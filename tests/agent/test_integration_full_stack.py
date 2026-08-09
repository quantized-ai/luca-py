"""End-to-end smoke test of the whole stack, driven the way an application is.

This is the "imaginary main.py": ONE runner composed exactly as the demo
composes it — `build_runner` gives the shell tools, the memory tools, the demo
math tools, the subagent tools and one shared `PermissionStrategy` — driven
through every public consumption form. Nothing is faked at the framework
boundary; the only substitutions are the LLM (a scripted faux transport) and
the workspace (a `tmp_path`, so the real shell tools do real filesystem work
somewhere disposable).

WHAT ONE SESSION COVERS, in four turns:

1. `start()` — eager, streaming, framework-driven subagents. Two subagents work
   in parallel with the real shell tools and each writes its own scratchpad.
2. `run()` — lazy, a gate in the MAIN conversation and a gate in a SUBAGENT at
   the same time, both answered through the TUI's own drive-then-prompt loop.
3. `run(autostart_subagents=False)` — the application drives the subagent
   itself, and DENIES its write.
4. `cancel()` at the announcement — the tree winds down and every link still
   resolves.

Then one declarative assertion over the final session: every conversation in
the catalog, its depth, its derived status and its full entry path.

This is a SMOKE test — it proves the pieces compose. It is not a substitute for
the focused suites (`tests/agent/subagents/`, `test_runner_tree.py`,
`tests/agent/contrib/`); when it fails, the focused test covering the same
ground says why.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from luca.agent.contrib.memory import MemoryPlugin
from luca.agent.contrib.resource_permissions import PermissionMode, PermissionStrategy
from luca.agent.contrib.subagents import RESULT_TOOL_NAME, SPAWN_TOOL_NAME
from luca.agent.contrib.tui.approvals import build_approval_prompts
from luca.agent.contrib.tui.wiring import build_runner
from luca.agent.core import (
    AgentSession,
    AgentSessionRunner,
    AssistantMessage,
    ChildConversation,
    ConversationStatus,
    ExecutionStatus,
    LLMConfig,
    RuntimeConfig,
    ToolExecution,
    TurnOutcome,
    pretty_print,
)
from luca.agent.core.events import ApprovalRequired, SubagentsSpawned
from luca.client.testing import (
    FauxProvider,
    FauxTransport,
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
)

MODEL = LLMConfig(model="fake-model", provider="faux")


# ── a faux transport that answers by conversation ────────────────────────────


class Hold:
    """A script entry that answers with a holding acknowledgement until the
    request carries every `needle`, then falls through to the next entry.

    A parent is re-engaged per subagent resolution, and whether two children
    resolving near-simultaneously coalesce into one wake round is task
    scheduling, not framework state — so the number of wake rounds a fixed
    script would have to predict is not deterministic. This entry absorbs
    them: reply `[waiting]` while results are still missing from the
    projection, and hand over to the real final once they have all arrived."""

    def __init__(self, needles: tuple[str, ...], text: str) -> None:
        self.needles = needles
        self.response = faux_assistant_message([faux_text(f"[waiting] {text}")], finish_reason="stop")


class ConversationScript(FauxTransport):
    """`FauxTransport` keyed by CONVERSATION instead of by arrival order.

    The stock transport pops one scripted response per call, so with subagents
    running in parallel which child gets which answer would depend on task
    scheduling. Every conversation opens with a user message — the seed prompt
    for a subagent, the posted message for the main one — so keying the script
    on that first message makes the whole tree deterministic without giving up
    the faux transport.
    """

    def __init__(self, scripts: dict[str, list]) -> None:
        super().__init__()
        self.scripts = {key: list(value) for key, value in scripts.items()}

    def _pop(self):
        request = self.requests[-1]
        opening = request.messages[0].content[0].text
        for key, queue in self.scripts.items():
            if not opening.startswith(key):
                continue
            flattened = "".join(
                block.text
                for message in request.messages
                for block in (message.content if isinstance(message.content, list) else [])
                if hasattr(block, "text") and isinstance(block.text, str)
            )
            while queue and isinstance(queue[0], Hold):
                hold = queue[0]
                if all(needle in flattened for needle in hold.needles):
                    queue.pop(0)  # satisfied — fall through to the real reply
                    continue
                return hold.response
            if not queue:
                raise AssertionError(f"the script for {key!r} ran out of responses")
            return queue.pop(0)
        raise AssertionError(f"no script for a conversation opening with {opening!r}")


# ── the application ──────────────────────────────────────────────────────────


def build(workspace: Path, transport: FauxTransport, **runtime):
    """`main.py`'s composition, minus the terminal."""
    session = AgentSessionRunner.new_session(
        MODEL,
        session_id="s_full",
        conversation_id="main",
        runtime_config=RuntimeConfig(subagents_enabled=True, **runtime),
    )
    runner, strategy, _questions = build_runner(
        session,
        workspace=workspace,
        provider=FauxProvider(transport=transport),
        mode=PermissionMode.ASK,
        subagents=True,
    )
    return session, runner, strategy


def answer(runner, strategy: PermissionStrategy, execution: ToolExecution, *, verdict: str = "once") -> None:
    """The TUI's modal policy, minus the modal.

    `build_approval_prompts` IS the app-side policy — it is what turns an
    execution into the steps a user answers, including the synthesized "run
    <name>" request a resourceless tool like `multiply` gets — so picking an
    option off its prompts exercises the real approval path rather than a
    reimplementation of it. `always` takes the tool's own suggested option
    (which writes a rule); a DENY kills the call, so it answers once and
    stops."""
    answers = []
    for prompt in build_approval_prompts(execution, strategy, main_conversation_id=runner.main_conversation_id):
        if verdict == "deny":
            option = next(candidate for candidate in prompt.options if candidate.is_deny)
        elif verdict == "always" and len(prompt.options) > 3:
            option = prompt.options[1]  # the tool's suggestion, scoped ALWAYS
        else:
            option = prompt.options[0]  # "Approve once"
        answers.append(option.answer)
        if option.is_deny:
            break
    strategy.apply_answer(execution, answers)


async def drive_until_idle(runner, strategy, sink: list, *, streaming: bool = False) -> None:
    """`TuiApp._drive`, in shape: drive, and if the conversation comes back
    BLOCKED answer its gates and drive again.

    THE DRIVE COMES BEFORE THE PROMPT — answering writes to the strategy, not to
    the execution, so a gate stays PENDING until a drive re-asks `decide()`.
    With subagents `pending_approvals()` is subtree-scoped, so this same loop
    answers a subagent's gate without knowing subagents exist."""
    for _ in range(8):  # bounded: a gate nothing can cover would loop
        if runner.idle():
            return
        run = runner.run(streaming=streaming)
        async with run:
            sink.extend([event async for event in run])
        if runner.blocked():
            for execution in runner.pending_approvals():
                answer(runner, strategy, execution)
    raise AssertionError("the conversation never went idle")


async def drive_child(run, runner, strategy, conversation_id: str, sink: list, *, verdict: str = "once") -> None:
    """Drive ONE subagent the way an application that owns it must.

    A run handle is single-use, and a subagent's drive ENDS the moment its gate
    defers, so resuming it is a FRESH handle — `run.child()` hands one back,
    which is the same two-step `runner.run()` is for the main conversation."""
    for _ in range(6):
        status = runner.session.get_conversation_status(conversation_id).status
        if status is ConversationStatus.IDLE:
            return
        if status is ConversationStatus.BLOCKED:
            for execution in runner.pending_approvals(conversation_id):
                answer(runner, strategy, execution, verdict=verdict)
                run.notify(execution)
        child = run.child(conversation_id)
        async with child:
            sink.extend([event async for event in child])
    raise AssertionError(f"subagent {conversation_id} never finished")


# ── scripted calls ───────────────────────────────────────────────────────────


def spawn(prompt: str, description: str, *, call_id: str, task_id: str):
    return faux_tool_call(
        SPAWN_TOOL_NAME,
        {"prompt": prompt, "description": description, "task_id": task_id},
        id=call_id,
    )


def read(path: Path, *, call_id: str):
    return faux_tool_call("read", {"file_path": str(path)}, id=call_id)


def write(path: Path, content: str, *, call_id: str):
    return faux_tool_call("write", {"file_path": str(path), "content": content}, id=call_id)


def note(content: str, *, call_id: str):
    return faux_tool_call("write_scratchpad", {"content": content}, id=call_id)


def todos(*contents: str, call_id: str):
    return faux_tool_call(
        "update_todos",
        {"todos": [{"content": content, "status": "pending"} for content in contents]},
        id=call_id,
    )


# ── reading the session back ─────────────────────────────────────────────────


def shape(session: AgentSession) -> dict[str, tuple[int, str, tuple[str, ...]]]:
    """Every conversation as (depth, derived status, entry types).

    The `[waiting]` wake-round acknowledgements are dropped before comparing:
    how many a turn records depends on whether near-simultaneous resolutions
    coalesced into one wake — task scheduling, not framework state — and the
    `Hold` script entries absorb exactly that variance."""

    def keep(entry) -> bool:
        return not (
            isinstance(entry, AssistantMessage)
            and entry.parts
            and entry.parts[0].type == "text"
            and entry.parts[0].text.startswith("[waiting]")
        )

    return {
        cid: (
            conversation.depth,
            session.get_conversation_status(cid).status.value,
            tuple(session.entries[node].type for node in conversation.nodes if keep(session.entries[node])),
        )
        for cid, conversation in session.conversations.items()
    }


def subagents(session: AgentSession) -> list[str]:
    return [cid for cid, c in session.conversations.items() if c.depth == 1]


def links(session: AgentSession, conversation_id: str = "main") -> list[ChildConversation]:
    return [
        entry
        for node in session.conversations[conversation_id].nodes
        if isinstance(entry := session.entries[node], ChildConversation)
    ]


def executions(session: AgentSession, conversation_id: str) -> list[ToolExecution]:
    return [
        entry
        for node in session.conversations[conversation_id].nodes
        if isinstance(entry := session.entries[node], ToolExecution)
    ]


def texts(session: AgentSession, conversation_id: str) -> list[str]:
    return [
        part.text
        for node in session.conversations[conversation_id].nodes
        if isinstance(entry := session.entries[node], AssistantMessage)
        for part in entry.parts
        if part.type == "text"
    ]


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch) -> Path:
    """The project directory, with HOME a SIBLING of it — the shell plugin
    already allows reads under the workspace, so a skill nested inside it would
    open whether or not the skill grant exists."""
    root = tmp_path / "workspace"
    home = tmp_path / "home"
    root.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # Instruction discovery reads the luca config directory, which XDG names.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    (root / "alpha.txt").write_text("alpha contents\n")
    (root / "beta.txt").write_text("beta contents\n")
    # Global, so its bundled files exercise the skill grant not the workspace one.
    skill = home / ".claude" / "skills" / "greeting"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: greeting\ndescription: How to greet someone.\n---\nAlways say hello twice.\n"
    )
    (skill / "references" / "tone.md").write_text("Warm, not effusive.\n")
    return root


# ── the whole thing ──────────────────────────────────────────────────────────


async def test_the_whole_stack_composes(workspace: Path):
    gamma = workspace / "gamma.txt"
    delta = workspace / "delta.txt"
    transport = ConversationScript(
        {
            # ── the main conversation: four turns, in order ──────────────────
            "Summarize": [
                # 1 · plan, then two subagents in parallel
                faux_assistant_message(
                    [
                        faux_thinking("Two files, two subagents.", signature="sig-1"),
                        todos("read alpha", "read beta", call_id="tc_todo"),
                        spawn("Read alpha.txt and report back", "read alpha", call_id="tc_a", task_id="A"),
                        spawn("Read beta.txt and report back", "read beta", call_id="tc_b", task_id="B"),
                    ],
                    finish_reason="tool_use",
                ),
                # wake rounds (the todo result, then each resolution) hold
                # until both children's answers are in the projection
                Hold(("alpha contents", "beta contents"), "the readers are still working"),
                faux_assistant_message([faux_text("Both files summarized.")], finish_reason="stop"),
                # 2 · a gate here AND a gate in the subagent, at once
                faux_assistant_message(
                    [
                        faux_tool_call("multiply", {"a": 6, "b": 7}, id="tc_mul"),
                        spawn(f"Write {gamma}", "write gamma", call_id="tc_c", task_id="C"),
                    ],
                    finish_reason="tool_use",
                ),
                Hold(("gamma written",), "gamma is still being written"),
                faux_assistant_message([faux_text("42, and gamma is written.")], finish_reason="stop"),
                # 3 · one subagent, driven by the application, denied
                faux_assistant_message(
                    [spawn(f"Write {delta}", "write delta", call_id="tc_d", task_id="D")],
                    finish_reason="tool_use",
                ),
                faux_assistant_message([faux_text("delta was refused.")], finish_reason="stop"),
                # 4 · two subagents nobody drives — cancelled at the announcement
                faux_assistant_message(
                    [
                        spawn("Never driven", "doomed", call_id="tc_e", task_id="E"),
                        spawn("Never driven either", "doomed too", call_id="tc_f", task_id="F"),
                    ],
                    finish_reason="tool_use",
                ),
            ],
            "Read alpha.txt": [
                faux_assistant_message(
                    [read(workspace / "alpha.txt", call_id="tc_ra"), note("A read alpha", call_id="tc_na")],
                    finish_reason="tool_use",
                ),
                faux_assistant_message([faux_text("alpha contents")], finish_reason="stop"),
            ],
            "Read beta.txt": [
                faux_assistant_message(
                    [read(workspace / "beta.txt", call_id="tc_rb"), note("B read beta", call_id="tc_nb")],
                    finish_reason="tool_use",
                ),
                faux_assistant_message([faux_text("beta contents")], finish_reason="stop"),
            ],
            f"Write {gamma}": [
                faux_assistant_message([write(gamma, "gamma!", call_id="tc_wg")], finish_reason="tool_use"),
                faux_assistant_message([faux_text("gamma written")], finish_reason="stop"),
            ],
            f"Write {delta}": [
                faux_assistant_message([write(delta, "delta!", call_id="tc_wd")], finish_reason="tool_use"),
                faux_assistant_message([faux_text("I was not allowed to write delta.")], finish_reason="stop"),
            ],
        }
    )
    session, runner, strategy = build(workspace, transport, subagent_hard_max_steps=6)
    memory = next(plugin for plugin in runner.plugins if isinstance(plugin, MemoryPlugin))

    # ── TURN 1 · start(), streaming, framework-driven subagents ───────────────
    runner.post_message("Summarize alpha.txt and beta.txt")
    turn_one: list = []
    run = runner.start(streaming=True)
    async with run:
        turn_one.extend([event async for event in run])
    result = await run

    assert result.status is ConversationStatus.IDLE
    assert result.outcome is TurnOutcome.COMPLETED
    assert runner.idle()

    first_wave = subagents(session)
    assert len(first_wave) == 2
    # THE STREAM IS THE TREE'S: an eager run's buffer carries the subagents'
    # events too, each tagged with the conversation that produced it
    assert {event.conversation_id for event in turn_one} == {"main", *first_wave}
    assert sum(isinstance(event, SubagentsSpawned) for event in turn_one) == 1
    # both children resolved before the parent answered
    assert sorted(link.execution_result.content[0].text for link in links(session)) == [
        "alpha contents",
        "beta contents",
    ]

    # the REAL shell tools did real work — reads inside the workspace never
    # prompt, because `ShellAccessPlugin` seeds the read tier
    reads = [
        execution
        for child in first_wave
        for execution in executions(session, child)
        if execution.raw_tool_call.name == "read"
    ]
    assert len(reads) == 2
    assert all(execution.status is ExecutionStatus.COMPLETED for execution in reads)
    assert sorted("alpha contents" in execution.result.content[0].text for execution in reads) == [False, True]

    # PER-CONVERSATION MEMORY. One plugin instance serves the whole tree, so an
    # unkeyed store would have the two subagents overwriting each other and
    # this would read one value.
    assert sorted(slot["content"] for slot in memory.scratchpad_store.values()) == ["A read alpha", "B read beta"]
    assert set(memory.scratchpad_store) == set(first_wave)
    assert set(memory.todo_store) == {"main"}  # the parent's list, nobody else's

    # ── TURN 2 · run(), a gate in main AND in a subagent, TUI-shaped loop ─────
    runner.post_message("Multiply 6 by 7 and have a subagent write gamma.txt")
    turn_two: list = []
    await drive_until_idle(runner, strategy, turn_two)

    assert runner.idle()
    assert gamma.read_text() == "gamma!"
    # BOTH gates reached the one subtree-scoped door: the main conversation's
    # `multiply` (a resourceless tool, so its request is synthesized) and the
    # subagent's `write`
    gated = {
        execution.conversation_id
        for event in turn_two
        if isinstance(event, ApprovalRequired)
        for execution in event.executions
    }
    assert gated == {"main", *[cid for cid in subagents(session) if cid not in first_wave]}
    assert "42" in texts(session, "main")[-1]

    # ── TURN 3 · autostart_subagents=False, and a DENY ────────────────────────
    runner.post_message("Have a subagent write delta.txt")
    parent_events: list = []
    child_events: list = []
    run = runner.run(autostart_subagents=False)
    async with run:
        async for event in run:
            parent_events.append(event)
            if isinstance(event, SubagentsSpawned):
                [child_id] = event.conversation_ids
                assert run.child(child_id) is not None  # the handle exists at announcement
                await drive_child(
                    run,
                    runner,
                    strategy,
                    child_id,
                    child_events,
                    verdict="deny",
                )

    assert runner.idle()
    assert not delta.exists()
    # NO DOUBLE DELIVERY: forwarding follows ownership, so an app-driven child's
    # events arrive on the child handle and NOT on the parent's stream
    assert {event.conversation_id for event in parent_events} == {"main"}
    assert {event.conversation_id for event in child_events} == {child_id}
    denied = [execution for execution in executions(session, child_id) if execution.raw_tool_call.name == "write"]
    assert [execution.status for execution in denied] == [ExecutionStatus.REJECTED]
    assert links(session)[-1].execution_result is not None  # the parent still resolved

    # ── TURN 4 · cancel the whole tree at the announcement ────────────────────
    runner.post_message("Spawn two and change your mind")
    run = runner.run(autostart_subagents=False)
    async with run:
        async for event in run:
            if isinstance(event, SubagentsSpawned):
                run.cancel(error="changed my mind")

    assert runner.idle()
    # EVERY LINK RESOLVED. An unresolved child on a CLOSED turn is
    # unprojectable, so the wind-down writes the cancellation result itself.
    assert all(link.execution_result is not None for link in links(session))
    closing = session.entries[session.conversations["main"].nodes[-1]]
    assert (closing.type, closing.outcome, closing.error) == ("turn_finish", TurnOutcome.CANCELLED, "changed my mind")

    # ── invariants that hold across the whole session ─────────────────────────

    # PRIVATE TOOLS never reached the wire, in any conversation…
    assert all(RESULT_TOOL_NAME not in [tool.name for tool in (request.tools or [])] for request in transport.requests)
    # …and the SPAWN tool was offered to the main conversation only: the gate is
    # depth-based, and every subagent here sits at the cap
    with_spawn = [
        request for request in transport.requests if SPAWN_TOOL_NAME in [tool.name for tool in (request.tools or [])]
    ]
    main_requests = [
        request for request in transport.requests if request.messages[0].content[0].text.startswith("Summarize")
    ]
    assert with_spawn == main_requests

    # the runtime still dispatched the private tool, once per child that ran
    private = [
        execution for execution in executions(session, "main") if execution.raw_tool_call.name == RESULT_TOOL_NAME
    ]
    assert len(private) == 4  # A, B, C, D — the two cancelled ones never ran it
    assert all(execution.tool_spec.is_private for execution in private)

    # PROVENANCE: every execution names the conversation it was born in
    for cid in session.conversations:
        assert all(execution.conversation_id == cid for execution in executions(session, cid))

    # USAGE is conversation-scoped and self-describing
    assert set(session.usages) <= set(session.conversations)
    for cid, records in session.usages.items():
        assert all(usage.conversation_id == cid for usage in records.values())

    # ── THE SESSION, DECLARATIVELY ────────────────────────────────────────────
    final = shape(session)
    assert final["main"] == (
        0,
        "idle",
        (
            # 1 · todos + two spawns → two children → two private results
            "user",
            "turn_start",
            "assistant",
            "tool_execution",
            "tool_execution",
            "tool_execution",
            "child_conversation",
            "child_conversation",
            "tool_execution",
            "tool_execution",
            "assistant",
            "turn_finish",
            # 2 · a gated multiply beside a spawn whose child also gated
            "user",
            "turn_start",
            "assistant",
            "tool_execution",
            "tool_execution",
            "child_conversation",
            "tool_execution",
            "assistant",
            "turn_finish",
            # 3 · one app-driven child whose write was denied
            "user",
            "turn_start",
            "assistant",
            "tool_execution",
            "child_conversation",
            "tool_execution",
            "assistant",
            "turn_finish",
            # 4 · two spawns, then the cancel wind-down
            "user",
            "turn_start",
            "assistant",
            "tool_execution",
            "tool_execution",
            "child_conversation",
            "child_conversation",
            "cancel_requested",
            "turn_finish",
        ),
    )
    # six subagents: four that ran, two that were cancelled at birth
    assert len(final) == 7
    assert sorted(depth for depth, _, _ in final.values()) == [0, 1, 1, 1, 1, 1, 1]
    # NOTHING IN THE CATALOG IS LEFT MID-FLIGHT
    assert {status for _, status, _ in final.values()} == {"idle"}
    ran = [cid for cid, (_, _, types) in final.items() if cid != "main" and "assistant" in types]
    assert len(ran) == 4
    for cid in ran:
        depth, status, types = final[cid]
        assert (depth, status) == (1, "idle")
        assert types[0] == "user"  # the seed prompt, and nothing of the parent's
        assert types[-1] == "turn_finish"
    for cid in set(final) - {"main", *ran}:
        # cancelled before they ever ran: the cascade opens the bracket the
        # cancellation needs to close, so a cancelled conversation is a settled
        # one whether or not it had started
        assert final[cid] == (1, "idle", ("user", "turn_start", "cancel_requested", "turn_finish"))

    # EVERYTHING ROUND TRIPS
    reloaded = AgentSession.model_validate_json(session.model_dump_json())
    assert reloaded == session
    assert shape(reloaded) == final
    assert json.loads(session.model_dump_json())["main_conversation_id"] == "main"


# ── the composition facts that deserve their own failure ─────────────────────


async def test_a_subagents_gate_is_answered_live_while_its_sibling_works(workspace: Path):
    """The one shape polling cannot reach: a subagent gates while its sibling
    keeps working, so the run does not return and there is no between-drives
    moment to check `pending_approvals()` in. `run.approvals` + `notify()`."""
    target = workspace / "live.txt"
    transport = ConversationScript(
        {
            "Do both": [
                faux_assistant_message(
                    [
                        spawn("Read alpha.txt", "read", call_id="tc_a", task_id="A"),
                        spawn(f"Write {target}", "write", call_id="tc_b", task_id="B"),
                    ],
                    finish_reason="tool_use",
                ),
                faux_assistant_message([faux_text("both done")], finish_reason="stop"),
            ],
            "Read alpha.txt": [
                faux_assistant_message([read(workspace / "alpha.txt", call_id="tc_r")], finish_reason="tool_use"),
                faux_assistant_message([faux_text("alpha")], finish_reason="stop"),
            ],
            f"Write {target}": [
                faux_assistant_message([write(target, "live!", call_id="tc_w")], finish_reason="tool_use"),
                faux_assistant_message([faux_text("written")], finish_reason="stop"),
            ],
        }
    )
    session, runner, strategy = build(workspace, transport)
    runner.post_message("Do both at once")
    seen: list = []

    run = runner.run()

    async def answer_gates() -> None:
        async for execution in run.approvals:
            seen.append(execution)
            answer(runner, strategy, execution)
            run.notify(execution)

    watcher = asyncio.ensure_future(answer_gates())
    async with run:
        _ = [event async for event in run]
    await watcher

    assert runner.idle()
    assert target.read_text() == "live!"
    # answered from the stream, in ONE run — the tree never came back blocked
    assert [execution.raw_tool_call.name for execution in seen] == ["write"]
    assert seen[0].conversation_id in subagents(session)
    assert all(link.execution_result is not None for link in links(session))


async def test_the_read_before_write_guard_does_not_leak_across_conversations(workspace: Path):
    """`FileReadTracker` is keyed by conversation, and that is a safety
    property: the main agent reading a file must not satisfy the guard for a
    subagent that never read it."""
    alpha = workspace / "alpha.txt"
    transport = ConversationScript(
        {
            "Read alpha then": [
                faux_assistant_message([read(alpha, call_id="tc_r")], finish_reason="tool_use"),
                faux_assistant_message(
                    [spawn(f"Overwrite {alpha}", "overwrite", call_id="tc_s", task_id="A")],
                    finish_reason="tool_use",
                ),
                faux_assistant_message([faux_text("the subagent could not")], finish_reason="stop"),
            ],
            "Overwrite": [
                faux_assistant_message([write(alpha, "clobbered", call_id="tc_w")], finish_reason="tool_use"),
                faux_assistant_message([faux_text("I had not read it")], finish_reason="stop"),
            ],
        }
    )
    session, runner, strategy = build(workspace, transport)
    runner.post_message("Read alpha then have a subagent overwrite it")

    await drive_until_idle(runner, strategy, [])

    assert alpha.read_text() == "alpha contents\n"  # untouched
    [child_id] = subagents(session)
    [attempt] = [e for e in executions(session, child_id) if e.raw_tool_call.name == "write"]
    assert attempt.result.is_error is True
    assert "has not been read yet" in attempt.result.content[0].text
    # …and the parent still finished: a failed tool inside a subagent is a
    # finished subagent, not an exception travelling upward
    assert runner.idle()


async def test_a_skill_is_advertised_loaded_and_its_bundled_files_open_ungated(workspace: Path):
    """Description in the prompt, body only via the tool, bundled file read
    without ever reaching the gate."""
    # Outside the workspace: only the skill grant makes this readable.
    reference = Path.home() / ".claude" / "skills" / "greeting" / "references" / "tone.md"
    assert workspace not in reference.parents
    transport = ConversationScript(
        {
            "Greet": [
                faux_assistant_message(
                    [faux_tool_call("skill", {"name": "greeting"}, id="tc_skill")],
                    finish_reason="tool_use",
                ),
                faux_assistant_message([read(reference, call_id="tc_ref")], finish_reason="tool_use"),
                faux_assistant_message([faux_text("Hello, hello.")], finish_reason="stop"),
            ],
        }
    )
    session, runner, strategy = build(workspace, transport)
    runner.post_message("Greet the user")

    await drive_until_idle(runner, strategy, [])

    prompt = runner.build_system_message(session.main_conversation_id)
    assert "- greeting: How to greet someone." in prompt
    assert "Always say hello twice." not in prompt  # the body is NOT in the prompt
    loaded, opened = [e for e in executions(session, "main") if e.raw_tool_call.name in ("skill", "read")]
    assert "Always say hello twice." in loaded.result.content[0].text
    assert str(reference.parent.parent) in loaded.result.content[0].text  # the skill dir
    assert "Warm, not effusive." in opened.result.content[0].text
    # `drive_until_idle` auto-answers gates, so a completed read proves nothing;
    # without the grant this log reads ['pending', 'allow'].
    assert [d.decision.value for d in opened.approval_decisions] == ["allow"]
    assert runner.idle()


async def test_pretty_print_renders_any_conversation_in_the_tree(workspace: Path):
    transport = ConversationScript(
        {
            "Read alpha via": [
                faux_assistant_message(
                    [spawn("Read alpha.txt", "read alpha", call_id="tc_a", task_id="A")],
                    finish_reason="tool_use",
                ),
                faux_assistant_message([faux_text("done")], finish_reason="stop"),
            ],
            "Read alpha.txt": [
                faux_assistant_message([read(workspace / "alpha.txt", call_id="tc_r")], finish_reason="tool_use"),
                faux_assistant_message([faux_text("alpha")], finish_reason="stop"),
            ],
        }
    )
    session, runner, _ = build(workspace, transport)
    runner.post_message("Read alpha via a subagent")

    await runner.run()

    [child_id] = subagents(session)
    assert f"Subagent · {child_id}" in pretty_print(session)
    assert "· depth 1" in pretty_print(session, child_id)
    assert "Read alpha.txt" in pretty_print(session, child_id)
    assert "· depth" not in pretty_print(session)  # the main transcript is unchanged


async def test_a_reloaded_session_keeps_driving_the_same_tree(workspace: Path):
    transport = ConversationScript(
        {
            "Read alpha via": [
                faux_assistant_message(
                    [spawn("Read alpha.txt", "read alpha", call_id="tc_a", task_id="A")],
                    finish_reason="tool_use",
                ),
                faux_assistant_message([faux_text("done")], finish_reason="stop"),
                # the reloaded session posts into the SAME conversation, so its
                # second turn is answered from this same script
                faux_assistant_message([faux_text("still here")], finish_reason="stop"),
            ],
            "Read alpha.txt": [
                faux_assistant_message([read(workspace / "alpha.txt", call_id="tc_r")], finish_reason="tool_use"),
                faux_assistant_message([faux_text("alpha")], finish_reason="stop"),
            ],
        }
    )
    session, runner, _ = build(workspace, transport)
    runner.post_message("Read alpha via a subagent")
    await runner.run()

    # cold reload into a fresh runner — the durable graph is the whole truth
    reloaded = AgentSession.model_validate_json(session.model_dump_json())
    fresh, _, _ = build_runner(
        reloaded,
        workspace=workspace,
        provider=FauxProvider(transport=transport),
        mode=PermissionMode.ASK,
        subagents=True,
    )
    fresh.post_message("Carry on")

    await fresh.run()

    assert fresh.idle()
    assert len(reloaded.conversations) == 2
    assert reloaded.main_conversation_id == "main"


async def test_the_subagents_really_run_at_the_same_time(workspace: Path):
    """Two subagents whose tool bodies both sleep: serialized, the wall clock
    would be the SUM; the interleaving is what proves it is the max."""
    from pydantic import BaseModel, ConfigDict

    from luca.agent.contrib.plugins import PluginAgentSessionRunner
    from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry, YoloPermissionPolicy
    from luca.agent.contrib.subagents import SubagentsPlugin
    from luca.agent.contrib.tools import Tool
    from luca.agent.core import CancellationToken, ToolKind

    order: list[str] = []

    class SlowTool(Tool):
        name = "slow"
        description = "Sleeps, then echoes its label."
        tool_kind = ToolKind.OTHER

        class Args(BaseModel):
            model_config = ConfigDict(extra="forbid")

            label: str

        async def _execute(
            self,
            args,
            session,
            conversation_id,
            *,
            tool_name: str,
            tool_call_id: str,
            cancellation_token: CancellationToken,
        ) -> str:
            order.append(f"enter:{args['label']}")
            await asyncio.sleep(0.05)
            order.append(f"exit:{args['label']}")
            return args["label"]

    transport = ConversationScript(
        {
            "Do two": [
                faux_assistant_message(
                    [
                        spawn("Task A", "a", call_id="tc_a", task_id="A"),
                        spawn("Task B", "b", call_id="tc_b", task_id="B"),
                    ],
                    finish_reason="tool_use",
                ),
                faux_assistant_message([faux_text("both")], finish_reason="stop"),
            ],
            "Task A": [
                faux_assistant_message([faux_tool_call("slow", {"label": "A"}, id="tcA")], finish_reason="tool_use"),
                faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            ],
            "Task B": [
                faux_assistant_message([faux_tool_call("slow", {"label": "B"}, id="tcB")], finish_reason="tool_use"),
                faux_assistant_message([faux_text("B done")], finish_reason="stop"),
            ],
        }
    )
    session = AgentSessionRunner.new_session(
        MODEL,
        session_id="s_conc",
        conversation_id="main",
        runtime_config=RuntimeConfig(subagents_enabled=True),
    )
    runner = PluginAgentSessionRunner(
        session,
        tool_registry=SimpleToolRegistry(tools=[SlowTool()], permission_policy=YoloPermissionPolicy()),
        plugins=[SubagentsPlugin()],
        provider=FauxProvider(transport=transport),
    )
    runner.post_message("Do two things at once")

    await runner.run()

    # INTERLEAVED: serialized execution would give enter:A, exit:A, enter:B…
    assert order == ["enter:A", "enter:B", "exit:A", "exit:B"]
    assert runner.idle()
    assert sorted(link.execution_result.content[0].text for link in links(session)) == ["A done", "B done"]


async def test_one_permission_strategy_serves_the_whole_tree(workspace: Path):
    """A rule an ALWAYS answer wrote while one subagent was asking covers the
    next subagent too: the strategy belongs to the application, not to a
    conversation."""
    first, second = workspace / "one.txt", workspace / "two.txt"
    transport = ConversationScript(
        {
            "Write both": [
                faux_assistant_message(
                    [spawn(f"Write {first}", "one", call_id="tc_a", task_id="A")],
                    finish_reason="tool_use",
                ),
                faux_assistant_message(
                    [spawn(f"Write {second}", "two", call_id="tc_b", task_id="B")],
                    finish_reason="tool_use",
                ),
                faux_assistant_message([faux_text("both written")], finish_reason="stop"),
            ],
            f"Write {first}": [
                faux_assistant_message([write(first, "1", call_id="tc_w1")], finish_reason="tool_use"),
                faux_assistant_message([faux_text("one written")], finish_reason="stop"),
            ],
            f"Write {second}": [
                faux_assistant_message([write(second, "2", call_id="tc_w2")], finish_reason="tool_use"),
                faux_assistant_message([faux_text("two written")], finish_reason="stop"),
            ],
        }
    )
    session, runner, strategy = build(workspace, transport)
    runner.post_message("Write both files, one subagent each")
    gates: list = []

    for _ in range(8):
        if runner.idle():
            break
        run = runner.run()
        async with run:
            _ = [event async for event in run]
        for execution in runner.pending_approvals():
            gates.append(execution)
            answer(runner, strategy, execution, verdict="always")

    assert (first.read_text(), second.read_text()) == ("1", "2")
    # the SECOND subagent never had to ask
    assert [execution.conversation_id for execution in gates] == [subagents(session)[0]]
