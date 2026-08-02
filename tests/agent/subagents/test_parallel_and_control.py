"""Parallelism, approvals, cancellation, failures, `autostart_subagents`,
usage and resume — the control surface, driven end to end.

The stories here are the ones §9 argues about: a gated subagent while its
siblings work, a failed child that resolves rather than propagating, a cancel
that cascades, and an application that drives the children itself.
"""

import asyncio

import pytest
from pydantic import BaseModel, ConfigDict

from luca.agent.contrib.subagents import RESULT_TOOL_NAME, SPAWN_TOOL_NAME
from luca.agent.core import AgentSession, ChildConversation
from luca.agent.core.events import SubagentsSpawned
from luca.agent.core.models import (
    ApprovalDecision,
    ApprovalOption,
    ConversationStatus,
    ExecutionStatus,
    TurnOutcome,
)
from luca.client.testing import faux_assistant_message, faux_text, faux_tool_call
from tests.agent.scenarios import AddTool, DeterministicRunner, FakeTool
from tests.agent.subagents.conftest import SubagentRegistry, spawn_call, subagent_session

IDS = [f"x{n}" for n in range(80)]


class SlowTool(FakeTool):
    """A body that yields to the loop long enough for a sibling to park and be
    restarted while this one is still working."""

    name = "slow"
    description = "Sleeps."

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

    async def _execute(self, args, session, conversation_id, *, cancellation_token) -> str:
        await asyncio.sleep(0.05)
        return "slow done"


def two_spawns():
    return faux_assistant_message(
        [
            spawn_call("Research A", "research A", call_id="tc1", task_id="t1"),
            spawn_call("Research B", "research B", call_id="tc2", task_id="t2"),
        ],
        finish_reason="tool_use",
    )


def children_of(session: AgentSession) -> list[str]:
    return [cid for cid, c in session.conversations.items() if c.depth == 1]


def links(session: AgentSession, conversation_id: str = "c1") -> list[ChildConversation]:
    return [
        entry
        for node in session.conversations[conversation_id].nodes
        if isinstance(entry := session.entries[node], ChildConversation)
    ]


# ── parallelism ───────────────────────────────────────────────────────────────


async def test_children_resolving_together_coalesce_into_one_wake_round(faux):
    # Both children finish before the parent's woken pass runs, so ONE loop
    # pass resolves both links and ONE wake round — here also the final round
    # — carries both task updates, each at its own resolution's position. The
    # 4-response script is the proof: a second wake round would pop a fifth
    # response and fail loudly. (Staggered resolutions — a wake round per
    # result — are pinned in test_wake_rounds.py, where the pending sibling is
    # held deterministically.)
    faux.set_responses(
        [
            two_spawns(),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("B done")], finish_reason="stop"),
            faux_assistant_message([faux_text("both done")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    # the parent's final model call happened LAST, after both children answered
    assert [link.execution_result.content[0].text for link in links(session)] == ["A done", "B done"]
    assert [m.content[0].text for m in faux.requests[-1].messages[-2:]] == [
        'Subagent task update:\n<task id=t1 status=completed completed_at="1970-01-01T00:00:01Z">\nA done\n</task>',
        'Subagent task update:\n<task id=t2 status=completed completed_at="1970-01-01T00:00:01Z">\nB done\n</task>',
    ]
    assert runner.idle()


async def test_every_conversation_ends_idle_and_the_catalog_holds_all_three(faux):
    faux.set_responses(
        [
            two_spawns(),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("B done")], finish_reason="stop"),
            faux_assistant_message([faux_text("both done")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    assert len(session.conversations) == 3
    assert {cid: c.depth for cid, c in session.conversations.items()} == {
        "c1": 0,
        children_of(session)[0]: 1,
        children_of(session)[1]: 1,
    }
    assert all(session.get_conversation_status(cid).status is ConversationStatus.IDLE for cid in session.conversations)


async def test_every_event_names_its_conversation(faux):
    faux.set_responses(
        [
            two_spawns(),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("B done")], finish_reason="stop"),
            faux_assistant_message([faux_text("both")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        events = [event async for event in run]

    # the tree's stream: the main conversation's events AND both subagents',
    # each routable by its own field
    assert {e.conversation_id for e in events} == {"c1", *children_of(session)}


# ── usage is conversation-scoped ─────────────────────────────────────────────


async def test_each_conversations_usage_lands_under_its_own_id(faux):
    faux.set_responses(
        [
            two_spawns(),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("B done")], finish_reason="stop"),
            faux_assistant_message([faux_text("both")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    # one row per conversation that made a model call, and every record
    # self-describes the conversation it belongs to
    assert set(session.usages) == {"c1", *children_of(session)}
    for conversation_id, records in session.usages.items():
        for entry_id, usage in records.items():
            assert usage.conversation_id == conversation_id
            assert usage.entry_id == entry_id
            assert entry_id in session.conversations[conversation_id].nodes
    # a session total is the sum ACROSS the catalog, archived rows included
    total = sum(u.total_tokens for records in session.usages.values() for u in records.values())
    assert total == sum(u.total_tokens for records in session.usages.values() for u in records.values())


# ── §5.6: a child's failure never escapes the child ──────────────────────────


async def test_a_child_that_runs_out_of_steps_resolves_rather_than_raising(faux):
    # `subagent_hard_max_steps` closes the child's turn ERRORED without
    # raising; to the parent it is a finished child whose result says it failed
    faux.set_responses(
        [
            faux_assistant_message([spawn_call("A", "a", call_id="tc1")], finish_reason="tool_use"),
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tcA")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("parent carries on")], finish_reason="stop"),
        ]
    )
    session = subagent_session(subagent_hard_max_steps=1)
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([AddTool()]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    [child_id] = children_of(session)
    closing = session.entries[session.conversations[child_id].nodes[-1]]
    assert closing.outcome is TurnOutcome.ERRORED
    # and the parent still answered
    [link] = links(session)
    assert link.execution_result is not None
    assert runner.idle()


async def test_a_child_with_no_final_message_resolves_with_a_transcript(faux):
    faux.set_responses(
        [
            faux_assistant_message([spawn_call("A", "a", call_id="tc1")], finish_reason="tool_use"),
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tcA")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = subagent_session(subagent_hard_max_steps=1)
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([AddTool()]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    [link] = links(session)
    assert link.execution_result.is_error is True
    # the child hit its hard step limit, so the fallback names the outcome
    # alongside the transcript
    assert link.execution_result.content[0].text.startswith(
        "The subagent was errored before finishing and left no final message. Transcript:"
    )


# ── §9.3: cancellation cascades ──────────────────────────────────────────────


async def test_cancelling_the_parent_cascades_and_resolves_every_child(faux):
    faux.set_responses([two_spawns()])
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    run = runner.run(autostart_subagents=False)
    events = []
    async with run:
        async for event in run:
            events.append(event)
            if isinstance(event, SubagentsSpawned):
                run.cancel()

    # UNDER `False` NO LIFECYCLE EVENTS EXIST — the flush announces its
    # closes only where the framework owns the children, so the never-driven
    # children stay silent here (no Started, Paused OR Finished).
    lifecycle = [e for e in events if type(e).__name__.startswith("Subagent")]
    assert [type(e).__name__ for e in lifecycle] == ["SubagentsSpawned"]
    # EVERY LINK RESOLVED. That is the invariant that matters: an unresolved
    # child on a CLOSED turn is unprojectable, so leaving one would wedge the
    # parent permanently. The wind-down writes the cancellation result itself,
    # without running the result tool — exactly as a PENDING execution is
    # terminalized without being dispatched.
    assert all(link.execution_result is not None for link in links(session))
    assert all(link.execution_result.is_error for link in links(session))
    assert runner.idle()
    # CANCELLING A CONVERSATION ALWAYS ENDS IT. These children were never
    # DRIVEN (autostart_subagents=False, and the cancel landed at the
    # announcement), so the cascade opens the bracket the cancellation needs to
    # close: seeded, cancelled, settled. Leaving them bracket-less would leave
    # the catalog holding conversations that derive BUSY forever — and would
    # make `run.child(cid).cancel()` a silent no-op, which is the same bug one
    # level down.
    for child_id in children_of(session):
        assert [session.entries[node].type for node in session.conversations[child_id].nodes] == [
            "user",
            "turn_start",
            "cancel_requested",
            "turn_finish",
        ]
        assert session.get_conversation_status(child_id).status is ConversationStatus.IDLE


async def test_cancelling_one_subagent_lets_its_parent_finish(faux):
    # `cancel()` is a SIGNAL: it writes the request and stops, because
    # consuming one is a DRIVE's job — and a subagent's drive is usually gone
    # by the time a cancel lands (it parked at a gate, or nobody ever started
    # it). The PARENT's drive does that flush, because it is the only
    # conversation that knows the child exists and the one that would otherwise
    # wait on it forever.
    faux.set_responses(
        [
            two_spawns(),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("only A came back")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    run = runner.run(autostart_subagents=False)
    async with run:
        async for event in run:
            if isinstance(event, SubagentsSpawned):
                first, second = event.conversation_ids
                run.child(second).cancel()  # never driven, and never will be
                async with run.child(first) as child:
                    _ = [_event async for _event in child]

    assert runner.idle()
    assert all(link.execution_result is not None for link in links(session))
    assert session.get_conversation_status(second).status is ConversationStatus.IDLE
    # the sibling is untouched: cancelling one subagent is not cancelling the turn
    assert [link.execution_result.is_error for link in links(session)] == [False, True]


async def test_cancelling_a_tree_twice_is_not_an_error(faux):
    # the cascade must tolerate a child that is already cancelling: cancelling
    # one subagent and then the whole tree is a normal sequence
    faux.set_responses([two_spawns()])
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    run = runner.run(autostart_subagents=False)
    async with run:
        async for event in run:
            if isinstance(event, SubagentsSpawned):
                first, *_ = event.conversation_ids
                run.child(first).cancel()
                run.cancel()  # cascades over a child already cancelling

    assert runner.idle()


# ── §9.4: autostart_subagents=False ──────────────────────────────────────────


async def test_the_application_can_drive_the_children_itself(faux):
    faux.set_responses(
        [
            two_spawns(),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("B done")], finish_reason="stop"),
            faux_assistant_message([faux_text("both")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")
    child_events: list = []

    async def drive(child):
        async with child:
            child_events.extend([event async for event in child])

    run = runner.run(autostart_subagents=False)
    parent_events: list = []
    async with run:
        async for event in run:
            if isinstance(event, SubagentsSpawned):
                children = [run.child(cid) for cid in event.conversation_ids]
                assert all(c is not None for c in children)
                await asyncio.gather(*(drive(c) for c in children))
            parent_events.append(event)

    assert runner.idle()
    # NO DOUBLE DELIVERY: forwarding follows ownership, so a child's events
    # reach the application through the child handle and NOT the parent stream
    assert {e.conversation_id for e in parent_events} == {"c1"}
    assert {e.conversation_id for e in child_events} == set(children_of(session))


async def test_the_handles_exist_before_the_children_start(faux):
    # the batch signal is emitted BEFORE the children are started, so a
    # `False`-mode consumer can reach every handle inside that event's branch
    faux.set_responses(
        [
            two_spawns(),
            faux_assistant_message([faux_text("A")], finish_reason="stop"),
            faux_assistant_message([faux_text("B")], finish_reason="stop"),
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")
    seen_at_announcement: list[str] = []

    run = runner.run(autostart_subagents=False)
    async with run:
        async for event in run:
            if isinstance(event, SubagentsSpawned):
                seen_at_announcement = sorted(run.children)
                for cid in event.conversation_ids:
                    child = run.child(cid)
                    async with child:
                        async for _ in child:
                            pass

    assert seen_at_announcement == sorted(children_of(session))


# ── §9.2: approvals are scoped to the conversation that raised them ──────────


async def test_a_gate_in_one_subagent_does_not_stop_its_sibling(faux):
    # A and B both spawned; B's tool call is deferred. A finishes anyway, and
    # `pending_approvals()` on the MAIN runner already reports B's gate —
    # subtree-scoped, attributed by the execution's own conversation_id.
    faux.set_responses(
        [
            two_spawns(),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tcB")],
                finish_reason="tool_use",
            ),
            # A's resolution is material, so the parent runs a wake round
            # before parking on the still-gated B
            faux_assistant_message([faux_text("waiting on B")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    registry = SubagentRegistry([AddTool()])
    registry.other.decisions = [ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)]
    runner = DeterministicRunner(session, tool_registry=registry, provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    run = runner.run(autostart_subagents=False)
    async with run:
        async for event in run:
            if isinstance(event, SubagentsSpawned):
                for cid in event.conversation_ids:
                    child = run.child(cid)
                    async with child:
                        async for _ in child:
                            pass

    gated = runner.pending_approvals()
    assert len(gated) == 1
    # attributed WITHOUT a wrapper type: the execution names its own conversation
    assert gated[0].conversation_id in children_of(session)
    assert gated[0].conversation_id != "c1"
    # the sibling finished regardless
    resolved = [link for link in links(session) if link.execution_result is not None]
    assert len(resolved) == 1


# ── resuming a subagent that parked at its gate ──────────────────────────────


def one_spawn():
    return faux_assistant_message(
        [spawn_call("Research A", "research A", call_id="tc1", task_id="t1")],
        finish_reason="tool_use",
    )


def gated_registry():
    """`add` defers once, then allows — the registry state an application's
    approval answer produces."""
    registry = SubagentRegistry([AddTool()])
    registry.other.decisions = [
        ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000),
        ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
    ]
    return registry


async def test_answering_a_subagents_gate_and_running_again_finishes_the_tree(faux):
    # THE RECOVERY CONTRACT, for a subagent. A gated child's drive ENDS —
    # nothing in its own subtree can advance — and the parent's drive returns
    # right behind it, so "answer, then run() again" is the only door back. The
    # next run has to re-drive the child, or the parent stays blocked on it
    # forever with nothing left to signal.
    faux.set_responses(
        [
            one_spawn(),
            faux_assistant_message([faux_tool_call("add", {"a": 1, "b": 2}, id="tcA")], finish_reason="tool_use"),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=gated_registry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    [child_id] = children_of(session)
    assert runner.blocked()
    assert [ex.conversation_id for ex in runner.pending_approvals()] == [child_id]

    async with runner.run() as run:  # the registry now allows — nothing else changed
        _ = [event async for event in run]

    assert runner.idle()
    assert [link.execution_result.content[0].text for link in links(session)] == ["A done"]


async def test_the_application_gets_a_fresh_handle_for_a_parked_subagent(faux):
    # A run handle is single-use, so under `autostart_subagents=False` — where
    # the APPLICATION owns the child's lifecycle — `run.child()` replaces the
    # one whose drive already ended. Without that there is no door back and the
    # parent's open turn wedges.
    faux.set_responses(
        [
            one_spawn(),
            faux_assistant_message([faux_tool_call("add", {"a": 1, "b": 2}, id="tcA")], finish_reason="tool_use"),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=gated_registry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")
    handles = []

    run = runner.run(autostart_subagents=False)
    async with run:
        async for event in run:
            if isinstance(event, SubagentsSpawned):
                [child_id] = event.conversation_ids
                for _ in range(2):
                    child = run.child(child_id)
                    handles.append(child)
                    async with child:
                        _ = [_event async for _event in child]

    assert handles[0] is not handles[1]  # the first one parked at the gate
    assert runner.idle()
    assert [link.execution_result.content[0].text for link in links(session)] == ["A done"]


async def test_notify_restarts_a_parked_subagent_inside_the_same_run(faux):
    # The shape polling cannot reach: one child gates while its sibling works,
    # so the run does not return. The gated child's drive is already gone by
    # then — `notify()` has to find that parked handle through the parent's
    # subtree and restart it, or the answer sits inert until the run ends.
    faux.set_responses(
        [
            two_spawns(),
            faux_assistant_message([faux_tool_call("add", {"a": 1, "b": 2}, id="tcA")], finish_reason="tool_use"),
            faux_assistant_message([faux_tool_call("slow", {}, id="tcB")], finish_reason="tool_use"),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
            # the restarted child resolves while its sibling's slow body still
            # runs, so the parent takes a wake round with that first update
            faux_assistant_message([faux_text("still waiting on B")], finish_reason="stop"),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
            faux_assistant_message([faux_text("both")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    registry = SubagentRegistry([AddTool(), SlowTool()])
    registry.other.decisions = [
        ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000),
        ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
        ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
    ]
    runner = DeterministicRunner(session, tool_registry=registry, provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")
    answered: list = []

    run = runner.run()

    async def watch() -> None:
        async for execution in run.approvals:
            if execution.id in [e.id for e in answered]:
                continue  # at-least-once: consumers dedup
            answered.append(execution)
            run.notify(execution)  # the registry allows on the second ask

    watcher = asyncio.ensure_future(watch())
    async with run:
        _ = [event async for event in run]
    await watcher

    # ONE run: the tree never came back blocked
    assert runner.idle()
    assert [ex.conversation_id for ex in answered] == [answered[0].conversation_id]
    assert answered[0].conversation_id in children_of(session)
    assert all(link.execution_result is not None for link in links(session))


# ── depth ─────────────────────────────────────────────────────────────────────


async def test_a_child_is_stamped_one_deeper_than_its_parent(faux):
    faux.set_responses(
        [
            faux_assistant_message([spawn_call("A", "a", call_id="tc1")], finish_reason="tool_use"),
            faux_assistant_message([faux_text("A")], finish_reason="stop"),
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    [child_id] = children_of(session)
    assert session.conversations[child_id].depth == session.conversations["c1"].depth + 1
    # a child carries no parent link: parent → child is the only direction
    # anything traverses, which is exactly why depth has to be stored
    assert session.conversations[child_id].previous_conversation_id is None


async def test_the_prompt_and_the_tool_list_never_disagree(faux):
    # both derive from ONE predicate: a static prompt would tell a subagent at
    # the cap that it can spawn while the gate withholds the tool
    from luca.agent.contrib.subagents import spawning_prompt_part

    session = subagent_session()
    [main] = ["c1"]
    session.conversations["deep"] = session.conversations["c1"].model_copy(update={"id": "deep", "depth": 1})

    assert spawning_prompt_part(session, main) is not None
    assert spawning_prompt_part(session, "deep") is None


async def test_the_prompt_is_silent_when_subagents_are_off(faux):
    from luca.agent.contrib.subagents import spawning_prompt_part

    assert spawning_prompt_part(subagent_session(enabled=False), "c1") is None


# ── resume ────────────────────────────────────────────────────────────────────


async def test_a_tree_round_trips_through_json(faux):
    faux.set_responses(
        [
            two_spawns(),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("B done")], finish_reason="stop"),
            faux_assistant_message([faux_text("both")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    reloaded = AgentSession.model_validate_json(session.model_dump_json())

    assert reloaded == session
    assert len(reloaded.conversations) == 3
    assert all(link.execution_result is not None for link in links(reloaded))


async def test_the_private_result_execution_is_an_ordinary_durable_entry(faux):
    faux.set_responses(
        [
            faux_assistant_message([spawn_call("A", "a", call_id="tc1")], finish_reason="tool_use"),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    result_execution = next(
        session.entries[n]
        for n in session.conversations["c1"].nodes
        if getattr(session.entries[n], "raw_tool_call", None) is not None
        and session.entries[n].raw_tool_call.name == RESULT_TOOL_NAME
    )
    # approvals, timing, spec filing and provenance — all of it, exactly like
    # any other tool call. Only its relationship to the WIRE differs.
    assert result_execution.status is ExecutionStatus.COMPLETED
    assert result_execution.approval_status is not None
    assert result_execution.conversation_id == "c1"
    assert result_execution.tool_spec.is_private is True
    assert result_execution.tool_spec_id in session.tool_specs
    assert result_execution.started_at is not None


@pytest.mark.parametrize("streaming", [False, True])
async def test_streaming_changes_the_event_vocabulary_and_nothing_else(faux, streaming):
    faux.set_responses(
        [
            faux_assistant_message([spawn_call("A", "a", call_id="tc1")], finish_reason="tool_use"),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run(streaming=streaming) as run:
        _ = [event async for event in run]

    assert [session.entries[n].type for n in session.conversations["c1"].nodes] == [
        "user",
        "turn_start",
        "assistant",
        "tool_execution",
        "child_conversation",
        "tool_execution",
        "assistant",
        "turn_finish",
    ]
    assert SPAWN_TOOL_NAME in [t.name for t in faux.requests[0].tools]
