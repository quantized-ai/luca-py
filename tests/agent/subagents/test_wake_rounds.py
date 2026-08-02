"""Wake rounds: the parent re-engages the model per subagent resolution.

The core v2 contract: a parent is no longer parked until every child
resolves — each resolution (and any other unseen material) triggers a model
round, with the result delivered as a task update at the RESULT EXECUTION's
path position, while the turn still cannot close COMPLETED before the last
child resolves. These stories pin the wake mechanics end to end; the
steering-post stories live in `test_post_messages.py`, the stop stories in
`test_stop_subagents.py`.

FIFO determinism: every story keeps at most ONE live LLM caller at a time —
sequential spawns (a new child per wake round), or a child that hangs in a
tool body. Two free-running children coalesce nondeterministically and are
asserted structurally elsewhere (`test_parallel_and_control.py`).
"""

import asyncio

import pytest
from pydantic import BaseModel, ConfigDict

from luca.agent.core import TurnOutcome
from luca.agent.core.events import SubagentFinished, SubagentsSpawned
from luca.agent.core.models import (
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    ChildConversation,
    ConversationStatus,
    ExecutionResult,
    ExecutionStatus,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    TurnFinish,
    TurnStart,
    UserMessage,
)
from luca.client.exceptions import ClientError
from luca.client.testing import faux_assistant_message, faux_error, faux_text, faux_tool_call
from tests.agent.scenarios import (
    MODEL,
    DeterministicRunner,
    FakeTool,
    conversation,
    make_session,
    spec,
)
from tests.agent.subagents.conftest import SubagentRegistry, spawn_call, subagent_session

IDS = [f"x{n}" for n in range(120)]


def update(task_id: str, text: str, *, status: str = "completed") -> str:
    return (
        f"Subagent task update:\n"
        f'<task id={task_id} status={status} completed_at="1970-01-01T00:00:01Z">\n{text}\n</task>'
    )


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PostingHangTool(FakeTool):
    """Posts a message, then hangs until cancelled — the deterministic wake
    trigger: the poster makes no further LLM calls, so every later response
    belongs to whoever the post woke."""

    name = "probe"
    description = "Posts, then hangs."
    Args = _NoArgs

    def __init__(self) -> None:
        self.post = None  # wired by the test

    async def _execute(self, args, session, conversation_id, *, cancellation_token) -> str:
        self.post()
        await asyncio.sleep(3600)
        return "unreachable"


async def test_each_resolution_wakes_the_parent_which_can_spawn_again(faux):
    # THE STAGGERED STORY, kept deterministic by sequencing: one child per
    # round, a wake round per resolution, the next spawn issued FROM a wake
    # round — and every update rendered below the parent's own last reply.
    faux.set_responses(
        [
            faux_assistant_message(
                [spawn_call("Find the bug", "finder", call_id="tc1", task_id="tA")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("A found the bug")], finish_reason="stop"),
            faux_assistant_message(
                [spawn_call("Write the fix", "fixer", call_id="tc2", task_id="tB")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("B wrote the fix")], finish_reason="stop"),
            faux_assistant_message([faux_text("bug found and fixed")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    assert runner.idle()
    assert len(faux.requests) == 5
    # the first wake round saw A's update as its newest message…
    assert faux.requests[2].messages[-1].content[0].text == update("tA", "A found the bug")
    # …and was offered the management tools alongside the spawn tool
    assert {tool.name for tool in faux.requests[2].tools} >= {"spawn_subagent", "stop_subagent", "list_subagents"}
    # the child itself was never offered them: it has no tasks of its own
    assert "stop_subagent" not in {tool.name for tool in (faux.requests[1].tools or [])}
    # the second wake round: B's update below the parent's own spawn round
    assert faux.requests[4].messages[-1].content[0].text == update("tB", "B wrote the fix")
    # the whole turn, in order: spawn round / update / spawn round / update /
    # final — one open bracket throughout
    assert [type(session.entries[n]).__name__ for n in session.conversations["c1"].nodes] == [
        "UserMessage",
        "TurnStart",
        "AssistantMessage",  # spawn A
        "ToolExecution",
        "ChildConversation",
        "ToolExecution",  # A's private result — the wake material
        "AssistantMessage",  # wake round: spawn B
        "ToolExecution",
        "ChildConversation",
        "ToolExecution",  # B's private result
        "AssistantMessage",  # the final answer
        "TurnFinish",
    ]


def resolved_unseen_session():
    """A parent saved right after a resolution, before the wake round: the
    result execution sits after the last assistant message — durable material."""
    spawn_spec = spec(
        "spawn_subagent",
        output_schema={"type": "object", "properties": {"is_subagent_spawn": {"type": "boolean"}}},
    )
    result_spec = spec("create_conversation_result", is_private=True)
    call = ToolCall(
        id="tc1",
        name="spawn_subagent",
        arguments={"prompt": "Research A", "description": "research", "task_id": "t1"},
    )
    return make_session(
        id="s_reload_wake",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
            "a1": AssistantMessage(
                id="a1",
                parent_id="ts",
                created_at=500,
                parts=[call],
                llm_config=MODEL,
                stop_reason="tool_use",
            ),
            "te1": ToolExecution(
                id="te1",
                conversation_id="c1",
                parent_id="a1",
                created_at=500,
                tool_call_id="tc1",
                raw_tool_call=call,
                tool_spec=spawn_spec,
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(
                    content=[TextContent(text="spawned")],
                    structured_content={
                        "is_subagent_spawn": True,
                        "task_id": "t1",
                        "prompt": "Research A",
                        "description": "research",
                        "process_subagent_result_tool_name": "create_conversation_result",
                    },
                ),
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=500)],
                started_at=500,
                ended_at=500,
                updated_at=500,
            ),
            "ch1": ChildConversation(
                id="ch1",
                parent_id="te1",
                created_at=500,
                conversation_id="c2",
                tool_execution_id="te1",
                execution_result=ExecutionResult(content=[TextContent(text="A done")]),
                result_execution_id="ter1",
            ),
            "ter1": ToolExecution(
                id="ter1",
                conversation_id="c1",
                parent_id="ch1",
                created_at=600,
                tool_call_id="rtc1",
                raw_tool_call=ToolCall(id="rtc1", name="create_conversation_result"),
                tool_spec=result_spec,
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(content=[TextContent(text="A done")]),
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=600)],
                started_at=600,
                ended_at=600,
                updated_at=600,
            ),
            "u_seed": UserMessage(id="u_seed", created_at=500, parts=[TextContent(text="Research A")]),
            "ts_c": TurnStart(id="ts_c", parent_id="u_seed", created_at=500),
            "a_c": AssistantMessage(
                id="a_c",
                parent_id="ts_c",
                created_at=500,
                parts=[TextContent(text="A done")],
                llm_config=MODEL,
                stop_reason="stop",
            ),
            "tf_c": TurnFinish(id="tf_c", parent_id="a_c", created_at=500),
        },
        conversations={
            "c1": conversation("c1", ["u1", "ts", "a1", "te1", "ch1", "ter1"]),
            "c2": conversation("c2", ["u_seed", "ts_c", "a_c", "tf_c"], depth=1),
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )


async def test_a_reload_right_after_a_resolution_still_wakes_the_model(faux):
    # The durable half of the wake condition: the runtime-minted result
    # execution after the last assistant message IS the record that a child
    # resolved unseen — a session loaded cold from that window derives BUSY
    # and wakes the model exactly once, with no runtime state.
    faux.set_responses([faux_assistant_message([faux_text("summary of A")], finish_reason="stop")])
    session = resolved_unseen_session()
    assert session.get_conversation_status("c1").status is ConversationStatus.BUSY
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry(),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )

    result = await runner.run()

    assert result.outcome is TurnOutcome.COMPLETED
    assert runner.idle()
    assert len(faux.requests) == 1
    assert faux.requests[0].messages[-1].content[0].text == (
        'Subagent task update:\n<task id=t1 status=completed completed_at="1970-01-01T00:00:00Z">\nA done\n</task>'
    )


async def test_hard_max_steps_cancels_the_running_children_and_closes_errored(faux):
    # The hard limit is a COST stop and it wins over the children: parking
    # would leave a hot-loop BUSY state, so the runner cancels the in-flight
    # subagents, settles every link, and closes ERRORED.
    faux.set_responses(
        [
            faux_assistant_message(
                [spawn_call("Research A", "research", call_id="tc1", task_id="t1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_text("digging in"), faux_tool_call("probe", {}, id="tc2")],
                finish_reason="tool_use",
            ),
        ]
    )
    session = subagent_session(hard_max_steps=1)
    probe = PostingHangTool()
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([probe]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    probe.post = lambda: runner.post_message("how is it going?")
    runner.post_message("go")

    async with runner.run() as run:
        events = [event async for event in run]

    # the post was material and the drive headed for the model — but the hard
    # limit fired first, cancelled the child mid-tool, and closed the turn
    closing = session.entries[session.conversations["c1"].nodes[-1]]
    assert (closing.type, closing.outcome) == ("turn_finish", TurnOutcome.ERRORED)
    assert closing.error == "Hard max steps limit reached: 1"
    [link] = [e for e in session.entries.values() if isinstance(e, ChildConversation)]
    assert link.execution_result == ExecutionResult(
        content=[TextContent(text="[subagent cancelled]")],
        is_error=True,
    )
    assert link.result_execution_id is None  # settled without the result tool
    child_id = link.conversation_id
    assert session.get_conversation_status(child_id).status is ConversationStatus.IDLE
    assert SubagentFinished(conversation_id=child_id, outcome=TurnOutcome.CANCELLED) in events
    assert len(faux.requests) == 2  # the model was never called for the limit


async def test_a_main_llm_failure_with_children_leaves_the_turn_open(faux):
    # A wake round's provider failure must not destroy the children's work:
    # the MAIN conversation's turn stays OPEN, the failure re-raises, and the
    # next engagement resumes the same bracket.
    faux.set_responses(
        [
            faux_assistant_message(
                [spawn_call("Research A", "research", call_id="tc1", task_id="t1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_text("digging in"), faux_tool_call("probe", {}, id="tc2")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([], error=faux_error("provider went away")),  # the wake round fails
        ]
    )
    session = subagent_session()
    probe = PostingHangTool()
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([probe]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    probe.post = lambda: runner.post_message("how is it going?")
    runner.post_message("go")

    with pytest.raises(ClientError, match="provider went away"):
        async with runner.run() as run:
            _ = [event async for event in run]

    # the bracket is still open — no TurnFinish — and the child still runs,
    # so the conversation stays engageable rather than closing ERRORED
    types = [session.entries[n].type for n in session.conversations["c1"].nodes]
    assert "turn_finish" not in types
    assert session.get_conversation_status("c1").status is ConversationStatus.BUSY
    [link] = [e for e in session.entries.values() if isinstance(e, ChildConversation)]
    assert link.execution_result is None

    # recovery is the caller's explicit action — here: give up and cancel;
    # the flush settles the child and the link like any wind-down
    runner.cancel()
    await runner.run()
    assert runner.idle()
    # re-read: resolution replaced the entry object
    [link] = [e for e in session.entries.values() if isinstance(e, ChildConversation)]
    assert link.execution_result.is_error is True


async def test_a_subagent_parents_llm_failure_settles_and_closes_errored(faux):
    # The depth split: nothing outside a subagent ever retries it mid-run, so
    # an open failed turn would strand the tree — a nested parent's failure
    # cancels its own children, closes ERRORED, and resolves upward as a
    # finished child whose result says so.
    faux.set_responses(
        [
            faux_assistant_message(
                [spawn_call("Coordinate the leaves", "nested", call_id="tc1", task_id="tN")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [spawn_call("Leaf work", "leaf", call_id="tc2", task_id="tL")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_text("leaf digging"), faux_tool_call("probe", {}, id="tc3")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([], error=faux_error("child provider down")),  # the CHILD's wake fails
            faux_assistant_message([faux_text("the nested task failed; wrapping up")], finish_reason="stop"),
        ]
    )
    session = subagent_session(max_depth=2)
    probe = PostingHangTool()
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([probe]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    holder: dict = {}
    probe.post = lambda: runner.post_message("status?", conversation_id=holder["nested"])
    runner.post_message("go")

    async with runner.run() as run:
        events = []
        async for event in run:
            events.append(event)
            if isinstance(event, SubagentsSpawned) and event.conversation_id == "c1":
                holder["nested"] = event.conversation_ids[0]

    assert runner.idle()
    nested_id = holder["nested"]
    # the nested parent's turn CLOSED (ERRORED), its hanging leaf settled
    closing = session.entries[session.conversations[nested_id].nodes[-1]]
    assert (closing.type, closing.outcome) == ("turn_finish", TurnOutcome.ERRORED)
    assert "child provider down" in closing.error
    [leaf_id] = [cid for cid, c in session.conversations.items() if c.depth == 2]
    assert session.get_conversation_status(leaf_id).status is ConversationStatus.IDLE
    assert SubagentFinished(conversation_id=nested_id, outcome=TurnOutcome.ERRORED) in events
    # …and the failure reached the MAIN conversation as an ordinary finished
    # child, never as an exception
    main_link = next(
        e for e in session.entries.values() if isinstance(e, ChildConversation) and e.conversation_id == nested_id
    )
    assert main_link.execution_result.is_error is True


class CancelOnInterrupted:
    """Middleware double: calls the wired cancel when the hanging child's tool
    body is killed — which happens INSIDE `_settle_children`'s awaits, the
    exact window where a user cancel must still control the close."""

    def __init__(self) -> None:
        self.cancel = None  # wired by the test to runner.cancel

    def after_tool_execution(self, execution, exception=None):
        if execution.status is ExecutionStatus.INTERRUPTED:
            self.cancel()
        return execution


async def test_a_cancel_landing_during_the_hard_max_settle_controls_the_close(faux):
    # The settle awaits the children's wind-downs — real awaits between the
    # loop-top cancel check and the close. A cancel() landing there must not
    # be buried inside an ERRORED bracket where nothing would ever consume it.
    faux.set_responses(
        [
            faux_assistant_message(
                [spawn_call("Research A", "research", call_id="tc1", task_id="t1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_text("digging in"), faux_tool_call("probe", {}, id="tc2")],
                finish_reason="tool_use",
            ),
        ]
    )
    session = subagent_session(hard_max_steps=1)
    probe = PostingHangTool()
    mw = CancelOnInterrupted()
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([probe]),
        provider=faux,
        ids=list(IDS),
        now=1000,
        middleware=[mw],
    )
    probe.post = lambda: runner.post_message("how is it going?")
    mw.cancel = runner.cancel
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    # the user's cancel won over the hard-limit's ERRORED close, and it was
    # consumed — one closing marker, the requested outcome
    main_closing = session.entries[session.conversations["c1"].nodes[-1]]
    assert (main_closing.type, main_closing.outcome) == ("turn_finish", TurnOutcome.CANCELLED)
    assert sum(1 for n in session.conversations["c1"].nodes if session.entries[n].type == "turn_finish") == 1
    assert runner.idle()


async def test_a_main_llm_failure_resume_retries_and_completes(faux):
    # THE RECOVERY CONTRACT for the open-turn failure: the next run() re-enters
    # the same bracket, the unchanged material re-engages the model, and the
    # orchestration finishes — here by stopping the hung task.
    faux.set_responses(
        [
            faux_assistant_message(
                [spawn_call("Research A", "research", call_id="tc1", task_id="t1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_text("digging in"), faux_tool_call("probe", {}, id="tc2")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([], error=faux_error("provider went away")),  # the wake round fails
        ]
    )
    session = subagent_session()
    probe = PostingHangTool()
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([probe]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    probe.post = lambda: runner.post_message("how is it going?")
    runner.post_message("go")

    with pytest.raises(ClientError, match="provider went away"):
        async with runner.run() as run:
            _ = [event async for event in run]

    # the caller's explicit retry: the SAME bracket resumes — no new TurnStart
    # — the model is re-engaged with the still-unseen post, stops the task,
    # and the turn closes COMPLETED once the child resolves
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("stop_subagent", {"task_id": "t1"}, id="tc3")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("stopping it")], finish_reason="stop"),
            faux_assistant_message([faux_text("wrapped up")], finish_reason="stop"),
        ]
    )
    result = await runner.run()

    assert result.outcome is TurnOutcome.COMPLETED
    assert runner.idle()
    types = [session.entries[n].type for n in session.conversations["c1"].nodes]
    assert types.count("turn_start") == 1
    assert types.count("turn_finish") == 1
    [link] = [e for e in session.entries.values() if isinstance(e, ChildConversation)]
    assert link.execution_result.is_error is True
