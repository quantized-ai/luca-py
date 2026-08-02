"""The stop handshake, runner-side: how a completed `is_subagent_stop`
payload becomes a cancellation — and, just as important, when it becomes
nothing.

The happy path (post → wake → stop → cancelled child reports back) is the
flagship story in `test_post_messages.py`; the contrib tool's own verdicts are
in `tests/agent/contrib/test_subagents.py`. What is pinned here is the
runner's matching discipline: the position bound that keeps an old stop from
killing a later task that reuses its id, and the no-ops for targets that do
not exist.
"""

import asyncio

import pytest
from pydantic import BaseModel, ConfigDict

from luca.agent.core import AgentError, TurnOutcome
from luca.agent.core.models import ChildConversation, ExecutionResult, TextContent
from luca.client.testing import faux_assistant_message, faux_text, faux_tool_call
from tests.agent.scenarios import DeterministicRunner, FakeTool
from tests.agent.subagents.conftest import (
    STOP_TOOL_NAME,
    SubagentRegistry,
    spawn_call,
    subagent_session,
)

IDS = [f"x{n}" for n in range(120)]


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PostingHangTool(FakeTool):
    name = "probe"
    description = "Posts, then hangs."
    Args = _NoArgs

    def __init__(self) -> None:
        self.post = None  # wired by the test

    async def _execute(self, args, session, conversation_id, *, cancellation_token) -> str:
        self.post()
        await asyncio.sleep(3600)
        return "unreachable"


class RogueStopTool(FakeTool):
    """Emits a stop payload of its own — the custom-tool half of the stop
    contract, and the door to the runner's no-op / fail-loud paths that the
    shipped tool's own validation never reaches."""

    name = "rogue_stop"
    description = "Emits an arbitrary stop payload."
    Args = _NoArgs

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def execute(self, args, session, conversation_id, *, cancellation_token):
        return ExecutionResult(
            content=[TextContent(text="signalled")],
            structured_content=self.payload,
        )


async def test_a_new_task_reusing_a_stopped_task_id_is_not_killed(faux):
    # THE POSITION BOUND. A stop execution stays on the path for the rest of
    # the turn, and the loop re-scans every pass — so it must only ever match
    # children spawned BEFORE it. The model stops t1, then spawns a NEW task
    # also named t1; the old stop must not touch it.
    faux.set_responses(
        [
            faux_assistant_message(
                [spawn_call("Research A", "first t1", call_id="tc1", task_id="t1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_text("digging"), faux_tool_call("probe", {}, id="tc2")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_tool_call(STOP_TOOL_NAME, {"task_id": "t1"}, id="tc3")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("stopping the first t1")], finish_reason="stop"),
            # the cancelled child's update wakes the parent, which REUSES the id
            faux_assistant_message(
                [spawn_call("Research A properly", "second t1", call_id="tc4", task_id="t1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("second t1 done")], finish_reason="stop"),
            faux_assistant_message([faux_text("all done")], finish_reason="stop"),
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
    probe.post = lambda: runner.post_message("stop the research")
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    assert runner.idle()
    links = [e for e in session.entries.values() if isinstance(e, ChildConversation)]
    assert [link.execution_result.is_error for link in links] == [True, False]
    assert links[1].execution_result.content == [TextContent(text="second t1 done")]
    # both children finished; only the first was cancelled
    outcomes = [session.entries[session.conversations[link.conversation_id].nodes[-1]].outcome for link in links]
    assert outcomes == [TurnOutcome.CANCELLED, TurnOutcome.COMPLETED]


async def test_a_stop_payload_for_an_unknown_task_is_a_no_op(faux):
    # a rogue payload naming nothing: the runner scans, matches nothing, and
    # the turn completes — no cancel, no raise
    faux.set_responses(
        [
            faux_assistant_message([faux_tool_call("rogue_stop", {}, id="tc1")], finish_reason="tool_use"),
            faux_assistant_message([faux_text("nothing to stop")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    rogue = RogueStopTool({"is_subagent_stop": True, "task_id": "ghost"})
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([rogue]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")

    result = await runner.run()

    assert result.outcome is TurnOutcome.COMPLETED
    assert runner.idle()
    assert session.conversations.keys() == {"c1"}  # nothing was ever spawned


async def test_a_stop_payload_without_a_task_id_is_a_contract_violation(faux):
    # the runner validates the keys it is about to ACT on, exactly like the
    # spawn payload's required fields
    faux.set_responses(
        [
            faux_assistant_message([faux_tool_call("rogue_stop", {}, id="tc1")], finish_reason="tool_use"),
        ]
    )
    session = subagent_session()
    rogue = RogueStopTool({"is_subagent_stop": True})
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([rogue]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")

    with pytest.raises(AgentError, match="missing its task_id"):
        async with runner.run() as run:
            _ = [event async for event in run]

    # a contract violation aborts the run with the turn left OPEN and
    # resumable, like every other fail-loud raise — nothing was cancelled
    types = [session.entries[n].type for n in session.conversations["c1"].nodes]
    assert "turn_finish" not in types
    assert session.conversations.keys() == {"c1"}


async def test_a_stop_for_an_id_that_already_resolved_reaches_the_live_sibling(faux):
    # Duplicate ids, resolved-BEFORE-the-stop shape: A(t1) completed normally,
    # then a same-id B(t1) hangs; the stop plainly names B. The resolved A
    # must be SKIPPED (its result execution precedes the stop on the path),
    # never treated as consuming the signal.
    faux.set_responses(
        [
            faux_assistant_message(
                [spawn_call("Research A", "first t1", call_id="tc1", task_id="t1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            # A's resolution wake: spawn a SECOND task reusing the id
            faux_assistant_message(
                [spawn_call("Research harder", "second t1", call_id="tc2", task_id="t1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_text("hmm"), faux_tool_call("probe", {}, id="tc3")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_tool_call(STOP_TOOL_NAME, {"task_id": "t1"}, id="tc4")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("stopping the second one")], finish_reason="stop"),
            faux_assistant_message([faux_text("all wrapped up")], finish_reason="stop"),
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
    probe.post = lambda: runner.post_message("stop t1")
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    assert runner.idle()
    links = [e for e in session.entries.values() if isinstance(e, ChildConversation)]
    # A's normal result stands; B — the live same-id sibling — was stopped
    assert [link.execution_result.is_error for link in links] == [False, True]
    assert links[0].execution_result.content == [TextContent(text="A done")]
    outcomes = [session.entries[session.conversations[link.conversation_id].nodes[-1]].outcome for link in links]
    assert outcomes == [TurnOutcome.COMPLETED, TurnOutcome.CANCELLED]


async def test_a_rogue_stop_for_a_finished_child_is_a_no_op(faux):
    # The runner's own re-check, past the shipped tool's validation: a payload
    # with the flag UP for a task that already finished must do nothing —
    # the signal is consumed, not an error.
    faux.set_responses(
        [
            faux_assistant_message(
                [spawn_call("Research A", "research", call_id="tc1", task_id="t1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            # the resolution wake: a rogue stop naming the now-finished task
            faux_assistant_message([faux_tool_call("rogue_stop", {}, id="tc2")], finish_reason="tool_use"),
            faux_assistant_message([faux_text("nothing to do")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    rogue = RogueStopTool({"is_subagent_stop": True, "task_id": "t1"})
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([rogue]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")

    result = await runner.run()

    assert result.outcome is TurnOutcome.COMPLETED
    assert runner.idle()
    [link] = [e for e in session.entries.values() if isinstance(e, ChildConversation)]
    # the finished child's normal result was never disturbed
    assert link.execution_result == ExecutionResult(content=[TextContent(text="A done")])
    [child_id] = [cid for cid in session.conversations if cid != "c1"]
    closing = session.entries[session.conversations[child_id].nodes[-1]]
    assert (closing.type, closing.outcome) == ("turn_finish", TurnOutcome.COMPLETED)
