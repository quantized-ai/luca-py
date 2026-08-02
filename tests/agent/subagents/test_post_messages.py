"""Mid-turn user messages across a LIVE subagent tree, end to end.

The cold-loaded acceptance matrix lives in
`tests/agent/test_runner_post_message.py`; these stories prove the rules
during a real drive: a post to the parent while its children work WAKES it —
the model sees the message promptly and can steer, including stopping a
subagent — a live subagent accepts posts into its own open turn and answers
them before it closes, a post landing under a wake round's own in-flight LLM
call is answered immediately (the `last_seen` fingerprint, not the durable
material predicate, is what catches it), and the parent's turn still cannot
close COMPLETED before its last child resolves.

Posting "during" the run is simulated deterministically: a tool body posts
(the child-side probes), or an `after_llm_response` middleware posts during a
specific LLM response. Never a timing trick. The steering stories keep the
FIFO faux provider deterministic by construction: the child posts and then
HANGS in its tool body, so every subsequent request is the parent's until the
stop lands.
"""

import asyncio

from pydantic import BaseModel, ConfigDict

from luca.agent.core import TurnOutcome
from luca.agent.core.events import SubagentFinished
from luca.agent.core.models import ExecutionStatus, TextContent
from luca.client.testing import faux_assistant_message, faux_text, faux_tool_call
from tests.agent.scenarios import DeterministicRunner, FakeTool
from tests.agent.subagents.conftest import (
    STOP_TOOL_NAME,
    SubagentRegistry,
    spawn_call,
    subagent_session,
)

IDS = [f"x{n}" for n in range(80)]


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PostingHangTool(FakeTool):
    """A child-side tool that posts a steering message to the MAIN
    conversation and then hangs — the deterministic stand-in for a user typing
    while a subagent grinds. The hang is what keeps the FIFO script exact: the
    child makes no further LLM calls, so every later response is the parent's,
    until a stop cancels the body (INTERRUPTED under the default zero grace)."""

    name = "probe"
    description = "Posts upward, then hangs."
    Args = _NoArgs

    def __init__(self) -> None:
        self.post = None  # wired by the test to runner.post_message

    async def _execute(self, args, session, conversation_id, *, cancellation_token) -> str:
        self.post()
        await asyncio.sleep(3600)
        return "unreachable"


class SelfPostingTool(FakeTool):
    """A child-side tool that posts to its OWN conversation — a live subagent
    accepts input like any conversation with an open turn."""

    name = "note"
    description = "Posts to its own conversation."
    Args = _NoArgs

    def __init__(self) -> None:
        self.post = None  # wired by the test; receives the conversation id

    async def _execute(self, args, session, conversation_id, *, cancellation_token) -> str:
        self.post(conversation_id)
        return "noted"


class PostOnCall:
    """Middleware double: posts to the main conversation during the Nth LLM
    response — `after_llm_response` fires before the record, which is
    indistinguishable from a post landing during the in-flight call."""

    def __init__(self, call_number: int, text: str) -> None:
        self.post = None  # wired by the test to runner.post_message
        self.calls = 0
        self.call_number = call_number
        self.text = text

    def after_llm_response(self, message):
        self.calls += 1
        if self.calls == self.call_number:
            self.post(self.text)
        return message


async def test_a_post_while_children_work_wakes_the_parent_which_stops_the_task(faux):
    # THE STEERING STORY, end to end. The child posts to main and hangs; the
    # parent — parked on its unresolved link — is woken, sees the message,
    # stops the task with `stop_subagent`, acknowledges, and gets the
    # cancelled child's final report as a task update before answering.
    faux.set_responses(
        [
            faux_assistant_message(
                [spawn_call("Research A", "research A", call_id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_text("Starting research..."), faux_tool_call("probe", {}, id="tc2")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_tool_call(STOP_TOOL_NAME, {"task_id": "t1", "reason": "user asked to stop"}, id="tc3")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Stopping the research task.")], finish_reason="stop"),
            faux_assistant_message([faux_text("Research stopped; nothing else to do.")], finish_reason="stop"),
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
    probe.post = lambda: runner.post_message("Please stop the research, it is taking too long")
    runner.post_message("go")

    async with runner.run() as run:
        events = [event async for event in run]

    assert runner.idle()
    # The parent's path tells the whole story in order: the post landed inside
    # the turn, the stop round and its ack are ordinary assistant steps, and
    # the child's resolution (the private result execution) lands after them.
    assert [type(session.entries[n]).__name__ for n in session.conversations["c1"].nodes] == [
        "UserMessage",  # go
        "TurnStart",
        "AssistantMessage",  # spawn round
        "ToolExecution",  # the spawn call
        "ChildConversation",  # resolved, in the end, as stopped
        "UserMessage",  # the steering post — accepted mid-orchestration
        "AssistantMessage",  # the wake round: stop_subagent
        "ToolExecution",  # the stop call
        "AssistantMessage",  # the ack round
        "ToolExecution",  # the private result tool
        "AssistantMessage",  # the final answer
        "TurnFinish",
    ]
    # the wake round SAW the steering message, and was offered the control
    # tools (children exist, so stop/list are on the list)
    assert faux.requests[2].messages[-1].content[0].text == "Please stop the research, it is taking too long"
    assert {tool.name for tool in faux.requests[2].tools} >= {"stop_subagent", "list_subagents"}
    # the ack round saw the stop tool's own result
    assert faux.requests[3].messages[-1].content[0].text == (
        "Stop signal received for task t1. Reason: user asked to stop"
    )
    # the final round saw the stopped child's report as a task update, AFTER
    # the parent's own ack — append-only, never a rewrite of earlier history
    assert faux.requests[4].messages[-1].content[0].text == (
        "Subagent task update:\n"
        '<task id=t1 status=failed completed_at="1970-01-01T00:00:01Z">\n'
        "The subagent was cancelled before finishing. Its last message:\n\nStarting research...\n"
        "</task>"
    )
    # the child was cancelled: its hanging body INTERRUPTED, its link resolved
    # with the stopped report, and its ending announced as CANCELLED
    child_id = next(cid for cid, c in session.conversations.items() if c.depth == 1)
    child_executions = [
        session.entries[n] for n in session.conversations[child_id].nodes if session.entries[n].type == "tool_execution"
    ]
    assert [execution.status for execution in child_executions] == [ExecutionStatus.INTERRUPTED]
    link = next(e for e in session.entries.values() if e.type == "child_conversation")
    assert link.execution_result.is_error is True
    assert link.result_execution_id is not None
    assert SubagentFinished(conversation_id=child_id, outcome=TurnOutcome.CANCELLED) in events


async def test_a_post_during_a_wake_rounds_call_is_answered_immediately(faux):
    # The `last_seen` regression: a message posted while the wake round's OWN
    # LLM call is in flight lands BEFORE the recorded assistant entry, where
    # the durable material predicate cannot see it. The live drive must run
    # another round immediately rather than parking until the next child
    # resolution.
    faux.set_responses(
        [
            faux_assistant_message(
                [spawn_call("Research A", "research A", call_id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_text("Working..."), faux_tool_call("probe", {}, id="tc2")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Still working on it.")], finish_reason="stop"),
            faux_assistant_message(
                [faux_tool_call(STOP_TOOL_NAME, {"task_id": "t1"}, id="tc3")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Stopping.")], finish_reason="stop"),
            faux_assistant_message([faux_text("Stopped everything.")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    probe = PostingHangTool()
    mw = PostOnCall(3, "actually just stop it")
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([probe]),
        provider=faux,
        ids=list(IDS),
        now=1000,
        middleware=[mw],
    )
    probe.post = lambda: runner.post_message("how is it going?")
    mw.post = runner.post_message
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    # the extra round ran IMMEDIATELY: its projection ends with the mid-call
    # post and the premature answer, exactly like v1's close-race round
    assert [m.content[0].text for m in faux.requests[3].messages[-2:]] == [
        "actually just stop it",
        "Still working on it.",
    ]
    assert len(faux.requests) == 6
    assert runner.idle()


async def test_a_post_to_a_live_subagent_is_answered_within_its_turn(faux):
    faux.set_responses(
        [
            faux_assistant_message(
                [spawn_call("Research A", "research A", call_id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_tool_call("note", {}, id="tc2")], finish_reason="tool_use"),
            faux_assistant_message([faux_text("done as asked")], finish_reason="stop"),
            faux_assistant_message([faux_text("all done")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    note = SelfPostingTool()
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([note]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    note.post = lambda cid: runner.post_message("Also check the tests", conversation_id=cid)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    # the message sits INSIDE the child's turn, after the tool round it landed
    # during, and the child's final call carried it — answered before the close
    child_id = next(cid for cid, c in session.conversations.items() if c.depth == 1)
    child_nodes = session.conversations[child_id].nodes
    assert [type(session.entries[n]).__name__ for n in child_nodes] == [
        "UserMessage",  # the seed prompt — the FIRST user message, not the only one
        "TurnStart",
        "AssistantMessage",
        "ToolExecution",
        "UserMessage",  # the mid-turn post
        "AssistantMessage",
        "TurnFinish",
    ]
    assert session.entries[child_nodes[4]].parts == [TextContent(text="Also check the tests")]
    assert faux.requests[2].messages[-1].content[0].text == "Also check the tests"
    assert runner.idle()


async def test_the_parent_accepts_mid_turn_posts_once_children_resolved(faux):
    # Call order: parent spawn → child final → the resolution wake round (the
    # middleware posts during THIS response) → the extra round the close-site
    # check forces to answer the post.
    faux.set_responses(
        [
            faux_assistant_message(
                [spawn_call("Research A", "research A", call_id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("summary")], finish_reason="stop"),
            faux_assistant_message([faux_text("noted")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    mw = PostOnCall(3, "one more thing")
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry(),
        provider=faux,
        ids=list(IDS),
        now=1000,
        middleware=[mw],
    )
    mw.post = runner.post_message
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    assert len(faux.requests) == 4  # the extra round ran
    assert [type(session.entries[n]).__name__ for n in session.conversations["c1"].nodes] == [
        "UserMessage",
        "TurnStart",
        "AssistantMessage",  # the spawn round
        "ToolExecution",  # the spawn call
        "ChildConversation",  # resolved
        "ToolExecution",  # the private result tool
        "UserMessage",  # the accepted mid-turn post
        "AssistantMessage",  # the premature "summary" — recorded, never dropped
        "AssistantMessage",  # the extra round that answered the post
        "TurnFinish",
    ]
    assert session.entries[session.conversations["c1"].nodes[-1]].outcome is TurnOutcome.COMPLETED
    # the extra round's projection carried the post and the premature answer
    assert [m.content[0].text for m in faux.requests[3].messages[-2:]] == ["one more thing", "summary"]
    assert runner.idle()
