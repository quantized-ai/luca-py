"""Mid-turn user messages across a LIVE subagent tree, end to end.

The cold-loaded acceptance matrix (including the reload-durability of the
subagents-active rejection) lives in `tests/agent/test_runner_post_message.py`;
these stories prove the same rules during a real drive: a post to the parent
while its children work is refused, a live subagent accepts posts into its own
open turn and answers them before it closes, and the parent accepts posts
again the moment its last child resolves — with the close-site check then
forcing the extra round that answers them.

Posting "during" the run is simulated deterministically: a tool body posts
(the child-side probes), or an `after_llm_response` middleware posts during a
specific LLM response. Never a timing trick.
"""

from pydantic import BaseModel, ConfigDict

from luca.agent.core import SubagentsActiveError, TurnOutcome
from luca.agent.core.models import TextContent
from luca.client.testing import faux_assistant_message, faux_text, faux_tool_call
from tests.agent.scenarios import DeterministicRunner, FakeTool
from tests.agent.subagents.conftest import SubagentRegistry, spawn_call, subagent_session

IDS = [f"x{n}" for n in range(80)]


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UpwardPostingTool(FakeTool):
    """A child-side tool that posts to the MAIN conversation while it — the
    child — is the very reason that conversation is mid-orchestration. The
    runner must refuse: the parent's open turn has an unresolved child."""

    name = "probe"
    description = "Posts upward."
    Args = _NoArgs

    def __init__(self) -> None:
        self.post = None  # wired by the test to runner.post_message
        self.raised: list[type] = []

    async def _execute(self, args, session, conversation_id, *, cancellation_token) -> str:
        try:
            self.post()
        except Exception as exc:  # the recorded TYPE is the assertion
            self.raised.append(type(exc))
        return "probed"


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


async def test_a_post_to_the_parent_while_children_work_raises_live(faux):
    # Call order with a single child is deterministic: parent spawn → child
    # tool round (the probe posts upward HERE, while the parent is parked on
    # its unresolved link) → child final → parent final.
    faux.set_responses(
        [
            faux_assistant_message(
                [spawn_call("Research A", "research A", call_id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_tool_call("probe", {}, id="tc2")], finish_reason="tool_use"),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("all done")], finish_reason="stop"),
        ]
    )
    session = subagent_session()
    probe = UpwardPostingTool()
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([probe]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    probe.post = lambda: runner.post_message("hurry up")
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    assert probe.raised == [SubagentsActiveError]
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
    # Call order: parent spawn → child final → parent "final" (the middleware
    # posts during THIS response — the link is already resolved, so the post
    # is accepted) → the extra round the close-site check forces.
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
