"""Runner ↔ ConversationProjector integration: the projector is the single
source of the wire history AND of the ToolExecuted presentation fields.

The projector's own projection rules live in `test_projection.py`; here the
subject is the collaborator seam: a runner built without one gets a default
`ConversationProjector`; a supplied projector's `project()` is what reaches
the LLM request (there is no projection middleware); its
`project_tool_execution()` output lands identically in the `ToolExecuted`
event and in the next request's correlated `ToolMessage`; and the projector
participates in runner configuration equality.
"""

from luca.agent.core.models import (
    AgentSession,
    Conversation,
    SessionConfig,
    TextContent,
    ToolExecution,
    UserMessage,
)
from luca.agent.core.projection import ConversationProjector
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
)
from luca.client.types import TextBlock as LucaTextBlock
from luca.client.types import ToolMessage
from luca.client.types import UserMessage as LucaUserMessage

from tests.agent.scenarios import (
    MODEL,
    AddTool,
    DeterministicRunner,
    FakeToolRegistry,
)


class RedactingProjector(ConversationProjector):
    """Rewrites every tool output — event and wire must agree."""

    def project_tool_execution(self, entry: ToolExecution, entries) -> ToolMessage:
        return ToolMessage(
            tool_call_id=entry.tool_call_id,
            content=[LucaTextBlock(text="[tool output redacted]")],
            is_error=False,
        )


class PrefixingProjector(ConversationProjector):
    """Injects synthetic history — full `project()` override policy."""

    def project(self, conversation, entries):
        return [
            LucaUserMessage(content=[LucaTextBlock(text="[SYNTHETIC PREAMBLE]")]),
            *super().project(conversation, entries),
        ]


def test_runner_defaults_to_a_fresh_conversation_projector():
    session = AgentSession(
        id="s_default_projector",
        active_conversation=Conversation(id="c1", nodes=[], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )

    runner = DeterministicRunner(session, now=1000)

    assert type(runner.conversation_projector) is ConversationProjector


async def test_supplied_projector_owns_the_llm_message_history():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("ok")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_projector_history",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "a1", "tf"], now=1000,
        conversation_projector=PrefixingProjector(),
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert faux.requests[0].messages == [
        LucaUserMessage(content=[LucaTextBlock(text="[SYNTHETIC PREAMBLE]")]),
        LucaUserMessage(content=[LucaTextBlock(text="Hi")]),
    ]


async def test_custom_tool_projection_reaches_event_and_wire_identically():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("done")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_projector_tool",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
        conversation_projector=RedactingProjector(),
    )

    async with runner.run() as run:
        events = [event async for event in run]

    # the event presents exactly what the model is told...
    executed = events[3]
    assert executed.type == "tool_executed"
    assert executed.result_text == "[tool output redacted]"
    assert executed.is_error is False
    # ...and the next request carries the same projection
    assert faux.requests[1].messages[-1] == ToolMessage(
        tool_call_id="tc1",
        content=[LucaTextBlock(text="[tool output redacted]")],
        is_error=False,
    )
    # the durable record keeps the real result — projection is derived state
    assert runner.session.entries["te1"].result.content == [TextContent(text="3")]


def test_projector_participates_in_runner_equality():
    session = AgentSession(
        id="s_projector_eq",
        active_conversation=Conversation(id="c1", nodes=[], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )

    default_a = DeterministicRunner(session, now=1000)
    default_b = DeterministicRunner(session, now=1000)
    custom = DeterministicRunner(
        session, now=1000, conversation_projector=RedactingProjector(),
    )

    assert default_a == default_b  # equivalent default projectors
    assert default_a != custom  # a different projection policy
