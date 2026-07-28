"""Smoke tests for the adapter's two remaining translations: a KNOWN client
assistant message renders into KNOWN agent message parts (the inbound
direction), and a KNOWN `ToolSpec` projects to a KNOWN client Tool definition.
Declarative — hardcoded invariant in, full expected out. No logic, no helpers.
(Conversation → LLM-message projection lives in `test_projection.py`.)
"""

from luca.agent.core.adapter import message_to_parts, tool_spec_to_luca_tool
from luca.agent.core.models import (
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolKind,
    ToolSpec,
)
from luca.client.types import (
    AssistantMessage as LucaAssistantMessage,
    TextBlock,
    ThinkingBlock,
    Tool as LucaTool,
    ToolCall as LucaToolCall,
)

# A tool's arguments as the core carries them: a plain JSON Schema dict, never
# a Pydantic class. Shared by the spec under test and the expected wire tool so
# "straight through" is what the assertion reads as.
ADD_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
    "additionalProperties": False,
}


def test_message_to_parts_preserves_block_order():
    message = LucaAssistantMessage(
        content=[
            ThinkingBlock(text="Let me add."),
            TextBlock(text="Sure —"),
            LucaToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        ],
    )

    assert message_to_parts(message) == [
        ThinkingContent(thinking="Let me add."),
        TextContent(text="Sure —"),
        ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
    ]


def test_tool_spec_projects_to_client_tool():
    # `input_schema` goes to `parameters` verbatim, and the framework-only
    # classification fields (kind, namespace, version, timeout, metadata) have
    # no place on the wire.
    spec = ToolSpec(
        name="add",
        description="Add two numbers.",
        input_schema=ADD_SCHEMA,
        metadata={"owner": "builtin"},
        tool_kind=ToolKind.OTHER,
        namespace="builtin.math",
        version="0.0.1",
        timeout_in_ms=5_000,
    )

    assert tool_spec_to_luca_tool(spec) == LucaTool(
        name="add",
        description="Add two numbers.",
        parameters=ADD_SCHEMA,
    )


def test_message_to_parts_keeps_the_thinking_signature():
    # dropping it here is what made a replayed Anthropic turn unacceptable
    parts = message_to_parts(
        LucaAssistantMessage(
            content=[
                ThinkingBlock(text="reasoning", signature="sig-abc"),
            ]
        ),
    )

    assert parts == [ThinkingContent(thinking="reasoning", signature="sig-abc")]


def test_message_to_parts_keeps_a_redacted_block():
    parts = message_to_parts(
        LucaAssistantMessage(
            content=[
                ThinkingBlock(text="", signature="encrypted", redacted=True),
            ]
        ),
    )

    assert parts == [
        ThinkingContent(thinking="", signature="encrypted", redacted=True),
    ]
