"""Smoke tests for the adapter's two remaining translations: a KNOWN client
assistant message renders into KNOWN agent message parts (the inbound
direction), and a KNOWN `ToolSpec` projects to a KNOWN client Tool definition.
Declarative — hardcoded invariant in, full expected out. No logic, no helpers.
(Conversation → LLM-message projection lives in `test_projection.py`.)
"""

import pytest

from luca.agent.core.adapter import message_to_parts, tool_spec_to_luca_tool
from luca.agent.core.exceptions import AgentError
from luca.agent.core.models import (
    AssistantMessage as AgentAssistantMessage,
    LLMConfig,
    PrivateProviderContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolKind,
    ToolSpec,
    URLCitation,
    WebFetchContent,
    WebPageContent,
    WebSearchContent,
)
from luca.agent.core.projection import ConversationProjector
from luca.client.types import (
    AssistantMessage as LucaAssistantMessage,
    PrivateProviderBlock,
    TextBlock,
    ThinkingBlock,
    Tool as LucaTool,
    ToolCall as LucaToolCall,
    ToolResultBlock,
    URLCitationAnnotation,
    WebFetchBlock,
    WebPagePart,
    WebSearchBlock,
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
    # Order is the client's positional contract: privates beside their
    # portable block, split cited spans after their merged text.
    message = LucaAssistantMessage(
        content=[
            ThinkingBlock(text="Let me add."),
            PrivateProviderBlock(format="openai.responses", data={"id": "ws_1", "type": "web_search_call"}),
            WebSearchBlock(queries=["adding"]),
            TextBlock(text="Sure —"),
            LucaToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        ],
    )

    assert message_to_parts(message) == [
        ThinkingContent(thinking="Let me add."),
        PrivateProviderContent(format="openai.responses", data={"id": "ws_1", "type": "web_search_call"}),
        WebSearchContent(queries=["adding"]),
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


def test_message_to_parts_keeps_the_reasoning_item_id():
    # OpenAI's Responses API replays a reasoning item only when its `rs_…` id
    # comes back with the encrypted payload; dropping the id here loses half
    # the identity and the block becomes unreplayable.
    parts = message_to_parts(
        LucaAssistantMessage(
            content=[
                ThinkingBlock(text="reasoning", id="rs_1", signature="enc-1"),
            ]
        ),
    )

    assert parts == [ThinkingContent(thinking="reasoning", id="rs_1", signature="enc-1")]


def test_a_private_provider_block_converts_verbatim():
    parts = message_to_parts(
        LucaAssistantMessage(
            content=[
                PrivateProviderBlock(
                    format="anthropic.messages",
                    data={"type": "server_tool_use", "id": "srvtoolu_1", "input": {"query": "apple"}},
                ),
            ]
        ),
    )

    assert parts == [
        PrivateProviderContent(
            format="anthropic.messages",
            data={"type": "server_tool_use", "id": "srvtoolu_1", "input": {"query": "apple"}},
        ),
    ]


def test_a_web_search_block_converts_with_its_results_and_extras():
    parts = message_to_parts(
        LucaAssistantMessage(
            content=[
                WebSearchBlock(
                    queries=["apple results"],
                    results=[WebPagePart(url="https://apple.com", title="Apple", content="snippet")],
                    extras={"id": "srvtoolu_1"},
                ),
            ]
        ),
    )

    assert parts == [
        WebSearchContent(
            queries=["apple results"],
            results=[WebPageContent(url="https://apple.com", title="Apple", content="snippet")],
            extras={"id": "srvtoolu_1"},
        ),
    ]


def test_a_resultless_web_search_block_keeps_none_not_empty():
    # results=None (metadata not returned) and results=[] (empty result set)
    # are different facts; the conversion must not conflate them.
    parts = message_to_parts(
        LucaAssistantMessage(content=[WebSearchBlock(queries=["apple"], results=None)]),
    )

    assert parts == [WebSearchContent(queries=["apple"], results=None)]


def test_a_web_fetch_block_converts_with_its_page():
    parts = message_to_parts(
        LucaAssistantMessage(
            content=[
                WebFetchBlock(
                    web_page=WebPagePart(url="https://apple.com", title="Apple", content="Page text."),
                    extras={"id": "ws_2"},
                ),
            ]
        ),
    )

    assert parts == [
        WebFetchContent(
            web_page=WebPageContent(url="https://apple.com", title="Apple", content="Page text."),
            extras={"id": "ws_2"},
        ),
    ]


def test_text_annotations_survive_the_conversion():
    parts = message_to_parts(
        LucaAssistantMessage(
            content=[
                TextBlock(
                    text="Apple rose 2.8% today.",
                    annotations=[
                        URLCitationAnnotation(
                            url="https://example.com/apple",
                            title="Apple shares rise",
                            start_index=0,
                            end_index=22,
                        )
                    ],
                ),
            ]
        ),
    )

    assert parts == [
        TextContent(
            text="Apple rose 2.8% today.",
            annotations=[
                URLCitation(
                    url="https://example.com/apple",
                    title="Apple shares rise",
                    start_index=0,
                    end_index=22,
                )
            ],
        ),
    ]


def test_an_unknown_client_block_raises():
    # `model_construct` bypasses the content union so a block the adapter has
    # no branch for (here a ToolResultBlock standing in for whatever the
    # client ships next) actually reaches the fall-through.
    message = LucaAssistantMessage.model_construct(
        content=[ToolResultBlock(tool_call_id="tc1", content="output")],
    )

    with pytest.raises(AgentError, match="tool_result"):
        message_to_parts(message)


def test_adapter_and_projector_are_inverse_on_the_new_parts():
    # Round-trip identity, scoped to the three new part types + annotated
    # text (whole-message identity is impossible by design: signatures drop
    # inbound, ToolCall.extras are not projected).
    parts = [
        PrivateProviderContent(format="openai.responses", data={"id": "ws_1", "type": "web_search_call"}),
        WebSearchContent(
            queries=["apple"],
            results=[WebPageContent(url="https://apple.com", title="Apple")],
            extras={"id": "ws_1"},
        ),
        WebFetchContent(web_page=WebPageContent(url="https://apple.com"), extras={"id": "ws_2"}),
        TextContent(
            text="Apple rose.",
            annotations=[URLCitation(url="https://apple.com", title="Apple", start_index=0, end_index=11)],
        ),
    ]
    entry = AgentAssistantMessage(
        id="a1",
        created_at=1,
        parts=parts,
        llm_config=LLMConfig(model="gpt-5.1", provider="openai"),
        stop_reason="stop",
    )

    projected = ConversationProjector().project_assistant_message(entry, {})

    assert message_to_parts(projected) == parts
