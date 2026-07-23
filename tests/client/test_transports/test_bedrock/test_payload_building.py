"""What BedrockTransport sends on the wire.

The payload builder is pure — no HTTP — so these call it directly and assert
the whole dict.
"""

import pytest

from luca.client.exceptions import BadRequestError, UnsupportedParameterError
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    TextBlock,
    ThinkingBlock,
    ToolMessage,
    UserMessage,
)
from luca.client.types.content import ToolCall
from luca.client.types.tools import Tool


def _build(transport, **kwargs):
    request = ChatCompletionRequest(provider="bedrock", **kwargs)
    return transport._build_chat_completion_payload(request)


def test_a_plain_user_turn_becomes_a_content_array(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    payload = _build(
        transport, model="us.amazon.nova-lite-v1:0",
        messages=[UserMessage(content="Hi")],
    )
    assert payload == {
        "messages": [{"role": "user", "content": [{"text": "Hi"}]}],
    }


def test_the_system_message_is_hoisted_to_a_top_level_array(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    payload = _build(
        transport, model="us.amazon.nova-lite-v1:0",
        messages=[UserMessage(content="Hi")],
        system_message="Be brief.",
    )
    assert payload == {
        "messages": [{"role": "user", "content": [{"text": "Hi"}]}],
        "system": [{"text": "Be brief."}],
    }


def test_the_model_id_is_only_in_the_url_never_the_body(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    request = ChatCompletionRequest(
        provider="bedrock", model="us.amazon.nova-lite-v1:0",
        messages=[UserMessage(content="Hi")],
    )
    assert "model" not in transport._build_chat_completion_payload(request)
    assert transport._chat_completion_url(request) == (
        "https://bedrock-runtime.us-east-1.amazonaws.com"
        "/model/us.amazon.nova-lite-v1:0/converse"
    )
    assert transport._chat_completion_url(request, stream=True).endswith("/converse-stream")


def test_sampling_and_stop_go_under_inference_config(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    payload = _build(
        transport, model="us.amazon.nova-lite-v1:0",
        messages=[UserMessage(content="Hi")],
        temperature=0.5, top_p=0.9, max_tokens=256, stop="END",
    )
    assert payload == {
        "messages": [{"role": "user", "content": [{"text": "Hi"}]}],
        "inferenceConfig": {
            "maxTokens": 256, "temperature": 0.5, "topP": 0.9,
            "stopSequences": ["END"],
        },
    }


def test_max_tokens_is_omitted_when_the_caller_does_not_set_one(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    payload = _build(
        transport, model="us.amazon.nova-lite-v1:0",
        messages=[UserMessage(content="Hi")],
    )
    assert "inferenceConfig" not in payload


def test_tool_arguments_are_a_json_object_not_a_string(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    payload = _build(
        transport, model="us.amazon.nova-lite-v1:0",
        messages=[
            UserMessage(content="17*23?"),
            AssistantMessage(content=[
                ToolCall(id="t1", name="multiply", arguments={"a": 17, "b": 23}),
            ]),
        ],
    )
    assert payload["messages"][1] == {
        "role": "assistant",
        "content": [{
            "toolUse": {
                "toolUseId": "t1", "name": "multiply",
                "input": {"a": 17, "b": 23},
            },
        }],
    }


def test_a_tool_definition_nests_the_schema_under_tool_spec(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    tool = Tool(
        name="multiply", description="Multiply two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    )
    payload = _build(
        transport, model="us.amazon.nova-lite-v1:0",
        messages=[UserMessage(content="17*23?")], tools=[tool],
    )
    assert payload["toolConfig"] == {
        "tools": [{
            "toolSpec": {
                "name": "multiply",
                "description": "Multiply two numbers",
                "inputSchema": {"json": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                }},
            },
        }],
    }


@pytest.mark.parametrize("choice, expected", [
    ("auto", {"auto": {}}),
    ("required", {"any": {}}),
    ({"name": "multiply"}, {"tool": {"name": "multiply"}}),
])
def test_tool_choice_maps_to_the_converse_shape(bedrock_transport_factory, choice, expected):
    transport = bedrock_transport_factory()
    tool = Tool(name="multiply", description="x", parameters={"type": "object", "properties": {}})
    payload = _build(
        transport, model="us.amazon.nova-lite-v1:0",
        messages=[UserMessage(content="Hi")], tools=[tool], tool_choice=choice,
    )
    assert payload["toolConfig"]["toolChoice"] == expected


def test_tool_choice_none_is_omitted_because_converse_has_no_equivalent(
    bedrock_transport_factory,
):
    transport = bedrock_transport_factory()
    tool = Tool(name="multiply", description="x", parameters={"type": "object", "properties": {}})
    payload = _build(
        transport, model="us.amazon.nova-lite-v1:0",
        messages=[UserMessage(content="Hi")], tools=[tool], tool_choice="none",
    )
    assert "toolChoice" not in payload["toolConfig"]


def test_a_tool_result_is_a_user_message_and_merges_with_a_following_user_turn(
    bedrock_transport_factory,
):
    transport = bedrock_transport_factory()
    payload = _build(
        transport, model="us.amazon.nova-lite-v1:0",
        messages=[
            UserMessage(content="17*23?"),
            AssistantMessage(content=[
                ToolCall(id="t1", name="multiply", arguments={"a": 17, "b": 23}),
            ]),
            ToolMessage(tool_call_id="t1", content="391"),
            UserMessage(content="thanks"),
        ],
    )
    # Converse needs strict alternation: the toolResult and the follow-up user
    # turn coalesce into one user message.
    assert payload["messages"][2] == {
        "role": "user",
        "content": [
            {"toolResult": {"toolUseId": "t1", "content": [{"text": "391"}]}},
            {"text": "thanks"},
        ],
    }


def test_a_failed_tool_result_carries_an_error_status(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    payload = _build(
        transport, model="us.amazon.nova-lite-v1:0",
        messages=[ToolMessage(tool_call_id="t1", content="boom", is_error=True)],
    )
    assert payload["messages"][0]["content"][0] == {
        "toolResult": {
            "toolUseId": "t1", "content": [{"text": "boom"}], "status": "error",
        },
    }


def test_a_signed_thinking_block_is_replayed_as_reasoning_content(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    payload = _build(
        transport, model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[AssistantMessage(content=[
            ThinkingBlock(text="work", signature="sig-abc"),
        ])],
    )
    assert payload["messages"][0]["content"][0] == {
        "reasoningContent": {
            "reasoningText": {"text": "work", "signature": "sig-abc"},
        },
    }


def test_an_unsigned_thinking_block_is_dropped(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    payload = _build(
        transport, model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[AssistantMessage(content=[
            ThinkingBlock(text="unsigned", signature=None),
            TextBlock(text="visible"),
        ])],
    )
    assert payload["messages"][0]["content"] == [{"text": "visible"}]


def test_reasoning_goes_into_additional_model_request_fields(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    payload = _build(
        transport, model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[UserMessage(content="Hi")], reasoning="low",
    )
    assert payload["additionalModelRequestFields"] == {
        "thinking": {"type": "enabled", "budget_tokens": 6_400},
    }
    assert payload["inferenceConfig"]["maxTokens"] == 7_424


def test_provider_options_win_over_resolved_reasoning(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    payload = _build(
        transport, model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[UserMessage(content="Hi")], reasoning="high",
        provider_options={"bedrock": {"additionalModelRequestFields": {"custom": 1}}},
    )
    assert payload["additionalModelRequestFields"] == {"custom": 1}


def test_temperature_on_a_thinking_model_with_reasoning_active_raises(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    with pytest.raises(UnsupportedParameterError):
        _build(
            transport, model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=[UserMessage(content="Hi")],
            reasoning="high", temperature=0.2,
        )


def test_top_k_is_refused_rather_than_silently_dropped(bedrock_transport_factory):
    transport = bedrock_transport_factory()
    with pytest.raises(UnsupportedParameterError, match="top_k"):
        _build(
            transport, model="us.amazon.nova-lite-v1:0",
            messages=[UserMessage(content="Hi")], top_k=40,
        )


def test_a_jpg_media_type_is_normalised_to_the_jpeg_converse_expects(bedrock_transport_factory):
    from luca.client.types.content import ImageBlock
    from luca.client.types.media import MediaBase64

    transport = bedrock_transport_factory()
    payload = _build(
        transport, model="us.amazon.nova-lite-v1:0",
        messages=[UserMessage(content=[
            ImageBlock(source=MediaBase64(data="AAAA", media_type="image/jpg")),
        ])],
    )
    assert payload["messages"][0]["content"][0] == {
        "image": {"format": "jpeg", "source": {"bytes": "AAAA"}},
    }


def test_a_url_image_cannot_be_sent_to_converse(bedrock_transport_factory):
    from luca.client.types.content import ImageBlock
    from luca.client.types.media import MediaURL

    transport = bedrock_transport_factory()
    with pytest.raises(BadRequestError):
        _build(
            transport, model="us.amazon.nova-lite-v1:0",
            messages=[UserMessage(content=[
                ImageBlock(source=MediaURL(url="https://example.com/x.png")),
            ])],
        )
