"""BedrockStreamer handlers — one Converse event in, exactly these public
events out, plus the message state. The streamer is never entered. Frame
decoding is covered separately in test_streamer_parse.py."""

import pytest

from luca.client.exceptions import StreamError
from luca.client.transports.bedrock.streamer import SyncBedrockStreamer
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    FinishEvent,
    StartEvent,
    TextBlock,
    TextDeltaEvent,
    TextStartEvent,
    ThinkingBlock,
    ThinkingDeltaEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
    UsageEvent,
    UserMessage,
)

REQUEST = ChatCompletionRequest(
    model="us.amazon.nova-pro-v1:0",
    provider="bedrock",
    messages=[UserMessage(content="hi")],
)


def _streamer() -> SyncBedrockStreamer:
    return SyncBedrockStreamer(REQUEST)


def test_message_start_opens_the_stream():
    assert _streamer().handle({"type": "messageStart", "role": "assistant"}) == [StartEvent()]


def test_a_text_block_synthesizes_its_start_on_the_first_delta():
    s = _streamer()

    events = s.handle({"type": "contentBlockDelta", "contentBlockIndex": 0, "delta": {"text": "Hi"}})

    assert events == [TextStartEvent(index=0), TextDeltaEvent(index=0, delta="Hi")]
    assert s.message.content == [TextBlock(text="Hi")]


def test_a_reasoning_block_synthesizes_its_start_and_takes_the_signature():
    s = _streamer()

    text = s.handle(
        {"type": "contentBlockDelta", "contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "hmm"}}}
    )
    signature = s.handle(
        {"type": "contentBlockDelta", "contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig-1"}}}
    )

    assert text == [ThinkingStartEvent(index=0), ThinkingDeltaEvent(index=0, delta="hmm")]
    assert signature == [ThinkingDeltaEvent(index=0, delta="")]
    assert s.message.content == [ThinkingBlock(text="hmm", signature="sig-1")]


def test_a_redacted_reasoning_block_arrives_whole():
    s = _streamer()

    events = s.handle(
        {
            "type": "contentBlockDelta",
            "contentBlockIndex": 0,
            "delta": {"reasoningContent": {"redactedContent": "encrypted"}},
        }
    )

    assert events == [ThinkingStartEvent(index=0)]
    assert s.message.content == [ThinkingBlock(text="", signature="encrypted", redacted=True)]


def test_a_tool_use_block_streams_start_deltas_and_stop():
    s = _streamer()

    opened = s.handle(
        {
            "type": "contentBlockStart",
            "contentBlockIndex": 1,
            "start": {"toolUse": {"toolUseId": "tool_1", "name": "add"}},
        }
    )
    delta = s.handle({"type": "contentBlockDelta", "contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"a":1}'}}})
    stopped = s.handle({"type": "contentBlockStop", "contentBlockIndex": 1})

    call = ToolCall(id="tool_1", name="add", arguments={"a": 1}, complete=True)
    assert opened == [ToolCallStartEvent(index=0, id="tool_1", name="add")]
    assert delta == [ToolCallDeltaEvent(index=0, arguments_delta='{"a":1}')]
    assert stopped == [ToolCallEndEvent(index=0, tool_call=call)]
    assert s.message.content == [call]


def test_a_tool_use_block_without_an_id_is_a_stream_error():
    with pytest.raises(StreamError, match="without a toolUseId or name"):
        _streamer().handle({"type": "contentBlockStart", "contentBlockIndex": 0, "start": {"toolUse": {"name": "add"}}})


def test_metadata_carries_the_cache_token_counts():
    s = _streamer()
    usage = Usage(
        input_tokens=10,
        output_tokens=4,
        total_tokens=14,
        cached_input_tokens=6,
        cache_write_tokens=2,
    )

    events = s.handle(
        {
            "type": "metadata",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 4,
                "totalTokens": 14,
                "cacheReadInputTokens": 6,
                "cacheWriteInputTokens": 2,
            },
        }
    )

    assert events == [UsageEvent(usage=usage)]
    assert s.usage == usage


def test_the_finish_is_built_at_wire_end_from_the_latched_stop_reason():
    s = _streamer()
    s.handle({"type": "messageStart", "role": "assistant"})
    s.handle({"type": "contentBlockDelta", "contentBlockIndex": 0, "delta": {"text": "Hi"}})
    s.handle({"type": "contentBlockStop", "contentBlockIndex": 0})

    latched = s.handle({"type": "messageStop", "stopReason": "end_turn"})
    usage = s.handle({"type": "metadata", "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}})
    finish = s.handle_wire_end()

    assert latched == []
    assert [e.type for e in usage] == ["usage"]
    assert finish == [
        FinishEvent(
            message=AssistantMessage(
                content=[TextBlock(text="Hi")],
                finish_reason="stop",
                provider_finish_reason="end_turn",
                provider="bedrock",
                model="us.amazon.nova-pro-v1:0",
                usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
            ),
            finish_reason="stop",
            provider_finish_reason="end_turn",
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
            tool_calls=[],
        )
    ]


def test_the_wire_end_without_a_stop_reason_reports_nothing():
    assert _streamer().handle_wire_end() == []
