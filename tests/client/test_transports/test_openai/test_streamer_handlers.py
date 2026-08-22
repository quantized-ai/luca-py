"""OpenAIChatCompletionsStreamer.handle() — one raw chunk in, exactly these
public events out, plus the message state that goes with them. The streamer
is never entered: handling is wire knowledge, independent of iteration."""

import pytest

from luca.client.exceptions import StreamError
from luca.client.transports.openai.streamer import SyncOpenAIChatCompletionsStreamer
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    FinishEvent,
    StartEvent,
    TextBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingBlock,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallStartEvent,
    Usage,
    UsageEvent,
    UserMessage,
)

REQUEST = ChatCompletionRequest(
    model="gpt-4o",
    provider="openai",
    messages=[UserMessage(content="hi")],
)


def _streamer() -> SyncOpenAIChatCompletionsStreamer:
    return SyncOpenAIChatCompletionsStreamer(REQUEST)


def test_the_first_chunk_opens_the_stream_and_the_text_block():
    s = _streamer()

    events = s.handle({"choices": [{"index": 0, "delta": {"content": "Hi"}}]})

    assert events == [StartEvent(), TextStartEvent(index=0), TextDeltaEvent(index=0, delta="Hi")]
    assert s.message.content == [TextBlock(text="Hi")]


def test_reasoning_deltas_stream_into_a_thinking_block():
    s = _streamer()

    events = s.handle({"choices": [{"index": 0, "delta": {"reasoning": "hmm"}}]})

    assert events == [StartEvent(), ThinkingStartEvent(index=0), ThinkingDeltaEvent(index=0, delta="hmm")]
    assert s.message.content == [ThinkingBlock(text="hmm")]


def test_reasoning_content_is_the_deepseek_spelling_of_the_same_field():
    s = _streamer()

    events = s.handle({"choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}}]})

    assert events == [StartEvent(), ThinkingStartEvent(index=0), ThinkingDeltaEvent(index=0, delta="hmm")]
    assert s.message.content == [ThinkingBlock(text="hmm")]


def test_tool_arguments_before_the_id_and_name_are_buffered_then_flushed():
    s = _streamer()

    first = s.handle(
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"a"'}}]}}]}
    )
    second = s.handle(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "add", "arguments": ":1}"}}]
                    },
                }
            ]
        }
    )

    assert first == [StartEvent()]
    assert second == [
        ToolCallStartEvent(index=0, id="call_1", name="add"),
        ToolCallDeltaEvent(index=0, arguments_delta='{"a"'),
        ToolCallDeltaEvent(index=0, arguments_delta=":1}"),
    ]
    assert s.message.content == [
        ToolCall(id="call_1", name="add", arguments={}, partial_arguments='{"a":1}', complete=False)
    ]


def test_the_finish_chunk_closes_every_open_block_in_order():
    s = _streamer()
    s.handle({"choices": [{"index": 0, "delta": {"reasoning": "think"}}]})
    s.handle({"choices": [{"index": 0, "delta": {"content": "Hi"}}]})

    events = s.handle({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})

    assert events == [ThinkingEndEvent(index=0, content="think"), TextEndEvent(index=1, content="Hi")]
    assert s.stop_reason == "stop"


def test_a_tool_that_never_resolved_fails_the_finish():
    s = _streamer()
    s.handle({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]}}]})

    with pytest.raises(StreamError, match="missing id or name"):
        s.handle({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})


def test_malformed_tool_arguments_fail_at_the_close():
    s = _streamer()
    s.handle(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "add", "arguments": '{"a":'}}]
                    },
                }
            ]
        }
    )

    with pytest.raises(StreamError, match="malformed JSON"):
        s.handle({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})


def test_a_mid_stream_error_chunk_raises_with_the_wire_message():
    s = _streamer()

    with pytest.raises(StreamError, match="upstream fell over"):
        s.handle({"error": {"code": 502, "message": "upstream fell over"}, "choices": []})


def test_the_usage_chunk_is_latched_and_surfaced():
    s = _streamer()
    usage = Usage(input_tokens=1, output_tokens=2, total_tokens=3, reasoning_tokens=1)

    events = s.handle(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
        }
    )

    assert events == [StartEvent(), UsageEvent(usage=usage)]
    assert s.usage == usage


def test_done_routes_to_the_wire_end_and_builds_the_finish():
    s = _streamer()
    s.handle({"choices": [{"index": 0, "delta": {"content": "Hi"}}]})
    s.handle({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})

    assert s.parse("data: [DONE]") == [{"done": True}]
    assert s.handle({"done": True}) == [
        FinishEvent(
            message=AssistantMessage(
                content=[TextBlock(text="Hi")],
                finish_reason="stop",
                provider_finish_reason="stop",
                provider="openai",
                model="gpt-4o",
                usage=Usage(),
            ),
            finish_reason="stop",
            provider_finish_reason="stop",
            usage=Usage(),
            tool_calls=[],
        )
    ]


def test_the_wire_end_without_a_finish_reason_reports_nothing():
    assert _streamer().handle_wire_end() == []
