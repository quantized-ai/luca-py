"""AnthropicStreamer handlers — one raw wire event in, exactly these public
events out, plus the message state. The streamer is never entered."""

import pytest

from luca.client.exceptions import StreamError
from luca.client.transports.anthropic.streamer import SyncAnthropicStreamer
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    FinishEvent,
    StartEvent,
    TextBlock,
    TextStartEvent,
    ThinkingBlock,
    ThinkingDeltaEvent,
    ThinkingStartEvent,
    Usage,
    UsageEvent,
    UserMessage,
)

REQUEST = ChatCompletionRequest(
    model="claude-test",
    provider="anthropic",
    messages=[UserMessage(content="hi")],
)

_MESSAGE_START = {
    "type": "message_start",
    "message": {"id": "msg_1", "usage": {"input_tokens": 5, "output_tokens": 0}},
}


def _streamer() -> SyncAnthropicStreamer:
    return SyncAnthropicStreamer(REQUEST)


def test_message_start_opens_the_stream_and_latches_input_tokens():
    s = _streamer()

    events = s.handle(_MESSAGE_START)

    assert events == [StartEvent()]
    assert s.input_tokens == 5


def test_an_unknown_block_type_is_ignored_with_its_deltas_and_stop():
    s = _streamer()
    s.handle(_MESSAGE_START)

    opened = s.handle(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "crystal_ball_use", "id": "cb_1", "name": "scry"},
        }
    )
    delta = s.handle(
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}}
    )
    stopped = s.handle({"type": "content_block_stop", "index": 0})
    # The next KNOWN block still lands at the next dense content index.
    text = s.handle({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}})

    assert opened == []
    assert delta == []
    assert stopped == []
    assert text == [TextStartEvent(index=0)]
    assert s.message.content == [TextBlock(text="")]


def test_a_redacted_thinking_block_arrives_whole_at_the_start():
    s = _streamer()
    s.handle(_MESSAGE_START)

    events = s.handle(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "redacted_thinking", "data": "encrypted-payload"},
        }
    )

    assert events == [ThinkingStartEvent(index=0)]
    assert s.message.content == [ThinkingBlock(text="", signature="encrypted-payload", redacted=True)]


def test_the_signature_delta_lands_on_the_block_as_a_zero_text_delta():
    s = _streamer()
    s.handle(_MESSAGE_START)
    s.handle({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}})

    events = s.handle(
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-abc"}}
    )

    assert events == [ThinkingDeltaEvent(index=0, delta="")]
    assert s.message.content == [ThinkingBlock(text="", signature="sig-abc")]


def test_a_tool_use_block_without_an_id_is_a_stream_error():
    s = _streamer()
    s.handle(_MESSAGE_START)

    with pytest.raises(StreamError, match="without an id or name"):
        s.handle({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "name": "add"}})


def test_malformed_tool_arguments_fail_at_the_block_stop():
    s = _streamer()
    s.handle(_MESSAGE_START)
    s.handle(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "add"},
        }
    )
    s.handle(
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"a":'}}
    )

    with pytest.raises(StreamError, match="malformed JSON"):
        s.handle({"type": "content_block_stop", "index": 0})


def test_message_stop_emits_usage_then_the_finish():
    s = _streamer()
    s.handle(_MESSAGE_START)
    s.handle({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}})

    events = s.handle({"type": "message_stop"})

    usage = Usage(input_tokens=5, output_tokens=2, total_tokens=7)
    assert events == [
        UsageEvent(usage=usage),
        FinishEvent(
            message=AssistantMessage(
                content=[],
                finish_reason="stop",
                provider_finish_reason="end_turn",
                provider="anthropic",
                model="claude-test",
                usage=usage,
            ),
            finish_reason="stop",
            provider_finish_reason="end_turn",
            usage=usage,
            tool_calls=[],
        ),
    ]


def test_message_stop_without_a_stop_reason_emits_usage_only():
    s = _streamer()
    s.handle(_MESSAGE_START)

    events = s.handle({"type": "message_stop"})

    assert events == [UsageEvent(usage=Usage(input_tokens=5, output_tokens=0, total_tokens=5))]


def test_an_error_event_raises_with_the_wire_message():
    s = _streamer()

    with pytest.raises(StreamError, match="Overloaded"):
        s.handle({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})


def test_message_stop_after_a_pause_turn_delta_finishes_with_pause():
    # the classification happens once, in the shared finish_event() through
    # the inherited wire mixin — the streamer needs no pause code of its own
    s = _streamer()
    s.handle(_MESSAGE_START)
    s.handle({"type": "message_delta", "delta": {"stop_reason": "pause_turn"}, "usage": {"output_tokens": 2}})

    events = s.handle({"type": "message_stop"})

    usage = Usage(input_tokens=5, output_tokens=2, total_tokens=7)
    assert events == [
        UsageEvent(usage=usage),
        FinishEvent(
            message=AssistantMessage(
                content=[],
                finish_reason="pause",
                provider_finish_reason="pause_turn",
                provider="anthropic",
                model="claude-test",
                usage=usage,
            ),
            finish_reason="pause",
            provider_finish_reason="pause_turn",
            usage=usage,
            tool_calls=[],
        ),
    ]
