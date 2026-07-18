"""Stream event union basics + accumulator semantics."""

import pytest

from luca.client.types import (
    AssistantMessage,
    FinishEvent,
    StartEvent,
    StreamEvent,
    TextBlock,
    TextDeltaEvent,
    ToolCall,
    Usage,
)
from luca.client.types.streaming import (
    RawBlockStart,
    RawBlockStop,
    RawFinish,
    RawTextDelta,
    RawToolArgumentsDelta,
    RawUsage,
    _ChatCompletionAccumulator,
)


class _FakeRequest:
    model = "test-model"
    response_format = None


def _classify(provider_value, message):
    """Simple test classifier — pass-through for 'stop', collapse 'content_filter' to error."""
    if provider_value == "content_filter":
        return ("error", "Provider safety filter (content_filter)")
    if provider_value == "stop":
        return ("stop", None)
    if provider_value == "tool_calls":
        return ("tool_use", None)
    if provider_value == "length":
        return ("length", None)
    return (provider_value, None)


def test_accumulator_text_block_lifecycle():
    acc = _ChatCompletionAccumulator(request=_FakeRequest(), provider="test")
    events = []
    events += list(acc.handle_raw(RawBlockStart(index=0, block_type="text")))
    events += list(acc.handle_raw(RawTextDelta(index=0, text="Hi")))
    events += list(acc.handle_raw(RawBlockStop(index=0)))
    events += list(acc.handle_raw(RawFinish(reason="stop")))

    assert events[0].type == "text_start"
    assert events[1].type == "text_delta"
    assert events[1].delta == "Hi"
    assert events[2].type == "text_end"
    assert events[2].content == "Hi"
    assert acc._terminal == "stop"


def test_accumulator_finish_event_canonicalizes():
    acc = _ChatCompletionAccumulator(request=_FakeRequest(), provider="test")
    list(acc.handle_raw(RawBlockStart(index=0, block_type="text")))
    list(acc.handle_raw(RawTextDelta(index=0, text="hi")))
    list(acc.handle_raw(RawBlockStop(index=0)))
    list(acc.handle_raw(RawFinish(reason="content_filter")))

    finish = acc.build_terminal_finish(classify_finish=_classify)
    assert finish.finish_reason == "error"
    assert finish.provider_finish_reason == "content_filter"
    assert finish.message.error_message == "Provider safety filter (content_filter)"


def test_accumulator_tool_call_assembles_arguments():
    acc = _ChatCompletionAccumulator(request=_FakeRequest(), provider="test")
    list(acc.handle_raw(RawBlockStart(index=0, block_type="tool_call", tool_id="c1", tool_name="get_weather")))
    list(acc.handle_raw(RawToolArgumentsDelta(index=0, arguments_delta='{"loca')))
    list(acc.handle_raw(RawToolArgumentsDelta(index=0, arguments_delta='tion":"NYC"}')))
    end_events = list(acc.handle_raw(RawBlockStop(index=0)))

    assert end_events[0].type == "tool_call_end"
    tc = end_events[0].tool_call
    assert tc.arguments == {"location": "NYC"}
    assert tc.complete is True
    # Same instance as in content.
    assert acc._message.content[0] is tc


def test_accumulator_dense_index_violation_raises_stream_error():
    from luca.client.exceptions import StreamError

    acc = _ChatCompletionAccumulator(request=_FakeRequest(), provider="test")
    with pytest.raises(StreamError):
        list(acc.handle_raw(RawBlockStart(index=5, block_type="text")))
