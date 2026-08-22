"""The shared iteration machinery: event order, the pending buffer,
exactly one terminal, the entry guards, and the post-exit accessors."""

import httpx
import pytest

from luca.client.exceptions import ConnectionError as ClientConnectionError, StreamError
from luca.client.transports import OpenAITransport
from luca.client.transports.faux import (
    faux_assistant_message,
    faux_error,
    faux_text,
    faux_tool_call,
)
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    FinishEvent,
    StartEvent,
    TextBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCall,
    Usage,
    UsageEvent,
    UserMessage,
)
from tests.client._helpers.httpx_mocks import make_sync_client, sse_response

from .conftest import drain, faux_astream, faux_stream

USAGE = Usage(input_tokens=3, output_tokens=5, total_tokens=8)


def test_the_full_event_sequence_arrives_in_order():
    scripted = faux_assistant_message([faux_text("Hello")], finish_reason="stop", usage=USAGE)

    with faux_stream(scripted) as s:
        events = list(s)

    assert events == [
        StartEvent(),
        TextStartEvent(index=0),
        TextDeltaEvent(index=0, delta="Hello"),
        TextEndEvent(index=0, content="Hello"),
        UsageEvent(usage=USAGE),
        FinishEvent(
            message=AssistantMessage(
                content=[TextBlock(text="Hello")],
                finish_reason="stop",
                provider_finish_reason="stop",
                provider="faux",
                model="test-model",
                usage=USAGE,
            ),
            finish_reason="stop",
            provider_finish_reason="stop",
            usage=USAGE,
            tool_calls=[],
        ),
    ]


def test_one_wire_read_fans_out_through_the_pending_buffer(openai_transport_factory):
    # One CC chunk carries the text delta, the finish reason, and usage:
    # five public events from a single read, delivered one per pull.
    chunks = [
        (
            b'data: {"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
        ),
        b"data: [DONE]\n\n",
    ]
    transport = openai_transport_factory(http_client=make_sync_client(sse_response(chunks)))
    request = ChatCompletionRequest(model="gpt-4o", provider="openai", messages=[UserMessage(content="hi")])

    with transport.completion_stream(request) as s:
        events = list(s)

    assert [e.type for e in events] == ["start", "text_start", "text_delta", "text_end", "usage", "finish"]


def test_a_scripted_error_is_exactly_one_terminal_error_event():
    scripted = faux_assistant_message([faux_text("partial")], error=faux_error("boom"))

    with faux_stream(scripted) as s:
        events = list(s)

    assert [e.type for e in events] == ["start", "text_start", "text_delta", "text_end", "error"]
    assert isinstance(events[-1].error, StreamError)
    assert str(events[-1].error) == "boom"


def test_the_error_event_carries_the_last_usage_seen(openai_transport_factory):
    chunks = [
        b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n\n',
        b"data: {not json\n\n",
    ]
    transport = openai_transport_factory(http_client=make_sync_client(sse_response(chunks)))
    request = ChatCompletionRequest(model="gpt-4o", provider="openai", messages=[UserMessage(content="hi")])

    with transport.completion_stream(request) as s:
        events = list(s)

    assert [e.type for e in events] == ["start", "usage", "error"]
    assert events[-1].usage == Usage(input_tokens=1, output_tokens=2, total_tokens=3)


def test_an_immediate_wire_failure_still_narrates_start_then_terminal(openai_transport_factory):
    # [DONE] with no finish reason before it: the wire produced nothing, and
    # the stream still opens its narration before the terminal.
    transport = openai_transport_factory(http_client=make_sync_client(sse_response([b"data: [DONE]\n\n"])))
    request = ChatCompletionRequest(model="gpt-4o", provider="openai", messages=[UserMessage(content="hi")])

    with transport.completion_stream(request) as s:
        events = list(s)

    assert [e.type for e in events] == ["start", "error"]
    assert "without a terminal event" in str(events[-1].error)


def test_a_mid_stream_transport_error_maps_to_a_terminal_error_event(openai_transport_factory):
    class ExplodingBody(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n'
            raise httpx.ReadError("connection lost")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ExplodingBody(), headers={"content-type": "text/event-stream"})

    transport = openai_transport_factory(http_client=make_sync_client(handler))
    request = ChatCompletionRequest(model="gpt-4o", provider="openai", messages=[UserMessage(content="hi")])

    with transport.completion_stream(request) as s:
        events = list(s)

    assert [e.type for e in events] == ["start", "text_start", "text_delta", "error"]
    assert isinstance(events[-1].error, ClientConnectionError)
    assert events[-1].error.provider == "openai"
    assert s.message.content == [TextBlock(text="Hi")]


def test_the_streamer_stamps_the_instance_provider_label():
    # A generic OpenAI-compatible host rides OpenAITransport; the message
    # carries the HOST's name, never the wire class's.
    chunks = [
        b'data: {"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    transport = OpenAITransport(
        provider="deepseek",
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test",
        http_client=make_sync_client(sse_response(chunks)),
    )
    request = ChatCompletionRequest(model="deepseek-chat", provider="deepseek", messages=[UserMessage(content="hi")])

    with transport.completion_stream(request) as s:
        events = list(s)

    assert events[-1].message.provider == "deepseek"
    assert s.message.provider == "deepseek"


def test_a_consumed_stream_refuses_re_iteration():
    with faux_stream(faux_assistant_message([faux_text("once")])) as s:
        list(s)
        with pytest.raises(StreamError, match="already been consumed"):
            list(s)


async def test_a_consumed_async_stream_refuses_re_iteration():
    async with faux_astream(faux_assistant_message([faux_text("once")])) as s:
        await drain(s)
        with pytest.raises(StreamError, match="already been consumed"):
            await drain(s)


def test_iterating_a_never_entered_stream_raises():
    s = faux_stream(faux_assistant_message([faux_text("hi")]))

    with pytest.raises(StreamError, match="never entered"):
        next(iter(s))


async def test_iterating_a_never_entered_async_stream_raises():
    s = faux_astream(faux_assistant_message([faux_text("hi")]))

    with pytest.raises(StreamError, match="never entered"):
        await anext(s.__aiter__())


def test_accessors_stay_readable_after_the_stream_closes():
    scripted = faux_assistant_message(
        [faux_text("Hi"), faux_tool_call("add", {"a": 1}, id="t1")],
        finish_reason="tool_use",
        usage=USAGE,
    )

    with faux_stream(scripted) as s:
        list(s)

    call = ToolCall(id="t1", name="add", arguments={"a": 1}, complete=True)
    assert s.message.content == [TextBlock(text="Hi"), call]
    assert s.text == "Hi"
    assert s.tool_calls == [call]
    assert s.usage == USAGE
    assert s.finish_reason == "tool_use"
    assert s.provider_finish_reason == "tool_use"
    assert s.cancelled is False
