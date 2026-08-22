"""cancel(): a terminal FinishEvent(cancelled=True, finish_reason=None)
carrying the partial message and any usage already seen — a finish, not an
error."""

import asyncio

import httpx
import pytest

from luca.client.exceptions import StreamError
from luca.client.transports.faux import faux_assistant_message, faux_hang, faux_text
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    FinishEvent,
    TextBlock,
    Usage,
    UserMessage,
)
from tests.client._helpers.httpx_mocks import make_sync_client, sse_response

from .conftest import drain, faux_astream, faux_stream

CANCELLED_FINISH = FinishEvent(
    message=AssistantMessage(
        content=[TextBlock(text="Hello")],
        finish_reason=None,
        provider_finish_reason=None,
        cancelled=True,
        provider="faux",
        model="test-model",
        usage=Usage(),
    ),
    finish_reason=None,
    provider_finish_reason=None,
    cancelled=True,
    usage=Usage(),
    tool_calls=[],
)


def test_cancel_mid_stream_yields_a_cancelled_finish():
    with faux_stream(faux_assistant_message([faux_text("Hello")])) as s:
        it = iter(s)
        head = [next(it), next(it), next(it)]
        s.cancel()
        tail = list(it)

    assert [e.type for e in head] == ["start", "text_start", "text_delta"]
    assert tail == [CANCELLED_FINISH]
    assert s.cancelled is True


async def test_async_cancel_between_events_yields_a_cancelled_finish():
    async with faux_astream(faux_assistant_message([faux_text("Hello")])) as s:
        head = [await anext(s), await anext(s), await anext(s)]
        await s.cancel()
        tail = await drain(s)

    assert [e.type for e in head] == ["start", "text_start", "text_delta"]
    assert tail == [CANCELLED_FINISH]
    assert s.cancelled is True


async def test_async_cancel_interrupts_a_read_in_flight():
    scripted = faux_assistant_message([faux_text("Hi"), faux_hang()])

    async with faux_astream(scripted) as s:
        consumer = asyncio.ensure_future(drain(s))
        await asyncio.sleep(0.05)  # the consumer is parked at the hang
        await s.cancel()
        events = await consumer

    assert [e.type for e in events] == ["start", "text_start", "text_delta", "text_end", "finish"]
    assert events[-1].cancelled is True
    assert events[-1].finish_reason is None
    assert events[-1].message.content == [TextBlock(text="Hi")]


async def test_async_cancel_during_the_open_raises_a_stream_error(openai_transport_factory):
    # The stream never opened, so there is no cancelled finish to emit — and
    # the consumer task, which nobody cancelled, must not see a bare
    # CancelledError.
    async def stalled(request: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(stalled))
    transport = openai_transport_factory(async_http_client=client)
    request = ChatCompletionRequest(model="gpt-4o", provider="openai", messages=[UserMessage(content="hi")])
    stream = transport.acompletion_stream(request)

    async def enter():
        async with stream:
            pass

    consumer = asyncio.ensure_future(enter())
    await asyncio.sleep(0.05)  # parked in the send
    await stream.cancel()
    with pytest.raises(StreamError, match="cancelled before it opened"):
        await consumer

    assert not consumer.cancelled()
    await client.aclose()


def test_cancel_after_the_finish_is_a_no_op():
    with faux_stream(faux_assistant_message([faux_text("done")])) as s:
        events = list(s)
        s.cancel()
        with pytest.raises(StopIteration):
            next(s)

    assert events[-1].type == "finish"
    assert events[-1].cancelled is False


def test_a_cancelled_finish_carries_the_usage_already_seen(openai_transport_factory):
    chunks = [
        b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    transport = openai_transport_factory(http_client=make_sync_client(sse_response(chunks)))
    request = ChatCompletionRequest(model="gpt-4o", provider="openai", messages=[UserMessage(content="hi")])

    with transport.completion_stream(request) as s:
        it = iter(s)
        head = [next(it), next(it)]
        s.cancel()
        tail = list(it)

    assert [e.type for e in head] == ["start", "usage"]
    assert [e.type for e in tail] == ["finish"]
    assert tail[0].cancelled is True
    assert tail[0].usage == Usage(input_tokens=1, output_tokens=2, total_tokens=3)
