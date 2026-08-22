"""The async deadline — a call_later timer that cancels the streamer-owned
read task. Expiry mid-read and between events is one terminal ErrorEvent
carrying the SDK TimeoutError; expiry during the open raises it."""

import asyncio

import httpx
import pytest

from luca.client.exceptions import TimeoutError as SDKTimeoutError
from luca.client.transports.faux import faux_assistant_message, faux_hang, faux_text
from luca.client.types import ChatCompletionRequest, StartEvent, UserMessage

from .conftest import drain, faux_astream


async def test_expiry_interrupts_an_in_flight_read():
    scripted = faux_assistant_message([faux_text("Hel"), faux_hang()], finish_reason="stop")

    async with faux_astream(scripted, timeout=0.05) as s:
        events = await drain(s)

    assert [e.type for e in events] == ["start", "text_start", "text_delta", "text_end", "error"]
    error = events[-1].error
    assert isinstance(error, SDKTimeoutError)
    assert str(error) == "stream exceeded timeout=0.05s"
    assert error.provider == "faux"


async def test_expiry_between_events_is_caught_before_the_next_read():
    # The timer fires while the consumer holds control (no read in flight);
    # the flag is picked up at the top of the next pull.
    async with faux_astream(faux_assistant_message([faux_text("Hi")]), timeout=0.05) as s:
        first = await anext(s)
        await asyncio.sleep(0.1)
        rest = await drain(s)

    assert first == StartEvent()
    assert [e.type for e in rest] == ["error"]
    assert isinstance(rest[0].error, SDKTimeoutError)


async def test_expiry_during_the_open_raises(openai_transport_factory):
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        return httpx.Response(200, content=b"", headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(slow))
    transport = openai_transport_factory(async_http_client=client)
    request = ChatCompletionRequest(model="gpt-4o", provider="openai", messages=[UserMessage(content="hi")])

    with pytest.raises(SDKTimeoutError, match=r"request did not open within timeout=0\.05s"):
        async with transport.acompletion_stream(request, timeout=0.05):
            pass

    await client.aclose()


async def test_expiry_interrupts_a_stalled_error_body(openai_transport_factory):
    # The whole open — the error-body read included — rides the streamer's
    # own task, so a server that answers 429 headers and then stalls the
    # body cannot hold `async with` past the deadline.
    class StalledBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b""
            await asyncio.Event().wait()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, stream=StalledBody(), headers={"content-type": "application/json"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = openai_transport_factory(async_http_client=client)
    request = ChatCompletionRequest(model="gpt-4o", provider="openai", messages=[UserMessage(content="hi")])

    with pytest.raises(SDKTimeoutError, match=r"request did not open within timeout=0\.05s"):
        async with transport.acompletion_stream(request, timeout=0.05):
            pass

    await client.aclose()


async def test_a_generous_deadline_is_inert():
    async with faux_astream(faux_assistant_message([faux_text("Hi")]), timeout=60.0) as s:
        events = await drain(s)

    assert events[-1].type == "finish"
    assert not any(e.type == "error" for e in events)
