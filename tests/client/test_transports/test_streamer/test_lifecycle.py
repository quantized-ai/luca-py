"""Teardown hygiene, under the suite-wide -W error::ResourceWarning: aclose
idempotence, the agent runner's external-cancel pattern, and abandonment."""

import asyncio
import gc

import pytest

from luca.client.transports.faux import faux_assistant_message, faux_hang, faux_text
from luca.client.types import ChatCompletionRequest, StartEvent, UserMessage
from tests.client._helpers.httpx_mocks import make_sync_client, sse_response

from .conftest import faux_astream


async def test_aclose_is_idempotent():
    async with faux_astream(faux_assistant_message([faux_text("Hi")])) as s:
        await anext(s)
        await s.aclose()
        await s.aclose()


async def test_external_cancel_then_aclose_then_aexit_leaks_nothing():
    # The agent runner's teardown: a hard-cancelled __anext__ task, then
    # aclose(), then __aexit__. External cancellation re-raises untouched.
    async with faux_astream(faux_assistant_message([faux_hang()])) as s:
        first = await anext(s)
        step = asyncio.ensure_future(s.__anext__())
        await asyncio.sleep(0.05)  # the step is parked at the hang
        step.cancel()
        with pytest.raises(asyncio.CancelledError):
            await step
        await s.aclose()

    assert first == StartEvent()
    assert s.last_task.done()


def test_a_drained_real_stream_leaves_nothing_open_when_garbage_collected(openai_transport_factory):
    chunks = [
        b'data: {"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    transport = openai_transport_factory(http_client=make_sync_client(sse_response(chunks)))
    request = ChatCompletionRequest(model="gpt-4o", provider="openai", messages=[UserMessage(content="hi")])

    with transport.completion_stream(request) as s:
        list(s)

    del s
    gc.collect()  # a leaked response would warn here, and warnings are errors


async def test_an_abandoned_async_read_task_is_recovered_by_aexit():
    # No explicit aclose: __aexit__ alone must kill the in-flight read task.
    async with faux_astream(faux_assistant_message([faux_hang()])) as s:
        await anext(s)
        step = asyncio.ensure_future(s.__anext__())
        await asyncio.sleep(0.05)
        step.cancel()
        with pytest.raises(asyncio.CancelledError):
            await step

    assert s.last_task.done()
