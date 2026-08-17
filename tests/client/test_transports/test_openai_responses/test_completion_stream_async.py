"""OpenAIResponsesTransport.acompletion_stream() — the async path over the same cases."""

import pytest

from tests.client._helpers.httpx_mocks import make_async_client, sse_response

from .test_completion_stream_sync import CASES


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
async def test_responses_transport_acompletion_stream(case, responses_transport_factory):
    client = make_async_client(sse_response(case.sse_chunks))
    transport = responses_transport_factory(async_http_client=client)
    try:
        async with transport.acompletion_stream(case.request) as s:
            events = [event async for event in s]
    finally:
        await transport.aclose()

    assert events == case.expected_events
