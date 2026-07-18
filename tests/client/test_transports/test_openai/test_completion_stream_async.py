import pytest

from tests.client._helpers.httpx_mocks import make_async_client, sse_response
from tests.client._helpers.stream_iteration import acollect_events_with_snapshots

from .test_completion_stream_sync import CASES


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
async def test_openai_transport_acompletion_stream(case, openai_transport_factory):
    client = make_async_client(sse_response(case.sse_chunks))
    transport = openai_transport_factory(async_http_client=client)
    try:
        async with transport.acompletion_stream(case.request) as s:
            events = await acollect_events_with_snapshots(s)
        assert events == case.expected_events
    finally:
        await transport.aclose()
