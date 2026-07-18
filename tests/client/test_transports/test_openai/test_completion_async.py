import pytest

from tests.client._helpers.httpx_mocks import json_response, make_async_client

from .test_completion_sync import CASES


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
async def test_openai_transport_acompletion(case, openai_transport_factory):
    client = make_async_client(json_response(case.mock_response_json))
    transport = openai_transport_factory(async_http_client=client)
    try:
        actual = await transport.acompletion(case.request)
    finally:
        await transport.aclose()
    expected = case.expected.model_copy(update={"raw": case.mock_response_json})
    assert actual == expected
