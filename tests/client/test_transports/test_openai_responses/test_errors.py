"""HTTP error → typed ClientError mapping on the Responses transport.

The mapping is shared with the chat-completions transport via
`OpenAIErrorMappingMixin` (same envelope, same status codes). This file is what
keeps the shared mixin honest for BOTH transports — an MRO mistake in either
class shows up as a NotImplementedError here."""

import httpx
import pytest

from luca.client.exceptions import (
    ConnectionError as ClientConnectionError,
    TimeoutError as ClientTimeoutError,
)
from luca.client.types import ChatCompletionRequest, UserMessage
from tests.client._helpers.httpx_mocks import error_response, make_sync_client

from ..test_openai.test_errors import CASES

REQUEST = ChatCompletionRequest(
    model="gpt-5.4",
    provider="openai",
    messages=[UserMessage(content="hi")],
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_responses_transport_http_error_mapping(case, responses_transport_factory):
    client = make_sync_client(error_response(case.status_code, case.body, case.headers))
    transport = responses_transport_factory(http_client=client)

    with pytest.raises(case.expected_exc) as exc_info:
        transport.completion(REQUEST)

    assert exc_info.value.provider == "openai"
    assert exc_info.value.original_exception is not None
    if case.extra_assertion is not None:
        assert case.extra_assertion(exc_info.value)


def test_responses_transport_timeout_maps_to_timeout_error(responses_transport_factory):
    def handler(request):
        raise httpx.TimeoutException("timeout", request=request)

    transport = responses_transport_factory(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ClientTimeoutError) as exc_info:
        transport.completion(REQUEST)
    assert exc_info.value.provider == "openai"


def test_responses_transport_connection_error_maps(responses_transport_factory):
    def handler(request):
        raise httpx.ConnectError("conn refused", request=request)

    transport = responses_transport_factory(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ClientConnectionError) as exc_info:
        transport.completion(REQUEST)
    assert exc_info.value.provider == "openai"
