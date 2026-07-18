"""HTTP error → typed ClientError mapping."""

from dataclasses import dataclass

import httpx
import pytest

from luca.client.exceptions import (
    AuthenticationError,
    BadRequestError,
    ContextLengthExceededError,
    ModelNotFoundError,
    ProviderAPIError,
    RateLimitError,
)
from luca.client.exceptions import ConnectionError as ClientConnectionError
from luca.client.exceptions import TimeoutError as ClientTimeoutError
from luca.client.types import ChatCompletionRequest, UserMessage
from tests.client._helpers.httpx_mocks import error_response, make_sync_client


REQUEST = ChatCompletionRequest(
    model="gpt-4o", provider="openai",
    messages=[UserMessage(content="hi")],
)


@dataclass(frozen=True)
class ErrorCase:
    name: str
    status_code: int
    body: dict
    headers: dict
    expected_exc: type
    extra_assertion: object = None


CASES = [
    ErrorCase("401_auth", 401,
              {"error": {"type": "invalid_request_error", "message": "Invalid key"}},
              {}, AuthenticationError),
    ErrorCase("429_with_retry_after", 429,
              {"error": {"type": "rate_limit_exceeded", "message": "Too many"}},
              {"retry-after": "30"}, RateLimitError,
              extra_assertion=lambda exc: exc.retry_after == 30.0),
    ErrorCase("400_context_length", 400,
              {"error": {"type": "context_length_exceeded", "message": "..."}},
              {}, ContextLengthExceededError),
    ErrorCase("400_generic", 400,
              {"error": {"type": "invalid_request_error", "message": "..."}},
              {}, BadRequestError),
    ErrorCase("404_model_not_found", 404,
              {"error": {"type": "not_found", "message": "..."}},
              {}, ModelNotFoundError),
    ErrorCase("500_server_error", 500,
              {"error": {"type": "server_error", "message": "..."}},
              {}, ProviderAPIError),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_openai_transport_http_error_mapping(case, openai_transport_factory):
    client = make_sync_client(error_response(case.status_code, case.body, case.headers))
    transport = openai_transport_factory(http_client=client)
    with pytest.raises(case.expected_exc) as exc_info:
        transport.completion(REQUEST)

    assert exc_info.value.provider == "openai"
    assert exc_info.value.original_exception is not None
    if case.extra_assertion is not None:
        assert case.extra_assertion(exc_info.value)


def test_openai_transport_timeout_maps_to_timeout_error(openai_transport_factory):
    def handler(request):
        raise httpx.TimeoutException("timeout", request=request)

    transport = openai_transport_factory(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ClientTimeoutError) as exc_info:
        transport.completion(REQUEST)
    assert exc_info.value.provider == "openai"


def test_openai_transport_connection_error_maps(openai_transport_factory):
    def handler(request):
        raise httpx.ConnectError("conn refused", request=request)

    transport = openai_transport_factory(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ClientConnectionError) as exc_info:
        transport.completion(REQUEST)
    assert exc_info.value.provider == "openai"
