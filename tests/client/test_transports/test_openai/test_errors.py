"""HTTP error → typed ClientError mapping.

The same table is driven through all three entry points — `completion`,
`completion_stream` and `acompletion_stream` — because the mapping used to
exist on the non-streaming path only. On a stream the rejection surfaces from
`_open()`/`_aopen()`, which run before the `try` that maps mid-stream
failures, so a 402 reached the caller as a bare `httpx.HTTPStatusError`
pointing at MDN and the provider's own message ("This request requires more
credits…") was discarded. Whether a call streams is a rendering choice; the
same rejection must produce the same typed error either way.
"""

from dataclasses import dataclass

import httpx
import pytest

from luca.client.exceptions import (
    AuthenticationError,
    BadRequestError,
    ConnectionError as ClientConnectionError,
    ContextLengthExceededError,
    ModelNotFoundError,
    ProviderAPIError,
    RateLimitError,
    TimeoutError as ClientTimeoutError,
)
from luca.client.types import ChatCompletionRequest, UserMessage
from tests.client._helpers.httpx_mocks import error_response, make_async_client, make_sync_client

REQUEST = ChatCompletionRequest(
    model="gpt-4o",
    provider="openai",
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
    ErrorCase(
        "401_auth", 401, {"error": {"type": "invalid_request_error", "message": "Invalid key"}}, {}, AuthenticationError
    ),
    ErrorCase(
        "429_with_retry_after",
        429,
        {"error": {"type": "rate_limit_exceeded", "message": "Too many"}},
        {"retry-after": "30"},
        RateLimitError,
        extra_assertion=lambda exc: exc.retry_after == 30.0,
    ),
    ErrorCase(
        "400_context_length",
        400,
        {"error": {"type": "context_length_exceeded", "message": "..."}},
        {},
        ContextLengthExceededError,
    ),
    ErrorCase("400_generic", 400, {"error": {"type": "invalid_request_error", "message": "..."}}, {}, BadRequestError),
    ErrorCase("404_model_not_found", 404, {"error": {"type": "not_found", "message": "..."}}, {}, ModelNotFoundError),
    ErrorCase("500_server_error", 500, {"error": {"type": "server_error", "message": "..."}}, {}, ProviderAPIError),
    # OpenRouter's real 402 body, captured 2026-08-08. No `type` key and a
    # numeric `code`, so it lands on the catch-all — which is enough, because
    # what the user needs is the message, and the message says what to do.
    ErrorCase(
        "402_out_of_credits",
        402,
        {
            "error": {
                "message": (
                    "This request requires more credits, or fewer max_tokens. You requested up to "
                    "65536 tokens, but can only afford 44818. To increase, visit "
                    "https://openrouter.ai/settings/credits and add more credits"
                ),
                "code": 402,
                "metadata": {
                    "limit_source": "openrouter_credits",
                    "remedy_hint": "Add credits at https://openrouter.ai/settings/credits, "
                    "or lower max_tokens / prompt size to fit your remaining balance.",
                },
            }
        },
        {},
        ProviderAPIError,
        extra_assertion=lambda exc: "requires more credits" in str(exc),
    ),
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


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_openai_transport_stream_http_error_mapping(case, openai_transport_factory):
    """A rejected STREAM raises the same typed error, at the open."""
    client = make_sync_client(error_response(case.status_code, case.body, case.headers))
    transport = openai_transport_factory(http_client=client)
    with pytest.raises(case.expected_exc) as exc_info, transport.completion_stream(REQUEST) as stream:
        list(stream)

    assert exc_info.value.provider == "openai"
    assert exc_info.value.original_exception is not None
    if case.extra_assertion is not None:
        assert case.extra_assertion(exc_info.value)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
async def test_openai_transport_astream_http_error_mapping(case, openai_transport_factory):
    client = make_async_client(error_response(case.status_code, case.body, case.headers))
    transport = openai_transport_factory(async_http_client=client)
    try:
        with pytest.raises(case.expected_exc) as exc_info:  # noqa: PT012
            async with transport.acompletion_stream(REQUEST) as stream:
                async for _event in stream:
                    pass
    finally:
        await transport.aclose()

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
