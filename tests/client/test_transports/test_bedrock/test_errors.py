"""HTTP error → typed ClientError, driven off the `x-amzn-errortype` header.

Bedrock puts the exception class in a header whose value has a `:<url>` suffix,
so the mapping matches the leading token, not the whole string.
"""

from dataclasses import dataclass, field

import httpx
import pytest

from luca.client.exceptions import (
    AuthenticationError,
    BadRequestError,
    ContextLengthExceededError,
    ModelNotFoundError,
    RateLimitError,
)
from luca.client.exceptions import ConnectionError as ClientConnectionError
from luca.client.exceptions import TimeoutError as ClientTimeoutError
from luca.client.types import ChatCompletionRequest, UserMessage

REQUEST = ChatCompletionRequest(
    provider="bedrock", model="us.amazon.nova-lite-v1:0",
    messages=[UserMessage(content="Hi")],
)
# The header value Bedrock actually sends: exception class plus a coral URL.
_SUFFIX = ":http://internal.amazon.com/coral/com.amazon.bedrock/"


@dataclass(frozen=True)
class ErrorCase:
    name: str
    status_code: int
    errortype: str
    message: str
    expected_exc: type
    headers: dict = field(default_factory=dict)
    extra_assertion: object = None


CASES = [
    ErrorCase("throttling", 429, "ThrottlingException", "Slow down",
              RateLimitError, headers={"retry-after": "12"},
              extra_assertion=lambda e: e.retry_after == 12.0),
    ErrorCase("validation", 400, "ValidationException", "bad shape",
              BadRequestError),
    ErrorCase("context_length", 400, "ValidationException",
              "Input is too long for the model", ContextLengthExceededError),
    ErrorCase("access_denied", 403, "AccessDeniedException", "no access",
              AuthenticationError),
    # The exact response an ungated Anthropic model returns on this account.
    ErrorCase("gated_model_404", 404, "ResourceNotFoundException",
              "Model use case details have not been submitted for this account.",
              ModelNotFoundError,
              extra_assertion=lambda e: "use case" in str(e)),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_bedrock_transport_http_error_mapping(case, bedrock_transport_factory):
    def handler(request):
        return httpx.Response(
            case.status_code,
            headers={"x-amzn-errortype": case.errortype + _SUFFIX, **case.headers},
            json={"message": case.message},
        )

    transport = bedrock_transport_factory(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(case.expected_exc) as exc_info:
        transport.completion(REQUEST)

    assert exc_info.value.provider == "bedrock"
    assert exc_info.value.original_exception is not None
    if case.extra_assertion is not None:
        assert case.extra_assertion(exc_info.value)


def test_bedrock_transport_timeout_maps_to_timeout_error(bedrock_transport_factory):
    def handler(request):
        raise httpx.TimeoutException("timeout", request=request)

    transport = bedrock_transport_factory(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ClientTimeoutError) as exc_info:
        transport.completion(REQUEST)
    assert exc_info.value.provider == "bedrock"


def test_bedrock_transport_connection_error_maps(bedrock_transport_factory):
    def handler(request):
        raise httpx.ConnectError("conn refused", request=request)

    transport = bedrock_transport_factory(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ClientConnectionError) as exc_info:
        transport.completion(REQUEST)
    assert exc_info.value.provider == "bedrock"
