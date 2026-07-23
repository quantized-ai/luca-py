"""HTTP error → typed ClientError, driven off the `x-amzn-errortype` header.

Bedrock puts the exception class in a header whose value has a `:<url>` suffix,
so the mapping matches the leading token, not the whole string.
"""

import httpx
import pytest

from luca.client.exceptions import (
    AuthenticationError,
    BadRequestError,
    ContextLengthExceededError,
    ModelNotFoundError,
    RateLimitError,
)
from luca.client.types import ChatCompletionRequest, UserMessage

_SUFFIX = ":http://internal.amazon.com/coral/com.amazon.bedrock/"


def _call_with_error(transport_factory, status, errortype, message):
    def handler(request):
        return httpx.Response(
            status,
            headers={"x-amzn-errortype": errortype + _SUFFIX},
            json={"message": message},
        )

    transport = transport_factory(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(Exception) as caught:
        transport.completion(ChatCompletionRequest(
            provider="bedrock", model="us.amazon.nova-lite-v1:0",
            messages=[UserMessage(content="Hi")],
        ))
    return caught.value


def test_a_throttling_exception_becomes_a_rate_limit_error(bedrock_transport_factory):
    err = _call_with_error(
        bedrock_transport_factory, 429, "ThrottlingException", "Slow down",
    )
    assert isinstance(err, RateLimitError)


def test_a_validation_exception_becomes_a_bad_request_error(bedrock_transport_factory):
    err = _call_with_error(
        bedrock_transport_factory, 400, "ValidationException", "bad shape",
    )
    assert isinstance(err, BadRequestError)


def test_an_over_long_validation_exception_becomes_a_context_length_error(
    bedrock_transport_factory,
):
    err = _call_with_error(
        bedrock_transport_factory, 400, "ValidationException",
        "Input is too long for the model",
    )
    assert isinstance(err, ContextLengthExceededError)


def test_an_access_denied_exception_becomes_an_authentication_error(bedrock_transport_factory):
    err = _call_with_error(
        bedrock_transport_factory, 403, "AccessDeniedException", "no access",
    )
    assert isinstance(err, AuthenticationError)


def test_the_gated_model_404_becomes_a_model_not_found_error(bedrock_transport_factory):
    # The exact response an ungated Anthropic model returns on this account.
    err = _call_with_error(
        bedrock_transport_factory, 404, "ResourceNotFoundException",
        "Model use case details have not been submitted for this account.",
    )
    assert isinstance(err, ModelNotFoundError)
    assert "use case" in str(err)
