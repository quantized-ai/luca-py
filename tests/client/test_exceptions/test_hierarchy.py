import pytest

from luca.client.exceptions import (
    AuthenticationError,
    BadRequestError,
    ClientError,
    ConfigurationError,
    ConnectionError as ClientConnectionError,
    ContextLengthExceededError,
    InvalidModelError,
    ModelNotFoundError,
    ProviderAPIError,
    ProviderNotFoundError,
    RateLimitError,
    StreamError,
    StructuredOutputError,
    TimeoutError as ClientTimeoutError,
    UnsupportedParameterError,
)


@pytest.mark.parametrize(
    ("subclass", "parent"),
    [
        (AuthenticationError, ConfigurationError),
        (ConfigurationError, ClientError),
        (BadRequestError, ClientError),
        (ContextLengthExceededError, BadRequestError),
        (InvalidModelError, BadRequestError),
        (UnsupportedParameterError, BadRequestError),
        (ProviderNotFoundError, ClientError),
        (ModelNotFoundError, ClientError),
        (RateLimitError, ClientError),
        (ProviderAPIError, ClientError),
        (ClientConnectionError, ClientError),
        (ClientTimeoutError, ClientError),
        (StructuredOutputError, ClientError),
        (StreamError, ClientError),
    ],
)
def test_subclass_relationship(subclass, parent):
    assert issubclass(subclass, parent)


def test_rate_limit_error_carries_retry_after():
    e = RateLimitError("limit", provider="openai", retry_after=30.0)
    assert e.provider == "openai"
    assert e.retry_after == 30.0


def test_client_error_carries_original_exception():
    base = ValueError("bad")
    e = ClientError("wrap", provider="openai", original_exception=base)
    assert e.original_exception is base
    assert e.provider == "openai"
