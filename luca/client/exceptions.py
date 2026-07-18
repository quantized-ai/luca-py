"""Typed exception hierarchy for the SDK.

Every exception inherits from ClientError. Each carries `provider` (the host
name) and `original_exception` (the underlying cause when applicable).

LLM-side moderation outcomes (refusal / content filter / safety) are NOT
exceptions — they arrive as a normal ChatCompletionResponse / FinishEvent with
`finish_reason="error"`. See api_prd.md §9 and architecture.md §11.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types.messages import AssistantMessage


class ClientError(Exception):
    """Base exception. Every SDK exception inherits from this."""

    def __init__(
        self,
        message: str = "",
        *,
        provider: str | None = None,
        original_exception: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.original_exception = original_exception


class ConfigurationError(ClientError):
    """Auth setup wrong, missing env vars, misconfigured provider."""


class AuthenticationError(ConfigurationError):
    """401 / invalid API key."""


class BadRequestError(ClientError):
    """400 / malformed request."""


class ContextLengthExceededError(BadRequestError):
    """Request too long for the model's context window."""


class InvalidModelError(BadRequestError):
    """Provider does not recognize the model id."""


class UnsupportedParameterError(BadRequestError):
    """A passed param is not supported on this (model, provider) pair."""


class ProviderNotFoundError(ClientError):
    """Unknown provider name (no PROVIDERS entry, no overrides)."""


class ModelNotFoundError(ClientError):
    """404 from upstream / model retired or unknown."""


class RateLimitError(ClientError):
    """429. Carries `retry_after` (seconds) when the upstream provides one."""

    def __init__(
        self,
        message: str = "",
        *,
        provider: str | None = None,
        original_exception: BaseException | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, provider=provider, original_exception=original_exception)
        self.retry_after = retry_after


class ProviderAPIError(ClientError):
    """5xx / generic upstream failure."""


class ConnectionError(ClientError):
    """Network error reaching the provider."""


class TimeoutError(ClientError):
    """Request timed out."""


class StructuredOutputError(ClientError):
    """response_format was set but the content didn't validate."""


class StreamError(ClientError):
    """Mid-stream failure. `partial_message` carries everything received so far."""

    def __init__(
        self,
        message: str = "",
        *,
        provider: str | None = None,
        original_exception: BaseException | None = None,
        partial_message: "AssistantMessage | None" = None,
    ) -> None:
        super().__init__(message, provider=provider, original_exception=original_exception)
        self.partial_message = partial_message


__all__ = [
    "ClientError",
    "ConfigurationError",
    "AuthenticationError",
    "BadRequestError",
    "ContextLengthExceededError",
    "InvalidModelError",
    "UnsupportedParameterError",
    "ProviderNotFoundError",
    "ModelNotFoundError",
    "RateLimitError",
    "ProviderAPIError",
    "ConnectionError",
    "TimeoutError",
    "StructuredOutputError",
    "StreamError",
]
