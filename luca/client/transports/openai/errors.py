"""OpenAI's HTTP error envelope, mapped to the SDK exception hierarchy.

Both OpenAI wire protocols — Chat Completions and Responses — return the same
`{"error": {"type", "message"}}` body and the same status codes, so the mapping
is shared. It lives in a mixin rather than a base class because the two
transports have nothing else in common: every projection and parse hook
differs, and inheriting one from the other would mean inheriting wrong methods.
"""

from __future__ import annotations

import httpx

from ...exceptions import (
    AuthenticationError,
    BadRequestError,
    ClientError,
    ConnectionError as ClientConnectionError,
    ContextLengthExceededError,
    InvalidModelError,
    ModelNotFoundError,
    ProviderAPIError,
    RateLimitError,
    TimeoutError as ClientTimeoutError,
    UnsupportedParameterError,
)


class OpenAIErrorMappingMixin:
    """Maps httpx failures to ClientError subclasses. Reads `self._provider`,
    which BaseTransport sets."""

    def _map_chat_completion_http_error(self, exc: httpx.HTTPError) -> ClientError:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            body = self._safe_json(exc.response)
            err_obj = (body or {}).get("error") if isinstance(body, dict) else {}
            err_type = (err_obj or {}).get("type", "") if isinstance(err_obj, dict) else ""
            msg = (err_obj or {}).get("message", str(exc)) if isinstance(err_obj, dict) else str(exc)

            if status == 401:
                return AuthenticationError(
                    msg,
                    provider=self._provider,  # type: ignore[attr-defined]
                    original_exception=exc,
                )
            if status == 429:
                return RateLimitError(
                    msg,
                    provider=self._provider,  # type: ignore[attr-defined]
                    original_exception=exc,
                    retry_after=self._retry_after(exc.response),
                )
            if status == 400:
                if "context_length" in err_type or "context_length" in msg.lower():
                    return ContextLengthExceededError(
                        msg,
                        provider=self._provider,  # type: ignore[attr-defined]
                        original_exception=exc,
                    )
                if "model" in err_type.lower():
                    return InvalidModelError(
                        msg,
                        provider=self._provider,  # type: ignore[attr-defined]
                        original_exception=exc,
                    )
                if "unsupported" in err_type.lower():
                    return UnsupportedParameterError(
                        msg,
                        provider=self._provider,  # type: ignore[attr-defined]
                        original_exception=exc,
                    )
                return BadRequestError(
                    msg,
                    provider=self._provider,  # type: ignore[attr-defined]
                    original_exception=exc,
                )
            if status == 404:
                return ModelNotFoundError(
                    msg,
                    provider=self._provider,  # type: ignore[attr-defined]
                    original_exception=exc,
                )
            if 500 <= status < 600:
                return ProviderAPIError(
                    msg,
                    provider=self._provider,  # type: ignore[attr-defined]
                    original_exception=exc,
                )
            return ProviderAPIError(
                msg,
                provider=self._provider,  # type: ignore[attr-defined]
                original_exception=exc,
            )

        if isinstance(exc, httpx.TimeoutException):
            return ClientTimeoutError(
                str(exc),
                provider=self._provider,  # type: ignore[attr-defined]
                original_exception=exc,
            )
        if isinstance(exc, httpx.NetworkError):
            return ClientConnectionError(
                str(exc),
                provider=self._provider,  # type: ignore[attr-defined]
                original_exception=exc,
            )
        return ProviderAPIError(
            str(exc),
            provider=self._provider,  # type: ignore[attr-defined]
            original_exception=exc,
        )

    @staticmethod
    def _safe_json(response: httpx.Response) -> dict | None:
        try:
            return response.json()
        except Exception:
            return None

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        val = response.headers.get("retry-after")
        if val is None:
            return None
        try:
            return float(val)
        except ValueError:
            return None
