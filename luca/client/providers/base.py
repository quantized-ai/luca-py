"""BaseProvider + ChatCompletionMixin.

Concrete classes — no ABC, no Protocol. Provider methods are forwarding wrappers
around the owned transport's methods.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import httpx

from ..exceptions import ConfigurationError

if TYPE_CHECKING:
    from ..transports.base import BaseTransport
    from ..types.completion import ChatCompletionRequest, ChatCompletionResponse
    from ..types.streaming import AsyncChatCompletionStream, ChatCompletionStream


class BaseProvider:
    """Vendor facade. Holds host defaults and owns the transport instance."""

    name: str = ""
    default_base_url: str = ""
    default_api_key_env_var: str | None = None
    default_transport_class: type | None = None

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        transport_class: type | None = None,
        transport: "BaseTransport | None" = None,
        timeout: float | None = 60.0,
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if transport is not None:
            self._transport = transport
            return

        resolved_api_key = api_key
        if resolved_api_key is None and self.default_api_key_env_var:
            resolved_api_key = os.environ.get(self.default_api_key_env_var)

        resolved_base_url = base_url or self.default_base_url
        if not resolved_base_url:
            raise ConfigurationError(
                f"Provider {self.name!r} has no default_base_url and none was passed.",
                provider=self.name,
            )

        cls = transport_class or self.default_transport_class
        if cls is None:
            raise ConfigurationError(
                f"Provider {self.name!r} has no default_transport_class.",
                provider=self.name,
            )

        self._transport = cls(
            provider=self.name,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            timeout=timeout,
            http_client=http_client,
            async_http_client=async_http_client,
        )

    @property
    def transport(self) -> "BaseTransport":
        return self._transport

    def close(self) -> None:
        self._transport.close()

    async def aclose(self) -> None:
        await self._transport.aclose()

    def __enter__(self) -> "BaseProvider":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    async def __aenter__(self) -> "BaseProvider":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()


class ChatCompletionMixin:
    """Provider-level chat completion. Forwards every method to the transport."""

    def completion(
        self, request: "ChatCompletionRequest",
    ) -> "ChatCompletionResponse":
        return self._transport.completion(request)  # type: ignore[attr-defined]

    async def acompletion(
        self, request: "ChatCompletionRequest",
    ) -> "ChatCompletionResponse":
        return await self._transport.acompletion(request)  # type: ignore[attr-defined]

    def completion_stream(
        self, request: "ChatCompletionRequest",
    ) -> "ChatCompletionStream":
        return self._transport.completion_stream(request)  # type: ignore[attr-defined]

    def acompletion_stream(
        self, request: "ChatCompletionRequest",
    ) -> "AsyncChatCompletionStream":
        return self._transport.acompletion_stream(request)  # type: ignore[attr-defined]
