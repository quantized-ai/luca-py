"""BaseTransport + ChatCompletionTransportMixin.

BaseTransport owns the httpx lifecycle, headers, and provenance bookkeeping.
ChatCompletionTransportMixin owns the chat-completion HTTP orchestration via
hook methods that concrete subclasses override.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from ..exceptions import ClientError

if TYPE_CHECKING:
    from ..types.completion import ChatCompletionRequest, ChatCompletionResponse
    from ..types.streaming import AsyncChatCompletionStream, ChatCompletionStream


class BaseTransport:
    """httpx client lifecycle, headers, and provider provenance.

    Every concrete transport subclasses this plus one or more capability mixins.
    `transport_id` identifies the wire protocol (e.g. "openai", "anthropic");
    `_provider` (on the instance) is the host name stamped onto responses for
    provenance — set by the constructor, not by the class."""

    transport_id: str = ""

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float | None = 60.0,
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._provider = provider
        self._base_url = base_url.rstrip("/") if base_url else ""
        self._api_key = api_key
        self._timeout = timeout

        self._owned_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout)
        self._owned_aclient = async_http_client is None
        self._aclient: httpx.AsyncClient | None = async_http_client

    # --- auth + headers (override for non-Bearer schemes) ---

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _ensure_aclient(self) -> httpx.AsyncClient:
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(timeout=self._timeout)
            self._owned_aclient = True
        return self._aclient

    # --- lifecycle ---

    def close(self) -> None:
        if self._owned_client:
            try:
                self._client.close()
            except Exception:
                pass

    async def aclose(self) -> None:
        if self._aclient is not None and self._owned_aclient:
            try:
                await self._aclient.aclose()
            except Exception:
                pass

    def __enter__(self) -> "BaseTransport":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    async def __aenter__(self) -> "BaseTransport":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()


class ChatCompletionTransportMixin:
    """Chat completion HTTP orchestration. Subclasses override the hook methods
    listed below; orchestration is shared.

    Required hooks:
      _build_chat_completion_payload(request, *, stream=False) -> dict
      _parse_chat_completion_response(httpx_response, request) -> ChatCompletionResponse
      _classify_finish(provider_value, message) -> tuple[str | None, str | None]
      _map_chat_completion_http_error(exc) -> ClientError
      _chat_completion_stream_class() -> type
      _async_chat_completion_stream_class() -> type

    Optional overrides:
      _chat_completion_url() -> str
      _build_chat_completion_httpx_request(request) -> httpx.Request
    """

    # type-ignored: methods reference self._client etc., set by BaseTransport.

    def completion(
        self, request: "ChatCompletionRequest",
    ) -> "ChatCompletionResponse":
        httpx_request = self._build_chat_completion_httpx_request(request)
        try:
            response = self._client.send(httpx_request)  # type: ignore[attr-defined]
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise self._map_chat_completion_http_error(e)
        return self._parse_chat_completion_response(response, request)

    async def acompletion(
        self, request: "ChatCompletionRequest",
    ) -> "ChatCompletionResponse":
        aclient = self._ensure_aclient()  # type: ignore[attr-defined]
        # build_request requires a client — use the async one for the URL host.
        httpx_request = aclient.build_request(
            "POST",
            self._chat_completion_url(),
            json=self._build_chat_completion_payload(request),
            headers=self._headers(),  # type: ignore[attr-defined]
        )
        try:
            response = await aclient.send(httpx_request)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise self._map_chat_completion_http_error(e)
        return self._parse_chat_completion_response(response, request)

    def completion_stream(
        self, request: "ChatCompletionRequest",
    ) -> "ChatCompletionStream":
        cls = self._chat_completion_stream_class()
        return cls(transport=self, request=request)

    def acompletion_stream(
        self, request: "ChatCompletionRequest",
    ) -> "AsyncChatCompletionStream":
        cls = self._async_chat_completion_stream_class()
        return cls(transport=self, request=request)

    # --- default hook implementations ---

    def _chat_completion_url(self) -> str:
        return f"{self._base_url}/chat/completions"  # type: ignore[attr-defined]

    def _build_chat_completion_httpx_request(
        self, request: "ChatCompletionRequest",
    ) -> httpx.Request:
        return self._client.build_request(  # type: ignore[attr-defined]
            "POST",
            self._chat_completion_url(),
            json=self._build_chat_completion_payload(request),
            headers=self._headers(),  # type: ignore[attr-defined]
        )

    # --- abstract hooks ---

    def _build_chat_completion_payload(
        self, request: "ChatCompletionRequest", *, stream: bool = False,
    ) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_chat_completion_payload"
        )

    def _parse_chat_completion_response(
        self, response: httpx.Response, request: "ChatCompletionRequest",
    ) -> "ChatCompletionResponse":
        raise NotImplementedError(
            f"{type(self).__name__} must implement _parse_chat_completion_response"
        )

    def _classify_finish(
        self, provider_value: str | None, message: Any,
    ) -> tuple[str | None, str | None]:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _classify_finish"
        )

    def _map_chat_completion_http_error(self, exc: httpx.HTTPError) -> ClientError:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _map_chat_completion_http_error"
        )

    def _chat_completion_stream_class(self) -> type:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _chat_completion_stream_class"
        )

    def _async_chat_completion_stream_class(self) -> type:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _async_chat_completion_stream_class"
        )
