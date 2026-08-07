"""BaseTransport + ChatCompletionTransportMixin.

BaseTransport owns the httpx lifecycle, headers, and provenance bookkeeping.
ChatCompletionTransportMixin owns the chat-completion HTTP orchestration via
hook methods that concrete subclasses override.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from ..exceptions import BadRequestError, ClientError
from ..types.content import NATIVE_TOOL_CALL_TYPES, ToolCall
from ..types.tools import BaseTool, ToolProjector

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

    # --- replaying provider-owned attestations ---

    def _attestation_is_replayable(
        self,
        message: Any,
        request: ChatCompletionRequest,
    ) -> bool:
        """Whether an assistant message's opaque provider data (a thinking
        signature, an encrypted reasoning item) may go back on THIS request.

        A signature is minted by one (provider, model) pair and is meaningless
        to any other — Anthropic 400s on a foreign one, and a foreign
        `rs_…` id is not a reasoning item OpenAI knows. So an attestation
        replays only when the message that carries it was produced by the pair
        we are about to call. Reachable in one keystroke: the agent TUI's
        `/model` rewrites the session's model mid-conversation and the next
        request replays the whole history.

        Provenance ABSENT (`provider`/`model` are None, as on a hand-built
        message) means the caller is driving and is trusted — refusing there
        would silently strip data the caller deliberately supplied."""
        provider = getattr(message, "provider", None)
        model = getattr(message, "model", None)
        if provider is None or model is None:
            return True
        return provider == self._provider and model == request.model

    def _ensure_aclient(self) -> httpx.AsyncClient:
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(timeout=self._timeout)
            self._owned_aclient = True
        return self._aclient

    # --- lifecycle ---

    def close(self) -> None:
        if self._owned_client:
            with contextlib.suppress(Exception):
                self._client.close()

    async def aclose(self) -> None:
        if self._aclient is not None and self._owned_aclient:
            with contextlib.suppress(Exception):
                await self._aclient.aclose()

    def __enter__(self) -> BaseTransport:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    async def __aenter__(self) -> BaseTransport:
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
      _chat_completion_url(request, *, stream=False) -> str
      _build_chat_completion_httpx_request(request, client) -> httpx.Request
    """

    # type-ignored: methods reference self._client etc., set by BaseTransport.

    def completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        httpx_request = self._build_chat_completion_httpx_request(
            request,
            self._client,  # type: ignore[attr-defined]
        )
        try:
            response = self._client.send(httpx_request)  # type: ignore[attr-defined]
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise self._map_chat_completion_http_error(e) from e
        return self._parse_chat_completion_response(response, request)

    async def acompletion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        aclient = self._ensure_aclient()  # type: ignore[attr-defined]
        # Same builder as the sync path so transports that customise request
        # construction are not bypassed on async.
        httpx_request = self._build_chat_completion_httpx_request(request, aclient)
        try:
            response = await aclient.send(httpx_request)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise self._map_chat_completion_http_error(e) from e
        return self._parse_chat_completion_response(response, request)

    def completion_stream(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionStream:
        cls = self._chat_completion_stream_class()
        return cls(transport=self, request=request)

    def acompletion_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncChatCompletionStream:
        cls = self._async_chat_completion_stream_class()
        return cls(transport=self, request=request)

    # --- default hook implementations ---

    def _chat_completion_url(
        self,
        request: ChatCompletionRequest,
        *,
        stream: bool = False,
    ) -> str:
        return f"{self._base_url}/chat/completions"  # type: ignore[attr-defined]

    def _build_chat_completion_httpx_request(
        self,
        request: ChatCompletionRequest,
        client: Any,
    ) -> httpx.Request:
        return client.build_request(
            "POST",
            self._chat_completion_url(request),
            json=self._build_chat_completion_payload(request),
            headers=self._headers(),  # type: ignore[attr-defined]
        )

    # --- abstract hooks ---

    def _build_chat_completion_payload(
        self,
        request: ChatCompletionRequest,
        *,
        stream: bool = False,
    ) -> dict:
        raise NotImplementedError(f"{type(self).__name__} must implement _build_chat_completion_payload")

    def _parse_chat_completion_response(
        self,
        response: httpx.Response,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        raise NotImplementedError(f"{type(self).__name__} must implement _parse_chat_completion_response")

    def _classify_finish(
        self,
        provider_value: str | None,
        message: Any,
    ) -> tuple[str | None, str | None]:
        raise NotImplementedError(f"{type(self).__name__} must implement _classify_finish")

    def _map_chat_completion_http_error(self, exc: httpx.HTTPError) -> ClientError:
        raise NotImplementedError(f"{type(self).__name__} must implement _map_chat_completion_http_error")

    def _chat_completion_stream_class(self) -> type:
        raise NotImplementedError(f"{type(self).__name__} must implement _chat_completion_stream_class")

    def _async_chat_completion_stream_class(self) -> type:
        raise NotImplementedError(f"{type(self).__name__} must implement _async_chat_completion_stream_class")

    # --- tool projection (shared mechanics; shapes live in projectors) ---

    # Each transport family binds this to its projector base class. The check
    # is class-based, not transport_id-based, so it composes with transport
    # subclassing: OpenRouter inherits OpenAITransport's base and accepts the
    # same projectors.
    TOOL_PROJECTOR_BASE: ClassVar[type] = ToolProjector

    def _default_tool_projector(self) -> ToolProjector:
        raise NotImplementedError(f"{type(self).__name__} must implement _default_tool_projector")

    def _resolve_projector(self, tool: BaseTool) -> ToolProjector:
        """The projector for a DECLARED tool, checked against this transport
        — an incompatible native tool fails before any HTTP."""
        projector = tool.get_projector() or self._default_tool_projector()
        if not isinstance(projector, self.TOOL_PROJECTOR_BASE):
            raise BadRequestError(
                f"{type(tool).__name__}'s projector targets another transport; "
                f"this request uses {type(self).__name__}.",
                provider=self._provider,  # type: ignore[attr-defined]
            )
        return projector

    def _project_tools(self, tools: list) -> list[dict]:
        return [self._resolve_projector(t).project_tool_to_llm(t) for t in tools]

    def _resolve_call(self, tool_call: ToolCall) -> tuple[ToolProjector, ToolCall] | None:
        """The projector that replays a ToolCall from history and the call to
        hand it — the lineage entry every transport records.

        The call is normalized through `as_native()` first, so a native
        expressed generically (`extras={"custom_type": …}`) reaches the same
        projector, with the same fields, as the typed subclass: the two forms
        are equivalent by construction rather than by parallel code paths.

        None when the call was minted by another transport family (e.g.
        `/model` switched providers mid-session). It means the call is dropped
        together with its result — the foreign-attestation policy: one lost
        exchange beats a 400 that kills the conversation."""
        call = tool_call.as_native()
        projector_class = type(call).projector_class
        if projector_class is None:
            return (self._default_tool_projector(), call)
        projector = projector_class()
        if isinstance(projector, self.TOOL_PROJECTOR_BASE):
            return (projector, call)
        return None

    def _native_projector_for_item(self, item_type: str | None) -> ToolProjector | None:
        """Parse-side registry lookup, compatibility-checked: a foreign
        family's registration can never mint calls on this wire. A miss (or
        an incompatible hit) means the item is ignored, exactly like any
        unknown hosted-tool item."""
        if item_type is None:
            return None
        cls = NATIVE_TOOL_CALL_TYPES.get(item_type)
        if cls is None or cls.projector_class is None:
            return None
        projector = cls.projector_class()
        if isinstance(projector, self.TOOL_PROJECTOR_BASE):
            return projector
        return None

    def _declared_tool_named(self, name: Any, tools: list | None) -> BaseTool | None:
        """First declared tool whose name matches — tool_choice resolution.
        Name collisions between a custom Tool and a native tool are
        documented, not policed; first match wins."""
        for t in tools or []:
            if getattr(t, "name", None) == name:
                return t
        return None
