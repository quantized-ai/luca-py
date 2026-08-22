"""WireFormatMixin + BaseTransport + ChatCompletionTransportMixin.

WireFormatMixin holds the wire knowledge shared by BOTH request paths — the
transport's non-streaming calls and the streamer's streaming calls: headers,
URL, attestation policy, tool projection, and the per-wire hook stubs.
BaseTransport owns the httpx lifecycle and provenance bookkeeping.
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
    from .streamer import BaseStreamer


class WireFormatMixin:
    """Per-wire knowledge, stateless over `self._provider` / `self._base_url` /
    `self._api_key` — never clients or timeouts, so the same mixin serves the
    transport and the streamer. Each wire package subclasses this into its
    `<X>WireMixin` and overrides the hooks below."""

    # Each transport family binds this to its projector base class. The check
    # is class-based, not transport_id-based, so it composes with transport
    # subclassing: OpenRouter inherits OpenAITransport's base and accepts the
    # same projectors.
    TOOL_PROJECTOR_BASE: ClassVar[type] = ToolProjector
    # Reasoning display policy for wires that resolve it (Anthropic, Bedrock).
    THINKING_DISPLAY: ClassVar[str | None] = None
    # The wire format this transport mints PrivateProviderBlocks under, and
    # therefore the only `PrivateProviderBlock.format` it replays. None means
    # the wire mints no privates — every private block is foreign and drops.
    WIRE_FORMAT: ClassVar[str | None] = None
    # Whether this wire has a shape for an `AudioBlock` on a user message.
    # Declared because a MODEL's audio capability does not imply it: the
    # catalog knows `gpt-realtime-2.1` hears, and cannot know that OpenAI
    # routes to the Responses wire, which has no audio part. Text, images and
    # documents reach every transport, so audio is the only block worth a flag.
    # `tests/client/test_transports/test_audio_declaration.py` pins this
    # against what the transports actually do.
    SUPPORTS_AUDIO_INPUT: ClassVar[bool] = False

    # --- auth + headers (override for non-Bearer schemes) ---

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:  # type: ignore[attr-defined]
            h["Authorization"] = f"Bearer {self._api_key}"  # type: ignore[attr-defined]
        return h

    def _chat_completion_url(
        self,
        request: ChatCompletionRequest,
        *,
        stream: bool = False,
    ) -> str:
        return f"{self._base_url}/chat/completions"  # type: ignore[attr-defined]

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
        return provider == self._provider and model == request.model  # type: ignore[attr-defined]

    # --- wire hooks (implemented by each <X>WireMixin) ---

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

    # --- effective-message selection ---

    def select_assistant_blocks(self, message: Any, request: ChatCompletionRequest) -> list:
        """The assistant blocks that actually serialize onto this wire for
        `request` — THE single copy of the per-block keep/drop rules, shared
        by payload build and `effective_messages`. Payload build serializes
        exactly these blocks; the query returns exactly these blocks.

        Takes the whole message (Anthropic's merged-cited rule is positional
        over the content list) and the request (thinking replay compares the
        attestation's provenance against `request.model`)."""
        raise NotImplementedError(f"{type(self).__name__} must implement select_assistant_blocks")

    def effective_messages(self, messages: list, request: ChatCompletionRequest) -> list:
        """The subset of `messages` that will actually serialize onto this
        wire for `request`, in client types — the same walk the payload build
        runs: per-message block selection through `select_assistant_blocks`,
        plus the ToolCall/ToolMessage lineage rule (a foreign-family native
        call drops, and the ToolMessage answering a dropped call falls with
        it, `_resolve_call`).

        A fully-filtered assistant message is returned as
        `AssistantMessage(content=[])`, never omitted — transports skip empty
        assistant messages at build time, so the payload invariant holds."""
        from ..types.messages import AssistantMessage, ToolMessage

        out: list = []
        # Call id → survives. Last write wins, exactly like the build walks'
        # lineage dicts, so a duplicate id across messages resolves the same
        # way in both paths.
        call_survives: dict[str, bool] = {}
        for msg in messages:
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, ToolCall):
                        call_survives[block.id] = self._resolve_call(block) is not None
                kept = self.select_assistant_blocks(msg, request)
                out.append(msg if kept == msg.content else msg.model_copy(update={"content": kept}))
            elif isinstance(msg, ToolMessage) and not call_survives.get(msg.tool_call_id, True):
                continue
            else:
                out.append(msg)
        return out

    def _map_chat_completion_http_error(self, exc: httpx.HTTPError) -> ClientError:
        raise NotImplementedError(f"{type(self).__name__} must implement _map_chat_completion_http_error")

    # --- tool projection (shared mechanics; shapes live in projectors) ---

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


class BaseTransport(WireFormatMixin):
    """httpx client lifecycle and provider provenance.

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
    """Chat completion HTTP orchestration. The wire hooks it calls
    (`_build_chat_completion_payload`, `_parse_chat_completion_response`,
    `_classify_finish`, `_map_chat_completion_http_error`) are declared on
    WireFormatMixin and implemented by each wire package's `<X>WireMixin`.

    For streaming the transport is a pure factory: `STREAMER` /
    `ASYNC_STREAMER` name the wire's sync/async streamer combo classes and
    the two methods forward data only. `timeout=` is a per-call total
    wall-clock deadline; on the non-streaming path it rides the httpx
    request as a per-phase override (best-effort).

    Optional overrides:
      _chat_completion_url(request, *, stream=False) -> str
      _build_chat_completion_httpx_request(request, client, timeout=None) -> httpx.Request
    """

    STREAMER: ClassVar[type | None] = None
    ASYNC_STREAMER: ClassVar[type | None] = None

    # type-ignored: methods reference self._client etc., set by BaseTransport.

    def completion(
        self,
        request: ChatCompletionRequest,
        *,
        timeout: float | None = None,
    ) -> ChatCompletionResponse:
        httpx_request = self._build_chat_completion_httpx_request(
            request,
            self._client,  # type: ignore[attr-defined]
            timeout=timeout,
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
        *,
        timeout: float | None = None,
    ) -> ChatCompletionResponse:
        aclient = self._ensure_aclient()  # type: ignore[attr-defined]
        # Same builder as the sync path so transports that customise request
        # construction are not bypassed on async.
        httpx_request = self._build_chat_completion_httpx_request(request, aclient, timeout=timeout)
        try:
            response = await aclient.send(httpx_request)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise self._map_chat_completion_http_error(e) from e
        return self._parse_chat_completion_response(response, request)

    def completion_stream(
        self,
        request: ChatCompletionRequest,
        *,
        timeout: float | None = None,
    ) -> BaseStreamer:
        return self.STREAMER(
            request,
            provider=self._provider,  # type: ignore[attr-defined]
            httpx_client=self._client,  # type: ignore[attr-defined]
            api_key=self._api_key,  # type: ignore[attr-defined]
            base_url=self._base_url,  # type: ignore[attr-defined]
            timeout=timeout,
        )

    def acompletion_stream(
        self,
        request: ChatCompletionRequest,
        *,
        timeout: float | None = None,
    ) -> BaseStreamer:
        return self.ASYNC_STREAMER(
            request,
            provider=self._provider,  # type: ignore[attr-defined]
            httpx_client=self._ensure_aclient(),  # type: ignore[attr-defined]
            api_key=self._api_key,  # type: ignore[attr-defined]
            base_url=self._base_url,  # type: ignore[attr-defined]
            timeout=timeout,
        )

    def _build_chat_completion_httpx_request(
        self,
        request: ChatCompletionRequest,
        client: Any,
        *,
        timeout: float | None = None,
    ) -> httpx.Request:
        return client.build_request(
            "POST",
            self._chat_completion_url(request),
            json=self._build_chat_completion_payload(request),
            headers=self._headers(),  # type: ignore[attr-defined]
            timeout=httpx.USE_CLIENT_DEFAULT if timeout is None else timeout,
        )
