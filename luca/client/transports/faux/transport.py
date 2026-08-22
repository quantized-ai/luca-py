"""Faux (scripted) transport — for tests. Does not use httpx.

Builders return small dataclasses representing intended scripted output;
FauxTransport plays them back as ChatCompletionResponse objects or through
the faux streamers (`streamer.py`).
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any

from ...exceptions import ClientError
from ...types.completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
)
from ...types.content import (
    PrivateProviderBlock,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    URLCitationAnnotation,
    WebFetchBlock,
    WebSearchBlock,
)
from ...types.messages import AssistantMessage
from ..base import BaseTransport, ChatCompletionTransportMixin, WireFormatMixin

# ---------------------------------------------------------------------------
# Scripted-response builders
# ---------------------------------------------------------------------------


@dataclass
class _FauxText:
    text: str
    annotations: list[URLCitationAnnotation] | None = None


@dataclass
class _FauxThinking:
    text: str
    signature: str | None = None
    redacted: bool = False


@dataclass
class _FauxToolCall:
    name: str
    arguments: dict
    id: str = "tool_call_faux"


@dataclass
class _FauxRefusal:
    text: str


@dataclass
class _FauxHang:
    """Marker block: playback hangs forever (cancellable) at this point —
    for cancellation / timeout tests. Async-only: the sync surfaces raise on
    it (a sync hang would just freeze the test)."""


@dataclass
class _FauxError:
    message: str
    error_class: type[ClientError] | None = None


@dataclass
class _FauxAssistantMessage:
    blocks: list
    finish_reason: str = "stop"
    error: _FauxError | None = None
    usage: Usage | None = None


def faux_text(text: str, annotations: list[URLCitationAnnotation] | None = None) -> _FauxText:
    return _FauxText(text=text, annotations=annotations)


def faux_thinking(
    text: str,
    signature: str | None = None,
    redacted: bool = False,
) -> _FauxThinking:
    return _FauxThinking(text=text, signature=signature, redacted=redacted)


# `id` mirrors ToolCall.id and is passed as id= at every call site.
def faux_tool_call(name: str, arguments: dict, id: str = "tool_call_faux") -> _FauxToolCall:  # noqa: A002
    return _FauxToolCall(name=name, arguments=arguments, id=id)


def faux_refusal(text: str) -> _FauxRefusal:
    return _FauxRefusal(text=text)


def faux_hang() -> _FauxHang:
    return _FauxHang()


def faux_error(message: str, error_class: type[ClientError] | None = None) -> _FauxError:
    return _FauxError(message=message, error_class=error_class)


def faux_assistant_message(
    blocks: list,
    finish_reason: str = "stop",
    error: _FauxError | None = None,
    usage: Usage | None = None,
) -> _FauxAssistantMessage:
    return _FauxAssistantMessage(
        blocks=blocks,
        finish_reason=finish_reason,
        error=error,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Wire mixin — shared by FauxTransport and FauxStreamer
# ---------------------------------------------------------------------------


class FauxWireMixin(WireFormatMixin):
    """The faux's wire knowledge: finish classification. The faux speaks
    SDK-canonical values directly — callers pass `finish_reason="stop"`,
    `"tool_use"`, `"error"`, `"length"` etc."""

    def select_assistant_blocks(self, message: AssistantMessage, request: ChatCompletionRequest) -> list:
        """Identity: the faux has no payload build, so nothing is ever
        filtered — the payload invariant is undefined here by construction."""
        return list(message.content)

    def effective_messages(self, messages: list, request: ChatCompletionRequest) -> list:
        """Identity, whole-walk: no build path means no lineage rule either —
        every message survives verbatim."""
        return list(messages)

    def _classify_finish(
        self,
        provider_value: str | None,
        message: AssistantMessage,
    ) -> tuple[str | None, str | None]:
        if provider_value == "error":
            # If any refusal block present, derive message from it.
            for b in message.content:
                if isinstance(b, RefusalBlock):
                    return ("error", f"Faux refusal: {b.text}")
            return ("error", "Faux error terminal")
        return (provider_value, None)

    def _map_chat_completion_http_error(self, exc: Any) -> ClientError:  # pragma: no cover
        return ClientError(str(exc), provider=self._provider, original_exception=exc)


# Imported here, between the wire mixin and the transport class, because the
# streamer module inherits the mixin above: at the top of the file this would
# be a circular import.
from .streamer import AsyncFauxStreamer, SyncFauxStreamer  # noqa: E402

# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class FauxTransport(BaseTransport, ChatCompletionTransportMixin, FauxWireMixin):
    """Scripted-response transport. set_responses(...) populates a queue;
    each completion() pops one. Thread-safe."""

    transport_id = "faux"

    STREAMER = SyncFauxStreamer
    ASYNC_STREAMER = AsyncFauxStreamer

    def __init__(
        self,
        *,
        provider: str = "faux",
        base_url: str = "",
        api_key: str | None = None,
        timeout: float | None = 60.0,
        http_client: Any = None,
        async_http_client: Any = None,
        tokens_per_second: float | int = 0,
    ) -> None:
        # Skip BaseTransport's httpx setup — we don't talk over HTTP.
        self._provider = provider
        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout
        self._owned_client = False
        self._owned_aclient = False
        self._client = None
        self._aclient = None

        self.tokens_per_second = tokens_per_second
        self._responses: list[_FauxAssistantMessage] = []
        # Every request that reaches this transport, in order — lets tests assert
        # what the caller actually sent (e.g. the projected tool list).
        self.requests: list[ChatCompletionRequest] = []
        self._lock = threading.Lock()

    def set_responses(self, responses: list[_FauxAssistantMessage]) -> None:
        with self._lock:
            self._responses = list(responses)

    def _pop(self) -> _FauxAssistantMessage:
        with self._lock:
            if not self._responses:
                raise RuntimeError("FauxTransport: no more scripted responses")
            return self._responses.pop(0)

    def _record_request(self, request: ChatCompletionRequest) -> None:
        with self._lock:
            self.requests.append(request)

    # --- lifecycle (no-op) ---
    def close(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    # --- non-streaming ---

    def completion(
        self,
        request: ChatCompletionRequest,
        *,
        timeout: float | None = None,
    ) -> ChatCompletionResponse:
        self._record_request(request)
        scripted = self._pop()
        if any(isinstance(b, _FauxHang) for b in scripted.blocks):
            raise RuntimeError("faux_hang() is async-only; use acompletion")
        return self._respond(scripted, request)

    async def acompletion(
        self,
        request: ChatCompletionRequest,
        *,
        timeout: float | None = None,
    ) -> ChatCompletionResponse:
        self._record_request(request)
        scripted = self._pop()
        if any(isinstance(b, _FauxHang) for b in scripted.blocks):
            await asyncio.Event().wait()  # hangs until cancelled / timed out
        return self._respond(scripted, request)

    def _respond(
        self,
        scripted: _FauxAssistantMessage,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        if scripted.error is not None:
            err_cls = scripted.error.error_class or ClientError
            raise err_cls(scripted.error.message, provider=self._provider)
        return self._build_response(scripted, request)

    # --- streaming (pop eagerly at call time, after recording) ---

    def completion_stream(
        self,
        request: ChatCompletionRequest,
        *,
        timeout: float | None = None,
    ) -> SyncFauxStreamer:
        self._record_request(request)
        scripted = self._pop()
        return self.STREAMER(
            request,
            provider=self._provider,
            scripted=scripted,
            timeout=timeout,
        )

    def acompletion_stream(
        self,
        request: ChatCompletionRequest,
        *,
        timeout: float | None = None,
    ) -> AsyncFauxStreamer:
        self._record_request(request)
        scripted = self._pop()
        return self.ASYNC_STREAMER(
            request,
            provider=self._provider,
            scripted=scripted,
            timeout=timeout,
        )

    def _build_response(
        self,
        scripted: _FauxAssistantMessage,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        content = self._materialize_blocks(scripted.blocks)
        message = AssistantMessage(
            content=content,
            provider=self._provider,
            model=request.model,
        )
        message.usage = scripted.usage or Usage()
        canonical, err_msg = self._classify_finish(scripted.finish_reason, message)
        message.finish_reason = canonical
        message.provider_finish_reason = scripted.finish_reason
        message.error_message = err_msg
        resp = ChatCompletionResponse(messages=[message])
        resp._response_format = request.response_format
        return resp

    def _materialize_blocks(self, blocks: list) -> list:
        out: list = []
        for b in blocks:
            if isinstance(b, _FauxText):
                out.append(TextBlock(text=b.text, annotations=list(b.annotations or [])))
            elif isinstance(b, (PrivateProviderBlock, WebSearchBlock, WebFetchBlock)):
                # Scripted web content is passed as the REAL client blocks —
                # played back verbatim (copied, so a reused script cannot
                # alias state into two responses).
                out.append(b.model_copy(deep=True))
            elif isinstance(b, _FauxThinking):
                out.append(
                    ThinkingBlock(
                        text=b.text,
                        signature=b.signature,
                        redacted=b.redacted,
                    )
                )
            elif isinstance(b, _FauxToolCall):
                out.append(
                    ToolCall(
                        id=b.id,
                        name=b.name,
                        arguments=b.arguments,
                        complete=True,
                    ),
                )
            elif isinstance(b, _FauxRefusal):
                out.append(RefusalBlock(text=b.text))
            else:
                raise ValueError(f"Unknown faux block type {type(b).__name__}")
        return out
