"""Faux (scripted) transport — for tests. Does not use httpx.

Builders return small dataclasses representing intended scripted output;
FauxTransport plays them back as ChatCompletionResponse / stream events.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

from ...exceptions import ClientError, StreamError
from ...exceptions import TimeoutError as SDKTimeoutError
from ...types.completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
)
from ...types.content import RefusalBlock, TextBlock, ThinkingBlock, ToolCall
from ...types.messages import AssistantMessage
from ...types.streaming import (
    AsyncChatCompletionStream,
    ChatCompletionStream,
    ErrorEvent,
    FinishEvent,
    RawBlockStart,
    RawBlockStop,
    RawFinish,
    RawRefusalDelta,
    RawStreamEvent,
    RawTextDelta,
    RawThinkingDelta,
    RawToolArgumentsDelta,
    RawUsage,
    StartEvent,
)
from ..base import BaseTransport, ChatCompletionTransportMixin


# ---------------------------------------------------------------------------
# Scripted-response builders
# ---------------------------------------------------------------------------


@dataclass
class _FauxText:
    text: str


@dataclass
class _FauxThinking:
    text: str
    signature: str | None = None


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
    for cancellation / total-timeout tests. Async-only: the sync surfaces
    raise on it (a sync hang would just freeze the test)."""


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


def faux_text(text: str) -> _FauxText:
    return _FauxText(text=text)


def faux_thinking(text: str, signature: str | None = None) -> _FauxThinking:
    return _FauxThinking(text=text, signature=signature)


def faux_tool_call(name: str, arguments: dict, id: str = "tool_call_faux") -> _FauxToolCall:
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
        blocks=blocks, finish_reason=finish_reason, error=error, usage=usage,
    )


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class FauxTransport(BaseTransport, ChatCompletionTransportMixin):
    """Scripted-response transport. set_responses(...) populates a queue;
    each completion() pops one. Thread-safe."""

    transport_id = "faux"

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

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self._record_request(request)
        scripted = self._pop()
        if any(isinstance(b, _FauxHang) for b in scripted.blocks):
            raise RuntimeError("faux_hang() is async-only; use acompletion")
        return self._respond(scripted, request)

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self._record_request(request)
        scripted = self._pop()
        if any(isinstance(b, _FauxHang) for b in scripted.blocks):
            await asyncio.Event().wait()  # hangs until cancelled / timed out
        return self._respond(scripted, request)

    def _respond(
        self, scripted: _FauxAssistantMessage, request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        if scripted.error is not None:
            err_cls = scripted.error.error_class or ClientError
            raise err_cls(scripted.error.message, provider=self._provider)
        return self._build_response(scripted, request)

    def completion_stream(self, request: ChatCompletionRequest) -> "FauxChatCompletionStream":
        self._record_request(request)
        scripted = self._pop()
        return FauxChatCompletionStream(
            transport=self, request=request, scripted=scripted,
        )

    def acompletion_stream(self, request: ChatCompletionRequest) -> "FauxAsyncChatCompletionStream":
        self._record_request(request)
        scripted = self._pop()
        return FauxAsyncChatCompletionStream(
            transport=self, request=request, scripted=scripted,
        )

    def _build_response(
        self, scripted: _FauxAssistantMessage, request: ChatCompletionRequest,
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
        resp = ChatCompletionResponse(message=message)
        resp._response_format = request.response_format
        return resp

    def _materialize_blocks(self, blocks: list) -> list:
        out: list = []
        for b in blocks:
            if isinstance(b, _FauxText):
                out.append(TextBlock(text=b.text))
            elif isinstance(b, _FauxThinking):
                out.append(ThinkingBlock(text=b.text, signature=b.signature))
            elif isinstance(b, _FauxToolCall):
                out.append(
                    ToolCall(
                        id=b.id, name=b.name, arguments=b.arguments, complete=True,
                    ),
                )
            elif isinstance(b, _FauxRefusal):
                out.append(RefusalBlock(text=b.text))
            else:
                raise ValueError(f"Unknown faux block type {type(b).__name__}")
        return out

    # --- finish-reason classification ---

    def _classify_finish(
        self, provider_value: str | None, message: AssistantMessage,
    ) -> tuple[str | None, str | None]:
        # The faux speaks SDK-canonical values directly — callers pass
        # `finish_reason="stop"`, `"tool_use"`, `"error"`, `"length"` etc.
        if provider_value == "error":
            # If any refusal block present, derive message from it.
            for b in message.content:
                if isinstance(b, RefusalBlock):
                    return ("error", f"Faux refusal: {b.text}")
            return ("error", "Faux error terminal")
        return (provider_value, None)

    # --- not used (faux has no httpx) ---
    def _build_chat_completion_payload(self, request, *, stream=False):  # pragma: no cover
        raise NotImplementedError("FauxTransport does not build wire payloads")

    def _parse_chat_completion_response(self, response, request):  # pragma: no cover
        raise NotImplementedError

    def _map_chat_completion_http_error(self, exc):  # pragma: no cover
        return ClientError(str(exc), provider=self._provider, original_exception=exc)

    def _chat_completion_stream_class(self) -> type:
        return FauxChatCompletionStream

    def _async_chat_completion_stream_class(self) -> type:
        return FauxAsyncChatCompletionStream


# ---------------------------------------------------------------------------
# Faux streaming
# ---------------------------------------------------------------------------


def _scripted_raw_events(
    scripted: _FauxAssistantMessage,
) -> Iterator[RawStreamEvent]:
    """Convert the scripted message into a sequence of RawStreamEvents."""
    next_idx = 0
    for block in scripted.blocks:
        if isinstance(block, _FauxText):
            i = next_idx
            next_idx += 1
            yield RawBlockStart(index=i, block_type="text")
            # Stream the text as one delta — token pacing only matters for
            # tests that read tokens_per_second; v1 emits one delta per block.
            if block.text:
                yield RawTextDelta(index=i, text=block.text)
            yield RawBlockStop(index=i)
        elif isinstance(block, _FauxThinking):
            i = next_idx
            next_idx += 1
            yield RawBlockStart(index=i, block_type="thinking")
            if block.text:
                yield RawThinkingDelta(index=i, text=block.text, signature=block.signature)
            yield RawBlockStop(index=i)
        elif isinstance(block, _FauxToolCall):
            import json as _json
            i = next_idx
            next_idx += 1
            yield RawBlockStart(
                index=i, block_type="tool_call",
                tool_id=block.id, tool_name=block.name,
            )
            args_str = _json.dumps(block.arguments) if block.arguments else "{}"
            yield RawToolArgumentsDelta(index=i, arguments_delta=args_str)
            yield RawBlockStop(index=i)
        elif isinstance(block, _FauxRefusal):
            i = next_idx
            next_idx += 1
            yield RawBlockStart(index=i, block_type="refusal")
            if block.text:
                yield RawRefusalDelta(index=i, text=block.text)
            yield RawBlockStop(index=i)
        elif isinstance(block, _FauxHang):
            yield block  # marker — each stream surface decides how to hang
        else:
            raise ValueError(f"Unknown faux block type {type(block).__name__}")

    if scripted.error is not None:
        raise StreamError(scripted.error.message)

    if scripted.usage is not None:
        yield RawUsage(usage=scripted.usage)
    yield RawFinish(reason=scripted.finish_reason)


class FauxChatCompletionStream(ChatCompletionStream):
    """Faux sync stream. No httpx — directly yields scripted raw events."""

    def __init__(
        self, *, transport: FauxTransport, request: ChatCompletionRequest,
        scripted: _FauxAssistantMessage,
    ) -> None:
        super().__init__(request=request, transport=transport)
        self._scripted = scripted

    def _open_http(self) -> Any:
        # No HTTP. Return a dummy CM-like object whose __exit__ is a no-op.
        class _NoopCM:
            def __enter__(_self):
                return _self

            def __exit__(_self, *exc):
                pass

            status_code = 200
            is_closed = False

            def read(_self):
                return b""

            def raise_for_status(_self):
                return None

            def close(_self):
                pass

        cm = _NoopCM()
        return cm

    def _open(self) -> None:
        if self._http_response is None:
            self._http_cm = self._open_http()
            self._http_response = self._http_cm.__enter__()

    def parse_chunks(self) -> Iterator[RawStreamEvent]:
        for ev in _scripted_raw_events(self._scripted):
            if isinstance(ev, _FauxHang):
                raise RuntimeError(
                    "faux_hang() is async-only; use acompletion_stream",
                )
            yield ev

    def _handle_iter_exception(self, exc: BaseException) -> Iterator[Any]:
        # Override: the faux is httpx-free, so the OpenAI/Anthropic-style
        # httpx mapping isn't useful. Cancellation and StreamError flow through
        # the base; everything else propagates.
        if self._cancelled:
            yield self._acc.build_terminal_finish(
                classify_finish=self._transport._classify_finish,
                cancelled=True,
            )
            return
        if isinstance(exc, StreamError):
            yield self._acc.build_error_event(exc)
            return
        raise exc


class FauxAsyncChatCompletionStream(AsyncChatCompletionStream):
    def __init__(
        self, *, transport: FauxTransport, request: ChatCompletionRequest,
        scripted: _FauxAssistantMessage,
    ) -> None:
        super().__init__(request=request, transport=transport)
        self._scripted = scripted

    async def _open_http(self) -> Any:
        class _NoopACM:
            async def __aenter__(_self):
                return _self

            async def __aexit__(_self, *exc):
                pass

            status_code = 200
            is_closed = False

            async def aread(_self):
                return b""

            def raise_for_status(_self):
                return None

            async def aclose(_self):
                pass

        return _NoopACM()

    async def _aopen(self) -> None:
        if self._http_response is None:
            self._http_cm = await self._open_http()
            self._http_response = await self._http_cm.__aenter__()

    async def parse_chunks(self) -> AsyncIterator[RawStreamEvent]:
        for ev in _scripted_raw_events(self._scripted):
            if isinstance(ev, _FauxHang):
                await asyncio.Event().wait()  # until cancelled / timed out
            yield ev

    async def _handle_iter_exception(self, exc: BaseException) -> AsyncIterator[Any]:
        if self._cancelled:
            yield self._acc.build_terminal_finish(
                classify_finish=self._transport._classify_finish,
                cancelled=True,
            )
            return
        if isinstance(exc, SDKTimeoutError):  # total_timeout expiry
            self._acc._message.error_message = str(exc)
            yield ErrorEvent(
                error=exc,
                partial_message=self._acc._message.model_copy(deep=True),
                usage=self._acc._usage,
            )
            return
        if isinstance(exc, StreamError):
            yield self._acc.build_error_event(exc)
            return
        raise exc
