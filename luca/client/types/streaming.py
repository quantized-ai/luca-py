"""Streaming protocol: the `StreamEvent` discriminated union, the
`RawStreamEvent` vocabulary (transport-internal), and the BaseStream /
ChatCompletionStream class hierarchy.

See architecture.md §10 for the full design. The public event union is
described in api_prd.md §12.6.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import warnings
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SerializeAsAny

from ..exceptions import ClientError, StreamError, TimeoutError as SDKTimeoutError
from .completion import Usage
from .content import RefusalBlock, TextBlock, ThinkingBlock, ToolCall
from .messages import AssistantMessage

if TYPE_CHECKING:
    from .completion import ChatCompletionResponse


# ---------------------------------------------------------------------------
# Public stream events (the StreamEvent discriminated union)
# ---------------------------------------------------------------------------


class StartEvent(BaseModel):
    type: Literal["start"] = "start"

    model_config = ConfigDict(extra="forbid")


class TextStartEvent(BaseModel):
    type: Literal["text_start"] = "text_start"
    index: int

    model_config = ConfigDict(extra="forbid")


class TextDeltaEvent(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    index: int
    delta: str

    model_config = ConfigDict(extra="forbid")


class TextEndEvent(BaseModel):
    type: Literal["text_end"] = "text_end"
    index: int
    content: str

    model_config = ConfigDict(extra="forbid")


class ThinkingStartEvent(BaseModel):
    type: Literal["thinking_start"] = "thinking_start"
    index: int

    model_config = ConfigDict(extra="forbid")


class ThinkingDeltaEvent(BaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    index: int
    delta: str

    model_config = ConfigDict(extra="forbid")


class ThinkingEndEvent(BaseModel):
    type: Literal["thinking_end"] = "thinking_end"
    index: int
    content: str

    model_config = ConfigDict(extra="forbid")


class ToolCallStartEvent(BaseModel):
    type: Literal["tool_call_start"] = "tool_call_start"
    index: int
    id: str
    name: str

    model_config = ConfigDict(extra="forbid")


class ToolCallDeltaEvent(BaseModel):
    type: Literal["tool_call_delta"] = "tool_call_delta"
    index: int
    arguments_delta: str

    model_config = ConfigDict(extra="forbid")


class ToolCallEndEvent(BaseModel):
    type: Literal["tool_call_end"] = "tool_call_end"
    index: int
    # SerializeAsAny: native ToolCall subclass fields must survive a dump,
    # same rule as AssistantMessage.content.
    tool_call: SerializeAsAny[ToolCall]

    model_config = ConfigDict(extra="forbid")


class RefusalStartEvent(BaseModel):
    type: Literal["refusal_start"] = "refusal_start"
    index: int

    model_config = ConfigDict(extra="forbid")


class RefusalDeltaEvent(BaseModel):
    type: Literal["refusal_delta"] = "refusal_delta"
    index: int
    delta: str

    model_config = ConfigDict(extra="forbid")


class RefusalEndEvent(BaseModel):
    type: Literal["refusal_end"] = "refusal_end"
    index: int
    content: str

    model_config = ConfigDict(extra="forbid")


class UsageEvent(BaseModel):
    type: Literal["usage"] = "usage"
    usage: Usage

    model_config = ConfigDict(extra="forbid")


class FinishEvent(BaseModel):
    type: Literal["finish"] = "finish"
    message: AssistantMessage
    finish_reason: str | None
    provider_finish_reason: str | None
    cancelled: bool = False
    usage: Usage
    tool_calls: list[SerializeAsAny[ToolCall]] = Field(default_factory=list)

    _response_format: Any | None = PrivateAttr(default=None)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def parse(self) -> Any:
        if self._response_format is None:
            raise ValueError("No response_format was set on the originating request; cannot parse().")
        from .structured import parse_structured_output

        text = "".join(block.text for block in self.message.content if isinstance(block, TextBlock))
        return parse_structured_output(text, self._response_format)


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    error: ClientError
    usage: Usage | None = None
    raw: Any = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)


StreamEvent = Annotated[
    StartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | RefusalStartEvent
    | RefusalDeltaEvent
    | RefusalEndEvent
    | UsageEvent
    | FinishEvent
    | ErrorEvent,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# RawStreamEvent vocabulary (transport-internal)
# ---------------------------------------------------------------------------


@dataclass
class RawBlockStart:
    index: int
    block_type: Literal["text", "thinking", "tool_call", "refusal"]
    tool_id: str | None = None
    tool_name: str | None = None
    # A redacted thinking block arrives whole in the start event — encrypted
    # payload, no deltas — so it has to be carried here or it is lost.
    signature: str | None = None
    redacted: bool = False
    # The provider's opaque id for a thinking block's underlying item, which
    # arrives when the item opens (OpenAI Responses: `rs_…`) — before any
    # delta, so RawThinkingDelta is too late to carry it.
    item_id: str | None = None
    # A transport-built typed ToolCall (native tools): appended as-is with
    # complete=False. The accumulator stays projector-blind — it never sees
    # more than the finished object.
    prebuilt: ToolCall | None = None


@dataclass
class RawBlockStop:
    index: int
    # The complete native ToolCall built when its wire item finished: swapped
    # in whole, so final arguments arrive without a JSON parse. None keeps
    # today's path (parse partial_arguments).
    replacement: ToolCall | None = None


@dataclass
class RawTextDelta:
    index: int
    text: str


@dataclass
class RawThinkingDelta:
    index: int
    text: str
    signature: str | None = None


@dataclass
class RawToolArgumentsDelta:
    index: int
    arguments_delta: str


@dataclass
class RawRefusalDelta:
    index: int
    text: str


@dataclass
class RawFinish:
    reason: str


@dataclass
class RawUsage:
    usage: Usage


RawStreamEvent = (
    RawBlockStart
    | RawTextDelta
    | RawThinkingDelta
    | RawToolArgumentsDelta
    | RawRefusalDelta
    | RawBlockStop
    | RawFinish
    | RawUsage
)


# ---------------------------------------------------------------------------
# BaseStream / AsyncBaseStream
# ---------------------------------------------------------------------------


class BaseStream:
    """Surface-agnostic streaming wrapper. Owns httpx-response lifecycle,
    cancellation, context manager, and the ResourceWarning safety net."""

    def __init__(self, *, request: Any, provider: str) -> None:
        self._request = request
        self._provider = provider
        self._http_cm: Any = None
        self._http_response: Any = None
        self._cancelled: bool = False
        self._consumed: bool = False

    # --- subclass hooks ---

    def _open_http(self) -> Any:
        raise NotImplementedError

    def parse_chunks(self) -> Iterator[RawStreamEvent]:
        raise NotImplementedError

    def _apply(self, event: Any) -> None:
        pass

    def _handle_iter_exception(self, exc: BaseException) -> Iterator[Any]:
        raise exc

    # --- lifecycle ---

    def _open(self) -> None:
        if self._http_response is None:
            self._http_cm = self._open_http()
            self._http_response = self._http_cm.__enter__()
            if self._http_response.status_code >= 400:
                self._http_response.read()
                self._http_response.raise_for_status()

    def _close(self) -> None:
        if self._http_cm is not None:
            try:
                self._http_cm.__exit__(None, None, None)
            except Exception:
                pass
            finally:
                self._http_cm = None
                self._http_response = None

    def __enter__(self) -> BaseStream:
        return self

    def __exit__(self, *exc: Any) -> None:
        self._close()

    def cancel(self) -> None:
        self._cancelled = True
        if self._http_response is not None:
            try:
                if not self._http_response.is_closed:
                    self._http_response.close()
            except Exception:
                pass

    def __iter__(self) -> Iterator[Any]:
        if self._consumed:
            raise StreamError(f"{type(self).__name__} has already been consumed")
        self._consumed = True

        self._open()
        try:
            for event in self.parse_chunks():
                self._apply(event)
                yield event
        except Exception as exc:
            yield from self._handle_iter_exception(exc)
        finally:
            self._close()

    def __del__(self) -> None:
        http_response = getattr(self, "_http_response", None)
        if http_response is not None:
            try:
                is_closed = http_response.is_closed
            except Exception:
                is_closed = True
            if not is_closed:
                warnings.warn(
                    "Stream was garbage-collected without being closed. "
                    "Use `with completion_stream(...) as s:` or call s.cancel() / s.collect().",
                    ResourceWarning,
                    stacklevel=2,
                    source=self,
                )
                with contextlib.suppress(Exception):
                    http_response.close()


class AsyncBaseStream:
    """Async mirror of BaseStream. __del__ can only sync-close."""

    def __init__(self, *, request: Any, provider: str) -> None:
        self._request = request
        self._provider = provider
        self._http_cm: Any = None
        self._http_response: Any = None
        self._cancelled: bool = False
        self._consumed: bool = False
        self._total_timeout: float | None = None
        self._deadline: float | None = None

    def _set_total_timeout(self, seconds: float) -> None:
        """Arm the wall-clock deadline (`acompletion_stream(total_timeout=)`):
        recorded at open, enforced on every chunk pull with the remaining
        time. Expiry yields exactly one terminal ErrorEvent carrying the SDK
        TimeoutError, then the stream closes."""
        self._total_timeout = seconds

    async def _open_http(self) -> Any:
        raise NotImplementedError

    async def parse_chunks(self) -> AsyncIterator[RawStreamEvent]:
        raise NotImplementedError
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]

    def _apply(self, event: Any) -> None:
        pass

    async def _handle_iter_exception(self, exc: BaseException) -> AsyncIterator[Any]:
        raise exc
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]

    async def _aopen(self) -> None:
        if self._http_response is None:
            self._http_cm = await self._open_http()
            self._http_response = await self._http_cm.__aenter__()
            if self._http_response.status_code >= 400:
                await self._http_response.aread()
                self._http_response.raise_for_status()

    async def _aclose(self) -> None:
        if self._http_cm is not None:
            try:
                await self._http_cm.__aexit__(None, None, None)
            except Exception:
                pass
            finally:
                self._http_cm = None
                self._http_response = None

    async def __aenter__(self) -> AsyncBaseStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._aclose()

    async def cancel(self) -> None:
        self._cancelled = True
        if self._http_response is not None:
            try:
                if not self._http_response.is_closed:
                    await self._http_response.aclose()
            except Exception:
                pass

    async def __aiter__(self) -> AsyncIterator[Any]:
        if self._consumed:
            raise StreamError(f"{type(self).__name__} has already been consumed")
        self._consumed = True

        await self._aopen()
        try:
            async for event in self.parse_chunks():
                self._apply(event)
                yield event
        except Exception as exc:
            async for ev in self._handle_iter_exception(exc):
                yield ev
        finally:
            await self._aclose()

    def __del__(self) -> None:
        http_response = getattr(self, "_http_response", None)
        if http_response is not None:
            try:
                is_closed = http_response.is_closed
            except Exception:
                is_closed = True
            if not is_closed:
                warnings.warn(
                    "AsyncStream was garbage-collected without being closed. "
                    "Use `async with acompletion_stream(...) as s:` or call await s.cancel() / await s.collect().",
                    ResourceWarning,
                    stacklevel=2,
                    source=self,
                )
                # Cannot await aclose() from __del__; rely on transport teardown.


# ---------------------------------------------------------------------------
# ChatCompletionStream (sync) — orchestrator + single mutator
# ---------------------------------------------------------------------------


class _ChatCompletionAccumulator:
    """Shared state machine used by both sync and async chat-completion streams.

    Owns the AssistantMessage being built, applies RawStreamEvents into it, and
    decides which public StreamEvents to emit. Single mutator of self._message;
    single emitter of public events."""

    def __init__(self, *, request: Any, provider: str) -> None:
        self._request = request
        self._provider = provider
        self._message = AssistantMessage(
            content=[],
            provider=provider,
            model=request.model,
        )
        self._terminal: str | None = None
        self._canonical_finish: str | None = None
        self._error_message: str | None = None
        self._usage: Usage | None = None
        self._open_block_indices: set[int] = set()

    @property
    def message(self) -> AssistantMessage:
        return self._message

    @property
    def usage(self) -> Usage | None:
        return self._usage

    @property
    def terminal(self) -> str | None:
        return self._terminal

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self._message.tool_calls

    @property
    def text(self) -> str:
        return "".join(b.text for b in self._message.content if isinstance(b, TextBlock))

    def start_event(self) -> StartEvent:
        return StartEvent()

    def handle_raw(self, raw: RawStreamEvent) -> Iterator[Any]:
        # Single mutator + single emitter; same logic across sync/async streams.

        def _check_start(i: int) -> None:
            if i != len(self._message.content):
                raise StreamError(
                    f"RawBlockStart.index={i} does not match expected next index "
                    f"{len(self._message.content)} (block indices must be dense and ordered)",
                )
            if i in self._open_block_indices:
                raise StreamError(
                    f"RawBlockStart.index={i} is already an open block",
                )

        def _check_delta(i: int, expected_type: type) -> None:
            if i not in self._open_block_indices:
                raise StreamError(
                    f"Delta references index {i} which is not an open block (open: {sorted(self._open_block_indices)})",
                )
            block = self._message.content[i]
            if not isinstance(block, expected_type):
                raise StreamError(
                    f"Delta type does not match block type {type(block).__name__} at index {i}",
                )

        if isinstance(raw, RawBlockStart):
            _check_start(raw.index)
            if raw.block_type == "text":
                self._message.content.append(TextBlock(text=""))
                self._open_block_indices.add(raw.index)
                yield TextStartEvent(index=raw.index)
            elif raw.block_type == "thinking":
                self._message.content.append(
                    ThinkingBlock(
                        text="",
                        id=raw.item_id,
                        signature=raw.signature,
                        redacted=raw.redacted,
                    ),
                )
                self._open_block_indices.add(raw.index)
                yield ThinkingStartEvent(index=raw.index)
            elif raw.block_type == "refusal":
                self._message.content.append(RefusalBlock(text=""))
                self._open_block_indices.add(raw.index)
                yield RefusalStartEvent(index=raw.index)
            elif raw.block_type == "tool_call":
                if raw.prebuilt is not None:
                    # Transport-built typed call (native tools), still
                    # in progress; the stop's `replacement` finishes it.
                    self._message.content.append(raw.prebuilt)
                    self._open_block_indices.add(raw.index)
                    yield ToolCallStartEvent(
                        index=raw.index,
                        id=raw.prebuilt.id,
                        name=raw.prebuilt.name,
                    )
                elif raw.tool_id is None or raw.tool_name is None:
                    raise StreamError(
                        f"RawBlockStart(tool_call, index={raw.index}) missing tool_id or tool_name",
                    )
                else:
                    self._message.content.append(
                        ToolCall(
                            id=raw.tool_id,
                            name=raw.tool_name,
                            arguments={},
                            partial_arguments="",
                            complete=False,
                        )
                    )
                    self._open_block_indices.add(raw.index)
                    yield ToolCallStartEvent(
                        index=raw.index,
                        id=raw.tool_id,
                        name=raw.tool_name,
                    )
            else:
                raise StreamError(
                    f"Unknown block_type={raw.block_type!r}",
                )

        elif isinstance(raw, RawTextDelta):
            _check_delta(raw.index, TextBlock)
            block = self._message.content[raw.index]
            assert isinstance(block, TextBlock)
            block.text += raw.text
            yield TextDeltaEvent(index=raw.index, delta=raw.text)

        elif isinstance(raw, RawThinkingDelta):
            _check_delta(raw.index, ThinkingBlock)
            block = self._message.content[raw.index]
            assert isinstance(block, ThinkingBlock)
            block.text += raw.text
            if raw.signature is not None:
                block.signature = raw.signature
            yield ThinkingDeltaEvent(index=raw.index, delta=raw.text)

        elif isinstance(raw, RawToolArgumentsDelta):
            _check_delta(raw.index, ToolCall)
            block = self._message.content[raw.index]
            assert isinstance(block, ToolCall)
            block.partial_arguments += raw.arguments_delta
            yield ToolCallDeltaEvent(index=raw.index, arguments_delta=raw.arguments_delta)

        elif isinstance(raw, RawRefusalDelta):
            _check_delta(raw.index, RefusalBlock)
            block = self._message.content[raw.index]
            assert isinstance(block, RefusalBlock)
            block.text += raw.text
            yield RefusalDeltaEvent(index=raw.index, delta=raw.text)

        elif isinstance(raw, RawBlockStop):
            if raw.index not in self._open_block_indices:
                raise StreamError(
                    f"RawBlockStop.index={raw.index} is not an open block (open: {sorted(self._open_block_indices)})",
                )
            self._open_block_indices.remove(raw.index)
            block = self._message.content[raw.index]
            if isinstance(block, TextBlock):
                yield TextEndEvent(index=raw.index, content=block.text)
            elif isinstance(block, ThinkingBlock):
                yield ThinkingEndEvent(index=raw.index, content=block.text)
            elif isinstance(block, RefusalBlock):
                yield RefusalEndEvent(index=raw.index, content=block.text)
            elif isinstance(block, ToolCall):
                if raw.replacement is not None:
                    # The complete native call: final arguments arrive whole,
                    # no JSON parse of partial_arguments.
                    self._message.content[raw.index] = raw.replacement
                    yield ToolCallEndEvent(
                        index=raw.index,
                        tool_call=raw.replacement,
                    )
                else:
                    try:
                        block.arguments = json.loads(block.partial_arguments) if block.partial_arguments else {}
                    except json.JSONDecodeError as e:
                        raise StreamError(
                            f"Tool call {raw.index} ({block.name!r}) returned malformed JSON",
                        ) from e
                    block.complete = True
                    block.partial_arguments = ""
                    yield ToolCallEndEvent(
                        index=raw.index,
                        tool_call=block,
                    )

        elif isinstance(raw, RawFinish):
            self._terminal = raw.reason

        elif isinstance(raw, RawUsage):
            self._usage = raw.usage
            yield UsageEvent(usage=raw.usage)

        else:
            raise StreamError(
                f"Unknown raw event type {type(raw).__name__!r}",
            )

    def build_terminal_finish(self, *, classify_finish: Any, cancelled: bool = False) -> FinishEvent:
        """Run classify_finish (if a terminal was received) and build FinishEvent."""
        if self._terminal is not None:
            canonical, err_msg = classify_finish(self._terminal, self._message)
            self._canonical_finish = canonical
            self._error_message = err_msg

        usage = self._usage or Usage()
        self._message.finish_reason = self._canonical_finish
        self._message.provider_finish_reason = self._terminal
        self._message.error_message = self._error_message
        self._message.cancelled = cancelled
        self._message.usage = usage

        finish = FinishEvent(
            message=self._message.model_copy(deep=True),
            finish_reason=self._canonical_finish,
            provider_finish_reason=self._terminal,
            cancelled=cancelled,
            usage=usage,
            tool_calls=list(self._message.tool_calls),
        )
        finish._response_format = getattr(self._request, "response_format", None)
        return finish

    def build_error_event(self, exc: StreamError) -> ErrorEvent:
        self._message.error_message = str(exc)
        return ErrorEvent(error=exc, usage=self._usage)


class ChatCompletionStream(BaseStream):
    """Sync chat-completion stream. Single mutator of self._message via the
    accumulator. Single emitter of public StreamEvents."""

    def __init__(self, *, request: Any, transport: Any) -> None:
        super().__init__(request=request, provider=transport._provider)
        self._transport = transport
        self._acc = _ChatCompletionAccumulator(request=request, provider=transport._provider)

    # --- live accessors ---
    @property
    def message(self) -> AssistantMessage:
        return self._acc.message

    @property
    def text(self) -> str:
        return self._acc.text

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self._acc.tool_calls

    @property
    def finish_reason(self) -> str | None:
        return self._acc._canonical_finish

    @property
    def provider_finish_reason(self) -> str | None:
        return self._acc._terminal

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def usage(self) -> Usage | None:
        return self._acc._usage

    def _open(self) -> None:
        """Map a rejected request to a `ClientError`, exactly as the
        non-streaming path does.

        `BaseStream._open` ends in `raise_for_status()`, and it runs BEFORE the
        `try` that routes httpx failures through the transport's mapper — so
        without this the caller gets a bare `httpx.HTTPStatusError` ("Client
        error '402 Payment Required' … see MDN") and the body OpenRouter
        actually sent ("This request requires more credits, or fewer
        max_tokens…") would be discarded. Every pre-stream status code rides
        on this: 401, 402, 429 (and its `retry_after`), 400, 404.

        It RAISES rather than yielding a terminal `ErrorEvent`: nothing has
        been streamed yet, so there is no stream to terminate, and the same
        rejection must not change shape just because `stream=True` — the
        non-streaming path raises the same mapped error
        (`transports/base.py`). Mid-stream failures keep their `ErrorEvent`
        (`_handle_iter_exception`): by then a stream exists to terminate.

        Here and not in `BaseStream`, because `_transport` — and therefore a
        mapper — only exists from this class down."""
        import httpx as _httpx  # deferred, as everywhere else in this module

        try:
            super()._open()
        except _httpx.HTTPError as exc:
            raise self._transport._map_chat_completion_http_error(exc) from exc

    def __iter__(self) -> Iterator[Any]:
        if self._consumed:
            raise StreamError(f"{type(self).__name__} has already been consumed")
        self._consumed = True

        self._open()
        try:
            yield self._acc.start_event()
            for raw in self.parse_chunks():
                yield from self._acc.handle_raw(raw)

            if self._acc._terminal is None:
                raise StreamError(
                    "Stream ended without RawFinish — provider closed the wire without sending a finish reason",
                )

            yield self._acc.build_terminal_finish(
                classify_finish=self._transport._classify_finish,
                cancelled=False,
            )
        except Exception as exc:
            yield from self._handle_iter_exception(exc)
        finally:
            self._close()

    def _handle_iter_exception(self, exc: BaseException) -> Iterator[Any]:
        if self._cancelled:
            yield self._acc.build_terminal_finish(
                classify_finish=self._transport._classify_finish,
                cancelled=True,
            )
            return

        if isinstance(exc, StreamError):
            yield self._acc.build_error_event(exc)
            return

        # Transport-layer (HTTP / network / etc.) — delegate to transport mapper.
        import httpx as _httpx

        if isinstance(exc, _httpx.HTTPError):
            err = self._transport._map_chat_completion_http_error(exc)
            self._acc._message.error_message = str(err)
            yield ErrorEvent(error=err, usage=self._acc._usage)
            return

        raise exc

    def collect(self) -> ChatCompletionResponse:
        from .completion import ChatCompletionResponse

        with self:
            for event in self:
                if event.type == "finish":
                    response = ChatCompletionResponse(messages=[event.message])
                    response._response_format = getattr(self._request, "response_format", None)
                    return response
                if event.type == "error":
                    raise event.error
        raise StreamError(
            "Stream ended without terminal event",
        )


class AsyncChatCompletionStream(AsyncBaseStream):
    """Async chat-completion stream — structural mirror of ChatCompletionStream."""

    def __init__(self, *, request: Any, transport: Any) -> None:
        super().__init__(request=request, provider=transport._provider)
        self._transport = transport
        self._acc = _ChatCompletionAccumulator(request=request, provider=transport._provider)

    @property
    def message(self) -> AssistantMessage:
        return self._acc.message

    @property
    def text(self) -> str:
        return self._acc.text

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self._acc.tool_calls

    @property
    def finish_reason(self) -> str | None:
        return self._acc._canonical_finish

    @property
    def provider_finish_reason(self) -> str | None:
        return self._acc._terminal

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def usage(self) -> Usage | None:
        return self._acc._usage

    async def _aopen(self) -> None:
        """Async twin of `ChatCompletionStream._open` — see it for why the
        mapping lives on this class and why this raises."""
        import httpx as _httpx  # deferred, as everywhere else in this module

        try:
            await super()._aopen()
        except _httpx.HTTPError as exc:
            raise self._transport._map_chat_completion_http_error(exc) from exc

    async def __aiter__(self) -> AsyncIterator[Any]:
        if self._consumed:
            raise StreamError(f"{type(self).__name__} has already been consumed")
        self._consumed = True

        await self._aopen()
        if self._total_timeout is not None and self._deadline is None:
            self._deadline = asyncio.get_running_loop().time() + self._total_timeout
        chunks = self.parse_chunks()
        try:
            yield self._acc.start_event()
            while True:
                try:
                    raw = await self._next_chunk(chunks)
                except StopAsyncIteration:
                    break
                for ev in self._acc.handle_raw(raw):
                    yield ev

            if self._acc._terminal is None:
                raise StreamError(
                    "Stream ended without RawFinish — provider closed the wire without sending a finish reason",
                )

            yield self._acc.build_terminal_finish(
                classify_finish=self._transport._classify_finish,
                cancelled=False,
            )
        except Exception as exc:
            async for ev in self._handle_iter_exception(exc):
                yield ev
        finally:
            await chunks.aclose()
            await self._aclose()

    async def _next_chunk(self, chunks: AsyncIterator[RawStreamEvent]) -> RawStreamEvent:
        """One parse_chunks() pull, bounded by the remaining total_timeout."""
        if self._deadline is None:
            return await chunks.__anext__()
        remaining = self._deadline - asyncio.get_running_loop().time()
        try:
            async with asyncio.timeout(max(remaining, 0.0)):
                return await chunks.__anext__()
        except TimeoutError as exc:  # the builtin, raised by asyncio.timeout
            raise SDKTimeoutError(
                f"stream exceeded total_timeout={self._total_timeout}s",
                provider=self._provider,
                original_exception=exc,
            ) from exc

    async def _handle_iter_exception(self, exc: BaseException) -> AsyncIterator[Any]:
        if self._cancelled:
            yield self._acc.build_terminal_finish(
                classify_finish=self._transport._classify_finish,
                cancelled=True,
            )
            return

        if isinstance(exc, SDKTimeoutError):  # total_timeout expiry
            self._acc._message.error_message = str(exc)
            yield ErrorEvent(error=exc, usage=self._acc._usage)
            return

        if isinstance(exc, StreamError):
            yield self._acc.build_error_event(exc)
            return

        import httpx as _httpx

        if isinstance(exc, _httpx.HTTPError):
            err = self._transport._map_chat_completion_http_error(exc)
            self._acc._message.error_message = str(err)
            yield ErrorEvent(error=err, usage=self._acc._usage)
            return

        raise exc

    async def collect(self) -> ChatCompletionResponse:
        from .completion import ChatCompletionResponse

        async with self:
            async for event in self:
                if event.type == "finish":
                    response = ChatCompletionResponse(messages=[event.message])
                    response._response_format = getattr(self._request, "response_format", None)
                    return response
                if event.type == "error":
                    raise event.error
        raise StreamError(
            "Stream ended without terminal event",
        )
