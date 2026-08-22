"""The streaming core: BaseStreamer + SyncStreamerMixin + AsyncStreamerMixin.

Each transport package subclasses BaseStreamer into a WIRE CLASS that owns its
wire end-to-end — request building, chunk parsing, event creation, error
mapping — as overridable methods (`transports/<x>/streamer.py`). Sync/async
iteration and the timeout machinery live in the two mixins, written once. A
concrete streamer is an empty mixin × wire combination:

    class AsyncAnthropicStreamer(AsyncStreamerMixin, AnthropicStreamer): pass

Lifecycle: constructing a streamer touches nothing; `__enter__`/`__aenter__`
sends the HTTP request and reads the response headers (a rejected request
raises the mapped typed exception there); each iteration step is one wire
read → `parse()` → `handle()` → one public event; block exit closes the
response and any owned client. Exactly one terminal event (`FinishEvent` or
`ErrorEvent`) per opened stream, and no exception leaks from iteration except
external task cancellation, which re-raises untouched.

The deadline (`timeout=`) is a total wall clock. Async enforcement is a
`call_later` timer that cancels the streamer-owned read task — never the
consumer's frame. Sync enforcement is cooperative: the clock is checked
before every read, so a read already blocked on the socket runs to
completion first (a platform limit, documented as best-effort).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import warnings
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from ..exceptions import StreamError, TimeoutError as SDKTimeoutError
from ..types.completion import Usage
from ..types.content import TextBlock, ToolCall
from ..types.messages import AssistantMessage
from ..types.streaming import ErrorEvent, FinishEvent, StartEvent

if TYPE_CHECKING:
    from ..types.completion import ChatCompletionRequest, ChatCompletionResponse

_PREMATURE_END = "Stream ended without a terminal event — the provider closed the wire without sending a finish reason"


class BaseStreamer:
    """Shared state, SSE parsing, handler dispatch, and event builders.

    Subclasses fill `HANDLERS_BY_TYPE` with one named handler method per wire
    event type (or override `handle()` for wires without event types), latch
    `self.stop_reason` and `self.usage` as they arrive, and either emit the
    terminal from their finish handler or override `handle_wire_end()` when
    the wire delivers usage after its finish marker."""

    PROVIDER: ClassVar[str | None] = None
    # The wire's default base URL; `_chat_completion_url` composes the
    # endpoint from it. An explicit base_url= wins.
    URL: ClassVar[str | None] = None
    HANDLERS_BY_TYPE: ClassVar[dict[str, str]] = {}
    # The one httpx phase kept alive on stream requests: a black-holed
    # connect must fail in bounded time, but a shared client's read timeout
    # must not kill a slow stream (§ timeout mechanics).
    CONNECT_TIMEOUT: ClassVar[float] = 5.0

    def __init__(
        self,
        request: ChatCompletionRequest,
        *,
        provider: str | None = None,
        httpx_client: Any = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        # Underscored: the inherited wire-mixin methods read these names.
        self._provider = provider or self.PROVIDER
        self._api_key = api_key
        self._base_url = base_url or self.URL
        self.request = request
        self.timeout = timeout
        self.client = httpx_client
        self.owns_client = httpx_client is None
        self.message = AssistantMessage(
            content=[],
            provider=self._provider,
            model=request.model,
        )
        self.usage: Usage | None = None
        self.stop_reason: str | None = None
        self.finish_reason: str | None = None
        self.response: Any = None
        self.lines: Any = None  # None until entered → StreamError on iteration
        self.pending: list = []
        self.finished = False
        self.cancelled = False
        self.emitted_start = False

    # --- live accessors ---

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.message.content if isinstance(b, TextBlock))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.message.tool_calls

    @property
    def provider_finish_reason(self) -> str | None:
        return self.stop_reason

    # --- wire reading (overridden per wire where the framing differs) ---

    def build_request(self) -> httpx.Request:
        payload = self._build_chat_completion_payload(self.request, stream=True)
        url = self._chat_completion_url(self.request, stream=True)
        return self.client.build_request(
            "POST",
            url,
            json=payload,
            headers=self._headers(),
            timeout=httpx.Timeout(None, connect=self.CONNECT_TIMEOUT),
        )

    def iter_chunks(self, response: httpx.Response) -> Any:
        return response.iter_lines()

    def aiter_chunks(self, response: httpx.Response) -> Any:
        return response.aiter_lines()

    def open_wire(self) -> Any:
        """Send the request and return the chunk source. The sync half of the
        one seam the mixins call — FauxStreamer overrides both halves to
        install its scripted source instead."""
        if self.client is None:
            self.client = httpx.Client()
        try:
            self.response = self.client.send(self.build_request(), stream=True)
        except httpx.HTTPError as exc:
            raise self._map_chat_completion_http_error(exc) from exc
        if self.expired():
            self.response.close()
            raise SDKTimeoutError(
                f"request did not open within timeout={self.timeout}s",
                provider=self._provider,
            )
        if self.response.status_code >= 400:
            try:
                try:
                    self.response.read()
                finally:
                    self.response.close()
                self.response.raise_for_status()
            except httpx.HTTPError as exc:
                raise self._map_chat_completion_http_error(exc) from exc
        return self.iter_chunks(self.response)

    async def open_response(self) -> httpx.Response:
        """The whole HTTP open as ONE awaitable — send, status check, and the
        error-body read — so the deadline timer and cancel(), which cancel
        the streamer-owned task this runs on, can interrupt any phase of it,
        a stalled error body included. `self.response` is assigned as soon as
        the headers exist, so teardown can always close it."""
        response = await self.client.send(self.build_request(), stream=True)
        self.response = response
        if response.status_code >= 400:
            try:
                await response.aread()
            finally:
                await response.aclose()
            response.raise_for_status()
        return response

    async def aopen_wire(self) -> Any:
        """Async half of the open seam: run `open_response()` on the
        streamer-owned task and convert its cancellation into the right
        outcome — the timer's into the open-timeout raise, `cancel()`'s into
        a StreamError (the stream never opened, so there is no cancelled
        finish to emit), external cancellation re-raises untouched."""
        if self.client is None:
            self.client = httpx.AsyncClient()
        self.last_task = asyncio.ensure_future(self.open_response())
        try:
            self.response = await self.last_task
        except asyncio.CancelledError:
            if self.expired:
                raise SDKTimeoutError(
                    f"request did not open within timeout={self.timeout}s",
                    provider=self._provider,
                ) from None
            if self.cancelled:
                raise StreamError(
                    "stream was cancelled before it opened",
                    provider=self._provider,
                ) from None
            # External cancellation: kill the in-flight open before
            # re-raising, or the task outlives the frame that owned it.
            await self.reap_read_task()
            raise
        except httpx.HTTPError as exc:
            raise self._map_chat_completion_http_error(exc) from exc
        return self.aiter_chunks(self.response)

    # --- chunk → raw events → public events ---

    def parse(self, chunk: str) -> list:
        """One wire chunk → a LIST of raw events: `[]` for non-data lines,
        several for framings that batch (Bedrock). The SSE default yields the
        decoded `data:` payload; `[DONE]` is wire noise here (the CC wire
        overrides — there it marks the end)."""
        if not chunk.startswith("data:"):
            return []
        data = chunk[5:].strip()
        if data == "[DONE]":
            return []
        try:
            return [json.loads(data)]
        except json.JSONDecodeError as e:
            raise StreamError(f"Stream produced non-JSON data: {data[:80]!r}") from e

    def handle(self, raw_event: dict) -> list:
        """One raw wire event → a LIST of public events (possibly empty)."""
        handler = self.HANDLERS_BY_TYPE.get(raw_event.get("type"))
        if handler is None:
            return []
        return getattr(self, handler)(raw_event)

    def handle_wire_end(self) -> list:
        """The wire closed before a terminal was emitted. Default `[]` means
        the provider hung up → the mixins emit the premature-end ErrorEvent.
        The CC and Bedrock wires override: they deliver usage AFTER their
        finish marker, so their FinishEvent can only be built here. Latched
        state only — no I/O, at most once."""
        return []

    # --- terminal builders (each marks the stream finished) ---

    def finish_event(self, *, cancelled: bool = False) -> FinishEvent:
        """The single FinishEvent builder: classifies the latched stop
        reason, stamps the message, and deep-copies it into the event."""
        self.finished = True
        if self.stop_reason is not None:
            canonical, error_message = self._classify_finish(self.stop_reason, self.message)
        else:
            canonical, error_message = (None, None)
        usage = self.usage or Usage()
        self.finish_reason = canonical
        self.message.finish_reason = canonical
        self.message.provider_finish_reason = self.stop_reason
        self.message.error_message = error_message
        self.message.cancelled = cancelled
        self.message.usage = usage
        finish = FinishEvent(
            message=self.message.model_copy(deep=True),
            finish_reason=canonical,
            provider_finish_reason=self.stop_reason,
            cancelled=cancelled,
            usage=usage,
            tool_calls=list(self.message.tool_calls),
        )
        finish._response_format = self.request.response_format
        return finish

    def error_event(self, error: Exception) -> ErrorEvent:
        self.finished = True
        self.message.error_message = str(error)
        return ErrorEvent(error=error, usage=self.usage)

    def timeout_event(self) -> ErrorEvent:
        return self.error_event(
            SDKTimeoutError(
                f"stream exceeded timeout={self.timeout}s",
                provider=self._provider,
            )
        )

    def emit(self, event: Any) -> Any:
        """The single exit gate both iterators return through: latch the
        terminal flag, and uphold the contract's "start first" — a stream
        whose first produced event is not a StartEvent (an immediate wire
        error, a timeout before any chunk, a pre-read cancel) still narrates
        start → terminal."""
        if isinstance(event, (FinishEvent, ErrorEvent)):
            self.finished = True
        if not self.emitted_start:
            self.emitted_start = True
            if not isinstance(event, StartEvent):
                self.pending.insert(0, event)
                return StartEvent()
        return event

    def __del__(self) -> None:
        response = getattr(self, "response", None)
        if response is None:
            return
        try:
            is_closed = response.is_closed
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
                response.close()


class SyncStreamerMixin:
    """Sync iteration. The deadline is a monotonic clock checked before every
    read — cooperative and best-effort by nature: httpx read timeouts are
    disabled on stream requests, so a read blocked on a live socket cannot be
    interrupted from this thread (`cancel()` from another thread closes the
    response and unblocks it)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.deadline: float | None = None

    def __enter__(self) -> Any:
        if self.lines is not None:
            return self
        if self.timeout is not None:
            self.deadline = time.monotonic() + self.timeout
        try:
            self.lines = self.open_wire()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.response is not None:
            with contextlib.suppress(Exception):
                self.response.close()
        if self.owns_client and self.client is not None:
            with contextlib.suppress(Exception):
                self.client.close()

    def expired(self) -> bool:
        return self.deadline is not None and time.monotonic() > self.deadline

    def cancel(self) -> None:
        """Interrupt the stream: the next pull returns a terminal
        `FinishEvent(cancelled=True)` with the partial message. Closing the
        response also unblocks a read blocked on the socket (the resulting
        httpx error routes to the same cancelled finish)."""
        self.cancelled = True
        if self.response is not None:
            with contextlib.suppress(Exception):
                if not self.response.is_closed:
                    self.response.close()

    def __iter__(self) -> Any:
        if self.finished:
            raise StreamError(f"{type(self).__name__} has already been consumed")
        return self

    def __next__(self) -> Any:
        if self.lines is None:
            raise StreamError("stream was never entered — use 'with completion_stream(...) as s'")
        if self.pending:
            return self.emit(self.pending.pop(0))
        if self.finished:
            raise StopIteration

        while True:
            if self.expired():
                return self.emit(self.timeout_event())
            if self.cancelled:
                return self.emit(self.finish_event(cancelled=True))
            try:
                chunk = next(self.lines)
                events = []
                for raw in self.parse(chunk):
                    events += self.handle(raw)
            except StopIteration:
                events = self.handle_wire_end()
                if not events:
                    return self.emit(self.error_event(StreamError(_PREMATURE_END)))
            except StreamError as exc:
                if self.cancelled:
                    return self.emit(self.finish_event(cancelled=True))
                return self.emit(self.error_event(exc))
            except (httpx.HTTPError, httpx.StreamError) as exc:
                if self.cancelled:
                    return self.emit(self.finish_event(cancelled=True))
                return self.emit(self.error_event(self._map_chat_completion_http_error(exc)))
            if not events:
                continue

            event, *rest = events
            self.pending += rest
            return self.emit(event)

    def collect(self) -> ChatCompletionResponse:
        from ..types.completion import ChatCompletionResponse

        with self:
            for event in self:
                if event.type == "finish":
                    response = ChatCompletionResponse(messages=[event.message])
                    response._response_format = self.request.response_format
                    return response
                if event.type == "error":
                    raise event.error
        raise StreamError("Stream ended without terminal event")


class AsyncStreamerMixin:
    """Async iteration. The deadline is a `call_later` timer whose callback
    sets `expired` and then cancels the in-flight read task — always a task
    this mixin owns, never the consumer's frame. `CancelledError` on the read
    await therefore has three sources, disambiguated by flags: the timer
    (`expired` → timeout event), `cancel()` (`cancelled` → cancelled finish),
    and external cancellation, which re-raises untouched."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.timer: Any = None
        self.last_task: asyncio.Future | None = None
        self.expired = False

    async def __aenter__(self) -> Any:
        if self.lines is not None:
            return self
        if self.timeout is not None:
            self.timer = asyncio.get_running_loop().call_later(self.timeout, self.expire)
        try:
            self.lines = await self.aopen_wire()
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self.timer is not None:
            self.timer.cancel()
        await self.aclose()
        if self.owns_client and self.client is not None:
            with contextlib.suppress(Exception):
                await self.client.aclose()

    def expire(self) -> None:
        # Set before cancelling: the read await wakes to CancelledError and
        # reads the flag to tell the timer from external cancellation.
        self.expired = True
        if self.last_task is not None and not self.last_task.done():
            self.last_task.cancel()

    async def reap_read_task(self) -> None:
        """Cancel and drain the in-flight read task. The CancelledError that
        await raises is normally the reaped task's own; if OUR task is being
        cancelled too, re-raise it — eating the caller's cancellation here
        would silently un-cancel them."""
        if self.last_task is None or self.last_task.done():
            return
        self.last_task.cancel()
        try:
            await self.last_task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
        except Exception:
            pass

    async def cancel(self) -> None:
        """Interrupt the stream: the next pull (or the read already in
        flight) produces a terminal `FinishEvent(cancelled=True)` with the
        partial message and any usage already seen."""
        self.cancelled = True
        await self.reap_read_task()
        if self.response is not None:
            with contextlib.suppress(Exception):
                if not self.response.is_closed:
                    await self.response.aclose()

    async def aclose(self) -> None:
        """Idempotent teardown: kill any in-flight read, close the source
        iterator and the response. The agent runner's pattern — external
        task-cancel of `__anext__`, then `aclose()`, then `__aexit__` — must
        leak nothing."""
        await self.reap_read_task()
        if self.lines is not None and hasattr(self.lines, "aclose"):
            with contextlib.suppress(Exception):
                await self.lines.aclose()
        if self.response is not None:
            with contextlib.suppress(Exception):
                if not self.response.is_closed:
                    await self.response.aclose()

    def __aiter__(self) -> Any:
        if self.finished:
            raise StreamError(f"{type(self).__name__} has already been consumed")
        return self

    async def __anext__(self) -> Any:
        if self.lines is None:
            raise StreamError("stream was never entered — use 'async with acompletion_stream(...) as s'")
        if self.pending:
            return self.emit(self.pending.pop(0))
        if self.finished:
            raise StopAsyncIteration

        while True:
            if self.expired:
                return self.emit(self.timeout_event())
            if self.cancelled:
                return self.emit(self.finish_event(cancelled=True))
            self.last_task = asyncio.ensure_future(anext(self.lines))
            try:
                chunk = await self.last_task
                events = []
                for raw in self.parse(chunk):
                    events += self.handle(raw)
            except StopAsyncIteration:
                events = self.handle_wire_end()
                if not events:
                    return self.emit(self.error_event(StreamError(_PREMATURE_END)))
            except asyncio.CancelledError:
                if self.expired:
                    return self.emit(self.timeout_event())
                if self.cancelled:
                    return self.emit(self.finish_event(cancelled=True))
                raise
            except StreamError as exc:
                if self.cancelled:
                    return self.emit(self.finish_event(cancelled=True))
                return self.emit(self.error_event(exc))
            except (httpx.HTTPError, httpx.StreamError) as exc:
                if self.cancelled:
                    return self.emit(self.finish_event(cancelled=True))
                return self.emit(self.error_event(self._map_chat_completion_http_error(exc)))
            if not events:
                continue

            event, *rest = events
            self.pending += rest
            return self.emit(event)

    async def collect(self) -> ChatCompletionResponse:
        from ..types.completion import ChatCompletionResponse

        async with self:
            async for event in self:
                if event.type == "finish":
                    response = ChatCompletionResponse(messages=[event.message])
                    response._response_format = self.request.response_format
                    return response
                if event.type == "error":
                    raise event.error
        raise StreamError("Stream ended without terminal event")
