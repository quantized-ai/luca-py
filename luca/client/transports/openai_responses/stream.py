"""OpenAI Responses streaming: typed SSE events → RawStreamEvent vocabulary.

Unlike chat completions, every SSE frame here is a named event over the whole
response object rather than a `delta` over one choice. That makes the parser a
dispatch on `event["type"]`, and it makes UNKNOWN TYPES SAFE TO IGNORE — the
hosted-tool events (`response.web_search_call.*`, …) and any event OpenAI adds
later carry no canonical block, so skipping them is correct rather than lossy.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ...exceptions import StreamError
from ...types.completion import Usage
from ...types.streaming import (
    AsyncChatCompletionStream,
    ChatCompletionStream,
    RawBlockStart,
    RawBlockStop,
    RawFinish,
    RawRefusalDelta,
    RawStreamEvent,
    RawTextDelta,
    RawThinkingDelta,
    RawToolArgumentsDelta,
    RawUsage,
)

# Synthetic content_index slots, so reasoning and function-call items share the
# one allocator with real content parts without colliding with content_index 0.
# A provider call takes the function-call slot: one output item is one call
# either way, and the two can never share an output_index.
_REASONING_SLOT = -1
_FUNCTION_CALL_SLOT = -2

# Between two summary parts of the same reasoning item — they are separate
# paragraphs of one train of thought, and read as one block.
_SUMMARY_PART_SEPARATOR = "\n\n"


class _ResponsesParserState:
    """Mutable parser state shared between the sync and async parsers.

    The accumulator requires content indices to be dense and ordered
    (`RawBlockStart.index == len(message.content)`), which holds because the
    Responses API streams its output items in order: an index is allocated the
    first time an item or content part is seen and never revisited."""

    def __init__(self, provider_calls: dict[str, tuple[str, str]] | None = None) -> None:
        self.next_content_index: int = 0
        # (output_index, content_index | slot) -> our content index
        self.content_index: dict[tuple[int, int], int] = {}
        # Blocks opened and not yet closed, so the terminal can close them.
        self.open_indices: set[int] = set()
        self.finished: bool = False
        # item type -> (tool name, provider type), for the provider tools this
        # request offered. A provider call item carries no tool name, so the
        # only thing that resolves one is what was sent.
        self.provider_calls: dict[str, tuple[str, str]] = provider_calls or {}

    def allocate(self, key: tuple[int, int]) -> int:
        index = self.next_content_index
        self.next_content_index += 1
        self.content_index[key] = index
        self.open_indices.add(index)
        return index

    def resolve(self, key: tuple[int, int]) -> int | None:
        return self.content_index.get(key)

    def close(self, index: int) -> bool:
        """True if the block was open, so a duplicate done-event cannot emit a
        second RawBlockStop."""
        if index not in self.open_indices:
            return False
        self.open_indices.remove(index)
        return True


def _parse_sse_line(line: str) -> str | None:
    """Return the JSON payload from a `data: ...` SSE line, or None.

    The Responses stream also sends `event:` lines naming the same type that is
    inside the payload; the payload is authoritative, so the header is skipped."""
    if not line:
        return None
    if not line.startswith("data:"):
        return None
    return line[5:].strip()


def _usage_from(response: dict) -> Usage | None:
    usage = response.get("usage")
    if usage is None:
        return None
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    return Usage(
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        cached_input_tokens=input_details.get("cached_tokens"),
        reasoning_tokens=output_details.get("reasoning_tokens"),
    )


_TERMINAL_EVENT_STATUS = {
    "response.completed": "completed",
    "response.incomplete": "incomplete",
    "response.failed": "failed",
}


def _terminal_reason(event_type: str, response: dict) -> str:
    """Same composition as the non-streaming transport, so both paths report
    the identical `provider_finish_reason`. A terminal missing `status` falls
    back to its EVENT NAME, never to "completed", so a failed response is
    never reported as a successful stop."""
    status = response.get("status") or _TERMINAL_EVENT_STATUS.get(event_type, "completed")
    if status == "incomplete":
        reason = (response.get("incomplete_details") or {}).get("reason")
        return f"incomplete:{reason}" if reason else "incomplete"
    return status


def _process_event(state: _ResponsesParserState, event: dict) -> Iterator[RawStreamEvent]:
    """Translate one Responses SSE event into RawStreamEvents."""
    event_type = event.get("type")

    if event_type == "response.output_item.added":
        yield from _item_added(state, event)

    elif event_type == "response.output_item.done":
        yield from _item_done(state, event)

    elif event_type == "response.content_part.added":
        part = event.get("part") or {}
        key = (event.get("output_index", 0), event.get("content_index", 0))
        if part.get("type") == "output_text":
            yield RawBlockStart(index=state.allocate(key), block_type="text")
        elif part.get("type") == "refusal":
            yield RawBlockStart(index=state.allocate(key), block_type="refusal")

    elif event_type == "response.content_part.done":
        index = state.resolve((event.get("output_index", 0), event.get("content_index", 0)))
        if index is not None and state.close(index):
            yield RawBlockStop(index=index)

    elif event_type == "response.output_text.delta":
        index = state.resolve((event.get("output_index", 0), event.get("content_index", 0)))
        if index is not None and event.get("delta"):
            yield RawTextDelta(index=index, text=event["delta"])

    elif event_type == "response.refusal.delta":
        index = state.resolve((event.get("output_index", 0), event.get("content_index", 0)))
        if index is not None and event.get("delta"):
            yield RawRefusalDelta(index=index, text=event["delta"])

    elif event_type == "response.reasoning_summary_text.delta":
        index = state.resolve((event.get("output_index", 0), _REASONING_SLOT))
        if index is not None and event.get("delta"):
            yield RawThinkingDelta(index=index, text=event["delta"])

    elif event_type == "response.reasoning_summary_part.added":
        # The first part opens the block that output_item.added already started;
        # every later one continues the same block after a paragraph break.
        index = state.resolve((event.get("output_index", 0), _REASONING_SLOT))
        if index is not None and event.get("summary_index", 0) > 0:
            yield RawThinkingDelta(index=index, text=_SUMMARY_PART_SEPARATOR)

    elif event_type == "response.function_call_arguments.delta":
        index = state.resolve((event.get("output_index", 0), _FUNCTION_CALL_SLOT))
        if index is not None and event.get("delta"):
            yield RawToolArgumentsDelta(index=index, arguments_delta=event["delta"])

    elif event_type in _TERMINAL_EVENT_STATUS:
        yield from _terminal(state, event, event_type)

    elif event_type == "error":
        raise StreamError(
            f"OpenAI Responses stream returned an error: {event.get('message') or event}",
        )


def _item_added(state: _ResponsesParserState, event: dict) -> Iterator[RawStreamEvent]:
    item = event.get("item") or {}
    output_index = event.get("output_index", 0)
    item_type = item.get("type")

    if item_type == "reasoning":
        index = state.allocate((output_index, _REASONING_SLOT))
        yield RawBlockStart(index=index, block_type="thinking", item_id=item.get("id"))
    elif item_type == "function_call":
        call_id = item.get("call_id")
        name = item.get("name")
        if call_id is None or name is None:
            raise StreamError(
                f"OpenAI Responses stream opened a function_call item without a call_id or name (output_index={output_index})",
            )
        index = state.allocate((output_index, _FUNCTION_CALL_SLOT))
        yield RawBlockStart(
            index=index,
            block_type="tool_call",
            tool_id=call_id,
            tool_name=name,
        )
    elif item_type in state.provider_calls:
        # `apply_patch_call` / `shell_call`. The item opens with a call_id and
        # nothing else; its arguments only exist on output_item.done, so the
        # block opens here and fills there.
        call_id = item.get("call_id")
        if call_id is None:
            raise StreamError(
                f"OpenAI Responses stream opened a {item_type} item without a call_id (output_index={output_index})",
            )
        name, provider_type = state.provider_calls[item_type]
        index = state.allocate((output_index, _FUNCTION_CALL_SLOT))
        yield RawBlockStart(
            index=index,
            block_type="tool_call",
            tool_id=call_id,
            tool_name=name,
            provider_type=provider_type,
        )
    # A `message` item's blocks open on content_part.added, one per part.


def _item_done(state: _ResponsesParserState, event: dict) -> Iterator[RawStreamEvent]:
    item = event.get("item") or {}
    output_index = event.get("output_index", 0)
    item_type = item.get("type")

    if item_type == "reasoning":
        index = state.resolve((output_index, _REASONING_SLOT))
        if index is None:
            return
        # The encrypted payload only exists once the item is complete, and it
        # is the half that makes the block replayable — carried on a zero-text
        # delta, which the accumulator writes straight onto the block.
        encrypted = item.get("encrypted_content")
        if encrypted is not None:
            yield RawThinkingDelta(index=index, text="", signature=encrypted)
        if state.close(index):
            yield RawBlockStop(index=index)
    elif item_type == "function_call":
        index = state.resolve((output_index, _FUNCTION_CALL_SLOT))
        if index is not None and state.close(index):
            yield RawBlockStop(index=index)
    elif item_type in state.provider_calls:
        index = state.resolve((output_index, _FUNCTION_CALL_SLOT))
        if index is None:
            return
        # There is no arguments-delta event for these: the payload arrives
        # whole, once, on the done event. One delta carries it so the
        # accumulator parses it exactly as it parses a function call's.
        payload = item.get("operation") if item_type == "apply_patch_call" else item.get("action")
        yield RawToolArgumentsDelta(index=index, arguments_delta=json.dumps(payload or {}))
        if state.close(index):
            yield RawBlockStop(index=index)
    # A `message` item's parts already closed on content_part.done.


def _terminal(
    state: _ResponsesParserState,
    event: dict,
    event_type: str,
) -> Iterator[RawStreamEvent]:
    if state.finished:
        return
    state.finished = True

    # A truncated response can go terminal mid-block. Closing through
    # RawBlockStop is load-bearing: the accumulator completes a tool call in
    # that branch, so an unclosed one would reach the caller as
    # `arguments={}, complete=False` beside a `tool_use` finish and be
    # dispatched with its arguments erased.
    for index in sorted(state.open_indices):
        yield RawBlockStop(index=index)
    state.open_indices.clear()

    response = event.get("response") or {}
    usage = _usage_from(response)
    if usage is not None:
        yield RawUsage(usage=usage)
    yield RawFinish(reason=_terminal_reason(event_type, response))


class OpenAIResponsesStream(ChatCompletionStream):
    def _open_http(self) -> Any:
        payload = self._transport._build_chat_completion_payload(self._request, stream=True)
        return self._transport._client.stream(
            "POST",
            self._transport._chat_completion_url(self._request, stream=True),
            json=payload,
            headers=self._transport._headers(),
        )

    def parse_chunks(self) -> Iterator[RawStreamEvent]:
        state = _ResponsesParserState(self._transport.native_call_items(self._request))
        for line in self._http_response.iter_lines():
            data = _parse_sse_line(line)
            if data is None:
                continue
            if data == "[DONE]":
                return
            try:
                event = json.loads(data)
            except json.JSONDecodeError as e:
                raise StreamError(
                    f"OpenAI Responses stream produced non-JSON data: {data[:80]!r}",
                ) from e
            yield from _process_event(state, event)


class OpenAIResponsesAsyncStream(AsyncChatCompletionStream):
    async def _open_http(self) -> Any:
        payload = self._transport._build_chat_completion_payload(self._request, stream=True)
        aclient = self._transport._ensure_aclient()
        return aclient.stream(
            "POST",
            self._transport._chat_completion_url(self._request, stream=True),
            json=payload,
            headers=self._transport._headers(),
        )

    async def parse_chunks(self) -> AsyncIterator[RawStreamEvent]:
        state = _ResponsesParserState(self._transport.native_call_items(self._request))
        async for line in self._http_response.aiter_lines():
            data = _parse_sse_line(line)
            if data is None:
                continue
            if data == "[DONE]":
                return
            try:
                event = json.loads(data)
            except json.JSONDecodeError as e:
                raise StreamError(
                    f"OpenAI Responses stream produced non-JSON data: {data[:80]!r}",
                ) from e
            for raw in _process_event(state, event):
                yield raw
