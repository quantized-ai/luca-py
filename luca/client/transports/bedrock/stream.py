"""Bedrock streaming: `vnd.amazon.eventstream` frames → RawStreamEvent vocabulary.

Converse streams a binary framing format, not SSE. Each frame is:

    total_len(4) header_len(4) prelude_crc(4) headers payload message_crc(4)

and headers are `name_len(1) name type(1) [value_len(2) value]`. All Converse
event headers are strings. `:event-type` names the event; `:message-type`
distinguishes a normal event from an exception.

Two Converse quirks the decoder absorbs:

  - `iter_bytes()` hands back arbitrary chunks, so a frame can straddle a read
    and several frames can arrive in one. The decoder buffers and yields whole
    frames only.
  - Converse sends `contentBlockStart` only for tool-use blocks; text and
    reasoning blocks begin straight at the first delta. The accumulator
    downstream requires a start before any delta, so one is synthesised on the
    first delta for a not-yet-started index.
"""

from __future__ import annotations

import json
import struct
import zlib
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
    RawStreamEvent,
    RawTextDelta,
    RawThinkingDelta,
    RawToolArgumentsDelta,
    RawUsage,
)

_HEADER_TYPE_STRING = 7


class _BedrockParserState:
    def __init__(self) -> None:
        self.started: set[int] = set()
        self.stop_reason: str | None = None


def _parse_headers(raw: bytes) -> dict[str, str]:
    headers: dict[str, str] = {}
    p = 0
    while p < len(raw):
        name_len = raw[p]
        p += 1
        name = raw[p:p + name_len].decode()
        p += name_len
        value_type = raw[p]
        p += 1
        if value_type != _HEADER_TYPE_STRING:
            raise StreamError(
                f"Unexpected Bedrock event header type {value_type} for {name!r}",
            )
        value_len = struct.unpack(">H", raw[p:p + 2])[0]
        p += 2
        headers[name] = raw[p:p + value_len].decode()
        p += value_len
    return headers


def _take_frame(buf: bytearray) -> tuple[dict[str, str], bytes] | None:
    """Pull one complete frame off the front of `buf`, or None if incomplete."""
    if len(buf) < 12:
        return None
    total_len, header_len = struct.unpack(">II", buf[:8])
    if len(buf) < total_len:
        return None

    prelude_crc = struct.unpack(">I", buf[8:12])[0]
    if zlib.crc32(bytes(buf[:8])) != prelude_crc:
        raise StreamError("Bedrock event stream prelude CRC mismatch")
    message_crc = struct.unpack(">I", buf[total_len - 4:total_len])[0]
    if zlib.crc32(bytes(buf[:total_len - 4])) != message_crc:
        raise StreamError("Bedrock event stream message CRC mismatch")

    headers = _parse_headers(bytes(buf[12:12 + header_len]))
    payload = bytes(buf[12 + header_len:total_len - 4])
    del buf[:total_len]
    return headers, payload


def _dispatch(state: _BedrockParserState, headers: dict, payload: bytes) -> Iterator[RawStreamEvent]:
    if headers.get(":message-type") in ("exception", "error"):
        body = _safe_json(payload)
        msg = body.get("message", "Bedrock stream exception") if body else "Bedrock stream exception"
        kind = headers.get(":exception-type") or headers.get(":error-code") or "?"
        raise StreamError(f"Bedrock stream error [{kind}]: {msg}")
    event_type = headers.get(":event-type")
    if event_type is None:
        return
    data = _safe_json(payload) or {}
    yield from _process_event(state, event_type, data)


def _ensure_start(
    state: _BedrockParserState, index: int, block_type: str,
) -> Iterator[RawStreamEvent]:
    if index not in state.started:
        state.started.add(index)
        yield RawBlockStart(index=index, block_type=block_type)


def _process_event(
    state: _BedrockParserState, event_type: str, data: dict,
) -> Iterator[RawStreamEvent]:
    if event_type == "contentBlockStart":
        idx = data.get("contentBlockIndex", 0)
        start = data.get("start") or {}
        if "toolUse" in start:
            tool = start["toolUse"]
            state.started.add(idx)
            yield RawBlockStart(
                index=idx, block_type="tool_call",
                tool_id=tool.get("toolUseId"), tool_name=tool.get("name"),
            )

    elif event_type == "contentBlockDelta":
        idx = data.get("contentBlockIndex", 0)
        delta = data.get("delta") or {}
        if "text" in delta:
            yield from _ensure_start(state, idx, "text")
            yield RawTextDelta(index=idx, text=delta["text"])
        elif "toolUse" in delta:
            yield RawToolArgumentsDelta(
                index=idx, arguments_delta=delta["toolUse"].get("input", ""),
            )
        elif "reasoningContent" in delta:
            rc = delta["reasoningContent"]
            if "text" in rc:
                yield from _ensure_start(state, idx, "thinking")
                yield RawThinkingDelta(index=idx, text=rc["text"])
            elif "signature" in rc:
                yield from _ensure_start(state, idx, "thinking")
                yield RawThinkingDelta(index=idx, text="", signature=rc["signature"])
            elif "redactedContent" in rc and idx not in state.started:
                state.started.add(idx)
                yield RawBlockStart(
                    index=idx, block_type="thinking",
                    signature=rc["redactedContent"], redacted=True,
                )

    elif event_type == "contentBlockStop":
        idx = data.get("contentBlockIndex", 0)
        if idx in state.started:
            state.started.discard(idx)
            yield RawBlockStop(index=idx)

    elif event_type == "messageStop":
        state.stop_reason = data.get("stopReason")
        if state.stop_reason is not None:
            yield RawFinish(reason=state.stop_reason)

    elif event_type == "metadata":
        usage = data.get("usage") or {}
        yield RawUsage(usage=Usage(
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            total_tokens=usage.get("totalTokens", 0),
            cached_input_tokens=usage.get("cacheReadInputTokens"),
            cache_write_tokens=usage.get("cacheWriteInputTokens"),
        ))


def _safe_json(payload: bytes) -> dict | None:
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


class BedrockChatCompletionStream(ChatCompletionStream):
    def _open_http(self) -> Any:
        payload = self._transport._build_chat_completion_payload(self._request, stream=True)
        return self._transport._client.stream(
            "POST",
            self._transport._chat_completion_url(self._request, stream=True),
            json=payload,
            headers=self._transport._headers(),
        )

    def parse_chunks(self) -> Iterator[RawStreamEvent]:
        state = _BedrockParserState()
        buf = bytearray()
        for chunk in self._http_response.iter_bytes():
            buf.extend(chunk)
            while (frame := _take_frame(buf)) is not None:
                yield from _dispatch(state, frame[0], frame[1])


class BedrockAsyncChatCompletionStream(AsyncChatCompletionStream):
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
        state = _BedrockParserState()
        buf = bytearray()
        async for chunk in self._http_response.aiter_bytes():
            buf.extend(chunk)
            while (frame := _take_frame(buf)) is not None:
                for ev in _dispatch(state, frame[0], frame[1]):
                    yield ev
