"""Bedrock Converse streamer.

Converse streams `vnd.amazon.eventstream` — a binary framing, not SSE. Each
frame is:

    total_len(4) header_len(4) prelude_crc(4) headers payload message_crc(4)

and headers are `name_len(1) name type(1) [value_len(2) value]`; all Converse
event headers are strings. `:event-type` names the event; `:message-type`
distinguishes a normal event from an exception.

The chunk source is `iter_bytes()`, which hands back arbitrary chunks: a
frame can straddle a read and several frames can arrive in one, so `parse`
buffers bytes and returns every completed frame, each with its `:event-type`
stamped in as `"type"` (Converse payloads carry no type key) so the base
dispatch works unchanged. A partial trailing frame at wire end is dropped
silently.

Two more Converse quirks: `contentBlockStart` arrives only for tool-use
blocks — text and reasoning begin straight at the first delta, so their
starts are synthesized — and usage (`metadata`) arrives after `messageStop`,
so the finish is emitted from `handle_wire_end()`.
"""

from __future__ import annotations

import json
import struct
import zlib
from typing import ClassVar

from ...exceptions import StreamError
from ...types.completion import ChatCompletionRequest, Usage
from ...types.content import TextBlock, ThinkingBlock, ToolCall
from ...types.streaming import (
    StartEvent,
    StreamEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    UsageEvent,
)
from ..streamer import AsyncStreamerMixin, BaseStreamer, SyncStreamerMixin
from .transport import BedrockWireMixin

_HEADER_TYPE_STRING = 7


class BedrockStreamer(BedrockWireMixin, BaseStreamer):
    PROVIDER = "bedrock"

    HANDLERS_BY_TYPE: ClassVar[dict[str, str]] = {
        "messageStart": "handle_message_start",
        "contentBlockStart": "handle_content_block_start",
        "contentBlockDelta": "handle_content_block_delta",
        "contentBlockStop": "handle_content_block_stop",
        "messageStop": "handle_message_stop",
        "metadata": "handle_metadata",
    }

    def __init__(self, request: ChatCompletionRequest, **kwargs) -> None:
        super().__init__(request, **kwargs)
        self.buffer = bytearray()
        self.indices: dict[int, int] = {}  # wire block index -> message.content index

    # --- framing ---

    def iter_chunks(self, response):
        return response.iter_bytes()

    def aiter_chunks(self, response):
        return response.aiter_bytes()

    def parse(self, chunk: bytes) -> list:
        self.buffer.extend(chunk)
        events = []
        while (frame := self.take_frame()) is not None:
            headers, payload = frame
            if headers.get(":message-type") in ("exception", "error"):
                body = self.decode_payload(payload)
                msg = body.get("message", "Bedrock stream exception") if body else "Bedrock stream exception"
                kind = headers.get(":exception-type") or headers.get(":error-code") or "?"
                raise StreamError(f"Bedrock stream error [{kind}]: {msg}")
            event_type = headers.get(":event-type")
            if event_type is None:
                continue
            data = self.decode_payload(payload) or {}
            data["type"] = event_type
            events.append(data)
        return events

    def take_frame(self) -> tuple[dict[str, str], bytes] | None:
        """Pull one complete frame off the front of the buffer, or None."""
        buf = self.buffer
        if len(buf) < 12:
            return None
        total_len, header_len = struct.unpack(">II", buf[:8])
        if len(buf) < total_len:
            return None

        prelude_crc = struct.unpack(">I", buf[8:12])[0]
        if zlib.crc32(bytes(buf[:8])) != prelude_crc:
            raise StreamError("Bedrock event stream prelude CRC mismatch")
        message_crc = struct.unpack(">I", buf[total_len - 4 : total_len])[0]
        if zlib.crc32(bytes(buf[: total_len - 4])) != message_crc:
            raise StreamError("Bedrock event stream message CRC mismatch")

        headers = self.parse_frame_headers(bytes(buf[12 : 12 + header_len]))
        payload = bytes(buf[12 + header_len : total_len - 4])
        del buf[:total_len]
        return headers, payload

    def parse_frame_headers(self, raw: bytes) -> dict[str, str]:
        headers: dict[str, str] = {}
        p = 0
        while p < len(raw):
            name_len = raw[p]
            p += 1
            name = raw[p : p + name_len].decode()
            p += name_len
            value_type = raw[p]
            p += 1
            if value_type != _HEADER_TYPE_STRING:
                raise StreamError(f"Unexpected Bedrock event header type {value_type} for {name!r}")
            value_len = struct.unpack(">H", raw[p : p + 2])[0]
            p += 2
            headers[name] = raw[p : p + value_len].decode()
            p += value_len
        return headers

    def decode_payload(self, payload: bytes) -> dict | None:
        try:
            return json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    # --- handlers ---

    def handle_message_start(self, raw_event: dict) -> list[StreamEvent]:
        return [StartEvent()]

    def handle_content_block_start(self, raw_event: dict) -> list[StreamEvent]:
        # Converse sends this only for tool-use blocks; text and reasoning
        # synthesize their starts at the first delta.
        wire_index = raw_event.get("contentBlockIndex", 0)
        start = raw_event.get("start") or {}
        if "toolUse" not in start:
            return []
        tool = start["toolUse"]
        if tool.get("toolUseId") is None or tool.get("name") is None:
            raise StreamError(
                f"Bedrock stream opened a toolUse block without a toolUseId or name (index={wire_index})",
            )
        call = ToolCall(
            id=tool["toolUseId"],
            name=tool["name"],
            arguments={},
            partial_arguments="",
            complete=False,
        )
        self.message.content.append(call)
        index = self.indices[wire_index] = len(self.message.content) - 1
        return [ToolCallStartEvent(index=index, id=call.id, name=call.name)]

    def handle_content_block_delta(self, raw_event: dict) -> list[StreamEvent]:
        wire_index = raw_event.get("contentBlockIndex", 0)
        delta = raw_event.get("delta") or {}
        events: list[StreamEvent] = []
        if "text" in delta:
            index = self.indices.get(wire_index)
            if index is None:
                self.message.content.append(TextBlock(text=""))
                index = self.indices[wire_index] = len(self.message.content) - 1
                events.append(TextStartEvent(index=index))
            self.message.content[index].text += delta["text"]
            events.append(TextDeltaEvent(index=index, delta=delta["text"]))
        elif "toolUse" in delta:
            index = self.indices.get(wire_index)
            if index is None:
                return []
            piece = delta["toolUse"].get("input", "")
            self.message.content[index].partial_arguments += piece
            events.append(ToolCallDeltaEvent(index=index, arguments_delta=piece))
        elif "reasoningContent" in delta:
            rc = delta["reasoningContent"]
            if "text" in rc:
                index = self.indices.get(wire_index)
                if index is None:
                    self.message.content.append(ThinkingBlock(text=""))
                    index = self.indices[wire_index] = len(self.message.content) - 1
                    events.append(ThinkingStartEvent(index=index))
                self.message.content[index].text += rc["text"]
                events.append(ThinkingDeltaEvent(index=index, delta=rc["text"]))
            elif "signature" in rc:
                index = self.indices.get(wire_index)
                if index is None:
                    self.message.content.append(ThinkingBlock(text=""))
                    index = self.indices[wire_index] = len(self.message.content) - 1
                    events.append(ThinkingStartEvent(index=index))
                self.message.content[index].signature = rc["signature"]
                events.append(ThinkingDeltaEvent(index=index, delta=""))
            elif "redactedContent" in rc and wire_index not in self.indices:
                # Arrives whole: encrypted payload on one delta, no text.
                self.message.content.append(ThinkingBlock(text="", signature=rc["redactedContent"], redacted=True))
                index = self.indices[wire_index] = len(self.message.content) - 1
                events.append(ThinkingStartEvent(index=index))
        return events

    def handle_content_block_stop(self, raw_event: dict) -> list[StreamEvent]:
        index = self.indices.pop(raw_event.get("contentBlockIndex", 0), None)
        if index is None:
            return []
        block = self.message.content[index]
        if isinstance(block, ToolCall):
            try:
                block.arguments = json.loads(block.partial_arguments) if block.partial_arguments else {}
            except json.JSONDecodeError as e:
                raise StreamError(f"Tool call {index} ({block.name!r}) returned malformed JSON") from e
            block.partial_arguments = ""
            block.complete = True
            return [ToolCallEndEvent(index=index, tool_call=block)]
        if isinstance(block, ThinkingBlock):
            return [ThinkingEndEvent(index=index, content=block.text)]
        return [TextEndEvent(index=index, content=block.text)]

    def handle_message_stop(self, raw_event: dict) -> list[StreamEvent]:
        self.stop_reason = raw_event.get("stopReason")
        return []

    def handle_metadata(self, raw_event: dict) -> list[StreamEvent]:
        usage = raw_event.get("usage") or {}
        self.usage = Usage(
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            total_tokens=usage.get("totalTokens", 0),
            cached_input_tokens=usage.get("cacheReadInputTokens"),
            cache_write_tokens=usage.get("cacheWriteInputTokens"),
        )
        return [UsageEvent(usage=self.usage)]

    def handle_wire_end(self) -> list[StreamEvent]:
        if self.stop_reason is None:
            return []
        return [self.finish_event()]


class SyncBedrockStreamer(SyncStreamerMixin, BedrockStreamer):
    pass


class AsyncBedrockStreamer(AsyncStreamerMixin, BedrockStreamer):
    pass
