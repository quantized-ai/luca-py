"""Anthropic streaming: typed SSE event stream → RawStreamEvent vocabulary.

The Anthropic wire format is far cleaner than OpenAI's — every chunk is a
typed event (`message_start`, `content_block_start`, `content_block_delta`,
`content_block_stop`, `message_delta`, `message_stop`) with explicit block
boundaries.
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
    RawStreamEvent,
    RawTextDelta,
    RawThinkingDelta,
    RawToolArgumentsDelta,
    RawUsage,
)


class _AnthropicParserState:
    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.stop_reason: str | None = None


def _parse_event_envelope(lines: list[str]) -> tuple[str | None, str | None]:
    """Anthropic uses `event: <type>\\ndata: <json>` per message."""
    event_type: str | None = None
    data: str | None = None
    for line in lines:
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()
    return event_type, data


def _process_event(
    state: _AnthropicParserState, event_type: str, data: dict,
) -> Iterator[RawStreamEvent]:
    if event_type == "message_start":
        msg = data.get("message", {})
        usage = msg.get("usage") or {}
        state.input_tokens = usage.get("input_tokens", 0)
        state.output_tokens = usage.get("output_tokens", 0)

    elif event_type == "content_block_start":
        idx = data.get("index", 0)
        block = data.get("content_block", {})
        block_type = block.get("type")
        if block_type == "text":
            yield RawBlockStart(index=idx, block_type="text")
        elif block_type == "thinking":
            yield RawBlockStart(index=idx, block_type="thinking")
        elif block_type == "tool_use":
            yield RawBlockStart(
                index=idx, block_type="tool_call",
                tool_id=block.get("id"), tool_name=block.get("name"),
            )
        elif block_type == "redacted_thinking":
            yield RawBlockStart(index=idx, block_type="thinking")

    elif event_type == "content_block_delta":
        idx = data.get("index", 0)
        delta = data.get("delta", {})
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            yield RawTextDelta(index=idx, text=delta.get("text", ""))
        elif delta_type == "thinking_delta":
            yield RawThinkingDelta(index=idx, text=delta.get("thinking", ""))
        elif delta_type == "signature_delta":
            yield RawThinkingDelta(
                index=idx, text="", signature=delta.get("signature"),
            )
        elif delta_type == "input_json_delta":
            yield RawToolArgumentsDelta(
                index=idx, arguments_delta=delta.get("partial_json", ""),
            )

    elif event_type == "content_block_stop":
        idx = data.get("index", 0)
        yield RawBlockStop(index=idx)

    elif event_type == "message_delta":
        delta = data.get("delta", {})
        if (reason := delta.get("stop_reason")) is not None:
            state.stop_reason = reason
        usage = data.get("usage") or {}
        if "output_tokens" in usage:
            state.output_tokens = usage["output_tokens"]

    elif event_type == "message_stop":
        if state.stop_reason is not None:
            yield RawFinish(reason=state.stop_reason)
        yield RawUsage(usage=Usage(
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            total_tokens=state.input_tokens + state.output_tokens,
        ))

    elif event_type == "error":
        err = data.get("error") or {}
        msg = err.get("message", "Anthropic stream error")
        raise StreamError(f"Anthropic stream error: {msg}")


def _iter_event_blocks(line_iter: Iterator[str]) -> Iterator[list[str]]:
    block: list[str] = []
    for line in line_iter:
        if line == "":
            if block:
                yield block
                block = []
        else:
            block.append(line)
    if block:
        yield block


class AnthropicChatCompletionStream(ChatCompletionStream):
    def _open_http(self) -> Any:
        payload = self._transport._build_chat_completion_payload(self._request, stream=True)
        return self._transport._client.stream(
            "POST",
            self._transport._chat_completion_url(),
            json=payload,
            headers=self._transport._headers(),
        )

    def parse_chunks(self) -> Iterator[RawStreamEvent]:
        state = _AnthropicParserState()
        for block_lines in _iter_event_blocks(self._http_response.iter_lines()):
            event_type, data_str = _parse_event_envelope(block_lines)
            if event_type is None or data_str is None:
                continue
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError as e:
                raise StreamError(
                    f"Anthropic stream produced non-JSON data: {data_str[:80]!r}",
                ) from e
            yield from _process_event(state, event_type, data)


async def _aiter_event_blocks(line_iter: AsyncIterator[str]) -> AsyncIterator[list[str]]:
    block: list[str] = []
    async for line in line_iter:
        if line == "":
            if block:
                yield block
                block = []
        else:
            block.append(line)
    if block:
        yield block


class AnthropicAsyncChatCompletionStream(AsyncChatCompletionStream):
    async def _open_http(self) -> Any:
        payload = self._transport._build_chat_completion_payload(self._request, stream=True)
        aclient = self._transport._ensure_aclient()
        return aclient.stream(
            "POST",
            self._transport._chat_completion_url(),
            json=payload,
            headers=self._transport._headers(),
        )

    async def parse_chunks(self) -> AsyncIterator[RawStreamEvent]:
        state = _AnthropicParserState()
        async for block_lines in _aiter_event_blocks(self._http_response.aiter_lines()):
            event_type, data_str = _parse_event_envelope(block_lines)
            if event_type is None or data_str is None:
                continue
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError as e:
                raise StreamError(
                    f"Anthropic stream produced non-JSON data: {data_str[:80]!r}",
                ) from e
            for ev in _process_event(state, event_type, data):
                yield ev
