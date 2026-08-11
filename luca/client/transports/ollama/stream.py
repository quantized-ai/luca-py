"""Ollama streaming: newline-delimited JSON → RawStreamEvent.

One JSON object per line, each carrying the whole delta so far's increment:

    {"message":{"role":"assistant","content":"Hi"},"done":false}
    {"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop",
     "prompt_eval_count":31,"eval_count":2}

No SSE framing, no `[DONE]` sentinel — `done: true` terminates and the same
frame carries the token counts.
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


class _OllamaParserState:
    def __init__(self) -> None:
        self.next_content_index: int = 0
        self.text_content_index: int | None = None
        self.thinking_content_index: int | None = None
        self.finished: bool = False


def _open(state: _OllamaParserState, kind: str) -> Iterator[RawStreamEvent]:
    index = state.next_content_index
    state.next_content_index += 1
    if kind == "text":
        state.text_content_index = index
    else:
        state.thinking_content_index = index
    yield RawBlockStart(index=index, block_type=kind)


def _close_open_blocks(state: _OllamaParserState) -> Iterator[RawStreamEvent]:
    for index in (state.thinking_content_index, state.text_content_index):
        if index is not None:
            yield RawBlockStop(index=index)
    state.thinking_content_index = None
    state.text_content_index = None


def _process_frame(state: _OllamaParserState, frame: dict) -> Iterator[RawStreamEvent]:
    message = frame.get("message") or {}

    thinking = message.get("thinking")
    if thinking:
        if state.thinking_content_index is None:
            yield from _open(state, "thinking")
        yield RawThinkingDelta(index=state.thinking_content_index, text=thinking)

    text = message.get("content")
    if text:
        if state.text_content_index is None:
            yield from _open(state, "text")
        yield RawTextDelta(index=state.text_content_index, text=text)

    # Tool calls arrive whole rather than as argument fragments, so each one is
    # a complete block: start, the full arguments, stop.
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        index = state.next_content_index
        state.next_content_index += 1
        yield RawBlockStart(
            index=index,
            block_type="tool_call",
            tool_id=call.get("id") or function.get("name", ""),
            tool_name=function.get("name", ""),
        )
        yield RawToolArgumentsDelta(
            index=index,
            arguments_delta=json.dumps(function.get("arguments") or {}),
        )
        yield RawBlockStop(index=index)

    if frame.get("done"):
        state.finished = True
        yield from _close_open_blocks(state)
        input_tokens = frame.get("prompt_eval_count") or 0
        output_tokens = frame.get("eval_count") or 0
        yield RawUsage(
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
        )
        yield RawFinish(reason=frame.get("done_reason") or "stop")


def _decode(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise StreamError(f"Ollama stream produced non-JSON data: {line[:80]!r}") from exc


class OllamaChatCompletionStream(ChatCompletionStream):
    def _open_http(self) -> Any:
        payload = self._transport._build_chat_completion_payload(self._request, stream=True)
        return self._transport._client.stream(
            "POST",
            self._transport._chat_completion_url(self._request, stream=True),
            json=payload,
            headers=self._transport._headers(),
        )

    def parse_chunks(self) -> Iterator[RawStreamEvent]:
        state = _OllamaParserState()
        for line in self._http_response.iter_lines():
            frame = _decode(line)
            if frame is not None:
                yield from _process_frame(state, frame)


class OllamaAsyncChatCompletionStream(AsyncChatCompletionStream):
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
        state = _OllamaParserState()
        async for line in self._http_response.aiter_lines():
            frame = _decode(line)
            if frame is not None:
                for event in _process_frame(state, frame):
                    yield event
