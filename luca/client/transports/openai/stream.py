"""OpenAI streaming: SSE chunks → RawStreamEvent vocabulary."""

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


class _OpenAIParserState:
    """Mutable parser state shared between sync and async stream parsers."""

    def __init__(self) -> None:
        self.next_content_index: int = 0
        self.reasoning_content_index: int | None = None
        self.text_content_index: int | None = None
        # openai tool_idx -> our content_idx (block IS open)
        self.tool_content_index: dict[int, int] = {}
        # openai tool_idx -> {"id", "name", "buffered_args"}
        self.tool_pending: dict[int, dict] = {}
        self.finished: bool = False


def _parse_sse_line(line: str) -> str | None:
    """Return the JSON payload from a `data: ...` SSE line, or None."""
    if not line:
        return None
    if not line.startswith("data:"):
        return None
    return line[5:].strip()


def _process_chunk(state: _OpenAIParserState, chunk: dict) -> Iterator[RawStreamEvent]:
    """Translate one OpenAI SSE chunk into RawStreamEvents."""

    choices = chunk.get("choices") or []
    choice = choices[0] if choices else {}
    delta = choice.get("delta", {}) or {}
    finish = choice.get("finish_reason")
    usage = chunk.get("usage")

    # --- reasoning ("thinking") ---
    # Present only when the host carries it (OpenRouter: `reasoning`;
    # DeepSeek-style: `reasoning_content`). Reasoning streams first, so it
    # claims the earliest content index. Inert for pure OpenAI (no field).
    reasoning_piece = delta.get("reasoning") or delta.get("reasoning_content")
    if reasoning_piece:
        if state.reasoning_content_index is None:
            state.reasoning_content_index = state.next_content_index
            state.next_content_index += 1
            yield RawBlockStart(index=state.reasoning_content_index, block_type="thinking")
        yield RawThinkingDelta(index=state.reasoning_content_index, text=reasoning_piece)

    # --- text ---
    content_piece = delta.get("content")
    if content_piece:
        if state.text_content_index is None:
            state.text_content_index = state.next_content_index
            state.next_content_index += 1
            yield RawBlockStart(index=state.text_content_index, block_type="text")
        yield RawTextDelta(index=state.text_content_index, text=content_piece)

    # --- tool calls ---
    for tc_delta in delta.get("tool_calls") or []:
        openai_idx = tc_delta.get("index")
        if openai_idx is None:
            continue
        pending = state.tool_pending.setdefault(
            openai_idx,
            {"id": None, "name": None, "buffered_args": []},
        )

        if (id_piece := tc_delta.get("id")) is not None:
            pending["id"] = id_piece
        function = tc_delta.get("function") or {}
        if (name_piece := function.get("name")) is not None:
            pending["name"] = name_piece
        args_piece = function.get("arguments", "")

        if openai_idx not in state.tool_content_index:
            if pending["id"] is not None and pending["name"] is not None:
                content_idx = state.next_content_index
                state.next_content_index += 1
                state.tool_content_index[openai_idx] = content_idx
                yield RawBlockStart(
                    index=content_idx,
                    block_type="tool_call",
                    tool_id=pending["id"],
                    tool_name=pending["name"],
                )
                for buffered in pending["buffered_args"]:
                    yield RawToolArgumentsDelta(
                        index=content_idx,
                        arguments_delta=buffered,
                    )
                pending["buffered_args"].clear()
                if args_piece:
                    yield RawToolArgumentsDelta(
                        index=content_idx,
                        arguments_delta=args_piece,
                    )
            elif args_piece:
                pending["buffered_args"].append(args_piece)
        else:
            if args_piece:
                yield RawToolArgumentsDelta(
                    index=state.tool_content_index[openai_idx],
                    arguments_delta=args_piece,
                )

    # --- finish: close blocks + raw finish ---
    if finish is not None and not state.finished:
        state.finished = True
        unresolved = [k for k, p in state.tool_pending.items()
                      if k not in state.tool_content_index]
        if unresolved:
            raise StreamError(
                f"OpenAI stream finished with tool calls missing id or name "
                f"(openai tool_indices: {unresolved})"
            )
        if state.reasoning_content_index is not None:
            yield RawBlockStop(index=state.reasoning_content_index)
        if state.text_content_index is not None:
            yield RawBlockStop(index=state.text_content_index)
        for content_idx in state.tool_content_index.values():
            yield RawBlockStop(index=content_idx)
        yield RawFinish(reason=finish)

    # --- usage ---
    if usage is not None:
        details = usage.get("completion_tokens_details") or {}
        yield RawUsage(usage=Usage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            reasoning_tokens=details.get("reasoning_tokens"),
        ))


class OpenAIChatCompletionStream(ChatCompletionStream):
    def _open_http(self) -> Any:
        payload = self._transport._build_chat_completion_payload(self._request, stream=True)
        return self._transport._client.stream(
            "POST",
            self._transport._chat_completion_url(self._request, stream=True),
            json=payload,
            headers=self._transport._headers(),
        )

    def parse_chunks(self) -> Iterator[RawStreamEvent]:
        state = _OpenAIParserState()
        for line in self._http_response.iter_lines():
            data = _parse_sse_line(line)
            if data is None:
                continue
            if data == "[DONE]":
                return
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as e:
                raise StreamError(
                    f"OpenAI stream produced non-JSON data: {data[:80]!r}",
                ) from e
            yield from _process_chunk(state, chunk)


class OpenAIAsyncChatCompletionStream(AsyncChatCompletionStream):
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
        state = _OpenAIParserState()
        async for line in self._http_response.aiter_lines():
            data = _parse_sse_line(line)
            if data is None:
                continue
            if data == "[DONE]":
                return
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as e:
                raise StreamError(
                    f"OpenAI stream produced non-JSON data: {data[:80]!r}",
                ) from e
            for ev in _process_chunk(state, chunk):
                yield ev
