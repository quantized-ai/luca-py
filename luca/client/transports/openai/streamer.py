"""OpenAI Chat Completions streamer.

Chunks on this wire carry no event type — one chunk mixes text, tool
fragments, the finish reason, and usage — so the wire class overrides
`handle()` with shape-based routing instead of filling `HANDLERS_BY_TYPE`.
The wire also has no block start/stop events: the streamer synthesizes them
(start on first delta, end when the finish reason arrives). Usage arrives
AFTER the finish-reason chunk (and may not arrive at all), so the terminal
FinishEvent is built at wire end — `parse` maps the `[DONE]` marker to a
`{"done": true}` event that `handle` routes to `handle_wire_end()`, with
source exhaustion as the fallback for servers that just close.

OpenRouter/DeepSeek-style quirks live HERE, not in a subclass: mid-stream
`{"error": …}` chunks and `reasoning`/`reasoning_content` thinking deltas
serve every OpenAI-compatible host and are inert on pure OpenAI (the fields
never appear). The OpenRouter subclass changes only PROVIDER and URL.
"""

from __future__ import annotations

import json

from ...exceptions import StreamError
from ...types.completion import ChatCompletionRequest, Usage
from ...types.content import RefusalBlock, TextBlock, ThinkingBlock, ToolCall
from ...types.streaming import (
    RefusalDeltaEvent,
    RefusalEndEvent,
    RefusalStartEvent,
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
from .transport import OpenAIWireMixin


class OpenAIChatCompletionsStreamer(OpenAIWireMixin, BaseStreamer):
    PROVIDER = "openai"
    URL = "https://api.openai.com/v1"

    def __init__(self, request: ChatCompletionRequest, **kwargs) -> None:
        super().__init__(request, **kwargs)
        self.started = False
        self.text_index: int | None = None
        self.reasoning_index: int | None = None
        self.refusal_index: int | None = None
        # wire tool index -> message.content index (the call IS open)
        self.tool_indices: dict[int, int] = {}
        # wire tool index -> {"id", "name", "buffered_args"} — a call opens
        # only once BOTH id and name have arrived; argument fragments seen
        # before that are buffered and flushed on open.
        self.tool_pending: dict[int, dict] = {}

    def parse(self, chunk: str) -> list:
        if not chunk.startswith("data:"):
            return []
        data = chunk[5:].strip()
        if data == "[DONE]":
            # Wire knowledge: [DONE] IS the end of this stream. Routing it
            # through handle() lets the finish fire on the marker, so a
            # server that lingers with the socket open cannot delay it.
            return [{"done": True}]
        try:
            return [json.loads(data)]
        except json.JSONDecodeError as e:
            raise StreamError(f"Stream produced non-JSON data: {data[:80]!r}") from e

    def handle(self, chunk: dict) -> list[StreamEvent]:
        if chunk.get("done"):
            return self.handle_wire_end()
        # A failure the HOST reports mid-stream, inside an already-200 body
        # (OpenRouter's shape). Checked first: it arrives with EMPTY
        # `choices`, so every branch below would skip it and the stream
        # would end in a FinishEvent indistinguishable from a real answer.
        if (error := chunk.get("error")) is not None:
            detail = error.get("message") if isinstance(error, dict) else None
            raise StreamError(f"OpenAI stream returned an error: {detail or error}")

        events: list[StreamEvent] = []
        if not self.started:
            self.started = True
            events.append(StartEvent())
        if chunk.get("choices"):
            choice = chunk["choices"][0]
            delta = choice.get("delta") or {}
            # Reasoning streams first, so it claims the earliest index.
            if reasoning := (delta.get("reasoning") or delta.get("reasoning_content")):
                events += self.handle_reasoning_delta(reasoning)
            if content := delta.get("content"):
                events += self.handle_text_delta(content)
            if refusal := delta.get("refusal"):
                events += self.handle_refusal_delta(refusal)
            for fragment in delta.get("tool_calls") or []:
                events += self.handle_tool_fragment(fragment)
            if choice.get("finish_reason"):
                self.stop_reason = choice["finish_reason"]
                events += self.close_open_blocks()
        if (usage := chunk.get("usage")) is not None:
            events += self.handle_usage(usage)
        return events

    def handle_reasoning_delta(self, text: str) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if self.reasoning_index is None:
            self.message.content.append(ThinkingBlock(text=""))
            self.reasoning_index = len(self.message.content) - 1
            events.append(ThinkingStartEvent(index=self.reasoning_index))
        self.message.content[self.reasoning_index].text += text
        events.append(ThinkingDeltaEvent(index=self.reasoning_index, delta=text))
        return events

    def handle_text_delta(self, text: str) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if self.text_index is None:
            self.message.content.append(TextBlock(text=""))
            self.text_index = len(self.message.content) - 1
            events.append(TextStartEvent(index=self.text_index))
        self.message.content[self.text_index].text += text
        events.append(TextDeltaEvent(index=self.text_index, delta=text))
        return events

    def handle_refusal_delta(self, text: str) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if self.refusal_index is None:
            self.message.content.append(RefusalBlock(text=""))
            self.refusal_index = len(self.message.content) - 1
            events.append(RefusalStartEvent(index=self.refusal_index))
        self.message.content[self.refusal_index].text += text
        events.append(RefusalDeltaEvent(index=self.refusal_index, delta=text))
        return events

    def handle_tool_fragment(self, fragment: dict) -> list[StreamEvent]:
        wire_index = fragment.get("index")
        if wire_index is None:
            return []
        pending = self.tool_pending.setdefault(
            wire_index,
            {"id": None, "name": None, "buffered_args": []},
        )
        if (id_piece := fragment.get("id")) is not None:
            pending["id"] = id_piece
        function = fragment.get("function") or {}
        if (name_piece := function.get("name")) is not None:
            pending["name"] = name_piece
        args_piece = function.get("arguments", "")

        events: list[StreamEvent] = []
        if wire_index not in self.tool_indices:
            if pending["id"] is not None and pending["name"] is not None:
                call = ToolCall(
                    id=pending["id"],
                    name=pending["name"],
                    arguments={},
                    partial_arguments="",
                    complete=False,
                )
                self.message.content.append(call)
                index = len(self.message.content) - 1
                self.tool_indices[wire_index] = index
                events.append(ToolCallStartEvent(index=index, id=call.id, name=call.name))
                for buffered in pending["buffered_args"]:
                    call.partial_arguments += buffered
                    events.append(ToolCallDeltaEvent(index=index, arguments_delta=buffered))
                pending["buffered_args"].clear()
                if args_piece:
                    call.partial_arguments += args_piece
                    events.append(ToolCallDeltaEvent(index=index, arguments_delta=args_piece))
            elif args_piece:
                pending["buffered_args"].append(args_piece)
        elif args_piece:
            index = self.tool_indices[wire_index]
            self.message.content[index].partial_arguments += args_piece
            events.append(ToolCallDeltaEvent(index=index, arguments_delta=args_piece))
        return events

    def close_open_blocks(self) -> list[StreamEvent]:
        unresolved = [k for k in self.tool_pending if k not in self.tool_indices]
        if unresolved:
            raise StreamError(
                f"OpenAI stream finished with tool calls missing id or name (openai tool_indices: {unresolved})"
            )
        events: list[StreamEvent] = []
        if self.reasoning_index is not None:
            block = self.message.content[self.reasoning_index]
            events.append(ThinkingEndEvent(index=self.reasoning_index, content=block.text))
            self.reasoning_index = None
        if self.text_index is not None:
            block = self.message.content[self.text_index]
            events.append(TextEndEvent(index=self.text_index, content=block.text))
            self.text_index = None
        if self.refusal_index is not None:
            block = self.message.content[self.refusal_index]
            events.append(RefusalEndEvent(index=self.refusal_index, content=block.text))
            self.refusal_index = None
        for index in self.tool_indices.values():
            call = self.message.content[index]
            try:
                call.arguments = json.loads(call.partial_arguments) if call.partial_arguments else {}
            except json.JSONDecodeError as e:
                raise StreamError(f"Tool call {index} ({call.name!r}) returned malformed JSON") from e
            call.partial_arguments = ""
            call.complete = True
            events.append(ToolCallEndEvent(index=index, tool_call=call))
        self.tool_indices = {}
        self.tool_pending = {}
        return events

    def handle_usage(self, raw_usage: dict) -> list[StreamEvent]:
        details = raw_usage.get("completion_tokens_details") or {}
        self.usage = Usage(
            input_tokens=raw_usage.get("prompt_tokens", 0),
            output_tokens=raw_usage.get("completion_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
            reasoning_tokens=details.get("reasoning_tokens"),
        )
        return [UsageEvent(usage=self.usage)]

    def handle_wire_end(self) -> list[StreamEvent]:
        if self.stop_reason is None:
            return []
        return [self.finish_event()]


class SyncOpenAIChatCompletionsStreamer(SyncStreamerMixin, OpenAIChatCompletionsStreamer):
    pass


class AsyncOpenAIChatCompletionsStreamer(AsyncStreamerMixin, OpenAIChatCompletionsStreamer):
    pass
