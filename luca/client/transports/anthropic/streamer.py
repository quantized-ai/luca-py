"""Anthropic Messages streamer.

The cleanest of the wires: every chunk is a typed event with explicit block
boundaries (`message_start`, `content_block_start/delta/stop`,
`message_delta`, `message_stop`). Usage arrives before the terminal
(`message_start` carries input tokens, `message_delta` the authoritative
final counts), so the finish is emitted straight from `handle_message_stop`.

The streamer keeps its own wire-index → content-index map: an unknown
`content_block_start` type goes into an ignored set and its deltas and stop
are dropped, so a future server-tool block degrades to "ignored" instead of
a dense-index failure.

Two shapes here are less than obvious:

- ADJACENT TEXT BLOCKS MERGE. Anthropic splits a cited answer into per-span
  text blocks; canonically they are ONE TextBlock, so every adjacent wire
  text block continues the same open "run" and TextEndEvent waits until a
  non-text block (or the message end) breaks the adjacency. A citation
  arrives complete in `citations_delta` BEFORE its span's text, so it is
  held until the span's `content_block_stop`, when the cited length — and
  therefore its character range over the run — is known.

- SERVER TOOL BLOCKS BECOME PRIVATE BLOCKS. A `server_tool_use` streams its
  input like a tool call (or arrives whole under a code_execution caller);
  its `*_tool_result` arrives whole in the start event. Each is kept
  verbatim as a PrivateProviderBlock, and a completed web result appends
  the synthetic WebSearchBlock / WebFetchBlock beside it — the same two
  views the non-streaming parse produces.
"""

from __future__ import annotations

import contextlib
import json
from typing import ClassVar

from ...exceptions import StreamError
from ...types.completion import ChatCompletionRequest, Usage
from ...types.content import PrivateProviderBlock, TextBlock, ThinkingBlock, ToolCall, WebSearchBlock
from ...types.streaming import (
    StartEvent,
    StreamEvent,
    TextAnnotationEvent,
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
    WebEndEvent,
    WebFetchEvent,
    WebSearchEvent,
    WebSearchResultEvent,
    WebStartEvent,
)
from ..streamer import AsyncStreamerMixin, BaseStreamer, SyncStreamerMixin
from .transport import _SERVER_RESULT_TYPES, AnthropicWireMixin

# The two server tools whose operations get web events; code_execution and
# future server tools pass through as private blocks only.
_WEB_TOOL_NAMES = ("web_search", "web_fetch")


class AnthropicStreamer(AnthropicWireMixin, BaseStreamer):
    PROVIDER = "anthropic"
    URL = "https://api.anthropic.com"

    HANDLERS_BY_TYPE: ClassVar[dict[str, str]] = {
        "message_start": "handle_message_start",
        "content_block_start": "handle_content_block_start",
        "content_block_delta": "handle_content_block_delta",
        "content_block_stop": "handle_content_block_stop",
        "message_delta": "handle_message_delta",
        "message_stop": "handle_message_stop",
        "error": "handle_error",
    }

    def __init__(self, request: ChatCompletionRequest, **kwargs) -> None:
        super().__init__(request, **kwargs)
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens: int | None = None
        self.server_tool_usage: dict = {}
        self.indices: dict[int, int] = {}  # wire index -> message.content index
        self.ignored: set[int] = set()  # wire indices of unknown block types
        # The open merged text run (see module docstring), its per-wire-block
        # start offsets, citations held until their block's stop, and the
        # run's wire blocks rebuilt for private preservation of cited runs.
        self.text_run_index: int | None = None
        self.text_offsets: dict[int, int] = {}  # wire index -> run offset
        self.pending_citations: dict[int, list[dict]] = {}  # wire index -> citations
        self.text_wire_recs: dict[int, dict] = {}  # wire index -> wire block copy
        self.text_run_wire_blocks: list[dict] = []  # completed wire blocks of the run
        # Server-tool bookkeeping: streamed partial input per wire index, and
        # each call's data by id so its result can pair with it.
        self.server_tool_partials: dict[int, str] = {}
        self.web_calls: dict[str, dict] = {}

    def handle_message_start(self, raw_event: dict) -> list[StreamEvent]:
        usage = (raw_event.get("message") or {}).get("usage") or {}
        self.input_tokens = usage.get("input_tokens", 0)
        self.output_tokens = usage.get("output_tokens", 0)
        return [StartEvent()]

    def handle_content_block_start(self, raw_event: dict) -> list[StreamEvent]:
        wire_index = raw_event.get("index", 0)
        block = raw_event.get("content_block") or {}
        block_type = block.get("type")
        if block_type == "text":
            return self._open_text_block(wire_index, block)
        # Any other block type breaks text adjacency: the open run, if any,
        # is complete and its TextEndEvent precedes the new block's start.
        events = self._close_text_run()
        if block_type == "thinking":
            self.message.content.append(ThinkingBlock(text=""))
            index = self.indices[wire_index] = len(self.message.content) - 1
            return [*events, ThinkingStartEvent(index=index)]
        if block_type == "redacted_thinking":
            # A redacted block arrives whole in the start event — encrypted
            # payload, no deltas — so `data` has to be kept here or it is
            # lost for good.
            self.message.content.append(ThinkingBlock(text="", signature=block.get("data"), redacted=True))
            index = self.indices[wire_index] = len(self.message.content) - 1
            return [*events, ThinkingStartEvent(index=index)]
        if block_type == "tool_use":
            if block.get("id") is None or block.get("name") is None:
                raise StreamError(
                    f"Anthropic stream opened a tool_use block without an id or name (index={wire_index})",
                )
            call = ToolCall(
                id=block["id"],
                name=block["name"],
                arguments={},
                partial_arguments="",
                complete=False,
            )
            self.message.content.append(call)
            index = self.indices[wire_index] = len(self.message.content) - 1
            return [*events, ToolCallStartEvent(index=index, id=call.id, name=call.name)]
        if block_type == "server_tool_use":
            # Input either streams via input_json_delta (direct calls) or
            # arrives complete here (code_execution-caller calls); the fact
            # event waits for the stop either way. web_calls registers the
            # MODEL'S data dict — the one the stop finalizes in place — so
            # the result's pairing sees the completed input.
            private = PrivateProviderBlock(format=self.WIRE_FORMAT, data=block)
            self.message.content.append(private)
            self.indices[wire_index] = len(self.message.content) - 1
            if (call_id := private.data.get("id")) is not None:
                self.web_calls[call_id] = private.data
            if private.data.get("name") in _WEB_TOOL_NAMES:
                events.append(WebStartEvent(id=call_id or ""))
            return events
        if block_type in _SERVER_RESULT_TYPES:
            # Results arrive 100% complete in the start event, no deltas; the
            # synthetic block and its events land at the stop.
            self.message.content.append(PrivateProviderBlock(format=self.WIRE_FORMAT, data=block))
            self.indices[wire_index] = len(self.message.content) - 1
            return events
        self.ignored.add(wire_index)
        return events

    def _open_text_block(self, wire_index: int, block: dict) -> list[StreamEvent]:
        """Open the merged text run, or continue it for an adjacent wire text
        block. The block's run offset anchors its citations' ranges, and a
        copy of the start block seeds the wire-block rebuild."""
        events: list[StreamEvent] = []
        if self.text_run_index is None:
            self.message.content.append(TextBlock(text=""))
            self.text_run_index = len(self.message.content) - 1
            events.append(TextStartEvent(index=self.text_run_index))
        self.indices[wire_index] = self.text_run_index
        self.text_offsets[wire_index] = len(self.message.content[self.text_run_index].text)
        self.text_wire_recs[wire_index] = dict(block)
        return events

    def _close_text_run(self) -> list[StreamEvent]:
        """The run is complete. A cited run keeps its rebuilt wire blocks as
        private blocks after the merged TextBlock — same preservation, same
        adjacency, as the non-streaming parse (`_text_run_private_blocks`)."""
        if self.text_run_index is None:
            return []
        index, self.text_run_index = self.text_run_index, None
        self.message.content.extend(self._text_run_private_blocks(self.text_run_wire_blocks))
        self.text_run_wire_blocks = []
        return [TextEndEvent(index=index, content=self.message.content[index].text)]

    def handle_content_block_delta(self, raw_event: dict) -> list[StreamEvent]:
        wire_index = raw_event.get("index", 0)
        index = self.indices.get(wire_index)
        if index is None:
            return []
        delta = raw_event.get("delta") or {}
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            text = delta.get("text", "")
            self.message.content[index].text += text
            return [TextDeltaEvent(index=index, delta=text)]
        if delta_type == "thinking_delta":
            text = delta.get("thinking", "")
            self.message.content[index].text += text
            return [ThinkingDeltaEvent(index=index, delta=text)]
        if delta_type == "signature_delta":
            # The attestation, on a zero-text delta just before the stop.
            self.message.content[index].signature = delta.get("signature")
            return [ThinkingDeltaEvent(index=index, delta="")]
        if delta_type == "input_json_delta":
            partial = delta.get("partial_json", "")
            block = self.message.content[index]
            if isinstance(block, PrivateProviderBlock):
                # A server tool's input has no public delta events; it lands
                # whole on the private block at the stop.
                self.server_tool_partials[wire_index] = self.server_tool_partials.get(wire_index, "") + partial
                return []
            block.partial_arguments += partial
            return [ToolCallDeltaEvent(index=index, arguments_delta=partial)]
        if delta_type == "citations_delta":
            # Complete on arrival, but its range needs the cited span's text,
            # which streams AFTER it — held until the block's stop.
            self.pending_citations.setdefault(wire_index, []).append(delta.get("citation") or {})
            return []
        return []

    def handle_content_block_stop(self, raw_event: dict) -> list[StreamEvent]:
        wire_index = raw_event.get("index", 0)
        index = self.indices.get(wire_index)
        if index is None:
            return []
        block = self.message.content[index]
        if isinstance(block, PrivateProviderBlock):
            return self._close_server_tool_block(wire_index, block)
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
        return self._close_text_block(wire_index, block)

    def _close_text_block(self, wire_index: int, block: TextBlock) -> list[StreamEvent]:
        """One wire text block closed. The merged run stays open — an
        adjacent text block may continue it — but the cited span's length is
        now known, so held citations become annotations with their range,
        and the original wire block can be rebuilt for preservation."""
        start = self.text_offsets.pop(wire_index, 0)
        citations = self.pending_citations.pop(wire_index, [])
        events: list[StreamEvent] = []
        for citation in citations:
            annotation = self._citation_annotation(citation, start, len(block.text))
            if annotation is not None:
                block.annotations.append(annotation)
                events.append(TextAnnotationEvent(index=self.indices[wire_index], annotation=annotation))
        rec = self.text_wire_recs.pop(wire_index, None)
        if rec is not None:
            rec["text"] = block.text[start:]
            if citations:
                rec["citations"] = list(rec.get("citations") or []) + citations
            self.text_run_wire_blocks.append(rec)
        return events

    def _close_server_tool_block(self, wire_index: int, block: PrivateProviderBlock) -> list[StreamEvent]:
        """A server-tool block completed: finalize a call's streamed input
        and state its fact, or pair a result with its call into the synthetic
        block — the same projection the non-streaming parse makes."""
        data = block.data
        if data.get("type") == "server_tool_use":
            partial = self.server_tool_partials.pop(wire_index, "")
            if partial:
                # Unlike a client tool call's arguments — which the caller
                # must execute, so malformed JSON is a StreamError there —
                # this input is descriptive: the provider already ran the
                # tool. Unparseable input degrades to the start-time value
                # rather than killing a live stream.
                with contextlib.suppress(json.JSONDecodeError):
                    data["input"] = json.loads(partial)
            call_id = data.get("id") or ""
            tool_input = data.get("input") or {}
            if data.get("name") == "web_search":
                query = tool_input.get("query")
                return [WebSearchEvent(id=call_id, queries=[query] if query else [])]
            if data.get("name") == "web_fetch":
                url = tool_input.get("url")
                return [WebFetchEvent(id=call_id, urls=[url] if url else [])]
            return []
        synthetic = self._web_block(self.web_calls.get(data.get("tool_use_id")), data)
        if synthetic is None:
            return []  # code_execution results and future kin: private only
        self.message.content.append(synthetic)
        result_id = data.get("tool_use_id") or ""
        events: list[StreamEvent] = []
        if isinstance(synthetic, WebSearchBlock) and synthetic.results:
            events.append(WebSearchResultEvent(id=result_id, results=list(synthetic.results)))
        events.append(WebEndEvent(id=result_id))
        return events

    def handle_message_delta(self, raw_event: dict) -> list[StreamEvent]:
        delta = raw_event.get("delta") or {}
        if (reason := delta.get("stop_reason")) is not None:
            self.stop_reason = reason
        usage = raw_event.get("usage") or {}
        if "input_tokens" in usage:
            # Server tools re-bill input while they loop; this final count is
            # authoritative over message_start's.
            self.input_tokens = usage["input_tokens"]
        if "output_tokens" in usage:
            self.output_tokens = usage["output_tokens"]
        if (thinking_tokens := (usage.get("output_tokens_details") or {}).get("thinking_tokens")) is not None:
            self.reasoning_tokens = thinking_tokens
        if (server_tool_use := usage.get("server_tool_use")) is not None:
            self.server_tool_usage = server_tool_use
        return []

    def handle_message_stop(self, raw_event: dict) -> list[StreamEvent]:
        events: list[StreamEvent] = self._close_text_run()
        self.usage = Usage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
            reasoning_tokens=self.reasoning_tokens,
            tool_requests=self._tool_requests(self.server_tool_usage),
            provider_tool_usage=self.server_tool_usage,
        )
        events.append(UsageEvent(usage=self.usage))
        if self.stop_reason is not None:
            events.append(self.finish_event())
        return events

    def handle_error(self, raw_event: dict) -> list[StreamEvent]:
        err = raw_event.get("error") or {}
        raise StreamError(f"Anthropic stream error: {err.get('message', 'Anthropic stream error')}")


class SyncAnthropicStreamer(SyncStreamerMixin, AnthropicStreamer):
    pass


class AsyncAnthropicStreamer(AsyncStreamerMixin, AnthropicStreamer):
    pass
