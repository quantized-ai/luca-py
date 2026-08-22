"""OpenAI Responses streamer.

Every SSE frame on this wire is a NAMED event over the whole response object,
so the dispatch table routes on `type` and UNKNOWN TYPES ARE SAFE TO IGNORE —
hosted-tool events (`response.web_search_call.*`, …) and anything OpenAI adds
later carry no canonical block. The bookkeeping turns (output_index,
content_index) pairs back into dense, ordered content blocks; reasoning and
call items share the allocator through synthetic slots.

Native tool-call items (apply_patch_call, shell_call, …) stream start + end
with no public argument deltas: their in-progress payloads are raw-text
events that fall through the dispatch unhandled, the skeleton call is
prebuilt at `output_item.added`, and the complete item replaces it whole at
`output_item.done`.
"""

from __future__ import annotations

import json
from typing import ClassVar

from ...exceptions import StreamError
from ...types.completion import ChatCompletionRequest
from ...types.content import PrivateProviderBlock, RefusalBlock, TextBlock, ThinkingBlock, ToolCall
from ...types.streaming import (
    RefusalDeltaEvent,
    RefusalEndEvent,
    RefusalStartEvent,
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
    WebFindEvent,
    WebSearchEvent,
    WebSearchResultEvent,
    WebStartEvent,
)
from ..streamer import AsyncStreamerMixin, BaseStreamer, SyncStreamerMixin
from .transport import OpenAIResponsesWireMixin

# Synthetic content_index slots, so reasoning and call items share the one
# allocator with real content parts without colliding with content_index 0.
_REASONING_SLOT = -1
_FUNCTION_CALL_SLOT = -2
_NATIVE_CALL_SLOT = -3
_WEB_CALL_SLOT = -4

# Between two summary parts of the same reasoning item — they are separate
# paragraphs of one train of thought, and read as one block.
_SUMMARY_PART_SEPARATOR = "\n\n"

_TERMINAL_EVENT_STATUS = {
    "response.completed": "completed",
    "response.incomplete": "incomplete",
    "response.failed": "failed",
}


class OpenAIResponsesStreamer(OpenAIResponsesWireMixin, BaseStreamer):
    PROVIDER = "openai"
    URL = "https://api.openai.com/v1"

    HANDLERS_BY_TYPE: ClassVar[dict[str, str]] = {
        "response.output_item.added": "handle_output_item_added",
        "response.output_item.done": "handle_output_item_done",
        "response.content_part.added": "handle_content_part_added",
        "response.content_part.done": "handle_content_part_done",
        "response.output_text.delta": "handle_output_text_delta",
        "response.output_text.annotation.added": "handle_output_text_annotation_added",
        "response.refusal.delta": "handle_refusal_delta",
        "response.reasoning_summary_part.added": "handle_reasoning_summary_part_added",
        "response.reasoning_summary_text.delta": "handle_reasoning_summary_text_delta",
        "response.function_call_arguments.delta": "handle_function_call_arguments_delta",
        "response.completed": "handle_terminal",
        "response.incomplete": "handle_terminal",
        "response.failed": "handle_terminal",
        "error": "handle_error",
    }

    def __init__(self, request: ChatCompletionRequest, **kwargs) -> None:
        super().__init__(request, **kwargs)
        self.started = False
        # (output_index, content_index | slot) -> message.content index.
        # Dense and ordered because the wire streams its items in order: an
        # index is allocated the first time a key is seen, never revisited.
        self.indices: dict[tuple[int, int], int] = {}
        # Blocks opened and not yet closed, so the terminal can close them.
        self.open_indices: set[int] = set()

    def handle(self, raw_event: dict) -> list[StreamEvent]:
        events = super().handle(raw_event)
        if not self.started:
            self.started = True
            return [StartEvent(), *events]
        return events

    def allocate(self, key: tuple[int, int], block) -> int:
        self.message.content.append(block)
        index = len(self.message.content) - 1
        self.indices[key] = index
        self.open_indices.add(index)
        return index

    def close(self, index: int) -> bool:
        """True if the block was open, so a duplicate done-event cannot close
        it twice."""
        if index not in self.open_indices:
            return False
        self.open_indices.remove(index)
        return True

    # --- items ---

    def handle_output_item_added(self, raw_event: dict) -> list[StreamEvent]:
        item = raw_event.get("item") or {}
        output_index = raw_event.get("output_index", 0)
        item_type = item.get("type")

        if item_type == "reasoning":
            # The wire item id (`rs_…`) arrives when the item opens; the
            # encrypted payload only exists once it is done.
            index = self.allocate((output_index, _REASONING_SLOT), ThinkingBlock(text="", id=item.get("id")))
            return [ThinkingStartEvent(index=index)]
        if item_type == "function_call":
            call_id, name = item.get("call_id"), item.get("name")
            if call_id is None or name is None:
                raise StreamError(
                    f"OpenAI Responses stream opened a function_call item without a call_id or name "
                    f"(output_index={output_index})",
                )
            call = ToolCall(id=call_id, name=name, arguments={}, partial_arguments="", complete=False)
            index = self.allocate((output_index, _FUNCTION_CALL_SLOT), call)
            return [ToolCallStartEvent(index=index, id=call_id, name=name)]
        if item_type == "web_search_call":
            # The added-time item is a stub (id + status, no action); the
            # private block opens on it and the done event completes it. Not
            # in open_indices — web blocks have no partial state to close.
            self.message.content.append(PrivateProviderBlock(format=self.WIRE_FORMAT, data=item))
            self.indices[(output_index, _WEB_CALL_SLOT)] = len(self.message.content) - 1
            return [WebStartEvent(id=item.get("id") or "")]
        if (projector := self._native_projector_for_item(item_type)) is not None:
            # Observed SSE: output_item.added carries the item skeleton —
            # status "in_progress", call_id present, payload partial.
            if item.get("call_id") is None:
                raise StreamError(
                    f"OpenAI Responses stream opened a {item_type} item without a call_id "
                    f"(output_index={output_index})",
                )
            call = projector.build_tool_call(item)
            call.complete = False
            index = self.allocate((output_index, _NATIVE_CALL_SLOT), call)
            return [ToolCallStartEvent(index=index, id=call.id, name=call.name)]
        # A `message` item's blocks open on content_part.added, one per part.
        # Other hosted-tool items have no canonical block; ignored, not
        # guessed at.
        return []

    def handle_output_item_done(self, raw_event: dict) -> list[StreamEvent]:
        item = raw_event.get("item") or {}
        output_index = raw_event.get("output_index", 0)
        item_type = item.get("type")

        if item_type == "reasoning":
            index = self.indices.get((output_index, _REASONING_SLOT))
            if index is None or not self.close(index):
                return []  # unknown item, or a duplicate done
            events: list[StreamEvent] = []
            block = self.message.content[index]
            # The encrypted payload is the half that makes the block
            # replayable — surfaced as a zero-text delta, exactly as today.
            encrypted = item.get("encrypted_content")
            if encrypted is not None:
                block.signature = encrypted
                events.append(ThinkingDeltaEvent(index=index, delta=""))
            events.append(ThinkingEndEvent(index=index, content=block.text))
            return events
        if item_type == "function_call":
            index = self.indices.get((output_index, _FUNCTION_CALL_SLOT))
            if index is not None and self.close(index):
                return [self.close_tool_call(index)]
            return []
        if item_type == "web_search_call":
            # Popping the mapping is the duplicate-done guard: a second done
            # finds no index and cannot append the synthetic twice.
            index = self.indices.pop((output_index, _WEB_CALL_SLOT), None)
            if index is None:
                return []
            # The done-time item is the complete operation — action, and any
            # requested sources/results — so it replaces the stub whole.
            self.message.content[index].data = item
            events = self._web_fact_events(item)
            synthetic = self._web_block(item)
            if synthetic is not None:
                self.message.content.append(synthetic)
            events.append(WebEndEvent(id=item.get("id") or ""))
            return events
        if self._native_projector_for_item(item_type) is not None:
            index = self.indices.get((output_index, _NATIVE_CALL_SLOT))
            if index is None or not self.close(index):
                return []
            if item.get("call_id") is None:
                return [self.close_tool_call(index)]  # degenerate close, empty args
            # The complete item: swapped in whole, final arguments and all.
            projector = self._native_projector_for_item(item_type)
            replacement = projector.build_tool_call(item)
            self.message.content[index] = replacement
            return [ToolCallEndEvent(index=index, tool_call=replacement)]
        # A `message` item's parts already closed on content_part.done.
        return []

    def _web_fact_events(self, item: dict) -> list[StreamEvent]:
        """The user-facing facts one completed web_search_call states. The
        `.in_progress`/`.searching`/`.completed` progress events carry no
        payload, so everything is known — and emitted — at item done."""
        item_id = item.get("id") or ""
        action = item.get("action") or {}
        action_type = action.get("type")
        events: list[StreamEvent] = []
        if action_type == "search":
            events.append(WebSearchEvent(id=item_id, queries=self._web_queries(action)))
            results = self._web_search_results(item)
            if results:
                events.append(WebSearchResultEvent(id=item_id, results=results))
        elif action_type == "open_page":
            events.append(WebFetchEvent(id=item_id, urls=[action.get("url", "")]))
        elif action_type == "find_in_page":
            events.append(WebFindEvent(id=item_id, url=action.get("url", ""), pattern=action.get("pattern", "")))
        return events

    # --- content parts ---

    def handle_content_part_added(self, raw_event: dict) -> list[StreamEvent]:
        part = raw_event.get("part") or {}
        key = (raw_event.get("output_index", 0), raw_event.get("content_index", 0))
        if part.get("type") == "output_text":
            return [TextStartEvent(index=self.allocate(key, TextBlock(text="")))]
        if part.get("type") == "refusal":
            return [RefusalStartEvent(index=self.allocate(key, RefusalBlock(text="")))]
        return []

    def handle_content_part_done(self, raw_event: dict) -> list[StreamEvent]:
        key = (raw_event.get("output_index", 0), raw_event.get("content_index", 0))
        index = self.indices.get(key)
        if index is None or not self.close(index):
            return []
        block = self.message.content[index]
        if isinstance(block, RefusalBlock):
            return [RefusalEndEvent(index=index, content=block.text)]
        return [TextEndEvent(index=index, content=block.text)]

    def handle_output_text_delta(self, raw_event: dict) -> list[StreamEvent]:
        index = self.indices.get((raw_event.get("output_index", 0), raw_event.get("content_index", 0)))
        if index is None or not raw_event.get("delta"):
            return []
        self.message.content[index].text += raw_event["delta"]
        return [TextDeltaEvent(index=index, delta=raw_event["delta"])]

    def handle_refusal_delta(self, raw_event: dict) -> list[StreamEvent]:
        index = self.indices.get((raw_event.get("output_index", 0), raw_event.get("content_index", 0)))
        if index is None or not raw_event.get("delta"):
            return []
        self.message.content[index].text += raw_event["delta"]
        return [RefusalDeltaEvent(index=index, delta=raw_event["delta"])]

    def handle_output_text_annotation_added(self, raw_event: dict) -> list[StreamEvent]:
        # The annotation is complete when it arrives; its range indexes the
        # full final text, which may still be streaming.
        index = self.indices.get((raw_event.get("output_index", 0), raw_event.get("content_index", 0)))
        if index is None:
            return []
        annotation = self._parse_annotation(raw_event.get("annotation") or {})
        if annotation is None:
            return []
        self.message.content[index].annotations.append(annotation)
        return [TextAnnotationEvent(index=index, annotation=annotation)]

    # --- reasoning summaries ---

    def handle_reasoning_summary_part_added(self, raw_event: dict) -> list[StreamEvent]:
        # The first part opens the block that output_item.added already
        # started; every later one continues it after a paragraph break.
        index = self.indices.get((raw_event.get("output_index", 0), _REASONING_SLOT))
        if index is None or raw_event.get("summary_index", 0) == 0:
            return []
        self.message.content[index].text += _SUMMARY_PART_SEPARATOR
        return [ThinkingDeltaEvent(index=index, delta=_SUMMARY_PART_SEPARATOR)]

    def handle_reasoning_summary_text_delta(self, raw_event: dict) -> list[StreamEvent]:
        index = self.indices.get((raw_event.get("output_index", 0), _REASONING_SLOT))
        if index is None or not raw_event.get("delta"):
            return []
        self.message.content[index].text += raw_event["delta"]
        return [ThinkingDeltaEvent(index=index, delta=raw_event["delta"])]

    # --- function-call arguments ---

    def handle_function_call_arguments_delta(self, raw_event: dict) -> list[StreamEvent]:
        index = self.indices.get((raw_event.get("output_index", 0), _FUNCTION_CALL_SLOT))
        if index is None or not raw_event.get("delta"):
            return []
        self.message.content[index].partial_arguments += raw_event["delta"]
        return [ToolCallDeltaEvent(index=index, arguments_delta=raw_event["delta"])]

    def close_tool_call(self, index: int) -> ToolCallEndEvent:
        """Finish an open call from its accumulated argument buffer. The
        object is mutated, not rebuilt, so a native subclass stays itself."""
        call = self.message.content[index]
        try:
            call.arguments = json.loads(call.partial_arguments) if call.partial_arguments else {}
        except json.JSONDecodeError as e:
            raise StreamError(f"Tool call {index} ({call.name!r}) returned malformed JSON") from e
        call.partial_arguments = ""
        call.complete = True
        return ToolCallEndEvent(index=index, tool_call=call)

    # --- terminal ---

    def handle_terminal(self, raw_event: dict) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        # A truncated response can go terminal mid-block. Closing here is
        # load-bearing for tool calls: an unclosed one would reach the caller
        # as `arguments={}, complete=False` beside a `tool_use` finish and be
        # dispatched with its arguments erased.
        for index in sorted(self.open_indices):
            block = self.message.content[index]
            if isinstance(block, ToolCall):
                events.append(self.close_tool_call(index))
            elif isinstance(block, ThinkingBlock):
                events.append(ThinkingEndEvent(index=index, content=block.text))
            elif isinstance(block, RefusalBlock):
                events.append(RefusalEndEvent(index=index, content=block.text))
            else:
                events.append(TextEndEvent(index=index, content=block.text))
        self.open_indices.clear()

        response = raw_event.get("response") or {}
        if (usage := response.get("usage")) is not None:
            # Same parse as the non-streaming path (`tool_usage` included);
            # no model_info here, so no cost — as on every streamer.
            self.usage = self._parse_usage(usage, None, response.get("tool_usage"))
            events.append(UsageEvent(usage=self.usage))

        # Same composition as the non-streaming path, with the EVENT NAME as
        # the fallback for a terminal missing `status` — never "completed",
        # so a failed response cannot be reported as a successful stop.
        status = response.get("status") or _TERMINAL_EVENT_STATUS.get(raw_event.get("type"), "completed")
        if status == "incomplete":
            reason = (response.get("incomplete_details") or {}).get("reason")
            status = f"incomplete:{reason}" if reason else "incomplete"
        self.stop_reason = status
        events.append(self.finish_event())
        return events

    def handle_error(self, raw_event: dict) -> list[StreamEvent]:
        raise StreamError(
            f"OpenAI Responses stream returned an error: {raw_event.get('message') or raw_event}",
        )


class SyncOpenAIResponsesStreamer(SyncStreamerMixin, OpenAIResponsesStreamer):
    pass


class AsyncOpenAIResponsesStreamer(AsyncStreamerMixin, OpenAIResponsesStreamer):
    pass
