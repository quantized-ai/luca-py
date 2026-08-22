"""Faux streaming: a scripted message walked through the real streamer
machinery. No httpx anywhere — `open_wire`/`aopen_wire` install a generator
over the script as the chunk source, so the iteration, deadline, and
cancellation mechanics of the mixins run verbatim (which is what lets the
async timeout tests drive them through the faux)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any, ClassVar

from ...exceptions import StreamError
from ...types.completion import ChatCompletionRequest
from ...types.content import (
    PrivateProviderBlock,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    WebFetchBlock,
    WebSearchBlock,
)
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
    WebSearchEvent,
    WebSearchResultEvent,
    WebStartEvent,
)
from ..streamer import AsyncStreamerMixin, BaseStreamer, SyncStreamerMixin
from .transport import (
    FauxWireMixin,
    _FauxAssistantMessage,
    _FauxHang,
    _FauxRefusal,
    _FauxText,
    _FauxThinking,
    _FauxToolCall,
)


class FauxStreamer(FauxWireMixin, BaseStreamer):
    """One start/delta/end triple per scripted block, then the scripted error
    (always a `StreamError` — `error_class` is ignored in streaming), then
    usage and finish. The hang marker lives in the source, not a handler:
    the async source parks on it until cancelled, the sync source refuses."""

    PROVIDER = "faux"

    HANDLERS_BY_TYPE: ClassVar[dict[str, str]] = {
        "start": "handle_start",
        "block_start": "handle_block_start",
        "block_delta": "handle_block_delta",
        "block_stop": "handle_block_stop",
        "error": "handle_error",
        "usage": "handle_usage",
        "finish": "handle_finish",
    }

    def __init__(
        self,
        request: ChatCompletionRequest,
        *,
        scripted: _FauxAssistantMessage,
        provider: str | None = None,
        httpx_client: Any = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(
            request,
            provider=provider,
            httpx_client=httpx_client,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self.scripted = scripted

    # --- the scripted source (replaces the HTTP open) ---

    def script_events(self) -> Iterator[dict]:
        yield {"type": "start"}
        index = 0
        for block in self.scripted.blocks:
            if isinstance(block, _FauxHang):
                yield {"type": "hang"}
                continue
            yield {"type": "block_start", "index": index, "block": block}
            yield {"type": "block_delta", "index": index, "block": block}
            yield {"type": "block_stop", "index": index}
            index += 1
        if self.scripted.error is not None:
            yield {"type": "error", "message": self.scripted.error.message}
        if self.scripted.usage is not None:
            yield {"type": "usage"}
        yield {"type": "finish"}

    def open_wire(self) -> Iterator[dict]:
        return self.sync_source()

    async def aopen_wire(self) -> AsyncIterator[dict]:
        return self.async_source()

    def sync_source(self) -> Iterator[dict]:
        for raw in self.script_events():
            if raw["type"] == "hang":
                raise RuntimeError("faux_hang() is async-only; use acompletion_stream")
            yield raw

    async def async_source(self) -> AsyncIterator[dict]:
        for raw in self.script_events():
            if raw["type"] == "hang":
                await asyncio.Event().wait()  # until cancelled / timed out
                continue
            yield raw

    def parse(self, chunk: dict) -> list:
        return [chunk]  # the source yields raw dicts already

    # --- handlers ---

    def handle_start(self, raw_event: dict) -> list[StreamEvent]:
        return [StartEvent()]

    def handle_block_start(self, raw_event: dict) -> list[StreamEvent]:
        index, block = raw_event["index"], raw_event["block"]
        if isinstance(block, _FauxText):
            self.message.content.append(TextBlock(text=""))
            return [TextStartEvent(index=index)]
        if isinstance(block, (PrivateProviderBlock, WebSearchBlock, WebFetchBlock)):
            # Scripted web content: the REAL client block lands on the message
            # whole (copied — a reused script cannot alias two responses);
            # portable blocks open the web event vocabulary, privates stream
            # silently, exactly like the real streamers.
            self.message.content.append(block.model_copy(deep=True))
            if isinstance(block, (WebSearchBlock, WebFetchBlock)):
                return [WebStartEvent(id=block.extras.get("id") or "")]
            return []
        if isinstance(block, _FauxThinking):
            # A redacted block carries its payload on the start event and
            # streams no deltas, mirroring how a provider sends one.
            self.message.content.append(
                ThinkingBlock(
                    text="",
                    signature=block.signature if block.redacted else None,
                    redacted=block.redacted,
                )
            )
            return [ThinkingStartEvent(index=index)]
        if isinstance(block, _FauxToolCall):
            self.message.content.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments={},
                    partial_arguments="",
                    complete=False,
                )
            )
            return [ToolCallStartEvent(index=index, id=block.id, name=block.name)]
        if isinstance(block, _FauxRefusal):
            self.message.content.append(RefusalBlock(text=""))
            return [RefusalStartEvent(index=index)]
        raise ValueError(f"Unknown faux block type {type(block).__name__}")

    def handle_block_delta(self, raw_event: dict) -> list[StreamEvent]:
        index, block = raw_event["index"], raw_event["block"]
        if isinstance(block, _FauxText):
            events: list[StreamEvent] = []
            if block.text:
                self.message.content[index].text += block.text
                events.append(TextDeltaEvent(index=index, delta=block.text))
            for annotation in block.annotations or []:
                # A citation is complete when it arrives, covering text
                # already streamed — the annotation lands on the block AND
                # streams its own event, like the real streamers.
                self.message.content[index].annotations.append(annotation)
                events.append(TextAnnotationEvent(index=index, annotation=annotation))
            return events
        if isinstance(block, WebSearchBlock):
            op_id = block.extras.get("id") or ""
            events = [WebSearchEvent(id=op_id, queries=list(block.queries))]
            # the real streamers suppress the results event when the list is
            # empty or absent — the faux must not play back a shape
            # production never produces
            if block.results:
                events.append(WebSearchResultEvent(id=op_id, results=self.message.content[index].results))
            return events
        if isinstance(block, WebFetchBlock):
            return [WebFetchEvent(id=block.extras.get("id") or "", urls=[block.web_page.url])]
        if isinstance(block, PrivateProviderBlock):
            return []
        if isinstance(block, _FauxThinking):
            if block.redacted or not block.text:
                return []
            self.message.content[index].text += block.text
            if block.signature is not None:
                self.message.content[index].signature = block.signature
            return [ThinkingDeltaEvent(index=index, delta=block.text)]
        if isinstance(block, _FauxToolCall):
            arguments = json.dumps(block.arguments) if block.arguments else "{}"
            self.message.content[index].partial_arguments += arguments
            return [ToolCallDeltaEvent(index=index, arguments_delta=arguments)]
        if isinstance(block, _FauxRefusal):
            if not block.text:
                return []
            self.message.content[index].text += block.text
            return [RefusalDeltaEvent(index=index, delta=block.text)]
        return []

    def handle_block_stop(self, raw_event: dict) -> list[StreamEvent]:
        index = raw_event["index"]
        block = self.message.content[index]
        if isinstance(block, ToolCall):
            block.arguments = json.loads(block.partial_arguments) if block.partial_arguments else {}
            block.partial_arguments = ""
            block.complete = True
            return [ToolCallEndEvent(index=index, tool_call=block)]
        if isinstance(block, ThinkingBlock):
            return [ThinkingEndEvent(index=index, content=block.text)]
        if isinstance(block, RefusalBlock):
            return [RefusalEndEvent(index=index, content=block.text)]
        if isinstance(block, (WebSearchBlock, WebFetchBlock)):
            return [WebEndEvent(id=block.extras.get("id") or "")]
        if isinstance(block, PrivateProviderBlock):
            return []
        return [TextEndEvent(index=index, content=block.text)]

    def handle_error(self, raw_event: dict) -> list[StreamEvent]:
        raise StreamError(raw_event["message"])

    def handle_usage(self, raw_event: dict) -> list[StreamEvent]:
        self.usage = self.scripted.usage
        return [UsageEvent(usage=self.usage)]

    def handle_finish(self, raw_event: dict) -> list[StreamEvent]:
        self.stop_reason = self.scripted.finish_reason
        return [self.finish_event()]


class SyncFauxStreamer(SyncStreamerMixin, FauxStreamer):
    pass


class AsyncFauxStreamer(AsyncStreamerMixin, FauxStreamer):
    pass
