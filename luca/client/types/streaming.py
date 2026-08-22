"""Streaming protocol: the public `StreamEvent` discriminated union.

The events are produced by the streamer classes in `luca/client/transports/`
(one streamer per wire protocol — see `transports/streamer.py`). The union is
described in api_prd.md §12.6.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SerializeAsAny

from ..exceptions import ClientError
from .completion import Usage
from .content import TextBlock, ToolCall, URLCitationAnnotation, WebPagePart
from .messages import AssistantMessage


class StartEvent(BaseModel):
    type: Literal["start"] = "start"

    model_config = ConfigDict(extra="forbid")


class TextStartEvent(BaseModel):
    type: Literal["text_start"] = "text_start"
    index: int

    model_config = ConfigDict(extra="forbid")


class TextDeltaEvent(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    index: int
    delta: str

    model_config = ConfigDict(extra="forbid")


class TextEndEvent(BaseModel):
    type: Literal["text_end"] = "text_end"
    index: int
    content: str

    model_config = ConfigDict(extra="forbid")


class ThinkingStartEvent(BaseModel):
    type: Literal["thinking_start"] = "thinking_start"
    index: int

    model_config = ConfigDict(extra="forbid")


class ThinkingDeltaEvent(BaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    index: int
    delta: str

    model_config = ConfigDict(extra="forbid")


class ThinkingEndEvent(BaseModel):
    type: Literal["thinking_end"] = "thinking_end"
    index: int
    content: str

    model_config = ConfigDict(extra="forbid")


class ToolCallStartEvent(BaseModel):
    type: Literal["tool_call_start"] = "tool_call_start"
    index: int
    id: str
    name: str

    model_config = ConfigDict(extra="forbid")


class ToolCallDeltaEvent(BaseModel):
    type: Literal["tool_call_delta"] = "tool_call_delta"
    index: int
    arguments_delta: str

    model_config = ConfigDict(extra="forbid")


class ToolCallEndEvent(BaseModel):
    type: Literal["tool_call_end"] = "tool_call_end"
    index: int
    # SerializeAsAny: native ToolCall subclass fields must survive a dump,
    # same rule as AssistantMessage.content.
    tool_call: SerializeAsAny[ToolCall]

    model_config = ConfigDict(extra="forbid")


class RefusalStartEvent(BaseModel):
    type: Literal["refusal_start"] = "refusal_start"
    index: int

    model_config = ConfigDict(extra="forbid")


class RefusalDeltaEvent(BaseModel):
    type: Literal["refusal_delta"] = "refusal_delta"
    index: int
    delta: str

    model_config = ConfigDict(extra="forbid")


class RefusalEndEvent(BaseModel):
    type: Literal["refusal_end"] = "refusal_end"
    index: int
    content: str

    model_config = ConfigDict(extra="forbid")


class TextAnnotationEvent(BaseModel):
    """A citation landed on the text block at `index`. The annotation is
    complete when it arrives; its range covers text already streamed."""

    type: Literal["text_annotation"] = "text_annotation"
    index: int
    annotation: URLCitationAnnotation

    model_config = ConfigDict(extra="forbid")


class WebStartEvent(BaseModel):
    """A provider-hosted web operation opened. `id` is the provider's id for
    the operation and links every web event of the same operation."""

    type: Literal["web_start"] = "web_start"
    id: str

    model_config = ConfigDict(extra="forbid")


class WebSearchEvent(BaseModel):
    type: Literal["web_search"] = "web_search"
    id: str
    queries: list[str]

    model_config = ConfigDict(extra="forbid")


class WebSearchResultEvent(BaseModel):
    """The results one search returned, batched exactly as the provider
    delivered them."""

    type: Literal["web_search_result"] = "web_search_result"
    id: str
    results: list[WebPagePart]

    model_config = ConfigDict(extra="forbid")


class WebFetchEvent(BaseModel):
    type: Literal["web_fetch"] = "web_fetch"
    id: str
    urls: list[str]

    model_config = ConfigDict(extra="forbid")


class WebFindEvent(BaseModel):
    type: Literal["web_find"] = "web_find"
    id: str
    url: str
    pattern: str

    model_config = ConfigDict(extra="forbid")


class WebEndEvent(BaseModel):
    """The operation finished. The completed canonical blocks are on the
    final AssistantMessage; this event does not duplicate them."""

    type: Literal["web_end"] = "web_end"
    id: str

    model_config = ConfigDict(extra="forbid")


class UsageEvent(BaseModel):
    type: Literal["usage"] = "usage"
    usage: Usage

    model_config = ConfigDict(extra="forbid")


class FinishEvent(BaseModel):
    type: Literal["finish"] = "finish"
    message: AssistantMessage
    finish_reason: str | None
    provider_finish_reason: str | None
    cancelled: bool = False
    usage: Usage
    tool_calls: list[SerializeAsAny[ToolCall]] = Field(default_factory=list)

    _response_format: Any | None = PrivateAttr(default=None)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def parse(self) -> Any:
        if self._response_format is None:
            raise ValueError("No response_format was set on the originating request; cannot parse().")
        from .structured import parse_structured_output

        text = "".join(block.text for block in self.message.content if isinstance(block, TextBlock))
        return parse_structured_output(text, self._response_format)


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    error: ClientError
    usage: Usage | None = None
    raw: Any = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)


StreamEvent = Annotated[
    StartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | TextAnnotationEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | RefusalStartEvent
    | RefusalDeltaEvent
    | RefusalEndEvent
    | WebStartEvent
    | WebSearchEvent
    | WebSearchResultEvent
    | WebFetchEvent
    | WebFindEvent
    | WebEndEvent
    | UsageEvent
    | FinishEvent
    | ErrorEvent,
    Field(discriminator="type"),
]
