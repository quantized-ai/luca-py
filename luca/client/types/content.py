"""Content blocks (the canonical home for everything a turn carries).

Discriminated union on `type`. Every block has `extra="forbid"`.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from .media import MediaSource


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str
    signature: str | None = None

    model_config = ConfigDict(extra="forbid")


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    source: MediaSource

    model_config = ConfigDict(extra="forbid")


class AudioBlock(BaseModel):
    type: Literal["audio"] = "audio"
    source: MediaSource

    model_config = ConfigDict(extra="forbid")


class FileBlock(BaseModel):
    type: Literal["file"] = "file"
    source: MediaSource
    name: str | None = None

    model_config = ConfigDict(extra="forbid")


class ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    text: str
    signature: str | None = None
    redacted: bool = False

    model_config = ConfigDict(extra="forbid")


class ToolCall(BaseModel):
    """A tool call emitted by the model. ONE class, two views — same instances
    live in AssistantMessage.content AND surface via message.tool_calls /
    response.tool_calls / stream.tool_calls (filtered, never copied)."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str

    arguments: dict = Field(default_factory=dict)
    partial_arguments: str = ""
    complete: bool = True
    thought_signature: str | None = None

    def parse_arguments(self, schema: Any) -> Any:
        """Validate self.arguments against `schema`. Returns a typed object.

        Raises StructuredOutputError on validation failure."""
        from ..exceptions import StructuredOutputError

        try:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema.model_validate(self.arguments)
            if isinstance(schema, TypeAdapter):
                return schema.validate_python(self.arguments)
        except ValidationError as e:
            raise StructuredOutputError(
                f"Tool call arguments failed validation: {e}",
                original_exception=e,
            )
        raise StructuredOutputError(
            f"Cannot parse arguments against schema of type {type(schema).__name__}; "
            "pass a BaseModel subclass or a TypeAdapter."
        )

    model_config = ConfigDict(extra="forbid")


class ToolResultBlock(BaseModel):
    """Tool result embedded inline (Anthropic-style). For the top-level
    OpenAI-style pattern, use ToolMessage instead."""

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    content: str | list[Union[TextBlock, ImageBlock]]
    is_error: bool = False

    model_config = ConfigDict(extra="forbid")


class RefusalBlock(BaseModel):
    type: Literal["refusal"] = "refusal"
    text: str

    model_config = ConfigDict(extra="forbid")


ContentBlock = Annotated[
    Union[
        TextBlock, ThinkingBlock, RefusalBlock,
        ImageBlock, AudioBlock, FileBlock,
        ToolCall, ToolResultBlock,
    ],
    Field(discriminator="type"),
]
