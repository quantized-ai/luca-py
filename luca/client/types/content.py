"""Content blocks (the canonical home for everything a turn carries).

Discriminated union on `type`. Every block has `extra="forbid"`.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from .media import MediaSource

# Native tool-call classes keyed by their `type` literal — the canonical
# serialization discriminator, which for every first-party native equals the
# wire item type. Filled by ToolCall.__pydantic_init_subclass__ when the
# defining module imports; importing `luca.client` imports all first-party
# native modules, so any client import registers them.
NATIVE_TOOL_CALL_TYPES: dict[str, type[ToolCall]] = {}


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
    """The model's reasoning. `signature` is the provider's opaque attestation
    over it; `id` is the provider's opaque identity FOR it — the OpenAI
    Responses API only replays a reasoning item when both its `rs_…` id and
    its encrypted content come back, so an id-less block is unreplayable
    there. Both are provider-owned: they round-trip verbatim or not at all."""

    type: Literal["thinking"] = "thinking"
    text: str
    id: str | None = None
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

    # Bound by native subclasses to their ToolProjector subclass; None on the
    # base and on any non-native subclass means the transport's default
    # projector. ClassVar[Any] deliberately: the sibling `type` FIELD shadows
    # the builtin in this namespace during annotation evaluation.
    projector_class: ClassVar[Any] = None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        wire_type = cls.model_fields["type"].default
        if wire_type == "tool_call":
            return  # not a native subclass
        existing = NATIVE_TOOL_CALL_TYPES.get(wire_type)
        if existing is not None and existing is not cls:
            raise TypeError(f"tool-call type {wire_type!r} is already registered by {existing.__name__}")
        NATIVE_TOOL_CALL_TYPES[wire_type] = cls

    @model_validator(mode="wrap")
    @classmethod
    def _dispatch_native(cls, value: Any, handler: Any) -> Any:
        # Validating a native payload against the BASE class dispatches to the
        # registered subclass, so containers annotated with ToolCall
        # revalidate conversations that contain native calls. Deserializing a
        # native payload whose module was never imported fails validation.
        if cls is ToolCall and isinstance(value, dict):
            target = NATIVE_TOOL_CALL_TYPES.get(value.get("type", "tool_call"))
            if target is not None:
                return target.model_validate(value)
        return handler(value)

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
            ) from e
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
    content: str | list[TextBlock | ImageBlock]
    is_error: bool = False

    model_config = ConfigDict(extra="forbid")


class RefusalBlock(BaseModel):
    type: Literal["refusal"] = "refusal"
    text: str

    model_config = ConfigDict(extra="forbid")


ContentBlock = Annotated[
    TextBlock | ThinkingBlock | RefusalBlock | ImageBlock | AudioBlock | FileBlock | ToolCall | ToolResultBlock,
    Field(discriminator="type"),
]
