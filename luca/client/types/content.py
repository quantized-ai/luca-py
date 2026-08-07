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

# The one reserved key in `ToolCall.extras` / `ToolMessage.extras`: it names
# the native type the entry really is, and every other key is a field of that
# class. See ToolCall.as_native / as_generic.
CUSTOM_TYPE_KEY = "custom_type"


def _native_wire_type(cls: type[BaseModel]) -> str | None:
    """The native `type` literal a class declares, or None for a base class.

    `ToolCall` defaults `type` to "tool_call" and `ToolMessage` has no `type`
    field at all — the same two shapes `__pydantic_init_subclass__` treats as
    "not a native subclass"."""
    field = cls.model_fields.get("type")
    if field is None or field.default == "tool_call":
        return None
    return field.default


def _as_native(obj: Any, registry: dict[str, Any], label: str) -> Any:
    """The generic form -> the registered native class it names in `extras`.

    Returns `obj` unchanged when there is nothing to do: no `custom_type`, or
    an instance that is already native (its own `extras` are then inert)."""
    from ..exceptions import BadRequestError

    custom_type = obj.extras.get(CUSTOM_TYPE_KEY)
    if custom_type is None or _native_wire_type(type(obj)) is not None:
        return obj
    target = registry.get(custom_type)
    if target is None:
        raise BadRequestError(
            f"Unknown native {label} type {custom_type!r} in extras. Native "
            "types register themselves at import; the module defining this "
            "one was never imported.",
        )
    data = obj.model_dump(exclude={"type", "extras"})
    data.update({k: v for k, v in obj.extras.items() if k != CUSTOM_TYPE_KEY})
    try:
        return target.model_validate(data)
    except ValidationError as e:
        raise BadRequestError(
            f"extras do not describe a valid {target.__name__}: {e}",
            original_exception=e,
        ) from e


def _as_generic(obj: Any, base_cls: type[BaseModel]) -> Any:
    """A native instance -> the base class carrying its identity in `extras`.

    Mechanical and exactly inverse to `_as_native`: `custom_type` is the
    class's `type` literal, and every field it declares beyond the base
    becomes one entry."""
    wire_type = _native_wire_type(type(obj))
    if wire_type is None:
        return obj
    data = obj.model_dump()
    base_names = set(base_cls.model_fields)
    extras = {**obj.extras, CUSTOM_TYPE_KEY: wire_type}
    extras.update({k: v for k, v in data.items() if k not in base_names and k != "type"})
    return base_cls(
        **{k: v for k, v in data.items() if k in base_names and k not in ("type", "extras")},
        extras=extras,
    )


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

    # Free-form and inert, with ONE reserved key: `custom_type` names a
    # registered native type, making this the generic form of that native
    # call — every other key is then one of that class's own fields. The
    # transports normalize through as_native() before projecting, so the two
    # forms produce identical wire payloads. See as_native / as_generic.
    extras: dict[str, Any] = Field(default_factory=dict)

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

    def as_native(self) -> ToolCall:
        """This call as the native subclass its `extras` name.

        `ToolCall(extras={"custom_type": "apply_patch_call", "item_id": …})`
        becomes the equivalent `ApplyPatchToolCall`. Self when there is
        nothing to do. Raises BadRequestError on an unregistered
        `custom_type` or extras the class rejects."""
        return _as_native(self, NATIVE_TOOL_CALL_TYPES, "tool-call")

    def as_generic(self) -> ToolCall:
        """This call as a base ToolCall carrying its native identity in
        `extras` — the inverse of as_native(), and self when already generic.

        The transport-independent form: no provider class needed to store it,
        replay it, or hand it to another process."""
        return _as_generic(self, ToolCall)

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
