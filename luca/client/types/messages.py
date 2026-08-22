"""Per-role discriminated union: UserMessage | AssistantMessage | ToolMessage.

There is intentionally NO SystemMessage class and NO "system" role. The
system prompt is request-scoped on ChatCompletionRequest.system_message.

`AssistantMessage` carries self-describing terminal state (finish_reason,
provider_finish_reason, error_message, cancelled, usage, provider, model,
timestamp) so a serialized conversation reloads with full context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator

if TYPE_CHECKING:
    from .completion import Usage

from .content import (
    AudioBlock,
    FileBlock,
    ImageBlock,
    PrivateProviderBlock,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    WebFetchBlock,
    WebSearchBlock,
    _as_generic,
    _as_native,
)

# Native tool-message classes keyed by their `type` literal. The base
# ToolMessage has NO `type` field — payloads without one validate as the
# base, so existing serialized data is untouched. Filled by
# ToolMessage.__pydantic_init_subclass__ for subclasses that declare one.
NATIVE_TOOL_MESSAGE_TYPES: dict[str, type[ToolMessage]] = {}


class _UsageHolder:
    """Lazy-import indirection for Usage so messages.py doesn't import completion.py
    at module-import time (which would create a cycle since completion.py imports
    messages.py)."""


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[TextBlock | ImageBlock | AudioBlock | FileBlock]
    name: str | None = None
    timestamp: int | None = None

    model_config = ConfigDict(extra="forbid")


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    # SerializeAsAny: model_dump() must keep native ToolCall subclass fields
    # (item_id, status, …). Without it pydantic dumps per the declared base
    # and drops them, with a warning the test suite turns into an error.
    content: list[
        TextBlock
        | ThinkingBlock
        | SerializeAsAny[ToolCall]
        | RefusalBlock
        | PrivateProviderBlock
        | WebSearchBlock
        | WebFetchBlock
    ] = Field(
        default_factory=list,
    )

    finish_reason: str | None = None
    provider_finish_reason: str | None = None
    cancelled: bool = False
    error_message: str | None = None

    provider: str | None = None
    model: str | None = None
    response_model: str | None = None
    response_id: str | None = None

    # Usage is imported only under TYPE_CHECKING to avoid the circular import
    # (completion.py imports messages.py for the discriminated union). At runtime
    # the annotation stays an unresolved forward ref until types/__init__.py --
    # which has the real Usage in scope -- calls AssistantMessage.model_rebuild().
    usage: Usage | None = None
    timestamp: int | None = None

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Same ToolCall instances filtered out of self.content. Never copied —
        mutating through this view mutates self.content[i] too."""
        return [b for b in self.content if isinstance(b, ToolCall)]

    model_config = ConfigDict(extra="forbid")


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str | list[TextBlock | ImageBlock]
    name: str | None = None
    is_error: bool = False
    timestamp: int | None = None

    # Same contract as ToolCall.extras: free-form and inert, except for the
    # reserved `custom_type` key naming a registered native result type.
    extras: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    def as_native(self) -> ToolMessage:
        """This result as the native subclass its `extras` name.

        `ToolMessage(extras={"custom_type": "shell_call_output", "results":
        […]})` becomes the equivalent `ShellToolMessage`. Self when there is
        nothing to do. Raises BadRequestError on an unregistered
        `custom_type` or extras the class rejects."""
        return _as_native(self, NATIVE_TOOL_MESSAGE_TYPES, "tool-message")

    def as_generic(self) -> ToolMessage:
        """This result as a base ToolMessage carrying its native identity in
        `extras` — the inverse of as_native(), and self when already
        generic."""
        return _as_generic(self, ToolMessage)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        field = cls.model_fields.get("type")
        if field is None:
            return  # not a native subclass
        wire_type = field.default
        existing = NATIVE_TOOL_MESSAGE_TYPES.get(wire_type)
        if existing is not None and existing is not cls:
            raise TypeError(f"tool-message type {wire_type!r} is already registered by {existing.__name__}")
        NATIVE_TOOL_MESSAGE_TYPES[wire_type] = cls

    @model_validator(mode="wrap")
    @classmethod
    def _dispatch_native(cls, value: Any, handler: Any) -> Any:
        # Same dispatch as ToolCall: a dict with a registered "type" validated
        # against the BASE class becomes the subclass; an unknown "type" falls
        # through to the base and fails its extra="forbid".
        if cls is ToolMessage and isinstance(value, dict) and "type" in value:
            target = NATIVE_TOOL_MESSAGE_TYPES.get(value["type"])
            if target is not None:
                return target.model_validate(value)
        return handler(value)


Message = Annotated[
    UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role"),
]
