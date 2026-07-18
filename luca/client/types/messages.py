"""Per-role discriminated union: UserMessage | AssistantMessage | ToolMessage.

There is intentionally NO SystemMessage class and NO "system" role. The
system prompt is request-scoped on ChatCompletionRequest.system_message.

`AssistantMessage` carries self-describing terminal state (finish_reason,
provider_finish_reason, error_message, cancelled, usage, provider, model,
timestamp) so a serialized conversation reloads with full context.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from .content import (
    AudioBlock,
    FileBlock,
    ImageBlock,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
)


class _UsageHolder:
    """Lazy-import indirection for Usage so messages.py doesn't import completion.py
    at module-import time (which would create a cycle since completion.py imports
    messages.py)."""


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[Union[TextBlock, ImageBlock, AudioBlock, FileBlock]]
    name: str | None = None
    timestamp: int | None = None

    model_config = ConfigDict(extra="forbid")


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[Union[TextBlock, ThinkingBlock, ToolCall, RefusalBlock]] = Field(
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

    # Forward-declared as "Usage | None"; validated as Any here to avoid the
    # circular import (completion.py imports messages.py for the discriminated
    # union). The runtime type is the real Usage; pydantic handles it via the
    # forward-ref resolution done in types/__init__.py.
    usage: "Usage | None" = None
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
    content: str | list[Union[TextBlock, ImageBlock]]
    name: str | None = None
    is_error: bool = False
    timestamp: int | None = None

    model_config = ConfigDict(extra="forbid")


Message = Annotated[
    Union[UserMessage, AssistantMessage, ToolMessage],
    Field(discriminator="role"),
]
