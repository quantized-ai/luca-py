"""Adapter: the two inbound/registry translations next to conversation
projection (which lives in `projection.py` as the `ConversationProjector`
strategy):

- `message_to_parts()` — a client `AssistantMessage`'s blocks rendered into
  the agent message parts the runner persists (the inbound direction).
- `tool_spec_to_luca_tool()` — a `ToolSpec` projected onto the wire
  `luca.client.Tool` definition (registry/request preparation).

Both are deliberately separate from `ConversationProjector`: response
conversion and tool-definition conversion are not conversation projection,
and there is no general bidirectional adapter object.
"""

from __future__ import annotations

from luca.client.types import AssistantMessage as LucaAssistantMessage, Tool as LucaTool

from .models import TextContent, ThinkingContent, ToolCall, ToolSpec


def message_to_parts(
    message: LucaAssistantMessage,
) -> list[TextContent | ThinkingContent | ToolCall]:
    """Translate a client assistant message's blocks into agent message parts.
    A refusal renders as plain text in V1."""
    parts: list[TextContent | ThinkingContent | ToolCall] = []
    for block in message.content:
        if block.type == "thinking":
            parts.append(
                ThinkingContent(
                    thinking=block.text,
                    id=block.id,
                    signature=block.signature,
                    redacted=block.redacted,
                ),
            )
        elif block.type in ("text", "refusal"):
            parts.append(TextContent(text=block.text))
        elif block.type == "tool_call":
            parts.append(
                ToolCall(id=block.id, name=block.name, arguments=block.arguments),
            )
    return parts


def tool_spec_to_luca_tool(spec: ToolSpec) -> LucaTool:
    """Project a `ToolSpec` onto the wire `luca.client.Tool` the model sees.

    `input_schema` is already a JSON Schema dict, which `luca.client.Tool`
    accepts verbatim as `parameters` — so the core hands the transport plain
    data and never touches a Python class. The remaining spec fields
    (`tool_kind`, `namespace`, `version`, `timeout_in_ms`, `metadata`) are
    framework/app-space classification the wire has no place for."""
    return LucaTool(
        name=spec.name,
        description=spec.description,
        parameters=spec.input_schema,
    )
