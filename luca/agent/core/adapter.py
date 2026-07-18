"""Adapter: the two inbound/registry translations next to conversation
projection (which lives in `projection.py` as the `ConversationProjector`
strategy):

- `message_to_parts()` — a client `AssistantMessage`'s blocks rendered into
  the agent message parts the runner persists (the inbound direction).
- `tool_to_luca_tool()` — an agent `Tool` projected onto the wire
  `luca.client.Tool` definition (registry/request preparation).

Both are deliberately separate from `ConversationProjector`: response
conversion and tool-definition conversion are not conversation projection,
and there is no general bidirectional adapter object.
"""

from __future__ import annotations

from luca.client.types import AssistantMessage as LucaAssistantMessage
from luca.client.types import Tool as LucaTool

from .models import TextContent, ThinkingContent, ToolCall
from .tools import Tool


def message_to_parts(
    message: LucaAssistantMessage,
) -> list[TextContent | ThinkingContent | ToolCall]:
    """Translate a client assistant message's blocks into agent message parts.
    A refusal renders as plain text in V1."""
    parts: list[TextContent | ThinkingContent | ToolCall] = []
    for block in message.content:
        if block.type == "thinking":
            parts.append(ThinkingContent(thinking=block.text))
        elif block.type in ("text", "refusal"):
            parts.append(TextContent(text=block.text))
        elif block.type == "tool_call":
            parts.append(
                ToolCall(id=block.id, name=block.name, arguments=block.arguments),
            )
    return parts


def tool_to_luca_tool(tool: Tool) -> LucaTool:
    """Project an agent `Tool` onto the wire `luca.client.Tool` the model sees.

    `Tool.Args` (a Pydantic model class) is passed straight through as
    `parameters` — `luca.client.Tool` accepts a BaseModel and the transport
    normalizes it to JSON schema at send time."""
    return LucaTool(
        name=tool.name,
        description=tool.description,
        parameters=tool.Args,
    )
