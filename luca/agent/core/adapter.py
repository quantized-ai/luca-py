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

from luca.client.types import (
    AssistantMessage as LucaAssistantMessage,
    Tool as LucaTool,
    ToolCall as LucaToolCall,
)

from .exceptions import AgentError
from .models import (
    AssistantContentPart,
    PrivateProviderContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolSpec,
    URLCitation,
    WebFetchContent,
    WebPageContent,
    WebSearchContent,
)


def _url_citations(annotations) -> list[URLCitation]:
    return [
        URLCitation(
            url=annotation.url,
            title=annotation.title,
            start_index=annotation.start_index,
            end_index=annotation.end_index,
        )
        for annotation in annotations
    ]


def _web_page(part) -> WebPageContent:
    return WebPageContent(url=part.url, title=part.title, content=part.content, extras=part.extras)


def message_to_parts(
    message: LucaAssistantMessage,
) -> list[AssistantContentPart]:
    """Translate a client assistant message's blocks into agent message parts,
    ORDER PRESERVED — the client's positional contracts (a private result item
    beside its portable block, split cited spans after their merged text) ride
    through the session on part order alone. A refusal renders as plain text
    in V1. An unknown block type raises: the next client block type must not
    leak out of the session unnoticed."""
    parts: list[AssistantContentPart] = []
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
        elif block.type == "text":
            parts.append(TextContent(text=block.text, annotations=_url_citations(block.annotations)))
        elif block.type == "refusal":
            parts.append(TextContent(text=block.text))
        elif block.type == "private_provider":
            parts.append(PrivateProviderContent(format=block.format, data=block.data))
        elif block.type == "web_search":
            parts.append(
                WebSearchContent(
                    queries=block.queries,
                    results=None if block.results is None else [_web_page(r) for r in block.results],
                    extras=block.extras,
                ),
            )
        elif block.type == "web_fetch":
            parts.append(WebFetchContent(web_page=_web_page(block.web_page), extras=block.extras))
        elif isinstance(block, LucaToolCall):
            # isinstance, never `block.type == "tool_call"`: a provider-native
            # call is a ToolCall SUBCLASS whose `type` is its own wire literal
            # ("apply_patch_call"), and the default conversion must keep it
            # even with no adoption middleware installed. `as_generic()` folds
            # the native identity into `extras` losslessly (and is identity
            # for the ordinary base-class call), so the stored part carries
            # everything the client parsed.
            generic = block.as_generic()
            parts.append(
                ToolCall(
                    id=generic.id,
                    name=generic.name,
                    arguments=generic.arguments,
                    extras=generic.extras,
                ),
            )
        else:
            raise AgentError(
                f"Cannot store client block of type {getattr(block, 'type', type(block).__name__)!r}; "
                "the adapter has no agent part for it."
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
