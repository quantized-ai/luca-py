"""MCP wire shapes to luca core shapes. Pure functions, no I/O, no state.

TOOL NAMES are `mcp__<label>__<tool>`. The prefix is load-bearing:
`ProxyToolRegistry` raises on duplicate names across registries and does not use
`namespace` to disambiguate, so it is the only thing keeping two servers that
both expose `search` apart. Sanitized and length-capped because providers reject
the alternative.

The name is never parsed back to find the server; the identity is written into
`ToolSpec.metadata` instead, which rule 12 permits because it is a pure function
of the tool definition, and which gives `prepare()` a resolution path needing no
catalog. Parsing would break on a tool literally called `a__b`, and on any name
the cap truncated.

CONTENT. Core carries text, image and file, and the projector raises on anything
else, so audio and unrecognized embedded resources are flattened to text with
the original block kept in `ExecutionResult.metadata`.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Final

import mcp_types

from luca.agent.core import (
    ExecutionResult,
    FileContent,
    ImageContent,
    MediaBase64,
    TextContent,
    ToolKind,
    ToolSpec,
)

PREFIX: Final = "mcp"
SEPARATOR: Final = "__"

# Anthropic and OpenAI both cap tool names at 64 characters, and a name that
# fails validation fails the whole request, not just the tool.
MAX_NAME_LENGTH: Final = 64

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def wire_name(label: str, tool_name: str) -> str:
    """The name the model sees for one server's tool.

    Over the cap, the tool segment is truncated and a digest of the UNTRUNCATED
    name appended, so two long names sharing a prefix stay distinct.
    """
    composed = f"{PREFIX}{SEPARATOR}{_UNSAFE.sub('_', label)}{SEPARATOR}{_UNSAFE.sub('_', tool_name)}"
    if len(composed) <= MAX_NAME_LENGTH:
        return composed
    digest = hashlib.sha256(f"{label}{SEPARATOR}{tool_name}".encode()).hexdigest()[:6]
    return f"{composed[: MAX_NAME_LENGTH - len(digest) - 1]}_{digest}"


def to_tool_spec(label: str, tool: mcp_types.Tool, *, timeout_in_ms: int | None = None) -> ToolSpec:
    """One MCP tool as a `ToolSpec`.

    `tool_kind` stays `OTHER`: mapping `annotations.readOnlyHint` to
    `ToolKind.READ` would let a `tool_kind: read` allow-rule auto-approve a
    remote tool on the server's own unverified claim.

    Nothing volatile goes in `metadata` (rule 12) — the spec is hashed into
    `session.tool_specs`, so a per-call field would mint a row every call.
    """
    annotations = tool.annotations
    return ToolSpec(
        name=wire_name(label, tool.name),
        title=tool.title or (annotations.title if annotations else None),
        description=tool.description or "",
        input_schema=dict(tool.input_schema) if tool.input_schema else {"type": "object", "properties": {}},
        output_schema=dict(tool.output_schema) if tool.output_schema else None,
        tool_kind=ToolKind.OTHER,
        namespace=f"{PREFIX}.{label}",
        timeout_in_ms=timeout_in_ms,
        metadata={
            "mcp": {
                "server": label,
                "tool": tool.name,
                # by_alias: store the wire shape the server sent.
                "annotations": annotations.model_dump(mode="json", by_alias=True, exclude_none=True)
                if annotations
                else {},
            }
        },
    )


def spec_identity(spec: ToolSpec) -> tuple[str, str] | None:
    """The (server label, remote tool name) a spec came from, or None when the
    spec is not ours."""
    entry = (spec.metadata or {}).get("mcp")
    if not isinstance(entry, dict):
        return None
    label, tool = entry.get("server"), entry.get("tool")
    if isinstance(label, str) and isinstance(tool, str):
        return label, tool
    return None


def to_execution_result(result: mcp_types.CallToolResult) -> ExecutionResult:
    """One `tools/call` result as an `ExecutionResult`.

    `isError` becomes `is_error` and the execution still COMPLETES: it is the
    tool's verdict on its own work, not a failure of the call.
    """
    content: list[Any] = []
    unmapped: list[dict] = []
    for block in result.content:
        mapped, raw = _content_block(block)
        content.append(mapped)
        if raw is not None:
            unmapped.append(raw)

    structured = result.structured_content
    if structured is not None and not isinstance(structured, dict):
        # 2026-07-28 allows any JSON value; the core field is a dict.
        structured = {"result": structured}

    if not content:
        # No content blocks shows the model nothing, so render the structured
        # output when there is any.
        text = json.dumps(structured, separators=(",", ":")) if structured is not None else ""
        content = [TextContent(text=text)]

    metadata: dict[str, Any] = {}
    if result.meta:
        metadata["mcp_meta"] = dict(result.meta)
    if unmapped:
        metadata["mcp_unmapped_content"] = unmapped
    return ExecutionResult(
        content=content,
        structured_content=structured,
        metadata=metadata,
        is_error=bool(result.is_error),
    )


def _content_block(block: mcp_types.ContentBlock) -> tuple[Any, dict | None]:
    """One MCP content block as a core `ContentPart`, plus the original block
    when the mapping was lossy and the raw payload is worth keeping."""
    match block:
        case mcp_types.TextContent():
            return TextContent(text=block.text), None
        case mcp_types.ImageContent():
            return ImageContent(source=MediaBase64(data=block.data, media_type=block.mime_type)), None
        case mcp_types.AudioContent():
            # Core has no audio part, so describe rather than carry.
            return (
                TextContent(text=f"[audio content, {block.mime_type}, not rendered]"),
                block.model_dump(mode="json", by_alias=True, exclude_none=True),
            )
        case mcp_types.ResourceLink():
            described = f"[resource {block.uri}]"
            if block.description:
                described = f"{described} {block.description}"
            return TextContent(text=described), None
        case mcp_types.EmbeddedResource():
            return _embedded(block)
    return (
        TextContent(text="[unsupported MCP content block]"),
        block.model_dump(mode="json", by_alias=True, exclude_none=True) if hasattr(block, "model_dump") else None,
    )


def _embedded(block: mcp_types.EmbeddedResource) -> tuple[Any, dict | None]:
    """An embedded resource, mapped by what it actually holds. A PDF becomes a
    real `FileContent`; anything unreadable keeps its bytes in metadata."""
    resource = block.resource
    media_type = resource.mime_type or ""
    if isinstance(resource, mcp_types.TextResourceContents):
        return TextContent(text=resource.text), None
    if media_type.startswith("image/"):
        return ImageContent(source=MediaBase64(data=resource.blob, media_type=media_type)), None
    if media_type == "application/pdf":
        return FileContent(
            source=MediaBase64(data=resource.blob, media_type=media_type), name=_basename(resource.uri)
        ), None
    return (
        TextContent(text=f"[resource {resource.uri}, {media_type or 'unknown type'}, not rendered]"),
        block.model_dump(mode="json", by_alias=True, exclude_none=True),
    )


def _basename(uri: str) -> str | None:
    tail = uri.rstrip("/").rsplit("/", 1)[-1]
    return tail or None


def approval_context(label: str, tool_name: str) -> dict:
    """What the shared `PermissionStrategy` decides on for one MCP call.

    Reuses the existing `(permission, resource)` vocabulary, so
    `permissions.rules` already expresses `{"permission": "mcp", "resource":
    "github/*"}` with no new schema. Plain dicts, so this module stays
    importable without `resource_permissions`; the shape is exactly what
    `PermissionRequest.model_dump()` produces.
    """
    resource = f"{label}/{tool_name}"
    return {
        "requests": [
            {
                "resources": [{"permission": "mcp", "resource": resource}],
                "answer_options": [
                    {
                        "resource_permissions": [{"permission": "mcp", "resource": resource}],
                        "metadata": {"label": f"Allow {tool_name} from {label}"},
                    },
                    {
                        "resource_permissions": [{"permission": "mcp", "resource": f"{label}/*"}],
                        "metadata": {"label": f"Allow every tool from {label}"},
                    },
                ],
                # `preview` is what the approval prompt reads; without one it
                # asks you to approve `mcp__airtable__list_records`, which
                # names the wire format rather than the thing being approved.
                "metadata": {
                    "server": label,
                    "tool": tool_name,
                    "preview": f"Run {tool_name} on the {label} MCP server",
                },
            }
        ]
    }
