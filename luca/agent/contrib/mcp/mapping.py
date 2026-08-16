"""MCP wire shapes to luca core shapes. Pure functions, no I/O, no state.

Every decision that needs an opinion about the framework lives here, so the
transport below stays about bytes and the registry above stays about the
contract.

TOOL NAMES. A tool is advertised as `mcp__<label>__<tool>`. The prefix is not
decoration: `ProxyToolRegistry` raises on duplicate names across registries and
does NOT use `namespace` to disambiguate, so it is the only thing keeping two
servers that both expose `search` apart. Names are sanitized and length-capped
because providers reject the alternative.

The name is never parsed back to find the server. The previous attempt split on
`__`, which breaks for a server whose tool is literally called `a__b` and for
any name the cap truncated. Instead the identity is written into
`ToolSpec.metadata`, which contract rule 12 permits because it is a pure
function of the tool definition, and which gives `prepare()` a resolution path
that needs no catalog at all. That is what makes a cold cross-process resume
dispatch correctly.

CONTENT. Core carries text, image and file, and the projector raises on
anything else, so audio and unrecognized embedded resources are flattened to
text with the original block preserved in `ExecutionResult.metadata`. Nothing
is silently dropped, and nothing that reaches the projector can crash it.
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
    MediaURL,
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

    When the composed name is too long the TOOL segment is truncated and a
    digest of the full name is appended, so two long names that share a prefix
    stay distinct. The digest covers the untruncated input, which is what makes
    the mapping injective in practice.
    """
    composed = f"{PREFIX}{SEPARATOR}{_UNSAFE.sub('_', label)}{SEPARATOR}{_UNSAFE.sub('_', tool_name)}"
    if len(composed) <= MAX_NAME_LENGTH:
        return composed
    digest = hashlib.sha256(f"{label}{SEPARATOR}{tool_name}".encode()).hexdigest()[:6]
    return f"{composed[: MAX_NAME_LENGTH - len(digest) - 1]}_{digest}"


def to_tool_spec(label: str, tool: mcp_types.Tool, *, timeout_in_ms: int | None = None) -> ToolSpec:
    """One MCP tool as a `ToolSpec`.

    `tool_kind` stays `OTHER` on purpose. Mapping `annotations.readOnlyHint` to
    `ToolKind.READ` would let a `tool_kind: read` allow-rule auto-approve a
    remote tool on the server's own unverified claim, so the annotations are
    carried for display and kept away from the decision.

    Everything in `metadata` is a pure function of the tool definition, per
    contract rule 12. Nothing that varies per call or per listing goes in it:
    the spec is hashed into `session.tool_specs`, so a volatile field would
    mint a fresh stored row on every single call.
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
                # by_alias, so what is stored is the wire shape the server sent
                # rather than a luca-invented spelling of it.
                "annotations": annotations.model_dump(mode="json", by_alias=True, exclude_none=True)
                if annotations
                else {},
            }
        },
    )


def spec_identity(spec: ToolSpec) -> tuple[str, str] | None:
    """The (server label, remote tool name) a spec came from, or None when the
    spec is not ours. The inverse of what `to_tool_spec` wrote, and the reason
    `prepare()` never has to parse a name."""
    entry = (spec.metadata or {}).get("mcp")
    if not isinstance(entry, dict):
        return None
    label, tool = entry.get("server"), entry.get("tool")
    if isinstance(label, str) and isinstance(tool, str):
        return label, tool
    return None


def to_execution_result(result: mcp_types.CallToolResult) -> ExecutionResult:
    """One `tools/call` result as an `ExecutionResult`.

    `isError` becomes `is_error` and nothing more: the execution still
    COMPLETES, because in this framework `is_error` is the tool's own verdict
    on its work rather than a failure of the call.
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
        # 2026-07-28 loosened `structuredContent` to any JSON value, but
        # `ExecutionResult.structured_content` is a dict. Wrapping keeps the
        # payload rather than discarding it.
        structured = {"result": structured}

    if not content:
        # A result with no content blocks shows the model nothing. When there
        # is structured output, render it; otherwise an empty string still
        # beats a content list the projector would have to special-case.
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
            # Core has no audio part and the projector raises on anything it
            # does not know, so this is described rather than carried.
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
    """An embedded resource, mapped by what it actually holds.

    A PDF becomes a real `FileContent`, because luca has a file part and models
    that read documents will take it. An image blob becomes an image. Anything
    else keeps its bytes in metadata and shows the model a description, since
    inventing a media type it cannot read helps nobody.
    """
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


def resource_link_as_file(block: mcp_types.ResourceLink) -> FileContent:
    """A resource link as a file part, for the callers that can fetch a URL.
    Kept separate from `_content_block` because a tool result must not smuggle
    a remote URL into the model's context as if it were content."""
    return FileContent(source=MediaURL(url=block.uri, media_type=block.mime_type), name=block.name)


def _basename(uri: str) -> str | None:
    tail = uri.rstrip("/").rsplit("/", 1)[-1]
    return tail or None


def approval_context(label: str, tool_name: str) -> dict:
    """What the shared `PermissionStrategy` decides on for one MCP call.

    Uses the existing `(permission, resource)` vocabulary rather than inventing
    a per-server trust knob, which means today's `permissions.rules` block
    already expresses `{"permission": "mcp", "resource": "github/*"}` with the
    glob semantics it already has, and "always allow" from the prompt writes
    the same shape back.

    Built as plain dicts so this module stays importable without the
    `resource_permissions` package; the shape is exactly what
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
                "metadata": {"server": label, "tool": tool_name},
            }
        ]
    }
