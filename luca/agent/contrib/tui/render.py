"""Pure text formatting for transcript cells. No Textual imports."""

from __future__ import annotations

from collections.abc import Iterable

from luca.agent.core.models import (
    ContentPart,
    ExecutionStatus,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
)

RESULT_MAX_LINES = 12
RESULT_MAX_CHARS = 2_000
ARG_VALUE_MAX_CHARS = 80

STATUS_LABELS: dict[ExecutionStatus, str] = {
    ExecutionStatus.PENDING: "pending",
    ExecutionStatus.RUNNING: "running…",
    ExecutionStatus.COMPLETED: "done",
    ExecutionStatus.FAILED: "failed",
    ExecutionStatus.NOT_FOUND: "not found",
    ExecutionStatus.INVALID: "invalid arguments",
    ExecutionStatus.REJECTED: "denied",
    ExecutionStatus.CANCELLED: "cancelled",
    ExecutionStatus.INTERRUPTED: "interrupted",
    ExecutionStatus.TIMED_OUT: "timed out",
}


def _format_value(value: object) -> str:
    """One argument value bounded for the call line. Long or multiline strings
    collapse to a single-line preview with a char count; the durable session
    keeps the full value. Short values are shown verbatim (repr)."""
    if isinstance(value, str):
        collapsed = " ".join(value.split())
        if len(collapsed) > ARG_VALUE_MAX_CHARS:
            preview = collapsed[:ARG_VALUE_MAX_CHARS].rstrip() + "…"
            return f"{preview!r} ({len(value)} chars)"
        if collapsed != value:  # short, but had newlines/runs of whitespace
            return repr(collapsed)
        return repr(value)
    rendered = repr(value)
    if len(rendered) > ARG_VALUE_MAX_CHARS:
        return rendered[:ARG_VALUE_MAX_CHARS].rstrip() + "…"
    return rendered


def format_args(arguments: dict) -> str:
    return ", ".join(f"{key}={_format_value(value)}" for key, value in arguments.items())


def format_tool_call(call: ToolCall) -> str:
    return f"{call.name}({format_args(call.arguments)})"


def status_label(status: ExecutionStatus) -> str:
    return STATUS_LABELS.get(status, status.value)


REDACTED_REASONING_MARKER = "[reasoning withheld by the provider]"


def reasoning_transcript_text(part: ThinkingContent) -> str:
    """A thinking part as transcript text. A redacted block carries no
    readable body, so it gets a marker instead of rendering as nothing."""
    if part.redacted:
        return REDACTED_REASONING_MARKER
    return part.thinking if part.thinking.strip() else ""


def user_transcript_text(parts: Iterable[ContentPart]) -> str:
    """A user message's parts as transcript text: text verbatim, each image
    as a `[image: name]` placeholder line. Textual cannot draw images, and
    both the live and the replayed transcript render through here so they
    cannot drift."""
    lines: list[str] = []
    for part in parts:
        if isinstance(part, TextContent):
            lines.append(part.text)
        elif isinstance(part, ImageContent):
            label = (
                part.metadata.get("name")
                or part.source.media_type
                or "image"
            )
            lines.append(f"[image: {label}]")
    return "\n".join(lines)


def clip_text(
    text: str,
    max_lines: int = RESULT_MAX_LINES,
    max_chars: int = RESULT_MAX_CHARS,
) -> str:
    """Bound a tool result for cell display; the durable session keeps the
    full text."""
    clipped = text
    if len(clipped) > max_chars:
        clipped = clipped[:max_chars]
    lines = clipped.splitlines() or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        clipped = "\n".join(lines)
    if clipped != text:
        dropped = len(text) - len(clipped)
        clipped = f"{clipped}\n… (+{dropped} more characters)"
    return clipped
