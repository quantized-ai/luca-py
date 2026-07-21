"""Pure text formatting for transcript cells. No Textual imports."""

from __future__ import annotations

from luca.agent.core.models import (
    ExecutionStatus,
    ImageContent,
    TextContent,
    ToolCall,
)

RESULT_MAX_LINES = 12
RESULT_MAX_CHARS = 2_000

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


def format_args(arguments: dict) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in arguments.items())


def format_tool_call(call: ToolCall) -> str:
    return f"{call.name}({format_args(call.arguments)})"


def status_label(status: ExecutionStatus) -> str:
    return STATUS_LABELS.get(status, status.value)


def user_transcript_text(parts) -> str:
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
