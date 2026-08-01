"""Pure text formatting and session reads for transcript cells. No Textual
imports, so everything here is unit-testable without a running app."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from luca.agent.core.models import (
    AgentSession,
    ChildConversation,
    CompactionEntry,
    ContentPart,
    ExecutionStatus,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
)

RESULT_MAX_LINES = 12
RESULT_MAX_CHARS = 2_000
ARG_VALUE_MAX_CHARS = 80
TASK_MAX_LINES = 6
TASK_MAX_CHARS = 400

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
            label = part.metadata.get("name") or part.source.media_type or "image"
            lines.append(f"[image: {label}]")
    return "\n".join(lines)


def compaction_transcript_text(entry: CompactionEntry) -> str:
    """A compaction's summary as transcript text — the same `ContentPart`
    shape a user message carries, so images render as the same placeholder.
    Empty for an entry that produced nothing (scheduled, failed, or a no-op),
    which is exactly when the wire projection emits nothing either."""
    return user_transcript_text(entry.parts or [])


def compaction_subtitle(entry: CompactionEntry) -> str:
    replaced = len(entry.compacted_nodes or [])
    return f"replaced {replaced} {'entry' if replaced == 1 else 'entries'}"


# ── subagents ─────────────────────────────────────────────────────────────────


def is_runtime_plumbing(execution: ToolExecution) -> bool:
    """Does this execution render as something OTHER than a tool cell?

    Two kinds, both matched by DECLARATION rather than by name — the same
    contract the runner matches on, so an application that ships its own spawn
    and result tools renders exactly like the built-in pair:

    - a PRIVATE tool is invoked by the runtime and never advertised to the
      model (the subagent result tool is one). Showing plumbing as a tool call
      the model appears to have made is simply wrong.
    - a SPAWN tool already has a richer rendering: the panel its child
      conversation gets. A tool cell beside it says the same thing twice.

    An unresolved spec answers False: with nothing declared there is nothing to
    match, and an ordinary tool cell is the honest fallback."""
    spec = execution.tool_spec
    if spec is None:
        return False
    if spec.is_private:
        return True
    schema = spec.output_schema
    return isinstance(schema, dict) and "is_subagent_spawn" in (schema.get("properties") or {})


def subagent_task(session: AgentSession, entry: ChildConversation) -> tuple[str, str]:
    """One subagent's `(description, prompt)`, read from the spawn execution
    that created it.

    Both live on the spawn result's `structured_content` — the handshake
    payload, which never reaches the model and so costs the conversation
    nothing. The raw call's arguments are the fallback for the shape where a
    tool spawned without echoing its own payload, and the conversation id is
    the last resort: a panel with no label is worse than a panel labelled by
    id."""
    execution = session.entries.get(entry.tool_execution_id)
    sources: list[dict] = []
    if isinstance(execution, ToolExecution):
        if execution.result is not None and execution.result.structured_content:
            sources.append(execution.result.structured_content)
        sources.append(execution.raw_tool_call.arguments)
    for source in sources:
        description = source.get("description")
        if description:
            return str(description), str(source.get("prompt") or "")
    return f"subagent {entry.conversation_id[:8]}", ""


def child_links(session: AgentSession) -> Iterator[tuple[str, ChildConversation]]:
    """Every `ChildConversation` in the session, as `(parent_id, entry)`.

    Walks the CONVERSATIONS rather than the entry store, because the entry
    store alone cannot say which path a link hangs from — and a nested subagent
    (a link on another subagent's path) has to nest in the transcript too."""
    for parent_id, conversation in session.conversations.items():
        for node_id in conversation.nodes:
            entry = session.entries.get(node_id)
            if isinstance(entry, ChildConversation):
                yield parent_id, entry


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
