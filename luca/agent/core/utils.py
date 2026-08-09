"""Read-only helpers over a durable `AgentSession`.

`pretty_print` renders ONE conversation as a plain-text transcript for reading
and debugging a saved session — the main conversation by default, or any other
row in `session.conversations` by id: one block per turn, the messages in path
order, and every tool call with its approval verdict and outcome drawn as a
tree under the assistant message that requested it.

It reads the DURABLE session — never the wire projection. A tool result is the
`ExecutionResult` the framework stored, a failure is the structured
`ToolExecutionError`, and approval comes from `approval_status` plus the last
`ApprovalDecision`; nothing is derived through `ConversationProjector`, so the
transcript shows what was recorded rather than what the model was told.

Bounded by design: text wraps to `WIDTH`, tool results clip to
`RESULT_MAX_LINES` / `RESULT_MAX_CHARS`, and reasoning renders as a marker
(the durable session keeps all of it). A node id missing from the entry store
prints as a marker instead of raising — a debugging view of a broken session
must still print.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime

from .models import (
    AgentSession,
    AnyEntry,
    ApprovalStatus,
    AssistantMessage,
    CancelRequested,
    ChildConversation,
    CompactionEntry,
    ContentPart,
    ExecutionStatus,
    ImageContent,
    PrunedEntry,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    Usage,
    UserMessage,
)

WIDTH = 64
RULE = "─" * WIDTH
INDENT = "  "

RESULT_MAX_LINES = 12
RESULT_MAX_CHARS = 1_000
ARG_MAX_CHARS = 60

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

REASONING_MARKER = "[thinking hidden]"
REDACTED_REASONING_MARKER = "[thinking redacted]"

OUTCOME_MARKS: dict[TurnOutcome, str] = {
    TurnOutcome.COMPLETED: "✓",
    TurnOutcome.CANCELLED: "⊘",
    TurnOutcome.TIMED_OUT: "✗",
    TurnOutcome.ERRORED: "✗",
}

STATUS_LABELS: dict[ExecutionStatus, str] = {
    ExecutionStatus.RECEIVED: "RECEIVED",
    ExecutionStatus.PENDING: "PENDING",
    ExecutionStatus.RUNNING: "RUNNING",
    ExecutionStatus.AWAITING_RESULT: "AWAITING RESULT",
    ExecutionStatus.COMPLETED: "OK",
    ExecutionStatus.FAILED: "FAILED",
    ExecutionStatus.NOT_FOUND: "NOT FOUND",
    ExecutionStatus.INVALID: "INVALID",
    ExecutionStatus.REJECTED: "DENIED",
    ExecutionStatus.REFUSED: "REFUSED",
    ExecutionStatus.CANCELLED: "CANCELLED",
    ExecutionStatus.INTERRUPTED: "INTERRUPTED",
    ExecutionStatus.TIMED_OUT: "TIMED OUT",
}

APPROVAL_LABELS: dict[ApprovalStatus, str] = {
    ApprovalStatus.ALLOWED: "ALLOWED",
    ApprovalStatus.REJECTED: "DENIED",
    ApprovalStatus.PENDING: "AWAITING APPROVAL",
}

# `ApprovalDecision.metadata["via"]` is the provenance convention the contrib
# registries write. An unrecognized value prints verbatim.
VIA_PHRASES: dict[str, str] = {
    "rule": "permission rule",
    "mode": "permission mode",
    "policy": "permission policy",
    "user": "user",
}


def pretty_print(session: AgentSession, conversation_id: str | None = None) -> str:
    """One conversation as a readable text transcript (`None` = the main one).

    The status line is DERIVED here, like everywhere else — nothing on the
    session stores one. `depth` appears only for a subagent conversation, so
    the ordinary transcript is unchanged."""
    conversation = session.conversations[conversation_id or session.main_conversation_id]
    entries = [_resolve(node_id, session.entries) for node_id in conversation.nodes]
    executions = {entry.tool_call_id: entry for entry in entries if isinstance(entry, ToolExecution)}
    usages = session.usages.get(conversation.id, {})
    blocks = _split_turns(entries)

    status = session.get_conversation_status(conversation.id).status
    header = f"Conversation {conversation.id} · {status.value} · {_plural(_count(entries, TurnStart), 'turn')}"
    if conversation.depth:
        header += f" · depth {conversation.depth}"
    lines = [
        f"LUCA SESSION {session.id}",
        header,
        f"Default: {_model_label(session)}",
        RULE,
    ]
    turn_number = 0
    for block in blocks:
        if any(isinstance(entry, TurnStart) for entry in block):
            turn_number += 1
        lines += ["", *_turn_lines(block, turn_number, executions, usages), "", RULE]
    lines.append(
        f"TOTAL · {_plural(_count(entries, TurnStart), 'turn')} · "
        f"{_plural(_count(entries, AssistantMessage), 'model call')} · "
        f"{_plural(len(executions), 'tool call')} · "
        f"{_plural(_tokens(entries, usages), 'token')}",
    )
    return "\n".join(lines)


# ── session-level helpers ───────────────────────────────────────────────────


def _resolve(node_id: str, entries: Mapping[str, AnyEntry]) -> AnyEntry | str:
    """The entry a path node points at, or the node id itself when the store
    has no such entry — a broken session still has to print."""
    entry = entries.get(node_id)
    return node_id if entry is None else entry


def _model_label(session: AgentSession) -> str:
    config = session.session_config.llm_config
    label = f"{config.provider}/{config.model}"
    if config.reasoning:
        label = f"{label} · reasoning {config.reasoning}"
    return label


def _split_turns(entries: Sequence[AnyEntry | str]) -> list[list[AnyEntry | str]]:
    """The path cut into one block per turn bracket. Entries recorded before a
    `TurnStart` (the user message `post_message` appended) lead the block they
    opened; a trailing block with no bracket is the not-yet-run tail."""
    blocks: list[list[AnyEntry | str]] = []
    current: list[AnyEntry | str] = []
    for entry in entries:
        if isinstance(entry, TurnStart) and _count(current, TurnStart):
            blocks.append(current)
            current = []
        current.append(entry)
        if isinstance(entry, TurnFinish):
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _count(entries: Iterable[AnyEntry | str], entry_type: type) -> int:
    return sum(1 for entry in entries if isinstance(entry, entry_type))


def _tokens(entries: Iterable[AnyEntry | str], usages: Mapping[str, Usage]) -> int:
    """Provider-reported total over the entries of this conversation only —
    usage is conversation-scoped accessory data, never an entry fact."""
    return sum(
        usage.total_tokens
        for entry in entries
        if not isinstance(entry, str) and (usage := usages.get(entry.id)) is not None
    )


# ── turn block ──────────────────────────────────────────────────────────────


def _turn_lines(
    block: Sequence[AnyEntry | str],
    number: int,
    executions: Mapping[str, ToolExecution],
    usages: Mapping[str, Usage],
) -> list[str]:
    start = next((e for e in block if isinstance(e, TurnStart)), None)
    finish = next((e for e in block if isinstance(e, TurnFinish)), None)
    first = next((e for e in block if not isinstance(e, str)), None)
    stamp = _timestamp(start or first)
    header = f"TURN {number} · {stamp}" if start is not None else f"NEXT TURN · {stamp}"

    sections: list[list[str]] = []
    step = 0
    for entry in block:
        if isinstance(entry, AssistantMessage):
            step += 1
        section = _entry_lines(entry, step, executions)
        if section:
            sections.append(section)

    lines = [header]
    for index, section in enumerate(sections):
        lines += section if index == 0 else ["", *section]
    if finish is not None or start is not None:
        lines += ["", _turn_footer(block, finish, usages)]
    return lines


def _turn_footer(
    block: Sequence[AnyEntry | str],
    finish: TurnFinish | None,
    usages: Mapping[str, Usage],
) -> str:
    stop_reasons = [entry.stop_reason for entry in block if isinstance(entry, AssistantMessage)]
    parts = ["⋯ open"] if finish is None else [f"{OUTCOME_MARKS[finish.outcome]} {finish.outcome.value}"]
    if stop_reasons:
        parts.append(stop_reasons[-1])
    parts.append(_plural(_tokens(block, usages), "token"))
    if finish is not None and finish.error:
        parts.append(_one_line(finish.error))
    return " · ".join(parts)


# ── entries ─────────────────────────────────────────────────────────────────


def _entry_lines(
    entry: AnyEntry | str,
    step: int,
    executions: Mapping[str, ToolExecution],
) -> list[str]:
    """One entry as a titled section, or `[]` for the entries a transcript
    renders elsewhere (the turn markers, and the tool executions drawn under
    the assistant message that requested them)."""
    if isinstance(entry, str):
        return [f"[missing entry {entry}]"]
    if isinstance(entry, UserMessage):
        return ["User", *_wrap(_content_text(entry.parts))]
    if isinstance(entry, AssistantMessage):
        return _assistant_lines(entry, step, executions)
    if isinstance(entry, CompactionEntry):
        replaced = _plural(len(entry.compacted_nodes or []), "entry", "entries")
        header = f"Compaction · replaced {replaced}"
        text = _content_text(entry.parts or [])
        return [header, *_wrap(text)] if text else [header]
    if isinstance(entry, ChildConversation):
        header = f"Subagent · {entry.conversation_id}"
        if entry.execution_result is None:
            return [f"{header} · unresolved"]
        text = _content_text(entry.execution_result.content)
        return [header, *[INDENT + line for line in _clip(text)]]
    if isinstance(entry, PrunedEntry):
        return [
            f"Pruned {entry.pruned_entry_type} {entry.pruned_entry_id}",
            *_wrap(_content_text(entry.content)),
        ]
    if isinstance(entry, CancelRequested):
        line = f"Cancel requested · {entry.outcome.value}"
        return [f"{line} · {_one_line(entry.error)}" if entry.error else line]
    return []


def _assistant_lines(
    entry: AssistantMessage,
    step: int,
    executions: Mapping[str, ToolExecution],
) -> list[str]:
    header = f"Assistant · step {step} · {entry.llm_config.provider}/{entry.llm_config.model}"
    chunks: list[list[str]] = []
    calls = [part for part in entry.parts if isinstance(part, ToolCall)]
    for part in entry.parts:
        if isinstance(part, ThinkingContent):
            marker = REDACTED_REASONING_MARKER if part.redacted else REASONING_MARKER
            chunks.append([INDENT + marker])
        elif isinstance(part, TextContent):
            chunks.append(_wrap(part.text))
        elif isinstance(part, ToolCall) and part is calls[0]:
            # the whole tree lands where the first call is
            chunks.append(_tools_lines(calls, executions))
    lines = [header]
    for index, chunk in enumerate(chunks):
        lines += chunk if index == 0 else ["", *chunk]
    return lines


def _content_text(parts: Iterable[ContentPart]) -> str:
    """Content parts as transcript text: text verbatim, an image as a labelled
    placeholder (the transcript cannot draw one)."""
    rendered: list[str] = []
    for part in parts:
        if isinstance(part, TextContent):
            rendered.append(part.text)
        elif isinstance(part, ImageContent):
            label = part.metadata.get("name") or part.source.media_type or "image"
            rendered.append(f"[image: {label}]")
    return "\n".join(rendered)


# ── tool tree ───────────────────────────────────────────────────────────────


def _tools_lines(
    calls: Sequence[ToolCall],
    executions: Mapping[str, ToolExecution],
) -> list[str]:
    lines = [INDENT + "Tools"]
    for index, call in enumerate(calls):
        last = index == len(calls) - 1
        branch = INDENT + ("└─ " if last else "├─ ")
        stem = INDENT + ("   " if last else "│  ")
        lines += _call_lines(call, branch, stem)
        lines += _execution_lines(executions.get(call.id), stem)
        if not last:
            lines.append(INDENT + "│")
    return lines


def _call_lines(call: ToolCall, branch: str, stem: str) -> list[str]:
    """The call signature on one line, or one argument per line when the
    signature outgrows the transcript width."""
    arguments = [f"{name}={_format_argument(value)}" for name, value in call.arguments.items()]
    signature = f"{call.name}({', '.join(arguments)})"
    if len(signature) <= WIDTH:
        return [branch + signature]
    lines = [f"{branch}{call.name}("]
    for index, argument in enumerate(arguments):
        comma = "" if index == len(arguments) - 1 else ","
        lines.append(f"{stem}  {argument}{comma}")
    lines.append(stem + ")")
    return lines


def _format_argument(value: object) -> str:
    """One argument value as JSON, clipped for display; the durable session
    keeps the full value."""
    rendered = json.dumps(value, ensure_ascii=False, default=str)
    if len(rendered) > ARG_MAX_CHARS:
        return rendered[: ARG_MAX_CHARS - 1].rstrip() + "…"
    return rendered


def _execution_lines(execution: ToolExecution | None, stem: str) -> list[str]:
    """The approval verdict and the outcome under one call. A DENIED approval
    IS the outcome — its `REJECTED` status adds nothing and is not repeated."""
    if execution is None:
        return [f"{stem}└─ NO EXECUTION RECORDED"]
    approval = _approval_line(execution)
    outcome = (
        None if execution.status == ExecutionStatus.REJECTED and approval is not None else _outcome_line(execution)
    )
    lines: list[str] = []
    if approval is not None:
        lines.append(f"{stem}{'└─ ' if outcome is None else '├─ '}{approval}")
    if outcome is not None:
        lines.append(f"{stem}└─ {outcome}")
        lines += [f"{stem}   {line}" for line in _outcome_body(execution)]
    return lines


def _approval_line(execution: ToolExecution) -> str | None:
    """`approval_status` is the current state; the audit log's last entry only
    supplies the provenance phrase. `None` means no policy ever saw the call."""
    if execution.approval_status is None:
        return None
    label = APPROVAL_LABELS[execution.approval_status]
    decisions = execution.approval_decisions
    via = (decisions[-1].metadata or {}).get("via") if decisions else None
    if via is None:
        return label
    phrase = VIA_PHRASES.get(via, str(via))
    if execution.approval_status == ApprovalStatus.REJECTED:
        return f"{label} · approval denied by {phrase}"
    return f"{label} · {phrase}"


def _outcome_line(execution: ToolExecution) -> str:
    """The outcome and how long the call was OUTSTANDING — `duration_ms` is
    total time (the approval wait and any parked time included), not body
    wall-clock, which is now per-attempt. A call dispatched more than once says
    so, because the count is the only visible trace of a deferral."""
    label = STATUS_LABELS[execution.status]
    result = execution.result
    if execution.status == ExecutionStatus.COMPLETED and result is not None:
        label = "ERROR" if result.is_error else "OK"
    if len(execution.attempts) > 1:
        label = f"{label} · {len(execution.attempts)} attempts"
    duration = execution.duration_ms
    return label if duration is None else f"{label} · {duration:,} ms"


def _outcome_body(execution: ToolExecution) -> list[str]:
    """What the outcome carries: the stored result, or the structured error."""
    if execution.result is not None:
        return _clip(_content_text(execution.result.content))
    if execution.error is not None:
        text = f"{execution.error.error_type}: {execution.error.error_message}"
        if execution.error.details:
            text += "\n" + json.dumps(execution.error.details, default=str)
        return _clip(text)
    return []


# ── text ────────────────────────────────────────────────────────────────────


def _wrap(text: str, indent: str = INDENT) -> list[str]:
    """Message text at the transcript width. Hard line breaks are kept — lists
    and code stay readable — and only over-long lines are wrapped."""
    width = WIDTH - len(indent)
    lines: list[str] = []
    for line in text.strip().split("\n"):
        wrapped = textwrap.wrap(line, width) if line.strip() else []
        lines += [indent + chunk for chunk in wrapped] or [""]
    return lines


def _clip(text: str) -> list[str]:
    """Tool output bounded for display: lines truncated to the width, then the
    whole block to `RESULT_MAX_LINES` / `RESULT_MAX_CHARS`. Never wrapped —
    tool output is structured, and rewrapping it destroys the structure."""
    text = text.strip()
    width = WIDTH - len(INDENT) * 2
    rendered: list[str] = []
    shown = 0
    for line in text.split("\n")[:RESULT_MAX_LINES]:
        if shown >= RESULT_MAX_CHARS:
            break
        line = line[: RESULT_MAX_CHARS - shown]
        shown += len(line) + 1  # the line plus the newline that followed it
        rendered.append(f"{line[: width - 1]}…" if len(line) > width else line)
    shown = min(shown, len(text))
    if shown < len(text):
        rendered.append(f"… (+{_plural(len(text) - shown, 'more character')})")
    return rendered


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _timestamp(entry: AnyEntry | None) -> str:
    if entry is None or entry.created_at is None:
        return "—"
    return datetime.fromtimestamp(entry.created_at / 1000).strftime(TIMESTAMP_FORMAT)


def _plural(count: int, noun: str, plural: str | None = None) -> str:
    word = noun if count == 1 else (plural or f"{noun}s")
    return f"{count:,} {word}"
