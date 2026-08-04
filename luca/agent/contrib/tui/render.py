"""Pure derivations: agent entries/events → the `state` view-models.

No Textual imports — everything here is unit-testable without an app. The
live `AgentApp` and the resume replay both go through these functions, so the
two cannot drift.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

from luca.agent.core import declares_spawn
from luca.agent.core.models import (
    AgentSession,
    AssistantMessage,
    ChildConversation,
    ContentPart,
    ExecutionStatus,
    ImageContent,
    TextContent,
    ToolExecution,
    UserMessage,
)

from . import state as vm
from .format import fmt_duration

OUTPUT_HEAD_LINES = 8
ARG_VALUE_MAX_CHARS = 80
SUMMARY_MAX_CHARS = 84

# ── user content ──────────────────────────────────────────────────────────────


def user_transcript_text(parts: Iterable[ContentPart]) -> str:
    """A user message's parts as transcript text: text verbatim, each image
    as a `[image: name]` placeholder line."""
    lines: list[str] = []
    for part in parts:
        if isinstance(part, TextContent):
            lines.append(part.text)
        elif isinstance(part, ImageContent):
            label = part.metadata.get("name") or part.source.media_type or "image"
            lines.append(f"[image: {label}]")
    return "\n".join(lines)


# ── tool calls ────────────────────────────────────────────────────────────────

# The one argument the design shows beside each tool name.
_PRIMARY_ARGS = (
    "path",
    "file_path",
    "command",
    "pattern",
    "name",
    "prompt",
    "query",
)


def tool_arg(execution: ToolExecution) -> str:
    """The single argument string on the call header: the tool's primary
    argument when it has one, a compact `k=v` line otherwise."""
    arguments = execution.raw_tool_call.arguments
    for key in _PRIMARY_ARGS:
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return _collapse(value)
    if not arguments:
        return ""
    return ", ".join(f"{key}={_collapse(_plain(value))}" for key, value in arguments.items())


def _plain(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _collapse(value: str, limit: int = ARG_VALUE_MAX_CHARS) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) > limit:
        return collapsed[:limit].rstrip() + "…"
    return collapsed


_STATUS_MAP: dict[ExecutionStatus, vm.ToolStatus] = {
    ExecutionStatus.RECEIVED: "pending",
    ExecutionStatus.PENDING: "pending",
    ExecutionStatus.RUNNING: "running",
    ExecutionStatus.COMPLETED: "ok",
    ExecutionStatus.REJECTED: "denied",
    ExecutionStatus.FAILED: "error",
    ExecutionStatus.NOT_FOUND: "error",
    ExecutionStatus.INVALID: "error",
    ExecutionStatus.REFUSED: "error",
    ExecutionStatus.CANCELLED: "error",
    ExecutionStatus.INTERRUPTED: "error",
    ExecutionStatus.TIMED_OUT: "error",
}

_STATUS_WORD: dict[ExecutionStatus, str] = {
    ExecutionStatus.REJECTED: "denied",
    ExecutionStatus.FAILED: "failed",
    ExecutionStatus.NOT_FOUND: "not found",
    ExecutionStatus.INVALID: "invalid arguments",
    ExecutionStatus.REFUSED: "refused",
    ExecutionStatus.CANCELLED: "cancelled",
    ExecutionStatus.INTERRUPTED: "interrupted",
    ExecutionStatus.TIMED_OUT: "timed out",
}


def diff_stat(diff: str) -> vm.DiffStat | None:
    """`+N −M` from a unified diff (the shell tools put one in metadata)."""
    added = removed = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    if added == 0 and removed == 0:
        return None
    return vm.DiffStat(add=added or None, **{"del": removed or None})


def _duration_note(execution: ToolExecution) -> str | None:
    if execution.started_at is None or execution.ended_at is None:
        return None
    seconds = (execution.ended_at - execution.started_at) / 1000
    if seconds < 0.05:
        return None
    return fmt_duration(seconds) if seconds >= 1 else f"{seconds:.1f}s"


def tool_block(
    execution: ToolExecution,
    result_text: str | None = None,
    *,
    is_error: bool = False,
    auto_approved: bool = False,
    denied_by_user: bool = False,
) -> vm.ToolBlock:
    """One execution (with its projector-derived result text, if terminal) as
    a `ToolBlock`. Every automatic permission decision is stated: an
    auto-allowed gated call carries `approved by rule` on its result row, a
    denial says whether a rule or the user refused it."""
    name = execution.raw_tool_call.name
    status = _STATUS_MAP.get(execution.status, "pending")
    if status == "running":
        status = "pending"
    block = vm.ToolBlock(tool=name, arg=tool_arg(execution), status=status)

    if execution.status is ExecutionStatus.REJECTED:
        note = "by you" if denied_by_user else "by rule"
        return block.model_copy(update={"result": vm.ToolResult(summary="denied", note=note)})
    if execution.status in _STATUS_WORD:  # the non-REJECTED failures
        lines = (result_text or "").splitlines()
        summary = _STATUS_WORD[execution.status]
        if duration := _duration_note(execution):
            summary += f" · {duration}"
        return block.model_copy(update={"output": _output(lines, summary)})
    if execution.status is not ExecutionStatus.COMPLETED:
        return block  # pending / running — header only

    note = "approved by rule" if auto_approved else None
    metadata = execution.result.metadata if execution.result is not None else {}
    if isinstance(metadata.get("diff"), str):
        stat = diff_stat(metadata["diff"])
        return block.model_copy(update={"result": vm.ToolResult(stat=stat, note=note)})

    text = (result_text or "").rstrip("\n")
    lines = text.splitlines() or [""]
    if "exit" in metadata:
        exit_code = metadata["exit"] if metadata["exit"] is not None else "?"
        summary = f"[error]exit {exit_code}[/]" if is_error else f"exit {exit_code}"
        if duration := _duration_note(execution):
            summary += f" · {duration}"
        return block.model_copy(update={"output": _output(lines, summary)})
    if summary := _completed_summary(name, lines):
        return block.model_copy(update={"result": vm.ToolResult(summary=summary, note=note)})
    if len(lines) == 1:
        return block.model_copy(
            update={"result": vm.ToolResult(summary=_collapse(lines[0], SUMMARY_MAX_CHARS), note=note)}
        )
    return block.model_copy(update={"output": _output(lines, f"{len(lines)} lines")})


def _output(lines: list[str], summary: str) -> vm.ToolOutput:
    hidden = lines[OUTPUT_HEAD_LINES:]
    if hidden:
        summary += f" · {len(hidden)} lines hidden"
    return vm.ToolOutput(
        lines=lines[:OUTPUT_HEAD_LINES],
        hidden_lines=hidden,
        summary=summary,
        expand_hint=bool(hidden),
    )


def _completed_summary(tool: str, lines: list[str]) -> str | None:
    """The design's compact result rows for the read-only tools:
    `84 lines`, `6 matches · luca/cli.py, luca/events.py`, `12 files`."""
    if tool == "read" and len(lines) > 1:
        return f"{len(lines)} lines"
    if tool == "grep" and lines and lines[0].startswith("Found "):
        count = lines[0].removeprefix("Found ").split(" match")[0]
        files = [line.removesuffix(":") for line in lines[1:] if line.endswith(":")]
        suffix = f" · {', '.join(files[:3])}" if files else ""
        if len(files) > 3:
            suffix += ", …"
        return _collapse(f"{count} matches{suffix}", SUMMARY_MAX_CHARS)
    if tool == "glob" and len(lines) > 1:
        return f"{len(lines)} files"
    return None


# ── plan (todo) blocks ────────────────────────────────────────────────────────

_TODO_GLYPHS = {
    "completed": "done",
    "in_progress": "active",
    "pending": "pending",
    "cancelled": "done",
}


def plan_block(todos: list[dict]) -> vm.ListBlock:
    """The memory plugin's todo list as the design's plan block."""
    rows = [
        vm.ListRow(
            glyph=_TODO_GLYPHS.get(str(item.get("status", "pending")), "pending"), text=str(item.get("content", ""))
        )
        for item in todos
    ]
    done = sum(1 for item in todos if str(item.get("status")) == "completed")
    active = min(done + 1, len(todos))
    return vm.ListBlock(label=f"plan · {active} of {len(todos)}", rows=rows)


def is_plan_update(execution: ToolExecution) -> bool:
    return execution.raw_tool_call.name == "update_todos"


def plan_from_execution(execution: ToolExecution) -> vm.ListBlock | None:
    todos = execution.raw_tool_call.arguments.get("todos")
    if not isinstance(todos, list):
        return None
    return plan_block([item for item in todos if isinstance(item, dict)])


# ── subagents ─────────────────────────────────────────────────────────────────


def is_runtime_plumbing(execution: ToolExecution) -> bool:
    """Executions that render as something OTHER than a tool row: a private
    (runtime-invoked) tool renders nothing, a spawn renders as its task
    block. Matched by DECLARATION, exactly like the runner."""
    spec = execution.tool_spec
    if spec is None:
        return False
    return spec.is_private or declares_spawn(spec)


def subagent_task(session: AgentSession, entry: ChildConversation) -> tuple[str, str]:
    """One subagent's `(description, prompt)`, read from the spawn execution
    that created it."""
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
    """Every `ChildConversation` in the session, as `(parent_id, entry)`."""
    for parent_id, conversation in session.conversations.items():
        for node_id in conversation.nodes:
            entry = session.entries.get(node_id)
            if isinstance(entry, ChildConversation):
                yield parent_id, entry


# ── session previews (1i) ─────────────────────────────────────────────────────


def preview_rows(session: AgentSession, count: int = 3) -> list[str]:
    """The final transcript rows of a session, compact, span-tagged — the
    sessions screen's `preview · last turn` block, so resume is never blind."""
    nodes = session.conversations[session.main_conversation_id].nodes
    rows: list[str] = []
    for node_id in reversed(nodes):
        if len(rows) >= count:
            break
        entry = session.entries.get(node_id)
        if isinstance(entry, ToolExecution):
            if entry.raw_tool_call.name == "update_todos":
                todos = entry.raw_tool_call.arguments.get("todos") or []
                active = next(
                    (t.get("content") for t in todos if isinstance(t, dict) and t.get("status") == "in_progress"),
                    None,
                )
                if active:
                    rows.append(f"[accent]◉[/] {active}")
                    continue
            rows.append(f"▸ {entry.raw_tool_call.name.ljust(6)} {tool_arg(entry)}")
        elif isinstance(entry, UserMessage):
            first = user_transcript_text(entry.parts).strip().splitlines()
            if first:
                rows.append(f"[accent]›[/] {first[0]}")
        elif isinstance(entry, AssistantMessage):
            for part in entry.parts:
                if isinstance(part, TextContent) and part.text.strip():
                    rows.append(part.text.strip().splitlines()[0])
                    break
    return list(reversed(rows))
