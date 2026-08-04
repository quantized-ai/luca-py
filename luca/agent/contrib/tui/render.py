"""Pure derivations: agent entries/events → the `state` view-models.

No Textual imports — everything here is unit-testable without an app. The
live `AgentApp` and the resume replay both go through these functions, so the
two cannot drift.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence

from luca.agent.core import declares_spawn
from luca.agent.core.models import (
    NONTERMINAL_STATUSES,
    AgentSession,
    AssistantMessage,
    ChildConversation,
    CompactionEntry,
    ContentPart,
    Entry,
    ExecutionStatus,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolExecution,
    TurnFinish,
    TurnOutcome,
    UserMessage,
)

from . import state as vm
from .format import fmt_bytes, fmt_duration, fmt_tokens, home_path

OUTPUT_HEAD_LINES = 8
ARG_VALUE_MAX_CHARS = 80
SUMMARY_MAX_CHARS = 84

# ── user content ──────────────────────────────────────────────────────────────


def mention_of(part: ContentPart) -> dict | None:
    """The `@`-mention annotation on a part, or None for ordinary content.
    Read best-effort: a part written by an older build, or by an application
    that annotates differently, must not break the transcript."""
    mention = (getattr(part, "metadata", None) or {}).get("mention")
    return mention if isinstance(mention, dict) else None


def user_transcript_text(parts: Iterable[ContentPart]) -> str:
    """A user message's parts as transcript text: text verbatim, each image
    as a `[image: name]` placeholder line.

    `@`-mention parts are skipped — an inlined file belongs in its own read
    row, not dumped into the user's own words."""
    lines: list[str] = []
    for part in parts:
        if mention_of(part) is not None:
            continue
        if isinstance(part, TextContent):
            lines.append(part.text)
        elif isinstance(part, ImageContent):
            label = part.metadata.get("name") or part.source.media_type or "image"
            lines.append(f"[image: {label}]")
    return "\n".join(lines)


def mention_blocks(parts: Iterable[ContentPart]) -> list[vm.ToolBlock]:
    """The `▸ read` rows under a user turn, one per `@`-mention part.

    They render in the tool idiom but are NOT tool calls: no execution backs
    them, so they never carry an approval note. The path is shown home-
    contracted for reading; `metadata["mention"]["path"]` stays absolute and
    is the source of truth."""
    blocks: list[vm.ToolBlock] = []
    for part in parts:
        mention = mention_of(part)
        if mention is None:
            continue
        blocks.append(
            vm.ToolBlock(
                tool="read",
                arg=home_path(str(mention.get("path", ""))),
                status="ok" if mention.get("success") else "error",
                result=vm.ToolResult(summary=mention_summary(mention)),
            )
        )
    return blocks


def mention_summary(mention: dict) -> str:
    """`523 lines` when it was inlined; `× <reason>, defaulting to agent tool
    calling` when it was declined."""
    if mention.get("success"):
        lines = mention.get("lines")
        if lines is not None:
            return f"{lines} lines"
        size = mention.get("bytes")
        return fmt_bytes(size) if size is not None else "read"
    reason = mention.get("reason") or "not inlined"
    return f"[error]×[/] {reason}, defaulting to agent tool calling"


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


def was_auto_approved(execution: ToolExecution) -> bool:
    """The call passed through a permission gate (it carries an approval
    context) and no user answer stands against it — so a rule decided it, and
    the transcript has to say so. The live app narrows this further with the
    answers it collected in THIS process; a stored session has only this."""
    if execution.approval_status is None:
        return False
    return bool(execution.extras.get("approval_context"))


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


def task_status(entry: ChildConversation) -> vm.TaskStatus:
    """A subagent's settled status. THE LINK IS THE SOURCE OF TRUTH: a
    cancelled subagent resolves without its result tool ever running, so the
    link, not the tool, says how it ended."""
    if entry.execution_result is None:
        return "waiting"
    return "failed" if entry.execution_result.is_error else "done"


# ── a stored session as transcript blocks ─────────────────────────────────────

# A terminal execution's model-facing result: `(text, is_error)`. Injected
# rather than imported so this module never reaches for a runner — the app
# passes its projector, a fixture generator passes a bare one.
ResultResolver = Callable[[ToolExecution], tuple[str | None, bool]]


def entry_blocks(
    entry: Entry | None,
    *,
    resolve_result: ResultResolver | None = None,
    subagent: bool = False,
) -> list[vm.Block]:
    """Every transcript block one session entry renders as — the mapping the
    live replay and any fixture generator share, so the two cannot drift.

    Without `resolve_result` a tool row is its header alone. `subagent=True`
    drops user messages: a child conversation's only one is its seed, and that
    already reads on the task label. A `ChildConversation` yields nothing here
    — nesting needs the session, which `transcript_blocks` has.
    """
    if isinstance(entry, UserMessage):
        if subagent:
            return []
        # Derived from the parts' metadata: the inlined file is never re-read,
        # so the transcript reproduces what the model actually saw.
        return [vm.UserBlock(text=user_transcript_text(entry.parts)), *mention_blocks(entry.parts)]
    if isinstance(entry, AssistantMessage):
        blocks: list[vm.Block] = []
        for part in entry.parts:
            if isinstance(part, ThinkingContent):
                blocks.append(vm.ThinkingBlock())
            elif isinstance(part, TextContent) and part.text.strip():
                blocks.append(vm.TextBlock(text=part.text))
        return blocks
    if isinstance(entry, ToolExecution):
        return _execution_blocks(entry, resolve_result)
    if isinstance(entry, CompactionEntry):
        if not entry.parts:
            return []
        replaced = len(entry.compacted_nodes or [])
        return [vm.NoticeBlock(text=f"context compacted · {replaced} entries summarized")]
    if isinstance(entry, TurnFinish) and entry.outcome is TurnOutcome.CANCELLED:
        return [vm.NoticeBlock(text="turn cancelled")]
    return []


def _execution_blocks(entry: ToolExecution, resolve_result: ResultResolver | None) -> list[vm.Block]:
    if is_runtime_plumbing(entry):
        # A private tool renders nothing and a spawn renders as its task block
        # — but a refusal is the one thing the user has to be told about.
        if entry.status is ExecutionStatus.REFUSED and entry.error is not None:
            return [vm.NoticeBlock(text=entry.error.error_message, error=True)]
        return []
    if is_plan_update(entry):
        if entry.status is not ExecutionStatus.COMPLETED:
            return []
        plan = plan_from_execution(entry)
        return [plan] if plan is not None else []
    result_text, is_error = None, False
    if resolve_result is not None and entry.status not in NONTERMINAL_STATUSES:
        result_text, is_error = resolve_result(entry)
    return [
        tool_block(
            entry,
            result_text,
            is_error=is_error,
            auto_approved=was_auto_approved(entry),
        )
    ]


def transcript_blocks(
    session: AgentSession,
    conversation_id: str | None = None,
    *,
    resolve_result: ResultResolver | None = None,
) -> list[vm.Block]:
    """A whole stored conversation as transcript blocks, each subagent nested
    into its own `TaskBlock`. This is the pure form of what the app mounts on
    resume, so a session file is renderable without an agent, a provider or a
    key — which is what lets a real conversation be browsed in the gallery."""
    return _conversation_blocks(
        session,
        conversation_id or session.main_conversation_id,
        resolve_result,
        subagent=False,
    )


def _conversation_blocks(
    session: AgentSession,
    conversation_id: str,
    resolve_result: ResultResolver | None,
    *,
    subagent: bool,
) -> list[vm.Block]:
    conversation = session.conversations.get(conversation_id)
    if conversation is None:
        return []
    blocks: list[vm.Block] = []
    # The plan is ONE block that the run kept rewriting, not one per update —
    # the live app mutates it in place, so the derivation has to collapse it
    # the same way or a resumed session and a derived fixture disagree.
    plan_at: int | None = None
    for node_id in conversation.nodes:
        entry = session.entries.get(node_id)
        if isinstance(entry, ChildConversation):
            description, _prompt = subagent_task(session, entry)
            blocks.append(
                vm.TaskBlock(
                    description=description,
                    status=task_status(entry),
                    blocks=_conversation_blocks(session, entry.conversation_id, resolve_result, subagent=True),
                )
            )
            continue
        for block in entry_blocks(entry, resolve_result=resolve_result, subagent=subagent):
            if isinstance(block, vm.ListBlock):  # only a plan update makes one
                if plan_at is not None:
                    blocks[plan_at] = block
                    continue
                plan_at = len(blocks)
            blocks.append(block)
    return blocks


# ── overlay rows (palette, `@` picker) ────────────────────────────────────────


def filter_rows(rows: Sequence[vm.OverlayRow], query: str) -> list[vm.OverlayRow]:
    """The palette / menu filter: a case-insensitive substring of the command
    or its description. Empty query keeps everything."""
    if not query:
        return list(rows)
    needle = query.lower()
    return [row for row in rows if needle in row.primary.lower() or needle in (row.secondary or "").lower()]


def picker_rows(
    matches: Sequence[tuple[str, str, int]],
    checked: Collection[str] = (),
) -> list[vm.OverlayRow]:
    """`files.match_files` output → `@` picker rows. `checked` holds plain
    paths; the row's `primary` carries the match spans, so the two are
    deliberately not the same string."""
    return [
        vm.OverlayRow(primary=marked, annotation=fmt_tokens(tokens), checked=path in checked)
        for path, marked, tokens in matches
    ]


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
