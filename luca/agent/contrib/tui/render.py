"""Pure derivations: agent entries/events → the `state` view-models.

No Textual imports — everything here is unit-testable without an app. The
live `AgentApp` and the resume replay both go through these functions, so the
two cannot drift.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence

from luca.agent.contrib.memory import (
    changed_of,
    is_open,
    is_todo_tool,
    is_todo_update,
    todos_of,
)
from luca.agent.contrib.questions import (
    QUESTIONS_NAMESPACE,
    OptionsType,
    QuestionsTool,
)
from luca.agent.core import declares_spawn
from luca.agent.core.models import (
    NONTERMINAL_STATUSES,
    AgentSession,
    AssistantMessage,
    ChildConversation,
    CompactionEntry,
    ContentPart,
    Conversation,
    Entry,
    ExecutionStatus,
    FileContent,
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
        elif isinstance(part, FileContent):
            label = part.name or part.metadata.get("name") or part.source.media_type or "file"
            lines.append(f"[file: {label}]")
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


def user_prompts(session: AgentSession, conversation_id: str | None = None) -> list[str]:
    """Every message the user TYPED on one conversation, oldest first — the
    composer's history, and nothing the app has to keep a second copy of.

    TEXT parts only: an `@`-mention part is the harness's expansion of what was
    typed, and an image is not something to retype. Consecutive duplicates
    collapse, shell-style.

    Walks the compaction lineage, so `/compact` does not eat the history — the
    entries survive in the store, only the PATH to them is rewritten, and the
    predecessor still holds the one the successor dropped. Nodes are deduped
    because the successor's path re-lists the tail it kept."""
    chain: list[Conversation] = []
    conversation = session.conversations.get(conversation_id or session.main_conversation_id)
    while conversation is not None:
        chain.append(conversation)
        conversation = session.conversations.get(conversation.previous_conversation_id)

    prompts: list[str] = []
    seen: set[str] = set()
    for link in reversed(chain):
        for node_id in link.nodes:
            if node_id in seen:
                continue
            seen.add(node_id)
            entry = session.entries.get(node_id)
            if not isinstance(entry, UserMessage):
                continue
            text = "\n".join(
                part.text for part in entry.parts if isinstance(part, TextContent) and mention_of(part) is None
            ).strip()
            if text and text != (prompts[-1] if prompts else None):
                prompts.append(text)
    return prompts


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
    # the native shell's primary argument is a LIST of command strings
    commands = arguments.get("commands")
    if isinstance(commands, list) and commands:
        return _collapse("; ".join(str(command) for command in commands))
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
    # A parked tool reads as in flight, because that is what it is from the
    # transcript's point of view: dispatched, no result yet, nothing the user
    # has to do about the ROW (whatever prompt the tool raised is its own).
    ExecutionStatus.AWAITING_RESULT: "running",
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
    """How long the BODY ran, from the last `ExecutionAttempt` — deliberately
    not `duration_ms`, which is total time outstanding and would count the
    approval wait and any parked period as work."""
    if not execution.attempts:
        return None
    attempt = execution.attempts[-1]
    if attempt.ended_at is None:
        return None
    seconds = (attempt.ended_at - attempt.started_at) / 1000
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
    # The spec's `display_name` is `title` when the tool declares one and the
    # internal `name` otherwise — so a provider-native tool reads as "Apply
    # patch" here while staying `openai_apply_patch` everywhere identity
    # matters. An unresolved call has no spec; its raw name is all there is.
    name = execution.tool_spec.display_name if execution.tool_spec is not None else execution.raw_tool_call.name
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


# ── the sticky plan (todo) panel ──────────────────────────────────────────────

# Where THE APP keeps the memory plugin's two stores inside
# `AgentSession.extras`. The plugin takes both dicts as constructor arguments
# and never looks a key up, so these names are the application's choice alone
# — `wiring.py` hands the dicts over, this module reads them back, and the
# session save persists them because they live on the session. An application
# composing `MemoryPlugin` differently picks its own names, or none.
TODO_STORE_KEY = "todos"
SCRATCHPAD_STORE_KEY = "scratchpad"

# How many todo rows the panel shows before it collapses the rest into one
# summary row. The panel is docked, so every row it takes is a row the
# transcript never gets back.
PLAN_ROWS = 5

_TODO_GLYPHS = {
    "completed": "done",
    "in_progress": "active",
    "pending": "pending",
    "cancelled": "done",
}


def _plan_label(todos: list[dict]) -> str:
    """The panel's header. Zero terms are omitted, so a list nobody has
    started reads `3 tasks (3 open)` rather than `(0 done, 3 open)`."""
    total = len(todos)
    noun = "task" if total == 1 else "tasks"
    open_count = sum(1 for item in todos if is_open(str(item.get("status", ""))))
    if open_count == 0:
        return f"Done ({total} {noun} completed)"
    done = total - open_count
    counts = f"{done} done, {open_count} open" if done else f"{open_count} open"
    return f"{total} {noun} ({counts})"


def _plan_row(item: dict) -> vm.ListRow:
    settled = not is_open(str(item.get("status", "")))
    # The id is span-tagged rather than folded into the text so it stays faint
    # against the item itself, which is what makes `#12` scannable.
    return vm.ListRow(
        glyph=_TODO_GLYPHS.get(str(item.get("status", "pending")), "pending"),
        text=f"[faint]#{item.get('id', '?')} -[/] {item.get('content', '')}",
        strike=settled,
    )


def _overflow_row(hidden: list[dict]) -> vm.ListRow:
    """The `… +N` row. It names a status only when every hidden item shares
    one — a mixed remainder that claimed to be `completed` would be a lie
    about work still outstanding."""
    statuses = {is_open(str(item.get("status", ""))) for item in hidden}
    label = {frozenset({True}): "pending", frozenset({False}): "completed"}.get(frozenset(statuses), "more")
    return vm.ListRow(glyph="none", text=f"[faint]… +{len(hidden)} {label}[/]")


def _plan_order(todos: list[dict], changed: Sequence[int], running: bool) -> list[dict]:
    """Which end of the list the window sits on.

    A list that fits is left in its own order: reordering rows nobody was
    going to lose is motion for its own sake, and the numbers are what the
    user tracks. Past that, the order decides what survives the window —
    while a turn runs the panel follows the work, leading with the items that
    write just moved so completions scroll past as they happen; once it
    settles the panel goes back to being a to-do list, open items first."""
    if len(todos) <= PLAN_ROWS:
        return list(todos)
    if running and changed:
        moved = set(changed)
        return [item for item in todos if item.get("id") in moved] + [
            item for item in todos if item.get("id") not in moved
        ]
    return [item for item in todos if is_open(str(item.get("status", "")))] + [
        item for item in todos if not is_open(str(item.get("status", "")))
    ]


def plan_block(
    todos: list[dict],
    *,
    changed: Sequence[int] = (),
    running: bool = False,
) -> vm.ListBlock | None:
    """The memory plugin's todo list as the sticky plan panel, or None when
    there is nothing to show. `changed` is the ids the last write moved."""
    if not todos:
        return None
    ordered = _plan_order(todos, changed, running)
    shown, hidden = ordered[:PLAN_ROWS], ordered[PLAN_ROWS:]
    rows = [_plan_row(item) for item in shown]
    if hidden:
        rows.append(_overflow_row(hidden))
    return vm.ListBlock(label=_plan_label(todos), rows=rows)


def session_todos(session: AgentSession, conversation_id: str | None = None) -> list[dict]:
    """One conversation's todo list, read back out of the session the app
    stores it in. The SAME dict the tools write — the app hands it to
    `MemoryPlugin` at construction and it rides along on every save — so this
    is a lookup, never a reconstruction.

    A session written by an application that composed the plugin differently
    (or not at all) has nothing here, and the panel simply has no list."""
    conversation_id = conversation_id or session.main_conversation_id
    slot = session.extras.get(TODO_STORE_KEY, {}).get(conversation_id) or {}
    return slot.get("todos", [])


def plan_dismissed(session: AgentSession, conversation_id: str | None = None) -> bool:
    """The PANEL's own rule, and nothing the agent knows about: a plan with
    nothing left open stops taking rows from the transcript once the user has
    spoken past it. The list itself stays — its numbering is built on it and
    the model may reopen an item — so this is a question about the dock,
    changing nothing.

    Live and resumed go through it alike, which is what keeps a reopened
    session's panel identical to the one the user closed."""
    conversation_id = conversation_id or session.main_conversation_id
    todos = session_todos(session, conversation_id)
    if not todos or any(is_open(item["status"]) for item in todos):
        return False
    conversation = session.conversations.get(conversation_id)
    if conversation is None:
        return False
    spoken_since = False
    for node_id in conversation.nodes:
        entry = session.entries.get(node_id)
        if isinstance(entry, UserMessage):
            spoken_since = True
        elif isinstance(entry, ToolExecution) and is_todo_update(entry):
            spoken_since = False
    return spoken_since


def last_todo_update(session: AgentSession) -> ToolExecution | None:
    """The most recent `update_todos` on the main conversation."""
    for node_id in reversed(session.conversations[session.main_conversation_id].nodes):
        entry = session.entries.get(node_id)
        if isinstance(entry, ToolExecution) and is_todo_update(entry):
            return entry
    return None


def plan_from_session(session: AgentSession, *, running: bool = False) -> vm.ListBlock | None:
    """The panel a stored session shows. Rebuilt the same way the plugin
    rebuilds its store on resume, so the panel and the agent's own list cannot
    disagree."""
    if plan_dismissed(session):
        return None
    todos = session_todos(session)
    last = last_todo_update(session) if running else None
    return plan_block(todos, changed=changed_of(last) if last is not None else (), running=running)


# ── the agent's questions (`ask_user`) ────────────────────────────────────────

# The panel's inner measure at the 105-column design width: 105 − 2×2 margin −
# 2 border − 2 padding. The producer PRE-WRAPS the body to it, the way
# `ToolOutput.lines` are pre-wrapped — the widget never reflows prose.
QUESTION_MEASURE = 97
TAB_CELLS = 12

CUSTOM_LABEL = "Custom answer:"
CHAT_LABEL = "Chat about this"


def is_questions_tool(execution: ToolExecution) -> bool:
    """The `ask_user` call, matched by DECLARATION — namespace plus name — so
    an application's own tool of the same name is not this package's."""
    spec = execution.tool_spec
    return spec is not None and spec.namespace == QUESTIONS_NAMESPACE and spec.name == QuestionsTool.name


def question_tab(title: str, index: int, *, limit: int = TAB_CELLS) -> str:
    """A tab label for one question.

    The tab is DERIVED, not authored: the tool's `Question` carries a title and
    nothing shorter, and asking the model for a second label per question would
    be one more thing for it to get wrong. Words are taken until the limit, so
    a short question keeps its own words and a long one is cut at a word
    boundary; a title with no usable words falls back to its position."""
    cleaned = "".join(
        character for character in title if character.isalnum() or character.isspace() or character in "-_"
    )
    label = ""
    for word in cleaned.split():
        candidate = f"{label} {word}".strip()
        if label and len(candidate) > limit:
            break
        label = candidate[:limit]
    return label or f"question {index + 1}"


def question_set_state(questions: Sequence, *, measure: int = QUESTION_MEASURE) -> vm.QuestionSetState:
    """Turn the tool's `Question` models into the dock's view-model.

    The producer appends the custom and chat rows, so a question is never
    authored with them and they are always last — which is what lets the widget
    treat "the last two rows" as a fixed shape."""
    built: list[vm.Question] = []
    for index, question in enumerate(questions):
        body = question.body or ""
        lines: list[str] = []
        for paragraph in body.split("\n"):
            lines.extend(textwrap.wrap(paragraph, measure) or [""])
        built.append(
            vm.Question(
                tab=question_tab(question.title, index),
                title=question.title,
                body=lines if body else [],
                mode="multi" if question.options_type == OptionsType.MULTIPLE_SELECT else "single",
                options=[
                    *(vm.QuestionOption(label=option) for option in question.options),
                    vm.QuestionOption(label=CUSTOM_LABEL, kind="custom"),
                    vm.QuestionOption(label=CHAT_LABEL, kind="chat", key_hint="enter"),
                ],
            )
        )
    return vm.QuestionSetState(questions=built)


def answer_payload(state: vm.QuestionSetState) -> dict:
    """The dock's state as the payload `QuestionsTool.answer()` stores.

    Verbatim and unvalidated on purpose: the tool never rejects a payload, and
    everything here is text on its way to becoming a string the model reads."""
    answers = []
    for question in state.questions:
        chat = any(option.kind == "chat" and option.checked for option in question.options)
        selected: list[str] = []
        custom: str | None = None
        for index, option in enumerate(question.options):
            if option.kind == "chat":
                continue
            if option.kind == "custom":
                custom = option.text or None
                continue
            if question.mode == "multi":
                if option.checked:
                    selected.append(option.label)
            # `answer`, not `selected`: the caret is not the pick. Reading the
            # caret here would send the model whichever option the user last
            # arrowed over rather than the one they committed.
            elif question.answer == index:
                selected.append(option.label)
        answers.append(
            {
                "question": question.title,
                "chat_about_this": chat,
                "answers": selected,
                "custom_answer": custom,
            }
        )
    return {"answers": answers, "custom_notes": state.extra or None}


def question_transcript_blocks(execution: ToolExecution) -> list[vm.Block] | None:
    """How a `ask_user` execution renders in the transcript, or None when it is
    not one and the ordinary tool-block path applies.

    THE ONE DERIVATION, called by both the live event handler and the replay,
    so the two cannot draw different things for the same call — which they did
    while each carried its own status test: a set the user cancelled out of
    rendered as a collapsed `0 questions · 0 answered` live and as the real
    interrupted call on reload.

    - COMPLETED → the collapsed header + answer list.
    - NONTERMINAL (parked) → nothing: the set is still holding the dock, and a
      header row under it would say it is over when it is not.
    - any other terminal (INVALID at birth, INTERRUPTED by a cancel, a failed
      poll) → None, so the ordinary tool block states what went wrong."""
    if not is_questions_tool(execution):
        return None
    if execution.status is ExecutionStatus.COMPLETED:
        return question_blocks(execution)
    if execution.status in NONTERMINAL_STATUSES:
        return []
    return None


def question_blocks(execution: ToolExecution) -> list[vm.Block]:
    """The collapsed transcript rendering of a finished question set: the
    ordinary `▸ ask_user  N questions` header plus one `☑ tab → answer` row per
    question, in the gutter idiom the plan panel already uses.

    Built from `structured_content` — the tool's own record of what it holds —
    never from `raw_tool_call.arguments`, which carries the questions and none
    of the answers."""
    payload = (execution.result.structured_content if execution.result else None) or {}
    stored = payload.get("questions")
    stored = stored if isinstance(stored, list) else []
    answer = payload.get("answer")
    entries = (answer or {}).get("answers") if isinstance(answer, dict) else None
    entries = [entry for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []

    rows: list[vm.ListRow] = []
    declined = 0
    for index, question in enumerate(stored):
        title = str(question.get("title", "")) if isinstance(question, dict) else ""
        entry = next(
            (item for item in entries if item.get("question") == title),
            entries[index] if index < len(entries) else None,
        )
        text = question_tab(title, index)
        if entry is None:
            rows.append(vm.ListRow(glyph="pending", text=text, annotation="→  no answer"))
            continue
        if entry.get("chat_about_this"):
            declined += 1
            rows.append(vm.ListRow(glyph="pending", text=text, annotation="→  [accent]chat about this[/]"))
            continue
        picked = [str(item) for item in entry.get("answers") or []]
        custom = entry.get("custom_answer")
        if isinstance(custom, str) and custom.strip():
            picked.append(f"custom · {custom}")
        rows.append(
            vm.ListRow(
                glyph="done" if picked else "pending",
                text=text,
                annotation="→  " + (", ".join(picked) if picked else "no answer"),
            )
        )
    notes = (answer or {}).get("custom_notes") if isinstance(answer, dict) else None
    if isinstance(notes, str) and notes.strip():
        rows.append(vm.ListRow(glyph="none", text="", annotation=f"added · {notes}"))

    answered = sum(1 for row in rows if row.glyph == "done")
    note = f"{answered} answered"
    if declined:
        note += " · chat about this"
    header = vm.ToolBlock(
        tool=execution.tool_spec.display_name if execution.tool_spec else "ask_user",
        arg=f"{len(stored)} question{'s' if len(stored) != 1 else ''}",
        status="ok" if execution.status is ExecutionStatus.COMPLETED else "error",
        note_right=note,
    )
    return [header, vm.ListBlock(rows=rows, column=28)]


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
    if is_todo_tool(entry):
        # Both halves of the pair render as the sticky panel, which is not a
        # transcript block — so the transcript shows neither the write nor the
        # read that so often precedes it.
        return []
    questions = question_transcript_blocks(entry)
    if questions is not None:
        return questions
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
        blocks.extend(entry_blocks(entry, resolve_result=resolve_result, subagent=subagent))
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


def _preview_todo(execution: ToolExecution) -> str | None:
    """A todo call as one preview row: the item in progress, or failing that
    the first still open. A list with nothing open previews as nothing — the
    plan is over and the rows are better spent on what came after it."""
    todos = todos_of(execution) or []
    active = next((item for item in todos if item.get("status") == "in_progress"), None)
    if active is None:
        active = next((item for item in todos if is_open(str(item.get("status", "")))), None)
    if active is None:
        return None
    return f"[accent]◉[/] {active.get('content', '')}"


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
            if is_todo_tool(entry):
                # The same rule the transcript follows: a todo call is the
                # panel, not a row. The one thing worth previewing is what the
                # session would resume ON — the item it was working through.
                active = _preview_todo(entry)
                if active is not None:
                    rows.append(active)
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
