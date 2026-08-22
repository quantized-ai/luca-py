"""Agent events → ACP `session/update` notifications.

The translation is mostly mechanical, because the two vocabularies line up
almost exactly: `ToolKind` differs from ACP's by one name, `ExecutionStatus`
collapses onto four ACP statuses, and the streaming text and reasoning deltas
are already chunk-shaped. What is NOT mechanical is subagents, which have no
place in ACP at all — see `Translator` below.

Nothing here talks to a client. `translate()` returns the updates one event
produces, in order, and the caller sends them; that keeps the whole mapping
testable against a list.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from acp.helpers import (
    start_tool_call,
    text_block,
    tool_content,
    tool_diff_content,
    update_agent_message,
    update_agent_thought,
    update_plan,
    update_tool_call,
)
from acp.schema import PlanEntry

from luca.agent.contrib.memory import is_todo_tool, is_todo_update, todos_of
from luca.agent.contrib.subagents import SPAWN_TOOL_NAME
from luca.agent.core.events import (
    AgentEvent,
    ReasoningBlock,
    ReasoningDelta,
    SubagentsSpawned,
    TextBlock,
    TextDelta,
    ToolCallReceived,
    ToolExecuted,
    ToolExecutionStarted,
)
from luca.agent.core.models import (
    ExecutionStatus,
    ToolExecution,
    ToolKind,
)

logger = logging.getLogger(__name__)

# Ours to theirs. Every name matches but one: we call it `web_fetch`, ACP calls
# it `fetch`. ACP also has `think`, which no tool of ours declares.
TOOL_KINDS: dict[ToolKind, str] = {
    ToolKind.READ: "read",
    ToolKind.SEARCH: "search",
    ToolKind.WEB_FETCH: "fetch",
    ToolKind.EDIT: "edit",
    ToolKind.MOVE: "move",
    ToolKind.DELETE: "delete",
    ToolKind.EXECUTE: "execute",
    ToolKind.SWITCH_MODE: "switch_mode",
    ToolKind.OTHER: "other",
}

# Thirteen framework statuses onto ACP's four. Everything that is neither
# running nor successfully finished is `failed`, which is what a client should
# show for a call that was refused, denied, cancelled or never resolved.
TOOL_STATUSES: dict[ExecutionStatus, str] = {
    ExecutionStatus.RECEIVED: "pending",
    ExecutionStatus.PENDING: "pending",
    ExecutionStatus.AWAITING_RESULT: "pending",
    ExecutionStatus.RUNNING: "in_progress",
    ExecutionStatus.COMPLETED: "completed",
    ExecutionStatus.FAILED: "failed",
    ExecutionStatus.NOT_FOUND: "failed",
    ExecutionStatus.INVALID: "failed",
    ExecutionStatus.REJECTED: "failed",
    ExecutionStatus.REFUSED: "failed",
    ExecutionStatus.CANCELLED: "failed",
    ExecutionStatus.INTERRUPTED: "failed",
    ExecutionStatus.TIMED_OUT: "failed",
}

# The memory plugin's todo statuses onto ACP's plan-entry statuses. ACP has no
# `cancelled`, and a cancelled item is done being worked on, so it reads as
# completed rather than as something still pending.
PLAN_STATUSES = {
    "pending": "pending",
    "in_progress": "in_progress",
    "completed": "completed",
    "cancelled": "completed",
}

TITLE_ARG_LENGTH = 60


def tool_kind(execution: ToolExecution) -> str:
    """The call's ACP kind, which is only ever a hint for the client's icon."""
    spec = execution.tool_spec
    if spec is None or spec.tool_kind is None:
        return "other"
    return TOOL_KINDS.get(spec.tool_kind, "other")


def tool_title(execution: ToolExecution) -> str:
    """A one-line label for the call: the tool's name plus its most telling
    argument, which is what a client puts in the collapsed row.

    Deliberately plain text. `render.tool_arg` does the same job for the TUI
    but emits theme-token markup, and a JSON-RPC client would render the tags
    literally."""
    name = execution.raw_tool_call.name
    arguments = execution.raw_tool_call.arguments
    if not isinstance(arguments, dict) or not arguments:
        return name
    for key in ("file_path", "path", "command", "pattern", "description", "prompt"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            single = value.strip().splitlines()[0]
            if len(single) > TITLE_ARG_LENGTH:
                single = single[: TITLE_ARG_LENGTH - 1] + "…"
            return f"{name} {single}"
    return name


def tool_locations(execution: ToolExecution) -> list[dict[str, str]]:
    """The absolute paths this call touches, for the client's file
    highlighting. Read off the result rather than guessed from arguments,
    since only the tool knows what a relative path resolved to."""
    metadata = (execution.result.metadata if execution.result else None) or {}
    paths = [metadata["path"]] if isinstance(metadata.get("path"), str) else []
    paths.extend(
        entry["absolute_path"]
        for entry in metadata.get("files") or ()
        if isinstance(entry, dict) and isinstance(entry.get("absolute_path"), str)
    )
    return [{"path": path} for path in paths]


def tool_diffs(execution: ToolExecution) -> list:
    """The change this call made, as ACP diff content — one per file.

    Present only for the editing tools, which record `old_text` / `new_text`
    beside their unified diff. Everything else returns nothing and the client
    shows the result text instead."""
    metadata = (execution.result.metadata if execution.result else None) or {}
    entries: list[dict] = []
    if isinstance(metadata.get("path"), str) and "new_text" in metadata:
        entries.append(metadata)
    entries.extend(
        {**entry, "path": entry.get("absolute_path") or entry.get("path")}
        for entry in metadata.get("files") or ()
        if isinstance(entry, dict) and "new_text" in entry
    )
    return [
        tool_diff_content(
            path=entry["path"],
            new_text=entry["new_text"] or "",
            old_text=entry.get("old_text"),
        )
        for entry in entries
        if isinstance(entry.get("path"), str)
    ]


def plan_entries(execution: ToolExecution) -> list[PlanEntry]:
    """One `update_todos` call as an ACP plan.

    ACP plan entries carry a priority we have no source for, so every entry is
    `medium`. Inventing one from position would be worse than saying nothing:
    a client sorting or colouring by it would be reading noise."""
    return [
        PlanEntry(
            content=str(todo.get("content", "")),
            priority="medium",
            status=PLAN_STATUSES.get(str(todo.get("status")), "pending"),
        )
        for todo in todos_of(execution) or ()
    ]


def _raw_output(execution: ToolExecution) -> Any:
    """`rawOutput` for the client's expandable detail. The structured payload
    when the tool produced one, else nothing — the result text already travels
    as content, and repeating it here doubles every tool call on the wire."""
    result = execution.result
    if result is None or result.structured_content is None:
        return None
    try:
        json.dumps(result.structured_content)
    except (TypeError, ValueError):
        return None
    return result.structured_content


class Translator:
    """Turns one conversation subtree's events into one linear ACP stream.

    ACP HAS NO SUBAGENTS. A session is one stream of updates, while a luca run
    yields events for the main conversation and every child at once, tagged by
    `conversation_id`. So a child's events are folded into the tool call that
    spawned it: its text arrives as content on that call, and the client shows
    nested work as progress on the spawn rather than as a second voice
    interleaved with the first.

    A child whose parent call cannot be resolved is dropped and logged. That is
    a real possibility on a resumed session, where the spawn happened in a
    previous process and this translator never saw the `SubagentsSpawned`.

    Stateful on purpose: the parent map is built from events as they arrive,
    and `message_id` has to change exactly when a new assistant message starts.
    One instance per prompt turn.
    """

    def __init__(self, main_conversation_id: str, *, streaming: bool = True) -> None:
        self.main_conversation_id = main_conversation_id
        # WHICH TIER OF EVENTS TO FORWARD. Text and reasoning arrive twice
        # under `streaming=True`: once as deltas while tokens land, and once
        # more as the completed block. Forwarding both sends every sentence to
        # the client twice. Blocks are the only tier a non-streaming run emits,
        # so this is a choice, not a filter.
        self.streaming = streaming
        # child conversation id → the tool call that spawned it
        self._spawned_by: dict[str, str] = {}
        # the most recent spawn call seen on each conversation, which is what a
        # `SubagentsSpawned` batch belongs to
        self._last_spawn_call: dict[str, str] = {}
        self._message_id = 0
        self._open_message = False

    # ── message ids ───────────────────────────────────────────────────────────

    def _current_message_id(self) -> str:
        if not self._open_message:
            self._message_id += 1
            self._open_message = True
        return f"msg_{self._message_id}"

    def end_message(self) -> None:
        """Close the current assistant message, so the next chunk starts a new
        one. Called by the driver between model rounds."""
        self._open_message = False

    # ── routing ───────────────────────────────────────────────────────────────

    def _parent_call(self, conversation_id: str) -> str | None:
        return self._spawned_by.get(conversation_id)

    def _fold(self, conversation_id: str, block) -> list:
        """A child conversation's output, as content on its spawn call."""
        call_id = self._parent_call(conversation_id)
        if call_id is None:
            logger.debug("dropping an event from %s: no spawning tool call is known", conversation_id)
            return []
        return [update_tool_call(call_id, content=[tool_content(block)])]

    def translate(self, event: AgentEvent) -> list:
        """The `session/update` payloads this event produces, in order."""
        own = event.conversation_id == self.main_conversation_id
        match event:
            case SubagentsSpawned():
                call_id = self._last_spawn_call.get(event.conversation_id)
                for child in event.conversation_ids:
                    if call_id is not None:
                        self._spawned_by[child] = call_id
                return []

            case TextDelta() | TextBlock():
                if not event.text or isinstance(event, TextBlock) is self.streaming:
                    return []
                if not own:
                    return self._fold(event.conversation_id, text_block(event.text))
                chunk = update_agent_message(text_block(event.text))
                chunk.message_id = self._current_message_id()
                return [chunk]

            case ReasoningDelta() | ReasoningBlock():
                # A redacted block has an encrypted attestation and no body;
                # there is nothing to show and an empty bubble is worse than
                # none.
                if not event.text or getattr(event, "redacted", False):
                    return []
                if isinstance(event, ReasoningBlock) is self.streaming:
                    return []
                if not own:
                    return []
                chunk = update_agent_thought(text_block(event.text))
                chunk.message_id = self._current_message_id()
                return [chunk]

            case ToolCallReceived():
                self._open_message = False
                execution = event.execution
                if execution.raw_tool_call.name == SPAWN_TOOL_NAME:
                    self._last_spawn_call[event.conversation_id] = event.tool_call_id
                if not own:
                    return []
                return [
                    start_tool_call(
                        event.tool_call_id,
                        tool_title(execution),
                        kind=tool_kind(execution),
                        status=TOOL_STATUSES[execution.status],
                        raw_input=execution.raw_tool_call.arguments,
                    )
                ]

            case ToolExecutionStarted():
                if not own:
                    return []
                return [update_tool_call(event.tool_call_id, status="in_progress")]

            case ToolExecuted():
                execution = event.execution
                if is_todo_tool(execution):
                    # A todo write is the plan, not a tool row. `read_todo`
                    # wrote nothing, so it is neither.
                    entries = plan_entries(execution) if own and is_todo_update(execution) else None
                    return [update_plan(entries)] if entries else []
                if not own:
                    return []
                content = tool_diffs(execution) or (
                    [tool_content(text_block(event.result_text))] if event.result_text else None
                )
                return [
                    update_tool_call(
                        event.tool_call_id,
                        status="failed" if event.is_error else TOOL_STATUSES[execution.status],
                        content=content,
                        locations=tool_locations(execution) or None,
                        raw_output=_raw_output(execution),
                    )
                ]

        # FinishReason, ApprovalRequired, the compaction events, the three
        # subagent lifecycle events and the two `*Start` markers carry nothing
        # a client can render. Silence is the right translation.
        return []
