"""Transcript cell widgets.

Each cell is one block of the conversation: a bordered `Static` whose border
title names the role. Cells keep their text in plain Python state (`.text`,
`.result_text`, `.status`) so tests assert on attributes, not on rendered
output. Markup is disabled everywhere — model output is arbitrary text.

`SubagentPanel` is the one container: a subagent's cells are mounted INSIDE
it rather than beside it, which is what keeps several subagents' interleaved
output readable.
"""

from __future__ import annotations

from rich.markdown import Markdown
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from luca.agent.core.models import CompactionEntry, ExecutionStatus, ToolExecution

from .render import (
    TASK_MAX_CHARS,
    TASK_MAX_LINES,
    clip_text,
    compaction_subtitle,
    compaction_transcript_text,
    format_tool_call,
    status_label,
)


class TranscriptCell(Static):
    """Base cell: bordered block with a role title and plain-text content."""

    role = "cell"

    DEFAULT_CSS = """
    TranscriptCell {
        height: auto;
        margin: 0 1 1 1;
        padding: 0 1;
        border: round $panel-lighten-2;
        border-title-color: $text-muted;
    }
    """

    def __init__(self, text: str = "", *, classes: str | None = None) -> None:
        super().__init__(markup=False, classes=classes)
        self._text = text
        self.border_title = self.role
        self.update(self._renderable(text))

    @property
    def text(self) -> str:
        return self._text

    def _renderable(self, text: str):
        """What actually gets rendered for `text`. Base cells are plain; a
        subclass can format (e.g. markdown). `self._text` always stays raw."""
        return text

    def set_text(self, text: str) -> None:
        self._text = text
        self.update(self._renderable(text))

    def append_text(self, delta: str) -> None:
        # Streaming stays plain — partial markdown (a half-open fence) renders
        # badly; set_text on the block boundary snaps it to the formatted view.
        self._text += delta
        self.update(self._text)


class UserCell(TranscriptCell):
    role = "you"

    DEFAULT_CSS = """
    UserCell { border: round $accent; }
    """


class AssistantCell(TranscriptCell):
    role = "assistant"

    DEFAULT_CSS = """
    AssistantCell { border: round $primary; }
    """

    def _renderable(self, text: str):
        return Markdown(text) if text.strip() else ""


class ReasoningCell(TranscriptCell):
    role = "thinking"

    DEFAULT_CSS = """
    ReasoningCell {
        border: round $panel-lighten-1;
        color: $text-muted;
        text-style: italic;
    }
    """

    def _renderable(self, text: str):
        return Markdown(text) if text.strip() else ""


class NoticeCell(TranscriptCell):
    """Turn-level notices: cancellations, run failures."""

    role = "notice"

    DEFAULT_CSS = """
    NoticeCell { border: round $warning; color: $text-muted; }
    NoticeCell.-error { border: round $error; color: $error; }
    """

    def __init__(self, text: str, *, error: bool = False) -> None:
        super().__init__(text, classes="-error" if error else None)


class CompactionCell(TranscriptCell):
    """A compaction: the summary that replaced the older span, with how many
    entries it stands in for as the border subtitle. Nothing is lost — the
    replaced entries are still in the session, on the archived conversation."""

    role = "compacted"

    DEFAULT_CSS = """
    CompactionCell {
        border: round $success;
        color: $text-muted;
    }
    """

    def __init__(self, entry: CompactionEntry) -> None:
        super().__init__(compaction_transcript_text(entry))
        self.border_subtitle = compaction_subtitle(entry)

    def _renderable(self, text: str):
        return Markdown(text) if text.strip() else ""


class SubagentPanel(Vertical):
    """One subagent's whole conversation, boxed and indented.

    ATTRIBUTION CANNOT BE POSITIONAL. Several subagents advance at once and
    their events arrive interleaved on a single stream, so a flat transcript
    would splice three conversations into one column with nothing saying which
    line came from which. Every cell a subagent produces is mounted inside its
    own panel instead, keyed by conversation id — the same key the events
    carry.

    The border says what the subagent was asked to do and how it ended, so a
    finished panel reads as a unit without expanding anything: title = the
    spawn's description, subtitle = its status."""

    role = "subagent"

    DEFAULT_CSS = """
    SubagentPanel {
        height: auto;
        margin: 0 1 1 4;
        padding: 0 1;
        border: round $accent-lighten-2;
        border-title-color: $text-muted;
        border-subtitle-color: $text-muted;
    }
    SubagentPanel.-done { border: round $success; }
    SubagentPanel.-error { border: round $error; }
    SubagentPanel > .subagent-task {
        color: $text-muted;
        text-style: italic;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, conversation_id: str, description: str, prompt: str = "") -> None:
        super().__init__()
        self.conversation_id = conversation_id
        self.description = description
        self.prompt = prompt
        # "waiting" until a `SubagentStarted` flips it: a spawned subagent may
        # be queued behind `subagents_max_workers` before it ever runs.
        self.status = "waiting"
        self.border_title = f"subagent · {description}"
        self.border_subtitle = "waiting…"

    def compose(self) -> ComposeResult:
        # The task the subagent was given. Its conversation starts with exactly
        # this as a user message, so the seed is NOT replayed as a cell —
        # rendering both would say it twice.
        if self.prompt.strip():
            yield Static(
                clip_text(self.prompt, TASK_MAX_LINES, TASK_MAX_CHARS),
                markup=False,
                classes="subagent-task",
            )

    def set_activity(self, status: str) -> None:
        """The live note while the subagent is unresolved: "running" when its
        drive holds a worker slot, "waiting" when it is queued or paused at a
        gate. `settle()` owns the terminal state and wins over a late event."""
        if self.status in ("done", "failed"):
            return
        self.status = status
        self.border_subtitle = f"{status}…"

    def settle(self, *, is_error: bool) -> None:
        """Close the panel. Driven by the resolved `ChildConversation`, not by
        the result tool's event: a CANCELLED subagent is resolved without the
        tool ever running, and its panel still has to stop saying "running"."""
        self.status = "failed" if is_error else "done"
        self.border_subtitle = self.status
        self.remove_class("-done", "-error")
        self.add_class("-error" if is_error else "-done")


class ToolCallCell(TranscriptCell):
    """One tool call's whole lifecycle in a single cell: the call line at
    birth, a status border subtitle across transitions, and the (clipped)
    result or error text at the terminal outcome."""

    role = "tool"

    DEFAULT_CSS = """
    ToolCallCell { border: round $secondary; }
    ToolCallCell.-running { border: round $warning; }
    ToolCallCell.-ok { border: round $success; }
    ToolCallCell.-error { border: round $error; }
    """

    def __init__(self, execution: ToolExecution) -> None:
        self.call_text = format_tool_call(execution.raw_tool_call)
        super().__init__(self.call_text)
        self.status: ExecutionStatus = execution.status
        self.result_text: str | None = None
        self.is_error = False
        self.border_title = f"tool · {execution.raw_tool_call.name}"
        self._show_status()

    def _show_status(self) -> None:
        self.border_subtitle = status_label(self.status)

    def mark_running(self, execution: ToolExecution) -> None:
        self.status = execution.status
        self.add_class("-running")
        self._show_status()

    def finish(
        self,
        execution: ToolExecution,
        result_text: str,
        is_error: bool,
    ) -> None:
        self.status = execution.status
        self.result_text = result_text
        self.is_error = is_error
        self.remove_class("-running")
        self.add_class("-error" if is_error else "-ok")
        self._show_status()
        self.set_text(f"{self.call_text}\n→ {clip_text(result_text)}")
