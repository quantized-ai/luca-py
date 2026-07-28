"""Transcript cell widgets.

Each cell is one block of the conversation: a bordered `Static` whose border
title names the role. Cells keep their text in plain Python state (`.text`,
`.result_text`, `.status`) so tests assert on attributes, not on rendered
output. Markup is disabled everywhere — model output is arbitrary text.
"""

from __future__ import annotations

from rich.markdown import Markdown
from textual.widgets import Static

from luca.agent.contrib.subagents import SubAgentTask, TaskStatus
from luca.agent.core.models import CompactionEntry, ExecutionStatus, ToolExecution

from .render import (
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


class SubAgentCell(TranscriptCell):
    """One background sub-agent's lifecycle in a single cell: the task line, a
    status border subtitle with a running step count, and the final result (or
    error) at the terminal outcome. Coarse status only — the child's own token
    stream never reaches the transcript."""

    role = "sub-agent"

    DEFAULT_CSS = """
    SubAgentCell { border: round $secondary; }
    SubAgentCell.-running { border: round $warning; }
    SubAgentCell.-ok { border: round $success; }
    SubAgentCell.-error { border: round $error; }
    """

    def __init__(self, task: SubAgentTask) -> None:
        self.task_line = f"{task.agent_type}: {task.title}"
        super().__init__(self.task_line)
        self.task_id = task.id
        self.status: TaskStatus = task.status
        self.border_title = f"sub-agent · {task.agent_type}"
        self._apply(task)

    def update_task(self, task: SubAgentTask) -> None:
        self._apply(task)

    def _apply(self, task: SubAgentTask) -> None:
        self.status = task.status
        self.border_subtitle = (
            f"running · step {task.steps}" if task.status is TaskStatus.RUNNING else task.status.value
        )
        self.remove_class("-running", "-ok", "-error")
        if task.status is TaskStatus.RUNNING:
            self.add_class("-running")
        elif task.status is TaskStatus.DONE:
            self.add_class("-ok")
            self.set_text(f"{self.task_line}\n→ {clip_text(task.result or '')}")
        elif task.status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
            self.add_class("-error")
            self.set_text(f"{self.task_line}\n→ {task.error or task.status.value}")
