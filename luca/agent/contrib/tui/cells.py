"""Transcript cell widgets.

Each cell is one block of the conversation: a bordered `Static` whose border
title names the role. Cells keep their text in plain Python state (`.text`,
`.result_text`, `.status`) so tests assert on attributes, not on rendered
output. Markup is disabled everywhere — model output is arbitrary text.
"""

from __future__ import annotations

from textual.widgets import Static

from luca.agent.core.models import ExecutionStatus, ToolExecution

from .render import clip_text, format_tool_call, status_label


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
        super().__init__(text, markup=False, classes=classes)
        self._text = text
        self.border_title = self.role

    @property
    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text
        self.update(text)

    def append_text(self, delta: str) -> None:
        self.set_text(self._text + delta)


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


class ReasoningCell(TranscriptCell):
    role = "thinking"

    DEFAULT_CSS = """
    ReasoningCell {
        border: round $panel-lighten-1;
        color: $text-muted;
        text-style: italic;
    }
    """


class NoticeCell(TranscriptCell):
    """Turn-level notices: cancellations, run failures."""

    role = "notice"

    DEFAULT_CSS = """
    NoticeCell { border: round $warning; color: $text-muted; }
    NoticeCell.-error { border: round $error; color: $error; }
    """

    def __init__(self, text: str, *, error: bool = False) -> None:
        super().__init__(text, classes="-error" if error else None)


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
        self, execution: ToolExecution, result_text: str, is_error: bool,
    ) -> None:
        self.status = execution.status
        self.result_text = result_text
        self.is_error = is_error
        self.remove_class("-running")
        self.add_class("-error" if is_error else "-ok")
        self._show_status()
        self.set_text(f"{self.call_text}\n→ {clip_text(result_text)}")
