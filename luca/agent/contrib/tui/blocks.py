"""The transcript block widgets — the design handoff's block vocabulary.

Each block is independently mountable and mutable in place, so streaming
appends and retroactive status changes (a pending call turning denied) never
re-render the transcript. Widgets render from `state` view-models; colors
resolve through the registered theme at render time (`resolve_tokens`), so no
color ever originates here.

Geometry (cells) comes from the handoff: 1-space glyph gutter on user/thinking
rows, a 6-cell tool-name column, the `1 + │ + 2` result gutter, the
`[5][3][code]` diff columns, and the 84-column assistant measure (in TCSS).
"""

from __future__ import annotations

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Static

from . import state as vm
from .format import (
    GLYPH_THINKING,
    GLYPH_TOOL,
    GLYPH_USER,
    LIST_GLYPHS,
    MINUS,
    TOOL_NAME_COLUMN,
    code_spans,
    fmt_duration,
    spans,
)
from .theme import Tokens, resolve_tokens


class _Line(Static):
    """A Static that renders itself from its view-model on demand."""

    def __init__(self, *, classes: str | None = None) -> None:
        super().__init__(markup=False, classes=classes)

    @property
    def tokens(self) -> Tokens:
        return resolve_tokens(self.app)

    def render(self) -> Text:  # pragma: no cover - subclasses override
        return Text()


# ── user turn ─────────────────────────────────────────────────────────────────


class UserTurn(Horizontal):
    """`›` in accent, one space, the message — wrapped with a hanging indent."""

    def __init__(self, text: str) -> None:
        super().__init__(classes="block")
        self.text = text

    def compose(self) -> ComposeResult:
        yield Static(GLYPH_USER, markup=False, classes="turn-glyph")
        yield Static(self.text, markup=False, classes="turn-body")

    def set_text(self, text: str) -> None:
        self.text = text
        self.query_one(".turn-body", Static).update(text)


# ── thinking ──────────────────────────────────────────────────────────────────


class ThinkingLine(Horizontal):
    """`∴ thought for 3s`, or `∴ <activity>` while in progress."""

    def __init__(self, model: vm.ThinkingBlock) -> None:
        super().__init__(classes="block")
        self.model = model

    def _text(self) -> str:
        if self.model.activity is not None:
            return self.model.activity
        if self.model.duration is not None:
            return f"thought for {fmt_duration(self.model.duration)}"
        # A bare block is a SETTLED one whose duration was never recorded
        # (non-streaming tier, replay) — the live in-progress path always
        # sets an activity, so this must not read as in progress.
        return "thought"

    def compose(self) -> ComposeResult:
        yield Static(GLYPH_THINKING, markup=False, classes="turn-glyph")
        yield Static(self._text(), markup=False, classes="turn-body")

    def set_activity(self, activity: str) -> None:
        self.model = self.model.model_copy(update={"activity": activity, "duration": None})
        self.query_one(".turn-body", Static).update(self._text())

    def settle(self, duration: float) -> None:
        self.model = vm.ThinkingBlock(duration=duration)
        self.query_one(".turn-body", Static).update(self._text())


# ── assistant text ────────────────────────────────────────────────────────────


class AssistantText(_Line):
    """Assistant prose at the 84-column measure; `` `spans` `` in accent; a
    streaming block ends with the 1-cell block cursor."""

    def __init__(self, model: vm.TextBlock) -> None:
        super().__init__(classes="block")
        self.model = model

    def render(self) -> Text:
        tokens = self.tokens
        text = code_spans(self.model.text, tokens)
        if self.model.streaming:
            text.append("█", style=Style(color=tokens.foreground))
        return text

    @property
    def text(self) -> str:
        return self.model.text

    def append_text(self, delta: str) -> None:
        self.model = self.model.model_copy(update={"text": self.model.text + delta})
        self.refresh(layout=True)

    def settle(self, text: str) -> None:
        self.model = vm.TextBlock(text=text, streaming=False)
        self.refresh(layout=True)


# ── notice ────────────────────────────────────────────────────────────────────


class NoticeLine(_Line):
    """A turn-level notice: one faint row, `$error` for failures."""

    def __init__(self, text: str, *, error: bool = False) -> None:
        super().__init__(classes="block" + (" -error" if error else ""))
        self.text = text
        self.error = error

    def render(self) -> Text:
        return Text(self.text, no_wrap=False)


# ── tool call + result ────────────────────────────────────────────────────────


def _stat_text(stat: vm.DiffStat, tokens: Tokens) -> Text:
    text = Text()
    if stat.add is not None:
        text.append(f"+{stat.add}", style=Style(color=tokens.success))
    if stat.remove is not None:
        if text:
            text.append(" ")
        text.append(f"{MINUS}{stat.remove}", style=Style(color=tokens.error))
    return text


class ToolHeader(_Line):
    """`▸ name   arg`, the name in a 6-cell column, with an optional
    right-aligned annotation (diff stat, `loaded`)."""

    def __init__(self, view: ToolBlockView) -> None:
        super().__init__()
        self.view = view

    def render(self) -> Text:
        tokens = self.tokens
        model = self.view.model
        glyph_color = tokens.error if model.status in ("denied", "error") else tokens.accent
        left = Text(no_wrap=True, overflow="ellipsis")
        left.append(GLYPH_TOOL, style=Style(color=glyph_color))
        left.append(" ")
        left.append(model.tool.ljust(TOOL_NAME_COLUMN), style=Style(color=tokens.foreground))
        left.append(" ")
        left.append(model.arg, style=Style(color=tokens.muted))
        right = Text()
        if model.stat is not None:
            right = _stat_text(model.stat, tokens)
        elif model.note_right:
            right = Text(model.note_right, style=Style(color=tokens.faint))
        if not right:
            return left
        pad = self.size.width - left.cell_len - right.cell_len
        if pad < 1:
            return left
        left.append(" " * pad)
        left.append_text(right)
        return left


class ResultLine(_Line):
    """One gutter row: diff stat, summary, decision note — any combination.
    A denied call's summary renders in `$error`."""

    def __init__(self, view: ToolBlockView) -> None:
        super().__init__()
        self.view = view

    def render(self) -> Text:
        tokens = self.tokens
        model = self.view.model
        result = model.result or vm.ToolResult()
        text = Text(no_wrap=True, overflow="ellipsis")
        if result.stat is not None:
            text.append_text(_stat_text(result.stat, tokens))
        if result.summary:
            if text:
                text.append(" ")
            color = tokens.error if model.status == "denied" else tokens.faint
            text.append_text(spans(result.summary, tokens, Style(color=color)))
        if result.note:
            if text:
                text.append(" ")
            text.append(result.note, style=Style(color=tokens.faint))
        return text


class OutputLines(_Line):
    """Verbatim output inside the gutter, `$text-muted`, span-tagged lines
    allowed. `expanded` swaps in the hidden tail."""

    def __init__(self, view: ToolBlockView) -> None:
        super().__init__()
        self.view = view

    def render(self) -> Text:
        tokens = self.tokens
        output = self.view.model.output or vm.ToolOutput()
        lines = list(output.lines)
        if self.view.expanded:
            lines += output.hidden_lines
        text = Text()
        for index, line in enumerate(lines):
            if index:
                text.append("\n")
            rendered = spans(line, tokens, Style(color=tokens.muted))
            rendered.no_wrap = True
            rendered.overflow = "ellipsis"
            text.append_text(rendered)
        return text


class OutputSummary(_Line):
    """`exit 1 · 1.9s · 11 lines hidden  ^o expand` under the output."""

    def __init__(self, view: ToolBlockView) -> None:
        super().__init__()
        self.view = view

    def render(self) -> Text:
        tokens = self.tokens
        output = self.view.model.output or vm.ToolOutput()
        text = Text(no_wrap=True, overflow="ellipsis")
        if output.summary:
            text.append_text(spans(output.summary, tokens, Style(color=tokens.faint)))
        if output.expand_hint:
            if text:
                text.append(" ")
            hint = "^o collapse" if self.view.expanded else "^o expand"
            text.append(hint, style=Style(color=tokens.muted))
        return text


class GutterBox(Vertical):
    """The `1 + │ + 2` result gutter under a call; `$error` edge on failure."""

    def __init__(self, view: ToolBlockView) -> None:
        classes = "gutter" + (" -error" if view.model.status == "error" else "")
        super().__init__(classes=classes)
        self.view = view

    def compose(self) -> ComposeResult:
        model = self.view.model
        if model.output is not None:
            if model.output.lines or model.output.hidden_lines:
                yield OutputLines(self.view)
            if model.output.summary or model.output.expand_hint:
                yield OutputSummary(self.view)
        elif model.result is not None:
            yield ResultLine(self.view)


class ToolBlockView(Vertical):
    """One tool call's whole lifecycle: header, then its result rows inside
    the `│` gutter. `apply()` swaps the model and rebuilds the gutter, so a
    pending call becoming denied mutates in place."""

    def __init__(self, model: vm.ToolBlock) -> None:
        super().__init__(classes="block")
        self.model = model
        self.expanded = False

    def compose(self) -> ComposeResult:
        yield ToolHeader(self)
        if self._has_gutter():
            yield GutterBox(self)

    def _has_gutter(self) -> bool:
        return self.model.result is not None or self.model.output is not None

    async def apply(self, model: vm.ToolBlock) -> None:
        self.model = model
        for child in list(self.query(GutterBox)):
            await child.remove()
        if self._has_gutter():
            await self.mount(GutterBox(self))
        self.query_one(ToolHeader).refresh()

    def toggle_expanded(self) -> None:
        output = self.model.output
        if output is None or not output.hidden_lines:
            return
        self.expanded = not self.expanded
        for line in self.query(OutputLines):
            line.refresh(layout=True)
        for summary in self.query(OutputSummary):
            summary.refresh()


# ── list block ────────────────────────────────────────────────────────────────


class ListBlockView(_Line):
    """Label row + glyph rows: plan, active skills, context set, consumers."""

    def __init__(self, model: vm.ListBlock) -> None:
        super().__init__(classes="block")
        self.model = model

    def render(self) -> Text:
        tokens = self.tokens
        text = Text()
        if self.model.label:
            text.append_text(spans(self.model.label, tokens, Style(color=tokens.faint)))
        for index, row in enumerate(self.model.rows):
            if text or index:
                text.append("\n")
            text.append_text(self._row(row, tokens))
        return text

    def _row(self, row: vm.ListRow, tokens: Tokens) -> Text:
        glyph_color = tokens.accent if row.glyph in ("active", "included") else tokens.faint
        row_color = {
            "done": tokens.faint,
            "pending": tokens.faint,
            "active": tokens.foreground,
            "included": tokens.foreground,
            "none": tokens.muted,
        }[row.glyph]
        gap = "  " if self.model.column else " "
        text = Text(no_wrap=True, overflow="ellipsis")
        text.append(LIST_GLYPHS[row.glyph], style=Style(color=glyph_color))
        text.append(gap)
        body = row.text
        if self.model.column and row.annotation:
            body = body.ljust(self.model.column)
        text.append_text(spans(body, tokens, Style(color=row_color, strike=row.strike or None)))
        if row.annotation:
            text.append("  ")
            text.append_text(spans(row.annotation, tokens, Style(color=tokens.faint)))
        return text

    def apply(self, model: vm.ListBlock) -> None:
        self.model = model
        self.refresh(layout=True)


# ── diff block ────────────────────────────────────────────────────────────────


class DiffLineRow(_Line):
    """`[5: line no][3: sign][code]`; row colors and backgrounds via TCSS."""

    def __init__(self, line: vm.DiffLine) -> None:
        classes = {"+": "-add", "-": "-del", None: None}[line.sign]
        super().__init__(classes=classes)
        self.line = line

    def render(self) -> Text:
        sign = {"+": " + ", "-": f" {MINUS} ", None: "   "}[self.line.sign]
        number = "" if self.line.num is None else str(self.line.num)
        return Text(
            f"{number:>5}{sign}{self.line.text}",
            no_wrap=True,
            overflow="ellipsis",
        )


class DiffBlockView(Vertical):
    def __init__(self, model: vm.DiffBlock) -> None:
        super().__init__(classes="block")
        self.model = model

    def compose(self) -> ComposeResult:
        for line in self.model.lines:
            yield DiffLineRow(line)


# ── subagent task block (vocabulary extension) ────────────────────────────────

_TASK_STATUS_TOKEN = {
    "waiting": "faint",
    "running": "accent",
    "done": "success",
    "failed": "error",
}


class TaskLabel(_Line):
    def __init__(self, view: TaskBlockView) -> None:
        super().__init__()
        self.view = view

    def render(self) -> Text:
        tokens = self.tokens
        text = Text(no_wrap=True, overflow="ellipsis")
        text.append("task · ", style=Style(color=tokens.faint))
        text.append(self.view.status_model.description, style=Style(color=tokens.foreground))
        text.append(" · ", style=Style(color=tokens.faint))
        status = self.view.status_model.status
        color = getattr(tokens, _TASK_STATUS_TOKEN[status]) if status != "running" else tokens.accent
        text.append(status, style=Style(color=color))
        return text


class TaskBlockView(Vertical):
    """A subagent's conversation: `task · description · status` label, then
    the child's own blocks in a gutter. Children mount live via `body`."""

    def __init__(self, model: vm.TaskBlock) -> None:
        super().__init__(classes="block")
        self.status_model = model

    def compose(self) -> ComposeResult:
        yield TaskLabel(self)
        with Vertical(classes="task-children"):
            for block in self.status_model.blocks:
                yield build_block(block)

    @property
    def body(self) -> Vertical:
        return self.query_one(".task-children", Vertical)

    def set_status(self, status: vm.TaskStatus) -> None:
        self.status_model = self.status_model.model_copy(update={"status": status})
        self.query_one(TaskLabel).refresh()


# ── factory ───────────────────────────────────────────────────────────────────


def build_block(model: vm.Block) -> Widget:
    """One view-model block → its widget. The single mapping every producer
    (fixtures, replay, live events) goes through."""
    match model:
        case vm.UserBlock():
            return UserTurn(model.text)
        case vm.ThinkingBlock():
            return ThinkingLine(model)
        case vm.TextBlock():
            return AssistantText(model)
        case vm.ToolBlock():
            return ToolBlockView(model)
        case vm.ListBlock():
            return ListBlockView(model)
        case vm.DiffBlock():
            return DiffBlockView(model)
        case vm.NoticeBlock():
            return NoticeLine(model.text, error=model.error)
        case vm.TaskBlock():
            return TaskBlockView(model)
    raise ValueError(f"unknown block: {model!r}")  # pragma: no cover
