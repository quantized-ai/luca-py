"""The shared interactive shells: selection rows, the approval prompt, the
overlay list, and the modal-screen base.

The selection treatment exists ONCE: `SelectRow` allocates the 1-cell accent
bar and the caret column permanently (moving the highlight only changes color
and background — never layout), and the `.select-row.-selected` rule in
`app.tcss` owns the background. Approval options, palette rows, picker rows,
settings rows and session rows are all `SelectRow` subclasses supplying their
column layout through `body()`/`trailing()`.
"""

from __future__ import annotations

from typing import ClassVar

from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Static

from . import state as vm
from .chrome import HintLegend, StatusBar
from .format import GLYPH_BAR, GLYPH_CARET, GLYPH_CURSOR, spans
from .theme import Tokens, resolve_tokens

PALETTE_COLUMN = 13
PICKER_COLUMN = 36


class SpanLine(Static):
    """A one-line Static rendering theme-token spans; `$text-faint` default
    unless a class overrides the color."""

    def __init__(self, text: str, *, classes: str | None = None) -> None:
        super().__init__(markup=False, classes=classes)
        self.text_value = text

    def set_text(self, text: str) -> None:
        self.text_value = text
        self.refresh(layout=True)

    def render(self) -> Text:
        rendered = spans(self.text_value, resolve_tokens(self.app))
        rendered.no_wrap = True
        rendered.overflow = "ellipsis"
        return rendered


class SelectRow(Static):
    """[1-cell bar][1-cell caret][2 spaces][subclass columns]  [trailing].

    Both reserved columns are always drawn (blank when unselected), so
    selection never reflows. `has_caret=False` collapses the caret column to
    one space — the sessions table's shape."""

    has_caret: ClassVar[bool] = True

    def __init__(self, *, selected: bool = False, classes: str | None = None) -> None:
        merged = "select-row" + (f" {classes}" if classes else "")
        super().__init__(markup=False, classes=merged)
        self._selected = selected
        self.set_class(selected, "-selected")

    @property
    def selected(self) -> bool:
        return self._selected

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.set_class(selected, "-selected")
        self.refresh()

    def body(self, tokens: Tokens) -> Text:  # pragma: no cover - overridden
        return Text()

    def trailing(self, tokens: Tokens) -> Text | None:
        return None

    def render(self) -> Text:
        tokens = resolve_tokens(self.app)
        text = Text(no_wrap=True, overflow="ellipsis")
        text.append(GLYPH_BAR if self._selected else " ", style=Style(color=tokens.accent))
        if self.has_caret:
            caret = GLYPH_CARET if self._selected else " "
            text.append(caret, style=Style(color=tokens.accent))
            text.append("  ")
        else:
            text.append(" ")
        text.append_text(self.body(tokens))
        right = self.trailing(tokens)
        if right:
            pad = self.size.width - text.cell_len - right.cell_len - 1
            if pad >= 1:
                text.append(" " * pad)
                text.append_text(right)
        return text


# ── approval prompt (1c, 1h) ──────────────────────────────────────────────────


class ApprovalOptionRow(SelectRow):
    def __init__(self, index: int, option: vm.ApprovalOption, *, selected: bool) -> None:
        super().__init__(selected=selected)
        self.index = index
        self.option = option

    def body(self, tokens: Tokens) -> Text:
        text = Text()
        text.append(str(self.index + 1), style=Style(color=tokens.faint))
        text.append("  ")
        color = tokens.foreground if self.selected else tokens.muted
        text.append(self.option.label, style=Style(color=color))
        return text

    def trailing(self, tokens: Tokens) -> Text | None:
        if self.option.key_hint:
            return Text(self.option.key_hint, style=Style(color=tokens.faint))
        return None


class ApprovalPromptView(Vertical, can_focus=True):
    """The bordered prompt that replaces the composer. Digits pick directly,
    `esc` selects the LAST option (Cancel turn) — it never merely dismisses."""

    class Decided(Message):
        def __init__(self, view: ApprovalPromptView, index: int) -> None:
            super().__init__()
            self.view = view
            self.index = index

    def __init__(self, model: vm.ApprovalState) -> None:
        super().__init__()
        self.model = model

    def compose(self) -> ComposeResult:
        yield SpanLine(self.model.question, classes="prompt-question")
        yield Static(classes="inner-rule")
        with Vertical(classes="prompt-options"):
            for index, option in enumerate(self.model.options):
                yield ApprovalOptionRow(index, option, selected=index == self.model.selected)

    def _move(self, delta: int) -> None:
        count = len(self.model.options)
        selected = (self.model.selected + delta) % count
        self.model = self.model.model_copy(update={"selected": selected})
        for row in self.query(ApprovalOptionRow):
            row.set_selected(row.index == selected)

    def _decide(self, index: int) -> None:
        self.post_message(self.Decided(self, index))

    async def on_key(self, event: events.Key) -> None:
        key = event.key
        if key == "up":
            self._move(-1)
        elif key == "down":
            self._move(1)
        elif key == "enter":
            self._decide(self.model.selected)
        elif key == "escape":
            self._decide(len(self.model.options) - 1)
        elif key.isdigit() and 1 <= int(key) <= len(self.model.options):
            self._decide(int(key) - 1)
        else:
            return
        event.stop()
        event.prevent_default()


# ── overlay lists (1e palette, 1f context picker) ─────────────────────────────


class PaletteRow(SelectRow):
    def __init__(self, row: vm.OverlayRow, *, column: int, selected: bool) -> None:
        super().__init__(selected=selected)
        self.row = row
        self.column = column

    def body(self, tokens: Tokens) -> Text:
        text = Text()
        primary = spans(self.row.primary, tokens, Style(color=tokens.foreground))
        text.append_text(primary)
        pad = self.column - primary.cell_len
        if pad > 0:
            text.append(" " * pad)
        text.append("  ")
        if self.row.secondary:
            text.append_text(spans(self.row.secondary, tokens, Style(color=tokens.muted)))
        return text


class PickerRow(SelectRow):
    def __init__(self, row: vm.OverlayRow, *, column: int, selected: bool) -> None:
        super().__init__(selected=selected)
        self.row = row
        self.column = column

    def body(self, tokens: Tokens) -> Text:
        text = Text()
        checkbox = "☑" if self.row.checked else "☐"
        text.append(checkbox, style=Style(color=tokens.faint))
        text.append("  ")
        color = tokens.foreground if self.selected else tokens.muted
        primary = spans(self.row.primary, tokens, Style(color=color))
        text.append_text(primary)
        pad = self.column - primary.cell_len
        if pad > 0:
            text.append(" " * pad)
        text.append("  ")
        if self.row.annotation:
            text.append(self.row.annotation, style=Style(color=tokens.faint))
        return text


class QueryLine(Static):
    """`/ s█` … right-aligned `6 of 14`."""

    def __init__(self, view: OverlayListView) -> None:
        super().__init__(markup=False, classes="overlay-query")
        self.view = view

    def render(self) -> Text:
        tokens = resolve_tokens(self.view.app)
        model = self.view.model
        text = Text(no_wrap=True)
        text.append(model.sigil, style=Style(color=tokens.accent))
        text.append(" ")
        text.append(model.query, style=Style(color=tokens.foreground))
        text.append(GLYPH_CURSOR, style=Style(color=tokens.foreground))
        if model.counter:
            right = Text(model.counter, style=Style(color=tokens.faint))
            pad = self.size.width - text.cell_len - right.cell_len
            if pad >= 1:
                text.append(" " * pad)
                text.append_text(right)
        return text


class OverlayListView(Vertical, can_focus=True):
    """Palette / context picker / generic menu: option rows above, an inner
    rule, then the query row. Replaces the composer in place; the app owns
    filtering and answers `QueryChanged` with a fresh state."""

    class QueryChanged(Message):
        def __init__(self, view: OverlayListView, value: str) -> None:
            super().__init__()
            self.view = view
            self.value = value

    class Toggled(Message):
        def __init__(self, view: OverlayListView, index: int) -> None:
            super().__init__()
            self.view = view
            self.index = index

    class Committed(Message):
        def __init__(self, view: OverlayListView, index: int) -> None:
            super().__init__()
            self.view = view
            self.index = index

    class Dismissed(Message):
        def __init__(self, view: OverlayListView) -> None:
            super().__init__()
            self.view = view

    def __init__(self, model: vm.OverlayState) -> None:
        super().__init__()
        self.model = model

    @property
    def _column(self) -> int:
        if self.model.column is not None:
            return self.model.column
        return PICKER_COLUMN if self.model.mode == "picker" else PALETTE_COLUMN

    def _row_widget(self, index: int, row: vm.OverlayRow) -> SelectRow:
        selected = index == self.model.selected
        if self.model.mode == "picker":
            return PickerRow(row, column=self._column, selected=selected)
        return PaletteRow(row, column=self._column, selected=selected)

    def compose(self) -> ComposeResult:
        with Vertical(classes="overlay-options"):
            for index, row in enumerate(self.model.rows):
                yield self._row_widget(index, row)
        yield Static(classes="inner-rule")
        yield QueryLine(self)

    async def set_state(self, model: vm.OverlayState) -> None:
        """Swap in a fresh state (new rows after a query change)."""
        self.model = model
        options = self.query_one(".overlay-options", Vertical)
        await options.remove_children()
        await options.mount(*(self._row_widget(index, row) for index, row in enumerate(self.model.rows)))
        self.query_one(QueryLine).refresh()

    def _move(self, delta: int) -> None:
        count = len(self.model.rows)
        if count == 0:
            return
        selected = (self.model.selected + delta) % count
        self.model = self.model.model_copy(update={"selected": selected})
        for index, row in enumerate(self.query(SelectRow)):
            row.set_selected(index == selected)

    def _set_query(self, value: str) -> None:
        self.model = self.model.model_copy(update={"query": value})
        self.query_one(QueryLine).refresh()
        self.post_message(self.QueryChanged(self, value))

    async def on_key(self, event: events.Key) -> None:
        key = event.key
        if key == "up":
            self._move(-1)
        elif key == "down":
            self._move(1)
        elif key == "enter":
            self.post_message(self.Committed(self, self.model.selected))
        elif key == "escape":
            self.post_message(self.Dismissed(self))
        elif key == "space" and self.model.mode == "picker":
            self.post_message(self.Toggled(self, self.model.selected))
        elif key == "backspace":
            if self.model.query:
                self._set_query(self.model.query[:-1])
            else:
                self.post_message(self.Dismissed(self))
        elif event.is_printable and event.character:
            self._set_query(self.model.query + event.character)
        else:
            return
        event.stop()
        event.prevent_default()


# ── modal screen base (1i, 1j, 1k) ────────────────────────────────────────────


class LucaModalScreen(ModalScreen[None]):
    """Full-screen modal keeping the global frame: status bar (mark + cwd +
    right label), the rule, a body, the hint legend. No composer. `esc`
    returns."""

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "close_modal", show=False)]

    def __init__(self, status: vm.StatusState, hints: list[str]) -> None:
        super().__init__()
        self._status = status
        self._hints = hints

    def compose(self) -> ComposeResult:
        yield StatusBar(self._status)
        yield Static(classes="rule")
        with VerticalScroll(id="modal-body"):
            yield from self.compose_body()
        yield HintLegend(self._hints)

    def compose_body(self) -> ComposeResult:  # pragma: no cover - overridden
        yield from ()

    def action_close_modal(self) -> None:
        self.dismiss(None)
