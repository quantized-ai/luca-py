"""The modal screens: sessions (1i), settings (1j), cost (1k), mcp (1l).

Presentation-first: each screen renders its `state` view-model and posts
intent messages (`Resume`, `Adjusted`, `CompactRequested`, …) for the app to
act on — in the gallery nothing listens and the screens are static, which is
exactly what makes every state renderable from a fixture.
"""

from __future__ import annotations

from typing import ClassVar

from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Static

from . import state as vm
from .chrome import HintLegend
from .format import spans
from .shells import LucaModalScreen, SelectRow, SpanLine
from .theme import Tokens, resolve_tokens

# ── sessions (1i) ─────────────────────────────────────────────────────────────

# (name, width, right-aligned) — drop order below the design width:
# TOKENS first, then TURNS.
SESSION_COLUMNS = (
    ("WHEN", 11, False),
    ("FIRST MESSAGE", 36, False),
    ("TURNS", 7, True),
    ("TOKENS", 8, True),
    ("COST", 7, True),
)


def _visible_columns(width: int) -> list[tuple[str, int, bool]]:
    columns = list(SESSION_COLUMNS)
    needed = sum(w for _, w, _ in columns) + 2 * (len(columns) - 1) + 4
    if width >= needed:
        return columns
    columns = [c for c in columns if c[0] != "TOKENS"]
    needed = sum(w for _, w, _ in columns) + 2 * (len(columns) - 1) + 4
    if width >= needed:
        return columns
    return [c for c in columns if c[0] != "TURNS"]


def _column_cells(row: vm.SessionRow) -> dict[str, str]:
    return {
        "WHEN": row.when,
        "FIRST MESSAGE": row.first_message,
        "TURNS": row.turns,
        "TOKENS": row.tokens,
        "COST": row.cost,
    }


def _fit(value: str, width: int, right: bool) -> str:
    if len(value) > width:
        value = value[: width - 1] + "…"
    return value.rjust(width) if right else value.ljust(width)


class SessionsHeader(Static):
    def __init__(self, screen: SessionsScreen) -> None:
        super().__init__(markup=False, classes="table-header")
        self._screen = screen

    def render(self) -> Text:
        cells = [_fit(name, width, right) for name, width, right in _visible_columns(self.size.width)]
        return Text("  " + "  ".join(cells), no_wrap=True, overflow="ellipsis")


class SessionRowView(SelectRow):
    has_caret: ClassVar[bool] = False

    def __init__(self, index: int, row: vm.SessionRow, *, selected: bool) -> None:
        super().__init__(selected=selected)
        self.index = index
        self.row = row

    def body(self, tokens: Tokens) -> Text:
        values = _column_cells(self.row)
        text = Text()
        for name, width, right in _visible_columns(self.size.width):
            value = _fit(values[name], width, right)
            if name in ("WHEN", "FIRST MESSAGE"):
                color = tokens.foreground if self.selected else tokens.muted
            else:
                color = tokens.muted if self.selected else tokens.faint
            text.append(value, style=Style(color=color))
            text.append("  ")
        text.rstrip()
        return text


class PreviewGutter(Static):
    """`preview · last turn` rows in the `│` gutter, `$text-muted`."""

    def __init__(self, lines: list[str]) -> None:
        super().__init__(markup=False, classes="gutter")
        self.lines = lines

    def set_lines(self, lines: list[str]) -> None:
        self.lines = lines
        self.refresh(layout=True)

    def render(self) -> Text:
        tokens = resolve_tokens(self.app)
        text = Text()
        for index, line in enumerate(self.lines):
            if index:
                text.append("\n")
            rendered = spans(line, tokens, Style(color=tokens.muted))
            rendered.no_wrap = True
            rendered.overflow = "ellipsis"
            text.append_text(rendered)
        return text


class SessionsScreen(LucaModalScreen):
    """`↑↓ move · enter resume · f fork · d delete · esc back`. Delete asks
    for a second `d` (the confirm), `esc` clears it."""

    class Highlighted(Message):
        def __init__(self, screen: SessionsScreen, index: int) -> None:
            super().__init__()
            self.screen = screen
            self.index = index

    class Resume(Message):
        def __init__(self, screen: SessionsScreen, row: vm.SessionRow) -> None:
            super().__init__()
            self.screen = screen
            self.row = row

    class Fork(Message):
        def __init__(self, screen: SessionsScreen, row: vm.SessionRow) -> None:
            super().__init__()
            self.screen = screen
            self.row = row

    class Delete(Message):
        def __init__(self, screen: SessionsScreen, row: vm.SessionRow) -> None:
            super().__init__()
            self.screen = screen
            self.row = row

    def __init__(self, state: vm.SessionsState, status: vm.StatusState, hints: list[str]) -> None:
        super().__init__(status, hints)
        self.state = state
        self._pending_delete = False

    def compose_body(self) -> ComposeResult:
        yield SpanLine(self.state.count_line, classes="faint-line count-line")
        with Vertical(classes="sessions-table"):
            yield SessionsHeader(self)
            for index, row in enumerate(self.state.rows):
                yield SessionRowView(index, row, selected=index == self.state.selected)
        yield SpanLine("preview · last turn", classes="faint-line preview-label")
        yield PreviewGutter(self.state.preview)

    def update_preview(self, lines: list[str]) -> None:
        self.query_one(PreviewGutter).set_lines(lines)

    def _move(self, delta: int) -> None:
        count = len(self.state.rows)
        if count == 0:
            return
        selected = (self.state.selected + delta) % count
        self.state = self.state.model_copy(update={"selected": selected})
        for row in self.query(SessionRowView):
            row.set_selected(row.index == selected)
        self._pending_delete = False
        self.post_message(self.Highlighted(self, selected))

    def _selected_row(self) -> vm.SessionRow | None:
        if not self.state.rows:
            return None
        return self.state.rows[self.state.selected]

    async def on_key(self, event: events.Key) -> None:
        key = event.key
        row = self._selected_row()
        if key == "up":
            self._move(-1)
        elif key == "down":
            self._move(1)
        elif key == "enter" and row is not None:
            self.post_message(self.Resume(self, row))
        elif key == "f" and row is not None:
            self.post_message(self.Fork(self, row))
        elif key == "d" and row is not None:
            if self._pending_delete:
                self._pending_delete = False
                self.post_message(self.Delete(self, row))
            else:
                self._pending_delete = True
                self.notify_delete_pending()
        elif key == "escape" and self._pending_delete:
            self._pending_delete = False
            self.notify_delete_cleared()
        else:
            return
        event.stop()
        event.prevent_default()

    def notify_delete_pending(self) -> None:
        self.query_one(HintLegend).set_hints(["d again delete", "esc keep"])

    def notify_delete_cleared(self) -> None:
        self.query_one(HintLegend).set_hints(self._hints)


# ── mcp (1l) ──────────────────────────────────────────────────────────────────

MCP_LABEL_COLUMN = 18
MCP_TOOL_COLUMN = 26

# The dot beside each server. Colour carries the state, so the row reads at a
# glance and "not authenticated" never looks like a failure. `stale` sits muted
# between the green and the red: its tools still work.
MCP_STATE_COLORS: dict[str, str] = {
    "connected": "success",
    "stale": "muted",
    "connecting": "accent",
    "needs_auth": "accent",
    "inactive": "error",
    "disabled": "faint",
}

# Which states have tools to show. Enter drills into these and acts on the rest.
MCP_HAS_TOOLS = ("connected", "stale")


class McpRowView(SelectRow):
    has_caret: ClassVar[bool] = False

    def __init__(self, index: int, row: vm.McpRow, *, selected: bool) -> None:
        super().__init__(selected=selected)
        self.index = index
        self.row = row

    def body(self, tokens: Tokens) -> Text:
        colour = tokens.accent if self.row.busy else getattr(tokens, MCP_STATE_COLORS[self.row.state], tokens.muted)
        text = Text()
        text.append("● ", style=Style(color=colour))
        text.append(self.row.label.ljust(MCP_LABEL_COLUMN), style=Style(color=tokens.foreground))
        text.append(
            self.row.detail,
            style=Style(color=tokens.muted if self.selected or self.row.busy else tokens.faint),
        )
        if self.row.state in MCP_HAS_TOOLS and not self.row.busy:
            text.append("  ▸", style=Style(color=tokens.faint))
        text.no_wrap = True
        text.overflow = "ellipsis"
        return text


class McpToolRowView(SelectRow):
    has_caret: ClassVar[bool] = False

    def __init__(self, index: int, row: vm.McpToolRow, *, selected: bool) -> None:
        super().__init__(selected=selected)
        self.index = index
        self.row = row

    def body(self, tokens: Tokens) -> Text:
        excluded = self.row.excluded is not None
        text = Text()
        # Marked, not merely dimmed: a greyed row reads as a tool the model can
        # still call.
        text.append("✗ " if excluded else "  ", style=Style(color=tokens.error))
        text.append(
            # `_fit` truncates as well as pads: real servers ship names longer
            # than the column, and one of them must not shift every summary.
            _fit(self.row.name, MCP_TOOL_COLUMN, False),
            style=Style(color=tokens.faint if excluded else tokens.foreground),
        )
        text.append(
            self.row.excluded or self.row.summary,
            style=Style(color=tokens.muted if self.selected else tokens.faint),
        )
        text.no_wrap = True
        text.overflow = "ellipsis"
        return text


class McpScreen(LucaModalScreen):
    """`↑↓ move · enter tools · a authenticate · r reconnect · d disable`.

    Two views on one screen, the server list and one server's tools, so `esc`
    walks back out the way it came in.

    An action does NOT dismiss the screen: authorizing waits on a human in a
    browser, and the row saying `authorizing…` while `esc` cancels is the thing
    a dismissed screen could not do.
    """

    class Authenticate(Message):
        def __init__(self, screen: McpScreen, row: vm.McpRow) -> None:
            super().__init__()
            self.screen = screen
            self.row = row

    class Reconnect(Message):
        def __init__(self, screen: McpScreen, row: vm.McpRow) -> None:
            super().__init__()
            self.screen = screen
            self.row = row

    class Toggle(Message):
        def __init__(self, screen: McpScreen, row: vm.McpRow) -> None:
            super().__init__()
            self.screen = screen
            self.row = row

    class Inspect(Message):
        """Show me what this server actually gives the model."""

        def __init__(self, screen: McpScreen, row: vm.McpRow) -> None:
            super().__init__()
            self.screen = screen
            self.row = row

    class Cancelled(Message):
        """`esc` while an action is in flight."""

        def __init__(self, screen: McpScreen) -> None:
            super().__init__()
            self.screen = screen

    def __init__(self, state: vm.McpState, status: vm.StatusState, hints: list[str]) -> None:
        super().__init__(status, hints)
        self.state = state

    def compose_body(self) -> ComposeResult:
        if self.state.detail is not None:
            yield from self._compose_detail(self.state.detail)
            return
        yield SpanLine(self.state.count_line, classes="faint-line count-line")
        with Vertical(classes="mcp-table"):
            for index, row in enumerate(self.state.rows):
                yield McpRowView(index, row, selected=index == self.state.selected)
        if self.state.message:
            error = " mcp-message--error" if self.state.message_is_error else ""
            yield SpanLine(self.state.message, classes=f"faint-line mcp-message{error}")

    def _compose_detail(self, detail: vm.McpDetail) -> ComposeResult:
        yield SpanLine(f"{detail.label} · {detail.count_line}", classes="faint-line count-line")
        with Vertical(classes="mcp-table"):
            for index, row in enumerate(detail.rows):
                yield McpToolRowView(index, row, selected=index == detail.selected)

    @property
    def busy(self) -> bool:
        return any(row.busy for row in self.state.rows)

    async def set_state(self, state: vm.McpState, hints: list[str] | None = None) -> None:
        """Redraw in place. The screen stays up across an action, so this is how
        an outcome reaches it."""
        self.state = state
        if hints is not None:
            self._hints = hints
        await self.recompose()

    def _move(self, delta: int) -> None:
        detail = self.state.detail
        rows = detail.rows if detail is not None else self.state.rows
        if not rows:
            return
        if detail is not None:
            moved = detail.model_copy(update={"selected": (detail.selected + delta) % len(rows)})
            self.state = self.state.model_copy(update={"detail": moved})
            for view in self.query(McpToolRowView):
                view.set_selected(view.index == moved.selected)
            return
        selected = (self.state.selected + delta) % len(rows)
        self.state = self.state.model_copy(update={"selected": selected})
        for view in self.query(McpRowView):
            view.set_selected(view.index == selected)

    def _selected_row(self) -> vm.McpRow | None:
        if not self.state.rows:
            return None
        return self.state.rows[self.state.selected]

    async def on_key(self, event: events.Key) -> None:
        if await self._handle_key(event.key):
            event.stop()
            event.prevent_default()

    async def _handle_key(self, key: str) -> bool:
        if key in ("up", "down"):
            self._move(-1 if key == "up" else 1)
            return True
        if key == "escape":
            # The drill-in first, then an action in flight, then the screen.
            if self.state.detail is not None:
                await self.set_state(self.state.model_copy(update={"detail": None}))
                return True
            if self.busy:
                self.post_message(self.Cancelled(self))
                return True
            return False  # the base screen dismisses
        if self.state.detail is not None or self.busy:
            return False  # nothing else acts on a server from here
        row = self._selected_row()
        if row is None:
            return False
        if key == "enter":
            self.post_message(self._enter_action(row))
        elif key == "a":
            self.post_message(self.Authenticate(self, row))
        elif key == "r":
            self.post_message(self.Reconnect(self, row))
        elif key == "d":
            self.post_message(self.Toggle(self, row))
        else:
            return False
        return True

    def _enter_action(self, row: vm.McpRow) -> Message:
        """Whatever the row's own `action` names, because the useful thing to do
        to a server depends on the state it is in."""
        if row.state in MCP_HAS_TOOLS:
            return self.Inspect(self, row)
        if row.state == "disabled":
            return self.Toggle(self, row)
        if row.state == "needs_auth":
            return self.Authenticate(self, row)
        return self.Reconnect(self, row)


# ── settings (1j) ─────────────────────────────────────────────────────────────

_VALUE_COLORS = {
    "accent": "accent",
    "error": "error",
    "muted": "muted",
    "foreground": "foreground",
}

SETTINGS_NAME_COLUMN = 24


class SettingRowView(SelectRow):
    def __init__(self, index: int, row: vm.SettingRow, *, selected: bool) -> None:
        super().__init__(selected=selected)
        self.index = index
        self.row = row

    def body(self, tokens: Tokens) -> Text:
        text = Text()
        name_color = tokens.foreground if self.selected else tokens.muted
        text.append(self.row.name.ljust(SETTINGS_NAME_COLUMN), style=Style(color=name_color))
        if self.row.color is not None:
            value_color = getattr(tokens, _VALUE_COLORS[self.row.color])
        else:
            value_color = tokens.foreground if self.selected else tokens.muted
        text.append(self.row.value, style=Style(color=value_color))
        return text

    def trailing(self, tokens: Tokens) -> Text | None:
        if self.selected:
            return Text("← →", style=Style(color=tokens.faint))
        return None


class SwatchLine(Static):
    """`luca-dark  ▕▏██ ██ …` — the seven theme colors in scale order. The
    first swatch is the background color, drawn as a hairline outline pair so
    it stays visible on itself."""

    def __init__(self, label: str) -> None:
        super().__init__(markup=False, classes="swatch-line")
        self.label = label

    def render(self) -> Text:
        tokens = resolve_tokens(self.app)
        text = Text(no_wrap=True)
        text.append(self.label, style=Style(color=tokens.faint))
        text.append("  ")
        # Left eighth-block first, right eighth-block last: the hairline
        # strokes sit at the swatch's OUTER edges, outlining it.
        text.append("▏▕", style=Style(color=tokens.hairline))
        for color in (
            tokens.rule,
            tokens.muted,
            tokens.foreground,
            tokens.accent,
            tokens.success,
            tokens.error,
        ):
            text.append(" ")
            text.append("██", style=Style(color=color))
        return text


class SettingsScreen(LucaModalScreen):
    """Three labelled groups; `← →` adjusts the focused row, `esc` saves and
    closes. Value edits are posted as intents — the app owns what is real."""

    class Adjusted(Message):
        def __init__(self, screen: SettingsScreen, row: vm.SettingRow, delta: int) -> None:
            super().__init__()
            self.screen = screen
            self.row = row
            self.delta = delta

    class Closed(Message):
        def __init__(self, screen: SettingsScreen) -> None:
            super().__init__()
            self.screen = screen

    def __init__(self, state: vm.SettingsState, status: vm.StatusState, hints: list[str]) -> None:
        super().__init__(status, hints)
        self.state = state

    def action_close_modal(self) -> None:
        # esc is "save & close": the app hears Closed and writes through.
        self.post_message(self.Closed(self))
        self.dismiss(None)

    def _flat_rows(self) -> list[vm.SettingRow]:
        return [row for group in self.state.groups for row in group.rows]

    def compose_body(self) -> ComposeResult:
        index = 0
        for group in self.state.groups:
            yield SpanLine(group.label, classes="faint-line group-label")
            for row in group.rows:
                yield SettingRowView(index, row, selected=index == self.state.selected)
                index += 1
            yield Static(classes="group-gap")
        if self.state.swatch_label:
            yield SwatchLine(self.state.swatch_label)

    def set_state(self, state: vm.SettingsState) -> None:
        """Re-render values in place after the app applied an adjustment."""
        self.state = state
        rows = self._flat_rows()
        for view in self.query(SettingRowView):
            view.row = rows[view.index]
            view.set_selected(view.index == state.selected)

    def _move(self, delta: int) -> None:
        count = len(self._flat_rows())
        if count == 0:
            return
        selected = (self.state.selected + delta) % count
        self.state = self.state.model_copy(update={"selected": selected})
        for view in self.query(SettingRowView):
            view.set_selected(view.index == selected)

    async def on_key(self, event: events.Key) -> None:
        key = event.key
        rows = self._flat_rows()
        if key == "up":
            self._move(-1)
        elif key == "down":
            self._move(1)
        elif key in ("left", "right") and rows:
            delta = -1 if key == "left" else 1
            self.post_message(self.Adjusted(self, rows[self.state.selected], delta))
        elif key == "enter" and rows:
            self.post_message(self.Adjusted(self, rows[self.state.selected], 1))
        else:
            return
        event.stop()
        event.prevent_default()


# ── cost (1k) ─────────────────────────────────────────────────────────────────

METER_COLUMNS = (18, 8, 8)
CONTEXT_BAR_WIDTH = 60

_METER_COLORS = {
    "accent": "accent",
    "foreground": "foreground",
    "faint": "faint",
    "rule": "rule",
    "hairline": "hairline",
}


class CostHeadline(Static):
    def __init__(self, state: vm.CostState) -> None:
        super().__init__(markup=False, classes="cost-headline")
        self.state = state

    def render(self) -> Text:
        tokens = resolve_tokens(self.app)
        text = Text(no_wrap=True)
        text.append(self.state.headline, style=Style(color=tokens.foreground, bold=True))
        text.append("  ")
        text.append(self.state.subline, style=Style(color=tokens.faint))
        return text


class MeterRow(Static):
    """`input   26.3k  $0.079  ███…` — the meter is proportional to COST."""

    def __init__(self, item: vm.CostItem) -> None:
        super().__init__(markup=False, classes="meter-row")
        self.item = item

    def render(self) -> Text:
        tokens = resolve_tokens(self.app)
        label_w, tokens_w, cost_w = METER_COLUMNS
        text = Text(no_wrap=True)
        text.append(self.item.label.ljust(label_w), style=Style(color=tokens.muted))
        text.append(self.item.tokens.rjust(tokens_w), style=Style(color=tokens.foreground))
        text.append("  ")
        text.append(self.item.cost.rjust(cost_w), style=Style(color=tokens.faint))
        text.append("  ")
        available = self.size.width - text.cell_len
        filled = max(0, round(available * max(0.0, min(self.item.fraction, 1.0))))
        color = getattr(tokens, _METER_COLORS[self.item.color])
        text.append("█" * filled, style=Style(color=color))
        return text


class ContextWindowView(Static):
    """The stacked 60-cell context bar with its usage line and legend."""

    def __init__(self, state: vm.ContextWindowState) -> None:
        super().__init__(markup=False, classes="context-window")
        self.state = state

    def render(self) -> Text:
        tokens = resolve_tokens(self.app)
        text = Text()
        text.append("context window", style=Style(color=tokens.faint))
        text.append("\n")
        text.append(self.state.used, style=Style(color=tokens.muted))
        text.append("  ")
        text.append(self.state.percent, style=Style(color=tokens.faint))
        text.append("\n")
        width = min(CONTEXT_BAR_WIDTH, max(self.size.width, 10))
        context_cells = round(width * max(0.0, min(self.state.context_fraction, 1.0)))
        reply_cells = round(width * max(0.0, min(self.state.reply_fraction, 1.0)))
        free_cells = max(0, width - context_cells - reply_cells)
        text.append("█" * context_cells, style=Style(color=tokens.accent))
        text.append("█" * reply_cells, style=Style(color=tokens.foreground))
        text.append("█" * free_cells, style=Style(color=tokens.rule))
        text.append("\n")
        for index, item in enumerate(self.state.legend):
            if index:
                text.append("   ")
            text.append_text(spans(item, tokens, Style(color=tokens.faint)))
        return text


class ConsumersView(Static):
    """`biggest consumers` — 36-cell label column, faint token counts."""

    def __init__(self, consumers: list[vm.ConsumerRow]) -> None:
        super().__init__(markup=False, classes="consumers")
        self.consumers = consumers

    def render(self) -> Text:
        tokens = resolve_tokens(self.app)
        text = Text()
        text.append("biggest consumers", style=Style(color=tokens.faint))
        for row in self.consumers:
            text.append("\n")
            line = Text(no_wrap=True, overflow="ellipsis")
            line.append(row.label.ljust(36), style=Style(color=tokens.muted))
            line.append("  ")
            line.append(row.tokens, style=Style(color=tokens.faint))
            text.append_text(line)
        return text


class CostScreen(LucaModalScreen):
    class CompactRequested(Message):
        def __init__(self, screen: CostScreen) -> None:
            super().__init__()
            self.screen = screen

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+k", "compact", show=False),
    ]

    def __init__(self, state: vm.CostState, status: vm.StatusState, hints: list[str]) -> None:
        super().__init__(status, hints)
        self.state = state

    def compose_body(self) -> ComposeResult:
        yield CostHeadline(self.state)
        with Vertical(classes="cost-items"):
            for item in self.state.items:
                yield MeterRow(item)
        if self.state.context is not None:
            yield ContextWindowView(self.state.context)
        if self.state.consumers:
            yield ConsumersView(self.state.consumers)

    def action_compact(self) -> None:
        self.post_message(self.CompactRequested(self))
