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
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Static

from . import state as vm
from .chrome import HintLegend, StatusBar
from .format import (
    GLYPH_BAR,
    GLYPH_BOX,
    GLYPH_BOX_TICKED,
    GLYPH_CARET,
    GLYPH_CURSOR,
    GLYPH_USER,
    spans,
)
from .prompt import PromptInput
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


# ── the question set (1l — the agent's `ask_user` prompt) ─────────────────────


class QuestionTabs(Static):
    """One line: `☑ storage   ☑ reader   ▎☐ backfill   ☐ rollout   …   3 of 4`.

    The tabs ARE the progress indicator — there is no percent field and no
    counter widget — and the active one is marked with the same `▎` bar every
    selected row uses."""

    def __init__(self, view: QuestionSetView) -> None:
        super().__init__(markup=False, classes="question-tabs")
        self.view = view

    def render(self) -> Text:
        tokens = resolve_tokens(self.view.app)
        model = self.view.model
        text = Text(no_wrap=True, overflow="ellipsis")
        for index, question in enumerate(model.questions):
            active = index == model.active
            if index:
                text.append("  ")
            # The bar cell is ALWAYS allocated, blank when inactive — the same
            # rule `SelectRow` follows, so moving between tabs never reflows.
            text.append(GLYPH_BAR if active else " ", style=Style(color=tokens.accent))
            box = GLYPH_BOX_TICKED if question.answered else GLYPH_BOX
            text.append(box, style=Style(color=tokens.accent if question.answered else tokens.faint))
            text.append(" ")
            text.append(question.tab, style=Style(color=tokens.foreground if active else tokens.faint))
        right = Text(
            f"{model.active + 1} of {len(model.questions)}",
            style=Style(color=tokens.faint),
        )
        pad = self.size.width - text.cell_len - right.cell_len
        if pad >= 1:
            text.append(" " * pad)
            text.append_text(right)
        return text


class QuestionOptionRow(SelectRow):
    """`[bar][caret]  N  [☑/☐]  label                              [hint]`.

    Two column rules, both standing:

    - the tick column is ALWAYS allocated in multi mode and NEVER in single
      mode, so toggling never reflows;
    - a `kind == "chat"` row renders the tick column as BLANK CELLS, not as a
      glyph — the label stays aligned with every other label, but there is
      nothing there to tick, because it is the way out and not an answer.

    The CUSTOM row is not one of these — it is `QuestionCustomRow`, which
    nests a real text field."""

    def __init__(
        self,
        index: int,
        option: vm.QuestionOption,
        *,
        multi: bool,
        selected: bool,
    ) -> None:
        super().__init__(selected=selected, classes="option-row")
        self.index = index
        self.option = option
        self.multi = multi

    def body(self, tokens: Tokens) -> Text:
        text = Text()
        text.append_text(option_columns(self.index, self.option, tokens, multi=self.multi))
        color = tokens.foreground if self.selected else tokens.muted
        text.append(self.option.label, style=Style(color=color))
        return text

    def trailing(self, tokens: Tokens) -> Text | None:
        if self.option.key_hint:
            return Text(self.option.key_hint, style=Style(color=tokens.faint))
        return None


def option_columns(index: int, option: vm.QuestionOption, tokens: Tokens, *, multi: bool) -> Text:
    """The columns every option row shares, before its label: `N  [☑/☐]  `.

    ONE function, because the plain rows and the composite custom row have to
    land on the same cell — two copies of this drift by a space and the labels
    stop lining up. Two column rules, both standing:

    - the tick column is ALWAYS allocated in multi mode and NEVER in single
      mode, so toggling never reflows;
    - a `kind == "chat"` row renders the tick column as BLANK CELLS, not as a
      glyph — the label stays aligned with every other label, but there is
      nothing there to tick, because it is the way out and not an answer."""
    text = Text()
    text.append(str(index + 1), style=Style(color=tokens.faint))
    text.append("  ")
    if multi:
        if option.kind == "chat":
            text.append("   ")  # blank cells: aligned, but nothing to tick
        else:
            # A typed custom answer IS a tick: the row counts as chosen the
            # moment it holds text.
            ticked = option.checked or bool(option.text)
            text.append(
                GLYPH_BOX_TICKED if ticked else GLYPH_BOX,
                style=Style(color=tokens.accent if ticked else tokens.faint),
            )
            text.append("  ")
    return text


class QuestionCustomPrefix(SelectRow):
    """The custom row's fixed columns — `[bar][caret]  N  [☑/☐]  Custom answer:`
    — drawn by the same `SelectRow` machinery as every other option row, so the
    two cannot drift apart by a cell."""

    def __init__(self, index: int, option: vm.QuestionOption, *, multi: bool, selected: bool) -> None:
        super().__init__(selected=selected, classes="custom-prefix")
        self.index = index
        self.option = option
        self.multi = multi

    def body(self, tokens: Tokens) -> Text:
        text = option_columns(self.index, self.option, tokens, multi=self.multi)
        text.append(self.option.label + " ", style=Style(color=tokens.faint))
        return text


class QuestionCustomRow(Horizontal):
    """The custom-answer row: the shared column prefix, then a REAL text field.

    A nested `PromptInput` rather than the hand-rolled inline editor this used
    to be. The moment an answer runs past the panel's width the row has to
    scroll to follow the cursor — and a scrolling single-line field with
    selection, word-jump, paste and undo is precisely what `TextArea` already
    is, correct about character widths in a way a hand-rolled one is not.

    `enter` commits (`PromptInput.Submitted`), `escape` leaves the field with
    the text intact, and `⇥` still moves between questions from inside it —
    `TextArea` binds none of those three, so they reach `QuestionSetView`
    exactly as they did before. `↑`/`↓` DO belong to the editor while it has
    the keyboard; you leave the field to move the caret again."""

    def __init__(self, index: int, option: vm.QuestionOption, *, multi: bool, selected: bool) -> None:
        super().__init__(classes="select-row option-row question-custom")
        self.index = index
        self.option = option
        self.multi = multi
        self._selected = selected
        self.set_class(selected, "-selected")

    def compose(self) -> ComposeResult:
        yield QuestionCustomPrefix(self.index, self.option, multi=self.multi, selected=self._selected)
        # `soft_wrap` off is what keeps this ONE ROW: a long answer scrolls
        # sideways under the cursor instead of wrapping the whole panel taller.
        field = PromptInput()
        field.soft_wrap = False
        yield field

    def on_mount(self) -> None:
        field = self.field
        if self.option.text:
            field.load_text(self.option.text)
            field.move_cursor(field.document.end)

    @property
    def field(self) -> PromptInput:
        return self.query_one(PromptInput)

    @property
    def selected(self) -> bool:
        return self._selected

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.set_class(selected, "-selected")
        self.query_one(QuestionCustomPrefix).set_selected(selected)


class QuestionSetView(Vertical, can_focus=True):
    """The agent's questions, replacing the composer exactly as the approval
    prompt does — same border, same inner rules, same `▎❯` selection.

    Four rows, top to bottom: the tab strip, the question title, the long body
    (full 99-column measure, up to 8 rows, scrolling beyond), and the options.
    Body ABOVE options rather than beside them, so prose and inline `code` read
    the way they do everywhere else; the cost is that the panel's height
    changes as you tab between questions, which is accepted.

    THE WIDGET DECIDES NOTHING. It reports what the user did and the app owns
    advancing the active tab, flipping `phase` and rebuilding the state — the
    widget never decides which question comes next, and never decides that the
    set is done."""

    class Answered(Message):
        """A single-select pick, or a multi-select commit."""

        def __init__(self, view: QuestionSetView, index: int) -> None:
            super().__init__()
            self.view = view
            self.index = index

    class Toggled(Message):
        def __init__(self, view: QuestionSetView, index: int) -> None:
            super().__init__()
            self.view = view
            self.index = index

    class CustomChanged(Message):
        def __init__(self, view: QuestionSetView, text: str) -> None:
            super().__init__()
            self.view = view
            self.text = text

    class ChatRequested(Message):
        def __init__(self, view: QuestionSetView) -> None:
            super().__init__()
            self.view = view

    class ConfirmRequested(Message):
        """`enter` on the last open question — open the confirmation."""

        def __init__(self, view: QuestionSetView) -> None:
            super().__init__()
            self.view = view

    class Moved(Message):
        """The caret or the active tab moved; the app re-derives the hints."""

        def __init__(self, view: QuestionSetView) -> None:
            super().__init__()
            self.view = view

    def __init__(self, model: vm.QuestionSetState) -> None:
        super().__init__()
        self.model = model

    # ── reads ────────────────────────────────────────────────────────────────

    @property
    def question(self) -> vm.Question:
        return self.model.questions[self.model.active]

    @property
    def _multi(self) -> bool:
        return self.question.mode == "multi"

    def compose(self) -> ComposeResult:
        yield QuestionTabs(self)
        yield Static(classes="inner-rule")
        yield SpanLine(self.question.title, classes="question-title")
        if self.question.body:
            yield Static(classes="inner-rule")
            with VerticalScroll(classes="question-body"):
                for line in self.question.body:
                    yield SpanLine(line)
        yield Static(classes="inner-rule")
        with Vertical(classes="question-options"):
            yield from self._option_rows()

    def _option_rows(self):
        question = self.question
        for index, option in enumerate(question.options):
            # A blank row before "Chat about this" — it is the exit, not one
            # more answer, and the gap is what says so.
            if option.kind == "chat":
                yield Static(classes="option-gap")
            kind = QuestionCustomRow if option.kind == "custom" else QuestionOptionRow
            yield kind(index, option, multi=self._multi, selected=index == question.selected)

    async def set_state(self, model: vm.QuestionSetState) -> None:
        """Swap in a fresh state. The panel is rebuilt wholesale because the
        body's height changes with the active tab — there is nothing to
        preserve across a tab move that the state does not already carry."""
        self.model = model
        await self.remove_children()
        await self.mount(*list(self.compose()))

    # ── caret ────────────────────────────────────────────────────────────────

    def _select(self, index: int) -> None:
        question = self.question
        count = len(question.options)
        index %= count
        questions = list(self.model.questions)
        questions[self.model.active] = question.model_copy(update={"selected": index})
        self.model = self.model.model_copy(update={"questions": questions, "editing_custom": False})
        for row in self._rows():
            row.set_selected(row.index == index)
        self.post_message(self.Moved(self))

    def _rows(self) -> list:
        """Every option row, of BOTH kinds — selection treats the composite
        custom row exactly like a plain one."""
        return [*self.query(QuestionOptionRow), *self.query(QuestionCustomRow)]

    def custom_field(self) -> PromptInput | None:
        """The custom row's text field, once mounted. `frame.show_questions`
        asks for it so a rebuild hands the keyboard back to the field the user
        was typing in rather than to the panel around it."""
        rows = list(self.query(QuestionCustomRow))
        return rows[0].field if rows else None

    def _move(self, delta: int) -> None:
        self._select(self.question.selected + delta)

    def _tab(self, delta: int) -> None:
        count = len(self.model.questions)
        self.model = self.model.model_copy(
            update={"active": (self.model.active + delta) % count, "editing_custom": False},
        )
        self.post_message(self.Moved(self))

    # ── the custom-answer editor ─────────────────────────────────────────────

    def _custom_index(self) -> int | None:
        for index, option in enumerate(self.question.options):
            if option.kind == "custom":
                return index
        return None

    def _set_custom(self, text: str) -> None:
        """Mirror the field's text into the widget's model and report it.

        The FIELD is the source of truth while it has the keyboard — it holds
        the cursor, the scroll and the undo stack — so this only keeps the
        model in step for the rebuilds that reseed a fresh field from it. The
        prefix is refreshed because in multi mode its tick is derived from
        whether the row holds text."""
        index = self._custom_index()
        if index is None or (self.question.options[index].text or "") == text:
            # A rebuilt field reloads its own text on mount, which `TextArea`
            # reports as a change like any other. Nothing moved, so nothing is
            # announced.
            return
        options = list(self.question.options)
        options[index] = options[index].model_copy(update={"text": text or None})
        questions = list(self.model.questions)
        questions[self.model.active] = self.question.model_copy(update={"options": options})
        self.model = self.model.model_copy(update={"questions": questions})
        for row in self.query(QuestionCustomRow):
            row.option = options[index]
            prefix = row.query_one(QuestionCustomPrefix)
            prefix.option = options[index]
            prefix.refresh()
        self.post_message(self.CustomChanged(self, text))

    def _field_has_keyboard(self) -> bool:
        field = self.custom_field()
        return field is not None and field.has_focus

    def _enter_custom(self, index: int) -> None:
        """Hand the keyboard to the text field. The caret is already on the
        row; from here every printable key, digit and space is literal text,
        which is what makes the editor an editor."""
        self.model = self.model.model_copy(update={"editing_custom": True})
        field = self.custom_field()
        if field is not None:
            field.focus()
            field.move_cursor(field.document.end)
        self.post_message(self.Moved(self))

    def _leave_custom(self) -> None:
        """Back to the option rows, TEXT INTACT.

        `escape` leaves the field; it no longer clears it. Deleting what you
        typed is the editor's job now that the row holds a real one, and an
        `escape` that wiped the line would be the one keystroke in the panel
        that destroys work."""
        self.model = self.model.model_copy(update={"editing_custom": False})
        self.focus()
        self.post_message(self.Moved(self))

    def on_text_area_changed(self, message: PromptInput.Changed) -> None:
        """The field's text, as it is typed, pasted or undone.

        STOPPED HERE, always: the app binds `on_text_area_changed` to the
        composer's `/` and `@` triggers, and a custom answer that opens with
        `/` must be text, not a command palette."""
        message.stop()
        if isinstance(message.text_area, PromptInput) and self._custom_index() is not None:
            self._set_custom(message.text_area.text)

    def on_prompt_input_submitted(self, message: PromptInput.Submitted) -> None:
        """`enter` in the field commits the custom answer.

        STOPPED HERE, always: the app's own handler would otherwise read it as
        a prompt to send to the model."""
        message.stop()
        index = self._custom_index()
        if index is None:
            return
        # An EMPTY custom field is not an answer: committing it leaves the
        # question exactly as it was.
        if message.value.strip():
            self.post_message(self.Answered(self, index))
        else:
            self._leave_custom()

    # ── keys ─────────────────────────────────────────────────────────────────

    async def on_key(self, event: events.Key) -> None:
        # `escape` is the way OUT of the field. `TextArea` binds neither it nor
        # `⇥`, so both arrive here while the editor has the keyboard — which is
        # what keeps question navigation working from inside the field.
        if event.key == "escape" and self._field_has_keyboard():
            self._leave_custom()
            event.stop()
            event.prevent_default()
            return

        question = self.question
        options = question.options
        key = event.key
        if key == "escape":
            # `esc` NEVER LEAVES A QUESTION UNANSWERED. Its one job is leaving
            # the text field, handled above while that field holds the
            # keyboard — so out here it is inert and CONSUMED. Letting it
            # through would reach the app's cancel binding and end the turn,
            # and the only ways out of a question set are the custom-answer row
            # and Chat about this.
            pass
        elif key == "up":
            self._move(-1)
        elif key == "down":
            self._move(1)
        elif key == "tab":
            self._tab(1)
        elif key == "shift+tab":
            self._tab(-1)
        elif key == "enter":
            # `enter` COMMITS. In single select that is the pick; in multi it is
            # the ticks as they stand — it never toggles the caret's row, or the
            # keystroke that finishes the question would also change the answer
            # it is finishing.
            self._activate(question.selected)
        elif key == "space":
            kind = options[question.selected].kind
            # Inert on the chat row: only `enter` (or the row's digit) may end
            # the whole set.
            if kind == "chat":
                return
            self._activate(question.selected, ticking=True)
        elif key.isdigit() and 1 <= int(key) <= len(options):
            self._select(int(key) - 1)
            self._activate(int(key) - 1, ticking=True)
        else:
            return
        event.stop()
        event.prevent_default()

    def _activate(self, index: int, *, ticking: bool = False) -> None:
        """`ticking` = the keystroke was a digit or `space`, which in MULTI mode
        only toggles a tick. In single select every one of the three keys picks
        and advances, which is why the flag changes nothing there."""
        option = self.question.options[index]
        if option.kind == "chat":
            self.post_message(self.ChatRequested(self))
            return
        if option.kind == "custom":
            self._enter_custom(index)
            return
        if self._multi and ticking:
            self.post_message(self.Toggled(self, index))
            return
        self.post_message(self.Answered(self, index))


class QuestionConfirmView(Vertical, can_focus=True):
    """The confirmation, in the same dock and the same border. Two rows: the
    summary (one line per question, in tab order) and one optional free-text
    field, which is what replaced the old per-question notes.

    READ-ONLY: no caret, no selectable rows, no `↑↓`, no way to change an
    answer from here. `esc` is how you go edit something, and reaching the
    confirmation is never a dead end — submission always takes two deliberate
    presses."""

    class Submitted(Message):
        def __init__(self, view: QuestionConfirmView, extra: str) -> None:
            super().__init__()
            self.view = view
            self.extra = extra

    class Cancelled(Message):
        """Back to the questions — CARRYING the optional field's current text,
        because `esc` there goes back with EVERYTHING intact, and what the user
        typed is part of everything."""

        def __init__(self, view: QuestionConfirmView, extra: str) -> None:
            super().__init__()
            self.view = view
            self.extra = extra

    def __init__(self, model: vm.QuestionSetState) -> None:
        super().__init__()
        self.model = model

    def compose(self) -> ComposeResult:
        yield ConfirmHead(self)
        yield Static(classes="inner-rule")
        with Vertical(classes="confirm-summary"):
            for question in self.model.questions:
                yield SpanLine(summarize_answer(question))
        yield Static(classes="inner-rule")
        with Horizontal(classes="confirm-extra"):
            yield Static(GLYPH_USER, markup=False, classes="composer-glyph")
            yield PromptInput(placeholder="anything to add? optional")

    def on_mount(self) -> None:
        field = self.query_one(PromptInput)
        if self.model.extra:
            field.load_text(self.model.extra)
        field.focus()

    def on_prompt_input_submitted(self, message: PromptInput.Submitted) -> None:
        message.stop()
        self.post_message(self.Submitted(self, message.value.strip()))

    async def on_key(self, event: events.Key) -> None:
        if event.key != "escape":
            return
        self.post_message(self.Cancelled(self, self.query_one(PromptInput).text.strip()))
        event.stop()
        event.prevent_default()


class ConfirmHead(Static):
    """`Ready to submit` … right-aligned `4 answered`."""

    def __init__(self, view: QuestionConfirmView) -> None:
        super().__init__(markup=False, classes="confirm-head")
        self.view = view

    def render(self) -> Text:
        tokens = resolve_tokens(self.view.app)
        text = Text("Ready to submit", style=Style(color=tokens.foreground), no_wrap=True)
        answered = sum(1 for question in self.view.model.questions if question.answered)
        right = Text(f"{answered} answered", style=Style(color=tokens.faint))
        pad = self.size.width - text.cell_len - right.cell_len
        if pad >= 1:
            text.append(" " * pad)
            text.append_text(right)
        return text


def summarize_answer(question: vm.Question, *, column: int = 15) -> str:
    """One confirmation row: `☑  tab        →  the answer`.

    A span string rather than a `Text`, so it goes through `SpanLine` like
    every other derived row — and so the same derivation feeds the collapsed
    transcript block once the set is submitted."""
    box = GLYPH_BOX_TICKED if question.answered else GLYPH_BOX
    picked: list[str] = []
    for index, option in enumerate(question.options):
        if option.kind == "chat":
            continue
        if option.kind == "custom":
            if option.text:
                picked.append(f"custom · {option.text}")
            continue
        if question.mode == "multi":
            if option.checked:
                picked.append(option.label)
        # `enumerate`, never `options.index(option)`: `QuestionOption` is a
        # Pydantic model with value equality, so a question the model authored
        # with two identical labels would resolve both rows to the first one.
        elif question.answer == index:
            picked.append(option.label)
    answer = ", ".join(picked) if picked else "chat about this"
    return f"{box}  {question.tab.ljust(column)}→  {answer}"


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
        # Scrolls: a model list runs to hundreds of rows and the dock is capped
        # at a share of the screen, so the list has to move under the cursor
        # rather than push the transcript and the composer off.
        with VerticalScroll(classes="overlay-options"):
            for index, row in enumerate(self.model.rows):
                yield self._row_widget(index, row)
        yield Static(classes="inner-rule")
        yield QueryLine(self)

    async def set_state(self, model: vm.OverlayState) -> None:
        """Swap in a fresh state (new rows after a query change)."""
        self.model = model
        options = self.query_one(".overlay-options", VerticalScroll)
        await options.remove_children()
        await options.mount(*(self._row_widget(index, row) for index, row in enumerate(self.model.rows)))
        options.scroll_home(animate=False)  # a new result set starts at the top
        self.query_one(QueryLine).refresh()

    def _move(self, delta: int) -> None:
        count = len(self.model.rows)
        if count == 0:
            return
        selected = (self.model.selected + delta) % count
        self.model = self.model.model_copy(update={"selected": selected})
        rows = list(self.query(SelectRow))
        for index, row in enumerate(rows):
            row.set_selected(index == selected)
        # Keep the caret on screen; without this the cursor walks off the
        # bottom of a long list and the highlight is invisible.
        if selected < len(rows):
            self.query_one(".overlay-options", VerticalScroll).scroll_to_widget(rows[selected], animate=False)

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
