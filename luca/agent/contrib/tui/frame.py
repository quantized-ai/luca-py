"""`LucaApp` — the global frame, with no agent attached.

Status bar, the single rule, the scrolling transcript, one dock slot
(composer / approval prompt / question set / overlay list), the hint legend.
Everything a
screen can show is applied from `state.ScreenState` view-models, which is
what the gallery and the snapshot tests boot against; `AgentApp` subclasses
this and feeds the same APIs from live agent events.

Below 60 columns the frame is replaced by a single centred line; between 60
and the 105-column design width the status bar drops segments and the
assistant measure shrinks (TCSS).
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Static

from . import state as vm
from .blocks import ListBlockView, TaskBlockView, build_block
from .chrome import Composer, HintLegend, StatusBar
from .format import HINTS, MIN_COLUMNS, question_hints_for
from .modals import CostScreen, McpScreen, SessionsScreen, SettingsScreen
from .shells import (
    ApprovalPromptView,
    LucaModalScreen,
    OverlayListView,
    QuestionConfirmView,
    QuestionSetView,
)
from .theme import LUCA_DARK

DEFAULT_THEME = LUCA_DARK.name


class LucaApp(App):
    TITLE = "luca"
    CSS_PATH = "app.tcss"

    # `priority` so the frame wins over the focused widget: the composer's
    # TextArea claims ctrl+c (copy) and ctrl+d (delete right) for itself, and
    # App binds ctrl+c to its own help-quit.
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+q", "quit", show=False, priority=True),
        Binding("ctrl+d", "quit", show=False, priority=True),
        Binding("ctrl+c", "clear_prompt", show=False, priority=True),
        Binding("pageup", "transcript_page(-1)", show=False, priority=True),
        Binding("pagedown", "transcript_page(1)", show=False, priority=True),
    ]

    def __init__(self, *, theme: str = DEFAULT_THEME) -> None:
        super().__init__()
        self.register_theme(LUCA_DARK)
        self.theme = theme

    def get_theme_variable_defaults(self) -> dict[str, str]:
        # The design tokens stay resolvable under any registered theme, so a
        # stock-theme user still gets a working (if off-palette) frame.
        return dict(LUCA_DARK.variables)

    # ── frame ─────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield StatusBar(id="status")
        yield Static(classes="rule")
        yield VerticalScroll(id="transcript", can_focus=False)
        # Mounted once and hidden, never mounted on demand: the panel updates
        # on every todo write, and `set_plan` has to stay callable from the
        # synchronous paths (`_set_busy`) that mounting would not allow.
        yield Vertical(ListBlockView(vm.ListBlock()), id="plan-dock")
        yield Vertical(id="dock")
        yield HintLegend(id="hints")
        yield Static("luca needs 60 columns", id="too-narrow", markup=False)

    def on_mount(self) -> None:
        self._check_width()

    def on_resize(self) -> None:
        self._check_width()

    def _check_width(self) -> None:
        self.screen.set_class(self.size.width < MIN_COLUMNS, "-too-narrow")

    # ── status / hints ────────────────────────────────────────────────────────

    def set_status(self, status: vm.StatusState) -> None:
        self.query_one("#status", StatusBar).set_state(status)

    def set_hints(self, hints: list[str]) -> None:
        self.query_one("#hints", HintLegend).set_hints(hints)

    # ── transcript ────────────────────────────────────────────────────────────

    @property
    def transcript(self) -> VerticalScroll:
        return self.query_one("#transcript", VerticalScroll)

    async def clear_transcript(self) -> None:
        await self.transcript.remove_children()

    async def mount_block(self, model: vm.Block, into: TaskBlockView | None = None) -> Widget:
        widget = build_block(model)
        target: Widget = into.body if into is not None else self.transcript
        await target.mount(widget)
        self.scroll_transcript_end()
        return widget

    def scroll_transcript_end(self) -> None:
        self.transcript.scroll_end(animate=False)

    def action_transcript_page(self, direction: int) -> None:
        if direction < 0:
            self.transcript.scroll_page_up(animate=False)
        else:
            self.transcript.scroll_page_down(animate=False)

    def dim_transcript(self, dimmed: bool) -> None:
        self.transcript.set_class(dimmed, "-dimmed")

    # ── the sticky plan panel ─────────────────────────────────────────────────

    @property
    def plan_view(self) -> ListBlockView:
        return self.query_one("#plan-dock", Vertical).query_one(ListBlockView)

    def set_plan(self, plan: vm.ListBlock | None) -> None:
        """Show, update or hide the sticky todo panel. Synchronous by design —
        it is called from the turn-boundary paths, not only from event
        handlers."""
        view = self.plan_view
        view.display = plan is not None
        if plan is not None:
            view.apply(plan)

    # ── the dock: composer / approval / overlay ───────────────────────────────

    @property
    def dock_slot(self) -> Vertical:
        return self.query_one("#dock", Vertical)

    def composer(self) -> Composer | None:
        try:
            return self.query_one(Composer)
        except Exception:
            return None

    def action_clear_prompt(self) -> None:
        """Empty the composer. A no-op while the dock holds an approval prompt
        or an overlay — there is no text to throw away."""
        composer = self.composer()
        if composer is not None:
            composer.input.clear()

    async def show_composer(
        self,
        state: vm.ComposerState | None = None,
        *,
        commands: list[str] | None = None,
        focus: bool = True,
    ) -> Composer:
        await self.dock_slot.remove_children()
        self.dim_transcript(False)
        composer = Composer(state, commands=commands or self._composer_commands())
        await self.dock_slot.mount(composer)
        if focus:
            composer.input.focus()
        return composer

    def _composer_commands(self) -> list[str]:
        """Slash names for inline completion; the live app overrides."""
        return []

    async def show_approval(self, state: vm.ApprovalState, *, focus: bool = True) -> ApprovalPromptView:
        await self.dock_slot.remove_children()
        self.dim_transcript(False)
        prompt = ApprovalPromptView(state)
        await self.dock_slot.mount(prompt)
        if focus:
            prompt.focus()
        return prompt

    async def show_questions(
        self,
        state: vm.QuestionSetState,
        *,
        focus: bool = True,
    ) -> QuestionSetView | QuestionConfirmView:
        """The fourth dock. `phase` is the ONLY thing that decides which of the
        two views renders, so they can never both be up."""
        await self.dock_slot.remove_children()
        self.dim_transcript(False)
        view: QuestionSetView | QuestionConfirmView
        confirming = state.phase == "confirming"
        view = QuestionConfirmView(state) if confirming else QuestionSetView(state)
        await self.dock_slot.mount(view)
        # The CONFIRMATION focuses its own optional field on mount — it is a
        # real `PromptInput` doing the composer's job, and it has the keyboard
        # by default. Focusing the view here would steal it straight back.
        #
        # The custom-answer row is the same story mid-set: while it is being
        # edited the keyboard belongs to ITS field, and this rebuild must hand
        # it back there rather than to the panel around it — otherwise entering
        # the field is the very thing that kicks you out of it.
        if focus and not confirming:
            field = view.custom_field() if state.editing_custom else None
            (field or view).focus()
        return view

    async def show_overlay(self, state: vm.OverlayState, *, focus: bool = True) -> OverlayListView:
        await self.dock_slot.remove_children()
        self.dim_transcript(True)
        overlay = OverlayListView(state)
        await self.dock_slot.mount(overlay)
        if focus:
            overlay.focus()
        return overlay

    # ── whole-screen application (fixtures, replay) ───────────────────────────

    async def apply_state(self, state: vm.ScreenState) -> None:
        """Render one declarative screen state: status, transcript, dock,
        hints, and the modal if the state carries one."""
        self.set_status(state.status)
        await self.clear_transcript()
        for block in state.transcript:
            await self.mount_block(block)
        self.set_plan(state.plan)
        dock = state.dock()
        if dock == "approval":
            await self.show_approval(state.approval)
        elif dock == "questions":
            await self.show_questions(state.questions)
        elif dock == "overlay":
            await self.show_overlay(state.overlay)
        else:
            await self.show_composer(state.composer or vm.ComposerState(), focus=state.modal is None)
        self.set_hints(state.hints or self._default_hints(state))
        if isinstance(self.screen, LucaModalScreen):
            self.pop_screen()
        if state.modal is not None:
            await self.push_screen(self._build_modal(state))

    def _default_hints(self, state: vm.ScreenState) -> list[str]:
        if state.modal is not None:
            if state.modal.sessions is not None:
                return HINTS["sessions"]
            if state.modal.settings is not None:
                return HINTS["settings"]
            if state.modal.mcp is not None:
                return HINTS["mcp"]
            return HINTS["cost"]
        dock = state.dock()
        if dock == "approval":
            count = len(state.approval.options)
            return ["↑↓ move", "enter confirm", f"1–{count} pick"]
        if dock == "questions":
            return question_hints_for(state.questions)
        if dock == "overlay":
            return HINTS[state.overlay.mode]
        return HINTS["idle"]

    def _build_modal(self, state: vm.ScreenState) -> LucaModalScreen:
        modal = state.modal
        hints = state.hints or self._default_hints(state)
        if modal.sessions is not None:
            return SessionsScreen(modal.sessions, state.status, hints)
        if modal.settings is not None:
            return SettingsScreen(modal.settings, state.status, hints)
        if modal.cost is not None:
            return CostScreen(modal.cost, state.status, hints)
        if modal.mcp is not None:
            return McpScreen(modal.mcp, state.status, hints)
        raise ValueError("modal state carries no screen")
