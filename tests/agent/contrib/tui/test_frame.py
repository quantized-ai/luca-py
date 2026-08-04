"""`LucaApp.apply_state` renders any hand-built `ScreenState`: the status
bar, the transcript blocks in order, exactly one dock (composer / approval /
overlay), the hint legend, and the three modal screens."""

from textual.widgets import Static

from luca.agent.contrib.tui import state as vm
from luca.agent.contrib.tui.blocks import (
    AssistantText,
    DiffBlockView,
    ListBlockView,
    NoticeLine,
    TaskBlockView,
    ThinkingLine,
    ToolBlockView,
    UserTurn,
)
from luca.agent.contrib.tui.chrome import Composer, HintLegend, StatusBar
from luca.agent.contrib.tui.format import HINTS
from luca.agent.contrib.tui.frame import LucaApp
from luca.agent.contrib.tui.modals import CostScreen, SessionsScreen, SettingsScreen
from luca.agent.contrib.tui.shells import ApprovalPromptView, LucaModalScreen, OverlayListView

FULL_STATE = vm.ScreenState(
    status=vm.StatusState(
        cwd="~/quantized/luca",
        model="sonnet-4.5",
        branch="main",
        dirty=True,
        tokens="12.4k",
        cost="$0.08",
    ),
    transcript=[
        vm.UserBlock(text="add a --json flag"),
        vm.ThinkingBlock(duration=3),
        vm.TextBlock(text="Adding the flag now."),
        vm.ToolBlock(tool="read", arg="luca/cli.py", result=vm.ToolResult(summary="84 lines")),
        vm.ListBlock(label="plan · 1 of 2", rows=[vm.ListRow(glyph="done", text="read the CLI")]),
        vm.DiffBlock(lines=[vm.DiffLine(num=8, sign="+", text="    json_flag = True")]),
        vm.NoticeBlock(text="turn cancelled"),
        vm.TaskBlock(description="explore the repo", status="running", blocks=[vm.TextBlock(text="Looking around.")]),
    ],
)

APPROVAL = vm.ApprovalState(
    question="Run `pytest -q`?",
    options=[
        vm.ApprovalOption(label="Approve once"),
        vm.ApprovalOption(label="Approve always — this command"),
        vm.ApprovalOption(label="Deny — tell Luca what to do instead"),
        vm.ApprovalOption(label="Cancel turn", key_hint="esc"),
    ],
)

OVERLAY = vm.OverlayState(
    mode="palette",
    rows=[vm.OverlayRow(primary="/clear", secondary="start a fresh session")],
    counter="1 of 14",
)

SESSIONS = vm.SessionsState(
    count_line="2 sessions in this repo · .luca/sessions",
    rows=[
        vm.SessionRow(when="18m ago", first_message="migrate the store", turns="14", tokens="31.7k", cost="$0.21"),
        vm.SessionRow(when="2h ago", first_message="add a --json flag", turns="6", tokens="12.4k", cost="$0.08"),
    ],
    preview=["◉ port the reader"],
)

SETTINGS = vm.SettingsState(
    groups=[vm.SettingsGroup(label="appearance", rows=[vm.SettingRow(name="theme", value="luca-dark")])],
    swatch_label="luca-dark",
)

COST = vm.CostState(
    headline="$0.21",
    subline="14 turns · 22m · sonnet-4.5",
    items=[vm.CostItem(label="input", tokens="26.3k", cost="$0.079", fraction=1.0)],
)


async def test_apply_state_renders_status_transcript_composer_and_default_hints():
    app = LucaApp()
    async with app.run_test(size=(105, 35)) as pilot:
        await app.apply_state(FULL_STATE)
        await pilot.pause()

        assert app.query_one("#status", StatusBar).model == FULL_STATE.status
        assert app.query_one("#status", StatusBar).render().plain.split() == [
            "◧",
            "luca",
            "~/quantized/luca",
            "sonnet-4.5",
            "main*",
            "12.4k",
            "·",
            "$0.08",
        ]
        assert [type(block) for block in app.transcript.children] == [
            UserTurn,
            ThinkingLine,
            AssistantText,
            ToolBlockView,
            ListBlockView,
            DiffBlockView,
            NoticeLine,
            TaskBlockView,
        ]
        assert [type(widget) for widget in app.dock_slot.children] == [Composer]
        assert app.query_one("#hints", HintLegend).hints == HINTS["idle"]


async def test_explicit_hints_override_the_defaults():
    app = LucaApp()
    async with app.run_test(size=(105, 35)) as pilot:
        await app.apply_state(vm.ScreenState(hints=["enter send", "esc interrupt"]))
        await pilot.pause()

        assert app.query_one("#hints", HintLegend).hints == ["enter send", "esc interrupt"]


async def test_an_approval_state_swaps_the_composer_for_the_prompt():
    app = LucaApp()
    async with app.run_test(size=(105, 35)) as pilot:
        await app.apply_state(vm.ScreenState())

        await app.apply_state(vm.ScreenState(approval=APPROVAL))
        await pilot.pause()

        assert [type(widget) for widget in app.dock_slot.children] == [ApprovalPromptView]
        assert app.query_one(ApprovalPromptView).model == APPROVAL
        assert app.transcript.has_class("-dimmed") is False
        assert app.query_one("#hints", HintLegend).hints == ["↑↓ move", "enter confirm", "1–4 pick"]


async def test_an_overlay_state_dims_the_transcript_behind_it():
    app = LucaApp()
    async with app.run_test(size=(105, 35)) as pilot:
        await app.apply_state(vm.ScreenState())

        await app.apply_state(vm.ScreenState(overlay=OVERLAY))
        await pilot.pause()

        assert [type(widget) for widget in app.dock_slot.children] == [OverlayListView]
        assert app.query_one(OverlayListView).model == OVERLAY
        assert app.transcript.has_class("-dimmed") is True
        assert app.query_one("#hints", HintLegend).hints == HINTS["palette"]


async def test_a_composer_state_restores_the_dock_and_undims():
    app = LucaApp()
    async with app.run_test(size=(105, 35)) as pilot:
        await app.apply_state(vm.ScreenState(overlay=OVERLAY))

        await app.apply_state(vm.ScreenState(composer=vm.ComposerState(placeholder="working…", text="queued")))
        await pilot.pause()

        assert [type(widget) for widget in app.dock_slot.children] == [Composer]
        assert app.query_one(Composer).input.text == "queued"
        assert app.query_one(Composer).input.placeholder == "working…"
        assert app.transcript.has_class("-dimmed") is False
        assert app.query_one("#hints", HintLegend).hints == HINTS["idle"]


async def test_a_sessions_modal_state_pushes_the_sessions_screen():
    state = vm.ScreenState(
        status=vm.StatusState(cwd="~/quantized/luca", label="sessions"),
        modal=vm.ModalState(sessions=SESSIONS),
    )
    app = LucaApp()
    async with app.run_test(size=(105, 35)) as pilot:
        await app.apply_state(state)
        await pilot.pause()

        assert type(app.screen) is SessionsScreen
        assert app.screen.state == SESSIONS
        assert app.screen.query_one(HintLegend).hints == HINTS["sessions"]


async def test_a_settings_modal_state_pushes_the_settings_screen():
    state = vm.ScreenState(
        status=vm.StatusState(cwd="~/quantized/luca", label="settings"),
        modal=vm.ModalState(settings=SETTINGS),
    )
    app = LucaApp()
    async with app.run_test(size=(105, 35)) as pilot:
        await app.apply_state(state)
        await pilot.pause()

        assert type(app.screen) is SettingsScreen
        assert app.screen.state == SETTINGS
        assert app.screen.query_one(HintLegend).hints == HINTS["settings"]


async def test_a_cost_modal_state_pushes_the_cost_screen():
    state = vm.ScreenState(
        status=vm.StatusState(cwd="~/quantized/luca", label="cost"),
        modal=vm.ModalState(cost=COST),
    )
    app = LucaApp()
    async with app.run_test(size=(105, 35)) as pilot:
        await app.apply_state(state)
        await pilot.pause()

        assert type(app.screen) is CostScreen
        assert app.screen.state == COST
        assert app.screen.query_one(HintLegend).hints == HINTS["cost"]


async def test_applying_a_plain_state_pops_the_modal():
    app = LucaApp()
    async with app.run_test(size=(105, 35)) as pilot:
        await app.apply_state(vm.ScreenState(modal=vm.ModalState(cost=COST)))
        await pilot.pause()

        await app.apply_state(vm.ScreenState())
        await pilot.pause()

        assert not isinstance(app.screen, LucaModalScreen)


async def test_below_60_columns_the_guard_replaces_the_frame():
    app = LucaApp()
    async with app.run_test(size=(59, 20)) as pilot:
        await pilot.pause()

        assert app.screen.has_class("-too-narrow") is True
        assert app.query_one("#too-narrow", Static).content == "luca needs 60 columns"


async def test_at_the_design_width_the_guard_stays_hidden():
    app = LucaApp()
    async with app.run_test(size=(105, 35)) as pilot:
        await pilot.pause()

        assert app.screen.has_class("-too-narrow") is False
