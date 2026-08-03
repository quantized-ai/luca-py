"""The approval preview opens the real modal with deterministic state."""

from textual.containers import Container, VerticalScroll
from textual.widgets import Button

from luca.agent.contrib.tui.screens import ApprovalScreen

from .previews.approval_dialog import PREVIEW_PROMPT, ApprovalDialogPreview


async def test_approval_dialog_preview_opens_the_production_screen():
    app = ApprovalDialogPreview()

    async with app.run_test() as pilot:
        await pilot.pause()

        assert (
            type(app.screen),
            app.screen.prompt,
            [button.variant for button in app.screen.query(Button)],
        ) == (
            ApprovalScreen,
            PREVIEW_PROMPT,
            ["primary", "default", "error", "warning"],
        )


async def test_approval_dialog_preview_toggles_theme_without_closing_the_modal():
    app = ApprovalDialogPreview()

    async with app.run_test() as pilot:
        await pilot.press("t")

        assert (app.theme, app.sub_title, type(app.screen)) == (
            "atom-one-light",
            "atom-one-light",
            ApprovalScreen,
        )


async def test_long_approval_keeps_every_action_visible_at_normal_height():
    app = ApprovalDialogPreview()

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        dialog = app.screen.query_one("#approval-dialog", Container)
        details = app.screen.query_one("#approval-details", VerticalScroll)
        options = app.screen.query_one("#approval-options", VerticalScroll)
        buttons = list(app.screen.query(Button))

        assert (
            app.screen.focused,
            details.virtual_size.height > details.region.height,
            [options.region.contains_region(button.region) for button in buttons],
            buttons[-1].region.bottom == dialog.content_region.bottom,
        ) == (
            options,
            True,
            [True, True, True, True],
            True,
        )


async def test_long_approval_details_are_keyboard_scrollable():
    app = ApprovalDialogPreview()

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        details = app.screen.query_one("#approval-details", VerticalScroll)

        await pilot.press("shift+tab", "end")
        await pilot.pause()

        assert (
            app.screen.focused,
            details.scroll_y,
            details.max_scroll_y,
        ) == (
            details,
            97.0,
            97,
        )


async def test_long_approval_actions_scroll_in_a_short_terminal():
    app = ApprovalDialogPreview()

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        options = app.screen.query_one("#approval-options", VerticalScroll)
        buttons = list(app.screen.query(Button))

        await pilot.press("end")
        await pilot.pause()

        assert (
            options.scroll_y,
            options.region.contains_region(buttons[-1].region),
            [button.id for button in buttons],
        ) == (
            8.0,
            True,
            [
                "approval-option-0",
                "approval-option-1",
                "approval-option-2",
                "approval-option-3",
            ],
        )
