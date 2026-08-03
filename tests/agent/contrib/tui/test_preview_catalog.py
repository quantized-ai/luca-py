"""The TUI preview catalog hosts and navigates the real component stories."""

from pathlib import Path

from textual.widgets import ContentSwitcher, OptionList

from luca.agent.contrib.tui.context_bar import ContextBar
from luca.agent.contrib.tui.screens import ApprovalScreen

from .previews.approval_dialog import PREVIEW_PROMPT
from .previews.catalog import STORIES, TuiPreviewCatalog


async def test_catalog_opens_on_the_context_gauge_story():
    app = TuiPreviewCatalog()

    async with app.run_test() as pilot:
        await pilot.pause()

        assert (
            app.query_one("#story-switcher", ContentSwitcher).current,
            [option.prompt for option in app.query_one(OptionList).options],
            [bar.text for bar in app.query(ContextBar)],
        ) == (
            "context-gauge-story",
            [label for label, _story_id in STORIES],
            [
                "[green]▐█████░░░░░░░░░░░░░░░▌[/] 25%  50k/200k",
                "[yellow]▐████████████░░░░░░░░▌[/] 62%  124k/200k",
                "[red]▐██████████████████░░▌[/] 90%  180k/200k",
            ],
        )


async def test_catalog_selects_the_approval_story_and_opens_its_modal():
    app = TuiPreviewCatalog()

    async with app.run_test() as pilot:
        await pilot.press("down", "enter")
        await pilot.pause()

        assert (
            app.query_one("#story-switcher", ContentSwitcher).current,
            type(app.screen),
            app.screen.prompt,
        ) == (
            "approval-dialog-story",
            ApprovalScreen,
            PREVIEW_PROMPT,
        )


async def test_catalog_reads_the_theme_from_luca_json(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "luca.json").write_text('{"theme": {"name": "textual-light"}}')
    app = TuiPreviewCatalog()

    async with app.run_test() as pilot:
        await pilot.pause()

        assert (app.theme, app.sub_title) == (
            "textual-light",
            "Context gauge · textual-light",
        )


async def test_catalog_theme_toggle_returns_to_the_configured_theme(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "luca.json").write_text('{"theme": {"name": "nord"}}')
    app = TuiPreviewCatalog()

    async with app.run_test() as pilot:
        await pilot.press("t", "t")

        assert (app.theme, app.sub_title) == ("nord", "Context gauge · nord")
