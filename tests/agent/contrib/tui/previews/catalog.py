"""Storybook-style catalog for the production TUI components.

Run with:

    uv run textual run tests.agent.contrib.tui.previews.catalog

Select a component in the left column and press Enter. The catalog mounts the
real production widgets; the approval story opens its real modal screen.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import ContentSwitcher, Footer, Header, OptionList, Static

from luca.agent.contrib.tui.app import DEFAULT_THEME
from luca.agent.contrib.tui.config import load_luca_config, pick, resolve_config_path

from .approval_dialog import ApprovalDialogStory
from .context_gauge import ContextGaugeStory

STORIES = (
    ("Context gauge", "context-gauge-story"),
    ("Approval dialog", "approval-dialog-story"),
)


class TuiPreviewCatalog(App[None]):
    """One development app containing every manually inspectable TUI story."""

    TITLE = "luca · TUI component catalog"
    SUB_TITLE = "Context gauge"

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("t", "toggle_preview_theme", "Toggle theme", priority=True),
        Binding("r", "reopen_story", "Reopen story", priority=True),
        Binding("q", "quit", "Quit", priority=True),
    ]

    CSS = """
    #catalog-body {
        height: 1fr;
    }
    #catalog-navigation {
        width: 26;
        padding: 1;
        border-right: solid $panel-lighten-1;
        background: $panel;
    }
    #catalog-navigation > .navigation-title {
        margin-bottom: 1;
        text-style: bold;
    }
    #component-list {
        height: auto;
    }
    #story-switcher {
        width: 1fr;
        height: 1fr;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        config = load_luca_config(path=resolve_config_path())
        self._configured_theme = pick(None, config.theme.name, DEFAULT_THEME)
        self.theme = self._configured_theme
        self.sub_title = f"{STORIES[0][0]} · {self.theme}"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="catalog-body"):
            with Vertical(id="catalog-navigation"):
                yield Static("Components", classes="navigation-title")
                yield OptionList(*(label for label, _story_id in STORIES), id="component-list")
            with ContentSwitcher(initial=STORIES[0][1], id="story-switcher"):
                yield ContextGaugeStory(id=STORIES[0][1])
                yield ApprovalDialogStory(id=STORIES[1][1])
        yield Footer()

    def on_mount(self) -> None:
        options = self.query_one("#component-list", OptionList)
        options.highlighted = 0
        options.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        _label, story_id = STORIES[event.option_index]
        self.query_one("#story-switcher", ContentSwitcher).current = story_id
        self._refresh_sub_title()
        if story_id == "approval-dialog-story":
            self.call_after_refresh(self.query_one(ApprovalDialogStory).show_dialog)

    def action_reopen_story(self) -> None:
        switcher = self.query_one("#story-switcher", ContentSwitcher)
        if switcher.current == "approval-dialog-story":
            self.query_one(ApprovalDialogStory).show_dialog()

    def action_toggle_preview_theme(self) -> None:
        alternate = "textual-dark" if self._configured_theme == "atom-one-light" else "atom-one-light"
        self.theme = alternate if self.theme == self._configured_theme else self._configured_theme
        self._refresh_sub_title()

    def _refresh_sub_title(self) -> None:
        current = self.query_one("#story-switcher", ContentSwitcher).current
        label = next(label for label, story_id in STORIES if story_id == current)
        self.sub_title = f"{label} · {self.theme}"


app = TuiPreviewCatalog()


if __name__ == "__main__":
    app.run()
