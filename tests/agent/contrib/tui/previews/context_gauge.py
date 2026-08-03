"""Preview the production context gauge against theme-aware reference colors.

Run with:

    uv run textual run tests/agent/contrib/tui/previews/context_gauge.py

Press ``t`` to toggle between a dark and a light Textual theme. The "current"
rows mount the real :class:`ContextBar`; the comparison rows render the same
bar with Textual's semantic theme variables.
"""

from __future__ import annotations

from typing import ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Static

from luca.agent.contrib.tui.context_bar import ContextBar, render_context_bar


class SemanticContextBar(Horizontal):
    """The same gauge with its colored span driven by the active theme."""

    DEFAULT_CSS = """
    SemanticContextBar {
        height: 1;
        margin: 0 0 0 2;
    }
    SemanticContextBar > .gauge {
        width: auto;
    }
    SemanticContextBar > .readout {
        width: auto;
        color: $text-muted;
    }
    SemanticContextBar.-success > .gauge { color: $text-success; }
    SemanticContextBar.-warning > .gauge { color: $text-warning; }
    SemanticContextBar.-error > .gauge { color: $text-error; }
    """

    def __init__(self, rendered: str, tone: str) -> None:
        super().__init__(classes=f"-{tone}")
        plain = Text.from_markup(rendered).plain
        gauge, readout = plain.split("▌", 1)
        self._gauge = f"{gauge}▌"
        self._readout = readout

    def compose(self) -> ComposeResult:
        yield Static(self._gauge, classes="gauge")
        yield Static(self._readout, classes="readout")


class GaugeExample(Vertical):
    """One utilization level shown with the current and semantic colors."""

    DEFAULT_CSS = """
    GaugeExample {
        height: auto;
        margin: 0 1 1 1;
        padding: 1 1;
        border: round $panel-lighten-2;
    }
    GaugeExample > .example-title {
        text-style: bold;
        margin-bottom: 1;
    }
    GaugeExample > .row-label {
        color: $text-muted;
    }
    GaugeExample > ContextBar,
    GaugeExample > SemanticContextBar {
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        *,
        label: str,
        used: int,
        window: int = 200_000,
        threshold: float = 0.8,
    ) -> None:
        super().__init__()
        self._label = label
        self._rendered = render_context_bar(used, window, threshold)
        utilization = used / window
        self._tone = "success" if utilization < 0.6 else ("warning" if utilization < threshold else "error")

    def compose(self) -> ComposeResult:
        current = ContextBar(self._rendered)
        current.text = self._rendered
        yield Static(self._label, classes="example-title")
        yield Static("Current production ContextBar — literal Rich color", classes="row-label")
        yield current
        yield Static(f"Theme-aware reference — $text-{self._tone}", classes="row-label")
        yield SemanticContextBar(self._rendered, self._tone)


class ContextGaugeStory(VerticalScroll):
    """The reusable catalog story for the context gauge states."""

    DEFAULT_CSS = """
    ContextGaugeStory {
        padding: 1 1;
    }
    ContextGaugeStory > .instructions {
        margin: 0 1 1 1;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "The production ContextBar is paired with a semantic-color reference. Switch themes to compare them.",
            classes="instructions",
        )
        yield GaugeExample(label="Low · 25%", used=50_000)
        yield GaugeExample(label="Warning · 62%", used=124_000)
        yield GaugeExample(label="Critical · 90%", used=180_000)


class ContextGaugePreview(App[None]):
    """A manually runnable visual scenario for the context gauge."""

    TITLE = "luca · context gauge preview"
    SUB_TITLE = "textual-dark"

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("t", "toggle_preview_theme", "Toggle theme"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield ContextGaugeStory(id="context-gauge-story")
        yield Footer()

    def action_toggle_preview_theme(self) -> None:
        self.theme = "atom-one-light" if self.theme == "textual-dark" else "textual-dark"
        self.sub_title = self.theme


app = ContextGaugePreview()


if __name__ == "__main__":
    app.run()
