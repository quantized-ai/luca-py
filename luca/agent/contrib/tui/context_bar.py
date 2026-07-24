"""A one-line context-utilization gauge for the TUI.

The rendering is a pure function so it can be tested without a running app; the
widget is a thin `Static` that recomputes from the session and the compactor.
"""

from __future__ import annotations

from textual.widgets import Static

from luca.agent.core.models import AgentSession

_WIDTH = 20


def _fmt(n: int) -> str:
    return f"{n / 1000:.0f}k" if n >= 1000 else str(n)


def render_context_bar(used: int, window: int, threshold: float = 0.8, width: int = _WIDTH) -> str:
    """`context ▐████████░░░░░░░░▌ 62%  124k/200k`, colored by fill vs threshold."""
    pct = 0.0 if window <= 0 else min(1.0, used / window)
    filled = round(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    color = "green" if pct < 0.6 else ("yellow" if pct < threshold else "red")
    return f"context [{color}]▐{bar}▌[/] {pct * 100:.0f}%  {_fmt(used)}/{_fmt(window)}"


class ContextBar(Static):
    DEFAULT_CSS = """
    ContextBar {
        height: 1;
        margin: 0 1;
        color: $text-muted;
    }
    """

    #: the last string rendered — handy for assertions
    text: str = ""

    def update_from(self, session: AgentSession, compactor) -> None:
        self.text = render_context_bar(
            compactor.context_used(session),
            compactor.context_window(session),
            compactor.threshold,
        )
        self.update(self.text)
