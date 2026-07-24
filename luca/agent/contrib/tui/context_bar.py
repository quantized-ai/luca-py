"""The context-utilization gauge for the TUI.

A one-line block bar (green → yellow → red as it nears the compaction
threshold) with the percentage and token readout, left-aligned to the input's
text. The rendering is a pure function so it can be tested without an app.
"""

from __future__ import annotations

from textual.widgets import Static

from luca.agent.core.models import AgentSession

_WIDTH = 20


def _fmt(n: int) -> str:
    return f"{n / 1000:.0f}k" if n >= 1000 else str(n)


def render_context_bar(used: int, window: int, threshold: float = 0.8, width: int = _WIDTH) -> str:
    """`▐████████░░░░░░░░▌ 62%  124k/200k`, colored by fill vs the threshold."""
    pct = 0.0 if window <= 0 else min(1.0, used / window)
    filled = round(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    color = "green" if pct < 0.6 else ("yellow" if pct < threshold else "red")
    return f"[{color}]▐{bar}▌[/] {pct * 100:.0f}%  {_fmt(used)}/{_fmt(window)}"


class ContextBar(Static):
    DEFAULT_CSS = """
    ContextBar {
        height: 1;
        /* align with the input's text: its margin (1) + border (1) + padding (2) */
        margin: 0 0 0 4;
        color: $text-muted;
    }
    """

    #: last rendered string — handy for assertions
    text: str = ""

    def update_from(self, session: AgentSession, compactor) -> None:
        self.text = render_context_bar(
            compactor.context_used(session),
            compactor.context_window(session),
            compactor.threshold,
        )
        self.update(self.text)
