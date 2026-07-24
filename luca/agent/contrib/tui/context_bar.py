"""The context-utilization gauge for the TUI.

A native Textual `ProgressBar` (smooth sub-cell fill, theme-aware) between a
`context` label and the token readout, colored green → yellow → red as it
approaches the compaction threshold. The stats math is a pure function so it
can be tested without a running app.
"""

from __future__ import annotations

from textual.containers import Horizontal
from textual.widgets import Label, ProgressBar

from luca.agent.core.models import AgentSession

_WARN_AT = 60  # percent — green below, yellow from here to the threshold


def _fmt(n: int) -> str:
    return f"{n / 1000:.0f}k" if n >= 1000 else str(n)


def context_stats(used: int, window: int, threshold: float) -> tuple[int, str, str]:
    """`(percent, "used/window", state)` where state is ok | warn | danger."""
    percent = 0 if window <= 0 else min(100, round(100 * used / window))
    threshold_percent = round(threshold * 100)
    if percent >= threshold_percent:
        state = "danger"
    elif percent >= _WARN_AT:
        state = "warn"
    else:
        state = "ok"
    return percent, f"{_fmt(used)}/{_fmt(window)}", state


class ContextBar(Horizontal):
    DEFAULT_CSS = """
    ContextBar {
        height: 1;
        margin: 0 1;
    }
    ContextBar > Label {
        color: $text-muted;
    }
    ContextBar > #ctx-tokens {
        margin: 0 0 0 1;
    }
    ContextBar > ProgressBar {
        width: 34;
    }
    ContextBar > ProgressBar Bar > .bar--bar {
        color: $success;
    }
    ContextBar.-warn > ProgressBar Bar > .bar--bar {
        color: $warning;
    }
    ContextBar.-danger > ProgressBar Bar > .bar--bar {
        color: $error;
    }
    """

    #: last "context N% used/window" summary — handy for assertions
    text: str = ""

    def compose(self):
        yield Label("context")
        yield ProgressBar(total=100, show_eta=False, id="ctx-progress")
        yield Label("", id="ctx-tokens")

    def update_from(self, session: AgentSession, compactor) -> None:
        percent, tokens, state = context_stats(
            compactor.context_used(session),
            compactor.context_window(session),
            compactor.threshold,
        )
        self.query_one("#ctx-progress", ProgressBar).update(total=100, progress=percent)
        self.query_one("#ctx-tokens", Label).update(tokens)
        self.set_class(state == "warn", "-warn")
        self.set_class(state == "danger", "-danger")
        self.text = f"context {percent}% {tokens}"
