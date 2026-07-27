"""The context gauge: how full the active conversation is against the model's
window. Used by the policy's `should_compact` and by the TUI context bar, so it
lives on its own with no dependency on either."""

from __future__ import annotations

from luca.agent.core.models import AgentSession
from luca.client import catalog

DEFAULT_WINDOW = 200_000


def calculate_context_used(session: AgentSession) -> int:
    """Sum of intrinsic `context_tokens` over the active conversation path."""
    entries = session.entries
    return sum(entries[node_id].context_tokens for node_id in session.active_conversation.nodes)


def get_context_window_size(session: AgentSession, default: int = DEFAULT_WINDOW) -> int:
    """The model's window from the client catalog, or `default` when the model
    (or the field) is missing."""
    cfg = session.session_config.llm_config
    info = catalog.get(cfg.provider, cfg.model)
    if info is not None and info.context_window:
        return info.context_window
    return default


def calculate_utilization_ratio(session: AgentSession, *, default_window: int = DEFAULT_WINDOW) -> float:
    """`used / window`, clamped to `[0, 1]`."""
    window = get_context_window_size(session, default_window)
    if window <= 0:
        return 0.0
    return min(1.0, calculate_context_used(session) / window)
