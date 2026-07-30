"""A concrete `ContextManager` for the core contract, with compaction.

`SummarizingContextManager` keeps core's per-entry accounting and adds the
compaction pair: it decides when to compact (the context gauge) and produces the
summary; the core runner owns the transition. `keep_turns` sets how many recent
exchanges survive verbatim.
"""

from .manager import (
    DEFAULT_SUMMARY_PROMPT,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    SummarizingContextManager,
    calculate_context_used,
    calculate_utilization_ratio,
    get_context_window_size,
)

__all__ = [
    "SummarizingContextManager",
    "DEFAULT_SUMMARY_PROMPT",
    "DEFAULT_THRESHOLD",
    "DEFAULT_WINDOW",
    "calculate_context_used",
    "get_context_window_size",
    "calculate_utilization_ratio",
]
