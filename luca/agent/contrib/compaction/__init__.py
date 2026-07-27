"""A concrete compaction policy for the core `CompactionPolicy` contract.

`SummarizingCompactionPolicy` decides when to compact (the context gauge) and
produces the summary; the core runner owns the transition. `keep_turns` sets
how many recent exchanges survive verbatim.
"""

from .context import calculate_context_used, calculate_utilization_ratio, get_context_window_size
from .policy import DEFAULT_SUMMARY_PROMPT, SummarizingCompactionPolicy

__all__ = [
    "SummarizingCompactionPolicy",
    "DEFAULT_SUMMARY_PROMPT",
    "calculate_context_used",
    "get_context_window_size",
    "calculate_utilization_ratio",
]
