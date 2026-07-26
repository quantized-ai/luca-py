"""A concrete compaction policy for the core `CompactionPolicy` contract.

`SummarizingCompactionPolicy` decides when to compact (the context gauge) and
produces the summary; the core runner owns the transition. `keep_turns` sets
how many recent exchanges survive verbatim.
"""

from .context import context_used, context_window, utilization
from .policy import DEFAULT_SUMMARY_PROMPT, SummarizingCompactionPolicy

__all__ = [
    "SummarizingCompactionPolicy",
    "DEFAULT_SUMMARY_PROMPT",
    "context_used",
    "context_window",
    "utilization",
]
