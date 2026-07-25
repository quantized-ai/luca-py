"""A concrete compaction policy for the core `CompactionPolicy` contract.

`SummarizingCompactionPolicy` decides when to compact (the context gauge) and
produces the summary; the core runner owns the transition. Pluggable via
`CompactionStrategy` (what to keep verbatim).
"""

from .context import context_used, context_window, utilization
from .policy import DEFAULT_SUMMARY_PROMPT, SummarizingCompactionPolicy
from .strategy import CompactionStrategy, RecentTurnsStrategy

__all__ = [
    "SummarizingCompactionPolicy",
    "CompactionStrategy",
    "RecentTurnsStrategy",
    "DEFAULT_SUMMARY_PROMPT",
    "context_used",
    "context_window",
    "utilization",
]
