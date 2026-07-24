"""Conversation compaction: summarize a long session into a fresh one.

Pluggable via `CompactionStrategy` (what to keep verbatim); `Compactor` owns the
context gauge, the summary call, and the atomic new-session assembly. The source
session is never mutated.
"""

from .compactor import (
    DEFAULT_SUMMARY_PROMPT,
    Compactor,
    build_compacted_session,
    context_used,
)
from .strategy import CompactionStrategy, RecentTurnsStrategy

__all__ = [
    "CompactionStrategy",
    "RecentTurnsStrategy",
    "Compactor",
    "build_compacted_session",
    "context_used",
    "DEFAULT_SUMMARY_PROMPT",
]
