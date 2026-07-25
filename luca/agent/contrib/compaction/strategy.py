"""Compaction split strategies — which trailing nodes survive verbatim.

A strategy decides one thing: given the foldable path (the nodes a compaction
may replace), which trailing slice to keep. Everything before it is folded into
the summary. That single split is the whole difference between "summarize
everything" and "summarize everything except the last few turns".

Concrete base, no ABC — subclass and override `select_keep`, the same house
style as `PermissionPolicy` / `ContextManager`. The base keeps nothing (full
summary). Strategies return plain node ids; the policy assembles the plan.
"""

from __future__ import annotations

from luca.agent.core.models import AgentSession, TurnStart, UserMessage


class CompactionStrategy:
    """Split policy base. Keeps nothing → the whole span is summarized."""

    def select_keep(self, candidates: list[str], session: AgentSession) -> list[str]:
        return []


class RecentTurnsStrategy(CompactionStrategy):
    """Keep the last `keep_turns` exchanges verbatim; fold the rest.

    The cut lands on the `TurnStart` of the Nth-from-last turn, then extends
    back one node to include the `UserMessage` that prompted it — a luca turn
    bracket does not contain its own user message, and keeping the answer
    without the question reads oddly. It is a clean exchange boundary, never a
    mid-turn cut (which would strand a tool call from its result). With fewer
    than `keep_turns` turns there is nothing older to fold, so it keeps
    everything and the policy treats that as "nothing to compact"."""

    def __init__(self, keep_turns: int = 2) -> None:
        if keep_turns < 1:
            raise ValueError("keep_turns must be >= 1")
        self.keep_turns = keep_turns

    def select_keep(self, candidates: list[str], session: AgentSession) -> list[str]:
        entries = session.entries
        seen = 0
        for i in range(len(candidates) - 1, -1, -1):
            if isinstance(entries[candidates[i]], TurnStart):
                seen += 1
                if seen == self.keep_turns:
                    start = i
                    if start > 0 and isinstance(entries[candidates[start - 1]], UserMessage):
                        start -= 1
                    return list(candidates[start:])
        return list(candidates)
