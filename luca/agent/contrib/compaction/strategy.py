"""Compaction split strategies — the pluggable part of the feature.

A strategy decides ONE thing: which trailing nodes of a conversation to keep
verbatim. Everything before the first kept node is what the engine summarizes.
That single split is the whole difference between "summarize everything" and
"summarize everything except the last few turns", so it is the only hook.

Concrete base, no ABC — subclass and override `select_keep`, matching the
`PermissionPolicy` / `ContextManager` / `ConversationProjector` house style.
The base keeps nothing, which is the full-summary (Claude-Code `/compact`)
shape. Strategies return plain node ids; ids and clocks belong to the engine.
"""

from __future__ import annotations

from luca.agent.core.models import AgentSession, TurnStart, UserMessage


class CompactionStrategy:
    """Split policy base. Keeps nothing → the whole conversation is summarized."""

    def select_keep(self, session: AgentSession) -> list[str]:
        return []


class RecentTurnsStrategy(CompactionStrategy):
    """Keep the last `keep_turns` exchanges verbatim; summarize the rest.

    The cut lands on the `TurnStart` of the Nth-from-last turn, then extends
    back one node to include the `UserMessage` that prompted it — a luca turn
    bracket does not contain its own user message, and keeping the answer
    without the question reads oddly. This is a clean exchange boundary, never
    a mid-turn cut (which would strand a tool call from its result). With fewer
    than `keep_turns` turns there is nothing older to summarize, so it keeps
    everything (the engine reads that as "don't compact")."""

    def __init__(self, keep_turns: int = 2) -> None:
        if keep_turns < 1:
            raise ValueError("keep_turns must be >= 1")
        self.keep_turns = keep_turns

    def select_keep(self, session: AgentSession) -> list[str]:
        nodes = session.active_conversation.nodes
        entries = session.entries
        seen = 0
        for i in range(len(nodes) - 1, -1, -1):
            if isinstance(entries[nodes[i]], TurnStart):
                seen += 1
                if seen == self.keep_turns:
                    start = i
                    if start > 0 and isinstance(entries[nodes[start - 1]], UserMessage):
                        start -= 1
                    return list(nodes[start:])
        return list(nodes)  # fewer than keep_turns turns → keep everything
