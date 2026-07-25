"""A concrete `CompactionPolicy`: summarize the older span with the session's
own model and fold it into a `CompactionEntry`, keeping a strategy-chosen tail.

Core owns the transition (archive the old conversation, swap in the new one);
this owns the decision — when to compact (the context gauge) and what the new
conversation looks like (the summary plus the kept nodes).
"""

from __future__ import annotations

from luca.agent.core.compaction import CompactionPlan, CompactionPolicy, UsageCounters
from luca.agent.core.models import (
    AgentSession,
    CompactionEntry,
    Conversation,
    TextContent,
)
from luca.agent.core.projection import ConversationProjector
from luca.client import acompletion
from luca.client.types import TextBlock, UserMessage as ClientUserMessage

from .context import DEFAULT_WINDOW, utilization
from .strategy import CompactionStrategy

DEFAULT_THRESHOLD = 0.8

DEFAULT_SUMMARY_PROMPT = (
    "You are compacting a long agent conversation so it can continue in a fresh "
    "context window. Write a dense, faithful summary that a capable agent could "
    "read and pick up exactly where this left off. Cover, in order:\n"
    "1. The user's overall goal and any explicit constraints or preferences.\n"
    "2. What has been done so far, and the key decisions made and why.\n"
    "3. Files, commands, and resources touched, with their current state.\n"
    "4. The current state: what is working, what is broken, what was just tried.\n"
    "5. The immediate next steps and any open questions.\n"
    "Preserve concrete details (names, paths, ids, values) over generalities. Do "
    "not add commentary, greetings, or meta-notes about summarizing. Output only "
    "the summary."
)

_SUMMARY_REQUEST = "Summarize the conversation above per your instructions."


def _text_of(message) -> str:
    return "".join(block.text for block in message.content if isinstance(block, TextBlock))


class SummarizingCompactionPolicy(CompactionPolicy):
    """`should_compact` is the context gauge; `compact` summarizes the folded
    span and returns the new path. Swap the `CompactionStrategy` to change what
    is kept verbatim."""

    def __init__(
        self,
        strategy: CompactionStrategy | None = None,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        default_window: int = DEFAULT_WINDOW,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        enabled: bool = True,
        provider=None,
    ) -> None:
        self.strategy = strategy or CompactionStrategy()
        self.threshold = threshold
        self.default_window = default_window
        self.summary_prompt = summary_prompt
        self.enabled = enabled
        self.provider = provider

    def should_compact(self, session: AgentSession) -> bool:
        return self.enabled and (utilization(session, default_window=self.default_window) >= self.threshold)

    async def compact(
        self,
        session: AgentSession,
        nodes: tuple[str, ...],
        entry: CompactionEntry,
    ) -> CompactionPlan | None:
        candidates = [node_id for node_id in nodes if node_id != entry.id]
        kept = self.strategy.select_keep(candidates, session)
        folded = candidates[: len(candidates) - len(kept)]
        if not folded:
            return None  # nothing older than the kept tail — nothing to compact

        summary, usage = await self._summarize(session, folded)
        cfg = session.session_config.llm_config
        entry.parts = [TextContent(text=summary)]
        entry.compacted_nodes = folded
        entry.llm_config = cfg.model_copy(deep=True)
        entry.metadata = {"strategy": type(self.strategy).__name__, "kept": len(kept)}
        return CompactionPlan(entry=entry, nodes=[entry.id, *kept], usage=usage)

    async def _summarize(
        self,
        session: AgentSession,
        folded: list[str],
    ) -> tuple[str, UsageCounters]:
        cfg = session.session_config.llm_config
        head = Conversation(id="_compaction_head", nodes=list(folded), created_at=0, updated_at=0)
        messages = ConversationProjector().project(head, session.entries)
        messages = [*messages, ClientUserMessage(content=[TextBlock(text=_SUMMARY_REQUEST)])]
        response = await acompletion(
            model=f"{cfg.provider}:{cfg.model}",
            messages=messages,
            system_message=self.summary_prompt,
            provider=self.provider,
            reasoning=cfg.reasoning,
        )
        return _text_of(response.message), _usage_of(response.message)


def _usage_of(message) -> UsageCounters:
    usage = message.usage
    if usage is None:
        return UsageCounters()
    return UsageCounters(
        input=usage.input_tokens,
        output=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cache_read=usage.cached_input_tokens or 0,
        cache_write=usage.cache_write_tokens or 0,
    )
