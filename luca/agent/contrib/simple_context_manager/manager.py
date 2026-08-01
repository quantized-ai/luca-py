"""A concrete `ContextManager` that also compacts: the context gauge plus LLM
summarization of the older span.

Inherits core's per-entry accounting (`calculate_context`, `prune_entry`,
`process_tool_output`) untouched and implements the compaction pair on top —
`should_compact` measures one conversation against the model's window,
and `compact` summarizes the older span with the session's own model and folds
it into a `CompactionEntry`, keeping a trailing tail.

Core owns the transition (archive the old conversation, swap in the new one);
this owns the decision — when to compact (the gauge) and what the new
conversation looks like (the summary plus the kept nodes). `keep_turns` is the
one knob: 0 folds everything into the summary, N keeps the last N exchanges
verbatim.

To extend it, subclass and override one of the public seams — `should_compact`
(when), `select_keep` (what survives verbatim), or `summarize` (how the summary
is produced). `compact` orchestrates the three and is rarely overridden.

The three gauge functions are module-level and depend on nothing else here: the
TUI context bar draws the same measurement `should_compact` decides on.
"""

from __future__ import annotations

from luca.agent.core.compaction import CompactionPlan, UsageCounters
from luca.agent.core.context_manager import ContextManager
from luca.agent.core.models import (
    AgentSession,
    CompactionEntry,
    TextContent,
    TurnStart,
    UserMessage,
)
from luca.agent.core.projection import ConversationProjector
from luca.client import acompletion, catalog
from luca.client.types import TextBlock, UserMessage as ClientUserMessage

DEFAULT_WINDOW = 200_000
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

DEFAULT_SUMMARY_REQUEST = "Summarize the conversation above per your instructions."


# ── the context gauge ─────────────────────────────────────────────────────────


def calculate_context_used(session: AgentSession, conversation_id: str) -> int:
    """Sum of intrinsic `context_tokens` over one conversation's path.

    Takes the conversation because a session holds several — the main agent's
    and one per subagent — and each has its own window to fill. A gauge that
    summed "the" conversation would report a number with no single basis."""
    entries = session.entries
    return sum(entries[node_id].context_tokens for node_id in session.conversations[conversation_id].nodes)


def get_context_window_size(session: AgentSession, default: int = DEFAULT_WINDOW) -> int:
    """The model's window from the client catalog, or `default` when the model
    (or the field) is missing. Conversation-independent: the window is a fact
    about the model, and every conversation in a session uses the same one."""
    cfg = session.session_config.llm_config
    info = catalog.get(cfg.provider, cfg.model)
    if info is not None and info.context_window:
        return info.context_window
    return default


def calculate_utilization_ratio(
    session: AgentSession,
    conversation_id: str,
    *,
    default_window: int = DEFAULT_WINDOW,
) -> float:
    """`used / window` for one conversation, clamped to `[0, 1]`."""
    window = get_context_window_size(session, default_window)
    if window <= 0:
        return 0.0
    return min(1.0, calculate_context_used(session, conversation_id) / window)


# ── the manager ───────────────────────────────────────────────────────────────


class SummarizingContextManager(ContextManager):
    """`should_compact` is the context gauge; `compact` summarizes the folded
    span and returns the new path. `keep_turns` sets how many recent exchanges
    survive verbatim (0 = summarize everything). Everything core's
    `ContextManager` does per entry is inherited unchanged.

    The public override seams are `should_compact`, `select_keep`, and
    `summarize`; the `text_of` / `usage_of` static helpers are available for a
    custom `summarize`."""

    def __init__(
        self,
        *,
        keep_turns: int = 0,
        threshold: float = DEFAULT_THRESHOLD,
        default_window: int = DEFAULT_WINDOW,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        enabled: bool = True,
        provider=None,
    ) -> None:
        self.keep_turns = keep_turns
        self.threshold = threshold
        self.default_window = default_window
        self.summary_prompt = summary_prompt
        self.enabled = enabled
        self.provider = provider

    def should_compact(self, session: AgentSession, conversation_id: str) -> bool:
        return self.enabled and (
            calculate_utilization_ratio(
                session,
                conversation_id,
                default_window=self.default_window,
            )
            >= self.threshold
        )

    def select_keep(self, candidates: list[str], session: AgentSession) -> list[str]:
        """Which trailing nodes survive verbatim; everything before is folded.

        `keep_turns <= 0` keeps nothing (full summary). Otherwise the last
        `keep_turns` exchanges: the cut lands on the Nth-from-last `TurnStart`
        and extends back to include the `UserMessage` that prompted it (a luca
        turn bracket does not contain its own user message), a clean exchange
        boundary that never strands a tool call from its result. Override this
        for a different split."""
        if self.keep_turns <= 0:
            return []
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

    async def summarize(self, session: AgentSession, folded: list[str]) -> tuple[str, UsageCounters]:
        """Produce the summary of the folded span: project those nodes, ask the
        session's own model, and return `(text, usage)`. Override to change the
        prompt, the model, or how the request is built."""
        cfg = session.session_config.llm_config
        messages = ConversationProjector().project(folded, session.entries)
        messages = [*messages, ClientUserMessage(content=[TextBlock(text=DEFAULT_SUMMARY_REQUEST)])]
        response = await acompletion(
            model=f"{cfg.provider}:{cfg.model}",
            messages=messages,
            system_message=self.summary_prompt,
            provider=self.provider,
            reasoning=cfg.reasoning,
        )
        return self.text_of(response.message), self.usage_of(response.message)

    async def compact(
        self,
        session: AgentSession,
        conversation_id: str,
        nodes: tuple[str, ...],
        entry: CompactionEntry,
    ) -> CompactionPlan | None:
        candidates = [node_id for node_id in nodes if node_id != entry.id]
        kept = self.select_keep(candidates, session)
        # Never fold a trailing unanswered user message: when compaction fires
        # automatically (a message posted, not yet answered), that node is the
        # pending question, and folding it into the summary would drop it
        # unanswered. Core leaves this to the manager on purpose.
        if candidates and candidates[-1] not in kept and isinstance(session.entries[candidates[-1]], UserMessage):
            kept = [candidates[-1]]
        folded = candidates[: len(candidates) - len(kept)]
        if not folded:
            return None  # nothing older than the kept tail — nothing to compact

        summary, usage = await self.summarize(session, folded)
        cfg = session.session_config.llm_config
        entry.parts = [TextContent(text=summary)]
        entry.compacted_nodes = folded
        entry.llm_config = cfg.model_copy(deep=True)
        entry.metadata = {"keep_turns": self.keep_turns, "kept": len(kept)}
        return CompactionPlan(entry=entry, nodes=[entry.id, *kept], usage=usage)

    @staticmethod
    def text_of(message) -> str:
        return "".join(block.text for block in message.content if isinstance(block, TextBlock))

    @staticmethod
    def usage_of(message) -> UsageCounters:
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
