"""The compaction engine: measure context, summarize, and assemble a new session.

`Compactor` owns everything the strategies share — the context gauge, the LLM
summary call, and the atomic session-assembly transform. It creates a NEW
`AgentSession` seeded with a `CompactionEntry` and never mutates the source, so
compaction is all-or-nothing and the original stays on disk to reopen.

`build_compacted_session` is a pure function (ids and clocks are injectable for
determinism) that reconstructs exactly what the ledger's `append` would have
produced: the summary node, deep copies of any kept turns, a rebuilt
`tool_executions` index, and a fresh conversation id.
"""

from __future__ import annotations

from time import time_ns
from uuid import uuid4

from luca.client import acompletion, catalog
from luca.client.types import TextBlock
from luca.client.types import UserMessage as ClientUserMessage

from luca.agent.core.context_manager import ContextManager
from luca.agent.core.models import (
    AgentSession,
    CompactionEntry,
    Conversation,
    ConversationStatus,
    PrunedEntry,
    ToolExecution,
)
from luca.agent.core.projection import ConversationProjector

from .strategy import CompactionStrategy

DEFAULT_THRESHOLD = 0.8
DEFAULT_WINDOW = 200_000

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


def _new_id() -> str:
    return uuid4().hex[:8]


def _now_ms() -> int:
    return time_ns() // 1_000_000


def _text_of(message) -> str:
    return "".join(b.text for b in message.content if isinstance(b, TextBlock))


def context_used(session: AgentSession) -> int:
    """Sum of intrinsic `context_tokens` over the active conversation path."""
    entries = session.entries
    return sum(entries[node_id].context_tokens for node_id in session.active_conversation.nodes)


def build_compacted_session(
    session: AgentSession,
    summary: str,
    *,
    head_ids: list[str],
    keep_ids: list[str],
    strategy_name: str = "",
    context_manager: ContextManager | None = None,
    new_session_id: str | None = None,
    new_conversation_id: str | None = None,
    compaction_entry_id: str | None = None,
    now_ms: int | None = None,
) -> AgentSession:
    """A fresh session: a `CompactionEntry` summarizing `head_ids`, followed by
    verbatim deep copies of `keep_ids`. The source session is never touched."""
    cm = context_manager or ContextManager()
    now = _now_ms() if now_ms is None else now_ms
    session_id = new_session_id or _new_id()
    conversation_id = new_conversation_id or _new_id()
    entry_id = compaction_entry_id or _new_id()

    compaction = CompactionEntry(
        id=entry_id,
        created_at=now,
        summary=summary,
        summarized=list(head_ids),
        details={
            "source_session_id": session.id,
            "source_conversation_id": session.active_conversation.id,
            "created_at_ms": now,
            "tokens_before": context_used(session),
            "strategy": strategy_name,
        },
    )
    compaction.context_tokens = cm.calculate_context(compaction)

    entries: dict = {entry_id: compaction}

    def _copy(node_id: str) -> None:
        if node_id in entries:
            return
        original = session.entries[node_id]
        entries[node_id] = original.model_copy(deep=True)
        # A kept PrunedEntry projects through its referent, the one cross-slice
        # dependency, so the referent must travel too.
        if isinstance(original, PrunedEntry):
            _copy(original.pruned_entry_id)

    for node_id in keep_ids:
        _copy(node_id)

    tool_executions: dict = {}
    for node_id, entry in entries.items():
        if isinstance(entry, ToolExecution):
            tool_executions.setdefault(entry.tool_call_id, []).append(node_id)

    return AgentSession(
        id=session_id,
        entries=entries,
        tool_executions=tool_executions,
        usages={},
        active_conversation=Conversation(
            id=conversation_id,
            nodes=[entry_id, *keep_ids],
            created_at=now,
            updated_at=now,
            status=ConversationStatus.IDLE,
        ),
        conversation_history=[],
        session_config=session.session_config.model_copy(deep=True),
    )


class Compactor:
    """Context gauge plus the summarize-and-rebuild operation, parameterized by
    a `CompactionStrategy`. Swap the strategy to change what is kept verbatim."""

    def __init__(
        self,
        strategy: CompactionStrategy | None = None,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        default_window: int = DEFAULT_WINDOW,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        enabled: bool = True,
    ) -> None:
        self.strategy = strategy or CompactionStrategy()
        self.threshold = threshold
        self.default_window = default_window
        self.summary_prompt = summary_prompt
        self.enabled = enabled

    def context_used(self, session: AgentSession) -> int:
        return context_used(session)

    def context_window(self, session: AgentSession) -> int:
        cfg = session.session_config.llm_config
        info = catalog.get(cfg.provider, cfg.model)
        if info is not None and info.context_window:
            return info.context_window
        return self.default_window

    def utilization(self, session: AgentSession) -> float:
        window = self.context_window(session)
        if window <= 0:
            return 0.0
        return min(1.0, self.context_used(session) / window)

    def should_compact(self, session: AgentSession) -> bool:
        return self.enabled and self.utilization(session) >= self.threshold

    async def compact(self, session: AgentSession, *, provider=None) -> AgentSession:
        """Summarize the older span and return a new compacted session. Returns
        the session unchanged when there is nothing older than the kept tail."""
        nodes = session.active_conversation.nodes
        keep_ids = self.strategy.select_keep(session)
        cut = len(nodes) - len(keep_ids)
        if list(nodes[cut:]) != list(keep_ids):
            raise ValueError(
                "CompactionStrategy.select_keep must return a contiguous suffix "
                f"of the conversation nodes; got {keep_ids!r}",
            )
        head_ids = list(nodes[:cut])
        if not head_ids:
            return session
        summary = await self._summarize(session, head_ids, provider)
        return build_compacted_session(
            session, summary,
            head_ids=head_ids, keep_ids=list(keep_ids),
            strategy_name=type(self.strategy).__name__,
        )

    async def _summarize(self, session: AgentSession, head_ids: list[str], provider) -> str:
        cfg = session.session_config.llm_config
        head = Conversation(id="_compaction_head", nodes=head_ids, created_at=0, updated_at=0)
        messages = ConversationProjector().project(head, session.entries)
        messages = [*messages, ClientUserMessage(content=[TextBlock(text=_SUMMARY_REQUEST)])]
        response = await acompletion(
            model=f"{cfg.provider}:{cfg.model}",
            messages=messages,
            system_message=self.summary_prompt,
            provider=provider,
            reasoning=cfg.reasoning,
        )
        return _text_of(response.message)
