"""SummarizingCompactionPolicy: the context gauge, the split strategies, and
the plan it hands the core transition. No TUI; the summary call uses a
FauxProvider. Core owns archiving/swapping and is tested in tests/agent."""

import pytest

from luca.agent.contrib.compaction import (
    CompactionStrategy,
    RecentTurnsStrategy,
    SummarizingCompactionPolicy,
    context_used,
    context_window,
    utilization,
)
from luca.agent.core.compaction import CompactionPlan, UsageCounters
from luca.agent.core.models import (
    AgentSession,
    AssistantMessage,
    CompactionEntry,
    CompactionSource,
    Conversation,
    ConversationStatus,
    LLMConfig,
    SessionConfig,
    TextContent,
    TurnFinish,
    TurnStart,
    UserMessage,
)
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text

MODEL = LLMConfig(model="fake-model", provider="faux")


def two_turn_session() -> AgentSession:
    """Two complete exchanges: (u1 → a1) then (u2 → a2). Markers count 0."""
    return AgentSession(
        id="src",
        entries={
            "u1": UserMessage(id="u1", created_at=1, parts=[TextContent(text="Q1")], context_tokens=1),
            "ts1": TurnStart(id="ts1", parent_id="u1", created_at=2),
            "a1": AssistantMessage(
                id="a1",
                parent_id="ts1",
                created_at=2,
                parts=[TextContent(text="A1")],
                llm_config=MODEL,
                stop_reason="stop",
                context_tokens=1,
            ),
            "tf1": TurnFinish(id="tf1", parent_id="a1", created_at=2),
            "u2": UserMessage(id="u2", parent_id="tf1", created_at=3, parts=[TextContent(text="Q2")], context_tokens=1),
            "ts2": TurnStart(id="ts2", parent_id="u2", created_at=4),
            "a2": AssistantMessage(
                id="a2",
                parent_id="ts2",
                created_at=4,
                parts=[TextContent(text="A2")],
                llm_config=MODEL,
                stop_reason="stop",
                context_tokens=1,
            ),
            "tf2": TurnFinish(id="tf2", parent_id="a2", created_at=4),
        },
        active_conversation=Conversation(
            id="c1",
            nodes=["u1", "ts1", "a1", "tf1", "u2", "ts2", "a2", "tf2"],
            created_at=1,
            updated_at=4,
            status=ConversationStatus.IDLE,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )


def _entry() -> CompactionEntry:
    return CompactionEntry(id="cmp", created_at=5, source=CompactionSource.USER)


def _offered(session: AgentSession) -> tuple[str, ...]:
    # the path the runner offers: the active nodes, ending with the compaction entry
    return (*session.active_conversation.nodes, "cmp")


def _faux(summary: str) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses([faux_assistant_message([faux_text(summary)], finish_reason="stop")])
    return provider


# ── the gauge ──────────────────────────────────────────────────────────────


def test_context_used_sums_context_tokens_over_the_active_path():
    assert context_used(two_turn_session()) == 4


def test_context_window_falls_back_to_default_for_an_unknown_model():
    assert context_window(two_turn_session(), default=50_000) == 50_000


def test_utilization_is_the_ratio():
    assert utilization(two_turn_session(), default_window=8) == 0.5


def test_should_compact_gates_on_threshold_and_enabled():
    session = two_turn_session()
    assert SummarizingCompactionPolicy(default_window=8, threshold=0.4).should_compact(session) is True
    assert SummarizingCompactionPolicy(default_window=8, threshold=0.9).should_compact(session) is False
    assert SummarizingCompactionPolicy(default_window=8, threshold=0.4, enabled=False).should_compact(session) is False


# ── strategies ───────────────────────────────────────────────────────────────


def test_the_base_strategy_keeps_nothing():
    session = two_turn_session()
    assert CompactionStrategy().select_keep(list(session.active_conversation.nodes), session) == []


def test_recent_turns_keeps_the_last_exchange_including_its_user_message():
    session = two_turn_session()
    assert RecentTurnsStrategy(keep_turns=1).select_keep(
        list(session.active_conversation.nodes),
        session,
    ) == ["u2", "ts2", "a2", "tf2"]


def test_recent_turns_rejects_a_zero_keep():
    with pytest.raises(ValueError, match="keep_turns"):
        RecentTurnsStrategy(keep_turns=0)


# ── compact() → the plan ─────────────────────────────────────────────────────


async def test_full_summary_folds_the_whole_span_into_one_node():
    session = two_turn_session()
    policy = SummarizingCompactionPolicy(provider=_faux("THE SUMMARY"))

    plan = await policy.compact(session, _offered(session), _entry())

    assert plan == CompactionPlan(
        entry=_entry().model_copy(
            update={
                "parts": [TextContent(text="THE SUMMARY")],
                "compacted_nodes": ["u1", "ts1", "a1", "tf1", "u2", "ts2", "a2", "tf2"],
                "llm_config": MODEL,
                "metadata": {"strategy": "CompactionStrategy", "kept": 0},
            }
        ),
        nodes=["cmp"],
        usage=UsageCounters(),
    )


async def test_recent_turns_keeps_the_tail_and_folds_the_head():
    session = two_turn_session()
    policy = SummarizingCompactionPolicy(RecentTurnsStrategy(keep_turns=1), provider=_faux("HEAD SUMMARY"))

    plan = await policy.compact(session, _offered(session), _entry())

    assert plan == CompactionPlan(
        entry=_entry().model_copy(
            update={
                "parts": [TextContent(text="HEAD SUMMARY")],
                "compacted_nodes": ["u1", "ts1", "a1", "tf1"],
                "llm_config": MODEL,
                "metadata": {"strategy": "RecentTurnsStrategy", "kept": 4},
            }
        ),
        nodes=["cmp", "u2", "ts2", "a2", "tf2"],
        usage=UsageCounters(),
    )


async def test_a_trailing_unanswered_user_message_is_never_folded():
    # auto-compaction fires with a just-posted, unanswered question on the path;
    # folding it would drop the question. Full summary must still keep it.
    session = two_turn_session()
    session.entries["u3"] = UserMessage(
        id="u3",
        parent_id="tf2",
        created_at=5,
        parts=[TextContent(text="Q3")],
        context_tokens=1,
    )
    session.active_conversation.nodes.append("u3")
    offered = (*session.active_conversation.nodes, "cmp")
    policy = SummarizingCompactionPolicy(provider=_faux("SUMMARY"))  # full summary

    plan = await policy.compact(session, offered, _entry())

    assert plan.nodes == ["cmp", "u3"]  # the pending question survives
    assert "u3" not in plan.entry.compacted_nodes


async def test_compact_returns_none_when_nothing_is_older_than_the_kept_tail():
    session = two_turn_session()
    policy = SummarizingCompactionPolicy(RecentTurnsStrategy(keep_turns=5), provider=_faux("x"))

    assert await policy.compact(session, _offered(session), _entry()) is None
