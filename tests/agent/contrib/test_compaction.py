"""Compaction engine: split strategies, the context gauge, and the atomic
new-session transform. No Textual, no network (the LLM call uses FauxProvider).
"""

import pytest

from luca.agent.contrib.compaction import (
    CompactionStrategy,
    Compactor,
    RecentTurnsStrategy,
    build_compacted_session,
    context_used,
)
from luca.agent.core.models import (
    AgentSession,
    AssistantMessage,
    CompactionEntry,
    Conversation,
    ConversationStatus,
    LLMConfig,
    PrunedEntry,
    SessionConfig,
    TextContent,
    TurnFinish,
    TurnStart,
    UserMessage,
)
from luca.agent.core.projection import ConversationProjector
from luca.agent.core.runner import AgentSessionRunner
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
                id="a1", parent_id="ts1", created_at=2, parts=[TextContent(text="A1")],
                llm_config=MODEL, stop_reason="stop", context_tokens=1,
            ),
            "tf1": TurnFinish(id="tf1", parent_id="a1", created_at=2),
            "u2": UserMessage(id="u2", parent_id="tf1", created_at=3, parts=[TextContent(text="Q2")], context_tokens=1),
            "ts2": TurnStart(id="ts2", parent_id="u2", created_at=4),
            "a2": AssistantMessage(
                id="a2", parent_id="ts2", created_at=4, parts=[TextContent(text="A2")],
                llm_config=MODEL, stop_reason="stop", context_tokens=1,
            ),
            "tf2": TurnFinish(id="tf2", parent_id="a2", created_at=4),
        },
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts1", "a1", "tf1", "u2", "ts2", "a2", "tf2"],
            created_at=1, updated_at=4, status=ConversationStatus.IDLE,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )


# ── strategies ─────────────────────────────────────────────────────────────


def test_the_base_strategy_keeps_nothing():
    assert CompactionStrategy().select_keep(two_turn_session()) == []


def test_recent_turns_keeps_the_last_exchange_including_its_user_message():
    assert RecentTurnsStrategy(keep_turns=1).select_keep(two_turn_session()) == [
        "u2", "ts2", "a2", "tf2",
    ]


def test_recent_turns_keeps_multiple_exchanges():
    assert RecentTurnsStrategy(keep_turns=2).select_keep(two_turn_session()) == [
        "u1", "ts1", "a1", "tf1", "u2", "ts2", "a2", "tf2",
    ]


def test_recent_turns_keeps_everything_when_fewer_turns_than_asked():
    assert RecentTurnsStrategy(keep_turns=5).select_keep(two_turn_session()) == [
        "u1", "ts1", "a1", "tf1", "u2", "ts2", "a2", "tf2",
    ]


def test_recent_turns_rejects_a_zero_keep():
    with pytest.raises(ValueError):
        RecentTurnsStrategy(keep_turns=0)


# ── the context gauge ──────────────────────────────────────────────────────


def test_context_used_sums_context_tokens_over_the_active_path():
    assert context_used(two_turn_session()) == 4  # four content entries at 1 each


def test_context_window_reads_the_catalog_when_present():
    session = two_turn_session()
    session.session_config.llm_config = LLMConfig(model="gpt-4o", provider="openai")
    assert Compactor(default_window=999).context_window(session) == 128_000


def test_context_window_falls_back_to_the_default_for_an_unknown_model():
    assert Compactor(default_window=50_000).context_window(two_turn_session()) == 50_000


def test_utilization_is_the_ratio_clamped_to_one():
    assert Compactor(default_window=8).utilization(two_turn_session()) == 0.5
    assert Compactor(default_window=2).utilization(two_turn_session()) == 1.0


def test_should_compact_gates_on_threshold_and_enabled():
    session = two_turn_session()
    assert Compactor(default_window=8, threshold=0.4).should_compact(session) is True
    assert Compactor(default_window=8, threshold=0.9).should_compact(session) is False
    assert Compactor(default_window=8, threshold=0.4, enabled=False).should_compact(session) is False


# ── build_compacted_session ────────────────────────────────────────────────


def test_full_summary_build_is_a_single_compaction_node():
    source = two_turn_session()
    all_nodes = list(source.active_conversation.nodes)

    new = build_compacted_session(
        source, "the summary",
        head_ids=all_nodes, keep_ids=[], strategy_name="CompactionStrategy",
        new_session_id="n1", new_conversation_id="nc1", compaction_entry_id="cmp", now_ms=99,
    )

    assert new == AgentSession(
        id="n1",
        entries={
            "cmp": CompactionEntry(
                id="cmp", created_at=99, summary="the summary",
                summarized=all_nodes,
                details={
                    "source_session_id": "src",
                    "source_conversation_id": "c1",
                    "created_at_ms": 99,
                    "tokens_before": 4,
                    "strategy": "CompactionStrategy",
                },
                context_tokens=2,  # len("the summary") // 4
            ),
        },
        tool_executions={},
        usages={},
        active_conversation=Conversation(
            id="nc1", nodes=["cmp"], created_at=99, updated_at=99,
            status=ConversationStatus.IDLE,
        ),
        conversation_history=[],
        session_config=SessionConfig(llm_config=MODEL),
    )


def test_recent_turns_build_keeps_the_tail_verbatim_and_leaves_the_source_intact():
    source = two_turn_session()
    before = source.model_copy(deep=True)
    keep = RecentTurnsStrategy(keep_turns=1).select_keep(source)
    head = source.active_conversation.nodes[: len(source.active_conversation.nodes) - len(keep)]

    new = build_compacted_session(
        source, "summary of the start",
        head_ids=head, keep_ids=keep, strategy_name="RecentTurnsStrategy",
        new_session_id="n1", new_conversation_id="nc1", compaction_entry_id="cmp", now_ms=99,
    )

    # the compaction node followed by deep copies of the kept exchange
    assert new.active_conversation.nodes == ["cmp", "u2", "ts2", "a2", "tf2"]
    assert new.entries["u2"] == source.entries["u2"]
    assert new.entries["a2"] == source.entries["a2"]
    assert new.entries["u2"] is not source.entries["u2"]  # a copy, not the original
    assert new.entries["cmp"].summarized == ["u1", "ts1", "a1", "tf1"]
    assert new.active_conversation.id == "nc1"
    assert new.usages == {}
    # the source is untouched
    assert source == before


def test_a_kept_pruned_entry_pulls_its_referent_into_the_new_session():
    from luca.agent.core.models import ToolCall, ToolExecution, ExecutionStatus, ExecutionResult

    tool_exec = ToolExecution(
        id="te1", created_at=2, tool_call_id="call-1",
        raw_tool_call=ToolCall(id="call-1", name="t", arguments={}),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="out")]),
        context_tokens=1,
    )
    pruned = PrunedEntry(
        id="pr1", created_at=2, pruned_entry_id="te1",
        pruned_entry_type="tool_execution",
        content=[TextContent(text="[pruned]")], context_tokens=1,
    )
    source = AgentSession(
        id="src",
        entries={
            "u1": UserMessage(id="u1", created_at=1, parts=[TextContent(text="Q")]),
            "ts1": TurnStart(id="ts1", parent_id="u1", created_at=2),
            "te1": tool_exec,
            "pr1": pruned,
            "tf1": TurnFinish(id="tf1", parent_id="pr1", created_at=2),
        },
        tool_executions={"call-1": ["te1"]},
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts1", "pr1", "tf1"], created_at=1, updated_at=2,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )

    new = build_compacted_session(
        source, "s", head_ids=["u1"], keep_ids=["ts1", "pr1", "tf1"],
        new_session_id="n1", new_conversation_id="nc1", compaction_entry_id="cmp", now_ms=9,
    )

    # the PrunedEntry's referent travels even though it is not on the path
    assert "te1" in new.entries
    assert new.entries["te1"] == tool_exec
    assert new.tool_executions == {"call-1": ["te1"]}


def test_a_kept_tool_execution_is_reindexed_in_the_new_session():
    from luca.agent.core.models import ExecutionResult, ExecutionStatus, ToolCall, ToolExecution

    tool_exec = ToolExecution(
        id="te1", parent_id="ts2", created_at=4, tool_call_id="call-9",
        raw_tool_call=ToolCall(id="call-9", name="mul", arguments={"a": 6}),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="42")]), context_tokens=1,
    )
    source = two_turn_session()
    source.entries["te1"] = tool_exec
    # splice the execution into the second turn's path
    source.active_conversation.nodes = [
        "u1", "ts1", "a1", "tf1", "u2", "ts2", "a2", "te1", "tf2",
    ]
    source.tool_executions = {"call-9": ["te1"]}

    keep = RecentTurnsStrategy(keep_turns=1).select_keep(source)
    head = source.active_conversation.nodes[: len(source.active_conversation.nodes) - len(keep)]
    new = build_compacted_session(source, "S", head_ids=head, keep_ids=keep)

    assert "te1" in new.active_conversation.nodes
    assert new.tool_executions == {"call-9": ["te1"]}
    assert new.entries["te1"] == tool_exec


def test_the_compacted_session_loads_idle_and_projects_cleanly():
    source = two_turn_session()
    keep = RecentTurnsStrategy(keep_turns=1).select_keep(source)
    head = source.active_conversation.nodes[: len(source.active_conversation.nodes) - len(keep)]
    new = build_compacted_session(source, "S", head_ids=head, keep_ids=keep)

    runner = AgentSessionRunner(new)
    assert runner.status is ConversationStatus.IDLE
    messages = ConversationProjector().project(new.active_conversation, new.entries)
    # [user(summary), user(Q2), assistant(A2)] — the kept exchange after the summary
    assert [type(m).__name__ for m in messages] == [
        "UserMessage", "UserMessage", "AssistantMessage",
    ]


# ── the full compact() operation ───────────────────────────────────────────


async def test_compact_summarizes_the_head_and_seeds_a_new_session():
    source = two_turn_session()
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("A DENSE SUMMARY")])])

    new = await Compactor(RecentTurnsStrategy(keep_turns=1)).compact(source, provider=faux)

    assert new.id != source.id
    first = new.entries[new.active_conversation.nodes[0]]
    assert isinstance(first, CompactionEntry)
    assert first.summary == "A DENSE SUMMARY"
    assert new.active_conversation.nodes[1:] == ["u2", "ts2", "a2", "tf2"]
    assert source.active_conversation.nodes == ["u1", "ts1", "a1", "tf1", "u2", "ts2", "a2", "tf2"]


async def test_compact_is_a_noop_when_there_is_nothing_older_than_the_tail():
    source = two_turn_session()
    # keep_turns covering every turn leaves no head to summarize
    new = await Compactor(RecentTurnsStrategy(keep_turns=5)).compact(source, provider=FauxProvider())
    assert new is source


async def test_compact_rejects_a_strategy_that_returns_a_non_suffix_keep():
    class BadStrategy(CompactionStrategy):
        def select_keep(self, session):
            return ["u1"]  # not a trailing slice of the path

    with pytest.raises(ValueError, match="contiguous suffix"):
        await Compactor(BadStrategy()).compact(two_turn_session(), provider=FauxProvider())
