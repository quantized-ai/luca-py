"""SummarizingCompactionPolicy: the context gauge, the keep_turns split, and
the plan it hands the core transition. No TUI; the summary call uses a
FauxProvider. Core owns archiving/swapping and is tested in tests/agent.

Two preconditions: a hand-written two-turn session for the plain cases, and
`tests/agent/scenarios.py`'s `RICH_IDLE_SESSION` for the realistic one — a
path carrying tool executions (whose specs are normalized into
`session.tool_specs`), a pruned entry, a prior summary and cancel markers.
"""

from luca.agent.contrib.compaction import (
    SummarizingCompactionPolicy,
    calculate_context_used,
    calculate_utilization_ratio,
    get_context_window_size,
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
from tests.agent.scenarios import MODEL as RICH_MODEL, RICH_IDLE_SESSION

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


def pending_question_session() -> AgentSession:
    """The same two exchanges plus a just-posted, unanswered `u3`."""
    session = two_turn_session()
    session.entries["u3"] = UserMessage(
        id="u3",
        parent_id="tf2",
        created_at=5,
        parts=[TextContent(text="Q3")],
        context_tokens=1,
    )
    session.active_conversation.nodes.append("u3")
    return session


def _entry() -> CompactionEntry:
    return CompactionEntry(id="cmp", created_at=5, source=CompactionSource.USER)


def _offered(session: AgentSession) -> tuple[str, ...]:
    # the path the runner offers: the active nodes, ending with the compaction entry
    return (*session.active_conversation.nodes, "cmp")


def _faux(summary: str) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses([faux_assistant_message([faux_text(summary)], finish_reason="stop")])
    return provider


class FixedSummary(SummarizingCompactionPolicy):
    """Overrides the `summarize` seam: no provider, no LLM call."""

    async def summarize(self, session, folded):
        return "OVERRIDDEN SUMMARY", UsageCounters(input=1, output=2, total_tokens=3)


# ── the gauge ──────────────────────────────────────────────────────────────


def test_calculate_context_used_sums_context_tokens_over_the_active_path():
    assert calculate_context_used(two_turn_session()) == 4


def test_get_context_window_size_falls_back_to_default_for_an_unknown_model():
    assert get_context_window_size(two_turn_session(), default=50_000) == 50_000


def test_calculate_utilization_ratio_is_used_over_window():
    assert calculate_utilization_ratio(two_turn_session(), default_window=8) == 0.5


def test_should_compact_is_true_once_utilization_reaches_the_threshold():
    assert SummarizingCompactionPolicy(default_window=8, threshold=0.4).should_compact(two_turn_session()) is True


def test_should_compact_is_false_below_the_threshold():
    assert SummarizingCompactionPolicy(default_window=8, threshold=0.9).should_compact(two_turn_session()) is False


def test_should_compact_is_false_when_disabled_however_full_the_window_is():
    policy = SummarizingCompactionPolicy(default_window=8, threshold=0.4, enabled=False)

    assert policy.should_compact(two_turn_session()) is False


# ── select_keep (the keep_turns split) ───────────────────────────────────────


def test_keep_turns_zero_keeps_nothing():
    session = two_turn_session()
    assert SummarizingCompactionPolicy(keep_turns=0).select_keep(list(session.active_conversation.nodes), session) == []


def test_keep_turns_one_keeps_the_last_exchange_including_its_user_message():
    session = two_turn_session()
    assert SummarizingCompactionPolicy(keep_turns=1).select_keep(
        list(session.active_conversation.nodes),
        session,
    ) == ["u2", "ts2", "a2", "tf2"]


def test_the_cut_lands_on_an_exchange_boundary_and_never_strands_a_tool_call():
    # a realistic path: the kept tail opens at the user message that prompted
    # the last TurnStart, so `a3`'s tool call keeps `te3` and the earlier
    # exchange keeps its own executions in the folded span
    session = RICH_IDLE_SESSION.model_copy(deep=True)

    assert SummarizingCompactionPolicy(keep_turns=1).select_keep(
        list(session.active_conversation.nodes),
        session,
    ) == ["u3", "ts3", "a3", "te3", "cr1", "tf3"]


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
                "metadata": {"keep_turns": 0, "kept": 0},
            }
        ),
        nodes=["cmp"],
        usage=UsageCounters(),
    )


async def test_keep_turns_keeps_the_tail_and_folds_the_head():
    session = two_turn_session()
    policy = SummarizingCompactionPolicy(keep_turns=1, provider=_faux("HEAD SUMMARY"))

    plan = await policy.compact(session, _offered(session), _entry())

    assert plan == CompactionPlan(
        entry=_entry().model_copy(
            update={
                "parts": [TextContent(text="HEAD SUMMARY")],
                "compacted_nodes": ["u1", "ts1", "a1", "tf1"],
                "llm_config": MODEL,
                "metadata": {"keep_turns": 1, "kept": 4},
            }
        ),
        nodes=["cmp", "u2", "ts2", "a2", "tf2"],
        usage=UsageCounters(),
    )


async def test_a_trailing_unanswered_user_message_is_never_folded():
    # auto-compaction fires with a just-posted, unanswered question on the path;
    # folding it would drop the question. Full summary must still keep it.
    session = pending_question_session()
    policy = SummarizingCompactionPolicy(provider=_faux("SUMMARY"))  # full summary

    plan = await policy.compact(session, _offered(session), _entry())

    assert plan == CompactionPlan(
        entry=_entry().model_copy(
            update={
                "parts": [TextContent(text="SUMMARY")],
                "compacted_nodes": ["u1", "ts1", "a1", "tf1", "u2", "ts2", "a2", "tf2"],
                "llm_config": MODEL,
                "metadata": {"keep_turns": 0, "kept": 1},
            }
        ),
        nodes=["cmp", "u3"],
        usage=UsageCounters(),
    )


async def test_a_tool_bearing_span_folds_through_the_projected_summary_call():
    # the folded head carries a completed and a failed tool execution, a pruned
    # entry and an earlier summary — everything the projector emits for the
    # summarization request
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    policy = SummarizingCompactionPolicy(keep_turns=1, provider=_faux("EARLIER WORK"))

    plan = await policy.compact(session, _offered(session), _entry())

    assert plan == CompactionPlan(
        entry=_entry().model_copy(
            update={
                "parts": [TextContent(text="EARLIER WORK")],
                "compacted_nodes": [
                    "cmp0",
                    "u1",
                    "ts1",
                    "a1",
                    "te1",
                    "te2",
                    "pr1",
                    "a2",
                    "tf1",
                    "u2",
                    "ts2",
                    "tf2",
                ],
                "llm_config": RICH_MODEL,
                "metadata": {"keep_turns": 1, "kept": 6},
            }
        ),
        nodes=["cmp", "u3", "ts3", "a3", "te3", "cr1", "tf3"],
        usage=UsageCounters(),
    )


async def test_compact_returns_none_when_nothing_is_older_than_the_kept_tail():
    session = two_turn_session()
    policy = SummarizingCompactionPolicy(keep_turns=5, provider=_faux("x"))

    assert await policy.compact(session, _offered(session), _entry()) is None


async def test_a_subclass_can_override_the_summarize_seam():
    session = two_turn_session()

    plan = await FixedSummary().compact(session, _offered(session), _entry())

    assert plan == CompactionPlan(
        entry=_entry().model_copy(
            update={
                "parts": [TextContent(text="OVERRIDDEN SUMMARY")],
                "compacted_nodes": ["u1", "ts1", "a1", "tf1", "u2", "ts2", "a2", "tf2"],
                "llm_config": MODEL,
                "metadata": {"keep_turns": 0, "kept": 0},
            }
        ),
        nodes=["cmp"],
        usage=UsageCounters(input=1, output=2, total_tokens=3),
    )
