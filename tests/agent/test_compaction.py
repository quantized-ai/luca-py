"""Declarative tests for the compaction CONTRACT: the policy base, the plan
value objects, and `validate_plan` / `check_snapshot` / `has_content`.

Pure and synchronous — no runner, no ledger, no provider, nothing async. One
session literal, one snapshot, one plan per test. The validator's whole job is
STRUCTURE: every rejection here is something no session could hold, and every
acceptance is a plan whose *meaning* core deliberately refuses to judge.

The session under test is a two-turn conversation `c1` with an open compaction
bracket at its tail (`ts_c`, `cmp`) and one archived conversation `c0` whose
entry is still in the store — the two shapes every "not on this path" rule
needs.
"""

import pytest
from pydantic import ValidationError

from luca.agent.core.compaction import (
    CompactionPlan,
    CompactionPolicy,
    ConversationSnapshot,
    UsageCounters,
    check_snapshot,
    has_content,
    validate_plan,
)
from luca.agent.core.exceptions import CompactionPlanError
from luca.agent.core.models import (
    AgentSession,
    AssistantMessage,
    CompactionEntry,
    CompactionSource,
    Conversation,
    ImageBase64,
    ImageContent,
    LLMConfig,
    SessionConfig,
    TextContent,
    TurnFinish,
    TurnStart,
    UserMessage,
)

MODEL = LLMConfig(model="test-model", provider="faux")
CHEAP = LLMConfig(model="cheap-model", provider="faux")

SUMMARY = [TextContent(text="the story so far")]
IMAGE = ImageContent(source=ImageBase64(data="aGk=", media_type="image/png"))

SESSION = AgentSession(
    id="s",
    entries={
        "a0": AssistantMessage(
            id="a0",
            created_at=400,
            parts=[TextContent(text="archived")],
            llm_config=MODEL,
            stop_reason="stop",
        ),
        "u1": UserMessage(
            id="u1",
            created_at=500,
            parts=[TextContent(text="hello")],
        ),
        "ts1": TurnStart(id="ts1", parent_id="u1", created_at=500),
        "a1": AssistantMessage(
            id="a1",
            parent_id="ts1",
            created_at=500,
            parts=[TextContent(text="hi")],
            llm_config=MODEL,
            stop_reason="stop",
        ),
        "tf1": TurnFinish(id="tf1", parent_id="a1", created_at=500),
        "u2": UserMessage(
            id="u2",
            parent_id="tf1",
            created_at=500,
            parts=[TextContent(text="and now?")],
        ),
        "ts_c": TurnStart(id="ts_c", parent_id="u2", created_at=600),
        "cmp": CompactionEntry(
            id="cmp",
            parent_id="ts_c",
            created_at=600,
            source=CompactionSource.POLICY,
            started_at=600,
        ),
    },
    active_conversation=Conversation(
        id="c1",
        nodes=["u1", "ts1", "a1", "tf1", "u2", "ts_c", "cmp"],
        created_at=500,
        updated_at=600,
    ),
    conversation_history=[
        Conversation(id="c0", nodes=["a0"], created_at=400, updated_at=400),
    ],
    session_config=SessionConfig(llm_config=MODEL),
)

SNAPSHOT = ConversationSnapshot(
    id="c1",
    nodes=("u1", "ts1", "a1", "tf1", "u2", "ts_c", "cmp"),
    offered=("u1", "ts1", "a1", "tf1", "u2", "cmp"),
)


# ── has_content (G3) ──────────────────────────────────────────────────────────


def test_no_parts_at_all_is_not_content():
    assert has_content(None) is False


def test_an_empty_part_list_is_not_content():
    assert has_content([]) is False


def test_an_empty_text_part_is_not_content():
    assert has_content([TextContent(text="")]) is False


def test_a_whitespace_only_text_part_is_not_content():
    assert has_content([TextContent(text="  \n\t")]) is False


def test_a_text_part_with_one_character_is_content():
    assert has_content([TextContent(text="x")]) is True


def test_an_image_only_summary_is_content():
    assert has_content([IMAGE]) is True


def test_a_blank_text_part_beside_an_image_is_content():
    assert has_content([TextContent(text="  "), IMAGE]) is True


# ── check_snapshot (G2) ───────────────────────────────────────────────────────


def test_check_snapshot_passes_when_nothing_moved():
    assert check_snapshot(session=SESSION, snapshot=SNAPSHOT) is None


def test_check_snapshot_rejects_a_replaced_active_conversation():
    session = SESSION.model_copy(deep=True)
    session.active_conversation = Conversation(
        id="c9",
        nodes=list(SNAPSHOT.nodes),
        created_at=500,
        updated_at=600,
    )

    with pytest.raises(
        CompactionPlanError,
        match="the active conversation changed under the plan",
    ):
        check_snapshot(session=session, snapshot=SNAPSHOT)


def test_check_snapshot_rejects_a_path_that_grew_under_the_plan():
    session = SESSION.model_copy(deep=True)
    session.entries["u3"] = UserMessage(
        id="u3",
        created_at=700,
        parts=[TextContent(text="late")],
    )
    session.active_conversation.nodes.append("u3")

    with pytest.raises(
        CompactionPlanError,
        match="the active conversation's path changed under the plan",
    ):
        check_snapshot(session=session, snapshot=SNAPSHOT)


# ── validate_plan: the rejection list, in evaluation order ───────────────────


def test_a_plan_computed_against_another_conversation_is_rejected():
    session = SESSION.model_copy(deep=True)
    session.active_conversation.id = "c9"
    plan = CompactionPlan(
        entry=SESSION.entries["cmp"].model_copy(update={"parts": SUMMARY}),
        nodes=["cmp"],
    )

    with pytest.raises(
        CompactionPlanError,
        match=r"changed under the plan \('c1' → 'c9'\)",
    ):
        validate_plan(plan, entry_id="cmp", session=session, snapshot=SNAPSHOT)


def test_a_plan_computed_against_a_stale_path_is_rejected():
    session = SESSION.model_copy(deep=True)
    session.active_conversation.nodes.remove("u1")
    plan = CompactionPlan(
        entry=SESSION.entries["cmp"].model_copy(update={"parts": SUMMARY}),
        nodes=["cmp"],
    )

    with pytest.raises(CompactionPlanError, match="path changed under the plan"):
        validate_plan(plan, entry_id="cmp", session=session, snapshot=SNAPSHOT)


def test_an_empty_plan_is_rejected():
    plan = CompactionPlan(
        entry=SESSION.entries["cmp"].model_copy(update={"parts": SUMMARY}),
        nodes=[],
    )

    with pytest.raises(
        CompactionPlanError,
        match="an empty plan is not a compaction",
    ):
        validate_plan(plan, entry_id="cmp", session=SESSION, snapshot=SNAPSHOT)


def test_a_plan_referencing_an_unknown_id_is_rejected():
    plan = CompactionPlan(
        entry=SESSION.entries["cmp"].model_copy(update={"parts": SUMMARY}),
        nodes=["cmp", "nowhere"],
    )

    with pytest.raises(
        CompactionPlanError,
        match="plan references unknown entry 'nowhere'",
    ):
        validate_plan(plan, entry_id="cmp", session=SESSION, snapshot=SNAPSHOT)


def test_a_plan_carrying_an_entry_from_an_archived_conversation_is_rejected():
    # `a0` IS in the store — it is simply not on the path that was offered
    plan = CompactionPlan(
        entry=SESSION.entries["cmp"].model_copy(update={"parts": SUMMARY}),
        nodes=["cmp", "a0"],
    )

    with pytest.raises(
        CompactionPlanError,
        match="plan references entry 'a0', which is not on conversation 'c1'",
    ):
        validate_plan(plan, entry_id="cmp", session=SESSION, snapshot=SNAPSHOT)


def test_a_plan_carrying_the_brackets_own_turn_start_is_rejected():
    # the whole point of `offered`: reaching around it into the live path is
    # caught by the ordinary not-on-the-path rule, with no special case
    plan = CompactionPlan(
        entry=SESSION.entries["cmp"].model_copy(update={"parts": SUMMARY}),
        nodes=["ts_c", "cmp"],
    )

    with pytest.raises(
        CompactionPlanError,
        match="plan references entry 'ts_c', which is not on conversation 'c1'",
    ):
        validate_plan(plan, entry_id="cmp", session=SESSION, snapshot=SNAPSHOT)


def test_a_plan_referencing_the_same_id_twice_is_rejected():
    plan = CompactionPlan(
        entry=SESSION.entries["cmp"].model_copy(update={"parts": SUMMARY}),
        nodes=["cmp", "u2", "u2"],
    )

    with pytest.raises(
        CompactionPlanError,
        match="plan references entry 'u2' twice",
    ):
        validate_plan(plan, entry_id="cmp", session=SESSION, snapshot=SNAPSHOT)


def test_a_plan_omitting_the_compaction_entry_is_rejected():
    plan = CompactionPlan(
        entry=SESSION.entries["cmp"].model_copy(update={"parts": SUMMARY}),
        nodes=["u2"],
    )

    with pytest.raises(
        CompactionPlanError,
        match="plan omits the compaction entry 'cmp'",
    ):
        validate_plan(plan, entry_id="cmp", session=SESSION, snapshot=SNAPSHOT)


def test_a_plan_with_no_parts_is_rejected():
    plan = CompactionPlan(entry=SESSION.entries["cmp"], nodes=["cmp"])

    with pytest.raises(CompactionPlanError, match="plan carries no content"):
        validate_plan(plan, entry_id="cmp", session=SESSION, snapshot=SNAPSHOT)


def test_a_plan_with_empty_parts_is_rejected():
    plan = CompactionPlan(
        entry=SESSION.entries["cmp"].model_copy(update={"parts": []}),
        nodes=["cmp"],
    )

    with pytest.raises(CompactionPlanError, match="plan carries no content"):
        validate_plan(plan, entry_id="cmp", session=SESSION, snapshot=SNAPSHOT)


def test_a_plan_with_whitespace_only_parts_is_rejected():
    plan = CompactionPlan(
        entry=SESSION.entries["cmp"].model_copy(
            update={"parts": [TextContent(text="   \n")]},
        ),
        nodes=["cmp"],
    )

    with pytest.raises(CompactionPlanError, match="plan carries no content"):
        validate_plan(plan, entry_id="cmp", session=SESSION, snapshot=SNAPSHOT)


# ── validate_plan: what it accepts ────────────────────────────────────────────


def test_a_fold_everything_plan_is_accepted():
    plan = CompactionPlan(
        entry=SESSION.entries["cmp"].model_copy(update={"parts": SUMMARY}),
        nodes=["cmp"],
    )

    assert (
        validate_plan(
            plan,
            entry_id="cmp",
            session=SESSION,
            snapshot=SNAPSHOT,
        )
        is None
    )


def test_carrying_the_offered_nodes_verbatim_is_accepted():
    plan = CompactionPlan(
        entry=SESSION.entries["cmp"].model_copy(update={"parts": SUMMARY}),
        nodes=list(SNAPSHOT.offered),
    )

    assert (
        validate_plan(
            plan,
            entry_id="cmp",
            session=SESSION,
            snapshot=SNAPSHOT,
        )
        is None
    )


def test_a_plan_interleaving_created_entries_between_carried_ids_is_accepted():
    plan = CompactionPlan(
        entry=SESSION.entries["cmp"].model_copy(update={"parts": SUMMARY}),
        nodes=[
            UserMessage(parts=[TextContent(text="framing")]),
            "cmp",
            UserMessage(parts=[TextContent(text="and then")]),
            "u2",
        ],
    )

    assert (
        validate_plan(
            plan,
            entry_id="cmp",
            session=SESSION,
            snapshot=SNAPSHOT,
        )
        is None
    )


def test_a_plan_reordering_carried_ids_is_accepted():
    # a scrambled path is a policy bug, not a structural impossibility
    plan = CompactionPlan(
        entry=SESSION.entries["cmp"].model_copy(update={"parts": SUMMARY}),
        nodes=["u2", "a1", "cmp", "u1"],
    )

    assert (
        validate_plan(
            plan,
            entry_id="cmp",
            session=SESSION,
            snapshot=SNAPSHOT,
        )
        is None
    )


def test_an_image_only_summary_is_accepted():
    plan = CompactionPlan(
        entry=SESSION.entries["cmp"].model_copy(update={"parts": [IMAGE]}),
        nodes=["cmp"],
    )

    assert (
        validate_plan(
            plan,
            entry_id="cmp",
            session=SESSION,
            snapshot=SNAPSHOT,
        )
        is None
    )


def test_a_plan_with_every_usage_counter_set_is_accepted():
    plan = CompactionPlan(
        entry=SESSION.entries["cmp"].model_copy(update={"parts": SUMMARY}),
        nodes=["cmp"],
        usage=UsageCounters(
            input=100,
            output=20,
            cache_read=5,
            cache_write=3,
            total_tokens=128,
        ),
    )

    assert (
        validate_plan(
            plan,
            entry_id="cmp",
            session=SESSION,
            snapshot=SNAPSHOT,
        )
        is None
    )


def test_every_non_content_field_of_the_returned_entry_is_ignored():
    # the runner applies exactly `parts`, `llm_config` and `metadata`; a stale
    # or foreign id, source or span on the copy is discarded, never validated
    plan = CompactionPlan(
        entry=SESSION.entries["cmp"].model_copy(
            update={
                "id": "someone-elses-id",
                "source": CompactionSource.USER,
                "compacted_nodes": ["u1", "made", "up"],
                "started_at": 1,
                "ended_at": 2,
                "parts": SUMMARY,
            },
        ),
        nodes=["cmp"],
    )

    assert (
        validate_plan(
            plan,
            entry_id="cmp",
            session=SESSION,
            snapshot=SNAPSHOT,
        )
        is None
    )


# ── the plan value objects ────────────────────────────────────────────────────


def test_a_plan_keeps_the_created_entry_objects_it_was_given():
    created = UserMessage(parts=[TextContent(text="framing")])

    plan = CompactionPlan(
        entry=SESSION.entries["cmp"].model_copy(update={"parts": SUMMARY}),
        nodes=[created, "cmp"],
    )

    assert plan.nodes[0] is created


def test_a_plan_rejects_an_unknown_field():
    with pytest.raises(ValidationError):
        CompactionPlan(
            entry=SESSION.entries["cmp"],
            nodes=["cmp"],
            reason="because",
        )


def test_usage_counters_default_every_counter_to_zero():
    assert UsageCounters() == UsageCounters(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
    )


def test_usage_counters_reject_a_providers_own_counter_names():
    # the failure lands in the POLICY, at plan construction — never inside the
    # runner's ledger write, where it could not be a clean plan rejection
    with pytest.raises(ValidationError):
        UsageCounters(prompt_tokens=100, completion_tokens=20)


def test_usage_counters_dump_exactly_the_kwargs_record_usage_takes():
    counters = UsageCounters(
        input=100,
        output=20,
        cache_read=5,
        cache_write=3,
        total_tokens=128,
    )

    assert counters.model_dump() == {
        "input": 100,
        "output": 20,
        "cache_read": 5,
        "cache_write": 3,
        "total_tokens": 128,
    }


def test_a_snapshot_is_frozen():
    with pytest.raises(ValidationError):
        SNAPSHOT.id = "c9"


# ── the policy base ───────────────────────────────────────────────────────────


def test_the_base_policy_never_compacts():
    assert CompactionPolicy().should_compact(SESSION) is False


async def test_the_base_policy_has_no_compact_implementation():
    with pytest.raises(NotImplementedError):
        await CompactionPolicy().compact(
            SESSION,
            SNAPSHOT.offered,
            SESSION.entries["cmp"],
        )


async def test_a_subclass_implements_compact_over_session_nodes_and_entry():
    class Folding(CompactionPolicy):
        def should_compact(self, session):
            return True

        async def compact(self, session, nodes, entry):
            return CompactionPlan(
                entry=entry.model_copy(
                    update={"parts": SUMMARY, "llm_config": CHEAP},
                ),
                nodes=[entry.id],
                usage=UsageCounters(input=100, output=20, total_tokens=120),
            )

    policy = Folding()

    plan = await policy.compact(
        SESSION,
        SNAPSHOT.offered,
        SESSION.entries["cmp"].model_copy(deep=True),
    )

    assert policy.should_compact(SESSION) is True
    assert plan == CompactionPlan(
        entry=SESSION.entries["cmp"].model_copy(
            update={"parts": SUMMARY, "llm_config": CHEAP},
        ),
        nodes=["cmp"],
        usage=UsageCounters(input=100, output=20, total_tokens=120),
    )
