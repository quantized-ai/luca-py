"""Integration scenarios for compaction: the drive step, the brackets, the
transition, the events, and every way the operation can end.

The mandate this file answers: a strategy author's unit is `compact()`, so the
session/conversation handling around it must be proven coherent for EVERY
ending. Almost every scenario therefore asserts the same three things — the
new conversation is what it should be, the pre-existing entries are unmutated,
and the old conversation stays in `conversations` intact, named as the
successor's `previous_conversation_id`.

House style: precondition (a literal from `scenarios.py`, deep-copied, or a
cold reload) → one action → full-object postcondition. `DeterministicRunner`
with a scripted `ids` list and a frozen clock (`now=1000`, so entries written
by the drive are visually distinct from the literals' `created_at=500`);
`FakeContextManager` for the compacting manager; `FauxProvider` for the turn
that follows. Never race two timed things — a hanging manager waits on an event
the test releases.

The `ids` scripts are the determinism contract, in draw order:
`schedule_compaction()` and a drive-top open each draw `TurnStart` then
`CompactionEntry`; a non-transition ending draws the closing `TurnFinish`; a
transition draws each created entry in plan order, then the closing
`TurnFinish`, then the new conversation id.
"""

import asyncio

import pytest

from luca.agent.core.compaction import CompactionPlan, UsageCounters
from luca.agent.core.context_manager import ContextManager
from luca.agent.core.events import (
    CompactionFinished,
    CompactionScheduled,
    CompactionStarted,
)
from luca.agent.core.exceptions import (
    AgentError,
    AlreadyCancellingError,
    CompactionPlanError,
)
from luca.agent.core.models import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    CompactionEntry,
    CompactionSource,
    Conversation,
    ConversationStatus,
    ExecutionAttempt,
    ExecutionAttemptOutcome,
    ExecutionResult,
    ExecutionStatus,
    ImageContent,
    MediaBase64,
    RuntimeConfig,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    TurnFinish,
    TurnOutcome,
    Usage,
    UserMessage,
)
from luca.agent.core.projection import ConversationProjector
from luca.agent.core.runner import RunResult
from luca.client.exceptions import ProviderAPIError, TimeoutError as ClientTimeoutError
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text
from luca.client.types import TextBlock, UserMessage as LucaUserMessage
from tests.agent.scenarios import (
    ADD_SPEC,
    CANCEL_PARKED_SESSION,
    CHEAP,
    CLEARED_SESSION,
    COMPACTION_BURIED_SESSION,
    COMPACTION_CANCEL_PARKED_SESSION,
    COMPACTION_FAILED_SESSION,
    COMPACTION_INTERRUPTED_SESSION,
    COMPACTION_SCHEDULED_SESSION,
    GATED_SESSION,
    MODEL,
    MULTIPLY_SPEC,
    POST_COMPACTION_SESSION,
    READ_FILE_SPEC,
    RICH_IDLE_SESSION,
    RICH_SESSION,
    DeterministicRunner,
    FakeContextManager,
    conversation,
    main_conversation,
    spec,
)

SUMMARY = [TextContent(text="Everything so far, in brief.")]  # 28 chars → 7
USAGE = UsageCounters(input=500, output=40, total_tokens=540)

# The spec of a tool no session literal ever called, so a row for it in
# `tool_specs` can only have been filed by the transition door itself.
ECHO_SPEC = spec("echo")

# `c1`'s path before any compaction, and the ids a fold-everything plan
# replaces on the trailing-question session.
RICH_NODES = list(main_conversation(RICH_SESSION).nodes)
RICH_IDLE_NODES = list(main_conversation(RICH_IDLE_SESSION).nodes)


# ── policy doubles ────────────────────────────────────────────────────────────


class CancellingPolicy(FakeContextManager):
    """Stands in for a cancel arriving MID-SUMMARY: it requests the cancel
    itself, then hangs forever — so the token trips while `compact()` is
    genuinely in flight, with no second timer. The test wires `runner` after
    construction."""

    runner = None

    def __init__(self) -> None:
        super().__init__()
        self.hard_cancelled = False

    async def compact(self, session, conversation_id, nodes, entry):
        self.seen.append((session, nodes, entry))
        self.runner.cancel()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.hard_cancelled = True
            raise
        return


# ── plan builders (the policy's judgment, scripted) ───────────────────────────
#
# Each is a `plan=` callable for `FakeContextManager`: it receives the live
# session, the offered path, and the deep copy of the entry — whose `id` is
# the only thing a plan cannot know in advance.


def fold_everything(session, nodes, entry):
    return CompactionPlan(
        entry=entry.model_copy(
            update={
                "parts": SUMMARY,
                "llm_config": CHEAP,
                "metadata": {"strategy": "fold-everything"},
            },
        ),
        nodes=[entry.id],
        usage=USAGE,
    )


def fold_and_keep_the_question(session, nodes, entry):
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": SUMMARY, "llm_config": CHEAP}),
        nodes=[entry.id, "u4"],
        usage=USAGE,
    )


def keep_the_last_assistant(session, nodes, entry):
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": SUMMARY, "llm_config": CHEAP}),
        nodes=[entry.id, "a2"],
        usage=USAGE,
    )


def full_carry(session, nodes, entry):
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": SUMMARY, "llm_config": CHEAP}),
        nodes=list(nodes),
        usage=USAGE,
    )


def frame_then_summarize(session, nodes, entry):
    # opens with a created entry, interleaves another between carried ids, and
    # leaves a carried assistant message as the leaf
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": SUMMARY, "llm_config": CHEAP}),
        nodes=[
            UserMessage(parts=[TextContent(text="[history compacted]")]),
            entry.id,
            UserMessage(parts=[TextContent(text="[continue from here]")]),
            "a2",
        ],
        usage=USAGE,
    )


def fold_and_create_a_tool_execution(session, nodes, entry):
    # a plan may invent any entry type, a `ToolExecution` included — the only
    # way one ever reaches the transition door's `created` list
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": SUMMARY, "llm_config": CHEAP}),
        nodes=[
            entry.id,
            ToolExecution(
                tool_call_id="tc9",
                raw_tool_call=ToolCall(id="tc9", name="echo", arguments={}),
                tool_spec=ECHO_SPEC,
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(content=[TextContent(text="echoed")]),
            ),
        ],
        usage=USAGE,
    )


def reorder_the_path(session, nodes, entry):
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": SUMMARY, "llm_config": CHEAP}),
        nodes=["u1", entry.id, "a2"],
        usage=USAGE,
    )


def carry_the_bracket_turn_start(session, nodes, entry):
    # reaches around the offered view into the live path
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": SUMMARY}),
        nodes=["ts_c", entry.id],
        usage=USAGE,
    )


def carry_an_unknown_id(session, nodes, entry):
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": SUMMARY}),
        nodes=[entry.id, "nowhere"],
        usage=USAGE,
    )


def carry_an_archived_id(session, nodes, entry):
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": SUMMARY}),
        nodes=[entry.id, "u0"],
        usage=USAGE,
    )


def carry_the_same_id_twice(session, nodes, entry):
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": SUMMARY}),
        nodes=[entry.id, "u1", "u1"],
        usage=USAGE,
    )


def empty_plan(session, nodes, entry):
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": SUMMARY}),
        nodes=[],
        usage=USAGE,
    )


def omit_the_compaction_entry(session, nodes, entry):
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": SUMMARY}),
        nodes=["u1"],
        usage=USAGE,
    )


def summary_with_no_content(session, nodes, entry):
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": [TextContent(text="  \n")]}),
        nodes=[entry.id],
        usage=USAGE,
    )


def image_only_summary(session, nodes, entry):
    return CompactionPlan(
        entry=entry.model_copy(
            update={
                "parts": [
                    ImageContent(
                        source=MediaBase64(data="aGk=", media_type="image/png"),
                    ),
                ],
                "llm_config": CHEAP,
            },
        ),
        nodes=[entry.id],
        usage=USAGE,
    )


def replace_the_compacting_conversation(session, nodes, entry):
    """A manager that installed a successor and re-pointed the name under the
    runner — the shape G2 exists to catch."""
    session.conversations["c9"] = Conversation(
        id="c9",
        nodes=list(nodes),
        created_at=1000,
        updated_at=1000,
    )
    session.main_conversation_id = "c9"
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": SUMMARY}),
        nodes=[entry.id],
        usage=USAGE,
    )


def append_to_the_active_path(session, nodes, entry):
    session.entries["late"] = UserMessage(
        id="late",
        created_at=999,
        parts=[TextContent(text="late")],
    )
    main_conversation(session).nodes.append("late")
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": SUMMARY}),
        nodes=[entry.id],
        usage=USAGE,
    )


def carry_a_phantom_open_turn(session, nodes, entry):
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": SUMMARY, "llm_config": CHEAP}),
        nodes=[entry.id, "ts3", "u4"],
        usage=USAGE,
    )


def carry_the_failed_turn_finish(session, nodes, entry):
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": SUMMARY, "llm_config": CHEAP}),
        nodes=[entry.id, "u2", "ts2", "tf2"],
        usage=USAGE,
    )


def bracket_the_summary_with_a_carried_turn_start(session, nodes, entry):
    # the counterfeit shape: [ts3, cmp, …] reads as an open COMPACTION bracket
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": SUMMARY, "llm_config": CHEAP}),
        nodes=["ts3", entry.id, "u4"],
        usage=USAGE,
    )


def summarize_only_the_previous_summary(session, nodes, entry):
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": SUMMARY, "llm_config": CHEAP}),
        nodes=[node for node in nodes if node != "cmp0"],
        usage=USAGE,
    )


# ── A. the successful transition ──────────────────────────────────────────────


async def test_the_transition_archives_the_old_path_and_installs_the_new_one():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=fold_everything),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    assert runner.session.conversations["c0"] == RICH_SESSION.conversations["c0"]
    assert [
        RICH_SESSION.conversations["c0"],
        Conversation(
            id="c1",
            nodes=[*RICH_IDLE_NODES, "ts_c", "cmp", "tf_c"],
            created_at=500,
            updated_at=1000,
        ),
    ]
    assert main_conversation(runner.session) == Conversation(
        id="c2",
        previous_conversation_id="c1",
        nodes=["cmp"],
        created_at=1000,
        updated_at=1000,
    )


async def test_every_carried_and_compacted_entry_is_unmutated():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    before = {k: v.model_copy(deep=True) for k, v in session.entries.items()}
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=fold_everything),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    entries = runner.session.entries
    assert {k: v for k, v in entries.items() if k not in ("ts_c", "cmp", "tf_c")} == before
    assert set(entries) - set(before) == {"ts_c", "cmp", "tf_c"}


async def test_compacted_nodes_lists_exactly_the_replaced_ids_in_path_order():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=fold_everything),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    # the whole pre-bracket path, in order, and NOT the bracket's own ts_c
    assert runner.session.entries["cmp"].compacted_nodes == RICH_IDLE_NODES


async def test_the_bracket_stays_behind_and_only_the_entry_carries_over():
    session = RICH_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(
            should=True,
            plan=fold_and_keep_the_question,
        ),
        provider=_answering_provider(),
        ids=["ts_c", "cmp", "tf_c", "c2", "ts4", "a4", "tf4"],
        now=1000,
    )

    await runner.run()

    archived = runner.session.conversations["c1"]
    assert archived.nodes[-3:] == ["ts_c", "cmp", "tf_c"]
    assert main_conversation(runner.session).nodes[0] == "cmp"
    assert "ts_c" not in main_conversation(runner.session).nodes
    assert "tf_c" not in main_conversation(runner.session).nodes


async def test_the_pruned_referent_and_the_archived_conversation_stay_reachable():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=fold_everything),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    assert runner.session.entries["te0"] == RICH_SESSION.entries["te0"]
    assert runner.session.entries["u0"] == RICH_SESSION.entries["u0"]
    assert runner.session.conversations["c0"] == RICH_SESSION.conversations["c0"]


async def test_the_old_usage_records_survive():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=fold_everything),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    assert runner.session.usages["c0"] == RICH_SESSION.usages["c0"]
    assert runner.session.usages["c1"] == {
        **RICH_SESSION.usages["c1"],
        "cmp": Usage(
            conversation_id="c1",
            entry_id="cmp",
            input=500,
            output=40,
            total_tokens=540,
        ),
    }
    assert "c2" not in runner.session.usages


async def test_the_entry_is_self_describing_on_the_new_path():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=fold_everything),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    assert runner.session.entries["cmp"] == CompactionEntry(
        id="cmp",
        parent_id="ts_c",
        created_at=1000,
        context_tokens=7,
        source=CompactionSource.USER,
        parts=SUMMARY,
        compacted_nodes=RICH_IDLE_NODES,
        llm_config=CHEAP,  # the POLICY's model, not the session's
        started_at=1000,
        ended_at=1000,
        metadata={"strategy": "fold-everything"},
    )


async def test_a_plan_with_created_entries_stamps_and_threads_them_in_order():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=frame_then_summarize),
        ids=["ts_c", "cmp", "new1", "new2", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    assert main_conversation(runner.session).nodes == [
        "new1",
        "cmp",
        "new2",
        "a2",
    ]
    # ids in plan order, `parent_id` threaded left to right, one shared
    # `created_at` for the whole transition
    assert runner.session.entries["new1"] == UserMessage(
        id="new1",
        parent_id=None,
        created_at=1000,
        context_tokens=4,
        parts=[TextContent(text="[history compacted]")],
    )
    assert runner.session.entries["new2"] == UserMessage(
        id="new2",
        parent_id="cmp",
        created_at=1000,
        context_tokens=5,
        parts=[TextContent(text="[continue from here]")],
    )


async def test_a_plan_that_opens_with_a_created_entry_gives_it_no_parent():
    # a conversation's first node has no parent — already true of every
    # session's first entry
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=frame_then_summarize),
        ids=["ts_c", "cmp", "new1", "new2", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    assert runner.session.entries["new1"].parent_id is None


async def test_a_plan_may_reorder_carried_ids():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=reorder_the_path),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    # the policy chose the path; the runner validates structure, not sense
    assert main_conversation(runner.session).nodes == ["u1", "cmp", "a2"]


async def test_a_fold_everything_plan_leaves_the_compaction_entry_as_the_leaf():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=fold_everything),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    result = await runner.run()

    assert main_conversation(runner.session).nodes == ["cmp"]
    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.COMPLETED,
        pending_approvals=[],
    )


async def test_a_keep_last_assistant_plan_leaves_an_assistant_leaf():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=keep_the_last_assistant),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    result = await runner.run()

    assert main_conversation(runner.session).nodes == ["cmp", "a2"]
    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.COMPLETED,
        pending_approvals=[],
    )


async def test_a_full_carry_plan_commits_with_an_empty_compacted_span():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    policy = FakeContextManager(plan=full_carry)
    runner = DeterministicRunner(
        session,
        context_manager=policy,
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    # `[]`, not None — nothing was replaced, and that is the policy's call
    assert runner.session.entries["cmp"].compacted_nodes == []
    assert main_conversation(runner.session).nodes == [
        *RICH_IDLE_NODES,
        "cmp",
    ]


async def test_a_span_of_only_a_previous_summary_is_committed():
    # core does not judge whether re-summarizing a summary was worthwhile
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(
            plan=summarize_only_the_previous_summary,
        ),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    assert runner.session.entries["cmp"].compacted_nodes == ["cmp0"]
    assert main_conversation(runner.session).previous_conversation_id == "c1"


async def test_the_next_request_projects_only_the_new_path():
    faux = _answering_provider()
    session = RICH_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=FakeContextManager(
            should=True,
            plan=fold_and_keep_the_question,
        ),
        ids=["ts_c", "cmp", "tf_c", "c2", "ts4", "a4", "tf4"],
        now=1000,
    )

    await runner.run()

    assert faux.requests[0].messages == [
        LucaUserMessage(content=[TextBlock(text="Everything so far, in brief.")]),
        LucaUserMessage(content=[TextBlock(text="What is X?")]),
    ]


async def test_the_compacted_session_round_trips_through_json():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=fold_everything),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    reloaded = AgentSession.model_validate_json(
        runner.session.model_dump_json(),
    )

    assert reloaded == runner.session


async def test_a_created_tool_execution_has_its_spec_filed_and_survives_a_reload():
    # `transition_conversation` is the tool-spec write door no ordinary tool
    # call travels through: only a compaction plan puts an execution on it.
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(
            plan=fold_and_create_a_tool_execution,
        ),
        ids=["ts_c", "cmp", "te9", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    reloaded = AgentSession.model_validate_json(
        runner.session.model_dump_json(),
    )

    assert reloaded == runner.session
    # `conversation_id` stays None: a plan-INVENTED execution has no birth
    # conversation the runner can name — the one it lands in is minted inside
    # the transition, after this entry is built. The field is provenance for
    # executions the runner creates from a model tool call, and nothing
    # resolves a path through it.
    assert reloaded.entries["te9"] == ToolExecution(
        id="te9",
        parent_id="cmp",
        created_at=1000,
        context_tokens=1,  # "echoed" → 6 // 4
        tool_call_id="tc9",
        raw_tool_call=ToolCall(id="tc9", name="echo", arguments={}),
        tool_spec=ECHO_SPEC,
        tool_spec_id=ECHO_SPEC.spec_id(),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="echoed")]),
    )
    assert reloaded.tool_specs == {
        ADD_SPEC.spec_id(): ADD_SPEC,
        READ_FILE_SPEC.spec_id(): READ_FILE_SPEC,
        MULTIPLY_SPEC.spec_id(): MULTIPLY_SPEC,
        ECHO_SPEC.spec_id(): ECHO_SPEC,
    }


async def test_an_updated_tool_execution_has_its_spec_filed_and_survives_a_reload():
    # the same door, the other list: `updates` is public and typed
    # `list[AnyEntry]`, so an execution can arrive there too — here with its
    # spec REPLACED, which is also why the id is recomputed rather than kept.
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["c2"], now=1000)
    respecced = session.entries["te1"].model_copy(update={"tool_spec": ECHO_SPEC})

    runner.ledger.transition_conversation(
        session.main_conversation_id,
        updates=[respecced],
        created=[],
        closing=None,
        nodes=["te1"],
        ts=1000,
    )

    reloaded = AgentSession.model_validate_json(
        runner.session.model_dump_json(),
    )

    assert reloaded == runner.session
    assert reloaded.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=500,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ECHO_SPEC,
        tool_spec_id=ECHO_SPEC.spec_id(),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="3")]),
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[
            ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=500),
        ],
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=500, ended_at=500)],
        finished_at=500,
        updated_at=500,
    )
    # the displaced `add` row stays: `tool_specs` is append-only, and te0/te2/
    # te3 still reference the other two
    assert reloaded.tool_specs == {
        ADD_SPEC.spec_id(): ADD_SPEC,
        READ_FILE_SPEC.spec_id(): READ_FILE_SPEC,
        MULTIPLY_SPEC.spec_id(): MULTIPLY_SPEC,
        ECHO_SPEC.spec_id(): ECHO_SPEC,
    }


async def test_the_events_are_scheduled_started_finished():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=fold_everything),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )
    runner.schedule_compaction()

    async with runner.run() as run:
        events = [event async for event in run]

    assert events == [
        CompactionScheduled(
            conversation_id="c1",
            entry=CompactionEntry(
                id="cmp",
                parent_id="ts_c",
                created_at=1000,
                source=CompactionSource.USER,
            ),
        ),
        CompactionStarted(
            conversation_id="c1",
            entry=CompactionEntry(
                id="cmp",
                parent_id="ts_c",
                created_at=1000,
                source=CompactionSource.USER,
                started_at=1000,
            ),
        ),
        CompactionFinished(
            conversation_id="c1",
            entry=runner.session.entries["cmp"],
            outcome=TurnOutcome.COMPLETED,
            error=None,
            created=[],
            new_conversation_id="c2",
        ),
    ]


async def test_the_policy_is_handed_a_deep_copy_and_the_live_session():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    policy = FakeContextManager(plan=fold_everything)
    runner = DeterministicRunner(
        session,
        context_manager=policy,
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    seen_session, _, seen_entry = policy.seen[0]
    assert seen_session is runner.session
    assert seen_entry is not runner.session.entries["cmp"]


async def test_the_policy_is_offered_the_path_without_the_bracket_turn_start():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    policy = FakeContextManager(plan=fold_everything)
    runner = DeterministicRunner(
        session,
        context_manager=policy,
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    offered = policy.seen[0][1]
    assert offered == (*RICH_IDLE_NODES, "cmp")
    assert isinstance(offered, tuple)
    assert "ts_c" not in offered


async def test_a_policy_that_writes_to_its_copy_and_fails_cannot_inject_a_summary():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(
            mutate=True,
            raises=ValueError("kaboom"),
        ),
        ids=["ts_c", "cmp", "tf_c"],
        now=1000,
    )
    runner.schedule_compaction()

    with pytest.raises(ValueError, match="kaboom"):
        await runner.run()

    assert runner.session.entries["cmp"].parts is None
    assert ConversationProjector().project(
        main_conversation(runner.session).nodes,
        runner.session.entries,
    ) == ConversationProjector().project(
        main_conversation(RICH_IDLE_SESSION).nodes,
        RICH_IDLE_SESSION.entries,
    )


# ── B. nothing to compact ─────────────────────────────────────────────────────


async def test_a_policy_returning_none_closes_completed_without_transitioning():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=None),
        ids=["ts_c", "cmp", "tf_c"],
        now=1000,
    )
    runner.schedule_compaction()

    async with runner.run() as run:
        events = [event async for event in run]

    assert main_conversation(runner.session).nodes == [
        *RICH_IDLE_NODES,
        "ts_c",
        "cmp",
        "tf_c",
    ]
    assert runner.session.entries["cmp"].parts is None
    assert runner.session.entries["tf_c"].outcome == TurnOutcome.COMPLETED
    assert len(runner.session.conversations) == len(RICH_SESSION.conversations)
    assert events[-1] == CompactionFinished(
        conversation_id="c1",
        entry=runner.session.entries["cmp"],
        outcome=TurnOutcome.COMPLETED,
        error=None,
        created=[],
        new_conversation_id=None,
    )


async def test_scheduling_on_an_empty_session_compacts_nothing():
    session = AgentSession(
        id="s_fresh",
        conversations={
            "c1": conversation(
                "c1",
                [],
                created_at=500,
                updated_at=500,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=None),
        ids=["ts_c", "cmp", "tf_c"],
        now=1000,
    )

    runner.schedule_compaction()
    result = await runner.run()

    assert main_conversation(runner.session).nodes == ["ts_c", "cmp", "tf_c"]
    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.COMPLETED,
        pending_approvals=[],
    )


async def test_a_noop_compaction_does_not_bury_a_queued_message():
    faux = _answering_provider()
    session = RICH_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=FakeContextManager(should=True, plan=None),
        ids=["ts_c", "cmp", "tf_c", "ts4", "a4", "tf4"],
        now=1000,
    )

    await runner.run()

    # the drive goes on and answers u4 in the same run
    assert len(faux.requests) == 1
    assert runner.session.entries["a4"].parts == [TextContent(text="X is 42.")]
    assert runner.idle()


# ── C. rejected plans ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (carry_an_unknown_id, "plan references unknown entry 'nowhere'"),
        (carry_an_archived_id, "plan references entry 'u0', which is not on"),
        (carry_the_bracket_turn_start, "plan references entry 'ts_c'"),
        (carry_the_same_id_twice, "plan references entry 'u1' twice"),
        (empty_plan, "an empty plan is not a compaction"),
        (omit_the_compaction_entry, "plan omits the compaction entry 'cmp'"),
        (summary_with_no_content, "plan carries no content"),
    ],
)
async def test_a_malformed_plan_is_refused_and_the_conversation_is_unchanged(
    plan,
    message,
):
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=plan),
        ids=["ts_c", "cmp", "tf_c"],
        now=1000,
    )
    runner.schedule_compaction()

    with pytest.raises(CompactionPlanError, match=message):
        await runner.run()

    assert runner.session.entries["cmp"].parts is None
    assert main_conversation(runner.session).nodes == [
        *RICH_IDLE_NODES,
        "ts_c",
        "cmp",
        "tf_c",
    ]
    assert runner.session.entries["tf_c"].outcome == TurnOutcome.ERRORED
    assert message in runner.session.entries["tf_c"].error
    assert len(runner.session.conversations) == len(RICH_SESSION.conversations)


async def test_a_plan_computed_against_a_replaced_conversation_is_refused():
    # G2 by conversation id
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(
            plan=replace_the_compacting_conversation,
        ),
        ids=["ts_c", "cmp", "tf_c"],
        now=1000,
    )
    runner.schedule_compaction()

    with pytest.raises(
        CompactionPlanError,
        match="the compacting conversation changed",
    ):
        await runner.run()

    assert runner.session.entries["cmp"].parts is None
    assert runner.session.entries["tf_c"].outcome == TurnOutcome.ERRORED
    # The runner committed NOTHING: `c9` is the policy's own doing, and no
    # conversation names `c1` as the one it replaced.
    assert [c.id for c in runner.session.conversations.values() if c.previous_conversation_id == "c1"] == []


async def test_a_plan_computed_against_a_path_that_moved_is_refused():
    # G2 by path: the policy appended to `main_conversation(session).nodes`
    # under its own plan
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=append_to_the_active_path),
        ids=["ts_c", "cmp", "tf_c"],
        now=1000,
    )
    runner.schedule_compaction()

    with pytest.raises(CompactionPlanError, match="path changed under the plan"):
        await runner.run()

    assert runner.session.entries["cmp"].parts is None
    assert runner.session.entries["tf_c"].outcome == TurnOutcome.ERRORED
    # The runner committed NOTHING: `c9` is the policy's own doing, and no
    # conversation names `c1` as the one it replaced.
    assert [c.id for c in runner.session.conversations.values() if c.previous_conversation_id == "c1"] == []


async def test_an_image_only_summary_is_committed():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=image_only_summary),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    assert runner.session.entries["cmp"].context_tokens == 1_000
    assert main_conversation(runner.session).nodes == ["cmp"]


async def test_usage_is_recorded_for_a_rejected_plan():
    # the tokens were spent; a paid-for compaction that left no trace of the
    # spend would be the worse outcome
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=carry_an_unknown_id),
        ids=["ts_c", "cmp", "tf_c"],
        now=1000,
    )
    runner.schedule_compaction()

    with pytest.raises(CompactionPlanError):
        await runner.run()

    assert runner.session.usages["c1"]["cmp"] == Usage(
        conversation_id="c1",
        entry_id="cmp",
        input=500,
        output=40,
        total_tokens=540,
    )


async def test_a_policy_that_replaced_the_conversation_still_gets_g2s_error():
    # G2 is checked BEFORE the usage write, or `record_usage` raises its own
    # unrelated error about an entry that is no longer on the active path
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=replace_the_compacting_conversation),
        ids=["ts_c", "cmp", "tf_c"],
        now=1000,
    )
    runner.schedule_compaction()

    with pytest.raises(
        CompactionPlanError,
        match="the compacting conversation changed under the plan",
    ):
        await runner.run()


async def test_a_rejected_plan_leaves_the_entry_projecting_nothing():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=empty_plan),
        ids=["ts_c", "cmp", "tf_c"],
        now=1000,
    )
    runner.schedule_compaction()

    with pytest.raises(CompactionPlanError):
        await runner.run()

    assert ConversationProjector().project(
        main_conversation(runner.session).nodes,
        runner.session.entries,
    ) == ConversationProjector().project(
        main_conversation(RICH_IDLE_SESSION).nodes,
        RICH_IDLE_SESSION.entries,
    )


# ── D. failures and the source split ─────────────────────────────────────────


async def test_a_user_policy_raise_closes_errored_and_propagates():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(raises=ValueError("kaboom")),
        ids=["ts_c", "cmp", "tf_c"],
        now=1000,
    )
    runner.schedule_compaction()

    with pytest.raises(ValueError, match="kaboom"):
        await runner.run()

    assert runner.session.entries["tf_c"] == TurnFinish(
        id="tf_c",
        parent_id="cmp",
        created_at=1000,
        outcome=TurnOutcome.ERRORED,
        error="kaboom",
    )
    assert runner.idle()  # the closed bracket is transparent — no spin


async def test_an_iterated_user_failure_yields_finished_before_raising():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(raises=ValueError("kaboom")),
        ids=["ts_c", "cmp", "tf_c"],
        now=1000,
    )
    runner.schedule_compaction()
    events = []

    # The append loop is load-bearing: the error fires mid-iteration and the
    # assertions below read the events captured before it, which a comprehension
    # would discard. Hence the PERF401/PT012 suppressions.
    with pytest.raises(ValueError, match="kaboom"):  # noqa: PT012
        async with runner.run() as run:
            async for event in run:
                events.append(event)  # noqa: PERF401

    assert [event.type for event in events] == [
        "compaction_scheduled",
        "compaction_started",
        "compaction_finished",
    ]
    assert events[-1].outcome == TurnOutcome.ERRORED
    assert events[-1].error == "kaboom"


async def test_a_policy_source_failure_degrades_and_the_queued_turn_still_runs():
    faux = _answering_provider()
    session = RICH_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=FakeContextManager(
            should=True,
            raises=ValueError("kaboom"),
        ),
        ids=["ts_c", "cmp", "tf_c", "ts4", "a4", "tf4"],
        now=1000,
    )

    result = await runner.run()

    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.COMPLETED,  # the TURN's close, not the compaction's
        pending_approvals=[],
    )
    assert runner.session.entries["tf_c"].outcome == TurnOutcome.ERRORED
    # the failed compaction bracket AND the new turn both land on c1
    assert main_conversation(runner.session).nodes[-7:] == [
        "u4",
        "ts_c",
        "cmp",
        "tf_c",
        "ts4",
        "a4",
        "tf4",
    ]
    assert runner.session.main_conversation_id == "c1"  # no successor was installed
    assert set(runner.session.conversations) == {"c0", "c1"}
    assert len(faux.requests) == 1


async def test_a_degraded_failure_is_only_visible_on_the_event_stream():
    faux = _answering_provider()
    session = RICH_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=FakeContextManager(
            should=True,
            raises=ValueError("kaboom"),
        ),
        ids=["ts_c", "cmp", "tf_c", "ts4", "a4", "tf4"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    finished = [e for e in events if e.type == "compaction_finished"]
    assert finished[0].outcome == TurnOutcome.ERRORED
    assert finished[0].error == "kaboom"
    assert finished[0].new_conversation_id is None


async def test_a_client_timeout_from_the_policy_closes_timed_out():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(
            raises=ClientTimeoutError("the summarization call timed out"),
        ),
        ids=["ts_c", "cmp", "tf_c"],
        now=1000,
    )
    runner.schedule_compaction()

    with pytest.raises(ClientTimeoutError):
        await runner.run()

    assert runner.session.entries["tf_c"].outcome == TurnOutcome.TIMED_OUT


async def test_a_provider_error_from_the_policy_closes_errored():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(
            raises=ProviderAPIError("upstream is down"),
        ),
        ids=["ts_c", "cmp", "tf_c"],
        now=1000,
    )
    runner.schedule_compaction()

    with pytest.raises(ProviderAPIError):
        await runner.run()

    assert runner.session.entries["tf_c"].outcome == TurnOutcome.ERRORED


async def test_the_policy_deadline_closes_timed_out_and_a_user_compaction_raises():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    session.session_config.runtime_config = RuntimeConfig(
        client_completion_timeout_in_ms=50,
    )
    policy = FakeContextManager(hang=True)
    runner = DeterministicRunner(
        session,
        context_manager=policy,
        ids=["ts_c", "cmp", "tf_c"],
        now=1000,
    )
    runner.schedule_compaction()

    with pytest.raises(ClientTimeoutError):
        await runner.run()

    assert runner.session.entries["tf_c"].outcome == TurnOutcome.TIMED_OUT
    assert runner.session.entries["cmp"].parts is None


async def test_the_policy_deadline_degrades_for_a_policy_source_compaction():
    faux = _answering_provider()
    session = RICH_SESSION.model_copy(deep=True)
    session.session_config.runtime_config = RuntimeConfig(
        client_completion_timeout_in_ms=50,
    )
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=FakeContextManager(should=True, hang=True),
        ids=["ts_c", "cmp", "tf_c", "ts4", "a4", "tf4"],
        now=1000,
    )

    result = await runner.run()

    assert runner.session.entries["tf_c"].outcome == TurnOutcome.TIMED_OUT
    assert result.outcome == TurnOutcome.COMPLETED  # the turn still ran
    assert len(faux.requests) == 1


async def test_middleware_raising_during_preparation_leaves_the_path_unchanged():
    class RefusesTheSummary:
        def before_entry_written(self, session, conversation_id, entry):
            if isinstance(entry, CompactionEntry) and entry.parts:
                raise RuntimeError("no summaries here")
            return entry

    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=fold_everything),
        middleware=[RefusesTheSummary()],
        ids=["ts_c", "cmp", "tf_c"],
        now=1000,
    )
    runner.schedule_compaction()

    with pytest.raises(RuntimeError, match="no summaries here"):
        await runner.run()

    assert len(runner.session.conversations) == len(RICH_SESSION.conversations)
    assert runner.session.entries["cmp"].parts is None
    assert runner.session.entries["tf_c"].outcome == TurnOutcome.ERRORED


async def test_middleware_raising_while_closing_a_failed_bracket_leaves_it_open():
    class RefusesTheClose:
        def before_entry_written(self, session, conversation_id, entry):
            if isinstance(entry, CompactionEntry) and entry.ended_at is not None:
                raise RuntimeError("no closing here")
            return entry

    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(raises=ValueError("kaboom")),
        middleware=[RefusesTheClose()],
        ids=["ts_c", "cmp", "tf_c"],
        now=1000,
    )
    runner.schedule_compaction()

    with pytest.raises(RuntimeError, match="no closing here"):
        await runner.run()

    assert "tf_c" not in runner.session.entries  # the bracket is still open
    assert runner.busy()  # resumable


async def test_repeated_policy_failures_burn_one_attempt_per_drive():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("X is 42.")], finish_reason="stop"),
            faux_assistant_message([faux_text("Still 42.")], finish_reason="stop"),
        ]
    )
    session = RICH_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=FakeContextManager(
            should=True,
            raises=ValueError("kaboom"),
        ),
        ids=[
            "ts_c",
            "cmp",
            "tf_c",
            "ts4",
            "a4",
            "tf4",
            "u5",
            "ts_c2",
            "cmp2",
            "tf_c2",
            "ts5",
            "a5",
            "tf5",
        ],
        now=1000,
    )

    await runner.run()
    runner.post_message("again?")
    await runner.run()

    nodes = main_conversation(runner.session).nodes
    assert nodes.count("cmp") == 1
    assert nodes.count("cmp2") == 1
    assert len(runner.session.conversations) == len(RICH_SESSION.conversations)


# ── E. cancellation ───────────────────────────────────────────────────────────


async def test_cancelling_between_scheduled_and_started_closes_cancelled():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    policy = FakeContextManager(plan=fold_everything)
    runner = DeterministicRunner(
        session,
        context_manager=policy,
        ids=["ts_c", "cmp", "cr", "tf_c"],
        now=1000,
    )
    runner.schedule_compaction()

    async with runner.run() as run:
        events = []
        async for event in run:
            events.append(event)
            if event.type == "compaction_scheduled":
                run.cancel()

    assert policy.seen == []  # the policy was never called
    assert runner.session.entries["tf_c"].outcome == TurnOutcome.CANCELLED
    assert runner.session.entries["cmp"].parts is None
    assert len(runner.session.conversations) == len(RICH_SESSION.conversations)


async def test_cancelling_mid_summary_closes_cancelled_and_tears_the_policy_down():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    policy = CancellingPolicy()
    runner = DeterministicRunner(
        session,
        context_manager=policy,
        ids=["ts_c", "cmp", "cr", "tf_c"],
        now=1000,
    )
    policy.runner = runner
    runner.schedule_compaction()

    await runner.run()

    assert len(policy.seen) == 1  # it was called…
    assert policy.hard_cancelled  # …and torn down where it hung
    assert runner.session.entries["cmp"].parts is None  # nothing was produced
    assert runner.session.entries["tf_c"].outcome == TurnOutcome.CANCELLED
    assert len(runner.session.conversations) == len(RICH_SESSION.conversations)


async def test_a_cancel_stops_the_drive_and_the_queued_message_is_not_answered():
    faux = _answering_provider()
    session = RICH_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=FakeContextManager(should=True, plan=fold_everything),
        ids=["ts_c", "cmp", "cr", "tf_c"],
        now=1000,
    )

    async with runner.run() as run:
        async for event in run:
            if event.type == "compaction_scheduled":
                run.cancel()

    assert faux.requests == []
    assert runner.busy()  # u4 still drives on the next run


async def test_the_run_after_a_cancelled_compaction_carries_no_interrupted_marker():
    faux = _answering_provider()
    session = RICH_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=FakeContextManager(should=[True, False], plan=fold_everything),
        ids=["ts_c", "cmp", "cr", "tf_c", "ts4", "a4", "tf4"],
        now=1000,
    )
    async with runner.run() as run:
        async for event in run:
            if event.type == "compaction_scheduled":
                run.cancel()

    await runner.run()

    # the compaction bracket projects as NOTHING — the user cancelled a
    # compaction, and the model must not be told their question was interrupted
    assert faux.requests[0].messages[-1] == LucaUserMessage(
        content=[TextBlock(text="What is X?")],
    )


async def test_a_parked_cancel_flushes_without_calling_the_policy():
    session = COMPACTION_CANCEL_PARKED_SESSION.model_copy(deep=True)
    policy = FakeContextManager(plan=fold_everything)
    runner = DeterministicRunner(
        session,
        context_manager=policy,
        ids=["tf_c"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    assert [event.type for event in events] == ["compaction_finished"]
    assert policy.seen == []
    assert policy.should_calls == 0
    assert runner.session.entries["tf_c"].outcome == TurnOutcome.CANCELLED


async def test_an_immediate_cancel_on_start_parks_the_compaction_flush():
    session = RICH_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(should=True, plan=fold_everything),
        ids=["ts_c", "cmp", "cr", "tf_c"],
        now=1000,
    )
    run = runner.start()  # start() opened a COMPACTION bracket

    run.cancel()
    result = await run

    assert main_conversation(runner.session).nodes[-4:] == [
        "ts_c",
        "cmp",
        "cr",
        "tf_c",
    ]
    assert result.outcome == TurnOutcome.CANCELLED


async def test_a_second_cancel_inside_a_compaction_bracket_raises():
    session = COMPACTION_SCHEDULED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(),
        ids=["cr"],
        now=1000,
    )

    runner.cancel()

    with pytest.raises(AlreadyCancellingError):
        runner.cancel()


async def test_run_result_after_a_cancelled_compaction():
    session = COMPACTION_CANCEL_PARKED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(),
        ids=["tf_c"],
        now=1000,
    )

    result = await runner.run()

    assert result == RunResult(
        status=ConversationStatus.BUSY,  # u4 is still queued
        outcome=TurnOutcome.CANCELLED,
        pending_approvals=[],
    )


@pytest.mark.parametrize(
    "outcome",
    [TurnOutcome.ERRORED, TurnOutcome.TIMED_OUT],
)
async def test_a_cancel_with_a_non_cancelled_outcome_still_stops_the_drive(outcome):
    # `cancel()` takes its outcome as an argument and only COMPLETED is
    # forbidden, so keying the drive-stops check off the outcome's VALUE would
    # let these fall through into the conversational turn
    faux = _answering_provider()
    session = COMPACTION_SCHEDULED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=FakeContextManager(plan=fold_everything),
        ids=["cr", "tf_c"],
        now=1000,
    )
    runner.cancel(outcome)

    await runner.run()

    assert faux.requests == []
    assert runner.session.entries["tf_c"].outcome == outcome


# ── F. crash, resume, suspend (G6) ────────────────────────────────────────────


async def test_a_scheduled_compaction_survives_a_reload_and_resumes_in_place():
    session = COMPACTION_SCHEDULED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=fold_and_keep_the_question),
        provider=_answering_provider(),
        ids=["tf_c", "c2", "ts4", "a4", "tf4"],
        now=1000,
    )

    assert runner.busy()
    await runner.run()

    # the SAME entry was reused — no second bracket, no new id
    assert runner.session.conversations["c1"].nodes.count("cmp") == 1
    assert runner.session.entries["cmp"].parts == SUMMARY


async def test_an_interrupted_compaction_keeps_its_original_started_at():
    session = COMPACTION_INTERRUPTED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=fold_and_keep_the_question),
        provider=_answering_provider(),
        ids=["tf_c", "c2", "ts4", "a4", "tf4"],
        now=1000,
    )

    assert runner.busy()  # the stale RUNNING self-healed at construction
    await runner.run()

    assert runner.session.entries["cmp"].started_at == 600  # not re-stamped
    assert runner.session.entries["cmp"].ended_at == 1000


async def test_a_closed_failed_bracket_is_never_retried():
    session = COMPACTION_FAILED_SESSION.model_copy(deep=True)
    policy = FakeContextManager(should=True, plan=fold_everything)
    runner = DeterministicRunner(
        session,
        context_manager=policy,
        ids=[],
        now=1000,
    )

    assert runner.idle()
    with pytest.raises(AgentError, match="Nothing to run"):
        await runner.run()

    assert policy.should_calls == 0


async def test_a_closed_completed_bracket_does_not_bury_a_queued_message():
    faux = _answering_provider()
    session = COMPACTION_BURIED_SESSION.model_copy(deep=True)
    policy = FakeContextManager(should=False)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=policy,
        ids=["ts4", "a4", "tf4"],
        now=1000,
    )

    assert runner.busy()
    await runner.run()

    assert len(faux.requests) == 1  # u4 was answered
    assert main_conversation(runner.session).nodes.count("cmp") == 1


async def test_a_committed_compaction_inside_a_counterfeit_bracket_is_not_re_run():
    # `[ts3, cmp, u4]` READS as an open compaction bracket. Resuming it would
    # re-call compact() over finished work and overwrite the record of what
    # that summary replaced — the one hazard that damages an audit trail.
    session = POST_COMPACTION_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["ts3", "cmp", "u4"]
    committed = session.entries["cmp"].model_copy(deep=True)
    policy = FakeContextManager(should=True, plan=fold_everything)
    runner = DeterministicRunner(
        session,
        provider=_answering_provider(),
        context_manager=policy,
        ids=["a4", "tf4"],
        now=1000,
    )

    await runner.run()

    assert policy.seen == []
    assert policy.should_calls == 0
    assert runner.session.entries["cmp"] == committed


async def test_schedule_compaction_raises_on_a_counterfeit_bracket():
    session = POST_COMPACTION_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["ts3", "cmp", "u4"]
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(),
        ids=[],
        now=1000,
    )

    with pytest.raises(AgentError, match="requires a closed turn"):
        runner.schedule_compaction()

    assert main_conversation(runner.session).nodes == ["ts3", "cmp", "u4"]


async def test_suspending_a_lazy_run_mid_compaction_leaves_the_bracket_open():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=fold_everything),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )
    runner.schedule_compaction()

    async with runner.run() as run:
        async for event in run:
            if event.type == "compaction_started":
                break

    assert "tf_c" not in runner.session.entries
    assert runner.busy()
    assert runner.ledger.open_compaction_entry(session.main_conversation_id) is not None


async def test_an_on_event_raise_during_started_leaves_the_bracket_open():
    def explode(event):
        if event.type == "compaction_started":
            raise RuntimeError("renderer crashed")

    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=fold_everything),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )
    runner.schedule_compaction()

    with pytest.raises(RuntimeError, match="renderer crashed"):
        await runner.run(on_event=explode)

    assert "tf_c" not in runner.session.entries
    assert runner.busy()


async def test_a_reloaded_compacted_session_drives_normally():
    faux = _answering_provider()
    session = POST_COMPACTION_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=FakeContextManager(should=False),
        ids=["ts4", "a4", "tf4"],
        now=1000,
    )

    await runner.run()

    assert faux.requests[0].messages == [
        LucaUserMessage(
            content=[
                TextBlock(text="The user added 1 and 2 and was answered 3."),
            ]
        ),
        LucaUserMessage(content=[TextBlock(text="What is X?")]),
    ]


# ── G. the trailing entry ─────────────────────────────────────────────────────


async def test_a_carried_trailing_user_message_keeps_driving():
    faux = _answering_provider()
    session = RICH_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=FakeContextManager(
            should=True,
            plan=fold_and_keep_the_question,
        ),
        ids=["ts_c", "cmp", "tf_c", "c2", "ts4", "a4", "tf4"],
        now=1000,
    )

    result = await runner.run()

    assert main_conversation(runner.session).nodes == [
        "cmp",
        "u4",
        "ts4",
        "a4",
        "tf4",
    ]
    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.COMPLETED,  # the TURN's close
        pending_approvals=[],
    )


async def test_a_folded_trailing_user_message_is_committed_and_the_question_is_lost():
    # the ONE silent failure the framework does not prevent — a policy may
    # legitimately fold the question into the summary. Asserted so it cannot
    # change by accident.
    faux = _answering_provider()
    session = RICH_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=FakeContextManager(should=True, plan=fold_everything),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    await runner.run()

    assert main_conversation(runner.session).nodes == ["cmp"]
    assert runner.idle()
    assert faux.requests == []
    assert "u4" in runner.session.entries
    assert "u4" in runner.session.entries["cmp"].compacted_nodes


async def test_a_trailing_turn_finish_gives_a_compaction_only_drive():
    faux = _answering_provider()
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=FakeContextManager(plan=fold_everything),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    result = await runner.run()

    assert faux.requests == []
    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.COMPLETED,
        pending_approvals=[],
    )


async def test_a_carried_failed_turn_finish_is_not_retried_after_the_transition():
    faux = _answering_provider()
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=FakeContextManager(plan=carry_the_failed_turn_finish),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    # The carried TurnFinish(ERRORED) is a CLOSED bracket, so the installed
    # conversation derives IDLE and the compaction-only drive stops there. A
    # failed turn is no longer retry-ready — nothing re-sends the request.
    assert faux.requests == []
    assert runner.idle()
    assert main_conversation(runner.session).nodes == ["cmp", "u2", "ts2", "tf2"]


async def test_a_carried_phantom_open_turn_is_committed_as_given():
    faux = _answering_provider()
    session = RICH_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=FakeContextManager(
            should=True,
            plan=carry_a_phantom_open_turn,
        ),
        ids=["ts_c", "cmp", "tf_c", "c2", "a4", "tf4"],
        now=1000,
    )

    await runner.run()

    # ts3 came over without its TurnFinish: the drive resumes a turn that
    # never happened. The hazard is the policy's — core commits it as given.
    assert main_conversation(runner.session).nodes == [
        "cmp",
        "ts3",
        "u4",
        "a4",
        "tf4",
    ]
    assert len(faux.requests) == 1


async def test_a_phantom_bracket_around_the_summary_is_driven_as_a_turn():
    # the plan carries `ts3` with the summary immediately after it, so
    # `[ts3, cmp, u4]` READS as an open compaction bracket. It must be driven
    # as the phantom turn it is, with the committed record left alone.
    faux = _answering_provider()
    session = RICH_SESSION.model_copy(deep=True)
    policy = FakeContextManager(
        should=True,
        plan=bracket_the_summary_with_a_carried_turn_start,
    )
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=policy,
        ids=["ts_c", "cmp", "tf_c", "c2", "a4", "tf4"],
        now=1000,
    )

    await runner.run()

    assert len(policy.seen) == 1  # compact() ran once, not twice
    assert runner.session.entries["cmp"].compacted_nodes is not None
    assert main_conversation(runner.session).nodes == [
        "ts3",
        "cmp",
        "u4",
        "a4",
        "tf4",
    ]
    assert len(faux.requests) == 1


# ── H. preconditions and API ─────────────────────────────────────────────────


def test_schedule_compaction_writes_the_bracket_and_the_entry():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(),
        ids=["ts_c", "cmp"],
        now=1000,
    )

    entry_id = runner.schedule_compaction()

    assert entry_id == "cmp"
    assert main_conversation(runner.session).nodes == [*RICH_IDLE_NODES, "ts_c", "cmp"]
    assert runner.session.entries["cmp"] == CompactionEntry(
        id="cmp",
        parent_id="ts_c",
        created_at=1000,
        source=CompactionSource.USER,
    )
    assert runner.busy()


def test_schedule_compaction_is_idempotent():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(),
        ids=["ts_c", "cmp"],
        now=1000,
    )
    runner.schedule_compaction()
    after_first = runner.session.model_copy(deep=True)

    assert runner.schedule_compaction() == "cmp"  # no ids drawn, nothing written
    assert runner.session == after_first


def test_schedule_compaction_rejects_an_open_conversational_turn():
    session = CLEARED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(),
        ids=[],
        now=1000,
    )

    with pytest.raises(AgentError, match="requires a closed turn"):
        runner.schedule_compaction()

    assert runner.session == CLEARED_SESSION


def test_schedule_compaction_rejects_an_approval_gate():
    session = GATED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(),
        ids=[],
        now=1000,
    )

    with pytest.raises(AgentError, match="status=blocked"):
        runner.schedule_compaction()


async def test_schedule_compaction_against_a_manager_that_cannot_compact_fails_on_the_drive():
    # A `ContextManager` always exists, so scheduling cannot pre-check that one
    # implements compaction: the bracket is written, and core's base `compact`
    # raises on the drive — ERRORED and propagating, as any USER failure does.
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["ts_c", "cmp", "tf_c"], now=1000)
    runner.schedule_compaction()

    with pytest.raises(NotImplementedError):
        await runner.run()

    assert runner.session.entries["tf_c"] == TurnFinish(
        id="tf_c",
        parent_id="cmp",
        created_at=1000,
        outcome=TurnOutcome.ERRORED,
        error="",
    )
    assert runner.idle()


def test_post_message_is_illegal_while_a_compaction_is_scheduled():
    session = COMPACTION_SCHEDULED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(),
        ids=[],
        now=1000,
    )

    with pytest.raises(AgentError, match="compaction scheduled or in flight"):
        runner.post_message("meanwhile…")


def test_post_message_is_illegal_while_a_compaction_is_in_flight():
    # The check reads the bracket SHAPE, never the status — an in-flight
    # compaction derives BUSY, and a status-based check would wrongly accept.
    session = COMPACTION_INTERRUPTED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(),
        ids=[],
        now=1000,
    )

    with pytest.raises(AgentError, match="compaction scheduled or in flight"):
        runner.post_message("meanwhile…")


async def test_post_message_is_legal_once_the_compaction_has_been_driven():
    session = COMPACTION_SCHEDULED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=fold_everything),
        provider=_answering_provider(),
        ids=["tf_c", "c2", "u5"],
        now=1000,
    )

    await runner.run()
    runner.post_message("meanwhile…")

    assert main_conversation(runner.session).nodes == ["cmp", "u5"]


async def test_start_opens_a_compaction_bracket_when_one_is_due():
    faux = _answering_provider()
    session = RICH_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=FakeContextManager(
            should=[True, False],
            plan=fold_and_keep_the_question,
        ),
        ids=["ts_c", "cmp", "tf_c", "c2", "ts4", "a4", "tf4"],
        now=1000,
    )

    run = runner.start()
    await run

    # no bare TurnStart was opened before the compaction bracket
    assert runner.session.conversations["c1"].nodes[-3:] == [
        "ts_c",
        "cmp",
        "tf_c",
    ]
    assert main_conversation(runner.session).nodes == [
        "cmp",
        "u4",
        "ts4",
        "a4",
        "tf4",
    ]


async def test_start_with_an_already_scheduled_compaction_opens_nothing_extra():
    session = COMPACTION_SCHEDULED_SESSION.model_copy(deep=True)
    policy = FakeContextManager(plan=fold_everything)
    runner = DeterministicRunner(
        session,
        context_manager=policy,
        ids=["tf_c", "c2"],
        now=1000,
    )

    run = runner.start()
    await run

    assert policy.should_calls == 0  # a scheduled compaction wins
    assert runner.session.conversations["c1"].nodes.count("ts_c") == 1


async def test_a_scheduled_compaction_wins_over_the_policy():
    session = COMPACTION_SCHEDULED_SESSION.model_copy(deep=True)
    policy = FakeContextManager(should=True, plan=fold_everything)
    runner = DeterministicRunner(
        session,
        context_manager=policy,
        ids=["tf_c", "c2"],
        now=1000,
    )

    await runner.run()

    assert policy.should_calls == 0
    assert runner.session.entries["cmp"].source == CompactionSource.USER
    assert len(policy.seen) == 1


async def test_should_compact_is_not_consulted_while_a_turn_is_open():
    faux = _answering_provider()
    session = CLEARED_SESSION.model_copy(deep=True)
    policy = FakeContextManager(should=True, plan=fold_everything)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=policy,
        tool_registry=None,
        ids=["a2", "tf"],
        now=1000,
    )

    await runner.run()

    assert policy.should_calls == 0
    assert policy.seen == []


async def test_should_compact_is_not_consulted_on_an_idle_session():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    policy = FakeContextManager(should=True, plan=fold_everything)
    runner = DeterministicRunner(
        session,
        context_manager=policy,
        ids=[],
        now=1000,
    )

    with pytest.raises(AgentError, match="Nothing to run"):
        await runner.run()

    assert policy.should_calls == 0


async def test_at_most_one_compaction_per_drive():
    faux = _answering_provider()
    session = RICH_SESSION.model_copy(deep=True)
    policy = FakeContextManager(should=True, plan=fold_and_keep_the_question)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=policy,
        ids=["ts_c", "cmp", "tf_c", "c2", "ts4", "a4", "tf4"],
        now=1000,
    )

    await runner.run()

    # `should_compact` is still True, but the step sits outside the loop
    assert len(policy.seen) == 1
    assert len(runner.session.conversations) == len(RICH_IDLE_SESSION.conversations) + 1


async def test_a_should_compact_that_raises_propagates_from_run():
    class Exploding(FakeContextManager):
        def should_compact(self, session, conversation_id):
            raise RuntimeError("bad threshold arithmetic")

    session = RICH_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=Exploding(),
        ids=[],
        now=1000,
    )

    with pytest.raises(RuntimeError, match="bad threshold arithmetic"):
        await runner.run()


async def test_a_should_compact_that_raises_from_start_leaves_the_runner_usable():
    class Exploding(FakeContextManager):
        def should_compact(self, session, conversation_id):
            raise RuntimeError("bad threshold arithmetic")

    faux = _answering_provider()
    session = RICH_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=Exploding(),
        ids=["ts4", "a4", "tf4"],
        now=1000,
    )

    with pytest.raises(RuntimeError, match="bad threshold arithmetic"):
        runner.start()

    # the one-run guard was released — the runner is not wedged
    runner.context_manager = ContextManager()  # the default: accounts, never compacts
    await runner.run()
    assert len(faux.requests) == 1


def test_two_runners_with_different_context_managers_are_not_equal():
    session = RICH_SESSION.model_copy(deep=True)

    assert DeterministicRunner(
        session,
        context_manager=FakeContextManager(should=True),
    ) != DeterministicRunner(
        session,
        context_manager=FakeContextManager(should=False),
    )


def test_two_runners_with_equivalent_context_managers_are_equal():
    session = RICH_SESSION.model_copy(deep=True)

    assert DeterministicRunner(
        session,
        context_manager=FakeContextManager(should=True),
    ) == DeterministicRunner(
        session,
        context_manager=FakeContextManager(should=True),
    )


# ── I. usage, llm_config, context ────────────────────────────────────────────


async def test_no_usage_is_recorded_when_the_policy_returns_none():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=None),
        ids=["ts_c", "cmp", "tf_c"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    assert "cmp" not in runner.session.usages["c1"]


async def test_no_usage_is_recorded_when_the_policy_raises():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(raises=ValueError("x")),
        ids=["ts_c", "cmp", "tf_c"],
        now=1000,
    )
    runner.schedule_compaction()

    with pytest.raises(ValueError, match="x"):
        await runner.run()

    assert "cmp" not in runner.session.usages["c1"]


async def test_the_context_tokens_are_recalculated_when_the_parts_land():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=fold_everything),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )
    runner.schedule_compaction()

    assert runner.session.entries["cmp"].context_tokens == 0
    await runner.run()

    assert runner.session.entries["cmp"].context_tokens == 7  # 28 // 4


async def test_middleware_has_the_final_say_on_the_summarys_context_tokens():
    class Overrides:
        def before_entry_written(self, session, conversation_id, entry):
            if isinstance(entry, CompactionEntry):
                entry.context_tokens = 99
            return entry

    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=fold_everything),
        middleware=[Overrides()],
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    assert runner.session.entries["cmp"].context_tokens == 99


# ── K. RunResult ──────────────────────────────────────────────────────────────


async def test_run_result_after_a_compaction_then_a_turn_reports_the_turn():
    session = RICH_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        provider=_answering_provider(),
        context_manager=FakeContextManager(
            should=True,
            plan=fold_and_keep_the_question,
        ),
        ids=["ts_c", "cmp", "tf_c", "c2", "ts4", "a4", "tf4"],
        now=1000,
    )

    result = await runner.run()

    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.COMPLETED,
        pending_approvals=[],
    )


async def test_a_deadline_that_does_not_expire_lets_the_compaction_commit():
    # the wall-clock tier is converted the same way the LLM step converts it;
    # a policy that answers inside it commits normally
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    session.session_config.runtime_config = RuntimeConfig(
        client_completion_timeout_in_ms=30_000,
    )
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=fold_everything),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    result = await runner.run()

    assert main_conversation(runner.session).nodes == ["cmp"]
    assert result.outcome == TurnOutcome.COMPLETED


async def test_the_compaction_events_fire_in_streaming_mode_too():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=fold_everything),
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )
    runner.schedule_compaction()

    async with runner.run(streaming=True) as run:
        events = [event async for event in run]

    assert [event.type for event in events] == [
        "compaction_scheduled",
        "compaction_started",
        "compaction_finished",
    ]


def test_schedule_compaction_rejects_a_parked_cancel():
    session = CANCEL_PARKED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(),
        ids=[],
        now=1000,
    )

    with pytest.raises(AgentError, match="status=cancelling"):
        runner.schedule_compaction()

    assert runner.session == CANCEL_PARKED_SESSION


# ── helpers (module-level, never in a test body) ─────────────────────────────


def _answering_provider() -> FauxProvider:
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("X is 42.")], finish_reason="stop"),
        ]
    )
    return faux
