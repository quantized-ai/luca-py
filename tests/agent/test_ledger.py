"""Declarative matrix for the SessionLedger — the entry-derived queries the
runner's status cache is re-derived from, plus the uniform append/put
bookkeeping and the tool-spec normalization every write door performs. Known
session data in, exact answer out; no loop, no provider.

The execution vocabulary under test (over `status` + `approval_status`):
PENDING splits into UNDECIDED (`approval_status` None or PENDING — what the
runner offers to the policy), AWAITING (`approval_status` PENDING — the
policy explicitly deferred; drives AWAITING_APPROVAL), and READY
(`approval_status` ALLOWED — dispatchable). RUNNING executions are orphans at
drive start. Approval state is read from `approval_status`, never
reconstructed from the `approval_decisions` audit log.

The tool-spec doors under test: `append`, `put_entry`,
`transition_conversation` (over `updates`, `created` AND `closing`) and
`refresh_entry` each file a `ToolExecution`'s `tool_spec` in
`session.tool_specs` under its content id and stamp that id on the execution.
`prune()` is deliberately not one of them. Session literals carrying an
execution go through `scenarios.make_session()` — a hand-written
`AgentSession` would have to file every spec itself, which is never what the
test is about.
"""

import pytest

from luca.agent.core.exceptions import AgentError
from luca.agent.core.ledger import SessionLedger
from luca.agent.core.models import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    CancelRequested,
    CompactionEntry,
    CompactionSource,
    Conversation,
    ConversationRuntimeStatus,
    ConversationStatus,
    ExecutionResult,
    ExecutionStatus,
    LLMConfig,
    PrunedEntry,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    Usage,
    UserMessage,
)
from tests.agent.scenarios import (
    MODEL,
    conversation,
    main_conversation,
    make_session,
    spec,
)

PENDING_1000 = ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)
ALLOW_1000 = ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)
DENY_1000 = ApprovalDecision(decision=ApprovalOption.DENY, created_at=1000)

# The two specs the tool-spec doors are exercised with, plus the content ids
# they hash to. `REVISED_ADD_SPEC` differs from `ADD_SPEC` only in its
# description — enough for a different id, so a re-stamp is visible.
ADD_SPEC = spec("add")
ADD_SPEC_ID = ADD_SPEC.spec_id()
REVISED_ADD_SPEC = spec("add", description="Add two numbers, revised.")
REVISED_ADD_SPEC_ID = REVISED_ADD_SPEC.spec_id()
MULTIPLY_SPEC = spec("multiply")
MULTIPLY_SPEC_ID = MULTIPLY_SPEC.spec_id()


# ── open_turn_index ────────────────────────────────────────────────────────────


def test_open_turn_index_is_none_on_empty_path():
    session = AgentSession(
        id="s",
        conversations={"c1": conversation("c1", [], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_turn_index("c1") is None


def test_open_turn_index_is_none_after_turn_finish():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts": TurnStart(id="ts", created_at=1),
            "tf": TurnFinish(id="tf", created_at=2),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "ts", "tf"],
                created_at=0,
                updated_at=2,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_turn_index("c1") is None


def test_open_turn_index_finds_unclosed_turn_start():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts1": TurnStart(id="ts1", created_at=1),
            "tf1": TurnFinish(id="tf1", created_at=2),
            "u2": UserMessage(id="u2", created_at=3, parts=[TextContent(text="go")]),
            "ts2": TurnStart(id="ts2", created_at=4),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "ts1", "tf1", "u2", "ts2"],
                created_at=0,
                updated_at=4,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_turn_index("c1") == 4


# ── derive_status ──────────────────────────────────────────────────────────────


def test_derive_status_idle_on_empty_session():
    session = AgentSession(
        id="s",
        conversations={"c1": conversation("c1", [], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    assert session.get_conversation_status("c1").status == ConversationStatus.IDLE


def test_derive_status_pending_with_trailing_user_message():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    assert session.get_conversation_status("c1").status == ConversationStatus.BUSY


def test_derive_status_pending_with_open_turn_and_no_executions():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts": TurnStart(id="ts", created_at=1),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "ts"],
                created_at=0,
                updated_at=1,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    assert session.get_conversation_status("c1").status == ConversationStatus.BUSY


def test_derive_status_awaiting_when_approval_status_is_pending():
    session = make_session(
        id="s",
        entries={
            "ts": TurnStart(id="ts", created_at=1),
            "te1": ToolExecution(
                id="te1",
                conversation_id="c1",
                created_at=2,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=spec("add"),
                status=ExecutionStatus.PENDING,
                approval_status=ApprovalStatus.PENDING,
                approval_decisions=[PENDING_1000],
            ),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["ts", "te1"],
                created_at=0,
                updated_at=2,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    assert session.get_conversation_status("c1").status == ConversationStatus.BLOCKED


def test_derive_status_pending_when_execution_was_never_processed():
    # crash mid-decide: approval_status None — the strategy was never asked,
    # so the right move is a plain run() (which asks it), NOT the gate.
    session = make_session(
        id="s",
        entries={
            "ts": TurnStart(id="ts", created_at=1),
            "te1": ToolExecution(
                id="te1",
                conversation_id="c1",
                created_at=2,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=spec("add"),
                status=ExecutionStatus.PENDING,
                approval_status=None,
                approval_decisions=[],
            ),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["ts", "te1"],
                created_at=0,
                updated_at=2,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    assert session.get_conversation_status("c1").status == ConversationStatus.BUSY


def test_derive_status_pending_when_execution_is_allowed_but_unrun():
    session = make_session(
        id="s",
        entries={
            "ts": TurnStart(id="ts", created_at=1),
            "te1": ToolExecution(
                id="te1",
                conversation_id="c1",
                created_at=2,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=spec("add"),
                status=ExecutionStatus.PENDING,
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[PENDING_1000, ALLOW_1000],
            ),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["ts", "te1"],
                created_at=0,
                updated_at=2,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    assert session.get_conversation_status("c1").status == ConversationStatus.BUSY


def test_derive_status_pending_with_orphaned_running_execution():
    # a persisted RUNNING execution means: call run() — the next drive
    # recovers it to INTERRUPTED and continues the turn.
    session = make_session(
        id="s",
        entries={
            "ts": TurnStart(id="ts", created_at=1),
            "te1": ToolExecution(
                id="te1",
                conversation_id="c1",
                created_at=2,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=spec("add"),
                status=ExecutionStatus.RUNNING,
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[ALLOW_1000],
                started_at=3,
            ),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["ts", "te1"],
                created_at=0,
                updated_at=3,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    assert session.get_conversation_status("c1").status == ConversationStatus.BUSY


def test_derive_status_idle_after_closed_turn():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts": TurnStart(id="ts", created_at=1),
            "tf": TurnFinish(id="tf", created_at=2),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "ts", "tf"],
                created_at=0,
                updated_at=2,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    assert session.get_conversation_status("c1").status == ConversationStatus.IDLE


def test_derive_status_cancelling_beats_awaiting_approval():
    # an unconsumed CancelRequested wins over the approval gate: the next
    # drive is a flush, not a re-ask.
    session = make_session(
        id="s",
        entries={
            "ts": TurnStart(id="ts", created_at=1),
            "te1": ToolExecution(
                id="te1",
                conversation_id="c1",
                created_at=2,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=spec("add"),
                status=ExecutionStatus.PENDING,
                approval_status=ApprovalStatus.PENDING,
                approval_decisions=[PENDING_1000],
            ),
            "cr": CancelRequested(id="cr", created_at=3),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["ts", "te1", "cr"],
                created_at=0,
                updated_at=3,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    assert session.get_conversation_status("c1").status == ConversationStatus.CANCELLING


def test_derive_status_idle_after_timed_out_turn():
    # A closed bracket is IDLE whatever its outcome. A failed turn is no longer
    # retry-ready: recovering means posting a new message, not silently
    # re-sending the identical request.
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts": TurnStart(id="ts", created_at=1),
            "tf": TurnFinish(
                id="tf",
                created_at=2,
                outcome=TurnOutcome.TIMED_OUT,
                error="client timeout",
            ),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "ts", "tf"],
                created_at=0,
                updated_at=2,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    assert session.get_conversation_status("c1").status == ConversationStatus.IDLE


def test_derive_status_idle_after_errored_turn():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts": TurnStart(id="ts", created_at=1),
            "tf": TurnFinish(
                id="tf",
                created_at=2,
                outcome=TurnOutcome.ERRORED,
                error="boom",
            ),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "ts", "tf"],
                created_at=0,
                updated_at=2,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    assert session.get_conversation_status("c1").status == ConversationStatus.IDLE


def test_derive_status_idle_after_cancelled_turn():
    # CANCELLED is a deliberate close — the consumed CancelRequested inside
    # the bracket no longer derives CANCELLING.
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts": TurnStart(id="ts", created_at=1),
            "cr": CancelRequested(id="cr", created_at=2),
            "tf": TurnFinish(
                id="tf",
                created_at=3,
                outcome=TurnOutcome.CANCELLED,
            ),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "ts", "cr", "tf"],
                created_at=0,
                updated_at=3,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    assert session.get_conversation_status("c1").status == ConversationStatus.IDLE


# ── the execution-subset matrix ────────────────────────────────────────────────
#
# One session, five executions covering every lifecycle/approval combination
# the runner asks about: terminal (skipped by all pending queries), undecided
# (never processed), awaiting (policy deferred), ready (allowed, unrun), and
# orphaned RUNNING.

MATRIX_SESSION = make_session(
    id="s_matrix",
    entries={
        "ts": TurnStart(id="ts", created_at=1),
        "te_done": ToolExecution(
            id="te_done",
            conversation_id="c1",
            created_at=2,
            tool_call_id="tc0",
            raw_tool_call=ToolCall(id="tc0", name="add", arguments={"a": 1, "b": 2}),
            tool_spec=spec("add"),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="3")]),
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[ALLOW_1000],
            started_at=3,
            ended_at=3,
        ),
        "te_undecided": ToolExecution(
            id="te_undecided",
            conversation_id="c1",
            created_at=4,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add"),
            tool_spec=spec("add"),
            status=ExecutionStatus.PENDING,
            approval_status=None,
            approval_decisions=[],
        ),
        "te_awaiting": ToolExecution(
            id="te_awaiting",
            conversation_id="c1",
            created_at=5,
            tool_call_id="tc2",
            raw_tool_call=ToolCall(id="tc2", name="multiply"),
            tool_spec=spec("multiply"),
            status=ExecutionStatus.PENDING,
            approval_status=ApprovalStatus.PENDING,
            approval_decisions=[PENDING_1000],
        ),
        "te_ready": ToolExecution(
            id="te_ready",
            conversation_id="c1",
            created_at=6,
            tool_call_id="tc3",
            raw_tool_call=ToolCall(id="tc3", name="subtract"),
            tool_spec=spec("subtract"),
            status=ExecutionStatus.PENDING,
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[ALLOW_1000],
        ),
        "te_running": ToolExecution(
            id="te_running",
            conversation_id="c1",
            created_at=7,
            tool_call_id="tc4",
            raw_tool_call=ToolCall(id="tc4", name="add"),
            tool_spec=spec("add"),
            status=ExecutionStatus.RUNNING,
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[ALLOW_1000],
            started_at=8,
        ),
    },
    conversations={
        "c1": conversation(
            "c1",
            ["ts", "te_done", "te_undecided", "te_awaiting", "te_ready", "te_running"],
            created_at=0,
            updated_at=8,
        )
    },
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)


def test_execution_subsets_partition_by_status_and_approval():
    session = MATRIX_SESSION.model_copy(deep=True)
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_turn_executions("c1") == [
        session.entries["te_done"],
        session.entries["te_undecided"],
        session.entries["te_awaiting"],
        session.entries["te_ready"],
        session.entries["te_running"],
    ]
    assert ledger.open_turn_pending_executions("c1") == [
        session.entries["te_undecided"],
        session.entries["te_awaiting"],
        session.entries["te_ready"],
    ]
    assert ledger.open_turn_undecided_executions("c1") == [
        session.entries["te_undecided"],
        session.entries["te_awaiting"],
    ]
    assert ledger.open_turn_awaiting_executions("c1") == [
        session.entries["te_awaiting"],
    ]
    assert ledger.open_turn_ready_executions("c1") == [
        session.entries["te_ready"],
    ]
    assert ledger.open_turn_running_executions("c1") == [
        session.entries["te_running"],
    ]
    assert ledger.has_awaiting_approval("c1") is True


def test_execution_subsets_are_empty_without_open_turn():
    session = AgentSession(
        id="s",
        entries={
            "ts": TurnStart(id="ts", created_at=1),
            "tf": TurnFinish(id="tf", created_at=2),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["ts", "tf"],
                created_at=0,
                updated_at=2,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_turn_executions("c1") == []
    assert ledger.open_turn_pending_executions("c1") == []
    assert ledger.open_turn_undecided_executions("c1") == []
    assert ledger.open_turn_awaiting_executions("c1") == []
    assert ledger.open_turn_ready_executions("c1") == []
    assert ledger.open_turn_running_executions("c1") == []
    assert ledger.has_awaiting_approval("c1") is False


def test_approval_state_is_read_from_approval_status_not_the_log():
    # the audit log trails a PENDING decision, but approval_status says
    # ALLOWED — the status field wins (middleware may author such state).
    session = make_session(
        id="s",
        entries={
            "ts": TurnStart(id="ts", created_at=1),
            "te1": ToolExecution(
                id="te1",
                conversation_id="c1",
                created_at=2,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=spec("add"),
                status=ExecutionStatus.PENDING,
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[PENDING_1000],
            ),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["ts", "te1"],
                created_at=0,
                updated_at=2,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_turn_ready_executions("c1") == [session.entries["te1"]]
    assert ledger.open_turn_awaiting_executions("c1") == []
    assert ledger.open_turn_undecided_executions("c1") == []


# ── open_turn_cancel_requested ─────────────────────────────────────────────────


def test_open_turn_cancel_requested_finds_the_unconsumed_entry():
    session = AgentSession(
        id="s",
        entries={
            "ts": TurnStart(id="ts", created_at=1),
            "cr": CancelRequested(id="cr", created_at=2, error="user hit ESC"),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["ts", "cr"],
                created_at=0,
                updated_at=2,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_turn_cancel_requested("c1") == CancelRequested(
        id="cr",
        created_at=2,
        outcome=TurnOutcome.CANCELLED,
        error="user hit ESC",
    )


def test_open_turn_cancel_requested_ignores_consumed_instances():
    # the prior turn's CancelRequested sits before its TurnFinish — consumed;
    # the new open turn has none.
    session = AgentSession(
        id="s",
        entries={
            "ts1": TurnStart(id="ts1", created_at=1),
            "cr": CancelRequested(id="cr", created_at=2),
            "tf1": TurnFinish(
                id="tf1",
                created_at=3,
                outcome=TurnOutcome.CANCELLED,
            ),
            "u2": UserMessage(id="u2", created_at=4, parts=[TextContent(text="go")]),
            "ts2": TurnStart(id="ts2", created_at=5),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["ts1", "cr", "tf1", "u2", "ts2"],
                created_at=0,
                updated_at=5,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_turn_cancel_requested("c1") is None


def test_open_turn_cancel_requested_is_none_without_open_turn():
    session = AgentSession(
        id="s",
        entries={
            "ts": TurnStart(id="ts", created_at=1),
            "cr": CancelRequested(id="cr", created_at=2),
            "tf": TurnFinish(
                id="tf",
                created_at=3,
                outcome=TurnOutcome.CANCELLED,
            ),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["ts", "cr", "tf"],
                created_at=0,
                updated_at=3,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_turn_cancel_requested("c1") is None


# ── record_usage: the single write door onto AgentSession.usages ──────────────


def test_record_usage_builds_the_record_and_indexes_it_by_conversation():
    session = AgentSession(
        id="s",
        entries={
            "a1": AssistantMessage(
                id="a1",
                created_at=1,
                parts=[TextContent(text="hi")],
                llm_config=MODEL,
                stop_reason="stop",
            ),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["a1"],
                created_at=0,
                updated_at=1,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    record = ledger.record_usage(
        "c1",
        "a1",
        input=10,
        output=5,
        cache_read=2,
        total_tokens=15,
    )

    assert record == Usage(
        conversation_id="c1",
        entry_id="a1",
        input=10,
        output=5,
        cache_read=2,
        cache_write=0,
        total_tokens=15,
    )
    assert session.usages == {"c1": {"a1": record}}


def test_record_usage_replaces_the_pair_record_on_re_record():
    session = AgentSession(
        id="s",
        entries={
            "a1": AssistantMessage(
                id="a1",
                created_at=1,
                parts=[TextContent(text="hi")],
                llm_config=MODEL,
                stop_reason="stop",
            ),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["a1"],
                created_at=0,
                updated_at=1,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")
    ledger.record_usage("c1", "a1", input=10, total_tokens=10)

    ledger.record_usage("c1", "a1", input=20, total_tokens=20)

    assert session.usages == {
        "c1": {
            "a1": Usage(
                conversation_id="c1",
                entry_id="a1",
                input=20,
                total_tokens=20,
            ),
        },
    }


def test_record_usage_rejects_a_missing_entry():
    session = AgentSession(
        id="s",
        conversations={
            "c1": conversation(
                "c1",
                [],
                created_at=0,
                updated_at=0,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    with pytest.raises(AgentError, match="no such entry"):
        ledger.record_usage("c1", "ghost", input=10)


def test_record_usage_rejects_an_entry_outside_the_conversation_path():
    session = AgentSession(
        id="s",
        entries={
            "a1": AssistantMessage(
                id="a1",
                created_at=1,
                parts=[TextContent(text="hi")],
                llm_config=MODEL,
                stop_reason="stop",
            ),
        },
        conversations={
            "c1": conversation(
                "c1",
                [],
                created_at=0,
                updated_at=0,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    with pytest.raises(AgentError, match="not on conversation"):
        ledger.record_usage("c1", "a1", input=10)


# ── append / put bookkeeping ───────────────────────────────────────────────────


def test_append_links_parent_extends_path_and_stamps_updated_at():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "ts")

    entry = ledger.append(
        "c1",
        lambda entry_id, parent_id, ts: TurnStart(
            id=entry_id,
            parent_id=parent_id,
            created_at=ts,
        ),
    )

    assert entry == TurnStart(id="ts", parent_id="u1", created_at=1000)
    assert session.entries["ts"] == entry
    assert main_conversation(session).nodes == ["u1", "ts"]
    assert main_conversation(session).updated_at == 1000


def test_put_entry_stores_the_replacement_and_touches_conversation():
    session = make_session(
        id="s",
        entries={
            "te1": ToolExecution(
                id="te1",
                conversation_id="c1",
                created_at=0,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=spec("add"),
                status=ExecutionStatus.PENDING,
            ),
        },
        conversations={"c1": conversation("c1", ["te1"], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")
    replacement = session.entries["te1"].model_copy(
        update={
            "status": ExecutionStatus.REJECTED,
            "approval_status": ApprovalStatus.REJECTED,
            "approval_decisions": [DENY_1000],
            "ended_at": 1000,
            "updated_at": 1000,
        },
    )

    stored = ledger.put_entry("c1", replacement)

    assert stored is replacement
    assert session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        created_at=0,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add"),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.REJECTED,
        approval_status=ApprovalStatus.REJECTED,
        approval_decisions=[DENY_1000],
        ended_at=1000,
        updated_at=1000,
    )
    assert main_conversation(session).updated_at == 1000


# ── refresh_entry: the derived-field door ─────────────────────────────────────


def test_refresh_entry_stores_the_replacement_without_touching_the_conversation():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(
                id="u1",
                created_at=0,
                parts=[TextContent(text="hi")],
                context_tokens=3,
            ),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")
    replacement = session.entries["u1"].model_copy(update={"context_tokens": 12})

    stored = ledger.refresh_entry(replacement)

    assert stored is replacement
    assert session.entries["u1"] == UserMessage(
        id="u1",
        created_at=0,
        parts=[TextContent(text="hi")],
        context_tokens=12,
    )
    # re-deriving a stored estimate is not a mutation of the conversation:
    # unlike put_entry, this door leaves `updated_at` exactly where it was.
    assert main_conversation(session) == Conversation(
        id="c1",
        nodes=["u1"],
        created_at=0,
        updated_at=0,
    )


def test_refresh_entry_refuses_an_uncommitted_entry():
    session = AgentSession(
        id="s",
        entries={},
        conversations={"c1": conversation("c1", [], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    with pytest.raises(AgentError, match="Cannot refresh entry None"):
        ledger.refresh_entry(CompactionEntry(source=CompactionSource.USER))

    assert session.entries == {}


def test_refresh_entry_refuses_to_create_an_unknown_entry():
    session = AgentSession(
        id="s",
        entries={},
        conversations={"c1": conversation("c1", [], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    with pytest.raises(AgentError, match="Cannot refresh entry 'nope'"):
        ledger.refresh_entry(
            CompactionEntry(
                id="nope",
                created_at=0,
                source=CompactionSource.USER,
            ),
        )

    assert session.entries == {}


# ── tool specs: the write doors that normalize them ───────────────────────────
#
# `append`, `put_entry`, `transition_conversation` and `refresh_entry` each
# file an execution's `tool_spec` in `session.tool_specs` under its
# content-derived id and stamp that id on the execution. `prune` is
# deliberately not one of them.


def test_append_files_the_tool_spec_and_stamps_the_reference():
    session = AgentSession(
        id="s",
        entries={"ts": TurnStart(id="ts", created_at=0)},
        conversations={"c1": conversation("c1", ["ts"], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "te1")

    appended = ledger.append(
        "c1",
        lambda entry_id, parent_id, ts: ToolExecution(
            id=entry_id,
            parent_id=parent_id,
            created_at=ts,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add"),
            tool_spec=spec("add"),
        ),
    )

    # `conversation_id` stays None: the LEDGER never stamps it. It is part of
    # the identity set the RUNNER owns, alongside id / parent_id / created_at.
    assert appended == ToolExecution(
        id="te1",
        parent_id="ts",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add"),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
    )
    assert session.tool_specs == {ADD_SPEC_ID: ADD_SPEC}


def test_append_leaves_an_unresolved_execution_with_no_spec_reference():
    # a NOT_FOUND birth resolved to no tool at all: nothing to file, nothing
    # to stamp, and the store stays empty.
    session = AgentSession(
        id="s",
        entries={"ts": TurnStart(id="ts", created_at=0)},
        conversations={"c1": conversation("c1", ["ts"], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "te1")

    appended = ledger.append(
        "c1",
        lambda entry_id, parent_id, ts: ToolExecution(
            id=entry_id,
            parent_id=parent_id,
            created_at=ts,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="ghost"),
            status=ExecutionStatus.NOT_FOUND,
        ),
    )

    # `conversation_id` stays None: the LEDGER never stamps it. It is part of
    # the identity set the RUNNER owns, alongside id / parent_id / created_at.
    assert appended == ToolExecution(
        id="te1",
        parent_id="ts",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="ghost"),
        status=ExecutionStatus.NOT_FOUND,
    )
    assert session.tool_specs == {}


def test_the_same_spec_written_twice_is_stored_once():
    # normalization: equal content hashes to the same id, so the second write
    # finds the row already filed and adds nothing — and the execution is
    # pointed at that stored instance, so `tool_specs[id] is entry.tool_spec`
    # holds in memory exactly as it does after a reload.
    session = make_session(
        id="s",
        entries={
            "te1": ToolExecution(
                id="te1",
                conversation_id="c1",
                created_at=0,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=spec("add"),
            ),
        },
        conversations={"c1": conversation("c1", ["te1"], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "te2")

    appended = ledger.append(
        "c1",
        lambda entry_id, parent_id, ts: ToolExecution(
            id=entry_id,
            parent_id=parent_id,
            created_at=ts,
            tool_call_id="tc2",
            raw_tool_call=ToolCall(id="tc2", name="add"),
            tool_spec=spec("add"),
        ),
    )

    assert appended == ToolExecution(
        id="te2",
        parent_id="te1",
        created_at=1000,
        tool_call_id="tc2",
        raw_tool_call=ToolCall(id="tc2", name="add"),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
    )
    assert session.tool_specs == {ADD_SPEC_ID: ADD_SPEC}
    assert appended.tool_spec is session.tool_specs[ADD_SPEC_ID]
    assert session.entries["te1"].tool_spec is session.tool_specs[ADD_SPEC_ID]


def test_put_entry_re_stamps_a_spec_replaced_between_writes():
    # the id is recomputed on every write and never short-circuited on an
    # already-set one: middleware rewrote the spec, and a skipped recompute
    # would leave the execution pointing at the previous version.
    session = make_session(
        id="s",
        entries={
            "te1": ToolExecution(
                id="te1",
                conversation_id="c1",
                created_at=0,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=spec("add"),
            ),
        },
        conversations={"c1": conversation("c1", ["te1"], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")
    replacement = session.entries["te1"].model_copy(
        update={"tool_spec": REVISED_ADD_SPEC, "updated_at": 1000},
    )

    ledger.put_entry("c1", replacement)

    assert session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        created_at=0,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add"),
        tool_spec=REVISED_ADD_SPEC,
        tool_spec_id=REVISED_ADD_SPEC_ID,
        updated_at=1000,
    )
    # append-only: the superseded spec stays filed, so every older execution
    # referencing it still resolves.
    assert session.tool_specs == {
        ADD_SPEC_ID: ADD_SPEC,
        REVISED_ADD_SPEC_ID: REVISED_ADD_SPEC,
    }


def test_put_entry_drops_the_reference_when_the_spec_is_removed():
    # the mirror of the re-stamp: a dangling id is the one shape a session
    # refuses to load with, so clearing the spec must clear the id too.
    session = make_session(
        id="s",
        entries={
            "te1": ToolExecution(
                id="te1",
                conversation_id="c1",
                created_at=0,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=spec("add"),
            ),
        },
        conversations={"c1": conversation("c1", ["te1"], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")
    replacement = session.entries["te1"].model_copy(update={"tool_spec": None})

    ledger.put_entry("c1", replacement)

    assert session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        created_at=0,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add"),
    )
    assert session.tool_specs == {ADD_SPEC_ID: ADD_SPEC}


def test_refresh_entry_files_a_spec_that_arrives_late():
    # the fourth door: a recalculation pass hands back an execution whose spec
    # was never filed, and it is filed and stamped here like anywhere else.
    session = AgentSession(
        id="s",
        entries={
            "te1": ToolExecution(
                id="te1",
                conversation_id="c1",
                created_at=0,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
            ),
        },
        conversations={"c1": conversation("c1", ["te1"], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")
    replacement = session.entries["te1"].model_copy(
        update={"tool_spec": ADD_SPEC, "context_tokens": 5},
    )

    ledger.refresh_entry(replacement)

    assert session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        created_at=0,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add"),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        context_tokens=5,
    )
    assert session.tool_specs == {ADD_SPEC_ID: ADD_SPEC}


# ── prune: the single path-replacement door ────────────────────────────────────


def test_prune_replaces_the_node_in_place_and_keeps_the_original_entry():
    session = make_session(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "te1": ToolExecution(
                id="te1",
                conversation_id="c1",
                parent_id="u1",
                created_at=1,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=spec("add"),
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(content=[TextContent(text="3")]),
                started_at=1,
                ended_at=1,
            ),
            "a2": AssistantMessage(
                id="a2",
                parent_id="te1",
                created_at=2,
                parts=[TextContent(text="done")],
                llm_config=MODEL,
                stop_reason="stop",
            ),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "te1", "a2"],
                created_at=0,
                updated_at=2,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "p1")

    pruned = ledger.prune(
        "c1",
        "te1",
        lambda entry_id, parent_id, ts: PrunedEntry(
            id=entry_id,
            parent_id=parent_id,
            created_at=ts,
            pruned_entry_type="tool_execution",
            pruned_entry_id="te1",
            content=[TextContent(text="[pruned]")],
            context_tokens=2,
        ),
    )

    # the pruned entry took the original's position AND its parent link
    assert pruned == PrunedEntry(
        id="p1",
        parent_id="u1",
        created_at=1000,
        pruned_entry_type="tool_execution",
        pruned_entry_id="te1",
        content=[TextContent(text="[pruned]")],
        context_tokens=2,
    )
    assert session.entries["p1"] == pruned
    assert main_conversation(session).nodes == ["u1", "p1", "a2"]
    assert main_conversation(session).updated_at == 1000
    # the original is untouched and remains in the store
    assert session.entries["te1"].status == ExecutionStatus.COMPLETED


def test_prune_is_not_a_tool_spec_door():
    # the door that looks like one and deliberately is not: prune only ever
    # writes a `PrunedEntry`, so the spec store keeps exactly the rows the
    # precondition filed and the pruned execution keeps its own reference.
    session = make_session(
        id="s",
        entries={
            "te1": ToolExecution(
                id="te1",
                conversation_id="c1",
                created_at=0,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=spec("add"),
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(content=[TextContent(text="3")]),
                started_at=0,
                ended_at=0,
            ),
        },
        conversations={"c1": conversation("c1", ["te1"], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "p1")

    ledger.prune(
        "c1",
        "te1",
        lambda entry_id, parent_id, ts: PrunedEntry(
            id=entry_id,
            parent_id=parent_id,
            created_at=ts,
            pruned_entry_type="tool_execution",
            pruned_entry_id="te1",
            content=[TextContent(text="[pruned]")],
        ),
    )

    assert session.tool_specs == {ADD_SPEC_ID: ADD_SPEC}
    assert session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        created_at=0,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add"),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="3")]),
        started_at=0,
        ended_at=0,
    )


def test_prune_rejects_a_missing_entry():
    session = AgentSession(
        id="s",
        conversations={"c1": conversation("c1", [], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "p1")

    with pytest.raises(AgentError, match="no such entry"):
        ledger.prune("c1", "ghost", lambda entry_id, parent_id, ts: None)


def test_prune_rejects_an_entry_outside_the_active_path():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
        },
        conversations={"c1": conversation("c1", [], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "p1")

    with pytest.raises(AgentError, match="not on conversation"):
        ledger.prune("c1", "u1", lambda entry_id, parent_id, ts: None)


def test_prune_rejects_a_replacement_referencing_a_different_entry():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "p1")

    with pytest.raises(AgentError, match="replacing"):
        ledger.prune(
            "c1",
            "u1",
            lambda entry_id, parent_id, ts: PrunedEntry(
                id=entry_id,
                parent_id=parent_id,
                created_at=ts,
                pruned_entry_type="user",
                pruned_entry_id="other",
                content=[TextContent(text="[pruned]")],
            ),
        )


def test_prune_rejects_a_mismatched_pruned_entry_type():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "p1")

    with pytest.raises(AgentError, match="pruned_entry_type"):
        ledger.prune(
            "c1",
            "u1",
            lambda entry_id, parent_id, ts: PrunedEntry(
                id=entry_id,
                parent_id=parent_id,
                created_at=ts,
                pruned_entry_type="tool_execution",
                pruned_entry_id="u1",
                content=[TextContent(text="[pruned]")],
            ),
        )


def test_prune_rejects_a_nonterminal_execution():
    session = make_session(
        id="s",
        entries={
            "te1": ToolExecution(
                id="te1",
                conversation_id="c1",
                created_at=0,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=spec("add"),
                status=ExecutionStatus.RUNNING,
                started_at=0,
            ),
        },
        conversations={"c1": conversation("c1", ["te1"], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "p1")

    with pytest.raises(AgentError, match="nonterminal"):
        ledger.prune(
            "c1",
            "te1",
            lambda entry_id, parent_id, ts: PrunedEntry(
                id=entry_id,
                parent_id=parent_id,
                created_at=ts,
                pruned_entry_type="tool_execution",
                pruned_entry_id="te1",
                content=[TextContent(text="[pruned]")],
            ),
        )


# ── ConversationRuntimeStatus.get_runtime_status_from_agent_session ────────────────
#
# Known AgentSession assembled from entry literals → exact ConversationRuntimeStatus.
# No runner, no provider — pure derivation from entries.

_LLM = LLMConfig(model="test-model", provider="faux")
_AM = AssistantMessage(
    id="a1",
    created_at=1,
    parts=[TextContent(text="ok")],
    llm_config=_LLM,
    stop_reason="stop",
)


def test_runtime_status_empty_session():
    session = AgentSession(
        id="s",
        conversations={"c1": conversation("c1", [], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=_LLM),
    )
    assert session.get_conversation_status("c1") == ConversationRuntimeStatus(
        status=ConversationStatus.IDLE, turn_count=0, step_count=0
    )


def test_runtime_status_open_turn_no_steps():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts": TurnStart(id="ts", created_at=1),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "ts"],
                created_at=0,
                updated_at=1,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=_LLM),
    )
    assert session.get_conversation_status("c1") == ConversationRuntimeStatus(
        status=ConversationStatus.BUSY, turn_count=1, step_count=0
    )


def test_runtime_status_open_turn_one_step():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts": TurnStart(id="ts", created_at=1),
            "a1": _AM,
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "ts", "a1"],
                created_at=0,
                updated_at=1,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=_LLM),
    )
    assert session.get_conversation_status("c1") == ConversationRuntimeStatus(
        status=ConversationStatus.BUSY, turn_count=1, step_count=1
    )


def test_runtime_status_open_turn_two_steps():
    a2 = AssistantMessage(
        id="a2",
        created_at=2,
        parts=[TextContent(text="also ok")],
        llm_config=_LLM,
        stop_reason="stop",
    )
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts": TurnStart(id="ts", created_at=1),
            "a1": _AM,
            "a2": a2,
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "ts", "a1", "a2"],
                created_at=0,
                updated_at=2,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=_LLM),
    )
    assert session.get_conversation_status("c1") == ConversationRuntimeStatus(
        status=ConversationStatus.BUSY, turn_count=1, step_count=2
    )


def test_runtime_status_closed_turn_no_open_turn():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts": TurnStart(id="ts", created_at=1),
            "a1": _AM,
            "tf": TurnFinish(id="tf", created_at=2, outcome=TurnOutcome.COMPLETED),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "ts", "a1", "tf"],
                created_at=0,
                updated_at=2,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=_LLM),
    )
    # Closed turn: step_count=0 (no open turn), turn_count=1
    assert session.get_conversation_status("c1") == ConversationRuntimeStatus(
        status=ConversationStatus.IDLE, turn_count=1, step_count=0
    )


def test_runtime_status_second_open_turn():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="first")]),
            "ts1": TurnStart(id="ts1", created_at=1),
            "a1": _AM,
            "tf1": TurnFinish(id="tf1", created_at=2, outcome=TurnOutcome.COMPLETED),
            "u2": UserMessage(id="u2", created_at=3, parts=[TextContent(text="second")]),
            "ts2": TurnStart(id="ts2", created_at=4),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "ts1", "a1", "tf1", "u2", "ts2"],
                created_at=0,
                updated_at=4,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=_LLM),
    )
    # Second open turn with no steps yet; turn_count=2 (two TurnStarts)
    assert session.get_conversation_status("c1") == ConversationRuntimeStatus(
        status=ConversationStatus.BUSY, turn_count=2, step_count=0
    )


def test_runtime_status_second_open_turn_with_steps():
    a2 = AssistantMessage(
        id="a2",
        created_at=5,
        parts=[TextContent(text="reply")],
        llm_config=_LLM,
        stop_reason="stop",
    )
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="first")]),
            "ts1": TurnStart(id="ts1", created_at=1),
            "a1": _AM,
            "tf1": TurnFinish(id="tf1", created_at=2, outcome=TurnOutcome.COMPLETED),
            "u2": UserMessage(id="u2", created_at=3, parts=[TextContent(text="second")]),
            "ts2": TurnStart(id="ts2", created_at=4),
            "a2": a2,
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "ts1", "a1", "tf1", "u2", "ts2", "a2"],
                created_at=0,
                updated_at=5,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=_LLM),
    )
    # step_count counts only AssistantMessages in the CURRENT (second) open turn
    assert session.get_conversation_status("c1") == ConversationRuntimeStatus(
        status=ConversationStatus.BUSY, turn_count=2, step_count=1
    )


def test_put_entry_updates_a_compaction_entry_as_readily_as_an_execution():
    session = AgentSession(
        id="s",
        entries={
            "cmp": CompactionEntry(
                id="cmp",
                created_at=0,
                source=CompactionSource.USER,
            ),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["cmp"],
                created_at=0,
                updated_at=0,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")
    replacement = session.entries["cmp"].model_copy(update={"started_at": 1000})

    ledger.put_entry("c1", replacement)

    assert session.entries["cmp"] == CompactionEntry(
        id="cmp",
        created_at=0,
        source=CompactionSource.USER,
        started_at=1000,
    )
    assert main_conversation(session).nodes == ["cmp"]
    assert main_conversation(session).updated_at == 1000


def test_put_entry_refuses_an_uncommitted_entry():
    session = AgentSession(
        id="s",
        entries={},
        conversations={
            "c1": conversation(
                "c1",
                [],
                created_at=0,
                updated_at=0,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    with pytest.raises(AgentError, match="Cannot store an uncommitted entry"):
        ledger.put_entry("c1", CompactionEntry(source=CompactionSource.USER))

    assert session.entries == {}


def test_put_entry_refuses_to_create_an_unknown_entry():
    session = AgentSession(
        id="s",
        entries={},
        conversations={
            "c1": conversation(
                "c1",
                [],
                created_at=0,
                updated_at=0,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    with pytest.raises(AgentError, match="Cannot update entry 'nope': no such entry"):
        ledger.put_entry(
            "c1",
            CompactionEntry(
                id="nope",
                created_at=0,
                source=CompactionSource.USER,
            ),
        )

    assert session.entries == {}


# ── transition_conversation: the atomic swap ──────────────────────────────────

# The pre-transition shape every transition test starts from: one answered
# turn on `c1`, a queued question, and an open compaction bracket whose entry
# is about to receive its content.
TRANSITION_SESSION = AgentSession(
    id="s_transition",
    entries={
        "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="hi")]),
        "ts1": TurnStart(id="ts1", parent_id="u1", created_at=500),
        "a1": AssistantMessage(
            id="a1",
            parent_id="ts1",
            created_at=500,
            parts=[TextContent(text="hello")],
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
    conversations={
        "c1": conversation(
            "c1",
            ["u1", "ts1", "a1", "tf1", "u2", "ts_c", "cmp"],
            created_at=500,
            updated_at=600,
        )
    },
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)

SUMMARIZED = CompactionEntry(
    id="cmp",
    parent_id="ts_c",
    created_at=600,
    source=CompactionSource.POLICY,
    parts=[TextContent(text="the user said hi")],
    compacted_nodes=["u1", "ts1", "a1", "tf1"],
    started_at=600,
    ended_at=1000,
    context_tokens=4,
)

CLOSING = TurnFinish(id="tf_c", parent_id="cmp", created_at=1000)


def test_transition_archives_the_outgoing_path_and_installs_the_new_one():
    session = TRANSITION_SESSION.model_copy(deep=True)
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "c2")

    installed = ledger.transition_conversation(
        "c1",
        updates=[SUMMARIZED.model_copy(deep=True)],
        created=[],
        closing=CLOSING.model_copy(deep=True),
        nodes=["cmp", "u2"],
        ts=1000,
    )

    # The name moved; the outgoing conversation did NOT move — it stays a
    # first-class row, now reachable backwards from its successor.
    assert installed is main_conversation(session)
    assert session.main_conversation_id == "c2"
    assert session.conversations == {
        "c1": Conversation(
            id="c1",
            nodes=["u1", "ts1", "a1", "tf1", "u2", "ts_c", "cmp", "tf_c"],
            created_at=500,
            updated_at=1000,
        ),
        "c2": Conversation(
            id="c2",
            nodes=["cmp", "u2"],
            created_at=1000,
            updated_at=1000,
            previous_conversation_id="c1",
        ),
    }
    assert session.entries["cmp"] == SUMMARIZED
    assert session.entries["tf_c"] == CLOSING


def test_the_outgoing_conversation_is_never_the_installed_one():
    session = TRANSITION_SESSION.model_copy(deep=True)
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "c2")

    ledger.transition_conversation(
        "c1",
        updates=[SUMMARIZED.model_copy(deep=True)],
        created=[],
        closing=CLOSING.model_copy(deep=True),
        nodes=["cmp"],
        ts=1000,
    )

    assert session.conversations["c1"] is not main_conversation(session)


def test_transition_stores_created_entries():
    session = TRANSITION_SESSION.model_copy(deep=True)
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "c2")
    framing = UserMessage(
        id="new1",
        created_at=1000,
        parts=[TextContent(text="framing")],
    )
    execution = ToolExecution(
        id="new2",
        conversation_id="c1",
        parent_id="new1",
        created_at=1000,
        tool_call_id="tcX",
        raw_tool_call=ToolCall(id="tcX", name="add"),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="3")]),
        started_at=1000,
        ended_at=1000,
    )

    ledger.transition_conversation(
        "c1",
        updates=[SUMMARIZED.model_copy(deep=True)],
        created=[framing, execution],
        closing=CLOSING.model_copy(deep=True),
        nodes=["new1", "new2", "cmp"],
        ts=1000,
    )

    assert session.entries["new1"] == framing
    assert session.entries["new2"] == execution


def test_transition_without_a_closing_marker_archives_the_path_as_it_stands():
    session = TRANSITION_SESSION.model_copy(deep=True)
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "c2")

    ledger.transition_conversation(
        "c1",
        updates=[],
        created=[],
        closing=None,
        nodes=["u2"],
        ts=1000,
    )

    assert session.conversations["c1"].nodes == [
        "u1",
        "ts1",
        "a1",
        "tf1",
        "u2",
        "ts_c",
        "cmp",
    ]


def test_the_new_conversation_derives_idle_from_a_carried_completed_turn():
    session = TRANSITION_SESSION.model_copy(deep=True)
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "c2")

    installed = ledger.transition_conversation(
        "c1",
        updates=[SUMMARIZED.model_copy(deep=True)],
        created=[],
        closing=CLOSING.model_copy(deep=True),
        nodes=["cmp", "ts1", "a1", "tf1"],
        ts=1000,
    )

    assert session.get_conversation_status(installed.id).status == ConversationStatus.IDLE


def test_the_new_conversation_derives_busy_from_a_carried_user_message():
    session = TRANSITION_SESSION.model_copy(deep=True)
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "c2")

    installed = ledger.transition_conversation(
        "c1",
        updates=[SUMMARIZED.model_copy(deep=True)],
        created=[],
        closing=CLOSING.model_copy(deep=True),
        nodes=["cmp", "u2"],
        ts=1000,
    )

    assert session.get_conversation_status(installed.id).status == ConversationStatus.BUSY


def test_the_new_conversation_derives_idle_from_a_carried_failed_turn():
    session = TRANSITION_SESSION.model_copy(deep=True)
    session.entries["tf1"] = session.entries["tf1"].model_copy(
        update={"outcome": TurnOutcome.ERRORED, "error": "boom"},
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "c2")

    installed = ledger.transition_conversation(
        "c1",
        updates=[SUMMARIZED.model_copy(deep=True)],
        created=[],
        closing=CLOSING.model_copy(deep=True),
        nodes=["cmp", "ts1", "a1", "tf1"],
        ts=1000,
    )

    assert session.get_conversation_status(installed.id).status == ConversationStatus.IDLE


def test_the_new_conversation_derives_idle_when_the_summary_is_the_leaf():
    session = TRANSITION_SESSION.model_copy(deep=True)
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "c2")

    installed = ledger.transition_conversation(
        "c1",
        updates=[SUMMARIZED.model_copy(deep=True)],
        created=[],
        closing=CLOSING.model_copy(deep=True),
        nodes=["cmp"],
        ts=1000,
    )

    assert session.get_conversation_status(installed.id).status == ConversationStatus.IDLE


def test_an_unknown_update_id_raises_before_anything_is_written():
    session = TRANSITION_SESSION.model_copy(deep=True)
    before = session.model_copy(deep=True)
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "c2")

    with pytest.raises(AgentError, match="Cannot update entry 'ghost'"):
        ledger.transition_conversation(
            "c1",
            updates=[
                SUMMARIZED.model_copy(deep=True, update={"id": "ghost"}),
            ],
            created=[],
            closing=CLOSING.model_copy(deep=True),
            nodes=["cmp"],
            ts=1000,
        )

    assert session == before


def test_a_created_id_that_already_exists_raises_before_anything_is_written():
    session = TRANSITION_SESSION.model_copy(deep=True)
    before = session.model_copy(deep=True)
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "c2")

    with pytest.raises(AgentError, match="Cannot create entry 'u1'"):
        ledger.transition_conversation(
            "c1",
            updates=[],
            created=[
                UserMessage(
                    id="u1",
                    created_at=1000,
                    parts=[TextContent(text="clash")],
                ),
            ],
            closing=None,
            nodes=["cmp"],
            ts=1000,
        )

    assert session == before


def test_a_created_entry_with_no_id_raises_before_anything_is_written():
    session = TRANSITION_SESSION.model_copy(deep=True)
    before = session.model_copy(deep=True)
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "c2")

    with pytest.raises(AgentError, match="Cannot create an entry with no id"):
        ledger.transition_conversation(
            "c1",
            updates=[],
            created=[UserMessage(parts=[TextContent(text="template")])],
            closing=None,
            nodes=["cmp"],
            ts=1000,
        )

    assert session == before


# ── transition_conversation: the third tool-spec write door ───────────────────
#
# No ordinary tool call travels through this door, only a compaction plan that
# carries or creates a `ToolExecution` — and it can arrive on ANY of the three
# entry inputs, since `updates` and `created` are public `list[AnyEntry]`.

# The transition shape with an answered tool call already on the path.
TRANSITION_TOOL_SESSION = make_session(
    id="s_transition_tool",
    entries={
        "te1": ToolExecution(
            id="te1",
            conversation_id="c1",
            created_at=500,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add"),
            tool_spec=spec("add"),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="3")]),
            started_at=500,
            ended_at=500,
        ),
        "ts_c": TurnStart(id="ts_c", parent_id="te1", created_at=600),
        "cmp": CompactionEntry(
            id="cmp",
            parent_id="ts_c",
            created_at=600,
            source=CompactionSource.POLICY,
            started_at=600,
        ),
    },
    conversations={
        "c1": conversation(
            "c1",
            ["te1", "ts_c", "cmp"],
            created_at=500,
            updated_at=600,
        )
    },
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)


def test_transition_files_the_spec_of_an_updated_execution():
    session = TRANSITION_TOOL_SESSION.model_copy(deep=True)
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "c2")
    rewritten = session.entries["te1"].model_copy(
        update={"tool_spec": REVISED_ADD_SPEC},
    )

    ledger.transition_conversation(
        "c1",
        updates=[rewritten],
        created=[],
        closing=CLOSING.model_copy(deep=True),
        nodes=["cmp", "te1"],
        ts=1000,
    )

    assert session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        created_at=500,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add"),
        tool_spec=REVISED_ADD_SPEC,
        tool_spec_id=REVISED_ADD_SPEC_ID,
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="3")]),
        started_at=500,
        ended_at=500,
    )
    assert session.tool_specs == {
        ADD_SPEC_ID: ADD_SPEC,
        REVISED_ADD_SPEC_ID: REVISED_ADD_SPEC,
    }


def test_transition_files_the_spec_of_a_created_execution():
    session = TRANSITION_TOOL_SESSION.model_copy(deep=True)
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "c2")
    minted = ToolExecution(
        id="te2",
        conversation_id="c1",
        created_at=1000,
        tool_call_id="tc2",
        raw_tool_call=ToolCall(id="tc2", name="multiply"),
        tool_spec=MULTIPLY_SPEC,
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="12")]),
        started_at=1000,
        ended_at=1000,
    )

    ledger.transition_conversation(
        "c1",
        updates=[],
        created=[minted],
        closing=CLOSING.model_copy(deep=True),
        nodes=["cmp", "te2"],
        ts=1000,
    )

    assert session.entries["te2"] == ToolExecution(
        id="te2",
        conversation_id="c1",
        created_at=1000,
        tool_call_id="tc2",
        raw_tool_call=ToolCall(id="tc2", name="multiply"),
        tool_spec=MULTIPLY_SPEC,
        tool_spec_id=MULTIPLY_SPEC_ID,
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="12")]),
        started_at=1000,
        ended_at=1000,
    )
    assert session.tool_specs == {
        ADD_SPEC_ID: ADD_SPEC,
        MULTIPLY_SPEC_ID: MULTIPLY_SPEC,
    }


def test_transition_files_the_spec_of_a_closing_execution():
    # `closing` is typed `AnyEntry | None`, so the helper runs over it too —
    # an execution handed in as the outgoing path's last node is filed exactly
    # like one on `updates` or `created`.
    session = TRANSITION_TOOL_SESSION.model_copy(deep=True)
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "c2")
    closing = ToolExecution(
        id="te2",
        conversation_id="c1",
        created_at=1000,
        tool_call_id="tc2",
        raw_tool_call=ToolCall(id="tc2", name="multiply"),
        tool_spec=MULTIPLY_SPEC,
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="12")]),
        started_at=1000,
        ended_at=1000,
    )

    ledger.transition_conversation(
        "c1",
        updates=[],
        created=[],
        closing=closing,
        nodes=["cmp"],
        ts=1000,
    )

    assert session.entries["te2"] == ToolExecution(
        id="te2",
        conversation_id="c1",
        created_at=1000,
        tool_call_id="tc2",
        raw_tool_call=ToolCall(id="tc2", name="multiply"),
        tool_spec=MULTIPLY_SPEC,
        tool_spec_id=MULTIPLY_SPEC_ID,
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="12")]),
        started_at=1000,
        ended_at=1000,
    )
    assert session.tool_specs == {
        ADD_SPEC_ID: ADD_SPEC,
        MULTIPLY_SPEC_ID: MULTIPLY_SPEC,
    }
    assert session.conversations["c1"].nodes == ["te1", "ts_c", "cmp", "te2"]


# ── open_compaction_entry: is there a compaction to RESUME? ───────────────────

# One store, many paths: a scheduled compaction, a committed one, the markers
# that open and close their brackets, and an ordinary turn to contrast with.
RESUME_SESSION = AgentSession(
    id="s_resume",
    entries={
        "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="hi")]),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
        "a1": AssistantMessage(
            id="a1",
            parent_id="ts",
            created_at=500,
            parts=[TextContent(text="hello")],
            llm_config=MODEL,
            stop_reason="stop",
        ),
        "ts_c": TurnStart(id="ts_c", parent_id="u1", created_at=600),
        "cmp": CompactionEntry(
            id="cmp",
            parent_id="ts_c",
            created_at=600,
            source=CompactionSource.USER,
            started_at=600,
        ),
        "cmp_done": CompactionEntry(
            id="cmp_done",
            parent_id="ts_c",
            created_at=600,
            source=CompactionSource.USER,
            parts=[TextContent(text="the story so far")],
            compacted_nodes=["u1"],
            started_at=600,
            ended_at=700,
        ),
        "cr": CancelRequested(id="cr", parent_id="cmp", created_at=700),
        "tf_c": TurnFinish(id="tf_c", parent_id="cmp", created_at=700),
    },
    conversations={
        "c1": conversation(
            "c1",
            [],
            created_at=500,
            updated_at=700,
        )
    },
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)


def test_no_compaction_to_resume_on_an_empty_path():
    session = RESUME_SESSION.model_copy(deep=True)
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_compaction_entry("c1") is None


def test_no_compaction_to_resume_inside_a_bare_open_conversational_turn():
    session = RESUME_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["u1", "ts"]
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_compaction_entry("c1") is None


def test_no_compaction_to_resume_inside_an_open_conversational_turn():
    session = RESUME_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["u1", "ts", "a1"]
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_compaction_entry("c1") is None


def test_an_open_compaction_bracket_offers_its_entry():
    session = RESUME_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["u1", "ts_c", "cmp"]
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_compaction_entry("c1") is session.entries["cmp"]


def test_a_parked_cancel_inside_the_bracket_still_offers_the_entry():
    session = RESUME_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["u1", "ts_c", "cmp", "cr"]
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_compaction_entry("c1") is session.entries["cmp"]


def test_a_closed_compaction_bracket_offers_nothing():
    session = RESUME_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["u1", "ts_c", "cmp", "tf_c"]
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_compaction_entry("c1") is None


def test_an_open_bracket_around_a_committed_entry_offers_nothing():
    # G6: a plan can counterfeit the SHAPE of an interrupted compaction; it
    # cannot counterfeit a committed entry's content. Resuming this would
    # re-run compact() over finished work and overwrite `compacted_nodes`.
    session = RESUME_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["u1", "ts_c", "cmp_done"]
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_compaction_entry("c1") is None


# ── derive_status: a closed compaction bracket is transparent ─────────────────

# One store, many paths. `cmp` is the compaction entry, `ts_c`/`ts_c2` its
# openers, and the `tf_c_*` markers its four possible closes.
STATUS_MATRIX_SESSION = AgentSession(
    id="s_matrix",
    entries={
        "u3": UserMessage(
            id="u3",
            created_at=500,
            parts=[TextContent(text="what is X?")],
        ),
        "ts2": TurnStart(id="ts2", parent_id="u3", created_at=500),
        "tf2_ok": TurnFinish(id="tf2_ok", parent_id="ts2", created_at=500),
        "tf2_err": TurnFinish(
            id="tf2_err",
            parent_id="ts2",
            created_at=500,
            outcome=TurnOutcome.ERRORED,
            error="boom",
        ),
        "ts_c": TurnStart(id="ts_c", created_at=600),
        "cmp": CompactionEntry(
            id="cmp",
            parent_id="ts_c",
            created_at=600,
            source=CompactionSource.POLICY,
        ),
        "cmp_started": CompactionEntry(
            id="cmp_started",
            parent_id="ts_c",
            created_at=600,
            source=CompactionSource.POLICY,
            started_at=600,
        ),
        "cr": CancelRequested(id="cr", parent_id="cmp", created_at=700),
        "tf_c_ok": TurnFinish(id="tf_c_ok", parent_id="cmp", created_at=700),
        "tf_c_cancelled": TurnFinish(
            id="tf_c_cancelled",
            parent_id="cmp",
            created_at=700,
            outcome=TurnOutcome.CANCELLED,
        ),
        "tf_c_err": TurnFinish(
            id="tf_c_err",
            parent_id="cmp",
            created_at=700,
            outcome=TurnOutcome.ERRORED,
            error="the policy raised",
        ),
        "tf_c_timeout": TurnFinish(
            id="tf_c_timeout",
            parent_id="cmp",
            created_at=700,
            outcome=TurnOutcome.TIMED_OUT,
            error="compaction exceeded 0.05s",
        ),
        "ts_c2": TurnStart(id="ts_c2", created_at=800),
        "cmp2": CompactionEntry(
            id="cmp2",
            parent_id="ts_c2",
            created_at=800,
            source=CompactionSource.USER,
        ),
        "tf_c2": TurnFinish(id="tf_c2", parent_id="cmp2", created_at=800),
        "u5": UserMessage(
            id="u5",
            created_at=900,
            parts=[TextContent(text="still there?")],
        ),
    },
    conversations={
        "c1": conversation(
            "c1",
            [],
            created_at=500,
            updated_at=900,
        )
    },
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)


def test_a_completed_compaction_does_not_bury_a_queued_question():
    session = STATUS_MATRIX_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["u3", "ts_c", "cmp", "tf_c_ok"]
    assert session.get_conversation_status("c1").status == ConversationStatus.BUSY


def test_a_cancelled_compaction_does_not_drop_the_queued_question():
    # the user cancelled the COMPACTION, not their question
    session = STATUS_MATRIX_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["u3", "ts_c", "cmp", "tf_c_cancelled"]
    assert session.get_conversation_status("c1").status == ConversationStatus.BUSY


def test_a_failed_compaction_over_a_finished_turn_is_idle_not_retry_ready():
    # without the skip rule tf_c(ERRORED) reads as a retry-ready turn and a
    # `while not runner.idle()` loop opens a fresh bracket every drive
    session = STATUS_MATRIX_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["tf2_ok", "ts_c", "cmp", "tf_c_err"]
    assert session.get_conversation_status("c1").status == ConversationStatus.IDLE


def test_a_timed_out_compaction_over_a_finished_turn_is_idle():
    session = STATUS_MATRIX_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["tf2_ok", "ts_c", "cmp", "tf_c_timeout"]
    assert session.get_conversation_status("c1").status == ConversationStatus.IDLE


def test_a_completed_compaction_over_a_failed_turn_derives_idle():
    session = STATUS_MATRIX_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["tf2_err", "ts_c", "cmp", "tf_c_ok"]
    assert session.get_conversation_status("c1").status == ConversationStatus.IDLE


def test_a_compaction_on_an_otherwise_empty_path_is_idle():
    session = STATUS_MATRIX_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["ts_c", "cmp", "tf_c_ok"]
    assert session.get_conversation_status("c1").status == ConversationStatus.IDLE


def test_two_stacked_closed_compaction_brackets_are_both_skipped():
    session = STATUS_MATRIX_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = [
        "u3",
        "ts_c",
        "cmp",
        "tf_c_ok",
        "ts_c2",
        "cmp2",
        "tf_c2",
    ]
    assert session.get_conversation_status("c1").status == ConversationStatus.BUSY


def test_a_scheduled_compaction_derives_pending():
    session = STATUS_MATRIX_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["u3", "ts_c", "cmp"]
    assert session.get_conversation_status("c1").status == ConversationStatus.BUSY


def test_an_interrupted_compaction_derives_pending():
    session = STATUS_MATRIX_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["u3", "ts_c", "cmp_started"]
    assert session.get_conversation_status("c1").status == ConversationStatus.BUSY


def test_an_unconsumed_cancel_inside_a_compaction_bracket_derives_cancelling():
    session = STATUS_MATRIX_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["u3", "ts_c", "cmp", "cr"]
    assert session.get_conversation_status("c1").status == ConversationStatus.CANCELLING


def test_a_node_after_the_bracket_ends_the_skip():
    session = STATUS_MATRIX_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["ts_c", "cmp", "tf_c_ok", "u5"]
    assert session.get_conversation_status("c1").status == ConversationStatus.BUSY


def test_a_closed_conversational_bracket_is_never_skipped():
    # The regression guard: only a TurnStart whose next node is the compaction
    # entry opens a compaction bracket. The conversational bracket therefore
    # stays the leaf and derives IDLE from its own close — if it were skipped,
    # `u3` would resurface and derive BUSY.
    session = STATUS_MATRIX_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["u3", "ts2", "tf2_err"]
    assert session.get_conversation_status("c1").status == ConversationStatus.IDLE


def test_a_carried_turn_finish_with_no_opener_stops_the_skip():
    # A policy carried tf2_err without its TurnStart: it is NOT a compaction
    # bracket, so the skip stops there and the ordinary closed-turn rules
    # apply to it rather than swallowing the whole path.
    session = STATUS_MATRIX_SESSION.model_copy(deep=True)
    main_conversation(session).nodes = ["cmp", "u3", "tf2_err"]
    assert session.get_conversation_status("c1").status == ConversationStatus.IDLE


# ── derive_status: the unresolved-children branch, in precedence order ────────
#
# 1. advanceable executions → BUSY   2. a child that can advance → BUSY
# 3. a gated execution → BLOCKED     4. unseen material → BUSY   5. BLOCKED
#
# Rule 3 ABOVE rule 4 is the load-bearing ordering: a gated wake-round call
# plus unrelated material must derive BLOCKED — the next run() can only
# re-park at the gate, and BUSY would hot-loop every poll-for-BLOCKED
# consumer.


def _spawn_declaring_spec():
    return spec(
        "spawn_subagent",
        output_schema={"type": "object", "properties": {"is_subagent_spawn": {"type": "boolean"}}},
    )


def _orchestrating_session(**overrides) -> AgentSession:
    """`c1` parked mid-orchestration on one unresolved child whose own turn is
    gated (the child derives BLOCKED), nothing else in flight."""
    from luca.agent.core.models import ChildConversation

    call = ToolCall(id="tc1", name="spawn_subagent", arguments={"task_id": "t1"})
    entries = {
        "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
        "a1": AssistantMessage(
            id="a1",
            parent_id="ts",
            created_at=500,
            parts=[call],
            llm_config=MODEL,
            stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1",
            conversation_id="c1",
            parent_id="a1",
            created_at=500,
            tool_call_id="tc1",
            raw_tool_call=call,
            tool_spec=_spawn_declaring_spec(),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(
                content=[TextContent(text="spawned")],
                structured_content={"is_subagent_spawn": True, "task_id": "t1"},
            ),
            started_at=500,
            ended_at=500,
        ),
        "ch1": ChildConversation(
            id="ch1",
            parent_id="te1",
            created_at=500,
            conversation_id="c2",
            tool_execution_id="te1",
        ),
        "u_seed": UserMessage(id="u_seed", created_at=500, parts=[TextContent(text="child work")]),
        "ts_c": TurnStart(id="ts_c", parent_id="u_seed", created_at=500),
        "te_c": ToolExecution(
            id="te_c",
            conversation_id="c2",
            parent_id="ts_c",
            created_at=500,
            tool_call_id="tc_c",
            raw_tool_call=ToolCall(id="tc_c", name="write"),
            tool_spec=spec("write"),
            status=ExecutionStatus.PENDING,
            approval_status=ApprovalStatus.PENDING,
            approval_decisions=[PENDING_1000],
        ),
    }
    entries.update(overrides.pop("entries", {}))
    fields = {
        "id": "s_children_status",
        "entries": entries,
        "conversations": {
            "c1": conversation("c1", overrides.pop("nodes", ["u1", "ts", "a1", "te1", "ch1"])),
            "c2": conversation("c2", ["u_seed", "ts_c", "te_c"], depth=1),
        },
        "main_conversation_id": "c1",
        "session_config": SessionConfig(llm_config=MODEL),
    }
    fields.update(overrides)
    return make_session(**fields)


def test_derive_status_blocked_when_every_child_is_blocked_and_nothing_is_unseen():
    session = _orchestrating_session()

    assert session.get_conversation_status("c1").status == ConversationStatus.BLOCKED


def test_derive_status_busy_when_a_posted_message_awaits_the_model():
    session = _orchestrating_session(
        entries={"um": UserMessage(id="um", parent_id="ch1", created_at=600, parts=[TextContent(text="steer")])},
        nodes=["u1", "ts", "a1", "te1", "ch1", "um"],
    )

    # the children cannot advance, but the CONVERSATION can: the next run()
    # calls the model with the message
    assert session.get_conversation_status("c1").status == ConversationStatus.BUSY


def test_derive_status_busy_when_a_resolution_awaits_the_model():
    from luca.agent.core.models import ChildConversation

    resolved = ChildConversation(
        id="ch1",
        parent_id="te1",
        created_at=500,
        conversation_id="c2",
        tool_execution_id="te1",
        execution_result=ExecutionResult(content=[TextContent(text="done")]),
        result_execution_id="ter1",
    )
    second = ChildConversation(
        id="ch2",
        parent_id="ch1",
        created_at=500,
        conversation_id="c2",
        tool_execution_id="te1",
    )
    session = _orchestrating_session(
        entries={
            "ch1": resolved,
            "ch2": second,
            "ter1": ToolExecution(
                id="ter1",
                conversation_id="c1",
                parent_id="ch2",
                created_at=600,
                tool_call_id="rtc1",
                raw_tool_call=ToolCall(id="rtc1", name="create_conversation_result"),
                tool_spec=spec("create_conversation_result", is_private=True),
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(content=[TextContent(text="done")]),
                started_at=600,
                ended_at=600,
            ),
        },
        nodes=["u1", "ts", "a1", "te1", "ch1", "ch2", "ter1"],
    )

    # ch2 is still unresolved and its child is BLOCKED — but ch1's result
    # execution after the last assistant is unseen material
    assert session.get_conversation_status("c1").status == ConversationStatus.BUSY


def test_derive_status_a_gate_outranks_material():
    session = _orchestrating_session(
        entries={
            "um": UserMessage(id="um", parent_id="ch1", created_at=600, parts=[TextContent(text="steer")]),
            "te2": ToolExecution(
                id="te2",
                conversation_id="c1",
                parent_id="um",
                created_at=700,
                tool_call_id="tc2",
                raw_tool_call=ToolCall(id="tc2", name="write"),
                tool_spec=spec("write"),
                status=ExecutionStatus.PENDING,
                approval_status=ApprovalStatus.PENDING,
                approval_decisions=[PENDING_1000],
            ),
        },
        nodes=["u1", "ts", "a1", "te1", "ch1", "um", "te2"],
    )

    # material exists, but the next run() can only re-park at the gate — the
    # application must act first
    assert session.get_conversation_status("c1").status == ConversationStatus.BLOCKED


def test_derive_status_an_advanceable_execution_wins_over_everything():
    session = _orchestrating_session(
        entries={
            "te2": ToolExecution(
                id="te2",
                conversation_id="c1",
                parent_id="ch1",
                created_at=700,
                tool_call_id="tc2",
                raw_tool_call=ToolCall(id="tc2", name="write"),
                tool_spec=spec("write"),
                status=ExecutionStatus.RECEIVED,
            ),
        },
        nodes=["u1", "ts", "a1", "te1", "ch1", "te2"],
    )

    assert session.get_conversation_status("c1").status == ConversationStatus.BUSY


def test_derive_status_a_cancelling_child_outranks_a_gate():
    # rule 2 above rule 3: the instant after a stop_subagent lands, the child
    # is CANCELLING — the parent's drive can still advance it (wait it out and
    # resolve the link), so a gated parent execution does not make the
    # conversation BLOCKED yet
    session = _orchestrating_session(
        entries={
            "cr_c": CancelRequested(id="cr_c", parent_id="te_c", created_at=800),
            "te2": ToolExecution(
                id="te2",
                conversation_id="c1",
                parent_id="ch1",
                created_at=700,
                tool_call_id="tc2",
                raw_tool_call=ToolCall(id="tc2", name="write"),
                tool_spec=spec("write"),
                status=ExecutionStatus.PENDING,
                approval_status=ApprovalStatus.PENDING,
                approval_decisions=[PENDING_1000],
            ),
        },
        nodes=["u1", "ts", "a1", "te1", "ch1", "te2"],
    )
    session.conversations["c2"].nodes.append("cr_c")

    assert session.get_conversation_status("c2").status == ConversationStatus.CANCELLING
    assert session.get_conversation_status("c1").status == ConversationStatus.BUSY
