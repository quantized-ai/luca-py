"""Declarative matrix for the SessionLedger — the entry-derived queries the
runner's status cache is re-derived from, plus the uniform append/put
bookkeeping. Known session data in, exact answer out; no loop, no provider.

The execution vocabulary under test (over `status` + `approval_status`):
PENDING splits into UNDECIDED (`approval_status` None or PENDING — what the
runner offers to the policy), AWAITING (`approval_status` PENDING — the
policy explicitly deferred; drives AWAITING_APPROVAL), and READY
(`approval_status` ALLOWED — dispatchable). RUNNING executions are orphans at
drive start. Approval state is read from `approval_status`, never
reconstructed from the `approval_decisions` audit log.
"""

import pytest

from luca.agent.core.exceptions import AgentError
from luca.agent.core.models import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    CancelRequested,
    Conversation,
    ConversationStatus,
    ExecutionResult,
    ExecutionStatus,
    LLMConfig,
    PrunedEntry,
    SessionConfig,
    SessionRuntimeStatus,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolSpec,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    Usage,
    UserMessage,
)
from luca.agent.core.ledger import SessionLedger

from tests.agent.scenarios import MODEL

PENDING_1000 = ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)
ALLOW_1000 = ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)
DENY_1000 = ApprovalDecision(decision=ApprovalOption.DENY, created_at=1000)


# ── open_turn_index ────────────────────────────────────────────────────────────


def test_open_turn_index_is_none_on_empty_path():
    session = AgentSession(
        id="s",
        active_conversation=Conversation(id="c1", nodes=[], created_at=0, updated_at=0),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_turn_index() is None


def test_open_turn_index_is_none_after_turn_finish():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts": TurnStart(id="ts", created_at=1),
            "tf": TurnFinish(id="tf", created_at=2),
        },
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts", "tf"], created_at=0, updated_at=2,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_turn_index() is None


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
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts1", "tf1", "u2", "ts2"], created_at=0, updated_at=4,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_turn_index() == 4


# ── derive_status ──────────────────────────────────────────────────────────────


def test_derive_status_idle_on_empty_session():
    session = AgentSession(
        id="s",
        active_conversation=Conversation(id="c1", nodes=[], created_at=0, updated_at=0),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.derive_status() == ConversationStatus.IDLE


def test_derive_status_pending_with_trailing_user_message():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=0, updated_at=0),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.derive_status() == ConversationStatus.PENDING


def test_derive_status_pending_with_open_turn_and_no_executions():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts": TurnStart(id="ts", created_at=1),
        },
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts"], created_at=0, updated_at=1,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.derive_status() == ConversationStatus.PENDING


def test_derive_status_awaiting_when_approval_status_is_pending():
    session = AgentSession(
        id="s",
        entries={
            "ts": TurnStart(id="ts", created_at=1),
            "te1": ToolExecution(
                id="te1", created_at=2, tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=ToolSpec(name="add"),
                status=ExecutionStatus.PENDING,
                approval_status=ApprovalStatus.PENDING,
                approval_decisions=[PENDING_1000],
            ),
        },
        active_conversation=Conversation(
            id="c1", nodes=["ts", "te1"], created_at=0, updated_at=2,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.derive_status() == ConversationStatus.AWAITING_APPROVAL


def test_derive_status_pending_when_execution_was_never_processed():
    # crash mid-decide: approval_status None — the strategy was never asked,
    # so the right move is a plain run() (which asks it), NOT the gate.
    session = AgentSession(
        id="s",
        entries={
            "ts": TurnStart(id="ts", created_at=1),
            "te1": ToolExecution(
                id="te1", created_at=2, tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=ToolSpec(name="add"),
                status=ExecutionStatus.PENDING,
                approval_status=None,
                approval_decisions=[],
            ),
        },
        active_conversation=Conversation(
            id="c1", nodes=["ts", "te1"], created_at=0, updated_at=2,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.derive_status() == ConversationStatus.PENDING


def test_derive_status_pending_when_execution_is_allowed_but_unrun():
    session = AgentSession(
        id="s",
        entries={
            "ts": TurnStart(id="ts", created_at=1),
            "te1": ToolExecution(
                id="te1", created_at=2, tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=ToolSpec(name="add"),
                status=ExecutionStatus.PENDING,
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[PENDING_1000, ALLOW_1000],
            ),
        },
        active_conversation=Conversation(
            id="c1", nodes=["ts", "te1"], created_at=0, updated_at=2,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.derive_status() == ConversationStatus.PENDING


def test_derive_status_pending_with_orphaned_running_execution():
    # a persisted RUNNING execution means: call run() — the next drive
    # recovers it to INTERRUPTED and continues the turn.
    session = AgentSession(
        id="s",
        entries={
            "ts": TurnStart(id="ts", created_at=1),
            "te1": ToolExecution(
                id="te1", created_at=2, tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=ToolSpec(name="add"),
                status=ExecutionStatus.RUNNING,
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[ALLOW_1000],
                started_at=3,
            ),
        },
        active_conversation=Conversation(
            id="c1", nodes=["ts", "te1"], created_at=0, updated_at=3,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.derive_status() == ConversationStatus.PENDING


def test_derive_status_idle_after_closed_turn():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts": TurnStart(id="ts", created_at=1),
            "tf": TurnFinish(id="tf", created_at=2),
        },
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts", "tf"], created_at=0, updated_at=2,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.derive_status() == ConversationStatus.IDLE


def test_derive_status_cancelling_beats_awaiting_approval():
    # an unconsumed CancelRequested wins over the approval gate: the next
    # drive is a flush, not a re-ask.
    session = AgentSession(
        id="s",
        entries={
            "ts": TurnStart(id="ts", created_at=1),
            "te1": ToolExecution(
                id="te1", created_at=2, tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=ToolSpec(name="add"),
                status=ExecutionStatus.PENDING,
                approval_status=ApprovalStatus.PENDING,
                approval_decisions=[PENDING_1000],
            ),
            "cr": CancelRequested(id="cr", created_at=3),
        },
        active_conversation=Conversation(
            id="c1", nodes=["ts", "te1", "cr"], created_at=0, updated_at=3,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.derive_status() == ConversationStatus.CANCELLING


def test_derive_status_pending_after_timed_out_turn():
    # a failed close is retry-ready, not idle: run() opens a new bracket.
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts": TurnStart(id="ts", created_at=1),
            "tf": TurnFinish(
                id="tf", created_at=2,
                outcome=TurnOutcome.TIMED_OUT, error="client timeout",
            ),
        },
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts", "tf"], created_at=0, updated_at=2,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.derive_status() == ConversationStatus.PENDING


def test_derive_status_pending_after_errored_turn():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts": TurnStart(id="ts", created_at=1),
            "tf": TurnFinish(
                id="tf", created_at=2,
                outcome=TurnOutcome.ERRORED, error="boom",
            ),
        },
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts", "tf"], created_at=0, updated_at=2,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.derive_status() == ConversationStatus.PENDING


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
                id="tf", created_at=3,
                outcome=TurnOutcome.CANCELLED,
            ),
        },
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts", "cr", "tf"], created_at=0, updated_at=3,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.derive_status() == ConversationStatus.IDLE


# ── the execution-subset matrix ────────────────────────────────────────────────
#
# One session, five executions covering every lifecycle/approval combination
# the runner asks about: terminal (skipped by all pending queries), undecided
# (never processed), awaiting (policy deferred), ready (allowed, unrun), and
# orphaned RUNNING.

MATRIX_SESSION = AgentSession(
    id="s_matrix",
    entries={
        "ts": TurnStart(id="ts", created_at=1),
        "te_done": ToolExecution(
            id="te_done", created_at=2, tool_call_id="tc0",
            raw_tool_call=ToolCall(id="tc0", name="add", arguments={"a": 1, "b": 2}),
            tool_spec=ToolSpec(name="add"),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="3")]),
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[ALLOW_1000],
            started_at=3, ended_at=3,
        ),
        "te_undecided": ToolExecution(
            id="te_undecided", created_at=4, tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add"),
            tool_spec=ToolSpec(name="add"),
            status=ExecutionStatus.PENDING,
            approval_status=None,
            approval_decisions=[],
        ),
        "te_awaiting": ToolExecution(
            id="te_awaiting", created_at=5, tool_call_id="tc2",
            raw_tool_call=ToolCall(id="tc2", name="multiply"),
            tool_spec=ToolSpec(name="multiply"),
            status=ExecutionStatus.PENDING,
            approval_status=ApprovalStatus.PENDING,
            approval_decisions=[PENDING_1000],
        ),
        "te_ready": ToolExecution(
            id="te_ready", created_at=6, tool_call_id="tc3",
            raw_tool_call=ToolCall(id="tc3", name="subtract"),
            tool_spec=ToolSpec(name="subtract"),
            status=ExecutionStatus.PENDING,
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[ALLOW_1000],
        ),
        "te_running": ToolExecution(
            id="te_running", created_at=7, tool_call_id="tc4",
            raw_tool_call=ToolCall(id="tc4", name="add"),
            tool_spec=ToolSpec(name="add"),
            status=ExecutionStatus.RUNNING,
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[ALLOW_1000],
            started_at=8,
        ),
    },
    tool_executions={
        "tc0": ["te_done"], "tc1": ["te_undecided"], "tc2": ["te_awaiting"],
        "tc3": ["te_ready"], "tc4": ["te_running"],
    },
    active_conversation=Conversation(
        id="c1", nodes=["ts", "te_done", "te_undecided", "te_awaiting", "te_ready", "te_running"],
        created_at=0, updated_at=8,
    ),
    session_config=SessionConfig(llm_config=MODEL),
)


def test_execution_subsets_partition_by_status_and_approval():
    session = MATRIX_SESSION.model_copy(deep=True)
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_turn_executions() == [
        session.entries["te_done"],
        session.entries["te_undecided"],
        session.entries["te_awaiting"],
        session.entries["te_ready"],
        session.entries["te_running"],
    ]
    assert ledger.open_turn_pending_executions() == [
        session.entries["te_undecided"],
        session.entries["te_awaiting"],
        session.entries["te_ready"],
    ]
    assert ledger.open_turn_undecided_executions() == [
        session.entries["te_undecided"],
        session.entries["te_awaiting"],
    ]
    assert ledger.open_turn_awaiting_executions() == [
        session.entries["te_awaiting"],
    ]
    assert ledger.open_turn_ready_executions() == [
        session.entries["te_ready"],
    ]
    assert ledger.open_turn_running_executions() == [
        session.entries["te_running"],
    ]
    assert ledger.has_awaiting_approval() is True


def test_execution_subsets_are_empty_without_open_turn():
    session = AgentSession(
        id="s",
        entries={
            "ts": TurnStart(id="ts", created_at=1),
            "tf": TurnFinish(id="tf", created_at=2),
        },
        active_conversation=Conversation(
            id="c1", nodes=["ts", "tf"], created_at=0, updated_at=2,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_turn_executions() == []
    assert ledger.open_turn_pending_executions() == []
    assert ledger.open_turn_undecided_executions() == []
    assert ledger.open_turn_awaiting_executions() == []
    assert ledger.open_turn_ready_executions() == []
    assert ledger.open_turn_running_executions() == []
    assert ledger.has_awaiting_approval() is False


def test_approval_state_is_read_from_approval_status_not_the_log():
    # the audit log trails a PENDING decision, but approval_status says
    # ALLOWED — the status field wins (middleware may author such state).
    session = AgentSession(
        id="s",
        entries={
            "ts": TurnStart(id="ts", created_at=1),
            "te1": ToolExecution(
                id="te1", created_at=2, tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=ToolSpec(name="add"),
                status=ExecutionStatus.PENDING,
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[PENDING_1000],
            ),
        },
        active_conversation=Conversation(
            id="c1", nodes=["ts", "te1"], created_at=0, updated_at=2,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_turn_ready_executions() == [session.entries["te1"]]
    assert ledger.open_turn_awaiting_executions() == []
    assert ledger.open_turn_undecided_executions() == []


# ── open_turn_cancel_requested ─────────────────────────────────────────────────


def test_open_turn_cancel_requested_finds_the_unconsumed_entry():
    session = AgentSession(
        id="s",
        entries={
            "ts": TurnStart(id="ts", created_at=1),
            "cr": CancelRequested(id="cr", created_at=2, error="user hit ESC"),
        },
        active_conversation=Conversation(
            id="c1", nodes=["ts", "cr"], created_at=0, updated_at=2,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_turn_cancel_requested() == CancelRequested(
        id="cr", created_at=2, outcome=TurnOutcome.CANCELLED, error="user hit ESC",
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
                id="tf1", created_at=3,
                outcome=TurnOutcome.CANCELLED,
            ),
            "u2": UserMessage(id="u2", created_at=4, parts=[TextContent(text="go")]),
            "ts2": TurnStart(id="ts2", created_at=5),
        },
        active_conversation=Conversation(
            id="c1", nodes=["ts1", "cr", "tf1", "u2", "ts2"], created_at=0, updated_at=5,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_turn_cancel_requested() is None


def test_open_turn_cancel_requested_is_none_without_open_turn():
    session = AgentSession(
        id="s",
        entries={
            "ts": TurnStart(id="ts", created_at=1),
            "cr": CancelRequested(id="cr", created_at=2),
            "tf": TurnFinish(
                id="tf", created_at=3,
                outcome=TurnOutcome.CANCELLED,
            ),
        },
        active_conversation=Conversation(
            id="c1", nodes=["ts", "cr", "tf"], created_at=0, updated_at=3,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    assert ledger.open_turn_cancel_requested() is None


# ── record_usage: the single write door onto AgentSession.usages ──────────────


def test_record_usage_builds_the_record_and_indexes_it_by_conversation():
    session = AgentSession(
        id="s",
        entries={
            "a1": AssistantMessage(
                id="a1", created_at=1, parts=[TextContent(text="hi")],
                llm_config=MODEL, stop_reason="stop",
            ),
        },
        active_conversation=Conversation(
            id="c1", nodes=["a1"], created_at=0, updated_at=1,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    record = ledger.record_usage(
        "a1", input=10, output=5, cache_read=2, total_tokens=15,
    )

    assert record == Usage(
        conversation_id="c1", entry_id="a1",
        input=10, output=5, cache_read=2, cache_write=0, total_tokens=15,
    )
    assert session.usages == {"c1": {"a1": record}}


def test_record_usage_replaces_the_pair_record_on_re_record():
    session = AgentSession(
        id="s",
        entries={
            "a1": AssistantMessage(
                id="a1", created_at=1, parts=[TextContent(text="hi")],
                llm_config=MODEL, stop_reason="stop",
            ),
        },
        active_conversation=Conversation(
            id="c1", nodes=["a1"], created_at=0, updated_at=1,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")
    ledger.record_usage("a1", input=10, total_tokens=10)

    ledger.record_usage("a1", input=20, total_tokens=20)

    assert session.usages == {
        "c1": {
            "a1": Usage(
                conversation_id="c1", entry_id="a1", input=20, total_tokens=20,
            ),
        },
    }


def test_record_usage_rejects_a_missing_entry():
    session = AgentSession(
        id="s",
        active_conversation=Conversation(
            id="c1", nodes=[], created_at=0, updated_at=0,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    with pytest.raises(AgentError, match="no such entry"):
        ledger.record_usage("ghost", input=10)


def test_record_usage_rejects_an_entry_outside_the_conversation_path():
    session = AgentSession(
        id="s",
        entries={
            "a1": AssistantMessage(
                id="a1", created_at=1, parts=[TextContent(text="hi")],
                llm_config=MODEL, stop_reason="stop",
            ),
        },
        active_conversation=Conversation(
            id="c1", nodes=[], created_at=0, updated_at=0,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "x")

    with pytest.raises(AgentError, match="not on conversation"):
        ledger.record_usage("a1", input=10)


# ── append / put bookkeeping ───────────────────────────────────────────────────


def test_append_links_parent_extends_path_and_stamps_updated_at():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=0, updated_at=0),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "ts")

    entry = ledger.append(
        lambda entry_id, parent_id, ts: TurnStart(
            id=entry_id, parent_id=parent_id, created_at=ts,
        )
    )

    assert entry == TurnStart(id="ts", parent_id="u1", created_at=1000)
    assert session.entries["ts"] == entry
    assert session.active_conversation.nodes == ["u1", "ts"]
    assert session.active_conversation.updated_at == 1000


def test_append_tool_execution_indexes_by_tool_call_id():
    session = AgentSession(
        id="s",
        entries={"ts": TurnStart(id="ts", created_at=0)},
        active_conversation=Conversation(id="c1", nodes=["ts"], created_at=0, updated_at=0),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "te1")

    ledger.append(
        lambda entry_id, parent_id, ts: ToolExecution(
            id=entry_id, parent_id=parent_id, created_at=ts,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add"),
            tool_spec=ToolSpec(name="add"),
        )
    )

    assert session.tool_executions == {"tc1": ["te1"]}


def test_put_execution_stores_the_replacement_and_touches_conversation():
    session = AgentSession(
        id="s",
        entries={
            "te1": ToolExecution(
                id="te1", created_at=0, tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=ToolSpec(name="add"),
                status=ExecutionStatus.PENDING,
            ),
        },
        tool_executions={"tc1": ["te1"]},
        active_conversation=Conversation(id="c1", nodes=["te1"], created_at=0, updated_at=0),
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

    stored = ledger.put_execution(replacement)

    assert stored is replacement
    assert session.entries["te1"] == ToolExecution(
        id="te1", created_at=0, tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add"),
        tool_spec=ToolSpec(name="add"),
        status=ExecutionStatus.REJECTED,
        approval_status=ApprovalStatus.REJECTED,
        approval_decisions=[DENY_1000],
        ended_at=1000,
        updated_at=1000,
    )
    assert session.active_conversation.updated_at == 1000


# ── prune: the single path-replacement door ────────────────────────────────────


def test_prune_replaces_the_node_in_place_and_keeps_the_original_entry():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "te1": ToolExecution(
                id="te1", parent_id="u1", created_at=1, tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=ToolSpec(name="add"),
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(content=[TextContent(text="3")]),
                started_at=1, ended_at=1,
            ),
            "a2": AssistantMessage(
                id="a2", parent_id="te1", created_at=2,
                parts=[TextContent(text="done")],
                llm_config=MODEL, stop_reason="stop",
            ),
        },
        tool_executions={"tc1": ["te1"]},
        active_conversation=Conversation(
            id="c1", nodes=["u1", "te1", "a2"], created_at=0, updated_at=2,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "p1")

    pruned = ledger.prune(
        "te1",
        lambda entry_id, parent_id, ts: PrunedEntry(
            id=entry_id, parent_id=parent_id, created_at=ts,
            pruned_entry_type="tool_execution",
            pruned_entry_id="te1",
            content=[TextContent(text="[pruned]")],
            context_tokens=2,
        ),
    )

    # the pruned entry took the original's position AND its parent link
    assert pruned == PrunedEntry(
        id="p1", parent_id="u1", created_at=1000,
        pruned_entry_type="tool_execution",
        pruned_entry_id="te1",
        content=[TextContent(text="[pruned]")],
        context_tokens=2,
    )
    assert session.entries["p1"] == pruned
    assert session.active_conversation.nodes == ["u1", "p1", "a2"]
    assert session.active_conversation.updated_at == 1000
    # the original is untouched: still in the store, still indexed
    assert session.entries["te1"].status == ExecutionStatus.COMPLETED
    assert session.tool_executions == {"tc1": ["te1"]}


def test_prune_rejects_a_missing_entry():
    session = AgentSession(
        id="s",
        active_conversation=Conversation(id="c1", nodes=[], created_at=0, updated_at=0),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "p1")

    with pytest.raises(AgentError, match="no such entry"):
        ledger.prune("ghost", lambda entry_id, parent_id, ts: None)


def test_prune_rejects_an_entry_outside_the_active_path():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=[], created_at=0, updated_at=0),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "p1")

    with pytest.raises(AgentError, match="not on conversation"):
        ledger.prune("u1", lambda entry_id, parent_id, ts: None)


def test_prune_rejects_a_replacement_referencing_a_different_entry():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=0, updated_at=0),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "p1")

    with pytest.raises(AgentError, match="replacing"):
        ledger.prune(
            "u1",
            lambda entry_id, parent_id, ts: PrunedEntry(
                id=entry_id, parent_id=parent_id, created_at=ts,
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
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=0, updated_at=0),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "p1")

    with pytest.raises(AgentError, match="pruned_entry_type"):
        ledger.prune(
            "u1",
            lambda entry_id, parent_id, ts: PrunedEntry(
                id=entry_id, parent_id=parent_id, created_at=ts,
                pruned_entry_type="tool_execution",
                pruned_entry_id="u1",
                content=[TextContent(text="[pruned]")],
            ),
        )


def test_prune_rejects_a_nonterminal_execution():
    session = AgentSession(
        id="s",
        entries={
            "te1": ToolExecution(
                id="te1", created_at=0, tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add"),
                tool_spec=ToolSpec(name="add"),
                status=ExecutionStatus.RUNNING,
                started_at=0,
            ),
        },
        tool_executions={"tc1": ["te1"]},
        active_conversation=Conversation(id="c1", nodes=["te1"], created_at=0, updated_at=0),
        session_config=SessionConfig(llm_config=MODEL),
    )
    ledger = SessionLedger(session, clock=lambda: 1000, gen_id=lambda: "p1")

    with pytest.raises(AgentError, match="nonterminal"):
        ledger.prune(
            "te1",
            lambda entry_id, parent_id, ts: PrunedEntry(
                id=entry_id, parent_id=parent_id, created_at=ts,
                pruned_entry_type="tool_execution",
                pruned_entry_id="te1",
                content=[TextContent(text="[pruned]")],
            ),
        )


# ── SessionRuntimeStatus.get_runtime_status_from_agent_session ────────────────
#
# Known AgentSession assembled from entry literals → exact SessionRuntimeStatus.
# No runner, no provider — pure derivation from entries.

_LLM = LLMConfig(model="test-model", provider="faux")
_AM = AssistantMessage(
    id="a1", created_at=1,
    parts=[TextContent(text="ok")],
    llm_config=_LLM,
    stop_reason="stop",
)


def test_runtime_status_empty_session():
    session = AgentSession(
        id="s",
        active_conversation=Conversation(id="c1", nodes=[], created_at=0, updated_at=0),
        session_config=SessionConfig(llm_config=_LLM),
    )
    assert SessionRuntimeStatus.get_runtime_status_from_agent_session(session) == \
        SessionRuntimeStatus(status=ConversationStatus.IDLE, turn_count=0, step_count=0)


def test_runtime_status_open_turn_no_steps():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts": TurnStart(id="ts", created_at=1),
        },
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts"], created_at=0, updated_at=1,
            status=ConversationStatus.RUNNING,
        ),
        session_config=SessionConfig(llm_config=_LLM),
    )
    assert SessionRuntimeStatus.get_runtime_status_from_agent_session(session) == \
        SessionRuntimeStatus(status=ConversationStatus.RUNNING, turn_count=1, step_count=0)


def test_runtime_status_open_turn_one_step():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts": TurnStart(id="ts", created_at=1),
            "a1": _AM,
        },
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts", "a1"], created_at=0, updated_at=1,
            status=ConversationStatus.RUNNING,
        ),
        session_config=SessionConfig(llm_config=_LLM),
    )
    assert SessionRuntimeStatus.get_runtime_status_from_agent_session(session) == \
        SessionRuntimeStatus(status=ConversationStatus.RUNNING, turn_count=1, step_count=1)


def test_runtime_status_open_turn_two_steps():
    a2 = AssistantMessage(
        id="a2", created_at=2,
        parts=[TextContent(text="also ok")],
        llm_config=_LLM, stop_reason="stop",
    )
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts": TurnStart(id="ts", created_at=1),
            "a1": _AM,
            "a2": a2,
        },
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts", "a1", "a2"], created_at=0, updated_at=2,
            status=ConversationStatus.RUNNING,
        ),
        session_config=SessionConfig(llm_config=_LLM),
    )
    assert SessionRuntimeStatus.get_runtime_status_from_agent_session(session) == \
        SessionRuntimeStatus(status=ConversationStatus.RUNNING, turn_count=1, step_count=2)


def test_runtime_status_closed_turn_no_open_turn():
    session = AgentSession(
        id="s",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="hi")]),
            "ts": TurnStart(id="ts", created_at=1),
            "a1": _AM,
            "tf": TurnFinish(id="tf", created_at=2, outcome=TurnOutcome.COMPLETED),
        },
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts", "a1", "tf"], created_at=0, updated_at=2,
            status=ConversationStatus.IDLE,
        ),
        session_config=SessionConfig(llm_config=_LLM),
    )
    # Closed turn: step_count=0 (no open turn), turn_count=1
    assert SessionRuntimeStatus.get_runtime_status_from_agent_session(session) == \
        SessionRuntimeStatus(status=ConversationStatus.IDLE, turn_count=1, step_count=0)


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
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts1", "a1", "tf1", "u2", "ts2"],
            created_at=0, updated_at=4,
            status=ConversationStatus.RUNNING,
        ),
        session_config=SessionConfig(llm_config=_LLM),
    )
    # Second open turn with no steps yet; turn_count=2 (two TurnStarts)
    assert SessionRuntimeStatus.get_runtime_status_from_agent_session(session) == \
        SessionRuntimeStatus(status=ConversationStatus.RUNNING, turn_count=2, step_count=0)


def test_runtime_status_second_open_turn_with_steps():
    a2 = AssistantMessage(
        id="a2", created_at=5,
        parts=[TextContent(text="reply")],
        llm_config=_LLM, stop_reason="stop",
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
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts1", "a1", "tf1", "u2", "ts2", "a2"],
            created_at=0, updated_at=5,
            status=ConversationStatus.RUNNING,
        ),
        session_config=SessionConfig(llm_config=_LLM),
    )
    # step_count counts only AssistantMessages in the CURRENT (second) open turn
    assert SessionRuntimeStatus.get_runtime_status_from_agent_session(session) == \
        SessionRuntimeStatus(status=ConversationStatus.RUNNING, turn_count=2, step_count=1)
