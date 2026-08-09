"""Declarative approval-flow scenarios: the decide() gate, re-asks, cold resume.

Same shape as `test_runner.py` — precondition → one action → postcondition with
full-object asserts — but here the interesting collaborator is the registry's
decide(): a scripted `FakeToolRegistry` (`scenarios.py`) whose `seen` list
records exactly which executions the runner asked it to decide, in order. The
core invariants under test:

- decide() has ONE call site (the top of the loop): it fires for fresh
  executions and for every resume path identically, and ONLY for undecided
  executions (`approval_status` None or PENDING). It is handed the LIVE
  `AgentSession` — there is no `ToolContext` any more.
- A decide() response updates `approval_status` directly AND appends to the
  `approval_decisions` audit log. A DENY is terminal on the spot (REJECTED,
  `finished_at` stamped, `attempts` empty, no dispatch).
- An ALLOW is a permission answer and nothing else: a call that then fails to
  RESOLVE records NOT_FOUND from the prepare phase, never REJECTED, and emits
  no `ToolExecutionStarted` — that event fires iff the body was dispatched.
- A PENDING decision defers only THAT execution: every ALLOWED sibling
  proceeds to dispatch, and the run parks (AWAITING_APPROVAL,
  `ApprovalRequired` as the final event) only after all runnable work
  advanced. The model is never called while any execution is nonterminal.
- Re-entering run() asks the registry again — never raises; a resolved call
  is never re-decided (at most one ALLOW/DENY ever; only PENDING repeats).
- decide() exceptions propagate; the session stays consistent and resumable.

Determinism comes from `DeterministicRunner` (`scenarios.py`); its `ids` script
is consumed in this order per turn:
  TurnStart, (AssistantMessage, [ToolExecution per call])..., TurnFinish
"""

import pytest

from luca.agent.core.events import (
    ApprovalRequired,
    FinishReason,
    TextBlock,
    ToolCallReceived,
    ToolCallStart,
    ToolExecuted,
    ToolExecutionStarted,
)
from luca.agent.core.exceptions import ToolNotFound
from luca.agent.core.models import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    ConversationStatus,
    ExecutionAttempt,
    ExecutionAttemptOutcome,
    ExecutionResult,
    ExecutionStatus,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    TurnFinish,
    TurnStart,
    Usage,
    UserMessage,
)
from luca.agent.core.tool_registry import PreparedTool
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
)
from luca.client.types import TextBlock as LucaTextBlock, ToolMessage
from tests.agent.scenarios import (
    ADD_SPEC,
    CLEARED_SESSION,
    GATED_SESSION,
    MODEL,
    MULTIPLY_SPEC,
    READ_FILE_SPEC,
    STALE_RUNNING_SESSION,
    UNDECIDED_SESSION,
    AddTool,
    DeterministicRunner,
    FakeToolRegistry,
    MultiplyTool,
    ReadFileTool,
    conversation,
    make_session,
)

PENDING_1000 = ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)
ALLOW_1000 = ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)
DENY_1000 = ApprovalDecision(decision=ApprovalOption.DENY, created_at=1000)
ALLOW_2000 = ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=2000)

# A persisted execution carries the DURABLE reference into `session.tool_specs`,
# not just the restorable `tool_spec` cache, so every expected execution literal
# needs the id the ledger stamped on it.
ADD_SPEC_ID = ADD_SPEC.spec_id()
MULTIPLY_SPEC_ID = MULTIPLY_SPEC.spec_id()
READ_FILE_SPEC_ID = READ_FILE_SPEC.spec_id()

ADD_CALL = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})
MULTIPLY_CALL = ToolCall(id="tc2", name="multiply", arguments={"a": 3, "b": 4})


class ExplodingRegistry(FakeToolRegistry):
    """decide() always raises — the decide-failure double."""

    async def decide(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> ApprovalDecision:
        raise RuntimeError("strategy down")


class UnresolvableRegistry(FakeToolRegistry):
    """Births and ALLOWs normally, then cannot resolve at prepare() — the
    routing-registry shape where a name reaches the gate but no child owns it.
    Local to this file: `FakeToolRegistry` always resolves what it births."""

    async def prepare(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> PreparedTool:
        raise ToolNotFound(f"Unknown tool: {tool_execution.raw_tool_call.name!r}.")


class SessionRecordingRegistry(FakeToolRegistry):
    """Records the session object every decide() was handed. Local to this
    file: `FakeToolRegistry.seen` records executions, never sessions."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sessions: list[AgentSession] = []

    async def decide(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> ApprovalDecision:
        self.sessions.append(session)
        return await super().decide(session, conversation_id, tool_execution)


# ── reaching the gate ──────────────────────────────────────────────────────────


async def test_pending_decision_pauses_runner_and_records_it():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
        ]
    )
    session = make_session(
        id="s_ask",
        entries={
            "u1": UserMessage(
                id="u1",
                created_at=500,
                parts=[TextContent(text="Add 1 and 2")],
            ),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry([AddTool()], decisions=[PENDING_1000])
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    gated = birth.model_copy(
        update={
            "approval_status": ApprovalStatus.PENDING,
            "approval_decisions": [PENDING_1000],
            "updated_at": 1000,
        }
    )
    assert events == [
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc1", execution=birth),
        ApprovalRequired(conversation_id="c1", executions=[gated]),
    ]
    assert runner.blocked()
    assert runner.pending_approvals() == [gated]
    assert runner.session.entries["te1"] == gated
    # the ledger filed the spec once, under the id stamped on the execution
    assert runner.session.tool_specs == {ADD_SPEC_ID: ADD_SPEC}
    # decide() was asked exactly once, with the pre-decision snapshot
    assert registry.seen == [birth]


async def test_decide_is_handed_the_live_session():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
        ]
    )
    session = make_session(
        id="s_live",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = SessionRecordingRegistry([AddTool()], decisions=[PENDING_1000])
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1"],
        now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # not a copy and not a ToolContext: the same object the runner and the
    # ledger write through, so a policy can re-read current state from it
    assert len(registry.sessions) == 1
    assert registry.sessions[0] is runner.session


async def test_streaming_pauses_at_approval_gate():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
        ]
    )
    session = make_session(
        id="s_ask_stream",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()], decisions=[PENDING_1000]),
        provider=faux,
        ids=["ts", "a1", "te1"],
        now=1000,
    )

    async with runner.run(streaming=True) as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    gated = birth.model_copy(
        update={
            "approval_status": ApprovalStatus.PENDING,
            "approval_decisions": [PENDING_1000],
            "updated_at": 1000,
        }
    )
    assert events == [
        ToolCallStart(conversation_id="c1", tool_call_id="tc1", name="add"),
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc1", execution=birth),
        ApprovalRequired(conversation_id="c1", executions=[gated]),
    ]
    assert runner.blocked()


async def test_denied_call_is_rejected_and_loop_continues():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Okay, I won't.")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_deny",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()], decisions=[DENY_1000]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    rejected = birth.model_copy(
        update={
            "status": ExecutionStatus.REJECTED,
            "approval_status": ApprovalStatus.REJECTED,
            "approval_decisions": [DENY_1000],
            "finished_at": 1000,
            "updated_at": 1000,
        }
    )
    assert events == [
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc1", execution=birth),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=rejected,
            result_text="[tool execution rejected]",
            is_error=True,
        ),
        TextBlock(conversation_id="c1", text="Okay, I won't."),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert runner.session.entries["te1"] == rejected
    assert runner.session.get_conversation_status(runner.session.main_conversation_id).status == ConversationStatus.IDLE


async def test_mixed_decisions_reject_and_execute_in_one_batch():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    faux_tool_call("add", {"a": 1, "b": 2}, id="tc1"),
                    faux_tool_call("multiply", {"a": 3, "b": 4}, id="tc2"),
                ],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Only added.")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mixed",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry(
            [AddTool(), MultiplyTool()],
            decisions=[ALLOW_1000, DENY_1000],
        ),
        provider=faux,
        ids=["ts", "a1", "te1", "te2", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    birth1 = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    birth2 = ToolExecution(
        id="te2",
        conversation_id="c1",
        parent_id="te1",
        created_at=1000,
        tool_call_id="tc2",
        raw_tool_call=MULTIPLY_CALL,
        tool_spec=MULTIPLY_SPEC,
        tool_spec_id=MULTIPLY_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    running1 = birth1.model_copy(
        update={
            "status": ExecutionStatus.RUNNING,
            "approval_status": ApprovalStatus.ALLOWED,
            "approval_decisions": [ALLOW_1000],
            "attempts": [ExecutionAttempt(started_at=1000)],
            "updated_at": 1000,
        }
    )
    completed1 = running1.model_copy(
        update={
            "status": ExecutionStatus.COMPLETED,
            "result": ExecutionResult(content=[TextContent(text="3")]),
            "attempts": [ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
            "finished_at": 1000,
        }
    )
    rejected2 = birth2.model_copy(
        update={
            "status": ExecutionStatus.REJECTED,
            "approval_status": ApprovalStatus.REJECTED,
            "approval_decisions": [DENY_1000],
            "finished_at": 1000,
            "updated_at": 1000,
        }
    )
    # the denial is terminal at decision time, so its ToolExecuted precedes
    # the allowed sibling's dispatch pair
    assert events == [
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc1", execution=birth1),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc2", execution=birth2),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc2",
            execution=rejected2,
            result_text="[tool execution rejected]",
            is_error=True,
        ),
        ToolExecutionStarted(conversation_id="c1", tool_call_id="tc1", execution=running1),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=completed1,
            result_text="3",
            is_error=False,
        ),
        TextBlock(conversation_id="c1", text="Only added."),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert runner.session.entries["te1"] == completed1
    assert runner.session.entries["te2"] == rejected2
    assert runner.idle()


async def test_approval_context_lands_on_execution_and_reaches_strategy():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("read_file", {"path": "/etc/passwd"}, id="tc1")],
                finish_reason="tool_use",
            ),
        ]
    )
    session = make_session(
        id="s_ctx",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="read")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry([ReadFileTool()], decisions=[PENDING_1000])
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1"],
        now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(
            id="tc1",
            name="read_file",
            arguments={"path": "/etc/passwd"},
        ),
        tool_spec=READ_FILE_SPEC,
        tool_spec_id=READ_FILE_SPEC_ID,
        status=ExecutionStatus.PENDING,
        extras={
            "approval_context": {
                "resources": ["/etc/passwd"],
                "preview": "Read /etc/passwd",
                "remember_as": [{"resource": "/etc/*", "preview": "Allow /etc/*"}],
            },
        },
    )
    gated = birth.model_copy(
        update={
            "approval_status": ApprovalStatus.PENDING,
            "approval_decisions": [PENDING_1000],
            "updated_at": 1000,
        }
    )
    assert runner.session.entries["te1"] == gated
    assert registry.seen == [birth]


# ── an ALLOW is a permission answer, not a resolution ──────────────────────────


async def test_allowed_call_that_cannot_be_resolved_records_not_found():
    # A registry that ALLOWs a name it cannot resolve is not denying anything:
    # prepare() raises ToolNotFound, the call records NOT_FOUND with
    # details["phase"] == "prepare", the body is never dispatched
    # (`attempts` empty) and no ToolExecutionStarted is emitted.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("No such tool.")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_unresolvable",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=UnresolvableRegistry([AddTool()], decisions=[ALLOW_1000]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    not_found = birth.model_copy(
        update={
            "status": ExecutionStatus.NOT_FOUND,
            "approval_status": ApprovalStatus.ALLOWED,
            "approval_decisions": [ALLOW_1000],
            "error": ToolExecutionError(
                error_type="ToolNotFound",
                error_message="Unknown tool: 'add'.",
                details={"phase": "prepare"},
            ),
            "finished_at": 1000,
            "updated_at": 1000,
            "context_tokens": 5,
        }
    )
    assert events == [
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc1", execution=birth),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=not_found,
            result_text="Unknown tool: 'add'.",
            is_error=True,
        ),
        TextBlock(conversation_id="c1", text="No such tool."),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert runner.session.entries["te1"] == not_found
    assert runner.session.entries["te1"].dispatched is False
    assert runner.idle()


# ── re-asking the strategy ───────────────────────────────────────────────────────


async def test_rerun_reasks_strategy_and_accumulates_decisions():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_resume",
        entries={
            "u1": UserMessage(
                id="u1",
                created_at=500,
                parts=[TextContent(text="Add 1 and 2")],
            ),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry(
        [AddTool()],
        decisions=[PENDING_1000, ALLOW_1000],
    )
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:  # pauses at the gate
        _ = [event async for event in run]
    assert runner.blocked()
    async with runner.run() as run:  # NO exception: re-asks
        resume_events = [event async for event in run]

    birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    gated = birth.model_copy(
        update={
            "approval_status": ApprovalStatus.PENDING,
            "approval_decisions": [PENDING_1000],
            "updated_at": 1000,
        }
    )
    running = gated.model_copy(
        update={
            "status": ExecutionStatus.RUNNING,
            "approval_status": ApprovalStatus.ALLOWED,
            "approval_decisions": [PENDING_1000, ALLOW_1000],
            "attempts": [ExecutionAttempt(started_at=1000)],
        }
    )
    completed = running.model_copy(
        update={
            "status": ExecutionStatus.COMPLETED,
            "result": ExecutionResult(content=[TextContent(text="3")]),
            "attempts": [ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
            "finished_at": 1000,
        }
    )
    assert resume_events == [
        ToolExecutionStarted(conversation_id="c1", tool_call_id="tc1", execution=running),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=completed,
            result_text="3",
            is_error=False,
        ),
        TextBlock(conversation_id="c1", text="It's 3."),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert runner.session.entries["te1"] == completed
    # asked twice, each time with the then-current snapshot
    assert registry.seen == [birth, gated]
    assert runner.idle()


async def test_allowed_sibling_dispatches_before_the_run_parks():
    # A PENDING decision defers only that execution: the ALLOWED sibling runs
    # to completion first, ApprovalRequired is the FINAL event, and the model
    # is not called until every call has a terminal execution.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    faux_tool_call("add", {"a": 1, "b": 2}, id="tc1"),
                    faux_tool_call("multiply", {"a": 3, "b": 4}, id="tc2"),
                ],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_partial",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry(
        [AddTool(), MultiplyTool()],
        decisions=[ALLOW_1000, PENDING_1000],
    )
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1", "te2", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        first_events = [event async for event in run]

    birth1 = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    birth2 = ToolExecution(
        id="te2",
        conversation_id="c1",
        parent_id="te1",
        created_at=1000,
        tool_call_id="tc2",
        raw_tool_call=MULTIPLY_CALL,
        tool_spec=MULTIPLY_SPEC,
        tool_spec_id=MULTIPLY_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    running1 = birth1.model_copy(
        update={
            "status": ExecutionStatus.RUNNING,
            "approval_status": ApprovalStatus.ALLOWED,
            "approval_decisions": [ALLOW_1000],
            "attempts": [ExecutionAttempt(started_at=1000)],
            "updated_at": 1000,
        }
    )
    completed1 = running1.model_copy(
        update={
            "status": ExecutionStatus.COMPLETED,
            "result": ExecutionResult(content=[TextContent(text="3")]),
            "attempts": [ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
            "finished_at": 1000,
        }
    )
    gated2 = birth2.model_copy(
        update={
            "approval_status": ApprovalStatus.PENDING,
            "approval_decisions": [PENDING_1000],
            "updated_at": 1000,
        }
    )
    # the allowed sibling ran; only the deferred call holds the turn open
    assert first_events == [
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc1", execution=birth1),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc2", execution=birth2),
        ToolExecutionStarted(conversation_id="c1", tool_call_id="tc1", execution=running1),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=completed1,
            result_text="3",
            is_error=False,
        ),
        ApprovalRequired(conversation_id="c1", executions=[gated2]),
    ]
    assert runner.blocked()
    assert runner.pending_approvals() == [gated2]
    assert runner.session.entries["te1"] == completed1
    assert len(faux.requests) == 1  # no second model call while te2 is open

    registry.decisions.append(DENY_1000)
    async with runner.run() as run:
        events = [event async for event in run]

    rejected2 = gated2.model_copy(
        update={
            "status": ExecutionStatus.REJECTED,
            "approval_status": ApprovalStatus.REJECTED,
            "approval_decisions": [PENDING_1000, DENY_1000],
            "finished_at": 1000,
        }
    )
    assert events == [
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc2",
            execution=rejected2,
            result_text="[tool execution rejected]",
            is_error=True,
        ),
        TextBlock(conversation_id="c1", text="done"),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    # te1 was decided ONCE (never re-asked once resolved); te2 twice
    assert registry.seen == [birth1, birth2, gated2]
    assert runner.session.entries["te2"] == rejected2
    assert runner.idle()


# ── cold resume: persisted mid-state sessions loaded into a fresh runner ────────


async def test_loaded_gated_session_exposes_pending_approvals():
    session = GATED_SESSION.model_copy(deep=True)

    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()], decisions=[]),
        provider=FauxProvider(),
        now=1000,
    )

    assert runner.blocked()
    assert runner.pending_approvals() == [session.entries["te1"]]


async def test_loaded_gated_session_run_reasks_strategy_and_completes():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
        ]
    )
    session = GATED_SESSION.model_copy(deep=True)
    registry = FakeToolRegistry([AddTool()], decisions=[ALLOW_1000])
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:  # NO exception: re-asks
        events = [event async for event in run]

    assert [event.type for event in events] == [
        "tool_execution_started",
        "tool_executed",
        "text_block",
        "finish_reason",
    ]
    assert runner.session == make_session(
        id="s_gated",
        entries={
            "u1": UserMessage(
                id="u1",
                created_at=500,
                parts=[TextContent(text="Add 1 and 2")],
            ),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
            "a1": AssistantMessage(
                id="a1",
                parent_id="ts",
                created_at=500,
                parts=[ADD_CALL],
                llm_config=MODEL,
                stop_reason="tool_use",
            ),
            "te1": ToolExecution(
                id="te1",
                conversation_id="c1",
                parent_id="a1",
                created_at=500,
                tool_call_id="tc1",
                raw_tool_call=ADD_CALL,
                tool_spec=ADD_SPEC,
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(
                    content=[TextContent(text="3")],
                    is_error=False,
                ),
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[
                    ApprovalDecision(
                        decision=ApprovalOption.PENDING,
                        created_at=500,
                    ),
                    ALLOW_1000,
                ],
                attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
                finished_at=1000,
                updated_at=1000,
            ),
            "a2": AssistantMessage(
                id="a2",
                parent_id="te1",
                created_at=1000,
                parts=[TextContent(text="It's 3.")],
                llm_config=MODEL,
                stop_reason="stop",
                context_tokens=1,  # len("It's 3.") // 4
            ),
            "tf": TurnFinish(id="tf", parent_id="a2", created_at=1000),
        },
        usages={"c1": {"a2": Usage(conversation_id="c1", entry_id="a2")}},
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "ts", "a1", "te1", "a2", "tf"],
                created_at=500,
                updated_at=1000,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )


async def test_cleared_execution_dispatches_before_any_llm_call_without_redeciding():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
        ]
    )
    session = CLEARED_SESSION.model_copy(deep=True)
    cleared = session.entries["te1"].model_copy(deep=True)
    # an empty decision script: any decide() would raise
    registry = FakeToolRegistry([AddTool()], decisions=[])
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    running = cleared.model_copy(
        update={
            "status": ExecutionStatus.RUNNING,
            "attempts": [ExecutionAttempt(started_at=1000)],
            "updated_at": 1000,
        }
    )
    completed = running.model_copy(
        update={
            "status": ExecutionStatus.COMPLETED,
            "result": ExecutionResult(content=[TextContent(text="3")]),
            "attempts": [ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
            "finished_at": 1000,
        }
    )
    assert events == [
        ToolExecutionStarted(conversation_id="c1", tool_call_id="tc1", execution=running),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=completed,
            result_text="3",
            is_error=False,
        ),
        TextBlock(conversation_id="c1", text="It's 3."),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    # a resolved call is NEVER re-decided
    assert registry.seen == []
    # exactly one LLM call, made only after the cleared call executed: the
    # request already carries its tool result
    assert len(faux.requests) == 1
    assert faux.requests[0].messages[-1] == ToolMessage(
        tool_call_id="tc1",
        content=[LucaTextBlock(text="3")],
        is_error=False,
    )
    assert runner.session.entries["te1"] == completed
    assert runner.idle()


async def test_undecided_session_self_heals_and_run_asks_strategy():
    # crash mid-decide: execution persisted, approval_status None — NOT
    # awaiting approval (the strategy was never asked); a plain run() asks it.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
        ]
    )
    session = UNDECIDED_SESSION.model_copy(deep=True)
    undecided = session.entries["te1"].model_copy(deep=True)
    registry = FakeToolRegistry([AddTool()], decisions=[ALLOW_1000])
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["a2", "tf"],
        now=1000,
    )

    assert runner.busy()  # stale RUNNING self-healed; not AWAITING_APPROVAL
    async with runner.run() as run:
        events = [event async for event in run]

    completed = undecided.model_copy(
        update={
            "status": ExecutionStatus.COMPLETED,
            "result": ExecutionResult(content=[TextContent(text="3")]),
            "approval_status": ApprovalStatus.ALLOWED,
            "approval_decisions": [ALLOW_1000],
            "attempts": [ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
            "finished_at": 1000,
            "updated_at": 1000,
        }
    )
    assert [event.type for event in events] == [
        "tool_execution_started",
        "tool_executed",
        "text_block",
        "finish_reason",
    ]
    assert runner.session.entries["te1"] == completed
    assert registry.seen == [undecided]
    assert runner.idle()


async def test_stale_running_status_self_heals_on_construction():
    session = STALE_RUNNING_SESSION.model_copy(deep=True)
    assert session.get_conversation_status(session.main_conversation_id).status == ConversationStatus.BUSY

    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()], decisions=[]),
        provider=FauxProvider(),
        now=1000,
    )

    assert runner.busy()  # the open turn with a cleared call means: call run()


# ── decide() failure ─────────────────────────────────────────────────────────


async def test_strategy_exception_propagates_and_session_stays_resumable():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_boom_policy",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=ExplodingRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1"],
        now=1000,
    )

    with pytest.raises(RuntimeError, match="strategy down"):
        async with runner.run() as run:
            _ = [event async for event in run]

    # the execution was persisted eagerly and stays unprocessed...
    birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    assert runner.session.entries["te1"] == birth

    # ...so a fresh runner (same session) with a working registry completes it
    resumed = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()], decisions=[ALLOW_1000]),
        provider=faux,
        ids=["a2", "tf"],
        now=1000,
    )
    assert resumed.busy()
    async with resumed.run() as run:
        events = [event async for event in run]

    running = birth.model_copy(
        update={
            "status": ExecutionStatus.RUNNING,
            "approval_status": ApprovalStatus.ALLOWED,
            "approval_decisions": [ALLOW_1000],
            "attempts": [ExecutionAttempt(started_at=1000)],
            "updated_at": 1000,
        }
    )
    completed = running.model_copy(
        update={
            "status": ExecutionStatus.COMPLETED,
            "result": ExecutionResult(content=[TextContent(text="3")]),
            "attempts": [ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
            "finished_at": 1000,
        }
    )
    assert events == [
        ToolExecutionStarted(conversation_id="c1", tool_call_id="tc1", execution=running),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=completed,
            result_text="3",
            is_error=False,
        ),
        TextBlock(conversation_id="c1", text="It's 3."),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert resumed.idle()


# ── durability: the gate survives a full serialize / reload cycle ───────────────


async def test_gated_session_survives_restart_and_resumes():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_restart",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()], decisions=[PENDING_1000]),
        provider=faux,
        ids=["ts", "a1", "te1"],
        now=1000,
    )
    async with runner.run() as run:  # pauses at the gate
        _ = [event async for event in run]
    payload = runner.session.model_dump_json()  # "process exits" here

    # restart: reload the session into a fresh runner with a fresh registry
    reloaded = AgentSession.model_validate_json(payload)
    resumed = DeterministicRunner(
        reloaded,
        tool_registry=FakeToolRegistry([AddTool()], decisions=[ALLOW_2000]),
        provider=faux,
        ids=["a2", "tf"],
        now=2000,
    )

    gated = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.PENDING,
        approval_status=ApprovalStatus.PENDING,
        approval_decisions=[PENDING_1000],
        updated_at=1000,
    )
    # the serialized session carries the spec ONCE in `tool_specs`; loading it
    # restored the execution's `tool_spec` cache from `tool_spec_id`
    assert reloaded.tool_specs == {ADD_SPEC_ID: ADD_SPEC}
    assert reloaded.entries["te1"] == gated
    assert resumed.blocked()
    assert resumed.pending_approvals() == [gated]

    async with resumed.run() as run:
        events = [event async for event in run]

    running = gated.model_copy(
        update={
            "status": ExecutionStatus.RUNNING,
            "approval_status": ApprovalStatus.ALLOWED,
            "approval_decisions": [PENDING_1000, ALLOW_2000],
            "attempts": [ExecutionAttempt(started_at=2000)],
            "updated_at": 2000,
        }
    )
    completed = running.model_copy(
        update={
            "status": ExecutionStatus.COMPLETED,
            "result": ExecutionResult(content=[TextContent(text="3")]),
            "attempts": [ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=2000, ended_at=2000)],
            "finished_at": 2000,
        }
    )
    assert events == [
        ToolExecutionStarted(conversation_id="c1", tool_call_id="tc1", execution=running),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=completed,
            result_text="3",
            is_error=False,
        ),
        TextBlock(conversation_id="c1", text="It's 3."),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert resumed.idle()
    assert resumed.session.entries["te1"] == completed
    assert resumed.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="a2",
        created_at=2000,
    )
