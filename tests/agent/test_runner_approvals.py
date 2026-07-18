"""Declarative approval-flow scenarios: the decide() gate, re-asks, cold resume.

Same shape as `test_runner.py` — precondition → one action → postcondition with
full-object asserts — but here the interesting collaborator is the registry's
decide(): a scripted `FakeToolRegistry` (`scenarios.py`) whose `seen` list
records exactly which executions the runner asked it to decide, in order. The
core invariants under test:

- decide() has ONE call site (the top of the loop): it fires for fresh
  executions and for every resume path identically, and ONLY for undecided
  executions (`approval_status` None or PENDING).
- A decide() response updates `approval_status` directly AND appends to the
  `approval_decisions` audit log. A DENY is terminal on the spot (REJECTED,
  `ended_at` stamped, no dispatch).
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

from luca.agent.core.models import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    Conversation,
    ConversationStatus,
    ExecutionResult,
    ExecutionStatus,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolKind,
    ToolSpec,
    TurnFinish,
    TurnStart,
    Usage,
    UserMessage,
)
from luca.agent.core.events import (
    ApprovalRequired,
    FinishReason,
    TextBlock,
    ToolCallReceived,
    ToolCallStart,
    ToolExecuted,
    ToolExecutionStarted,
)
from luca.agent.core.context import ToolContext
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
)
from luca.client.types import TextBlock as LucaTextBlock
from luca.client.types import ToolMessage

from tests.agent.scenarios import (
    CLEARED_SESSION,
    GATED_SESSION,
    MODEL,
    STALE_RUNNING_SESSION,
    UNDECIDED_SESSION,
    AddTool,
    DeterministicRunner,
    FakeToolRegistry,
    MultiplyTool,
    ReadFileTool,
)

PENDING_1000 = ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)
ALLOW_1000 = ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)
DENY_1000 = ApprovalDecision(decision=ApprovalOption.DENY, created_at=1000)

ADD_SPEC = ToolSpec(name="add", description="Add two numbers.")
MULTIPLY_SPEC = ToolSpec(name="multiply", description="Multiply two numbers.")


class ExplodingRegistry(FakeToolRegistry):
    """decide() always raises — the decide-failure double."""

    async def decide(
        self, tool_execution: ToolExecution, context: ToolContext,
    ) -> ApprovalDecision:
        raise RuntimeError("strategy down")


# ── reaching the gate ──────────────────────────────────────────────────────────


async def test_pending_decision_pauses_runner_and_records_it():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
    ])
    session = AgentSession(
        id="s_ask",
        entries={
            "u1": UserMessage(
                id="u1", created_at=500, parts=[TextContent(text="Add 1 and 2")],
            ),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry([AddTool()], decisions=[PENDING_1000])
    runner = DeterministicRunner(
        session, tool_registry=registry, provider=faux,
        ids=["ts", "a1", "te1"], now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1", parent_id="a1", created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        status=ExecutionStatus.PENDING,
    )
    gated = birth.model_copy(update={
        "approval_status": ApprovalStatus.PENDING,
        "approval_decisions": [PENDING_1000],
        "updated_at": 1000,
    })
    assert events == [
        FinishReason(finish_reason="tool_use"),
        ToolCallReceived(tool_call_id="tc1", execution=birth),
        ApprovalRequired(executions=[gated]),
    ]
    assert runner.awaiting_approval()
    assert runner.pending_approvals() == [gated]
    assert runner.session.entries["te1"] == gated
    assert runner.session.tool_executions == {"tc1": ["te1"]}
    # decide() was asked exactly once, with the pre-decision snapshot
    assert registry.seen == [birth]


async def test_streaming_pauses_at_approval_gate():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
    ])
    session = AgentSession(
        id="s_ask_stream",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()], decisions=[PENDING_1000]),
        provider=faux, ids=["ts", "a1", "te1"], now=1000,
    )

    async with runner.run(streaming=True) as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1", parent_id="a1", created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        status=ExecutionStatus.PENDING,
    )
    assert events == [
        ToolCallStart(tool_call_id="tc1", name="add"),
        FinishReason(finish_reason="tool_use"),
        ToolCallReceived(tool_call_id="tc1", execution=birth),
        ApprovalRequired(executions=[
            birth.model_copy(update={
                "approval_status": ApprovalStatus.PENDING,
                "approval_decisions": [PENDING_1000],
                "updated_at": 1000,
            }),
        ]),
    ]
    assert runner.awaiting_approval()


async def test_denied_call_is_rejected_and_loop_continues():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("Okay, I won't.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_deny",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()], decisions=[DENY_1000]),
        provider=faux, ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1", parent_id="a1", created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        status=ExecutionStatus.PENDING,
    )
    rejected = birth.model_copy(update={
        "status": ExecutionStatus.REJECTED,
        "approval_status": ApprovalStatus.REJECTED,
        "approval_decisions": [DENY_1000],
        "ended_at": 1000,
        "updated_at": 1000,
    })
    assert events == [
        FinishReason(finish_reason="tool_use"),
        ToolCallReceived(tool_call_id="tc1", execution=birth),
        ToolExecuted(
            tool_call_id="tc1", execution=rejected,
            result_text="[tool execution rejected]", is_error=True,
        ),
        TextBlock(text="Okay, I won't."),
        FinishReason(finish_reason="stop"),
    ]
    assert runner.session.entries["te1"] == rejected
    assert runner.session.active_conversation.status == ConversationStatus.IDLE


async def test_mixed_decisions_reject_and_execute_in_one_batch():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1"),
             faux_tool_call("multiply", {"a": 3, "b": 4}, id="tc2")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("Only added.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_mixed",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry(
            [AddTool(), MultiplyTool()], decisions=[ALLOW_1000, DENY_1000],
        ),
        provider=faux, ids=["ts", "a1", "te1", "te2", "a2", "tf"], now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    # the denial is terminal at decision time, so its ToolExecuted precedes
    # the allowed sibling's dispatch pair
    assert [(event.type, event.tool_call_id) for event in events[1:6]] == [
        ("tool_call_received", "tc1"),
        ("tool_call_received", "tc2"),
        ("tool_executed", "tc2"),
        ("tool_execution_started", "tc1"),
        ("tool_executed", "tc1"),
    ]
    assert events[6:] == [
        TextBlock(text="Only added."),
        FinishReason(finish_reason="stop"),
    ]
    assert runner.session.entries["te1"].status == ExecutionStatus.COMPLETED
    assert runner.session.entries["te1"].approval_status == ApprovalStatus.ALLOWED
    assert runner.session.entries["te1"].approval_decisions == [ALLOW_1000]
    assert runner.session.entries["te2"].status == ExecutionStatus.REJECTED
    assert runner.session.entries["te2"].approval_status == ApprovalStatus.REJECTED
    assert runner.session.entries["te2"].approval_decisions == [DENY_1000]
    assert runner.session.entries["te2"].ended_at == 1000
    assert runner.session.entries["te2"].started_at is None
    assert runner.session.tool_executions == {"tc1": ["te1"], "tc2": ["te2"]}
    assert runner.idle()


async def test_approval_context_lands_on_execution_and_reaches_strategy():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("read_file", {"path": "/etc/passwd"}, id="tc1")],
            finish_reason="tool_use",
        ),
    ])
    session = AgentSession(
        id="s_ctx",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="read")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry([ReadFileTool()], decisions=[PENDING_1000])
    runner = DeterministicRunner(
        session, tool_registry=registry, provider=faux,
        ids=["ts", "a1", "te1"], now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    expected_context = {
        "resources": ["/etc/passwd"],
        "preview": "Read /etc/passwd",
        "remember_as": [{"resource": "/etc/*", "preview": "Allow /etc/*"}],
    }
    assert runner.session.entries["te1"].extras == {
        "approval_context": expected_context,
    }
    assert runner.session.entries["te1"].tool_spec.tool_kind == ToolKind.READ
    assert registry.seen[0].extras == {"approval_context": expected_context}


# ── re-asking the strategy ───────────────────────────────────────────────────────


async def test_rerun_reasks_strategy_and_accumulates_decisions():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_resume",
        entries={
            "u1": UserMessage(
                id="u1", created_at=500, parts=[TextContent(text="Add 1 and 2")],
            ),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry(
        [AddTool()], decisions=[PENDING_1000, ALLOW_1000],
    )
    runner = DeterministicRunner(
        session, tool_registry=registry, provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    async with runner.run() as run:  # pauses at the gate
        _ = [event async for event in run]
    assert runner.awaiting_approval()
    async with runner.run() as run:  # NO exception: re-asks
        resume_events = [event async for event in run]

    assert [event.type for event in resume_events] == [
        "tool_execution_started", "tool_executed", "text_block", "finish_reason",
    ]
    assert resume_events[1].result_text == "3"
    assert runner.session.entries["te1"].approval_status == ApprovalStatus.ALLOWED
    assert runner.session.entries["te1"].approval_decisions == [
        PENDING_1000, ALLOW_1000,
    ]
    # asked twice, each time with the then-current snapshot
    assert [list(ex.approval_decisions) for ex in registry.seen] == [
        [], [PENDING_1000],
    ]
    assert runner.idle()


async def test_allowed_sibling_dispatches_before_the_run_parks():
    # A PENDING decision defers only that execution: the ALLOWED sibling runs
    # to completion first, ApprovalRequired is the FINAL event, and the model
    # is not called until every call has a terminal execution.
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1"),
             faux_tool_call("multiply", {"a": 3, "b": 4}, id="tc2")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("done")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_partial",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry(
        [AddTool(), MultiplyTool()], decisions=[ALLOW_1000, PENDING_1000],
    )
    runner = DeterministicRunner(
        session, tool_registry=registry,
        provider=faux, ids=["ts", "a1", "te1", "te2", "a2", "tf"], now=1000,
    )

    async with runner.run() as run:
        first_events = [event async for event in run]

    # the allowed sibling ran; only the deferred call holds the turn open
    assert [(event.type, getattr(event, "tool_call_id", None)) for event in first_events] == [
        ("finish_reason", None),
        ("tool_call_received", "tc1"),
        ("tool_call_received", "tc2"),
        ("tool_execution_started", "tc1"),
        ("tool_executed", "tc1"),
        ("approval_required", None),
    ]
    assert runner.awaiting_approval()
    assert first_events[-1] == ApprovalRequired(executions=[
        ToolExecution(
            id="te2", parent_id="te1", created_at=1000,
            tool_call_id="tc2",
            raw_tool_call=ToolCall(id="tc2", name="multiply", arguments={"a": 3, "b": 4}),
            tool_spec=MULTIPLY_SPEC,
            status=ExecutionStatus.PENDING,
            approval_status=ApprovalStatus.PENDING,
            approval_decisions=[PENDING_1000],
            updated_at=1000,
        ),
    ])
    assert runner.pending_approvals() == [runner.session.entries["te2"]]
    assert runner.session.entries["te1"].status == ExecutionStatus.COMPLETED
    assert len(faux.requests) == 1  # no second model call while te2 is open

    registry.decisions.append(DENY_1000)
    async with runner.run() as run:
        events = [event async for event in run]

    assert [event.type for event in events] == [
        "tool_executed", "text_block", "finish_reason",
    ]
    assert events[0].tool_call_id == "tc2"
    assert events[0].result_text == "[tool execution rejected]"
    # te1 was decided ONCE (never re-asked once resolved); te2 twice
    assert [ex.id for ex in registry.seen] == ["te1", "te2", "te2"]
    assert runner.session.entries["te2"].approval_status == ApprovalStatus.REJECTED
    assert runner.session.entries["te2"].approval_decisions == [
        PENDING_1000, DENY_1000,
    ]
    assert runner.idle()


# ── cold resume: persisted mid-state sessions loaded into a fresh runner ────────


async def test_loaded_gated_session_exposes_pending_approvals():
    session = GATED_SESSION.model_copy(deep=True)

    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()], decisions=[]),
        provider=FauxProvider(), now=1000,
    )

    assert runner.awaiting_approval()
    assert runner.pending_approvals() == [session.entries["te1"]]


async def test_loaded_gated_session_run_reasks_strategy_and_completes():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = GATED_SESSION.model_copy(deep=True)
    registry = FakeToolRegistry([AddTool()], decisions=[ALLOW_1000])
    runner = DeterministicRunner(
        session, tool_registry=registry, provider=faux,
        ids=["a2", "tf"], now=1000,
    )

    async with runner.run() as run:  # NO exception: re-asks
        events = [event async for event in run]

    assert [event.type for event in events] == [
        "tool_execution_started", "tool_executed", "text_block", "finish_reason",
    ]
    assert runner.session == AgentSession(
        id="s_gated",
        entries={
            "u1": UserMessage(
                id="u1", created_at=500, parts=[TextContent(text="Add 1 and 2")],
            ),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
            "a1": AssistantMessage(
                id="a1", parent_id="ts", created_at=500,
                parts=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
                llm_config=MODEL, stop_reason="tool_use",
            ),
            "te1": ToolExecution(
                id="te1", parent_id="a1", created_at=500,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
                tool_spec=ADD_SPEC,
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(
                    content=[TextContent(text="3")], is_error=False,
                ),
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[
                    ApprovalDecision(
                        decision=ApprovalOption.PENDING, created_at=500,
                    ),
                    ALLOW_1000,
                ],
                started_at=1000, ended_at=1000,
                updated_at=1000,
            ),
            "a2": AssistantMessage(
                id="a2", parent_id="te1", created_at=1000,
                parts=[TextContent(text="It's 3.")],
                llm_config=MODEL, stop_reason="stop",
                context_tokens=1,  # len("It's 3.") // 4
            ),
            "tf": TurnFinish(id="tf", parent_id="a2", created_at=1000),
        },
        tool_executions={"tc1": ["te1"]},
        usages={"c1": {"a2": Usage(conversation_id="c1", entry_id="a2")}},
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts", "a1", "te1", "a2", "tf"],
            created_at=500, updated_at=1000, status=ConversationStatus.IDLE,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )


async def test_cleared_execution_dispatches_before_any_llm_call_without_redeciding():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = CLEARED_SESSION.model_copy(deep=True)
    # an empty decision script: any decide() would raise
    registry = FakeToolRegistry([AddTool()], decisions=[])
    runner = DeterministicRunner(
        session, tool_registry=registry, provider=faux,
        ids=["a2", "tf"], now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    assert [event.type for event in events] == [
        "tool_execution_started", "tool_executed", "text_block", "finish_reason",
    ]
    # a resolved call is NEVER re-decided
    assert registry.seen == []
    # exactly one LLM call, made only after the cleared call executed: the
    # request already carries its tool result
    assert len(faux.requests) == 1
    assert faux.requests[0].messages[-1] == ToolMessage(
        tool_call_id="tc1", content=[LucaTextBlock(text="3")], is_error=False,
    )
    assert runner.session.entries["te1"].status == ExecutionStatus.COMPLETED
    assert runner.idle()


async def test_undecided_session_self_heals_and_run_asks_strategy():
    # crash mid-decide: execution persisted, approval_status None — NOT
    # awaiting approval (the strategy was never asked); a plain run() asks it.
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = UNDECIDED_SESSION.model_copy(deep=True)
    registry = FakeToolRegistry([AddTool()], decisions=[ALLOW_1000])
    runner = DeterministicRunner(
        session, tool_registry=registry, provider=faux,
        ids=["a2", "tf"], now=1000,
    )

    assert runner.pending()  # stale RUNNING self-healed; not AWAITING_APPROVAL
    async with runner.run() as run:
        events = [event async for event in run]

    assert [event.type for event in events] == [
        "tool_execution_started", "tool_executed", "text_block", "finish_reason",
    ]
    assert runner.session.entries["te1"].approval_status == ApprovalStatus.ALLOWED
    assert runner.session.entries["te1"].approval_decisions == [ALLOW_1000]
    assert runner.idle()


async def test_stale_running_status_self_heals_on_construction():
    session = STALE_RUNNING_SESSION.model_copy(deep=True)
    assert session.active_conversation.status == ConversationStatus.RUNNING

    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()], decisions=[]),
        provider=FauxProvider(), now=1000,
    )

    assert runner.pending()  # the open turn with a cleared call means: call run()


# ── decide() failure ─────────────────────────────────────────────────────────


async def test_strategy_exception_propagates_and_session_stays_resumable():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_boom_policy",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=ExplodingRegistry([AddTool()]),
        provider=faux, ids=["ts", "a1", "te1"], now=1000,
    )

    with pytest.raises(RuntimeError, match="strategy down"):
        async with runner.run() as run:
            _ = [event async for event in run]

    # the execution was persisted eagerly and stays unprocessed...
    assert runner.session.entries["te1"].approval_status is None
    assert runner.session.entries["te1"].approval_decisions == []
    assert runner.session.entries["te1"].status == ExecutionStatus.PENDING

    # ...so a fresh runner (same session) with a working registry completes it
    resumed = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()], decisions=[ALLOW_1000]),
        provider=faux, ids=["a2", "tf"], now=1000,
    )
    assert resumed.pending()
    async with resumed.run() as run:
        events = [event async for event in run]
    assert events[1].type == "tool_executed"
    assert events[1].result_text == "3"
    assert resumed.idle()



# ── durability: the gate survives a full serialize / reload cycle ───────────────


async def test_gated_session_survives_restart_and_resumes():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_restart",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()], decisions=[PENDING_1000]),
        provider=faux, ids=["ts", "a1", "te1"], now=1000,
    )
    async with runner.run() as run:  # pauses at the gate
        _ = [event async for event in run]
    payload = runner.session.model_dump_json()  # "process exits" here

    # restart: reload the session into a fresh runner with a fresh registry
    reloaded = AgentSession.model_validate_json(payload)
    resumed = DeterministicRunner(
        reloaded,
        tool_registry=FakeToolRegistry([AddTool()], decisions=[
            ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=2000),
        ]),
        provider=faux, ids=["a2", "tf"], now=2000,
    )

    assert resumed.awaiting_approval()
    assert resumed.pending_approvals() == [reloaded.entries["te1"]]
    async with resumed.run() as run:
        events = [event async for event in run]

    assert [event.type for event in events] == [
        "tool_execution_started", "tool_executed", "text_block", "finish_reason",
    ]
    assert resumed.idle()
    assert resumed.session.entries["te1"].approval_status == ApprovalStatus.ALLOWED
    assert resumed.session.entries["te1"].approval_decisions == [
        PENDING_1000,
        ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=2000),
    ]
    assert resumed.session.entries["tf"] == TurnFinish(
        id="tf", parent_id="a2", created_at=2000,
    )
