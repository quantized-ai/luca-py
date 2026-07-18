"""Cancellation scenarios: the universal cancel() door, the durable
CancelRequested entry, the step-boundary wind-down, token races with grace,
and the parked-cancel flush.

Cancellation facts under test on the ToolExecution record: run cancellation
stamps `cancel_signalled_at` on every affected nonterminal execution (and on
an in-flight one BEFORE its grace window runs); a PENDING execution becomes
CANCELLED (resultless, errorless, approval state untouched); a RUNNING one
settles by the grace machinery — COMPLETED with its real result if it returns
in time (keeping `cancel_signalled_at`), FAILED if it raises, INTERRUPTED if
grace expires; an already-terminal execution is unaffected.

House style throughout: precondition → one action → full-object
postcondition, and NEVER race two timed things — one side of every race
hangs forever on an event nobody sets, the other is instant.
"""

import asyncio

import pytest

from luca.agent.core.context import CancellationToken, ToolContext
from luca.agent.core.models import (
    RuntimeConfig,
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
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    ToolSpec,
    TurnFinish,
    TurnOutcome,
    Usage,
    UserMessage,
)
from luca.agent.core.events import (
    FinishReason,
    ReasoningDelta,
    ReasoningStart,
    TextBlock,
    ToolCallReceived,
    ToolExecuted,
    ToolExecutionStarted,
)
from luca.agent.core.exceptions import AlreadyCancellingError
from luca.agent.core.runner import RunResult
from luca.agent.core.tools import Tool
from luca.client.exceptions import ProviderAPIError
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_error,
    faux_hang,
    faux_text,
    faux_thinking,
    faux_tool_call,
)

from tests.agent.scenarios import (
    CANCEL_PARKED_SESSION,
    GATED_SESSION,
    MODEL,
    AddTool,
    BinaryArgs,
    DeterministicRunner,
    FakeToolRegistry,
)

ALLOW_1000 = ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)
PENDING_1000 = ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)

ADD_SPEC = ToolSpec(name="add", description="Add two numbers.")


# ── tool doubles ───────────────────────────────────────────────────────────────


class HangingTool(Tool):
    """Hangs forever on an event nobody sets; records the hard cancel."""

    name = "hang"
    description = "Hangs until hard-cancelled."
    Args = BinaryArgs

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.hard_cancelled = False

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.hard_cancelled = True
            raise
        return "unreachable"


class CooperativeTool(Tool):
    """Returns a sentinel the instant the run's token trips."""

    name = "cooperative"
    description = "Returns partial output on cancellation."
    Args = BinaryArgs

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        self.started.set()
        await cancellation_token.wait_cancelled()
        return "partial sum"


class TimeoutRaisingTool(Tool):
    """Raises builtin TimeoutError the instant the run's token trips — the
    tool's OWN failure must not be mistaken for the grace deadline."""

    name = "deadline_confused"
    description = "Raises TimeoutError when cancelled."
    Args = BinaryArgs

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        self.started.set()
        await cancellation_token.wait_cancelled()
        raise TimeoutError("tool's own deadline")


class ReleasableProvider(FauxProvider):
    """FauxProvider whose acompletion parks until the test releases it — the
    deterministic stand-in for "the model finishes within the grace window"."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def acompletion(self, request):
        self.started.set()
        await self.release.wait()
        return await super().acompletion(request)


class CancellingRegistry(FakeToolRegistry):
    """Stands in for a cancel arriving while decide() deliberates: it
    requests the cancel itself, then punts. The test wires `runner` after
    construction."""

    runner = None

    async def decide(
        self, tool_execution: ToolExecution, context: ToolContext,
    ) -> ApprovalDecision:
        self.runner.cancel()
        return ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)


# ── cancel(): the no-op and diagnostic branches ───────────────────────────────


async def test_cancel_on_idle_session_is_a_noop():
    session = AgentSession(
        id="s_idle",
        active_conversation=Conversation(id="c1", nodes=[], created_at=900, updated_at=900),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, now=1000)

    runner.cancel()

    assert runner.idle()
    assert runner.session.entries == {}


async def test_cancel_on_fresh_pending_is_a_noop_and_the_turn_runs_normally():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_fresh",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "a1", "tf"], now=1000,
    )

    runner.cancel()  # no open turn yet — cancel targets the turn, not the handle

    assert runner.pending()
    assert runner.session.active_conversation.nodes == ["u1"]
    result = await runner.run()
    assert result.outcome == TurnOutcome.COMPLETED


async def test_second_cancel_raises_and_the_first_call_wins():
    session = GATED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()], decisions=[]),
        provider=FauxProvider(), ids=["cr"], now=1000,
    )
    runner.cancel(error="first")

    with pytest.raises(AlreadyCancellingError):
        runner.cancel(error="second")

    assert runner.cancelling()
    assert runner.session.entries["cr"] == CancelRequested(
        id="cr", parent_id="te1", created_at=1000,
        outcome=TurnOutcome.CANCELLED, error="first",
    )


# ── abandon at the gate + the flush drive ─────────────────────────────────────


async def test_cancel_at_the_gate_parks_and_the_next_run_flushes():
    faux = FauxProvider()
    session = GATED_SESSION.model_copy(deep=True)
    fake = FakeToolRegistry([AddTool()], decisions=[])  # empty decide script
    runner = DeterministicRunner(
        session, tool_registry=fake, provider=faux,
        ids=["cr", "tf"], now=1000,
    )

    runner.cancel()  # abandon the turn — not DENY: deny would feed the model

    assert runner.cancelling()
    assert runner.session.active_conversation.nodes == [
        "u1", "ts", "a1", "te1", "cr",
    ]

    async with runner.run() as run:  # the flush: no decide(), no LLM call
        events = [event async for event in run]

    cancelled = ToolExecution(
        id="te1", parent_id="a1", created_at=500,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        status=ExecutionStatus.CANCELLED,
        result=None,
        approval_status=ApprovalStatus.PENDING,  # untouched by the wind-down
        approval_decisions=[
            ApprovalDecision(decision=ApprovalOption.PENDING, created_at=500),
        ],
        cancel_signalled_at=1000,
        ended_at=1000,
        updated_at=1000,
    )
    assert events == [
        ToolExecuted(
            tool_call_id="tc1", execution=cancelled,
            result_text="[tool execution cancelled]", is_error=True,
        ),
    ]
    assert fake.seen == []
    assert faux.requests == []
    assert runner.session.entries["te1"] == cancelled
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf", parent_id="cr", created_at=1000,
        outcome=TurnOutcome.CANCELLED,
    )
    assert runner.idle()


async def test_flush_via_await_returns_a_cancelled_result():
    session = CANCEL_PARKED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()], decisions=[]),
        provider=FauxProvider(), ids=["tf"], now=1000,
    )

    assert runner.cancelling()
    result = await runner.run()

    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.CANCELLED,
        pending_approvals=[],
    )
    assert runner.idle()


async def test_parked_cancel_survives_save_and_cold_reload():
    session = GATED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()], decisions=[]),
        provider=FauxProvider(), ids=["cr"], now=1000,
    )
    runner.cancel(error="user abandoned the turn")
    payload = runner.session.model_dump_json()  # "process exits" here

    reloaded = AgentSession.model_validate_json(payload)
    resumed = DeterministicRunner(
        reloaded, tool_registry=FakeToolRegistry([AddTool()], decisions=[]),
        provider=FauxProvider(), ids=["tf"], now=2000,
    )

    assert resumed.cancelling()  # derive_status found the unconsumed entry
    result = await resumed.run()

    assert result.outcome == TurnOutcome.CANCELLED
    assert resumed.session.entries["tf"] == TurnFinish(
        id="tf", parent_id="cr", created_at=2000,
        outcome=TurnOutcome.CANCELLED, error="user abandoned the turn",
    )
    assert resumed.idle()


async def test_cancel_after_the_flush_completed_is_a_noop():
    session = CANCEL_PARKED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()], decisions=[]),
        provider=FauxProvider(), ids=["tf"], now=1000,
    )
    await runner.run()  # the flush
    nodes_after_flush = list(runner.session.active_conversation.nodes)

    runner.cancel()  # IDLE again — branch 3

    assert runner.idle()
    assert runner.session.active_conversation.nodes == nodes_after_flush


async def test_flush_leaves_an_already_terminal_execution_untouched():
    # the wind-down cancels only nonterminal executions: a call the policy
    # already denied (terminal REJECTED at decision time) is unaffected and
    # emits no second ToolExecuted.
    session = GATED_SESSION.model_copy(deep=True)
    session.entries["te1"] = session.entries["te1"].model_copy(update={
        "status": ExecutionStatus.REJECTED,
        "approval_status": ApprovalStatus.REJECTED,
        "approval_decisions": [
            ApprovalDecision(decision=ApprovalOption.PENDING, created_at=500),
            ApprovalDecision(decision=ApprovalOption.DENY, created_at=600),
        ],
        "ended_at": 600,
        "updated_at": 600,
    })
    session.entries["cr"] = CancelRequested(id="cr", parent_id="te1", created_at=600)
    session.active_conversation.nodes.append("cr")
    session.active_conversation.status = ConversationStatus.CANCELLING
    rejected_before_flush = session.entries["te1"].model_copy(deep=True)
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()], decisions=[]),
        provider=FauxProvider(), ids=["tf"], now=1000,
    )

    async with runner.run() as run:  # the flush
        events = [event async for event in run]

    assert events == []
    assert runner.session.entries["te1"] == rejected_before_flush
    assert runner.session.entries["tf"].outcome == TurnOutcome.CANCELLED
    assert runner.idle()


# ── live cancel: token races, grace, hard cancel ──────────────────────────────


async def test_live_cancel_hard_cancels_the_hanging_tool():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("hang", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
    ])
    tool = HangingTool()
    session = AgentSession(
        id="s_live",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([tool]), provider=faux,
        ids=["ts", "a1", "te1", "cr", "tf"], now=1000,
    )
    run = runner.start()
    await tool.started.wait()  # the tool is in-flight

    run.cancel()  # the handle delegates to runner.cancel()
    result = await run

    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.CANCELLED,
        pending_approvals=[],
    )
    assert tool.hard_cancelled is True  # grace 0 → straight to hard cancel
    interrupted = ToolExecution(
        id="te1", parent_id="a1", created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="hang", arguments={"a": 1, "b": 2}),
        tool_spec=ToolSpec(name="hang", description="Hangs until hard-cancelled."),
        status=ExecutionStatus.INTERRUPTED,
        result=None,
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[ALLOW_1000],
        started_at=1000,
        ended_at=1000,
        cancel_signalled_at=1000,  # stamped and persisted before the grace ran
        updated_at=1000,
    )
    assert runner.session.entries["te1"] == interrupted
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf", parent_id="cr", created_at=1000,
        outcome=TurnOutcome.CANCELLED,
    )
    async with run:
        events = [event async for event in run]
    assert [event.type for event in events] == [
        "finish_reason", "tool_call_received", "tool_execution_started",
        "tool_executed",
    ]
    assert events[3] == ToolExecuted(
        tool_call_id="tc1", execution=interrupted,
        result_text="[tool execution interrupted]", is_error=True,
    )


async def test_cooperative_tool_finishing_within_grace_records_its_real_result():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("cooperative", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
    ])
    tool = CooperativeTool()
    session = AgentSession(
        id="s_grace",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(
            llm_config=MODEL,
            # huge grace — never expires; the tool returns the instant the
            # token trips, so nothing here depends on real time
            runtime_config=RuntimeConfig(tool_cancellation_grace_period=30_000),
        ),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([tool]), provider=faux,
        ids=["ts", "a1", "te1", "cr", "tf"], now=1000,
    )
    run = runner.start()
    await tool.started.wait()

    runner.cancel()
    result = await run

    assert result.outcome == TurnOutcome.CANCELLED
    # it RETURNED — a real COMPLETED result, keeping cancel_signalled_at
    assert runner.session.entries["te1"] == ToolExecution(
        id="te1", parent_id="a1", created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="cooperative", arguments={"a": 1, "b": 2}),
        tool_spec=ToolSpec(
            name="cooperative", description="Returns partial output on cancellation.",
        ),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(
            content=[TextContent(text="partial sum")], is_error=False,
        ),
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[ALLOW_1000],
        started_at=1000,
        ended_at=1000,
        cancel_signalled_at=1000,
        updated_at=1000,
        context_tokens=2,  # len("partial sum") // 4
    )
    assert runner.session.entries["tf"].outcome == TurnOutcome.CANCELLED
    async with run:
        events = [event async for event in run]
    assert events[3].type == "tool_executed"
    assert events[3].result_text == "partial sum"
    assert events[3].is_error is False


async def test_pre_start_sibling_is_cancelled_when_the_first_is_interrupted():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("hang", {"a": 1, "b": 2}, id="tc1"),
             faux_tool_call("add", {"a": 3, "b": 4}, id="tc2")],
            finish_reason="tool_use",
        ),
    ])
    tool = HangingTool()
    session = AgentSession(
        id="s_sibling",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([tool, AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "te2", "cr", "tf"], now=1000,
    )
    run = runner.start()
    await tool.started.wait()  # tc1 in-flight; tc2 not yet started

    runner.cancel()
    result = await run

    assert result.outcome == TurnOutcome.CANCELLED
    assert runner.session.entries["te1"].status == ExecutionStatus.INTERRUPTED
    assert runner.session.entries["te1"].started_at == 1000
    assert runner.session.entries["te1"].cancel_signalled_at == 1000
    assert runner.session.entries["te2"].status == ExecutionStatus.CANCELLED
    assert runner.session.entries["te2"].started_at is None
    assert runner.session.entries["te2"].cancel_signalled_at == 1000
    assert runner.session.entries["te2"].result is None
    assert runner.session.entries["te2"].error is None
    async with run:
        events = [event async for event in run]
    assert [(event.type, getattr(event, "tool_call_id", None)) for event in events] == [
        ("finish_reason", None),
        ("tool_call_received", "tc1"),
        ("tool_call_received", "tc2"),
        ("tool_execution_started", "tc1"),
        ("tool_executed", "tc1"),
        ("tool_executed", "tc2"),
    ]
    assert events[4].result_text == "[tool execution interrupted]"
    assert events[5].result_text == "[tool execution cancelled]"


async def test_lazy_cancel_between_events_cancels_the_unstarted_execution():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
    ])
    session = AgentSession(
        id="s_lazy_cancel",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "cr", "tf"], now=1000,
    )

    async with runner.run() as run:
        events = []
        async for event in run:
            events.append(event)
            if event.type == "tool_call_received":
                run.cancel()  # before the policy or the batch ever sees it

    assert [event.type for event in events] == [
        "finish_reason", "tool_call_received", "tool_executed",
    ]
    assert events[2].result_text == "[tool execution cancelled]"
    execution = runner.session.entries["te1"]
    assert execution.status == ExecutionStatus.CANCELLED
    assert execution.approval_status is None  # the policy never processed it
    assert execution.approval_decisions == []
    assert execution.cancel_signalled_at == 1000
    assert runner.session.entries["tf"].outcome == TurnOutcome.CANCELLED
    assert runner.idle()


async def test_cancel_landing_mid_decide_winds_down_instead_of_pausing():
    # decide() is never cancelled; the pre-park check winds down BEFORE the
    # approval pause — no ApprovalRequired, the gate never opens.
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
    ])
    registry = CancellingRegistry([AddTool()])
    session = AgentSession(
        id="s_mid_decide",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=registry, provider=faux,
        ids=["ts", "a1", "te1", "cr", "tf"], now=1000,
    )
    registry.runner = runner

    async with runner.run() as run:
        events = [event async for event in run]

    assert [event.type for event in events] == [
        "finish_reason", "tool_call_received", "tool_executed",
    ]
    assert events[2].result_text == "[tool execution cancelled]"
    execution = runner.session.entries["te1"]
    assert execution.status == ExecutionStatus.CANCELLED
    # the deferral WAS processed before the cancel consumed the turn
    assert execution.approval_status == ApprovalStatus.PENDING
    assert execution.approval_decisions == [PENDING_1000]
    assert runner.session.entries["tf"].outcome == TurnOutcome.CANCELLED
    assert runner.idle()


async def test_mid_stream_cancel_drops_the_partial_assistant_message():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_thinking("Thinking…"), faux_hang()], finish_reason="stop",
        ),
    ])
    session = AgentSession(
        id="s_stream_cancel",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "cr", "tf"], now=1000,
    )
    run = runner.start(streaming=True)

    async with run:
        events = []
        async for event in run:
            events.append(event)
            if event == ReasoningDelta(text="Thinking…"):
                run.cancel()  # the stream is parked mid-generation

    assert events == [ReasoningStart(), ReasoningDelta(text="Thinking…")]
    assert run.result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.CANCELLED,
        pending_approvals=[],
    )
    # the partial assistant message was dropped — only bookkeeping landed
    assert runner.session.active_conversation.nodes == ["u1", "ts", "cr", "tf"]
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf", parent_id="cr", created_at=1000,
        outcome=TurnOutcome.CANCELLED,
    )
    assert runner.idle()


# ── cancel vs the grace window at the close sites ─────────────────────────────


async def test_llm_answer_within_grace_is_recorded_but_the_cancel_still_wins():
    faux = ReleasableProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Here you go.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_llm_grace",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(
            llm_config=MODEL,
            # huge grace — never expires; the faux answers the instant the
            # test releases it, so nothing here depends on real time
            runtime_config=RuntimeConfig(llm_completion_cancellation_grace_period=30_000),
        ),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "cr", "a1", "tf"], now=1000,
    )
    run = runner.start()
    await faux.started.wait()  # the LLM call is in flight

    runner.cancel()
    faux.release.set()  # the model answers within the grace window
    result = await run

    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.CANCELLED,
        pending_approvals=[],
    )
    # the within-grace answer is recorded ...
    assert runner.session.entries["a1"] == AssistantMessage(
        id="a1", parent_id="cr", created_at=1000,
        parts=[TextContent(text="Here you go.")],
        llm_config=MODEL, stop_reason="stop",
        context_tokens=3,  # len("Here you go.") // 4
    )
    # ... but the requested outcome controls the close
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf", parent_id="a1", created_at=1000,
        outcome=TurnOutcome.CANCELLED,
    )
    assert runner.session.active_conversation.nodes == [
        "u1", "ts", "cr", "a1", "tf",
    ]
    assert runner.idle()
    async with run:
        events = [event async for event in run]
    assert events == [
        TextBlock(text="Here you go."), FinishReason(finish_reason="stop"),
    ]


async def test_llm_failure_within_grace_closes_cancelled_and_returns_normally():
    faux = ReleasableProvider()
    faux.set_responses([
        faux_assistant_message(
            [], error=faux_error("provider 500", error_class=ProviderAPIError),
        ),
    ])
    session = AgentSession(
        id="s_llm_grace_fail",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(
            llm_config=MODEL,
            runtime_config=RuntimeConfig(llm_completion_cancellation_grace_period=30_000),
        ),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "cr", "tf"], now=1000,
    )
    run = runner.start()
    await faux.started.wait()

    runner.cancel(error="user abandoned the turn")
    faux.release.set()  # the call FAILS within the grace window
    result = await run  # no raise — the cancel controls this close too

    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.CANCELLED,
        pending_approvals=[],
    )
    # the failure was discarded: no assistant message, no failed TurnFinish
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf", parent_id="cr", created_at=1000,
        outcome=TurnOutcome.CANCELLED, error="user abandoned the turn",
    )
    assert runner.session.active_conversation.nodes == ["u1", "ts", "cr", "tf"]
    assert runner.idle()


async def test_eager_cancel_before_the_first_tick_flushes_without_an_llm_call():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_instant_cancel",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "cr", "tf"], now=1000,
    )
    run = runner.start()  # start() opened the bracket durably

    run.cancel()  # before the background task's first tick
    result = await run

    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.CANCELLED,
        pending_approvals=[],
    )
    assert faux.requests == []  # the flush never reached the model
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf", parent_id="cr", created_at=1000,
        outcome=TurnOutcome.CANCELLED,
    )
    assert runner.session.active_conversation.nodes == ["u1", "ts", "cr", "tf"]
    assert runner.idle()


async def test_tool_raising_timeout_error_within_grace_records_failed():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("deadline_confused", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
    ])
    tool = TimeoutRaisingTool()
    session = AgentSession(
        id="s_tool_timeout_grace",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(
            llm_config=MODEL,
            runtime_config=RuntimeConfig(tool_cancellation_grace_period=30_000),
        ),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([tool]), provider=faux,
        ids=["ts", "a1", "te1", "cr", "tf"], now=1000,
    )
    run = runner.start()
    await tool.started.wait()

    runner.cancel()  # the tool wakes and raises its OWN TimeoutError
    result = await run

    assert result.outcome == TurnOutcome.CANCELLED
    # the tool FAILED — it was not interrupted by the grace machinery
    execution = runner.session.entries["te1"]
    assert execution.status == ExecutionStatus.FAILED
    assert execution.result is None
    assert execution.error == ToolExecutionError(
        error_type="TimeoutError",
        error_message="tool's own deadline",
        details={"phase": "execution"},
    )
    assert execution.cancel_signalled_at == 1000
    assert runner.session.entries["tf"].outcome == TurnOutcome.CANCELLED
