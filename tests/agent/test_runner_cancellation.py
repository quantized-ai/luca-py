"""Cancellation scenarios: the universal cancel() door, the durable
CancelRequested entry, the step-boundary wind-down, the four registry races,
token races with grace, and the parked-cancel flush.

Cancellation facts under test on the ToolExecution record: run cancellation
stamps `cancel_signalled_at` on every affected nonterminal execution (and on
an in-flight one BEFORE its grace window runs); a PENDING execution becomes
CANCELLED (resultless, errorless, approval state untouched); a RUNNING one
settles by the grace machinery — COMPLETED with its real result if it returns
in time (keeping `cancel_signalled_at`), FAILED if it raises, INTERRUPTED if
grace expires; an already-terminal execution is unaffected.

The four registry awaits (`get_tools`, `create_execution`, `decide`,
`prepare`) are raced with ZERO grace and awaited unwinding, so no registry can
make cancel() a no-op: each one, hung forever, is unblocked by cancel() and
closes the turn with the requested outcome. A cancelled birth records
CANCELLED (never FAILED — a cancellation is not a tool failure), and a
cancellation up to AND INCLUDING `prepare()` settling means the body is never
dispatched: `started_at` stays None, `dispatched` stays False, and no
`ToolExecutionStarted` is emitted.

House style throughout: precondition → one action → full-object
postcondition, and NEVER race two timed things — one side of every race
hangs forever on an event nobody sets, the other is instant.
"""

import asyncio

import pytest

from luca.agent.core.context import CancellationToken
from luca.agent.core.events import (
    FinishReason,
    ReasoningDelta,
    ReasoningStart,
    TextBlock,
    ToolCallReceived,
    ToolExecuted,
)
from luca.agent.core.exceptions import AlreadyCancellingError
from luca.agent.core.models import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    CancelRequested,
    ConversationStatus,
    ExecutionResult,
    ExecutionStatus,
    RuntimeConfig,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    UserMessage,
)
from luca.agent.core.runner import RunResult
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
    ADD_SPEC,
    CANCEL_PARKED_SESSION,
    CLEARED_SESSION,
    GATED_SESSION,
    MODEL,
    MULTIPLY_SPEC,
    UNDECIDED_SESSION,
    AddTool,
    BinaryArgs,
    DeterministicRunner,
    FakeTool,
    FakeToolRegistry,
    MultiplyTool,
    conversation,
    main_conversation,
    make_session,
)

ALLOW_600 = ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=600)
ALLOW_1000 = ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)
PENDING_500 = ApprovalDecision(decision=ApprovalOption.PENDING, created_at=500)
PENDING_1000 = ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)

ADD_CALL = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})


# ── tool doubles ───────────────────────────────────────────────────────────────


class HangingTool(FakeTool):
    """Hangs forever on an event nobody sets; records the hard cancel."""

    name = "hang"
    description = "Hangs until hard-cancelled."
    Args = BinaryArgs

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.hard_cancelled = False

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.hard_cancelled = True
            raise
        return "unreachable"


class CooperativeTool(FakeTool):
    """Returns a sentinel the instant the run's token trips."""

    name = "cooperative"
    description = "Returns partial output on cancellation."
    Args = BinaryArgs

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        self.started.set()
        await cancellation_token.wait_cancelled()
        return "partial sum"


class TimeoutRaisingTool(FakeTool):
    """Raises builtin TimeoutError the instant the run's token trips — the
    tool's OWN failure must not be mistaken for the grace deadline."""

    name = "deadline_confused"
    description = "Raises TimeoutError when cancelled."
    Args = BinaryArgs

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        self.started.set()
        await cancellation_token.wait_cancelled()
        raise TimeoutError("tool's own deadline")


class RecordingAddTool(AddTool):
    """`add`, recording every invocation of its BODY — the witness for "the
    tool never ran". Same spec as `AddTool` (name, description and `Args` are
    inherited), so a precondition built on `ADD_SPEC` still matches."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        self.calls.append(args)
        return str(args["a"] + args["b"])


HANG_SPEC = HangingTool().get_tool_spec()
COOPERATIVE_SPEC = CooperativeTool().get_tool_spec()
DEADLINE_SPEC = TimeoutRaisingTool().get_tool_spec()


# ── registry doubles ───────────────────────────────────────────────────────────


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


class CancellingDecideRegistry(FakeToolRegistry):
    """Stands in for a cancel arriving while decide() deliberates: it
    requests the cancel itself, then punts. The test wires `runner` after
    construction."""

    runner = None

    async def decide(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> ApprovalDecision:
        self.seen.append(tool_execution)
        self.runner.cancel()
        return ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)


class HangingGetToolsRegistry(FakeToolRegistry):
    """`get_tools` parks forever on an event nobody sets."""

    def __init__(self, tools=()) -> None:
        super().__init__(tools)
        self.started = asyncio.Event()

    async def get_tools(self, session: AgentSession, conversation_id: str):
        self.started.set()
        await asyncio.Event().wait()


class HangingBirthRegistry(FakeToolRegistry):
    """`create_execution` parks forever on an event nobody sets."""

    def __init__(self, tools=()) -> None:
        super().__init__(tools)
        self.started = asyncio.Event()

    async def create_execution(self, session: AgentSession, conversation_id: str, call: ToolCall):
        self.started.set()
        await asyncio.Event().wait()


class HangingDecideRegistry(FakeToolRegistry):
    """`decide` parks forever on an event nobody sets."""

    def __init__(self, tools=()) -> None:
        super().__init__(tools)
        self.started = asyncio.Event()

    async def decide(self, session: AgentSession, conversation_id: str, tool_execution: ToolExecution):
        self.seen.append(tool_execution)
        self.started.set()
        await asyncio.Event().wait()


class HangingPrepareRegistry(FakeToolRegistry):
    """`prepare` parks forever on an event nobody sets."""

    def __init__(self, tools=()) -> None:
        super().__init__(tools)
        self.started = asyncio.Event()

    async def prepare(self, session: AgentSession, conversation_id: str, tool_execution: ToolExecution):
        self.started.set()
        await asyncio.Event().wait()


class CancellingPrepareRegistry(FakeToolRegistry):
    """Stands in for a cancel landing in the window between `prepare()`
    returning and the body being dispatched: it requests the cancel itself,
    then hands back a perfectly good callable. The test wires `runner` after
    construction."""

    runner = None

    async def prepare(self, session: AgentSession, conversation_id: str, tool_execution: ToolExecution):
        self.runner.cancel()
        return await super().prepare(session, conversation_id, tool_execution)


class CancelOnFirstBirthRegistry(FakeToolRegistry):
    """The `add` birth requests the cancel and still returns its draft; every
    other birth parks forever, so the runner has to synthesize one for it.
    The test wires `runner` after construction."""

    runner = None

    async def create_execution(self, session: AgentSession, conversation_id: str, call: ToolCall):
        if call.name != "add":
            await asyncio.Event().wait()  # parks forever — never returns
        self.runner.cancel()
        return await super().create_execution(session, conversation_id, call)


class TracedResource:
    """An async context manager writing its acquire/release into a shared
    trace — the witness that a cancelled registry call unwinds completely."""

    def __init__(self, trace: list) -> None:
        self.trace = trace

    async def __aenter__(self) -> "TracedResource":
        self.trace.append(("acquired", None))
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self.trace.append(("released", None))
        return False


class ResourceHoldingRegistry(FakeToolRegistry):
    """`create_execution` takes a resource in `async with`, then parks inside
    the block — the shape rule 6 of the registry contract describes."""

    def __init__(self, tools=(), trace: list | None = None) -> None:
        super().__init__(tools)
        self.trace = trace if trace is not None else []
        self.started = asyncio.Event()

    async def create_execution(self, session: AgentSession, conversation_id: str, call: ToolCall):
        async with TracedResource(self.trace):
            self.started.set()
            await asyncio.Event().wait()


class HookRecorder:
    """Middleware recording which tool/permission lifecycle hooks fired, in
    order, as `(hook, entry_id)` — the witness for "fired exactly once per
    execution" and "never fired at all"."""

    def __init__(self, trace: list | None = None) -> None:
        self.trace = trace if trace is not None else []

    def before_permission_check(self, session, conversation_id, execution: ToolExecution) -> ToolExecution:
        self.trace.append(("before_permission_check", execution.id))
        return execution

    def after_permission_decision(
        self,
        session,
        conversation_id,
        decision: ApprovalDecision,
        execution: ToolExecution,
    ) -> ApprovalDecision:
        self.trace.append(("after_permission_decision", execution.id))
        return decision

    def before_tool_execution(self, session, conversation_id, execution: ToolExecution) -> ToolExecution:
        self.trace.append(("before_tool_execution", execution.id))
        return execution

    def after_tool_execution(
        self,
        session,
        conversation_id,
        execution: ToolExecution,
        exception: Exception | None = None,
    ) -> ToolExecution:
        self.trace.append(("after_tool_execution", execution.id))
        return execution


# ── local mid-state session literals ──────────────────────────────────────────

# A crash mid-decide whose turn the user then abandoned: the execution was
# never offered to the policy (approval_status None, empty audit log) and a
# CancelRequested is already parked. The next drive is the FLUSH — the decide
# step is never reached, so no permission hook may fire at all.
UNDECIDED_PARKED_SESSION = UNDECIDED_SESSION.model_copy(
    deep=True,
    update={"id": "s_undecided_parked"},
)
UNDECIDED_PARKED_SESSION.entries["cr"] = CancelRequested(
    id="cr",
    parent_id="te1",
    created_at=600,
)
main_conversation(UNDECIDED_PARKED_SESSION).nodes.append("cr")
main_conversation(UNDECIDED_PARKED_SESSION).updated_at = 600

# Two approved-but-unrun calls from one assistant response: the batch
# precondition. Neither has been dispatched, so a cancel landing during the
# first one's dispatch has to terminalize both.
TWO_CLEARED_SESSION = make_session(
    id="s_two_cleared",
    entries={
        "u1": UserMessage(
            id="u1",
            created_at=500,
            parts=[TextContent(text="Add 1 and 2, then multiply 3 by 4")],
        ),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
        "a1": AssistantMessage(
            id="a1",
            parent_id="ts",
            created_at=500,
            parts=[
                ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
                ToolCall(id="tc2", name="multiply", arguments={"a": 3, "b": 4}),
            ],
            llm_config=MODEL,
            stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1",
            conversation_id="c1",
            parent_id="a1",
            created_at=500,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            tool_spec=ADD_SPEC,
            status=ExecutionStatus.PENDING,
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[ALLOW_600],
            updated_at=600,
        ),
        "te2": ToolExecution(
            id="te2",
            conversation_id="c1",
            parent_id="te1",
            created_at=500,
            tool_call_id="tc2",
            raw_tool_call=ToolCall(
                id="tc2",
                name="multiply",
                arguments={"a": 3, "b": 4},
            ),
            tool_spec=MULTIPLY_SPEC,
            status=ExecutionStatus.PENDING,
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[ALLOW_600],
            updated_at=600,
        ),
    },
    conversations={
        "c1": conversation(
            "c1",
            ["u1", "ts", "a1", "te1", "te2"],
            created_at=500,
            updated_at=600,
        )
    },
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)


def one_user_message_session(
    session_id: str,
    runtime_config: RuntimeConfig | None = None,
) -> AgentSession:
    """A fresh session with one unanswered question — the plain live-run
    precondition. A module-level factory, so the test body stays declarative
    and no two tests share a mutable literal."""
    return make_session(
        id=session_id,
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1"],
                created_at=500,
                updated_at=500,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(
            llm_config=MODEL,
            runtime_config=runtime_config or RuntimeConfig(),
        ),
    )


# ── cancel(): the no-op and diagnostic branches ───────────────────────────────


async def test_cancel_on_idle_session_is_a_noop():
    session = make_session(
        id="s_idle",
        conversations={"c1": conversation("c1", [], created_at=900, updated_at=900)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, now=1000)

    runner.cancel()

    assert runner.idle()
    assert runner.session.entries == {}


async def test_cancel_on_fresh_pending_is_a_noop_and_the_turn_runs_normally():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_fresh",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        provider=faux,
        ids=["ts", "a1", "tf"],
        now=1000,
    )

    runner.cancel()  # no open turn yet — cancel targets the turn, not the handle

    assert runner.busy()
    assert main_conversation(runner.session).nodes == ["u1"]
    result = await runner.run()
    assert result.outcome == TurnOutcome.COMPLETED


async def test_second_cancel_raises_and_the_first_call_wins():
    session = GATED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()], decisions=[]),
        provider=FauxProvider(),
        ids=["cr"],
        now=1000,
    )
    runner.cancel(error="first")

    with pytest.raises(AlreadyCancellingError):
        runner.cancel(error="second")

    assert runner.cancelling()
    assert runner.session.entries["cr"] == CancelRequested(
        id="cr",
        parent_id="te1",
        created_at=1000,
        outcome=TurnOutcome.CANCELLED,
        error="first",
    )


# ── abandon at the gate + the flush drive ─────────────────────────────────────


async def test_cancel_at_the_gate_parks_and_the_next_run_flushes():
    faux = FauxProvider()
    session = GATED_SESSION.model_copy(deep=True)
    fake = FakeToolRegistry([AddTool()], decisions=[])  # empty decide script
    runner = DeterministicRunner(
        session,
        tool_registry=fake,
        provider=faux,
        ids=["cr", "tf"],
        now=1000,
    )

    runner.cancel()  # abandon the turn — not DENY: deny would feed the model

    assert runner.cancelling()
    assert main_conversation(runner.session).nodes == [
        "u1",
        "ts",
        "a1",
        "te1",
        "cr",
    ]

    async with runner.run() as run:  # the flush: no decide(), no LLM call
        events = [event async for event in run]

    cancelled = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=500,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.CANCELLED,
        result=None,
        approval_status=ApprovalStatus.PENDING,  # untouched by the wind-down
        approval_decisions=[PENDING_500],
        cancel_signalled_at=1000,
        ended_at=1000,
        updated_at=1000,
    )
    assert events == [
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=cancelled,
            result_text="[tool execution cancelled]",
            is_error=True,
        ),
    ]
    assert fake.seen == []
    assert faux.requests == []
    assert runner.session.entries["te1"] == cancelled
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="cr",
        created_at=1000,
        outcome=TurnOutcome.CANCELLED,
    )
    assert runner.idle()


async def test_flush_via_await_returns_a_cancelled_result():
    session = CANCEL_PARKED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()], decisions=[]),
        provider=FauxProvider(),
        ids=["tf"],
        now=1000,
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
        session,
        tool_registry=FakeToolRegistry([AddTool()], decisions=[]),
        provider=FauxProvider(),
        ids=["cr"],
        now=1000,
    )
    runner.cancel(error="user abandoned the turn")
    payload = runner.session.model_dump_json()  # "process exits" here

    reloaded = AgentSession.model_validate_json(payload)
    resumed = DeterministicRunner(
        reloaded,
        tool_registry=FakeToolRegistry([AddTool()], decisions=[]),
        provider=FauxProvider(),
        ids=["tf"],
        now=2000,
    )

    assert resumed.cancelling()  # derive_status found the unconsumed entry
    result = await resumed.run()

    assert result.outcome == TurnOutcome.CANCELLED
    assert resumed.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="cr",
        created_at=2000,
        outcome=TurnOutcome.CANCELLED,
        error="user abandoned the turn",
    )
    assert resumed.idle()


async def test_cancel_after_the_flush_completed_is_a_noop():
    session = CANCEL_PARKED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()], decisions=[]),
        provider=FauxProvider(),
        ids=["tf"],
        now=1000,
    )
    await runner.run()  # the flush
    nodes_after_flush = list(main_conversation(runner.session).nodes)

    runner.cancel()  # IDLE again — branch 3

    assert runner.idle()
    assert main_conversation(runner.session).nodes == nodes_after_flush


async def test_flush_leaves_an_already_terminal_execution_untouched():
    # the wind-down cancels only nonterminal executions: a call the policy
    # already denied (terminal REJECTED at decision time) is unaffected and
    # emits no second ToolExecuted.
    session = GATED_SESSION.model_copy(deep=True)
    session.entries["te1"] = session.entries["te1"].model_copy(
        update={
            "status": ExecutionStatus.REJECTED,
            "approval_status": ApprovalStatus.REJECTED,
            "approval_decisions": [
                PENDING_500,
                ApprovalDecision(decision=ApprovalOption.DENY, created_at=600),
            ],
            "ended_at": 600,
            "updated_at": 600,
        }
    )
    session.entries["cr"] = CancelRequested(id="cr", parent_id="te1", created_at=600)
    main_conversation(session).nodes.append("cr")
    rejected_before_flush = session.entries["te1"].model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()], decisions=[]),
        provider=FauxProvider(),
        ids=["tf"],
        now=1000,
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
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("hang", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
        ]
    )
    tool = HangingTool()
    session = one_user_message_session("s_live")
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([tool]),
        provider=faux,
        ids=["ts", "a1", "te1", "cr", "tf"],
        now=1000,
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
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="hang", arguments={"a": 1, "b": 2}),
        tool_spec=HANG_SPEC,
        tool_spec_id=HANG_SPEC.spec_id(),
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
        id="tf",
        parent_id="cr",
        created_at=1000,
        outcome=TurnOutcome.CANCELLED,
    )
    async with run:
        events = [event async for event in run]
    assert [event.type for event in events] == [
        "finish_reason",
        "tool_call_received",
        "tool_execution_started",
        "tool_executed",
    ]
    assert events[3] == ToolExecuted(
        conversation_id="c1",
        tool_call_id="tc1",
        execution=interrupted,
        result_text="[tool execution interrupted]",
        is_error=True,
    )


async def test_cooperative_tool_finishing_within_grace_records_its_real_result():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("cooperative", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
        ]
    )
    tool = CooperativeTool()
    session = one_user_message_session(
        "s_grace",
        # huge grace — never expires; the tool returns the instant the token
        # trips, so nothing here depends on real time
        RuntimeConfig(tool_cancellation_grace_period=30_000),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([tool]),
        provider=faux,
        ids=["ts", "a1", "te1", "cr", "tf"],
        now=1000,
    )
    run = runner.start()
    await tool.started.wait()

    runner.cancel()
    result = await run

    assert result.outcome == TurnOutcome.CANCELLED
    # it RETURNED — a real COMPLETED result, keeping cancel_signalled_at
    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(
            id="tc1",
            name="cooperative",
            arguments={"a": 1, "b": 2},
        ),
        tool_spec=COOPERATIVE_SPEC,
        tool_spec_id=COOPERATIVE_SPEC.spec_id(),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(
            content=[TextContent(text="partial sum")],
            is_error=False,
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
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    faux_tool_call("hang", {"a": 1, "b": 2}, id="tc1"),
                    faux_tool_call("add", {"a": 3, "b": 4}, id="tc2"),
                ],
                finish_reason="tool_use",
            ),
        ]
    )
    tool = HangingTool()
    session = one_user_message_session("s_sibling")
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([tool, AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "te2", "cr", "tf"],
        now=1000,
    )
    run = runner.start()
    await tool.started.wait()  # tc1 in-flight; tc2 not yet started

    runner.cancel()
    result = await run

    assert result.outcome == TurnOutcome.CANCELLED
    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="hang", arguments={"a": 1, "b": 2}),
        tool_spec=HANG_SPEC,
        tool_spec_id=HANG_SPEC.spec_id(),
        status=ExecutionStatus.INTERRUPTED,  # the body ran, grace expired
        result=None,
        error=None,
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[ALLOW_1000],
        started_at=1000,
        ended_at=1000,
        cancel_signalled_at=1000,
        updated_at=1000,
    )
    assert runner.session.entries["te2"] == ToolExecution(
        id="te2",
        conversation_id="c1",
        parent_id="te1",
        created_at=1000,
        tool_call_id="tc2",
        raw_tool_call=ToolCall(id="tc2", name="add", arguments={"a": 3, "b": 4}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.CANCELLED,  # never dispatched
        result=None,
        error=None,
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[ALLOW_1000],
        started_at=None,
        ended_at=1000,
        cancel_signalled_at=1000,
        updated_at=1000,
    )
    async with run:
        events = [event async for event in run]
    assert [(event.type, getattr(event, "tool_call_id", None)) for event in events] == [
        ("finish_reason", None),
        ("tool_call_received", "tc1"),
        ("tool_call_received", "tc2"),
        ("tool_execution_started", "tc1"),  # only the dispatched call
        ("tool_executed", "tc1"),
        ("tool_executed", "tc2"),
    ]
    assert events[4].result_text == "[tool execution interrupted]"
    assert events[5].result_text == "[tool execution cancelled]"


async def test_lazy_cancel_between_events_cancels_the_unstarted_execution():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
        ]
    )
    session = one_user_message_session("s_lazy_cancel")
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "cr", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = []
        async for event in run:
            events.append(event)
            if event.type == "tool_call_received":
                run.cancel()  # before the policy or the batch ever sees it

    assert [event.type for event in events] == [
        "finish_reason",
        "tool_call_received",
        "tool_executed",
    ]
    assert events[2].result_text == "[tool execution cancelled]"
    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.CANCELLED,
        result=None,
        error=None,
        approval_status=None,  # the policy never processed it
        approval_decisions=[],
        started_at=None,
        ended_at=1000,
        cancel_signalled_at=1000,
        updated_at=1000,
    )
    assert runner.session.entries["tf"].outcome == TurnOutcome.CANCELLED
    assert runner.idle()


async def test_cancel_landing_mid_decide_winds_down_instead_of_pausing():
    # decide() returned before the token was checked, so the deferral IS
    # recorded; the pre-park check then winds down BEFORE the approval pause —
    # no ApprovalRequired, the gate never opens.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
        ]
    )
    registry = CancellingDecideRegistry([AddTool()])
    session = one_user_message_session("s_mid_decide")
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1", "cr", "tf"],
        now=1000,
    )
    registry.runner = runner

    async with runner.run() as run:
        events = [event async for event in run]

    assert [event.type for event in events] == [
        "finish_reason",
        "tool_call_received",
        "tool_executed",
    ]
    assert events[2].result_text == "[tool execution cancelled]"
    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.CANCELLED,
        result=None,
        error=None,
        # the deferral WAS processed before the cancel consumed the turn
        approval_status=ApprovalStatus.PENDING,
        approval_decisions=[PENDING_1000],
        started_at=None,
        ended_at=1000,
        cancel_signalled_at=1000,
        updated_at=1000,
    )
    assert runner.session.entries["tf"].outcome == TurnOutcome.CANCELLED
    assert runner.idle()


async def test_mid_stream_cancel_drops_the_partial_assistant_message():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_thinking("Thinking…"), faux_hang()],
                finish_reason="stop",
            ),
        ]
    )
    session = make_session(
        id="s_stream_cancel",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        provider=faux,
        ids=["ts", "cr", "tf"],
        now=1000,
    )
    run = runner.start(streaming=True)

    async with run:
        events = []
        async for event in run:
            events.append(event)
            if event == ReasoningDelta(conversation_id="c1", text="Thinking…"):
                run.cancel()  # the stream is parked mid-generation

    assert events == [ReasoningStart(conversation_id="c1"), ReasoningDelta(conversation_id="c1", text="Thinking…")]
    assert run.result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.CANCELLED,
        pending_approvals=[],
    )
    # the partial assistant message was dropped — only bookkeeping landed
    assert main_conversation(runner.session).nodes == ["u1", "ts", "cr", "tf"]
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="cr",
        created_at=1000,
        outcome=TurnOutcome.CANCELLED,
    )
    assert runner.idle()


# ── the four registry awaits: a hung phase can never eat a cancel ─────────────


async def test_cancel_during_get_tools_unblocks_the_run_and_closes_cancelled():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
        ]
    )
    registry = HangingGetToolsRegistry([AddTool()])
    session = one_user_message_session("s_hung_get_tools")
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "cr", "tf"],
        now=1000,
    )
    run = runner.start()
    await registry.started.wait()  # the tool listing is parked

    runner.cancel(error="user abandoned the turn")
    result = await run

    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.CANCELLED,
        pending_approvals=[],
    )
    assert faux.requests == []  # no tool list → no LLM call
    assert main_conversation(runner.session).nodes == ["u1", "ts", "cr", "tf"]
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="cr",
        created_at=1000,
        outcome=TurnOutcome.CANCELLED,
        error="user abandoned the turn",
    )
    assert runner.idle()


async def test_cancel_during_create_execution_records_cancelled_not_failed():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
        ]
    )
    registry = HangingBirthRegistry([AddTool()])
    session = one_user_message_session("s_hung_birth")
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        # The execution exists BEFORE the cancel can be requested: it was
        # appended with the assistant message that asked for it, and only its
        # birth was still in flight.
        ids=["ts", "a1", "te1", "cr", "tf"],
        now=1000,
    )
    run = runner.start()
    await registry.started.wait()  # the birth is parked

    runner.cancel()
    result = await run

    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.CANCELLED,
        pending_approvals=[],
    )
    born = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=None,  # the registry never got to resolve it
        status=ExecutionStatus.PENDING,  # a lost race is not a failure
    )
    cancelled = born.model_copy(
        update={
            "status": ExecutionStatus.CANCELLED,
            "error": None,  # CANCELLED, never FAILED
            "ended_at": 1000,
            "cancel_signalled_at": 1000,
            "updated_at": 1000,
        },
    )
    async with run:
        events = [event async for event in run]
    assert events == [
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc1", execution=born),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=cancelled,
            result_text="[tool execution cancelled]",
            is_error=True,
        ),
    ]
    assert runner.session.entries["te1"] == cancelled
    assert runner.session.tool_specs == {}
    assert runner.idle()


async def test_cancel_on_an_unborn_execution_settles_it_instead_of_stranding_it():
    # ── precondition ─────────────────────────────────────────────────────────
    # A run suspended at the tool-call boundary: the execution is durable and
    # RECEIVED — appended with the assistant message, never offered to the
    # registry. The wind-down has to settle it like any other undispatched
    # call; leaving it behind would close the turn over an execution no drive
    # will ever finish, and every later projection would fail loudly on it.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
        ]
    )
    session = one_user_message_session("s_cancel_unborn")
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "cr", "tf"],
        now=1000,
    )
    async with runner.run() as run:
        async for _ in run:
            break  # suspend before the birth step
    assert runner.session.entries["te1"].status is ExecutionStatus.RECEIVED

    # ── action ───────────────────────────────────────────────────────────────
    runner.cancel()
    result = await runner.run()

    # ── postcondition ────────────────────────────────────────────────────────
    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.CANCELLED,
        pending_approvals=[],
    )
    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=None,  # the registry was never consulted
        status=ExecutionStatus.CANCELLED,
        result=None,
        error=None,  # CANCELLED, never FAILED
        started_at=None,
        ended_at=1000,
        cancel_signalled_at=1000,
        updated_at=1000,
    )
    assert runner.idle()


async def test_cancel_during_decide_records_no_decision_and_closes_cancelled():
    registry = HangingDecideRegistry([AddTool()])
    recorder = HookRecorder()
    session = UNDECIDED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=FauxProvider(),
        ids=["cr", "tf"],
        now=1000,
        middleware=[recorder],
    )
    run = runner.start()
    await registry.started.wait()  # the decision is parked

    runner.cancel()
    result = await run

    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.CANCELLED,
        pending_approvals=[],
    )
    # no decision was applied, and `after_permission_decision` never fired.
    # The call was never selected for dispatch either, so it winds down
    # through the terminal tail alone.
    assert recorder.trace == [
        ("before_permission_check", "te1"),
        ("after_tool_execution", "te1"),
    ]
    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=500,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.CANCELLED,
        result=None,
        error=None,
        approval_status=None,
        approval_decisions=[],
        started_at=None,
        ended_at=1000,
        cancel_signalled_at=1000,
        updated_at=1000,
    )
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="cr",
        created_at=1000,
        outcome=TurnOutcome.CANCELLED,
    )
    assert runner.idle()


async def test_cancel_during_prepare_records_cancelled_without_dispatching():
    registry = HangingPrepareRegistry([AddTool()])
    session = CLEARED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=FauxProvider(),
        ids=["cr", "tf"],
        now=1000,
    )
    run = runner.start()
    await registry.started.wait()  # the preparation is parked

    runner.cancel()
    result = await run

    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.CANCELLED,
        pending_approvals=[],
    )
    cancelled = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=500,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.CANCELLED,
        result=None,
        error=None,
        approval_status=ApprovalStatus.ALLOWED,  # untouched by the wind-down
        approval_decisions=[PENDING_500, ALLOW_600],
        started_at=None,  # the body was never dispatched
        ended_at=1000,
        cancel_signalled_at=1000,
        updated_at=1000,
    )
    async with run:
        events = [event async for event in run]
    # no ToolExecutionStarted: it is emitted iff the body was dispatched
    assert events == [
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=cancelled,
            result_text="[tool execution cancelled]",
            is_error=True,
        ),
    ]
    assert runner.session.entries["te1"] == cancelled
    assert runner.session.entries["te1"].dispatched is False
    assert runner.idle()


async def test_cancel_after_prepare_returned_records_cancelled_and_never_runs_the_body():
    tool = RecordingAddTool()
    registry = CancellingPrepareRegistry([tool])
    session = CLEARED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=FauxProvider(),
        ids=["cr", "tf"],
        now=1000,
    )
    registry.runner = runner

    async with runner.run() as run:
        events = [event async for event in run]

    assert registry.prepared == ["add"]  # preparation SUCCEEDED
    assert tool.calls == []  # the grace window never starts new work
    cancelled = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=500,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.CANCELLED,
        result=None,
        error=None,
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[PENDING_500, ALLOW_600],
        started_at=None,
        ended_at=1000,
        cancel_signalled_at=1000,
        updated_at=1000,
    )
    assert events == [
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=cancelled,
            result_text="[tool execution cancelled]",
            is_error=True,
        ),
    ]
    assert runner.session.entries["te1"] == cancelled
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="cr",
        created_at=1000,
        outcome=TurnOutcome.CANCELLED,
    )
    assert runner.idle()


async def test_every_call_still_gets_an_execution_when_the_cancel_lands_mid_batch():
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
        ]
    )
    registry = CancelOnFirstBirthRegistry([AddTool(), MultiplyTool()])
    session = one_user_message_session("s_mid_batch_birth")
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        # Both executions exist before the cancel: they were appended with the
        # assistant message, and only their births were still in flight.
        ids=["ts", "a1", "te1", "te2", "cr", "tf"],
        now=1000,
    )
    registry.runner = runner

    async with runner.run() as run:
        events = [event async for event in run]

    # two calls → two executions, whichever births the cancel reached
    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=ADD_SPEC,  # this birth RETURNED before the race was lost
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.CANCELLED,
        result=None,
        error=None,
        approval_status=None,
        approval_decisions=[],
        started_at=None,
        ended_at=1000,
        cancel_signalled_at=1000,
        updated_at=1000,
    )
    assert runner.session.entries["te2"] == ToolExecution(
        id="te2",
        conversation_id="c1",
        parent_id="te1",
        created_at=1000,
        tool_call_id="tc2",
        raw_tool_call=ToolCall(
            id="tc2",
            name="multiply",
            arguments={"a": 3, "b": 4},
        ),
        tool_spec=None,  # runner-synthesized: the birth never returned
        status=ExecutionStatus.CANCELLED,
        result=None,
        error=None,
        approval_status=None,
        approval_decisions=[],
        started_at=None,
        ended_at=1000,
        cancel_signalled_at=1000,
        updated_at=1000,
    )
    assert [(event.type, getattr(event, "tool_call_id", None)) for event in events] == [
        ("finish_reason", None),
        ("tool_call_received", "tc1"),
        ("tool_call_received", "tc2"),
        ("tool_executed", "tc1"),
        ("tool_executed", "tc2"),
    ]
    assert runner.idle()


async def test_batch_cancelled_after_the_first_dispatch_ends_both_cancelled_alike():
    registry = HangingPrepareRegistry([AddTool(), MultiplyTool()])
    recorder = HookRecorder()
    session = TWO_CLEARED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=FauxProvider(),
        ids=["cr", "tf"],
        now=1000,
        middleware=[recorder],
    )
    run = runner.start()
    await registry.started.wait()  # tc1's preparation is parked; tc2 untouched

    runner.cancel()
    result = await run

    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.CANCELLED,
        pending_approvals=[],
    )
    assert registry.prepared == []  # neither call ever resolved
    # The two pipelines differ in the DISPATCH hook and agree on the terminal
    # one: te1 was selected for dispatch (its `prepare` parked, so the hook had
    # already fired) and is cancelled in place; te2 was never selected, so it
    # belongs to the wind-down and `before_tool_execution` never fires for it.
    # Both still reach the universal terminal tail exactly once.
    assert recorder.trace == [
        ("before_tool_execution", "te1"),
        ("after_tool_execution", "te1"),
        ("after_tool_execution", "te2"),
    ]
    # identical durable shape — the in-place cancellation and the wind-down
    # produce the same record
    first = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=500,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.CANCELLED,
        result=None,
        error=None,
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[ALLOW_600],
        started_at=None,
        ended_at=1000,
        cancel_signalled_at=1000,
        updated_at=1000,
    )
    second = ToolExecution(
        id="te2",
        conversation_id="c1",
        parent_id="te1",
        created_at=500,
        tool_call_id="tc2",
        raw_tool_call=ToolCall(
            id="tc2",
            name="multiply",
            arguments={"a": 3, "b": 4},
        ),
        tool_spec=MULTIPLY_SPEC,
        tool_spec_id=MULTIPLY_SPEC.spec_id(),
        status=ExecutionStatus.CANCELLED,
        result=None,
        error=None,
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[ALLOW_600],
        started_at=None,
        ended_at=1000,
        cancel_signalled_at=1000,
        updated_at=1000,
    )
    assert runner.session.entries["te1"] == first
    assert runner.session.entries["te2"] == second
    async with run:
        events = [event async for event in run]
    assert events == [
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=first,
            result_text="[tool execution cancelled]",
            is_error=True,
        ),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc2",
            execution=second,
            result_text="[tool execution cancelled]",
            is_error=True,
        ),
    ]
    assert runner.idle()


async def test_a_resource_held_inside_create_execution_is_released_before_the_wind_down():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
        ]
    )
    trace: list = []
    registry = ResourceHoldingRegistry([AddTool()], trace=trace)
    session = one_user_message_session("s_held_resource")
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1", "cr", "tf"],
        now=1000,
        middleware=[HookRecorder(trace)],
    )
    run = runner.start()
    await registry.started.wait()  # the birth is parked inside `async with`

    runner.cancel()
    result = await run

    assert result.outcome == TurnOutcome.CANCELLED
    # the kill is awaited out: the `async with` unwound completely before the
    # runner moved on to terminalize the call
    assert trace == [
        ("acquired", None),
        ("released", None),
        ("after_tool_execution", "te1"),
    ]
    assert runner.session.entries["te1"].status == ExecutionStatus.CANCELLED
    assert runner.idle()


async def test_a_pending_cancellation_fires_no_before_permission_check():
    registry = FakeToolRegistry([AddTool()], decisions=[])  # empty decide script
    recorder = HookRecorder()
    session = UNDECIDED_PARKED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=FauxProvider(),
        ids=["tf"],
        now=1000,
        middleware=[recorder],
    )

    async with runner.run() as run:  # the flush: the decide step is never reached
        events = [event async for event in run]

    assert registry.seen == []
    assert recorder.trace == [("after_tool_execution", "te1")]
    cancelled = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=500,
        tool_call_id="tc1",
        raw_tool_call=ADD_CALL,
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.CANCELLED,
        result=None,
        error=None,
        approval_status=None,
        approval_decisions=[],
        started_at=None,
        ended_at=1000,
        cancel_signalled_at=1000,
        updated_at=1000,
    )
    assert events == [
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=cancelled,
            result_text="[tool execution cancelled]",
            is_error=True,
        ),
    ]
    assert runner.session.entries["te1"] == cancelled
    assert runner.idle()


# ── cancel vs the grace window at the close sites ─────────────────────────────


async def test_llm_answer_within_grace_is_recorded_but_the_cancel_still_wins():
    faux = ReleasableProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Here you go.")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_llm_grace",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(
            llm_config=MODEL,
            # huge grace — never expires; the faux answers the instant the
            # test releases it, so nothing here depends on real time
            runtime_config=RuntimeConfig(llm_completion_cancellation_grace_period=30_000),
        ),
    )
    runner = DeterministicRunner(
        session,
        provider=faux,
        ids=["ts", "cr", "a1", "tf"],
        now=1000,
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
        id="a1",
        parent_id="cr",
        created_at=1000,
        parts=[TextContent(text="Here you go.")],
        llm_config=MODEL,
        stop_reason="stop",
        context_tokens=3,  # len("Here you go.") // 4
    )
    # ... but the requested outcome controls the close
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="a1",
        created_at=1000,
        outcome=TurnOutcome.CANCELLED,
    )
    assert main_conversation(runner.session).nodes == [
        "u1",
        "ts",
        "cr",
        "a1",
        "tf",
    ]
    assert runner.idle()
    async with run:
        events = [event async for event in run]
    assert events == [
        TextBlock(conversation_id="c1", text="Here you go."),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]


async def test_llm_failure_within_grace_closes_cancelled_and_returns_normally():
    faux = ReleasableProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [],
                error=faux_error("provider 500", error_class=ProviderAPIError),
            ),
        ]
    )
    session = make_session(
        id="s_llm_grace_fail",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(
            llm_config=MODEL,
            runtime_config=RuntimeConfig(llm_completion_cancellation_grace_period=30_000),
        ),
    )
    runner = DeterministicRunner(
        session,
        provider=faux,
        ids=["ts", "cr", "tf"],
        now=1000,
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
        id="tf",
        parent_id="cr",
        created_at=1000,
        outcome=TurnOutcome.CANCELLED,
        error="user abandoned the turn",
    )
    assert main_conversation(runner.session).nodes == ["u1", "ts", "cr", "tf"]
    assert runner.idle()


async def test_eager_cancel_before_the_first_tick_flushes_without_an_llm_call():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_instant_cancel",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        provider=faux,
        ids=["ts", "cr", "tf"],
        now=1000,
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
        id="tf",
        parent_id="cr",
        created_at=1000,
        outcome=TurnOutcome.CANCELLED,
    )
    assert main_conversation(runner.session).nodes == ["u1", "ts", "cr", "tf"]
    assert runner.idle()


async def test_tool_raising_timeout_error_within_grace_records_failed():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("deadline_confused", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
        ]
    )
    tool = TimeoutRaisingTool()
    session = one_user_message_session(
        "s_tool_timeout_grace",
        RuntimeConfig(tool_cancellation_grace_period=30_000),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([tool]),
        provider=faux,
        ids=["ts", "a1", "te1", "cr", "tf"],
        now=1000,
    )
    run = runner.start()
    await tool.started.wait()

    runner.cancel()  # the tool wakes and raises its OWN TimeoutError
    result = await run

    assert result.outcome == TurnOutcome.CANCELLED
    # the tool FAILED — it was not interrupted by the grace machinery
    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(
            id="tc1",
            name="deadline_confused",
            arguments={"a": 1, "b": 2},
        ),
        tool_spec=DEADLINE_SPEC,
        tool_spec_id=DEADLINE_SPEC.spec_id(),
        status=ExecutionStatus.FAILED,
        result=None,
        error=ToolExecutionError(
            error_type="TimeoutError",
            error_message="tool's own deadline",
            details={"phase": "execution"},
        ),
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[ALLOW_1000],
        started_at=1000,
        ended_at=1000,
        cancel_signalled_at=1000,
        updated_at=1000,
        context_tokens=4,  # len("tool's own deadline") // 4
    )
    assert runner.session.entries["tf"].outcome == TurnOutcome.CANCELLED
    assert runner.idle()
