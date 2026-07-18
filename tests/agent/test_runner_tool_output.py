"""Explicit, declarative tests for the tool-output plumbing.

Decision-support tests: each one lays out the FULL story of one tool-call
round — precondition (every object built inline, no scenarios.py constants),
one action, postcondition (the literal AgentSession and event list, every
`ToolExecution` field spelled out) — so the exact shape of the execution
lifecycle (`status` / `approval_status` / `result` / `error` /
`started_at` / `ended_at` / `cancel_signalled_at`) and of the three event
snapshots (`ToolCallReceived` at birth, `ToolExecutionStarted` at dispatch,
`ToolExecuted` at the terminal outcome) is visible on the page while the
tool-output design is being reviewed.

Everything is deliberately inlined and repeated. Do not factor helpers out of
this file; the duplication IS the point.
"""

import asyncio

from pydantic import BaseModel, ConfigDict

from luca.agent.core.context import CancellationToken, ToolContext
from luca.agent.core.events import (
    FinishReason,
    TextBlock,
    ToolCallReceived,
    ToolExecuted,
    ToolExecutionStarted,
)
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
    RuntimeConfig,
    SessionConfig,
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
from luca.agent.core.runner import AgentSessionRunner
from luca.agent.core.tool_registry import ToolRegistry
from luca.agent.core.tools import Tool
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text, faux_tool_call


class AddArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: int
    b: int


class AddTool(Tool):
    name = "add"
    description = "Add two numbers."
    Args = AddArgs

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        return str(args["a"] + args["b"])


class SleepArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SleepForeverTool(Tool):
    """Never returns on its own: whatever ends it (a cancel's hard cancel, a
    deadline) is what the recorded status reports."""

    name = "sleep_forever"
    description = "Sleep until something kills it."
    Args = SleepArgs

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        await asyncio.sleep(30)
        return "never happens"


class SlowTool(Tool):
    """Same, but with its own 50 ms deadline — it can only ever TIME OUT."""

    name = "slow"
    description = "Sleep past its own deadline."
    Args = SleepArgs
    timeout_in_ms = 50

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        await asyncio.sleep(30)
        return "never happens"


class CooperativeSleepTool(Tool):
    """Watches the `cancellation_token`, then spends 50 ms winding down
    before returning partial output. A real tool would poll `.cancelled`
    between units of work and flush what it has; the sleep IS the flush, and it
    is what the grace window has to cover — a tool that returned instantly
    would never need grace at all."""

    name = "cooperative_sleep"
    description = "Sleep, but wind down and return early when cancelled."
    Args = SleepArgs

    async def execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        await cancellation_token.wait_cancelled()
        await asyncio.sleep(0.05)  # flush partial work
        return ExecutionResult(
            content=[TextContent(text="cut short: 2 of 10 rows written")],
            metadata={"cut_short": True},
            is_error=False,
        )


class InlineToolRegistry(ToolRegistry):
    """Minimal inline registry over known-good calls: PENDING births carrying
    the tool's spec, ALLOW every decision (`created_at` frozen to 1000 — the
    production ApprovalDecision default stamps the wall clock, which would
    break the literal session asserts below), resolve-validate-invoke
    execute. The error-path registries live in `scenarios.FakeToolRegistry`;
    this file only ever calls tools that resolve and validate."""

    def __init__(self, tools: list[Tool]) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def get_tools(self, agent_session: AgentSession) -> list[Tool]:
        return list(self.tools_by_name.values())

    async def create_execution(
        self, call: ToolCall, context: ToolContext,
    ) -> ToolExecution:
        return ToolExecution(
            id="", created_at=0,
            tool_call_id=call.id, raw_tool_call=call,
            tool_spec=self.tools_by_name[call.name].get_tool_spec(),
        )

    async def decide(
        self, tool_execution: ToolExecution, context: ToolContext,
    ) -> ApprovalDecision:
        return ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)

    async def execute(
        self,
        tool_execution: ToolExecution,
        context: ToolContext,
        *,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        tool = self.tools_by_name[tool_execution.raw_tool_call.name]
        args = tool.Args.model_validate(tool_execution.raw_tool_call.arguments)
        return await tool.execute(
            args.model_dump(), context, cancellation_token=cancellation_token,
        )


class ScriptedRunner(AgentSessionRunner):
    """The production runner with its two determinism hooks overridden inline:
    `generate_id()` pops the next scripted id (consumed in entry-creation
    order), `now_ms()` is frozen to 1000."""

    def __init__(self, *args, ids: list[str], **kwargs) -> None:
        self._ids = iter(ids)
        super().__init__(*args, **kwargs)

    def generate_id(self) -> str:
        return next(self._ids)

    def now_ms(self) -> int:
        return 1000


async def test_successful_tool_call_full_session_shape():
    # ── precondition ─────────────────────────────────────────────────────────
    # An empty IDLE session; a runner with the single AddTool; a scripted
    # provider that answers the first LLM call with one tool call and the
    # second with the final text. Entry ids are consumed in creation order:
    #   u1 (post_message), ts (TurnStart), a1 (tool-call AssistantMessage),
    #   te1 (ToolExecution), a2 (final AssistantMessage), tf (TurnFinish).
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s1",
        entries={},
        active_conversation=Conversation(id="c1", nodes=[], created_at=500, updated_at=500),
        session_config=SessionConfig(
            llm_config=LLMConfig(model="test-model", provider="faux"),
        ),
    )
    runner = ScriptedRunner(
        session,
        tool_registry=InlineToolRegistry([AddTool()]),
        provider=faux,
        ids=["u1", "ts", "a1", "te1", "a2", "tf"],
    )

    # ── action ───────────────────────────────────────────────────────────────
    runner.post_message("Add 1 and 2")
    async with runner.run() as run:
        events = [event async for event in run]

    # ── postcondition ────────────────────────────────────────────────────────
    # Three snapshots of the SAME durable execution: born PENDING and never
    # yet seen by the policy; persisted RUNNING (approval ALLOWED,
    # `started_at` stamped) immediately before the body ran; COMPLETED with
    # the tool's own result. `ToolExecuted.result_text` / `is_error` are the
    # projection the model sees.
    assert events == [
        FinishReason(finish_reason="tool_use"),
        ToolCallReceived(
            tool_call_id="tc1",
            execution=ToolExecution(
                id="te1", parent_id="a1", created_at=1000,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
                tool_spec=ToolSpec(name="add", description="Add two numbers."),
                extras={},
                approval_status=None,
                approval_decisions=[],
                status=ExecutionStatus.PENDING,
                result=None,
                error=None,
                started_at=None,
                ended_at=None,
                cancel_signalled_at=None,
                updated_at=None,
                is_doom_loop_flagged=False,
                context_tokens=0,
            ),
        ),
        ToolExecutionStarted(
            tool_call_id="tc1",
            execution=ToolExecution(
                id="te1", parent_id="a1", created_at=1000,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
                tool_spec=ToolSpec(name="add", description="Add two numbers."),
                extras={},
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[
                    ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
                ],
                status=ExecutionStatus.RUNNING,
                result=None,
                error=None,
                started_at=1000,
                ended_at=None,
                cancel_signalled_at=None,
                updated_at=1000,
                is_doom_loop_flagged=False,
                context_tokens=0,
            ),
        ),
        ToolExecuted(
            tool_call_id="tc1",
            execution=ToolExecution(
                id="te1", parent_id="a1", created_at=1000,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
                tool_spec=ToolSpec(name="add", description="Add two numbers."),
                extras={},
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[
                    ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
                ],
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(
                    content=[TextContent(text="3")],
                    metadata={},
                    is_error=False,
                ),
                error=None,
                started_at=1000,
                ended_at=1000,
                cancel_signalled_at=None,
                updated_at=1000,
                is_doom_loop_flagged=False,
                context_tokens=0,
            ),
            result_text="3",
            is_error=False,
        ),
        TextBlock(text="It's 3."),
        FinishReason(finish_reason="stop"),
    ]
    assert runner.session == AgentSession(
        id="s1",
        entries={
            "u1": UserMessage(
                id="u1", parent_id=None, created_at=1000,
                parts=[TextContent(text="Add 1 and 2")],
                context_tokens=2,  # len("Add 1 and 2") // 4
            ),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=1000),
            "a1": AssistantMessage(
                id="a1", parent_id="ts", created_at=1000,
                parts=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
                llm_config=LLMConfig(model="test-model", provider="faux"),
                stop_reason="tool_use",
                context_tokens=4,  # (name + JSON args) // 4
            ),
            "te1": ToolExecution(
                id="te1", parent_id="a1", created_at=1000,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
                tool_spec=ToolSpec(name="add", description="Add two numbers."),
                extras={},
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[
                    ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
                ],
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(
                    content=[TextContent(text="3")],
                    metadata={},
                    is_error=False,
                ),
                error=None,
                started_at=1000,
                ended_at=1000,
                cancel_signalled_at=None,
                updated_at=1000,
                is_doom_loop_flagged=False,
                context_tokens=0,
            ),
            "a2": AssistantMessage(
                id="a2", parent_id="te1", created_at=1000,
                parts=[TextContent(text="It's 3.")],
                llm_config=LLMConfig(model="test-model", provider="faux"),
                stop_reason="stop",
                context_tokens=1,  # len("It's 3.") // 4
            ),
            "tf": TurnFinish(id="tf", parent_id="a2", created_at=1000),
        },
        tool_executions={"tc1": ["te1"]},
        usages={"c1": {
            "a1": Usage(conversation_id="c1", entry_id="a1"),
            "a2": Usage(conversation_id="c1", entry_id="a2"),
        }},
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts", "a1", "te1", "a2", "tf"],
            created_at=500,
            updated_at=1000,
            status=ConversationStatus.IDLE,
        ),
        session_config=SessionConfig(
            llm_config=LLMConfig(model="test-model", provider="faux"),
        ),
    )


async def test_interrupted_tool_call_full_session_shape():
    # ── precondition ─────────────────────────────────────────────────────────
    # Same empty IDLE session, but the tool never returns and the run is
    # cancelled while it is in flight. Entry ids in creation order:
    #   u1 (post_message), ts (TurnStart, appended eagerly by start()),
    #   a1 (tool-call AssistantMessage), te1 (ToolExecution),
    #   cr (CancelRequested, appended by run.cancel()), tf (TurnFinish).
    # The provider is scripted with ONE response: the turn is cancelled before
    # the runner ever asks the model a second time.
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("sleep_forever", {}, id="tc1")],
            finish_reason="tool_use",
        ),
    ])
    session = AgentSession(
        id="s1",
        entries={},
        active_conversation=Conversation(id="c1", nodes=[], created_at=500, updated_at=500),
        session_config=SessionConfig(
            llm_config=LLMConfig(model="test-model", provider="faux"),
        ),
    )
    runner = ScriptedRunner(
        session,
        tool_registry=InlineToolRegistry([SleepForeverTool()]),
        provider=faux,
        ids=["u1", "ts", "a1", "te1", "cr", "tf"],
    )

    # ── action ───────────────────────────────────────────────────────────────
    runner.post_message("Sleep forever")
    run = runner.start()
    await asyncio.sleep(0.05)  # the tool is now in flight
    run.cancel()
    async with run:
        events = [event async for event in run]

    # ── postcondition ────────────────────────────────────────────────────────
    # The tool started and was hard-cancelled: INTERRUPTED, resultless AND
    # errorless (the status is the complete lifecycle fact), with the full
    # timing story — `started_at` (it dispatched), `cancel_signalled_at`
    # (persisted before the grace window ran), `ended_at`. The model would see
    # the derived placeholder output.
    assert events == [
        FinishReason(finish_reason="tool_use"),
        ToolCallReceived(
            tool_call_id="tc1",
            execution=ToolExecution(
                id="te1", parent_id="a1", created_at=1000,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="sleep_forever", arguments={}),
                tool_spec=ToolSpec(
                    name="sleep_forever",
                    description="Sleep until something kills it.",
                ),
                extras={},
                approval_status=None,
                approval_decisions=[],
                status=ExecutionStatus.PENDING,
                result=None,
                error=None,
                started_at=None,
                ended_at=None,
                cancel_signalled_at=None,
                updated_at=None,
                is_doom_loop_flagged=False,
                context_tokens=0,
            ),
        ),
        ToolExecutionStarted(
            tool_call_id="tc1",
            execution=ToolExecution(
                id="te1", parent_id="a1", created_at=1000,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="sleep_forever", arguments={}),
                tool_spec=ToolSpec(
                    name="sleep_forever",
                    description="Sleep until something kills it.",
                ),
                extras={},
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[
                    ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
                ],
                status=ExecutionStatus.RUNNING,
                result=None,
                error=None,
                started_at=1000,
                ended_at=None,
                cancel_signalled_at=None,
                updated_at=1000,
                is_doom_loop_flagged=False,
                context_tokens=0,
            ),
        ),
        ToolExecuted(
            tool_call_id="tc1",
            execution=ToolExecution(
                id="te1", parent_id="a1", created_at=1000,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="sleep_forever", arguments={}),
                tool_spec=ToolSpec(
                    name="sleep_forever",
                    description="Sleep until something kills it.",
                ),
                extras={},
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[
                    ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
                ],
                status=ExecutionStatus.INTERRUPTED,
                result=None,
                error=None,
                started_at=1000,
                ended_at=1000,
                cancel_signalled_at=1000,
                updated_at=1000,
                is_doom_loop_flagged=False,
                context_tokens=0,
            ),
            result_text="[tool execution interrupted]",
            is_error=True,
        ),
    ]
    assert runner.session == AgentSession(
        id="s1",
        entries={
            "u1": UserMessage(
                id="u1", parent_id=None, created_at=1000,
                parts=[TextContent(text="Sleep forever")],
                context_tokens=3,  # len("Sleep forever") // 4
            ),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=1000),
            "a1": AssistantMessage(
                id="a1", parent_id="ts", created_at=1000,
                parts=[ToolCall(id="tc1", name="sleep_forever", arguments={})],
                llm_config=LLMConfig(model="test-model", provider="faux"),
                stop_reason="tool_use",
                context_tokens=3,  # (name + JSON args) // 4
            ),
            "te1": ToolExecution(
                id="te1", parent_id="a1", created_at=1000,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="sleep_forever", arguments={}),
                tool_spec=ToolSpec(
                    name="sleep_forever",
                    description="Sleep until something kills it.",
                ),
                extras={},
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[
                    ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
                ],
                status=ExecutionStatus.INTERRUPTED,
                result=None,
                error=None,
                started_at=1000,
                ended_at=1000,
                cancel_signalled_at=1000,
                updated_at=1000,
                is_doom_loop_flagged=False,
                context_tokens=0,
            ),
            "cr": CancelRequested(
                id="cr", parent_id="te1", created_at=1000,
                outcome=TurnOutcome.CANCELLED,
                error=None,
            ),
            "tf": TurnFinish(
                id="tf", parent_id="cr", created_at=1000,
                outcome=TurnOutcome.CANCELLED,
                error=None,
            ),
        },
        tool_executions={"tc1": ["te1"]},
        usages={"c1": {
            "a1": Usage(conversation_id="c1", entry_id="a1"),
        }},
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts", "a1", "te1", "cr", "tf"],
            created_at=500,
            updated_at=1000,
            status=ConversationStatus.IDLE,
        ),
        session_config=SessionConfig(
            llm_config=LLMConfig(model="test-model", provider="faux"),
        ),
    )


async def test_cooperative_cancellation_returns_a_real_result():
    # ── precondition ─────────────────────────────────────────────────────────
    # The cancel counterpart of the test above, with the two ingredients that
    # make cooperation possible: a non-zero `tool_cancellation_grace_period`
    # (default 0 = straight to hard cancel) and a tool that watches the token.
    # Cancelled at the same point, in flight; entry ids in creation order:
    #   u1, ts, a1, te1, cr (CancelRequested), tf (TurnFinish).
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("cooperative_sleep", {}, id="tc1")],
            finish_reason="tool_use",
        ),
    ])
    session = AgentSession(
        id="s1",
        entries={},
        active_conversation=Conversation(id="c1", nodes=[], created_at=500, updated_at=500),
        session_config=SessionConfig(
            llm_config=LLMConfig(model="test-model", provider="faux"),
            runtime_config=RuntimeConfig(tool_cancellation_grace_period=1000),
        ),
    )
    runner = ScriptedRunner(
        session,
        tool_registry=InlineToolRegistry([CooperativeSleepTool()]),
        provider=faux,
        ids=["u1", "ts", "a1", "te1", "cr", "tf"],
    )

    # ── action ───────────────────────────────────────────────────────────────
    runner.post_message("Sleep cooperatively")
    run = runner.start()
    await asyncio.sleep(0.05)  # the tool is now in flight
    run.cancel()
    async with run:
        events = [event async for event in run]

    # ── postcondition ────────────────────────────────────────────────────────
    # The tool returned inside the grace window, so it is NOT interrupted: the
    # execution is COMPLETED, carries the tool's own ExecutionResult (content,
    # metadata, and the is_error the tool chose), and KEEPS the
    # `cancel_signalled_at` stamp that was persisted before the grace window
    # ran — the model would see the tool's own text, not a placeholder. The
    # TURN is still cancelled: the CancelRequested closes it with
    # outcome=CANCELLED and the LLM is never called a second time.
    assert events[3] == ToolExecuted(
        tool_call_id="tc1",
        execution=ToolExecution(
            id="te1", parent_id="a1", created_at=1000,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="cooperative_sleep", arguments={}),
            tool_spec=ToolSpec(
                name="cooperative_sleep",
                description="Sleep, but wind down and return early when cancelled.",
            ),
            extras={},
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[
                ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
            ],
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(
                content=[TextContent(text="cut short: 2 of 10 rows written")],
                metadata={"cut_short": True},
                is_error=False,
            ),
            error=None,
            started_at=1000,
            ended_at=1000,
            cancel_signalled_at=1000,
            updated_at=1000,
            is_doom_loop_flagged=False,
            context_tokens=7,  # len("cut short: 2 of 10 rows written") // 4
        ),
        result_text="cut short: 2 of 10 rows written",
        is_error=False,
    )
    assert runner.session == AgentSession(
        id="s1",
        entries={
            "u1": UserMessage(
                id="u1", parent_id=None, created_at=1000,
                parts=[TextContent(text="Sleep cooperatively")],
                context_tokens=4,  # len("Sleep cooperatively") // 4
            ),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=1000),
            "a1": AssistantMessage(
                id="a1", parent_id="ts", created_at=1000,
                parts=[ToolCall(id="tc1", name="cooperative_sleep", arguments={})],
                llm_config=LLMConfig(model="test-model", provider="faux"),
                stop_reason="tool_use",
                context_tokens=4,  # (name + JSON args) // 4
            ),
            "te1": ToolExecution(
                id="te1", parent_id="a1", created_at=1000,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="cooperative_sleep", arguments={}),
                tool_spec=ToolSpec(
                    name="cooperative_sleep",
                    description="Sleep, but wind down and return early when cancelled.",
                ),
                extras={},
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[
                    ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
                ],
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(
                    content=[TextContent(text="cut short: 2 of 10 rows written")],
                    metadata={"cut_short": True},
                    is_error=False,
                ),
                error=None,
                started_at=1000,
                ended_at=1000,
                cancel_signalled_at=1000,
                updated_at=1000,
                is_doom_loop_flagged=False,
                context_tokens=7,
            ),
            "cr": CancelRequested(
                id="cr", parent_id="te1", created_at=1000,
                outcome=TurnOutcome.CANCELLED,
                error=None,
            ),
            "tf": TurnFinish(
                id="tf", parent_id="cr", created_at=1000,
                outcome=TurnOutcome.CANCELLED,
                error=None,
            ),
        },
        tool_executions={"tc1": ["te1"]},
        usages={"c1": {
            "a1": Usage(conversation_id="c1", entry_id="a1"),
        }},
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts", "a1", "te1", "cr", "tf"],
            created_at=500,
            updated_at=1000,
            status=ConversationStatus.IDLE,
        ),
        session_config=SessionConfig(
            llm_config=LLMConfig(model="test-model", provider="faux"),
            runtime_config=RuntimeConfig(tool_cancellation_grace_period=1000),
        ),
    )


async def test_timed_out_tool_call_full_session_shape():
    # ── precondition ─────────────────────────────────────────────────────────
    # Same empty IDLE session; the tool carries its own 50 ms deadline and
    # sleeps past it. Nothing cancels the turn, so the loop keeps going: the
    # timed-out call becomes an ordinary (resultless, errorless, is_error on
    # the wire) tool output and the model answers. Entry ids in creation order:
    #   u1 (post_message), ts (TurnStart), a1 (tool-call AssistantMessage),
    #   te1 (ToolExecution), a2 (final AssistantMessage), tf (TurnFinish).
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("slow", {}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It timed out.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s1",
        entries={},
        active_conversation=Conversation(id="c1", nodes=[], created_at=500, updated_at=500),
        session_config=SessionConfig(
            llm_config=LLMConfig(model="test-model", provider="faux"),
        ),
    )
    runner = ScriptedRunner(
        session,
        tool_registry=InlineToolRegistry([SlowTool()]),
        provider=faux,
        ids=["u1", "ts", "a1", "te1", "a2", "tf"],
    )

    # ── action ───────────────────────────────────────────────────────────────
    runner.post_message("Run the slow tool")
    async with runner.run() as run:
        events = [event async for event in run]

    # ── postcondition ────────────────────────────────────────────────────────
    # The deadline hard-cancelled the tool: TIMED_OUT, resultless, errorless,
    # `cancel_signalled_at` untouched (a deadline is NOT a run cancellation),
    # derived placeholder output — and, unlike a cancel, the turn runs to
    # completion.
    assert events[3] == ToolExecuted(
        tool_call_id="tc1",
        execution=ToolExecution(
            id="te1", parent_id="a1", created_at=1000,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="slow", arguments={}),
            tool_spec=ToolSpec(
                name="slow", description="Sleep past its own deadline.",
                timeout_in_ms=50,
            ),
            extras={},
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[
                ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
            ],
            status=ExecutionStatus.TIMED_OUT,
            result=None,
            error=None,
            started_at=1000,
            ended_at=1000,
            cancel_signalled_at=None,
            updated_at=1000,
            is_doom_loop_flagged=False,
        ),
        result_text="[tool execution timed_out]",
        is_error=True,
    )
    assert events[4] == TextBlock(text="It timed out.")
    assert events[5] == FinishReason(finish_reason="stop")
    assert runner.session == AgentSession(
        id="s1",
        entries={
            "u1": UserMessage(
                id="u1", parent_id=None, created_at=1000,
                parts=[TextContent(text="Run the slow tool")],
                context_tokens=4,  # len("Run the slow tool") // 4
            ),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=1000),
            "a1": AssistantMessage(
                id="a1", parent_id="ts", created_at=1000,
                parts=[ToolCall(id="tc1", name="slow", arguments={})],
                llm_config=LLMConfig(model="test-model", provider="faux"),
                stop_reason="tool_use",
                context_tokens=1,  # (name + JSON args) // 4
            ),
            "te1": ToolExecution(
                id="te1", parent_id="a1", created_at=1000,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="slow", arguments={}),
                tool_spec=ToolSpec(
                    name="slow", description="Sleep past its own deadline.",
                    timeout_in_ms=50,
                ),
                extras={},
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[
                    ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
                ],
                status=ExecutionStatus.TIMED_OUT,
                result=None,
                error=None,
                started_at=1000,
                ended_at=1000,
                cancel_signalled_at=None,
                updated_at=1000,
                is_doom_loop_flagged=False,
                context_tokens=0,
            ),
            "a2": AssistantMessage(
                id="a2", parent_id="te1", created_at=1000,
                parts=[TextContent(text="It timed out.")],
                llm_config=LLMConfig(model="test-model", provider="faux"),
                stop_reason="stop",
                context_tokens=3,  # len("It timed out.") // 4
            ),
            "tf": TurnFinish(id="tf", parent_id="a2", created_at=1000),
        },
        tool_executions={"tc1": ["te1"]},
        usages={"c1": {
            "a1": Usage(conversation_id="c1", entry_id="a1"),
            "a2": Usage(conversation_id="c1", entry_id="a2"),
        }},
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts", "a1", "te1", "a2", "tf"],
            created_at=500,
            updated_at=1000,
            status=ConversationStatus.IDLE,
        ),
        session_config=SessionConfig(
            llm_config=LLMConfig(model="test-model", provider="faux"),
        ),
    )
