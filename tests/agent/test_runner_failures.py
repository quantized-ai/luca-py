"""Timeout & failure-outcome scenarios: the tool deadline (per-tool override
vs RuntimeConfig), crash recovery of orphaned RUNNING executions, the LLM
catch/close/re-raise site (TIMED_OUT / ERRORED → status PENDING, retry-ready),
and the §5.5 post_message matrix.

House style: precondition → one action → full-object postcondition; never
race two timed things — every deadline test pairs a real (small) timer with
a hang-forever await, so the timer is the only clock that matters.
"""

import asyncio

import pytest

from luca.agent.core.context import CancellationToken, ToolContext
from luca.agent.core.exceptions import AgentError
from luca.agent.core.models import (
    RuntimeConfig,
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    Conversation,
    ConversationStatus,
    ExecutionStatus,
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
from luca.agent.core.events import (
    FinishReason,
    TextBlock,
    TextDelta,
    TextStart,
    ToolCallReceived,
    ToolExecuted,
    ToolExecutionStarted,
)
from luca.agent.core.tools import Tool
from luca.client.exceptions import ProviderAPIError, StreamError
from luca.client.exceptions import TimeoutError as ClientTimeoutError
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_error,
    faux_hang,
    faux_text,
    faux_tool_call,
)
from luca.client.types import TextBlock as LucaTextBlock
from luca.client.types import ToolMessage
from luca.client.types import UserMessage as LucaUserMessage

from tests.agent.scenarios import (
    CANCEL_PARKED_SESSION,
    GATED_SESSION,
    MODEL,
    POST_FAILURE_SESSION,
    RUNNING_ORPHAN_SESSION,
    UNDECIDED_SESSION,
    AddTool,
    BinaryArgs,
    DeterministicRunner,
    FakeToolRegistry,
)


# ── tool doubles ───────────────────────────────────────────────────────────────


class CooperatingHangTool(Tool):
    """Hangs forever; cleans up observably when the deadline hard-cancels it."""

    name = "hang"
    description = "Hangs until the deadline kills it."
    Args = BinaryArgs

    def __init__(self) -> None:
        self.cleaned_up = False

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cleaned_up = True
            raise
        return "unreachable"


class FastHangTool(CooperatingHangTool):
    """The same hanger with a tiny per-tool deadline — beats any config."""

    name = "fast_hang"
    description = "Hangs; carries its own 50ms deadline."
    timeout_in_ms = 50


class StubbornHangTool(Tool):
    """Ignores the hard cancel (stands in for detached thread work): swallows
    the CancelledError and keeps hanging until the TEST releases it."""

    name = "stubborn"
    description = "Survives the hard cancel."
    Args = BinaryArgs

    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await self.release.wait()  # still not done — like a blocked thread
        return "finally finished"


# ── tool timeout (§5.1) ────────────────────────────────────────────────────────


async def test_tool_timeout_records_timed_out_and_the_turn_continues():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("hang", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("Couldn't compute.")], finish_reason="stop"),
    ])
    tool = CooperatingHangTool()
    session = AgentSession(
        id="s_tool_to",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(
            llm_config=MODEL,
            runtime_config=RuntimeConfig(tool_execution_timeout_in_ms=50),
        ),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([tool]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    assert [event.type for event in events] == [
        "finish_reason", "tool_call_received", "tool_execution_started",
        "tool_executed", "text_block", "finish_reason",
    ]
    assert events[3].result_text == "[tool execution timed_out]"
    assert events[3].is_error is True
    assert tool.cleaned_up is True  # the hard cancel was delivered
    execution = runner.session.entries["te1"]
    assert execution.status == ExecutionStatus.TIMED_OUT
    assert execution.result is None
    assert execution.error is None  # the status IS the complete lifecycle fact
    assert execution.started_at == 1000 and execution.ended_at == 1000
    assert execution.cancel_signalled_at is None  # a deadline is not a cancel
    # a tool deadline is NOT a turn failure: the derived output fed the
    # next model call and the turn completed
    assert faux.requests[1].messages[-1] == ToolMessage(
        tool_call_id="tc1",
        content=[LucaTextBlock(text="[tool execution timed_out]")],
        is_error=True,
    )
    assert runner.session.entries["tf"].outcome == TurnOutcome.COMPLETED
    assert runner.idle()


async def test_non_cooperating_hanger_is_recorded_on_time_and_detached():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("stubborn", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("Moving on.")], finish_reason="stop"),
    ])
    tool = StubbornHangTool()
    session = AgentSession(
        id="s_stubborn",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(
            llm_config=MODEL,
            runtime_config=RuntimeConfig(tool_execution_timeout_in_ms=50),
        ),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([tool]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    result = await runner.run()

    # recorded ON TIME — the run finished while the tool is still stuck
    assert result.outcome == TurnOutcome.COMPLETED
    assert runner.session.entries["te1"].status == ExecutionStatus.TIMED_OUT
    assert runner.idle()
    # let the detached task finish; its swallowed result must not leak
    # (warnings-as-errors would fail the test otherwise)
    tool.release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)


async def test_per_tool_deadline_beats_the_config_deadline():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("fast_hang", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("ok")], finish_reason="stop"),
    ])
    tool = FastHangTool()
    session = AgentSession(
        id="s_override",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(
            llm_config=MODEL,
            # effectively infinite next to the tool's own 50ms
            runtime_config=RuntimeConfig(tool_execution_timeout_in_ms=600_000),
        ),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([tool]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    result = await runner.run()

    assert result.outcome == TurnOutcome.COMPLETED
    assert tool.cleaned_up is True
    assert runner.session.entries["te1"].status == ExecutionStatus.TIMED_OUT


async def test_instant_tool_under_a_huge_deadline_is_unaffected():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_inert_deadline",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(
            llm_config=MODEL,
            runtime_config=RuntimeConfig(tool_execution_timeout_in_ms=600_000),
        ),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    assert events[3].type == "tool_executed"
    assert events[3].result_text == "3"
    assert events[3].is_error is False
    assert runner.session.entries["te1"].status == ExecutionStatus.COMPLETED
    assert runner.idle()


# ── crash recovery: orphaned RUNNING executions ────────────────────────────────


async def test_orphaned_running_execution_recovers_to_interrupted_without_redispatch():
    # a persisted RUNNING execution has no live task on the next drive: it is
    # terminalized INTERRUPTED before anything else — after_tool_execution
    # runs, the body is NEVER re-dispatched (no ToolExecutionStarted, no
    # result), and durable state records nothing crash-specific.
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("It was interrupted.")], finish_reason="stop"),
    ])
    session = RUNNING_ORPHAN_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["a2", "tf"], now=1000,
    )

    assert runner.pending()  # stale RUNNING status self-healed on construction
    async with runner.run() as run:
        events = [event async for event in run]

    interrupted = ToolExecution(
        id="te1", parent_id="a1", created_at=500,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ToolSpec(name="add", description="Add two numbers."),
        status=ExecutionStatus.INTERRUPTED,
        result=None,
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[
            ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=500),
        ],
        started_at=500,
        ended_at=1000,
        updated_at=1000,
    )
    assert events == [
        ToolExecuted(
            tool_call_id="tc1", execution=interrupted,
            result_text="[tool execution interrupted]", is_error=True,
        ),
        TextBlock(text="It was interrupted."),
        FinishReason(finish_reason="stop"),
    ]
    assert runner.session.entries["te1"] == interrupted
    # the model was called only after recovery, with the derived tool output
    assert faux.requests[0].messages[-1] == ToolMessage(
        tool_call_id="tc1",
        content=[LucaTextBlock(text="[tool execution interrupted]")],
        is_error=True,
    )
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf", parent_id="a2", created_at=1000,
    )
    assert runner.idle()


async def test_orphan_recovery_precedes_a_parked_cancel_flush():
    # recovery runs before the wind-down, so a call whose body actually
    # started is INTERRUPTED (no cancel_signalled_at — the cancellation never
    # reached the live body), never CANCELLED.
    session = RUNNING_ORPHAN_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]),
        provider=FauxProvider(), ids=["cr", "tf"], now=1000,
    )
    runner.cancel()

    result = await runner.run()  # the flush

    assert result.outcome == TurnOutcome.CANCELLED
    assert runner.session.entries["te1"].status == ExecutionStatus.INTERRUPTED
    assert runner.session.entries["te1"].cancel_signalled_at is None
    assert runner.session.entries["te1"].ended_at == 1000
    assert runner.idle()


# ── LLM failure: record, close, re-raise (§5.4) ───────────────────────────────


async def test_llm_timeout_closes_the_turn_and_reraises_through_await():
    # the high tier: RuntimeConfig.client_completion_timeout_in_ms wires into
    # the client's total_timeout=, whose expiry raises the SDK TimeoutError
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_hang()])])
    session = AgentSession(
        id="s_llm_to",
        entries={
            "u1": UserMessage(
                id="u1", created_at=500, parts=[TextContent(text="Add 1 and 2")],
            ),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(
            llm_config=MODEL,
            runtime_config=RuntimeConfig(client_completion_timeout_in_ms=50),
        ),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "tf"], now=1000,
    )
    run = runner.run()

    with pytest.raises(ClientTimeoutError):
        await run

    assert run.result is None  # the raise path never produces a result
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf", parent_id="ts", created_at=1000,
        outcome=TurnOutcome.TIMED_OUT,
        error="completion exceeded total_timeout=0.05s",
    )
    assert runner.session.active_conversation.nodes == ["u1", "ts", "tf"]
    assert runner.pending()  # retry-ready, no AssistantMessage recorded


async def test_scripted_client_timeout_is_indistinguishable_from_a_real_one():
    # the low tier: the runner cannot and should not tell a scripted
    # TimeoutError from an httpx/total one — same class, same close
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [], error=faux_error("connect timeout", error_class=ClientTimeoutError),
        ),
    ])
    session = AgentSession(
        id="s_llm_to2",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "tf"], now=1000,
    )

    with pytest.raises(ClientTimeoutError, match="connect timeout"):
        await runner.run()

    assert runner.session.entries["tf"] == TurnFinish(
        id="tf", parent_id="ts", created_at=1000,
        outcome=TurnOutcome.TIMED_OUT, error="connect timeout",
    )
    assert runner.pending()


async def test_llm_error_closes_the_turn_and_reraises_through_iteration():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [], error=faux_error("provider 500", error_class=ProviderAPIError),
        ),
    ])
    session = AgentSession(
        id="s_llm_err",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "tf"], now=1000,
    )
    run = runner.run()

    with pytest.raises(ProviderAPIError, match="provider 500"):
        async with run:
            _ = [event async for event in run]

    assert run.result is None
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf", parent_id="ts", created_at=1000,
        outcome=TurnOutcome.ERRORED, error="provider 500",
    )
    assert runner.pending()


async def test_streaming_llm_error_closes_the_turn_after_the_deltas():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_text("Hel")], error=faux_error("boom mid-stream"),
        ),
    ])
    session = AgentSession(
        id="s_stream_err",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "tf"], now=1000,
    )
    run = runner.run(streaming=True)
    events = []

    with pytest.raises(StreamError, match="boom mid-stream"):
        async with run:
            async for event in run:
                events.append(event)

    assert events == [TextStart(), TextDelta(text="Hel")]
    assert run.result is None
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf", parent_id="ts", created_at=1000,
        outcome=TurnOutcome.ERRORED, error="boom mid-stream",
    )
    assert runner.pending()  # the partial assistant message was dropped


async def test_post_failure_session_reloads_cold_and_a_new_turn_reanswers():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = POST_FAILURE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts2", "a1", "tf2"], now=2000,
    )

    assert runner.pending()  # derived from the trailing failed TurnFinish
    result = await runner.run()

    assert result.outcome == TurnOutcome.COMPLETED
    assert runner.session.active_conversation.nodes == [
        "u1", "ts", "tf", "ts2", "a1", "tf2",
    ]
    # the failed bracket projected nothing to the wire — just the user message
    assert faux.requests[0].messages == [
        LucaUserMessage(content=[LucaTextBlock(text="Add 1 and 2")]),
    ]
    assert runner.idle()


# ── the §5.5 post_message matrix ──────────────────────────────────────────────


async def test_post_message_is_legal_after_a_failed_turn():
    session = POST_FAILURE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["u2"], now=2000)

    runner.post_message("Take your time, retry.")

    assert runner.pending()
    assert runner.session.active_conversation.nodes == ["u1", "ts", "tf", "u2"]


async def test_post_message_rejects_awaiting_approval():
    session = GATED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()], decisions=[]),
        provider=FauxProvider(), now=1000,
    )

    with pytest.raises(AgentError):
        runner.post_message("never mind")


async def test_post_message_rejects_cancelling():
    session = CANCEL_PARKED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()], decisions=[]),
        provider=FauxProvider(), now=1000,
    )

    with pytest.raises(AgentError):
        runner.post_message("never mind")


async def test_post_message_rejects_an_open_resumable_bracket():
    # PENDING status alone is not enough — the bracket must be closed
    session = UNDECIDED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()], decisions=[]),
        provider=FauxProvider(), now=1000,
    )

    assert runner.pending()
    with pytest.raises(AgentError):
        runner.post_message("also this")
