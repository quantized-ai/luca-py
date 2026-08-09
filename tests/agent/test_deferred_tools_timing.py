"""Deferred tool calling under the two clocks: the tool deadline and the
cancel grace window (spec 0007).

The park has no clock of its own. What the deadline bounds is ONE POLL — the
body of a single dispatch — and re-dispatch is what makes that distinction
observable, because the same call is now bodied several times with unbounded,
unmeasured time in between:

- a poll that hangs past the deadline is TERMINAL, exactly like a first
  dispatch that hangs, and the audit list is what says which poll died:
  `[DEFERRED, TIMED_OUT]`;
- a deferral returned INSIDE the cancel grace window still parks, and the
  close-side settle then takes it — INTERRUPTED with its attempt closed
  DEFERRED, which is the fingerprint of a body that answered rather than one
  the grace machinery killed;
- the PARKED PERIOD IS UNBOUNDED: a `timeout_in_ms` that would kill a slow
  body does nothing to a call that sits parked across drives for longer than
  it. Nothing in the framework ages a deferral out. The driver owns the
  cadence and the abandonment.

House style, and it matters twice as much here: NEVER race two timed things.
Every scenario pairs one real (small) timer with a side that is instant or
hangs forever, so the timer is the only clock that decides anything.

`DeterministicRunner` freezes `now_ms()` at 1000, so every attempt opened and
closed inside one drive carries `started_at == ended_at == 1000` — the real
elapsed time lives in the asyncio deadline, never in the record.
"""

import asyncio

from luca.agent.core.context import CancellationToken
from luca.agent.core.models import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    ConversationStatus,
    ExecutionAttempt,
    ExecutionAttemptOutcome,
    ExecutionDeferred,
    ExecutionResult,
    ExecutionStatus,
    RuntimeConfig,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolSpec,
    TurnOutcome,
    TurnStart,
    UserMessage,
)
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text
from luca.client.types import TextBlock as LucaTextBlock, ToolMessage
from tests.agent.scenarios import (
    MODEL,
    DeferringTool,
    DeterministicRunner,
    FakeToolRegistry,
    conversation,
    make_session,
)

ALLOW_500 = ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=500)

DEFERRED_ATTEMPT = ExecutionAttempt(
    outcome=ExecutionAttemptOutcome.DEFERRED,
    started_at=1000,
    ended_at=1000,
)
TIMED_OUT_ATTEMPT = ExecutionAttempt(
    outcome=ExecutionAttemptOutcome.TIMED_OUT,
    started_at=1000,
    ended_at=1000,
)


# ── tool doubles ───────────────────────────────────────────────────────────────


class DeferThenHangTool(DeferringTool):
    """Answers "not yet" on the first poll, then hangs forever on every one
    after it — the shape the PER-POLL deadline has to kill. It carries its own
    50ms deadline, so nothing here depends on the runtime config."""

    name = "slow_ask"
    description = "Defers once, then hangs; carries a 50ms deadline."
    timeout_in_ms = 50

    def __init__(self) -> None:
        super().__init__()
        self.cleaned_up = False

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult | ExecutionDeferred:
        self.dispatches.append(tool_call_id)
        if len(self.dispatches) == 1:
            return ExecutionDeferred()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cleaned_up = True
            raise
        return ExecutionResult(content=[TextContent(text="unreachable")])


class LateDeferringTool(DeferringTool):
    """Parks until the run's cancellation token trips, then answers "not yet" —
    a deferral landing INSIDE the cancel grace window. The token is the only
    clock: the tool returns the instant it fires, and the test's grace period
    is large enough never to expire."""

    name = "late_ask"
    description = "Defers the instant the run is cancelled."

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult | ExecutionDeferred:
        self.dispatches.append(tool_call_id)
        self.started.set()
        await cancellation_token.wait_cancelled()
        return ExecutionDeferred()


class TimedDeferringTool(DeferringTool):
    """A DEADLINE-CARRYING deferrer whose body is instant: every poll returns
    "not yet" immediately, so its own 50ms deadline can only ever fire if
    something outside a poll were being measured against it."""

    name = "timed_ask"
    description = "Defers instantly; carries a 50ms deadline."
    timeout_in_ms = 50


SLOW_ASK_SPEC = DeferThenHangTool().get_tool_spec()
LATE_ASK_SPEC = LateDeferringTool().get_tool_spec()
TIMED_ASK_SPEC = TimedDeferringTool().get_tool_spec()


# ── literal factories ──────────────────────────────────────────────────────────


def one_call_session(session_id: str, spec: ToolSpec, **runtime) -> AgentSession:
    """A turn already at the dispatch step: the model asked for `spec.name`,
    the execution was born and ALLOWED, and no body has run. The first `run()`
    dispatches it, so every scenario below is about what one poll does."""
    call = ToolCall(id="tc1", name=spec.name, arguments={"a": 1, "b": 2})
    return make_session(
        id=session_id,
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Which one?")]),
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
                tool_spec=spec,
                status=ExecutionStatus.PENDING,
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[ALLOW_500],
                updated_at=500,
            ),
        },
        conversations={"c1": conversation("c1", ["u1", "ts", "a1", "te1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(
            llm_config=MODEL,
            runtime_config=RuntimeConfig(**runtime) if runtime else RuntimeConfig(),
        ),
    )


def dispatched_execution(spec: ToolSpec, **over) -> ToolExecution:
    """The durable shape of `one_call_session`'s call after exactly one
    dispatch that parked."""
    fields: dict = {
        "id": "te1",
        "conversation_id": "c1",
        "parent_id": "a1",
        "created_at": 500,
        "tool_call_id": "tc1",
        "raw_tool_call": ToolCall(id="tc1", name=spec.name, arguments={"a": 1, "b": 2}),
        "tool_spec": spec,
        "tool_spec_id": spec.spec_id(),
        "status": ExecutionStatus.AWAITING_RESULT,
        "result": None,
        "error": None,
        "approval_status": ApprovalStatus.ALLOWED,
        "approval_decisions": [ALLOW_500],
        "attempts": [DEFERRED_ATTEMPT],
        "finished_at": None,
        "cancel_signalled_at": None,
        "updated_at": 1000,
        "context_tokens": 0,
    }
    fields.update(over)
    return ToolExecution(**fields)


# ── the deadline bounds ONE poll ───────────────────────────────────────────────


async def test_a_poll_that_hangs_past_the_deadline_times_the_whole_call_out():
    # ── precondition ─────────────────────────────────────────────────────────
    # One dispatch happened and parked; the tool will hang on the next one.
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("Gave up.")], finish_reason="stop")])
    tool = DeferThenHangTool()
    runner = DeterministicRunner(
        one_call_session("s_poll_timeout", SLOW_ASK_SPEC),
        tool_registry=FakeToolRegistry([tool]),
        provider=faux,
        ids=["a2", "tf"],
        now=1000,
    )
    await runner.run()

    # ── action ───────────────────────────────────────────────────────────────
    await runner.run()

    # ── postcondition ────────────────────────────────────────────────────────
    # A POLL THAT TIMES OUT IS TERMINAL. The deadline bounds a single body, and
    # a re-dispatched body is a body like any other: the second attempt closes
    # TIMED_OUT, the EXECUTION goes with it, and `[DEFERRED, TIMED_OUT]` is the
    # only place the record says which poll died. A deadline is not a cancel,
    # so `cancel_signalled_at` stays None.
    assert tool.dispatches == ["tc1", "tc1"]
    assert tool.cleaned_up is True  # the hard cancel was delivered
    assert runner.session.entries["te1"] == dispatched_execution(
        SLOW_ASK_SPEC,
        status=ExecutionStatus.TIMED_OUT,
        attempts=[DEFERRED_ATTEMPT, TIMED_OUT_ATTEMPT],
        finished_at=1000,
    )
    # terminal means the turn moves on: the derived output fed the model and
    # the turn completed, exactly as an undeferred timeout does
    assert faux.requests[0].messages[-1] == ToolMessage(
        tool_call_id="tc1",
        content=[LucaTextBlock(text="[tool execution timed_out]")],
        is_error=True,
    )
    assert runner.session.entries["tf"].outcome == TurnOutcome.COMPLETED
    assert runner.pending_deferred_tool_executions() == []
    assert runner.idle()


async def test_a_parked_call_is_not_aged_out_by_its_own_deadline():
    # ── precondition ─────────────────────────────────────────────────────────
    # A call parked by a tool carrying a 50ms deadline of its own.
    tool = TimedDeferringTool()
    runner = DeterministicRunner(
        one_call_session("s_unbounded_park", TIMED_ASK_SPEC),
        tool_registry=FakeToolRegistry([tool]),
        provider=FauxProvider(),
        now=1000,
    )
    await runner.run()

    # ── action ───────────────────────────────────────────────────────────────
    # Three times the tool's deadline, spent entirely PARKED.
    await asyncio.sleep(0.15)
    await runner.run()

    # ── postcondition ────────────────────────────────────────────────────────
    # THE PARKED PERIOD IS UNBOUNDED. `timeout_in_ms` covers the body of one
    # poll and nothing else — no clock spans the gap between polls, and nothing
    # in the framework ages a deferral out. The second poll runs and parks like
    # the first; the driver decides how long a deferral may live.
    assert tool.dispatches == ["tc1", "tc1"]
    assert runner.session.entries["te1"] == dispatched_execution(
        TIMED_ASK_SPEC,
        attempts=[DEFERRED_ATTEMPT, DEFERRED_ATTEMPT],
    )
    assert runner.session.get_conversation_status("c1").status == ConversationStatus.BLOCKED


# ── a deferral inside the cancel grace window ──────────────────────────────────


async def test_a_deferral_landing_within_the_cancel_grace_parks_then_settles_interrupted():
    # ── precondition ─────────────────────────────────────────────────────────
    # A live drive with the body in flight, under a grace period large enough
    # never to expire — the tool returns the instant the token trips, so the
    # token is the only clock in the scenario.
    tool = LateDeferringTool()
    runner = DeterministicRunner(
        one_call_session("s_grace_deferral", LATE_ASK_SPEC, tool_cancellation_grace_period=30_000),
        tool_registry=FakeToolRegistry([tool]),
        provider=FauxProvider(),
        ids=["cr", "tf"],
        now=1000,
    )
    run = runner.start()
    await tool.started.wait()

    # ── action ───────────────────────────────────────────────────────────────
    runner.cancel()
    result = await run

    # ── postcondition ────────────────────────────────────────────────────────
    # The deferral was HONOURED — it arrived within the grace window, so the
    # call parked — and the wind-down at the loop top then settled the park
    # INTERRUPTED, like any other parked call a close lands on. The audit list
    # is the proof of which of the two happened: ONE attempt, closed DEFERRED.
    # Had the grace expired instead, the same execution would carry a single
    # INTERRUPTED attempt, because the body would have been killed mid-run.
    assert tool.dispatches == ["tc1"]
    assert runner.session.entries["te1"] == dispatched_execution(
        LATE_ASK_SPEC,
        status=ExecutionStatus.INTERRUPTED,
        attempts=[DEFERRED_ATTEMPT],
        finished_at=1000,
        cancel_signalled_at=1000,
    )
    assert result.outcome == TurnOutcome.CANCELLED
    assert runner.pending_deferred_tool_executions() == []
    assert runner.idle()
