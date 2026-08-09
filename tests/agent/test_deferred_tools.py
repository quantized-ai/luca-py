"""Deferred tool calling: a tool that answers "not yet" (spec 0007).

A prepared callable may return `ExecutionDeferred()` instead of an
`ExecutionResult`. The runner records that attempt, parks the open turn at
`AWAITING_RESULT`, and returns; the application resolves whatever the tool is
waiting on and drives again.

THE ONE FACT EVERYTHING RESTS ON: there is no resume, only RE-DISPATCH. When a
tool defers, nothing about that call stays alive — no callable, no coroutine,
no runner state — so the next drive runs the identical dispatch path from
scratch: `before_tool_execution` → `prepare()` → a brand-new callable → invoke.
Three consequences fall out and are pinned below: `prepare()` re-runs every
poll, `status` cycles `AWAITING_RESULT → RUNNING → AWAITING_RESULT`, and
`ToolExecutionStarted` fires once per poll.

What this file pins, and what each one would break:

- the park itself: `AWAITING_RESULT`, one closed `ExecutionAttempt`, no
  `finished_at`, no `ToolExecuted`, no `after_tool_execution`, 0 context tokens;
- PARK PLACEMENT, twice. Ahead of the progress-continue, or a re-dispatched
  deferral counts as progress and the drive re-selects the same execution and
  spins as fast as Python runs. Ahead of the unresolved-children park, or a
  COMPLETED SIBLING wakes the model with no post anywhere;
- THE THREE STATUS TERMS. A parked call is not advanceable, is not runnable
  (the trap: the runnable fallback reads "nothing left but gates", and a parked
  call is not a gate), and blocks the subagent branch like a gate does;
- approval is asked once per CALL, never once per dispatch;
- the projection carve-out: `AWAITING_RESULT_OUTPUT`, `is_error=False`, and a
  post that reaches the model past a parked call;
- the SHARED close-side settle: `cancel()`, `hard_max_steps` and an LLM failure
  all give a parked call INTERRUPTED, and none of them touches the audit list —
  nothing was running when they landed, so its last attempt is already closed
  as DEFERRED;
- orphan recovery closing a dangling attempt as DISCARDED, which is the one
  case the audit log exists to explain;
- a RUNTIME-MINTED call may not defer: it has no durable re-selection path, so
  the deferral records as an ordinary tool failure.

Determinism comes from `DeterministicRunner` (`scenarios.py`) with the clock
frozen at 1000, so an attempt opened and closed inside one drive carries
`started_at == ended_at == 1000`.
"""

import asyncio

from luca.agent.core.context import CancellationToken
from luca.agent.core.events import (
    FinishReason,
    TextBlock,
    ToolCallReceived,
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
    RuntimeConfig,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    TurnOutcome,
    TurnStart,
    UserMessage,
)
from luca.agent.core.projection import ConversationProjector
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
    ASK_SPEC,
    MODEL,
    AddTool,
    DeferringTool,
    DeterministicRunner,
    FakeToolRegistry,
    conversation,
    main_conversation,
    make_session,
)

ASK_SPEC_ID = ASK_SPEC.spec_id()
ADD_SPEC_ID = ADD_SPEC.spec_id()

ASK_CALL = ToolCall(id="tc1", name="ask", arguments={"a": 1, "b": 2})
ADD_CALL = ToolCall(id="tc2", name="add", arguments={"a": 3, "b": 4})

ALLOW_500 = ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=500)
ALLOW_1000 = ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)

DEFERRED_ATTEMPT = ExecutionAttempt(
    outcome=ExecutionAttemptOutcome.DEFERRED,
    started_at=1000,
    ended_at=1000,
)
COMPLETED_ATTEMPT = ExecutionAttempt(
    outcome=ExecutionAttemptOutcome.COMPLETED,
    started_at=1000,
    ended_at=1000,
)


def asking_session(session_id: str = "s_ask", **runtime) -> AgentSession:
    """A turn already at the dispatch step: the model asked for `ask(1, 2)`,
    the execution was born and ALLOWED, and no body has run.

    Starting here rather than from an LLM round keeps every scenario about the
    DEFERRAL — the first `run()` dispatches, the tool says "not yet", and the
    drive parks without ever calling the model."""
    return make_session(
        id=session_id,
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Which one?")]),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
            "a1": AssistantMessage(
                id="a1",
                parent_id="ts",
                created_at=500,
                parts=[ASK_CALL],
                llm_config=MODEL,
                stop_reason="tool_use",
            ),
            "te1": ToolExecution(
                id="te1",
                conversation_id="c1",
                parent_id="a1",
                created_at=500,
                tool_call_id="tc1",
                raw_tool_call=ASK_CALL,
                tool_spec=ASK_SPEC,
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


def parked_execution(**over) -> ToolExecution:
    """The durable shape of one parked call, after exactly one dispatch."""
    fields: dict = {
        "id": "te1",
        "conversation_id": "c1",
        "parent_id": "a1",
        "created_at": 500,
        "tool_call_id": "tc1",
        "raw_tool_call": ASK_CALL,
        "tool_spec": ASK_SPEC,
        "tool_spec_id": ASK_SPEC_ID,
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


# ── the park ───────────────────────────────────────────────────────────────────


async def test_a_deferring_tool_parks_the_open_turn():
    # ── precondition ─────────────────────────────────────────────────────────
    # A turn at the dispatch step, and a tool with no answer for `tc1`.
    faux = FauxProvider()
    tool = DeferringTool()
    registry = FakeToolRegistry([tool])
    runner = DeterministicRunner(
        asking_session(),
        tool_registry=registry,
        provider=faux,
        now=1000,
    )

    # ── action ───────────────────────────────────────────────────────────────
    result = await runner.run()

    # ── postcondition ────────────────────────────────────────────────────────
    # AWAITING_RESULT with ONE closed attempt and no `finished_at` — the call
    # has not finished. `context_tokens` stays 0 (nonterminal), the turn is
    # still open (no `TurnFinish`), and the model was never called.
    assert runner.session.entries["te1"] == parked_execution()
    assert runner.session.conversations["c1"].nodes == ["u1", "ts", "a1", "te1"]
    assert faux.requests == []
    assert result.status == ConversationStatus.BLOCKED
    assert result.outcome is None
    assert result.pending_deferred_tool_executions == [parked_execution()]


async def test_the_parked_drive_emits_started_and_no_terminal_event():
    faux = FauxProvider()
    registry = FakeToolRegistry([DeferringTool()])
    runner = DeterministicRunner(asking_session(), tool_registry=registry, provider=faux, now=1000)

    async with runner.run() as run:
        events = [event async for event in run]

    # `ToolExecutionStarted` fires because the body WAS dispatched; nothing
    # terminal follows, because a deferral is not an outcome. That is exactly
    # why no consumer can tell parked from in-flight off the stream, and why
    # discovery is a poll over the session.
    assert events == [
        ToolExecutionStarted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=ToolExecution(
                id="te1",
                conversation_id="c1",
                parent_id="a1",
                created_at=500,
                tool_call_id="tc1",
                raw_tool_call=ASK_CALL,
                tool_spec=ASK_SPEC,
                tool_spec_id=ASK_SPEC_ID,
                status=ExecutionStatus.RUNNING,
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[ALLOW_500],
                attempts=[ExecutionAttempt(started_at=1000)],
                updated_at=1000,
            ),
        ),
    ]


async def test_a_deferral_runs_no_outcome_middleware():
    class Recorder:
        def __init__(self) -> None:
            self.after: list[ToolExecution] = []
            self.before: list[ToolExecution] = []

        def before_tool_execution(self, session, conversation_id, execution):
            self.before.append(execution)
            return execution

        def after_tool_execution(self, session, conversation_id, execution, exception=None):
            self.after.append(execution)
            return execution

    recorder = Recorder()
    runner = DeterministicRunner(
        asking_session(),
        tool_registry=FakeToolRegistry([DeferringTool()]),
        provider=FauxProvider(),
        now=1000,
        middleware=[recorder],
    )

    await runner.run()

    # The dispatch hook fires (a dispatch attempt really started); the OUTCOME
    # hook does not, because there is no outcome.
    assert [execution.id for execution in recorder.before] == ["te1"]
    assert recorder.after == []


# ── re-dispatch ────────────────────────────────────────────────────────────────


async def test_resolving_and_driving_again_completes_the_same_call():
    # ── precondition ─────────────────────────────────────────────────────────
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("Got it.")], finish_reason="stop")])
    tool = DeferringTool()
    registry = FakeToolRegistry([tool])
    runner = DeterministicRunner(
        asking_session(),
        tool_registry=registry,
        provider=faux,
        ids=["a2", "tf"],
        now=1000,
    )
    await runner.run()

    # ── action ───────────────────────────────────────────────────────────────
    tool.answers["tc1"] = "the second one"
    await runner.run()

    # ── postcondition ────────────────────────────────────────────────────────
    # ONE execution under ONE tool_call_id, with TWO attempts: the deferral and
    # the resolution. `prepare()` ran once per dispatch — it is contractually
    # re-callable, which is what makes re-dispatch-from-scratch legal.
    assert runner.session.entries["te1"] == parked_execution(
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="the second one")]),
        attempts=[DEFERRED_ATTEMPT, COMPLETED_ATTEMPT],
        finished_at=1000,
        context_tokens=len("the second one") // 4,
    )
    assert registry.prepared == ["ask", "ask"]
    assert tool.dispatches == ["tc1", "tc1"]
    assert runner.status == ConversationStatus.IDLE


async def test_a_second_drive_that_still_defers_polls_exactly_once():
    # PARK PLACEMENT RULE 1. A re-dispatched deferral IS progress by the
    # progress-continue's test, so a park branch placed after it would
    # re-select the same execution and spin as fast as Python runs — no LLM
    # cost, no limit tripped, and no test failure unless someone counts the
    # attempts. This is that count.
    tool = DeferringTool()
    runner = DeterministicRunner(
        asking_session(),
        tool_registry=FakeToolRegistry([tool]),
        provider=FauxProvider(),
        now=1000,
    )
    await runner.run()

    await runner.run()

    assert tool.dispatches == ["tc1", "tc1"]
    assert runner.session.entries["te1"] == parked_execution(attempts=[DEFERRED_ATTEMPT, DEFERRED_ATTEMPT])


async def test_a_completed_sibling_does_not_wake_the_model():
    # PARK PLACEMENT RULE 2. The unresolved-children park's wake rule counts
    # ANY terminal execution as fresh material, so a park branch sitting after
    # it would call the model because a SIBLING TOOL completed, with no post
    # anywhere. "A post is the only thing that projects a parked call" is what
    # placing the branch ahead of it keeps true.
    faux = FauxProvider()
    session = asking_session()
    session.entries["a1"].parts = [ASK_CALL, ADD_CALL]
    session.entries["te2"] = ToolExecution(
        id="te2",
        conversation_id="c1",
        parent_id="te1",
        created_at=500,
        tool_call_id="tc2",
        raw_tool_call=ADD_CALL,
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.PENDING,
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[ALLOW_500],
        updated_at=500,
    )
    session.tool_specs[ADD_SPEC_ID] = ADD_SPEC
    main_conversation(session).nodes.append("te2")
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([DeferringTool(), AddTool()]),
        provider=faux,
        now=1000,
    )

    result = await runner.run()

    assert runner.session.entries["te2"].status == ExecutionStatus.COMPLETED
    assert faux.requests == []
    assert result.status == ConversationStatus.BLOCKED


async def test_approval_is_asked_once_per_call_not_once_per_dispatch():
    tool = DeferringTool()
    registry = FakeToolRegistry([tool])
    runner = DeterministicRunner(
        asking_session(),
        tool_registry=registry,
        provider=FauxProvider(),
        now=1000,
    )
    await runner.run()

    await runner.run()

    # The decide step only ever considers PENDING executions, and a parked one
    # is AWAITING_RESULT — so a call approved once stays approved for every
    # poll. It is one tool call.
    assert registry.seen == []
    assert runner.session.entries["te1"].approval_decisions == [ALLOW_500]


# ── discovery ──────────────────────────────────────────────────────────────────


async def test_pending_deferred_tool_executions_before_and_after():
    tool = DeferringTool()
    runner = DeterministicRunner(
        asking_session(),
        tool_registry=FakeToolRegistry([tool]),
        provider=FauxProvider(),
        now=1000,
    )
    await runner.run()
    assert runner.pending_deferred_tool_executions() == [parked_execution()]

    tool.answers["tc1"] = "done"
    faux_final = FauxProvider()
    faux_final.set_responses([faux_assistant_message([faux_text("ok")], finish_reason="stop")])
    runner = DeterministicRunner(
        runner.session,
        tool_registry=FakeToolRegistry([tool]),
        provider=faux_final,
        ids=["a2", "tf"],
        now=1000,
    )
    await runner.run()

    assert runner.pending_deferred_tool_executions() == []


# ── status ─────────────────────────────────────────────────────────────────────


async def test_a_parked_call_derives_blocked():
    # TERM 2 — `open_turn_is_runnable`'s fallback, and the trap. It reads
    # "nothing left except gates"; a parked call is not a gate, so without
    # naming it explicitly this derives BUSY and a `while not idle(): run()`
    # driver spins on a drive that only parks.
    runner = DeterministicRunner(
        asking_session(),
        tool_registry=FakeToolRegistry([DeferringTool()]),
        provider=FauxProvider(),
        now=1000,
    )
    await runner.run()

    assert runner.session.get_conversation_status("c1").status == ConversationStatus.BLOCKED
    assert runner.blocked() is True


async def test_a_parked_call_is_not_advanceable():
    # TERM 1 — no code of its own; it falls through every clause. Pinned so a
    # future edit cannot quietly make it advanceable again.
    from luca.agent.core.models import open_turn_has_advanceable_executions, open_turn_has_parked_executions

    runner = DeterministicRunner(
        asking_session(),
        tool_registry=FakeToolRegistry([DeferringTool()]),
        provider=FauxProvider(),
        now=1000,
    )
    await runner.run()
    nodes = runner.session.conversations["c1"].nodes
    entries = runner.session.entries

    assert open_turn_has_advanceable_executions(nodes, entries) is False
    assert open_turn_has_parked_executions(nodes, entries) is True


# ── projection ─────────────────────────────────────────────────────────────────


async def test_a_parked_execution_projects_the_placeholder():
    runner = DeterministicRunner(
        asking_session(),
        tool_registry=FakeToolRegistry([DeferringTool()]),
        provider=FauxProvider(),
        now=1000,
    )
    await runner.run()

    # `build_messages()` is public, so a debug view or a token counter can
    # force this projection even with nobody posting anything.
    assert runner.build_messages("c1")[-1] == ToolMessage(
        tool_call_id="tc1",
        content=[LucaTextBlock(text=ConversationProjector.AWAITING_RESULT_OUTPUT)],
        is_error=False,
    )


async def test_a_post_reaches_the_model_past_a_parked_call_and_then_re_parks():
    # ── precondition ─────────────────────────────────────────────────────────
    tool = DeferringTool()
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("Still waiting on you.")], finish_reason="stop")])
    runner = DeterministicRunner(
        asking_session(),
        tool_registry=FakeToolRegistry([tool]),
        provider=faux,
        ids=["u2", "a2"],
        now=1000,
    )
    await runner.run()

    # ── action ───────────────────────────────────────────────────────────────
    runner.post_message("actually, hurry up")
    assert runner.session.get_conversation_status("c1").status == ConversationStatus.BUSY
    async with runner.run() as run:
        events = [event async for event in run]

    # ── postcondition ────────────────────────────────────────────────────────
    # Exactly ONE model round, with the parked call on the wire as its
    # placeholder — and then the drive re-parks. The turn does NOT close: a
    # live deferral keeps it open, or the call would be stranded outside any
    # open turn with its placeholder frozen into the history.
    #
    # ONE POLL, ONE ROUND. The drive re-dispatches once on its way to the model
    # — the answer may have landed while the user was typing — and the pass
    # AFTER the round does not poll again: nothing happened in between that
    # could have resolved the call, and only a `notify()` re-arms the poll
    # inside a live drive.
    assert len(faux.requests) == 1
    assert faux.requests[0].messages[-2] == ToolMessage(
        tool_call_id="tc1",
        content=[LucaTextBlock(text=ConversationProjector.AWAITING_RESULT_OUTPUT)],
        is_error=False,
    )
    assert events == [
        ToolExecutionStarted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=runner.session.entries["te1"].model_copy(
                update={
                    "status": ExecutionStatus.RUNNING,
                    "attempts": [DEFERRED_ATTEMPT, ExecutionAttempt(started_at=1000)],
                },
            ),
        ),
        TextBlock(conversation_id="c1", text="Still waiting on you."),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert runner.session.entries["te1"] == parked_execution(
        attempts=[DEFERRED_ATTEMPT, DEFERRED_ATTEMPT],
    )
    assert runner.session.get_conversation_status("c1").status == ConversationStatus.BLOCKED
    assert runner.pending_deferred_tool_executions() == [runner.session.entries["te1"]]
    assert tool.answers == {}


# ── the close-side settle ──────────────────────────────────────────────────────


async def test_cancel_while_parked_records_interrupted_with_one_attempt():
    runner = DeterministicRunner(
        asking_session(),
        tool_registry=FakeToolRegistry([DeferringTool()]),
        provider=FauxProvider(),
        ids=["cr", "tf"],
        now=1000,
    )
    await runner.run()

    runner.cancel()
    result = await runner.run()

    # INTERRUPTED, not CANCELLED: the body demonstrably STARTED, and CANCELLED
    # is defined as "cancellation prevented the body from starting". ONE
    # attempt, because nothing was running when the cancel landed — the last
    # one was already closed as DEFERRED.
    assert runner.session.entries["te1"] == parked_execution(
        status=ExecutionStatus.INTERRUPTED,
        finished_at=1000,
        cancel_signalled_at=1000,
    )
    assert result.outcome == TurnOutcome.CANCELLED
    assert runner.session.get_conversation_status("c1").status == ConversationStatus.IDLE


async def test_hard_max_steps_settles_a_parked_call_through_the_same_bucket():
    # THE SETTLE IS SHARED. A post drives a real round while a call is parked,
    # and those rounds count — so `hard_max_steps` can land on a live deferral.
    # A bucket in the cancel path alone would strand it outside any open turn.
    tool = DeferringTool()
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("hold on")], finish_reason="stop")])
    runner = DeterministicRunner(
        asking_session(hard_max_steps=1),
        tool_registry=FakeToolRegistry([tool]),
        provider=faux,
        ids=["u2", "tf"],
        now=1000,
    )
    await runner.run()

    runner.post_message("hurry")
    result = await runner.run()

    # TWO attempts: the drive polls once on its way to the model, and the
    # limit trips before the round happens. Both are DEFERRED — nothing was
    # running when the close landed, which is why the settle gives the
    # EXECUTION INTERRUPTED without touching the audit list.
    assert runner.session.entries["te1"] == parked_execution(
        status=ExecutionStatus.INTERRUPTED,
        attempts=[DEFERRED_ATTEMPT, DEFERRED_ATTEMPT],
        finished_at=1000,
        cancel_signalled_at=1000,
    )
    assert result.outcome == TurnOutcome.ERRORED
    assert faux.requests == []


# ── crash recovery ─────────────────────────────────────────────────────────────


async def test_a_crash_mid_poll_recovers_to_interrupted_with_a_discarded_attempt():
    # ── precondition ─────────────────────────────────────────────────────────
    # A session as a crashed process left it: the execution persisted RUNNING
    # with an OPEN attempt, no live task behind it.
    session = asking_session("s_crashed")
    session.entries["te1"] = session.entries["te1"].model_copy(
        update={
            "status": ExecutionStatus.RUNNING,
            "attempts": [ExecutionAttempt(started_at=600)],
            "updated_at": 600,
        },
    )
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("I'll try again.")], finish_reason="stop")])
    tool = DeferringTool()
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([tool]),
        provider=faux,
        ids=["a2", "tf"],
        now=1000,
    )

    # ── action ───────────────────────────────────────────────────────────────
    await runner.run()

    # ── postcondition ────────────────────────────────────────────────────────
    # The EXECUTION is INTERRUPTED and the dangling ATTEMPT is DISCARDED, and
    # the distinction is the point: INTERRUPTED means the framework killed the
    # body and knows it did not finish; DISCARDED means the process died and
    # nobody ever saw how that run went. The tool is never re-dispatched.
    assert runner.session.entries["te1"] == parked_execution(
        status=ExecutionStatus.INTERRUPTED,
        attempts=[
            ExecutionAttempt(
                outcome=ExecutionAttemptOutcome.DISCARDED,
                started_at=600,
                ended_at=1000,
            ),
        ],
        finished_at=1000,
        context_tokens=0,
    )
    assert tool.dispatches == []


# ── failures on a re-dispatch ──────────────────────────────────────────────────


class BreakingRegistry(FakeToolRegistry):
    """Resolves normally on the first `prepare()` and raises on every one
    after it — the shape a re-dispatch has to survive."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0

    async def prepare(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> PreparedTool:
        self.calls += 1
        if self.calls > 1:
            raise ToolNotFound(f"Unknown tool: {tool_execution.raw_tool_call.name!r}.")
        return await super().prepare(session, conversation_id, tool_execution)


async def test_a_prepare_that_raises_on_the_second_poll_appends_no_attempt():
    # THE PREPARE-FAILURE INVARIANT, on a re-dispatch: "a failed `prepare()`
    # appends no attempt". The older phrasing ("leaves `started_at` unset") is
    # false here — poll 1 already recorded one, so this terminal execution
    # demonstrably WAS dispatched once.
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("oh well")], finish_reason="stop")])
    runner = DeterministicRunner(
        asking_session(),
        tool_registry=BreakingRegistry([DeferringTool()]),
        provider=faux,
        ids=["a2", "tf"],
        now=1000,
    )
    await runner.run()

    await runner.run()

    execution = runner.session.entries["te1"]
    assert execution.status == ExecutionStatus.NOT_FOUND
    assert execution.attempts == [DEFERRED_ATTEMPT]
    assert execution.finished_at == 1000
    assert execution.dispatched is True


class HangingPrepareRegistry(FakeToolRegistry):
    """Resolves instantly on the first `prepare()` and hangs on every one
    after it — the window a `cancel()` lands in during a RE-dispatch."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0

    async def prepare(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> PreparedTool:
        self.calls += 1
        if self.calls > 1:
            await asyncio.Event().wait()  # forever; the cancel is the only clock
        return await super().prepare(session, conversation_id, tool_execution)


async def test_cancel_during_a_re_dispatch_records_interrupted_not_cancelled():
    # THE SAME USER ACTION MUST RECORD THE SAME STATUS. A cancel that lands
    # while the call sits parked settles it INTERRUPTED (the body demonstrably
    # started); one that lands a moment later, inside the re-dispatch's own
    # `prepare()`, must not record CANCELLED — the status defined as
    # "cancellation prevented the body from starting" — on an execution whose
    # audit list already holds a DEFERRED attempt.
    runner = DeterministicRunner(
        asking_session(),
        tool_registry=HangingPrepareRegistry([DeferringTool()]),
        provider=FauxProvider(),
        ids=["cr", "tf"],
        now=1000,
    )
    await runner.run()

    run = runner.run()
    task = asyncio.ensure_future(run._wait())
    await asyncio.sleep(0)
    runner.cancel()
    await task

    assert runner.session.entries["te1"] == parked_execution(
        status=ExecutionStatus.INTERRUPTED,
        finished_at=1000,
        cancel_signalled_at=1000,
    )


class RaisingOnSecondTool(DeferringTool):
    """Defers once, then raises — a poll that fails is TERMINAL."""

    async def execute(self, args, session, conversation_id, *, tool_name, tool_call_id, cancellation_token):
        self.dispatches.append(tool_call_id)
        if len(self.dispatches) == 1:
            from luca.agent.core.models import ExecutionDeferred

            return ExecutionDeferred()
        raise ValueError("kaboom")


async def test_a_poll_that_raises_is_terminal_and_closes_its_own_attempt():
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("oh well")], finish_reason="stop")])
    runner = DeterministicRunner(
        asking_session(),
        tool_registry=FakeToolRegistry([RaisingOnSecondTool()]),
        provider=faux,
        ids=["a2", "tf"],
        now=1000,
    )
    await runner.run()

    await runner.run()

    execution = runner.session.entries["te1"]
    assert execution.status == ExecutionStatus.FAILED
    assert execution.attempts == [
        DEFERRED_ATTEMPT,
        ExecutionAttempt(outcome=ExecutionAttemptOutcome.FAILED, started_at=1000, ended_at=1000),
    ]
    assert execution.finished_at == 1000


# ── derived properties ─────────────────────────────────────────────────────────


async def test_dispatched_and_duration_read_from_the_new_fields():
    tool = DeferringTool()
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("ok")], finish_reason="stop")])
    runner = DeterministicRunner(
        asking_session(),
        tool_registry=FakeToolRegistry([tool]),
        provider=faux,
        ids=["a2", "tf"],
        now=2500,
    )
    await runner.run()
    parked = runner.session.entries["te1"]

    tool.answers["tc1"] = "at last"
    await runner.run()

    # `dispatched` is `bool(attempts)`; `duration_ms` is TOTAL time outstanding
    # (`finished_at - created_at`), which is the number a deferred call makes
    # meaningful — body time is per-attempt.
    assert (parked.dispatched, parked.duration_ms) == (True, None)
    settled = runner.session.entries["te1"]
    assert (settled.dispatched, settled.duration_ms) == (True, 2000)


# ── a runtime-minted call may not defer ────────────────────────────────────────


async def test_a_runtime_minted_call_may_not_defer():
    # `_derive_child_result` mints a FRESH `ToolCall` per pass and only
    # resolves the `ChildConversation` on `ToolExecuted`, so a deferral there
    # would never resolve the link, the next iteration would mint ANOTHER
    # result execution, and the drive would spin — no LLM cost, no limit to
    # trip. It is a registry contract violation, recorded as a tool failure.
    runner = DeterministicRunner(
        asking_session(),
        tool_registry=FakeToolRegistry([DeferringTool()]),
        provider=FauxProvider(),
        ids=["te9"],
        now=1000,
    )
    call = ToolCall(id="rt1", name="ask", arguments={"a": 1, "b": 2})

    events = [event async for event in runner._invoke_runtime_tool("c1", call, CancellationToken())]

    executed = [event for event in events if isinstance(event, ToolExecuted)]
    assert len(executed) == 1
    execution = executed[0].execution
    assert execution.status == ExecutionStatus.FAILED
    assert execution.error is not None
    assert "cannot be re-dispatched" in execution.error.error_message
    assert execution.attempts == [
        ExecutionAttempt(outcome=ExecutionAttemptOutcome.DEFERRED, started_at=1000, ended_at=1000),
    ]


# ── the full first round, from an LLM call ─────────────────────────────────────


async def test_the_whole_first_round_from_a_model_request():
    # The one scenario that starts from an empty session, so the event stream
    # covers birth → dispatch → park in one place.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("ask", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
        ]
    )
    session = make_session(
        id="s_fresh",
        entries={},
        conversations={"c1": conversation("c1", [], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([DeferringTool()]),
        provider=faux,
        ids=["u1", "ts", "a1", "te1"],
        now=1000,
    )

    runner.post_message("Which one?")
    async with runner.run() as run:
        events = [event async for event in run]

    assert [type(event).__name__ for event in events] == [
        "FinishReason",
        "ToolCallReceived",
        "ToolExecutionStarted",
    ]
    assert isinstance(events[1], ToolCallReceived)
    assert runner.session.entries["te1"].status == ExecutionStatus.AWAITING_RESULT
    assert runner.session.get_conversation_status("c1").status == ConversationStatus.BLOCKED
