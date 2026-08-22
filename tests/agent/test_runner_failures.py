"""Timeout & failure-outcome scenarios: the tool deadline (per-tool override
vs RuntimeConfig), the whole failure surface of the prepare/execute split
(resolution fails → the body was never dispatched; the body raises → FAILED,
whatever the exception type), a raising `get_tools` (propagates, the turn
stays open, the next run asks again), crash recovery — of orphaned RUNNING
executions and of a `prepare()` that never got to write one — and the LLM
catch/close/re-raise site (TIMED_OUT / ERRORED → status PENDING, retry-ready).
The post_message acceptance matrix lives in `test_runner_post_message.py`.

House style: precondition → one action → full-object postcondition; never
race two timed things — every deadline test pairs a real (small) timer with
a hang-forever await, so the timer is the only clock that matters.

The event TYPE list is asserted verbatim in the dispatch scenarios because it
is the load-bearing fact of the split: `ToolExecutionStarted` is emitted if
and only if the body was actually dispatched. The full `ToolExecution` carries
the rest of the detail.
"""

import asyncio
import json

import pytest
from pydantic import ValidationError

from luca.agent.core.context import CancellationToken
from luca.agent.core.events import (
    FinishReason,
    TextBlock,
    TextDelta,
    TextStart,
    ToolExecuted,
)
from luca.agent.core.exceptions import (
    AgentError,
    IncompleteResponseError,
    InvalidToolArguments,
    ToolNotFound,
)
from luca.agent.core.models import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    ExecutionAttempt,
    ExecutionAttemptOutcome,
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
    UserMessage,
)
from luca.agent.core.tool_registry import PreparedTool
from luca.client.exceptions import ProviderAPIError, StreamError, TimeoutError as ClientTimeoutError
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_error,
    faux_hang,
    faux_text,
    faux_tool_call,
)
from luca.client.types import TextBlock as LucaTextBlock, ToolMessage, UserMessage as LucaUserMessage
from tests.agent.scenarios import (
    ADD_SPEC,
    CLEARED_SESSION,
    MODEL,
    MULTIPLY_SPEC,
    POST_FAILURE_SESSION,
    RUNNING_ORPHAN_SESSION,
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

# ── tool doubles ───────────────────────────────────────────────────────────────


class CooperatingHangTool(FakeTool):
    """Hangs forever; cleans up observably when the deadline hard-cancels it."""

    name = "hang"
    description = "Hangs until the deadline kills it."
    Args = BinaryArgs

    def __init__(self) -> None:
        self.cleaned_up = False

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
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


class StubbornHangTool(FakeTool):
    """Ignores the hard cancel (stands in for detached thread work): swallows
    the CancelledError and keeps hanging until the TEST releases it."""

    name = "stubborn"
    description = "Survives the hard cancel."
    Args = BinaryArgs

    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> str:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await self.release.wait()  # still not done — like a blocked thread
        return "finally finished"


class TightTool(FakeTool):
    """An instant body under a 10ms deadline. The deadline bounds the BODY
    only, so the same spec answers both halves of the prepare/execute split."""

    name = "tight"
    description = "Returns immediately; carries a 10ms deadline."
    Args = BinaryArgs
    timeout_in_ms = 10

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> str:
        return "tight"


class LookupTool(FakeTool):
    """A BODY that raises `ToolNotFound` looking up a sub-resource — the tool
    resolved fine, so this is a tool failure, not a resolution failure."""

    name = "lookup"
    description = "Raises ToolNotFound from inside the body."
    Args = BinaryArgs

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> str:
        raise ToolNotFound("No such record: 42.")


class ValidatingTool(FakeTool):
    """A BODY that validates a payload of its own and raises pydantic's
    `ValidationError` — the arguments were valid; the tool failed."""

    name = "validate"
    description = "Raises a pydantic ValidationError from inside the body."
    Args = BinaryArgs

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> str:
        BinaryArgs.model_validate({})
        return "unreachable"


HANG_SPEC = CooperatingHangTool().get_tool_spec()
FAST_HANG_SPEC = FastHangTool().get_tool_spec()
TIGHT_SPEC = TightTool().get_tool_spec()
LOOKUP_SPEC = LookupTool().get_tool_spec()
VALIDATE_SPEC = ValidatingTool().get_tool_spec()


# ── registry doubles ───────────────────────────────────────────────────────────


class RaisingPrepareRegistry(FakeToolRegistry):
    """Resolution or validation fails: `prepare` raises instead of returning a
    callable, so no body is ever dispatched."""

    def __init__(self, tools=(), *, raises: Exception) -> None:
        super().__init__(tools)
        self.raises = raises

    async def prepare(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> PreparedTool:
        raise self.raises


class NonCallablePrepareRegistry(FakeToolRegistry):
    """`prepare` hands back something that cannot be invoked for ONE tool and
    behaves normally for every other — the sibling-isolation precondition."""

    def __init__(self, tools=(), *, broken: str, returns) -> None:
        super().__init__(tools)
        self.broken = broken
        self.returns = returns

    async def prepare(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> PreparedTool:
        if tool_execution.raw_tool_call.name == self.broken:
            return self.returns
        return await super().prepare(session, conversation_id, tool_execution)


class SyncPrepareRegistry(FakeToolRegistry):
    """`prepare` returns a plain `def` that returns a string. It IS callable,
    so it is invoked — and only then does the missing awaitable surface."""

    async def prepare(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> PreparedTool:
        self.prepared.append(tool_execution.raw_tool_call.name)

        def run(*, cancellation_token: CancellationToken) -> str:
            return "not an awaitable"

        return run


class SlowPrepareRegistry(FakeToolRegistry):
    """`prepare` takes 50ms — five times the tool's own 10ms deadline, which
    bounds the body and nothing else."""

    async def prepare(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> PreparedTool:
        await asyncio.sleep(0.05)
        return await super().prepare(session, conversation_id, tool_execution)


class HangingCallableRegistry(FakeToolRegistry):
    """`prepare` returns instantly; the CALLABLE it returns hangs forever —
    the only thing the deadline is allowed to kill."""

    async def prepare(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> PreparedTool:
        self.prepared.append(tool_execution.raw_tool_call.name)

        async def run(*, cancellation_token: CancellationToken) -> ExecutionResult:
            await asyncio.Event().wait()
            return ExecutionResult(content=[TextContent(text="unreachable")])

        return run


class FlakyGetToolsRegistry(FakeToolRegistry):
    """`get_tools` raises the first time it is asked and answers normally
    afterwards — the registry whose catalog was briefly unreachable."""

    def __init__(self, tools=(), *, raises: Exception) -> None:
        super().__init__(tools)
        self.raises = raises
        self.list_calls = 0

    async def get_tools(self, session: AgentSession, conversation_id: str):
        self.list_calls += 1
        if self.list_calls == 1:
            raise self.raises
        return await super().get_tools(session, conversation_id)


class TornDownPrepareRegistry(FakeToolRegistry):
    """The process dying inside `prepare()`: the pending task is torn down, so
    `asyncio.CancelledError` comes out of the await. It is a `BaseException`,
    so no `except Exception` in the runner converts it into a tool failure —
    the drive unwinds having written nothing."""

    async def prepare(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> PreparedTool:
        raise asyncio.CancelledError()


class ProbeRegistry(FakeToolRegistry):
    """Records every invocation of the prepared callable, plus a snapshot of
    that record taken at the instant `prepare()` returns: a `prepare` that ran
    the body would leave a non-empty snapshot behind."""

    def __init__(self, tools=()) -> None:
        super().__init__(tools)
        self.invocations: list[str] = []
        self.invocations_when_prepared: list[list[str]] = []

    async def prepare(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> PreparedTool:
        name = tool_execution.raw_tool_call.name
        inner = await super().prepare(session, conversation_id, tool_execution)

        async def run(*, cancellation_token: CancellationToken) -> ExecutionResult:
            self.invocations.append(name)
            return await inner(cancellation_token=cancellation_token)

        self.invocations_when_prepared.append(list(self.invocations))
        return run


class ToolHookRecorder:
    """Middleware double recording the tool-lifecycle hook pair: which call
    fired it, in which status, and (for the outcome hook) with which live
    exception."""

    def __init__(self) -> None:
        self.before: list[tuple[str, ExecutionStatus]] = []
        self.after: list[tuple[str, ExecutionStatus, str | None]] = []

    def before_tool_execution(self, session, conversation_id, execution: ToolExecution) -> ToolExecution:
        self.before.append((execution.tool_call_id, execution.status))
        return execution

    def after_tool_execution(
        self,
        session,
        conversation_id,
        execution: ToolExecution,
        exception: Exception | None,
    ) -> ToolExecution:
        self.after.append(
            (
                execution.tool_call_id,
                execution.status,
                None if exception is None else type(exception).__name__,
            )
        )
        return execution


# ── literal factories ──────────────────────────────────────────────────────────


def _pydantic_error() -> ValidationError:
    """A real pydantic `ValidationError` — the one exception whose message and
    structured errors cannot be written as a literal."""
    try:
        BinaryArgs.model_validate({})
    except ValidationError as exc:
        return exc
    raise AssertionError("BinaryArgs.model_validate({}) must fail")


VALIDATION_ERROR = _pydantic_error()


def one_call_session(session_id: str) -> AgentSession:
    """A fresh session with one unanswered user message — the precondition
    every dispatch scenario below starts from."""
    return make_session(
        id=session_id,
        entries={
            "u1": UserMessage(
                id="u1",
                created_at=500,
                parts=[TextContent(text="go")],
            ),
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
        session_config=SessionConfig(llm_config=MODEL),
    )


def prepare_failure(
    status: ExecutionStatus,
    error: ToolExecutionError,
) -> ToolExecution:
    """The durable shape EVERY prepare-time failure produces for `add(1, 2)`
    as the only call of a fresh turn.

    Only `status` and `error` vary with the exception: the birth `tool_spec`
    stands (there is no dispatch-time re-snapshot), approval was granted and
    stays granted, and the body was never invoked — no attempt was ever
    appended, so `dispatched` is False, and there is no result."""
    return ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=status,
        result=None,
        error=error,
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[
            ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
        ],
        attempts=[],
        finished_at=1000,
        updated_at=1000,
        context_tokens=len(error.error_message) // 4,
    )


def resolution_facts(execution: ToolExecution) -> dict:
    """The facts a resolution failure must report identically whenever it is
    discovered. Approval state legitimately differs (a birth-terminal call was
    never decided) and so does `error.details`, whose phase is the whole point
    of recording it — those two are asserted separately."""
    return {
        "status": execution.status,
        "error_type": execution.error.error_type,
        "result": execution.result,
        "attempts": execution.attempts,
        "dispatched": execution.dispatched,
    }


# The orphan recovery outcome, shared by the two scenarios that produce it:
# an orphan is exactly another INTERRUPTED execution. Its dangling attempt —
# opened when the dead process persisted RUNNING — is closed as DISCARDED:
# recovery never learned how that invocation ended.
RECOVERED_ORPHAN = ToolExecution(
    id="te1",
    conversation_id="c1",
    parent_id="a1",
    created_at=500,
    tool_call_id="tc1",
    raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
    tool_spec=ADD_SPEC,
    tool_spec_id=ADD_SPEC.spec_id(),
    status=ExecutionStatus.INTERRUPTED,
    result=None,
    approval_status=ApprovalStatus.ALLOWED,
    approval_decisions=[
        ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=500),
    ],
    attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.DISCARDED, started_at=500, ended_at=1000)],
    finished_at=1000,
    updated_at=1000,
)


# ── tool timeout (§5.1) ────────────────────────────────────────────────────────


async def test_tool_timeout_records_timed_out_and_the_turn_continues():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("hang", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Couldn't compute.")], finish_reason="stop"),
        ]
    )
    tool = CooperatingHangTool()
    session = make_session(
        id="s_tool_to",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(
            llm_config=MODEL,
            runtime_config=RuntimeConfig(tool_execution_timeout_in_ms=50),
        ),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([tool]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    assert [event.type for event in events] == [
        "finish_reason",
        "tool_call_received",
        "tool_execution_started",
        "tool_executed",
        "text_block",
        "finish_reason",
    ]
    assert tool.cleaned_up is True  # the hard cancel was delivered
    # the status IS the complete lifecycle fact: no result, no error, and a
    # deadline is not a cancel (`cancel_signalled_at` stays None)
    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="hang", arguments={"a": 1, "b": 2}),
        tool_spec=HANG_SPEC,
        tool_spec_id=HANG_SPEC.spec_id(),
        status=ExecutionStatus.TIMED_OUT,
        result=None,
        error=None,
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[
            ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
        ],
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.TIMED_OUT, started_at=1000, ended_at=1000)],
        finished_at=1000,
        cancel_signalled_at=None,
        updated_at=1000,
    )
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
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("stubborn", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Moving on.")], finish_reason="stop"),
        ]
    )
    tool = StubbornHangTool()
    session = make_session(
        id="s_stubborn",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(
            llm_config=MODEL,
            runtime_config=RuntimeConfig(tool_execution_timeout_in_ms=50),
        ),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([tool]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
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
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("fast_hang", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    tool = FastHangTool()
    session = make_session(
        id="s_override",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(
            llm_config=MODEL,
            # effectively infinite next to the tool's own 50ms
            runtime_config=RuntimeConfig(tool_execution_timeout_in_ms=600_000),
        ),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([tool]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    result = await runner.run()

    assert result.outcome == TurnOutcome.COMPLETED
    assert tool.cleaned_up is True
    assert runner.session.entries["te1"].status == ExecutionStatus.TIMED_OUT
    assert runner.session.entries["te1"].tool_spec == FAST_HANG_SPEC


async def test_instant_tool_under_a_huge_deadline_is_unaffected():
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
        id="s_inert_deadline",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(
            llm_config=MODEL,
            runtime_config=RuntimeConfig(tool_execution_timeout_in_ms=600_000),
        ),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    completed = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="3")]),
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[
            ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
        ],
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
        finished_at=1000,
        updated_at=1000,
    )
    assert [event.type for event in events] == [
        "finish_reason",
        "tool_call_received",
        "tool_execution_started",
        "tool_executed",
        "text_block",
        "finish_reason",
    ]
    assert events[3] == ToolExecuted(
        conversation_id="c1",
        tool_call_id="tc1",
        execution=completed,
        result_text="3",
        is_error=False,
    )
    assert runner.session.entries["te1"] == completed
    assert runner.idle()


async def test_a_prepare_slower_than_the_tools_own_deadline_still_succeeds():
    # the deadline bounds the BODY only: 50ms of preparation under a 10ms
    # `timeout_in_ms` is not a timeout, because no clock covers `prepare()`.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("tight", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
        ]
    )
    session = one_call_session("s_slow_prepare")
    runner = DeterministicRunner(
        session,
        tool_registry=SlowPrepareRegistry([TightTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    result = await runner.run()

    assert result.outcome == TurnOutcome.COMPLETED
    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="tight", arguments={"a": 1, "b": 2}),
        tool_spec=TIGHT_SPEC,
        tool_spec_id=TIGHT_SPEC.spec_id(),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="tight")]),
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[
            ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
        ],
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
        finished_at=1000,
        updated_at=1000,
        context_tokens=1,
    )
    assert runner.idle()


async def test_the_same_deadline_against_a_slow_callable_records_timed_out():
    # the other half of the split: the identical 10ms `timeout_in_ms`, this
    # time with an instant `prepare()` and a body that never returns.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("tight", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Gave up.")], finish_reason="stop"),
        ]
    )
    session = one_call_session("s_slow_callable")
    runner = DeterministicRunner(
        session,
        tool_registry=HangingCallableRegistry([TightTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    result = await runner.run()

    assert result.outcome == TurnOutcome.COMPLETED
    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="tight", arguments={"a": 1, "b": 2}),
        tool_spec=TIGHT_SPEC,
        tool_spec_id=TIGHT_SPEC.spec_id(),
        status=ExecutionStatus.TIMED_OUT,
        result=None,
        error=None,
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[
            ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
        ],
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.TIMED_OUT, started_at=1000, ended_at=1000)],
        finished_at=1000,
        cancel_signalled_at=None,
        updated_at=1000,
    )
    assert runner.idle()


# ── prepare() failures: resolution never dispatches a body ─────────────────────

PREPARE_FAILURES = [
    pytest.param(
        ToolNotFound("Unknown tool: 'add'."),
        ExecutionStatus.NOT_FOUND,
        ToolExecutionError(
            error_type="ToolNotFound",
            error_message="Unknown tool: 'add'.",
            details={"phase": "prepare"},
        ),
        id="tool_not_found",
    ),
    pytest.param(
        InvalidToolArguments(
            "Arguments for tool 'add' are invalid.",
            errors=[{"type": "int_parsing", "loc": ["a"], "msg": "not an int"}],
        ),
        ExecutionStatus.INVALID,
        ToolExecutionError(
            error_type="InvalidToolArguments",
            error_message="Arguments for tool 'add' are invalid.",
            details={
                "phase": "prepare",
                "errors": [{"type": "int_parsing", "loc": ["a"], "msg": "not an int"}],
            },
        ),
        id="invalid_tool_arguments",
    ),
    pytest.param(
        VALIDATION_ERROR,
        ExecutionStatus.INVALID,
        ToolExecutionError(
            error_type="ValidationError",
            error_message=str(VALIDATION_ERROR),
            details={
                "phase": "prepare",
                "errors": json.loads(VALIDATION_ERROR.json(include_url=False)),
            },
        ),
        id="pydantic_validation_error",
    ),
    pytest.param(
        RuntimeError("the tool catalog is offline"),
        ExecutionStatus.FAILED,
        ToolExecutionError(
            error_type="RuntimeError",
            error_message="the tool catalog is offline",
            details={"phase": "prepare"},
        ),
        id="anything_else",
    ),
]


@pytest.mark.parametrize(("exception", "status", "error"), PREPARE_FAILURES)
async def test_a_raising_prepare_never_marks_the_body_dispatched(exception, status, error):
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("No luck.")], finish_reason="stop"),
        ]
    )
    session = one_call_session("s_prepare_raise")
    runner = DeterministicRunner(
        session,
        tool_registry=RaisingPrepareRegistry([AddTool()], raises=exception),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    # no `tool_execution_started`: the body was never dispatched
    assert [event.type for event in events] == [
        "finish_reason",
        "tool_call_received",
        "tool_executed",
        "text_block",
        "finish_reason",
    ]
    assert runner.session.entries["te1"] == prepare_failure(status, error)
    assert runner.session.entries["te1"].dispatched is False
    assert runner.session.entries["tf"].outcome == TurnOutcome.COMPLETED
    assert runner.idle()


async def test_a_resolution_failure_records_the_same_way_whenever_it_is_found():
    # one assistant response, two unresolvable calls: `ghost` is unknown at
    # birth, `add` resolves at birth and has vanished by `prepare()`.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    faux_tool_call("ghost", {"a": 1, "b": 2}, id="tc1"),
                    faux_tool_call("add", {"a": 1, "b": 2}, id="tc2"),
                ],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Neither tool exists.")], finish_reason="stop"),
        ]
    )
    session = one_call_session("s_two_resolutions")
    runner = DeterministicRunner(
        session,
        tool_registry=RaisingPrepareRegistry(
            [AddTool()],
            raises=ToolNotFound("Unknown tool: 'add'."),
        ),
        provider=faux,
        ids=["ts", "a1", "te1", "te2", "a2", "tf"],
        now=1000,
    )

    await runner.run()

    at_birth = runner.session.entries["te1"]
    at_prepare = runner.session.entries["te2"]
    assert (
        resolution_facts(at_birth)
        == resolution_facts(at_prepare)
        == {
            "status": ExecutionStatus.NOT_FOUND,
            "error_type": "ToolNotFound",
            "result": None,
            "attempts": [],
            "dispatched": False,
        }
    )
    # the two legitimate differences: approval state and the recorded phase
    assert (at_birth.approval_status, at_birth.approval_decisions) == (None, [])
    assert (at_prepare.approval_status, at_prepare.approval_decisions) == (
        ApprovalStatus.ALLOWED,
        [ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)],
    )
    assert at_birth.error.details == {}  # the registry authored this one
    assert at_prepare.error.details == {"phase": "prepare"}
    assert runner.idle()


@pytest.mark.parametrize(
    ("returns", "type_name"),
    [
        pytest.param(None, "NoneType", id="none"),
        pytest.param(("not", "callable"), "tuple", id="tuple"),
    ],
)
async def test_a_non_callable_prepare_fails_one_execution_and_leaves_the_rest(returns, type_name):
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
            faux_assistant_message([faux_text("One of them worked.")], finish_reason="stop"),
        ]
    )
    session = one_call_session("s_non_callable")
    runner = DeterministicRunner(
        session,
        tool_registry=NonCallablePrepareRegistry(
            [AddTool(), MultiplyTool()],
            broken="add",
            returns=returns,
        ),
        provider=faux,
        ids=["ts", "a1", "te1", "te2", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    # only the healthy sibling reports a dispatch
    assert [event.type for event in events] == [
        "finish_reason",
        "tool_call_received",
        "tool_call_received",
        "tool_executed",
        "tool_execution_started",
        "tool_executed",
        "text_block",
        "finish_reason",
    ]
    assert runner.session.entries["te1"] == prepare_failure(
        ExecutionStatus.FAILED,
        ToolExecutionError(
            error_type="AgentError",
            error_message=(f"prepare() for tool 'add' returned {type_name}, which is not callable."),
            details={"phase": "prepare"},
        ),
    )
    assert runner.session.entries["te2"] == ToolExecution(
        id="te2",
        conversation_id="c1",
        parent_id="te1",
        created_at=1000,
        tool_call_id="tc2",
        raw_tool_call=ToolCall(id="tc2", name="multiply", arguments={"a": 3, "b": 4}),
        tool_spec=MULTIPLY_SPEC,
        tool_spec_id=MULTIPLY_SPEC.spec_id(),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="12")]),
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[
            ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
        ],
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
        finished_at=1000,
        updated_at=1000,
    )
    assert runner.session.entries["tf"].outcome == TurnOutcome.COMPLETED
    assert runner.idle()


async def test_before_tool_execution_fires_exactly_once_when_prepare_raises():
    # the hook is the boundary: it fired ahead of `prepare()`, so the
    # terminalization belongs to the dispatch path and must not run the
    # undispatched pipeline (which would fire it a second time).
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("No luck.")], finish_reason="stop"),
        ]
    )
    recorder = ToolHookRecorder()
    session = one_call_session("s_hook_once")
    runner = DeterministicRunner(
        session,
        tool_registry=RaisingPrepareRegistry(
            [AddTool()],
            raises=ToolNotFound("Unknown tool: 'add'."),
        ),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
        middleware=[recorder],
    )

    await runner.run()

    assert recorder.before == [("tc1", ExecutionStatus.PENDING)]
    assert recorder.after == [("tc1", ExecutionStatus.NOT_FOUND, "ToolNotFound")]


async def test_prepare_does_not_invoke_the_callable_it_returns():
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
    registry = ProbeRegistry([AddTool()])
    session = one_call_session("s_probe")
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    await runner.run()

    assert registry.prepared == ["add"]
    assert registry.invocations_when_prepared == [[]]  # nothing had run yet
    assert registry.invocations == ["add"]  # invoked once, after prepare
    assert runner.session.entries["te1"].status == ExecutionStatus.COMPLETED


# ── failures AFTER dispatch: every raise is FAILED ────────────────────────────


async def test_a_tool_body_raising_tool_not_found_records_failed_at_execution():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("lookup", {"a": 4, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("It failed.")], finish_reason="stop"),
        ]
    )
    session = one_call_session("s_body_not_found")
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([LookupTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    # the body WAS dispatched — resolution succeeded, so the exception type
    # no longer selects the status
    assert [event.type for event in events] == [
        "finish_reason",
        "tool_call_received",
        "tool_execution_started",
        "tool_executed",
        "text_block",
        "finish_reason",
    ]
    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="lookup", arguments={"a": 4, "b": 2}),
        tool_spec=LOOKUP_SPEC,
        tool_spec_id=LOOKUP_SPEC.spec_id(),
        status=ExecutionStatus.FAILED,
        result=None,
        error=ToolExecutionError(
            error_type="ToolNotFound",
            error_message="No such record: 42.",
            details={"phase": "execution"},
        ),
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[
            ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
        ],
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.FAILED, started_at=1000, ended_at=1000)],
        finished_at=1000,
        updated_at=1000,
        context_tokens=len("No such record: 42.") // 4,
    )
    assert runner.session.entries["te1"].dispatched is True
    assert runner.idle()


async def test_a_tool_body_raising_a_validation_error_records_failed_not_invalid():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("validate", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("It failed.")], finish_reason="stop"),
        ]
    )
    session = one_call_session("s_body_invalid")
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([ValidatingTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    await runner.run()

    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="validate", arguments={"a": 1, "b": 2}),
        tool_spec=VALIDATE_SPEC,
        tool_spec_id=VALIDATE_SPEC.spec_id(),
        status=ExecutionStatus.FAILED,
        result=None,
        error=ToolExecutionError(
            error_type="ValidationError",
            error_message=str(VALIDATION_ERROR),
            details={
                "phase": "execution",
                "errors": json.loads(VALIDATION_ERROR.json(include_url=False)),
            },
        ),
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[
            ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
        ],
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.FAILED, started_at=1000, ended_at=1000)],
        finished_at=1000,
        updated_at=1000,
        context_tokens=len(str(VALIDATION_ERROR)) // 4,
    )
    assert runner.idle()


async def test_a_prepared_callable_that_returns_a_plain_value_fails_after_dispatch():
    # the callable was INVOKED, so the missing awaitable is a post-dispatch
    # failure like any other — the TypeError must not escape the run.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Broken tool.")], finish_reason="stop"),
        ]
    )
    session = one_call_session("s_sync_callable")
    runner = DeterministicRunner(
        session,
        tool_registry=SyncPrepareRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    assert [event.type for event in events] == [
        "finish_reason",
        "tool_call_received",
        "tool_execution_started",
        "tool_executed",
        "text_block",
        "finish_reason",
    ]
    error_message = "An asyncio.Future, a coroutine or an awaitable is required"
    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.FAILED,
        result=None,
        error=ToolExecutionError(
            error_type="TypeError",
            error_message=error_message,
            details={"phase": "execution"},
        ),
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[
            ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
        ],
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.FAILED, started_at=1000, ended_at=1000)],
        finished_at=1000,
        updated_at=1000,
        context_tokens=len(error_message) // 4,
    )
    assert runner.session.entries["te1"].dispatched is True
    assert runner.idle()


async def test_a_toolless_runner_terminalizes_a_loaded_ready_execution():
    # a session persisted with an approved-but-unrun call, reloaded into a
    # runner that has no registry at all: `prepare()` raises ToolNotFound so
    # the call records honestly instead of crashing the run.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("That tool is gone.")], finish_reason="stop"),
        ]
    )
    session = CLEARED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        tool_registry=None,
        provider=faux,
        ids=["a2", "tf"],
        now=1000,
    )

    assert runner.busy()
    async with runner.run() as run:
        events = [event async for event in run]

    assert [event.type for event in events] == [
        "tool_executed",
        "text_block",
        "finish_reason",
    ]
    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=500,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.NOT_FOUND,
        result=None,
        error=ToolExecutionError(
            error_type="ToolNotFound",
            error_message="Unknown tool: 'add'.",
            details={"phase": "prepare"},
        ),
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[
            ApprovalDecision(decision=ApprovalOption.PENDING, created_at=500),
            ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=600),
        ],
        attempts=[],
        finished_at=1000,
        updated_at=1000,
        context_tokens=len("Unknown tool: 'add'.") // 4,
    )
    assert faux.requests[0].messages[-1] == ToolMessage(
        tool_call_id="tc1",
        content=[LucaTextBlock(text="Unknown tool: 'add'.")],
        is_error=True,
    )
    assert runner.session.entries["tf"].outcome == TurnOutcome.COMPLETED
    assert runner.idle()


# ── get_tools failure: the run aborts, the turn stays open ────────────────────


async def test_get_tools_raising_aborts_the_run_and_the_next_one_asks_again():
    # `get_tools` is treated exactly like `decide`: the exception propagates
    # and aborts the run with the turn left open and resumable. The runner
    # never substitutes an empty tool list — calling the model with no tools
    # when the registry meant to offer some would silently change the answer.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Ready.")], finish_reason="stop"),
        ]
    )
    registry = FlakyGetToolsRegistry(
        [AddTool()],
        raises=RuntimeError("the tool catalog is offline"),
    )
    session = one_call_session("s_get_tools_raise")
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "tf"],
        now=1000,
    )

    with pytest.raises(RuntimeError, match="the tool catalog is offline"):
        await runner.run()

    # the bracket opened and stayed open, and no LLM call was made at all
    assert list(runner.session.entries) == ["u1", "ts"]
    assert faux.requests == []
    assert runner.busy()

    await runner.run()  # the resume asks again

    assert registry.list_calls == 2
    assert list(runner.session.entries) == ["u1", "ts", "a1", "tf"]
    assert runner.session.entries["tf"].outcome == TurnOutcome.COMPLETED
    assert runner.idle()


# ── crash recovery: orphaned RUNNING executions, and a lost prepare ───────────


async def test_orphaned_running_execution_recovers_to_interrupted_without_redispatch():
    # a persisted RUNNING execution has no live task on the next drive: it is
    # terminalized INTERRUPTED before anything else — after_tool_execution
    # runs, the body is NEVER re-dispatched (no ToolExecutionStarted, no
    # result), and durable state records nothing crash-specific.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("It was interrupted.")], finish_reason="stop"),
        ]
    )
    session = RUNNING_ORPHAN_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["a2", "tf"],
        now=1000,
    )

    assert runner.busy()  # stale RUNNING status self-healed on construction
    async with runner.run() as run:
        events = [event async for event in run]

    assert events == [
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=RECOVERED_ORPHAN,
            result_text="[tool execution interrupted]",
            is_error=True,
        ),
        TextBlock(conversation_id="c1", text="It was interrupted."),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert runner.session.entries["te1"] == RECOVERED_ORPHAN
    # the model was called only after recovery, with the derived tool output
    assert faux.requests[0].messages[-1] == ToolMessage(
        tool_call_id="tc1",
        content=[LucaTextBlock(text="[tool execution interrupted]")],
        is_error=True,
    )
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="a2",
        created_at=1000,
    )
    assert runner.idle()


async def test_orphan_recovery_precedes_a_parked_cancel_flush():
    # recovery runs before the wind-down, so a call whose body actually
    # started is INTERRUPTED (no cancel_signalled_at — the cancellation never
    # reached the live body), never CANCELLED.
    session = RUNNING_ORPHAN_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=FauxProvider(),
        ids=["cr", "tf"],
        now=1000,
    )
    runner.cancel()

    result = await runner.run()  # the flush

    assert result.outcome == TurnOutcome.CANCELLED
    assert runner.session.entries["te1"] == RECOVERED_ORPHAN
    assert runner.idle()


async def test_a_crash_during_prepare_leaves_it_pending_and_the_next_drive_prepares_again():
    # The mirror of the orphan case, and the reason the RUNNING row moved
    # AFTER `prepare()`: preparation writes nothing, so a process that dies
    # inside it leaves the execution exactly as it was — PENDING, ALLOWED, no
    # attempt appended — and the next drive prepares it again and completes.
    # Nothing is INTERRUPTED, because no body ever started.
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
    lost_attempt = ToolHookRecorder()
    session = one_call_session("s_prepare_crash")
    runner = DeterministicRunner(
        session,
        tool_registry=TornDownPrepareRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1"],
        now=1000,
        middleware=[lost_attempt],
    )

    with pytest.raises(asyncio.CancelledError):
        await runner.run()

    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.PENDING,
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)],
        updated_at=1000,
    )

    # the process restarts: the durable session is reloaded into a fresh runner
    resumed_attempt = ToolHookRecorder()
    registry = FakeToolRegistry([AddTool()])
    resumed = DeterministicRunner(
        AgentSession.model_validate_json(runner.session.model_dump_json()),
        tool_registry=registry,
        provider=faux,
        ids=["a2", "tf"],
        now=2000,
        middleware=[resumed_attempt],
    )

    await resumed.run()

    assert registry.prepared == ["add"]  # prepared for the first time here
    assert resumed.session.entries["te1"].status == ExecutionStatus.COMPLETED
    assert resumed.session.entries["te1"].result == ExecutionResult(
        content=[TextContent(text="3")],
    )
    # exactly once per dispatch ATTEMPT, not once per call for all time: the
    # lost attempt fired it, and so does the one that finished
    assert lost_attempt.before == [("tc1", ExecutionStatus.PENDING)]
    assert resumed_attempt.before == [("tc1", ExecutionStatus.PENDING)]
    assert resumed.session.entries["tf"].outcome == TurnOutcome.COMPLETED
    assert resumed.idle()


# ── LLM failure: record, close, re-raise (§5.4) ───────────────────────────────


async def test_llm_timeout_closes_the_turn_and_reraises_through_await():
    # the high tier: RuntimeConfig.client_completion_timeout_in_ms wires into
    # the client's total_timeout=, whose expiry raises the SDK TimeoutError
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_hang()])])
    session = make_session(
        id="s_llm_to",
        entries={
            "u1": UserMessage(
                id="u1",
                created_at=500,
                parts=[TextContent(text="Add 1 and 2")],
            ),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(
            llm_config=MODEL,
            runtime_config=RuntimeConfig(client_completion_timeout_in_ms=50),
        ),
    )
    runner = DeterministicRunner(
        session,
        provider=faux,
        ids=["ts", "tf"],
        now=1000,
    )
    run = runner.run()

    with pytest.raises(ClientTimeoutError):
        await run

    assert run.result is None  # the raise path never produces a result
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="ts",
        created_at=1000,
        outcome=TurnOutcome.TIMED_OUT,
        error="completion exceeded timeout=0.05s",
    )
    assert main_conversation(runner.session).nodes == ["u1", "ts", "tf"]
    assert runner.idle()  # a failed close is IDLE: recover by posting, not by re-running


async def test_scripted_client_timeout_is_indistinguishable_from_a_real_one():
    # the low tier: the runner cannot and should not tell a scripted
    # TimeoutError from an httpx/total one — same class, same close
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [],
                error=faux_error("connect timeout", error_class=ClientTimeoutError),
            ),
        ]
    )
    session = make_session(
        id="s_llm_to2",
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
        ids=["ts", "tf"],
        now=1000,
    )

    with pytest.raises(ClientTimeoutError, match="connect timeout"):
        await runner.run()

    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="ts",
        created_at=1000,
        outcome=TurnOutcome.TIMED_OUT,
        error="connect timeout",
    )
    assert runner.idle()  # a closed bracket is IDLE whatever its outcome


async def test_llm_error_closes_the_turn_and_reraises_through_iteration():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [],
                error=faux_error("provider 500", error_class=ProviderAPIError),
            ),
        ]
    )
    session = make_session(
        id="s_llm_err",
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
        ids=["ts", "tf"],
        now=1000,
    )
    run = runner.run()

    with pytest.raises(ProviderAPIError, match="provider 500"):
        async with run:
            _ = [event async for event in run]

    assert run.result is None
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="ts",
        created_at=1000,
        outcome=TurnOutcome.ERRORED,
        error="provider 500",
    )
    assert runner.idle()  # a closed bracket is IDLE whatever its outcome


async def test_streaming_llm_error_closes_the_turn_after_the_deltas():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_text("Hel")],
                error=faux_error("boom mid-stream"),
            ),
        ]
    )
    session = make_session(
        id="s_stream_err",
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
        ids=["ts", "tf"],
        now=1000,
    )
    run = runner.run(streaming=True)
    events = []

    # The append loop is load-bearing: the error fires mid-iteration and the
    # assertions below read the events captured before it, which a comprehension
    # would discard. Hence the PERF401/PT012 suppressions.
    with pytest.raises(StreamError, match="boom mid-stream"):  # noqa: PT012
        async with run:
            async for event in run:
                events.append(event)  # noqa: PERF401

    assert events == [TextStart(conversation_id="c1"), TextDelta(conversation_id="c1", text="Hel")]
    assert run.result is None
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="ts",
        created_at=1000,
        outcome=TurnOutcome.ERRORED,
        error="boom mid-stream",
    )
    assert runner.idle()  # closed bracket; the partial assistant message was dropped


# ── the model stopped without answering ────────────────────────────────────────
# `NON_ANSWER_FINISH_REASONS`. The round keys off `tool_calls`, which settles
# "stop" vs "tool_use" and says nothing about a model that produced neither an
# answer nor a call — so `length` and `error` used to close the turn COMPLETED,
# and a sentence cut off mid-word or a refused request was indistinguishable
# from a finished one.


async def test_a_length_truncation_errors_the_turn_and_keeps_what_was_produced():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Here is the first half of the ans")], finish_reason="length"),
        ]
    )
    session = make_session(
        id="s_len",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Write something long")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, provider=faux, ids=["ts", "a1", "tf"], now=1000)

    with pytest.raises(IncompleteResponseError, match="finish_reason 'length'"):
        await runner.run()

    # THE PARTIAL SURVIVES. A transport failure drops its partial because
    # nothing was really produced; here the model produced these tokens and on
    # a truncation they are the useful half.
    assert runner.session.entries["a1"] == AssistantMessage(
        id="a1",
        parent_id="ts",
        created_at=1000,
        context_tokens=8,
        parts=[TextContent(text="Here is the first half of the ans")],
        llm_config=MODEL,
        stop_reason="length",
    )
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="a1",
        created_at=1000,
        outcome=TurnOutcome.ERRORED,
        error="the model stopped with finish_reason 'length'",
    )
    assert main_conversation(runner.session).nodes == ["u1", "ts", "a1", "tf"]
    assert runner.idle()  # resumable: the recovery is another post_message ("continue")


async def test_a_refused_answer_errors_the_turn_under_the_providers_own_message():
    """Every transport canonicalizes a refusal / safety filter / guardrail to
    finish_reason "error" WITH an `error_message`. The runner reads it rather
    than inventing prose."""
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("I can't")], finish_reason="error"),
        ]
    )
    session = make_session(
        id="s_refused",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="do the thing")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, provider=faux, ids=["ts", "a1", "tf"], now=1000)

    with pytest.raises(IncompleteResponseError, match="Faux error terminal"):
        await runner.run()

    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="a1",
        created_at=1000,
        outcome=TurnOutcome.ERRORED,
        error="Faux error terminal",
    )
    assert runner.idle()


async def test_a_truncated_round_that_still_produced_tool_calls_runs_them():
    """`tool_calls` wins over the finish reason, unchanged. A response the cap
    cut off AFTER a complete call is not a failure — the call parsed, so it is
    dispatched and the turn goes on."""
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="length",
            ),
            faux_assistant_message([faux_text("3.")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_len_calls",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="add them")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    await runner.run()

    assert main_conversation(runner.session).nodes == ["u1", "ts", "a1", "te1", "a2", "tf"]
    assert runner.session.entries["tf"].outcome == TurnOutcome.COMPLETED


async def test_post_failure_session_reloads_cold_and_a_new_turn_reanswers():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
        ]
    )
    session = POST_FAILURE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        provider=faux,
        ids=["u2", "ts2", "a1", "tf2"],
        now=2000,
    )

    # A failed turn is IDLE, not retry-ready: a bare `run()` would silently
    # re-send the identical request, so recovery goes through post_message.
    assert runner.idle()
    with pytest.raises(AgentError, match="Nothing to run"):
        await runner.run()

    runner.post_message("Add 1 and 2")
    result = await runner.run()

    assert result.outcome == TurnOutcome.COMPLETED
    assert main_conversation(runner.session).nodes == [
        "u1",
        "ts",
        "tf",
        "u2",
        "ts2",
        "a1",
        "tf2",
    ]
    # the failed bracket projected nothing to the wire — just the two user
    # messages (the original and the one that re-armed the session)
    assert faux.requests[0].messages == [
        LucaUserMessage(content=[LucaTextBlock(text="Add 1 and 2")]),
        LucaUserMessage(content=[LucaTextBlock(text="Add 1 and 2")]),
    ]
    assert runner.idle()


# The post_message acceptance matrix lives in `test_runner_post_message.py`.


async def test_empty_resume_reasons_closes_completed_with_pause_recorded():
    # [] = pause-and-replay off: "pause" is not in NON_ANSWER_FINISH_REASONS,
    # so the turn closes COMPLETED exactly as before the feature — with the
    # durable stop_reason still recording what the provider said
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("Searching...")], finish_reason="pause")])
    session = make_session(
        id="s_pause_off",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="find apple")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(
            llm_config=MODEL,
            runtime_config=RuntimeConfig(resume_finish_reasons=[]),
        ),
    )
    runner = DeterministicRunner(session, provider=faux, ids=["ts", "a1", "tf"], now=1000)

    await runner.run()

    assert len(faux.requests) == 1
    assert runner.session.entries["a1"].stop_reason == "pause"
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="a1",
        created_at=1000,
        outcome=TurnOutcome.COMPLETED,
    )
