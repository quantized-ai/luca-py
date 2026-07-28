"""AgentSessionRunner: a resumable async agent loop over the core data model.

The runner is a small state machine wrapped around an `AgentSession`. A caller
drives it by polling its status and supplying input:

    runner = AgentSessionRunner(session, tool_registry=REGISTRY)
    while True:
        if runner.idle():
            runner.post_message(input("> "))
        elif runner.awaiting_approval():
            ...                                        # resolve on the registry
        else:                                          # PENDING / CANCELLING
            async with runner.run() as run:
                async for event in run:                # render the event
                    ...

A run is created by one of two methods, both returning an `AgentRun` handle:

- `run()` — "Lazy: nothing happens until awaited or iterated; stopping
  iteration stops the agent."
- `start()` — "Eager: begins immediately and completes regardless of
  observation; await to join, `.cancel()` to stop."

The handle supports three consumption forms: `await run` drives (lazy) or
joins (eager) to the next stopping point and returns a `RunResult`;
`async with run: async for event in run` iterates the events (iteration
REQUIRES the context manager — lazy iteration is the engine itself, eager
iteration reads a buffer the background task fills); `run.cancel()` requests
the turn end. Exiting a lazy run's block never advances the engine — exit
always suspends where it is, re-derives the status, and finalizes the handle;
the open bracket resumes via a later `run()`. `streaming=` selects only the
event vocabulary (block events vs block + delta events); the session updates
are identical.

Each run advances the agent from its current status as far as it can, then
stops at the next point that needs the user:

- the turn completed (no more tool calls)            → status IDLE;
- the registry left a call's approval PENDING        → status AWAITING_APPROVAL.

One logical *turn* (the agent's full response to a user message) is bracketed
by a single `TurnStart` / `TurnFinish` (the finish carries the `TurnOutcome`),
even when it pauses for approval across several runs (or a process restart): a
`TurnStart` with no later `TurnFinish` means the open turn is resumed rather
than re-opened. Provider usage is recorded per assistant entry in
`AgentSession.usages[conversation_id][entry_id]` — accessory
conversation-entry data, never embedded in entries or rolled up on markers.

Compaction — replacing the older span of a conversation with a summary of it —
is delegated the same way, to the `CompactionPolicy` the runner is constructed
with (`compaction.py`; `None` = compaction never happens). It runs as a step at
the top of a drive, before the conversational bracket opens, inside a turn
bracket of its own; a successful one archives the conversation and installs a
new one over the path the policy chose, in a single atomic swap. See
`schedule_compaction()` and `_compaction_step`.

The whole tool lifecycle is delegated to the `ToolRegistry` the runner is
constructed with (`tool_registry.py`; `None` = toolless agent). The runner
touches tools through exactly four registry methods: `get_tools` (queried
fresh per LLM call), `create_execution` (the birth draft — the runner stamps
identity and appends), `decide` (approval), and `prepare` (resolution +
validation, returning the callable that runs the body). All four are async,
all four take the live session, and all four are raced against the run's
cancellation token, so no registry or tool-owned code can make `cancel()` a
no-op. Because preparation is separate from the body, the durable `RUNNING`
row is written only once `prepare()` has returned: `started_at` /
`dispatched` mean "the body was dispatched", for every outcome, and NOT_FOUND
/ INVALID mean resolution and validation failed rather than that a body raised
a similarly-named exception. The loop has exactly ONE decide() call site — its
top: "any undecided
executions? → ask the registry" — which serves the fresh path (executions
created this iteration) and every resume path (a re-entered run, a reloaded
session) identically. An execution is created **eagerly** (persisted before
any decision exists, so a crash mid-decide loses nothing) and **atomically**
with its assistant message (no yield point between them — and a final
answer's `TurnFinish` lands in the same no-yield window, so a suspend can
neither strand a `tool_use` request without its executions nor leave a
fully-answered bracket open to a duplicate LLM call). All calls in one
assistant response are prepared as a SET, while every call keeps an
independent outcome: a decision updates `approval_status` and appends to the
`approval_decisions` audit log (only PENDING may repeat; a resolved call is
never re-asked), a DENY turns the execution `REJECTED` on the spot, and
every ALLOWED sibling proceeds to dispatch even while another call sits
deferred — the runner parks (`AWAITING_APPROVAL`) only after all currently
runnable work has advanced, and it never calls the model again until every
tool call in the preceding assistant response has a terminal execution and a
correlated tool output.

The wire payload is derived state: the runner's `ConversationProjector`
collaborator (`projection.py`, `conversation_projector=`) recomputes the
canonical client message list on every LLM call, and the same
`project_tool_execution` output feeds both the correlated tool message and
the `ToolExecuted` event's presentation fields.

Every id and timestamp the runner writes flows through two overridable hook
methods — `generate_id()` (uuid) and `now_ms()` (wall clock). The production
class carries no test parameters; tests subclass and override the hooks for
determinism (see `DeterministicRunner` in `tests/agent/scenarios.py`).
`provider=` is forwarded verbatim to the client (its public kwarg for passing
a provider instance), which is also how tests hand in a `FauxProvider`.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import json
import time
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from luca.client import acompletion, acompletion_stream
from luca.client.exceptions import TimeoutError as ClientTimeoutError
from luca.client.types import Tool as LucaTool

from . import adapter
from .compaction import (
    CompactionPlan,
    CompactionPolicy,
    ConversationSnapshot,
    check_snapshot,
    validate_plan,
)
from .context import CancellationToken
from .context_manager import ContextManager
from .events import (
    AgentEvent,
    ApprovalRequired,
    CompactionFinished,
    CompactionScheduled,
    CompactionStarted,
    FinishReason,
    ReasoningBlock,
    ReasoningDelta,
    ReasoningStart,
    TextBlock,
    TextDelta,
    TextStart,
    ToolCallReceived,
    ToolCallStart,
    ToolExecuted,
    ToolExecutionStarted,
)
from .exceptions import (
    AgentError,
    AlreadyCancellingError,
    InvalidToolArguments,
    ToolNotFound,
)
from .ledger import SessionLedger
from .models import (
    AgentSession,
    AnyEntry,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    CancelRequested,
    CompactionEntry,
    CompactionSource,
    ContentPart,
    Conversation,
    ConversationStatus,
    ExecutionStatus,
    Inf,
    LLMConfig,
    RuntimeConfig,
    SessionConfig,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    UserMessage,
)
from .projection import ConversationProjector, tool_message_text
from .system_prompt import (
    DefaultSystemPromptAssembler,
    SystemPromptAssembler,
    SystemPromptPartInput,
    coerce_system_prompt_part,
)
from .tool_registry import PreparedTool, ToolRegistry

EventCallback = Callable[[AgentEvent], "Awaitable[None] | None"]


class RunResult(BaseModel):
    """Where a run stopped: the DERIVED status there, plus how the last
    bracket this run closed ended (`None` if it closed none).

    - Turn completed → `status=IDLE`, `outcome` COMPLETED (or CANCELLED for a
      wind-down).
    - Approval pause → `status=AWAITING_APPROVAL`, `outcome=None`,
      `pending_approvals` non-empty.
    - Compaction-only drive → `status` from the new (or unchanged) path,
      `outcome` COMPLETED or CANCELLED. A caller that needs to tell "the agent
      answered" from "a compaction ran" reads `CompactionFinished`.

    Carries no usage: provider consumption lives in
    `AgentSession.usages[conversation_id][entry_id]`, one record per
    assistant entry — aggregate as needed.

    A timeout / LLM failure does NOT produce a result: the turn is closed
    (`TurnFinish(TIMED_OUT | ERRORED)`) and the exception re-raises through
    `await` / iteration — `run.result` stays None on the raise path. With a
    cancel pending, the wind-down consumes the failure instead and the run
    returns normally (outcome from the `CancelRequested`)."""

    model_config = ConfigDict(extra="forbid")

    status: ConversationStatus  # derived where the run stopped
    outcome: TurnOutcome | None  # set iff a bracket closed during this run
    pending_approvals: list[ToolExecution]  # non-empty iff AWAITING_APPROVAL


class AgentRun:
    """One run of the agent — the handle `runner.run()` / `runner.start()`
    return. Three consumption forms: `await run` → `RunResult`;
    `async with run: async for event in run` (iteration requires the context
    manager); `run.cancel()` → delegate to `runner.cancel()`.

    One logical pass: a single cursor per handle — `break` then a second
    `async for` continues where the first stopped; a second `await` returns
    the cached `RunResult` (or re-raises the stored exception). After a
    suspended lazy run is finalized by `__aexit__`, further `await`/iteration
    raises `AgentError` — resume with a fresh `runner.run()`.

    `on_event` (sync or async) is invoked inline with every event as it
    occurs, regardless of consumption form; combining it with iteration
    delivers events to both channels (supported, but pick one)."""

    def __init__(
        self,
        runner: AgentSessionRunner,
        *,
        streaming: bool,
        on_event: EventCallback | None,
        eager: bool,
    ) -> None:
        self._runner = runner
        self._streaming = streaming
        self._on_event = on_event
        self._eager = eager
        self.result: RunResult | None = None
        self._exception: BaseException | None = None
        self._engine: AsyncIterator[AgentEvent] | None = None  # lazy engine
        self._task: asyncio.Task | None = None  # eager background task
        self._buffer: list[AgentEvent] = []  # eager event history (grow-only)
        self._cursor = 0  # the handle's single logical pass
        self._wake = asyncio.Event()  # eager: buffer grew / task finished
        self._token: CancellationToken | None = None
        self._finished = False  # the engine produced its last event
        self._entered = False
        self._exited = False
        if eager:
            # Validates state synchronously at call time and spawns the
            # background task. The loop is resolved FIRST so a sync-context
            # start() fails before taking the one-engine guard; the bracket
            # opens durably at call time so an immediate cancel() has an open
            # turn to attach to (the first drive is then the flush) — and
            # which bracket that is has to be decided here too, or a
            # policy-driven compaction could never fire for an eager run.
            loop = asyncio.get_running_loop()
            self._runner._begin_run(self)
            try:
                self._runner._open_bracket_for_start()
            except BaseException:
                # `_begin_run` has already taken the one-run guard, and the
                # three sites that release it are all downstream of
                # `create_task`, which never runs. Without this the runner is
                # permanently unusable after one raise from application code
                # (`should_compact`, or `before_entry_written`).
                self._runner._end_run(self)
                raise
            self._task = loop.create_task(self._consume())

    # ── the three consumption forms ─────────────────────────────────────────

    def cancel(
        self,
        outcome: TurnOutcome = TurnOutcome.CANCELLED,
        error: str | None = None,
    ) -> None:
        """Delegates verbatim to `runner.cancel()` — session-scoped: it
        cancels the live TURN, whichever handle is driving it."""
        self._runner.cancel(outcome, error)

    def __await__(self):
        return self._wait().__await__()

    async def __aenter__(self) -> AgentRun:
        if self._exited:
            raise AgentError("this run handle is finalized; create a fresh one")
        self._entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self._exited = True
        if self._eager:
            await self._finalize_eager(swallow_failure=exc_type is not None)
            return False
        # Lazy: suspend — close the engine exactly where it is (never advance
        # it), re-derive the status, finalize the handle. No entry is written.
        if self._engine is not None and not self._finished:
            self._finished = True
            await self._engine.aclose()
            self._runner._refresh_status()
            self._runner._end_run(self)
        return False

    def __aiter__(self) -> AgentRun:
        if not self._entered or self._exited:
            raise AgentError(
                "iterate inside 'async with' (e.g. async with runner.run() as run: async for event in run: ...)"
            )
        return self

    async def __anext__(self) -> AgentEvent:
        if self._eager:
            return await self._next_buffered()
        return await self._pump()

    # ── lazy: the iterator IS the engine ────────────────────────────────────

    async def _pump(self) -> AgentEvent:
        """Advance the lazy engine one event (first drive creates it),
        delivering the event to `on_event`."""
        if self._exception is not None:
            raise self._exception
        if self._exited and self.result is None:
            # suspended (or never driven) and finalized — only a completed
            # run keeps answering through its cached result
            raise AgentError("this run was suspended and finalized; resume with a fresh runner.run()")
        if self._finished:
            raise StopAsyncIteration
        if self._engine is None:
            self._runner._begin_run(self)  # raises on IDLE / concurrent run
            self._engine = self._runner._drive(
                streaming=self._streaming,
                token=self._token,
            )
        try:
            event = await self._engine.__anext__()
        except StopAsyncIteration:
            self._finished = True
            self.result = self._runner._build_run_result()
            self._runner._end_run(self)
            raise
        except BaseException as exc:  # engine raised (incl. external cancel)
            self._finished = True
            self._exception = exc
            self._runner._refresh_status()
            self._runner._end_run(self)
            raise
        try:
            await self._deliver(event)
        except BaseException as exc:
            # on_event is app code: crash semantics — tear the engine down,
            # leave the bracket open (resumable), propagate.
            self._finished = True
            self._exception = exc
            await self._engine.aclose()
            self._runner._refresh_status()
            self._runner._end_run(self)
            raise
        return event

    async def _wait(self) -> RunResult:
        if self._eager:
            return await self._join()
        if self._exception is not None:
            raise self._exception
        while self.result is None:
            try:
                await self._pump()
            except StopAsyncIteration:
                break
        return self.result

    # ── eager: background task + buffer ─────────────────────────────────────

    async def _consume(self) -> RunResult:
        """The background task: drain the engine into the buffer, invoking
        `on_event` per event. Runs to the stopping point regardless of
        observation; a slow iterator never stalls the agent (a slow *callback*
        does — it is the app's own hook, awaited inline)."""
        engine = self._runner._drive(
            streaming=self._streaming,
            token=self._token,
        )
        try:
            while True:
                try:
                    event = await engine.__anext__()
                except StopAsyncIteration:
                    break
                self._buffer.append(event)
                self._wake.set()
                await self._deliver(event)
            self.result = self._runner._build_run_result()
            return self.result
        except BaseException as exc:
            self._exception = exc
            self._runner._refresh_status()
            raise
        finally:
            self._finished = True
            self._wake.set()
            try:
                await engine.aclose()  # no-op unless on_event/cancel left it open
            finally:
                self._runner._end_run(self)  # guard released after teardown

    async def _next_buffered(self) -> AgentEvent:
        while True:
            if self._cursor < len(self._buffer):
                event = self._buffer[self._cursor]
                self._cursor += 1
                return event
            if self._task.done():
                if self._task.cancelled():
                    raise asyncio.CancelledError()
                exc = self._task.exception()
                if exc is not None:
                    raise exc  # buffer drained → surface the failure
                raise StopAsyncIteration
            self._wake.clear()
            if self._cursor < len(self._buffer) or self._task.done():
                continue  # produced between the check and the clear
            await self._wake.wait()

    async def _join(self) -> RunResult:
        try:
            return await self._task
        except asyncio.CancelledError:
            if not self._task.done():
                # The JOIN was cancelled (task-group teardown), not the run:
                # hard-cancel the background task and await it — no orphan.
                self._task.cancel()
                with contextlib.suppress(BaseException):
                    await self._task
            raise

    async def _finalize_eager(self, swallow_failure: bool) -> None:
        try:
            await self._task
        except asyncio.CancelledError:
            if not self._task.done():
                self._task.cancel()
                with contextlib.suppress(BaseException):
                    await self._task
            raise
        except BaseException:
            # The block's own exception wins when both failed; the background
            # one stays retrievable via `await run`.
            if not swallow_failure:
                raise

    # ── shared ──────────────────────────────────────────────────────────────

    async def _deliver(self, event: AgentEvent) -> None:
        if self._on_event is None:
            return
        result = self._on_event(event)
        if inspect.isawaitable(result):
            await result


class AgentSessionRunner:
    """Stateful driver over an `AgentSession`. Owns the `ToolRegistry` (the
    single tool touch point; `None` = toolless agent), the system-prompt
    parts + assembler (see `system_prompt.py`), and the id/clock hooks;
    mutates the session in place (through its `SessionLedger`)."""

    @classmethod
    def new_session(
        cls,
        llm_config: LLMConfig,
        session_id: str | None = None,
        runtime_config: RuntimeConfig | None = None,
        conversation_id: str | None = None,
    ) -> AgentSession:
        """Build a fresh, empty `IDLE` session. The first user message is added
        later via `post_message`."""
        session_id = session_id or uuid4().hex[:8]
        ts = _now_ms()
        return AgentSession(
            id=session_id,
            active_conversation=Conversation(
                id=conversation_id or uuid4().hex[:8],
                nodes=[],
                created_at=ts,
                updated_at=ts,
                status=ConversationStatus.IDLE,
            ),
            session_config=SessionConfig(
                llm_config=llm_config,
                runtime_config=runtime_config or RuntimeConfig(),
            ),
        )

    def __init__(
        self,
        session: AgentSession,
        tool_registry: ToolRegistry | None = None,
        system_prompt_parts: list[SystemPromptPartInput] | None = None,
        system_prompt_assembler: SystemPromptAssembler | None = None,
        *,
        provider=None,
        conversation_projector: ConversationProjector | None = None,
        context_manager: ContextManager | None = None,
        compaction_policy: CompactionPolicy | None = None,
        middleware: list | None = None,
    ) -> None:
        self.session = session
        self.tool_registry = tool_registry
        # A single projector OBJECT (never a class to instantiate, never
        # stacked by plugins): lives on the runner, is never serialized, and
        # is invoked fresh whenever messages are prepared for an LLM call.
        self.conversation_projector = conversation_projector or ConversationProjector()
        # The context-accounting strategy (context_manager.py): calculates
        # every new entry's `context_tokens`, processes returned tool output,
        # and builds pruned replacements. Same collaborator pattern as the
        # projector — one object, never serialized, defaults to the simple
        # built-in policy.
        self.context_manager = context_manager or ContextManager()
        # Compaction's single extension point (compaction.py). `None` means
        # compaction never happens: `should_compact` is never consulted and
        # `schedule_compaction()` raises. The policy owns the whole decision —
        # when, what, with which model, which nodes survive, and whether to
        # try again; the runner only triggers, stamps and archives.
        self.compaction_policy = compaction_policy
        # Static parts (str / dict / SystemPromptPart) coerce eagerly — a bad
        # part fails at construction, not mid-turn. Callables resolve per call.
        self.system_prompt_parts = [
            part if callable(part) else coerce_system_prompt_part(part) for part in (system_prompt_parts or [])
        ]
        self.system_prompt_assembler = system_prompt_assembler or DefaultSystemPromptAssembler()
        self.provider = provider
        self.middleware = list(middleware or [])
        self.ledger = SessionLedger(session, self.now_ms, self.generate_id)
        self._active_run: AgentRun | None = None  # first-drive → finalization
        # Per-run state, reset in `_begin_run`. `_closed_outcome` is how the
        # LAST bracket this run closed ended — carried rather than re-read
        # from the path, because after a compaction the closing marker is on
        # the archived conversation. The two flags belong to the compaction
        # step: whether it consumed a cancellation (the drive must then stop
        # without answering the queued turn) and whether it did anything at
        # all (only then may a re-derived IDLE end the drive).
        self._closed_outcome: TurnOutcome | None = None
        self._compaction_consumed_cancel = False
        self._compaction_ran = False

        # Status is a denormalized cache of the entry state — re-derive it from
        # the entries when taking ownership of a (possibly loaded) session so a
        # stale RUNNING / drifted status self-heals.
        self.session.active_conversation.status = self.ledger.derive_status()

        rc = session.session_config.runtime_config
        if rc.soft_max_steps > 0 and rc.hard_max_steps > 0 and rc.soft_max_steps == rc.hard_max_steps:
            warnings.warn(
                f"soft_max_steps and hard_max_steps are both {rc.soft_max_steps}; "
                "hard_max_steps prevails — the turn will close with ERRORED instead "
                "of a graceful soft-limit stop.",
                UserWarning,
                stacklevel=2,
            )

    def __eq__(self, other: object) -> bool:
        """Configuration equivalence: two runners are equal when they would
        drive a session the same way — equal session state and equivalent
        tool registry, prompt parts, assembler, provider, compaction policy,
        and middleware.
        Collaborators without their own `__eq__` (registries, assemblers,
        middleware) compare by class + instance state rather than
        identity."""
        if not isinstance(other, AgentSessionRunner):
            return NotImplemented
        return (
            self.session == other.session
            and _equivalent(self.tool_registry, other.tool_registry)
            and self.system_prompt_parts == other.system_prompt_parts
            and _equivalent(
                self.system_prompt_assembler,
                other.system_prompt_assembler,
            )
            and _equivalent(
                self.conversation_projector,
                other.conversation_projector,
            )
            and _equivalent(self.context_manager, other.context_manager)
            and _equivalent(self.compaction_policy, other.compaction_policy)
            and self.provider == other.provider
            and _all_equivalent(self.middleware, other.middleware)
        )

    # ── id + clock hooks (override points; no test parameters) ─────────────

    def generate_id(self) -> str:
        """Mint the id for the next entry. Every entry the runner creates gets
        its id here; override (subclass or mock) for deterministic ids."""
        return uuid4().hex[:8]

    def now_ms(self) -> int:
        """The runner's clock (unix ms). Every timestamp the runner writes
        comes from here; override (subclass or mock) to freeze time."""
        return _now_ms()

    # ── status predicates ──────────────────────────────────────────────────

    @property
    def status(self) -> ConversationStatus:
        return self.session.active_conversation.status

    def idle(self) -> bool:
        return self.status == ConversationStatus.IDLE

    def pending(self) -> bool:
        return self.status == ConversationStatus.PENDING

    def running(self) -> bool:
        return self.status == ConversationStatus.RUNNING

    def awaiting_approval(self) -> bool:
        return self.status == ConversationStatus.AWAITING_APPROVAL

    def cancelling(self) -> bool:
        return self.status == ConversationStatus.CANCELLING

    # ── caller-facing mutations / queries ────────────────────────────────────

    def post_message(self, content: str | list[ContentPart]) -> str:
        """Append a user message and arm the runner. Legal when the bracket is
        CLOSED and the status is IDLE or PENDING: a fresh/finished session,
        after a failed turn (add or clarify before the retry), or behind an
        already-queued message (queueing — consecutive user messages are an
        established shape). An open turn — CANCELLING, AWAITING_APPROVAL, or a
        resumable bracket — always rejects, and so does an open COMPACTION
        bracket: a scheduled or in-flight compaction must be driven before the
        session takes new input, durably, across a reload.

        `content` is a bare string (the common case) or an ordered list of
        parts mixing text and images; `before_post_message` sees that list and
        returns the one that is persisted."""
        if (
            self.status not in (ConversationStatus.IDLE, ConversationStatus.PENDING)
            or self.ledger.open_turn_index() is not None
        ):
            raise AgentError(
                f"post_message requires a closed turn and IDLE/PENDING status (status={self.status.value})."
            )
        parts = self._run_middlewares(
            "before_post_message",
            _normalize_post_parts(content),
        )
        message = self._append(
            lambda entry_id, parent_id, ts: UserMessage(
                id=entry_id,
                parent_id=parent_id,
                created_at=ts,
                parts=parts,
            )
        )
        self._set_status(ConversationStatus.PENDING)
        return message.id

    def schedule_compaction(self) -> str:
        """Arm the session for compaction: open a compaction bracket, write a
        `CompactionEntry(source=USER)`, and return its id. Idempotent.

        It does NOT compact — the work happens on the next drive, so the
        ordinary shape is schedule-then-drive:

            runner.schedule_compaction()
            await runner.run()

        The intent is therefore DURABLE: a process that dies before the drive
        leaves a session that still knows a compaction was asked for, and an
        open bracket already derives PENDING ("work is queued, call run()").
        The price is that `post_message` — which requires a closed bracket —
        raises until that drive has run, across a reload; this is exactly the
        treatment every other open bracket already gets.

        Requires a CLOSED bracket. An open conversational turn (a resumable
        bracket, AWAITING_APPROVAL, CANCELLING) raises `AgentError`: the
        appended `TurnStart` would nest inside it, and `open_turn_index()`
        walks back to the NEAREST one, so the open turn's eventual
        `TurnFinish` would silently close the wrong bracket. Raises with no
        `compaction_policy` configured — the runner would open a bracket
        nothing can close."""
        if self.compaction_policy is None:
            raise AgentError(
                "schedule_compaction requires a compaction_policy; this runner was constructed without one."
            )
        existing = self.ledger.open_compaction_entry()
        if existing is not None:
            return existing.id  # idempotent — nothing is written
        if self.ledger.open_turn_index() is not None:
            raise AgentError(f"schedule_compaction requires a closed turn (status={self.status.value}).")
        entry = self._open_compaction_bracket(CompactionSource.USER)
        self._set_status(ConversationStatus.PENDING)
        return entry.id

    def pending_approvals(self) -> list[ToolExecution]:
        """The open turn's executions awaiting an out-of-band approval — those
        whose `approval_status` is PENDING. Each is self-contained
        (`raw_tool_call` + whatever the registry recorded in `extras`);
        resolve them on the registry's own state, then call `run()` again (it
        asks the registry again — no posting back through the runner)."""
        return self.ledger.open_turn_awaiting_executions()

    def cancel(
        self,
        outcome: TurnOutcome = TurnOutcome.CANCELLED,
        error: str | None = None,
    ) -> None:
        """The universal cancellation door — synchronous, session-scoped
        ("cancel the turn, not the handle"), works in every state. Exactly
        three behaviors:

        1. Open turn, no unconsumed cancel (live run, suspended run, approval
           pause, reloaded crashed session — all the same): append a durable
           `CancelRequested(outcome, error)`, THEN trip the live run's token
           (entry-before-token is mandatory — a woken engine must always find
           the entry), set status CANCELLING, return immediately. The
           wind-down happens at the engine's next step boundary (live run) or
           on the next `run()`/`start()` — the *flush*. An unconsumed cancel
           controls EVERY close: an LLM answer landing within the grace
           window is recorded but the turn still closes with the requested
           outcome, and an LLM failure within the window closes the same way
           with the run returning normally (the failure is discarded).
        2. Open turn with an unconsumed `CancelRequested` already present →
           `AlreadyCancellingError` (diagnostic only; the first request's
           outcome/error stand).
        3. No open turn → no-op (nothing to cancel; `start()` opens the
           bracket at call time, so a started run is always cancellable —
           this branch is an undriven lazy handle, or no run at all)."""
        if self.ledger.open_turn_index() is None:
            return
        if self.ledger.open_turn_cancel_requested() is not None:
            raise AlreadyCancellingError(
                "a cancellation is already pending for the open turn; the first request's outcome/error stand"
            )
        self._append(
            lambda entry_id, parent_id, ts: CancelRequested(
                id=entry_id,
                parent_id=parent_id,
                created_at=ts,
                outcome=outcome,
                error=error,
            )
        )
        run = self._active_run
        if run is not None and run._token is not None:
            run._token.cancel()
        self._set_status(ConversationStatus.CANCELLING)

    # ── the two run methods ──────────────────────────────────────────────────

    def run(
        self,
        *,
        streaming: bool = False,
        on_event: EventCallback | None = None,
    ) -> AgentRun:
        """Lazy: nothing happens until awaited or iterated; stopping iteration
        stops the agent.

        Creating (and discarding) the handle is harmless — no work, no
        validation; the IDLE/concurrent-run guards fire at first drive.
        `streaming` selects the event vocabulary only (block events vs block +
        delta events). Events reach `on_event` even when the run is only
        awaited; without it, an awaited run discards them."""
        return AgentRun(self, streaming=streaming, on_event=on_event, eager=False)

    def start(
        self,
        *,
        streaming: bool = False,
        on_event: EventCallback | None = None,
    ) -> AgentRun:
        """Eager: begins immediately and completes regardless of observation;
        await to join, `.cancel()` to stop.

        Validates state synchronously at call time, opens the turn bracket
        durably (a `TurnStart` is appended if none is open — an immediate
        `cancel()` therefore parks the flush rather than no-opping), and
        spawns one background `asyncio.Task` (requires a running loop,
        resolved before any state is taken). Events are buffered
        regardless of observation — a late first consumer sees the full
        history from event 0 — and also delivered to `on_event`."""
        return AgentRun(self, streaming=streaming, on_event=on_event, eager=True)

    # ── run lifecycle plumbing (used by AgentRun) ────────────────────────────

    def _begin_run(self, run: AgentRun) -> None:
        """First-drive gate: one engine at a time, runnable status, and the
        run's own CancellationToken. There is no per-run context object:
        registries and tools receive the live session (`context.py`)."""
        if self._active_run is not None:
            raise AgentError("another run is already active on this runner; finish or finalize it first")
        if self.idle():
            raise AgentError("Nothing to run; call post_message() first.")
        run._token = CancellationToken()
        # Per-run, so a handle re-driven after an approval pause (or a
        # suspended lazy run resumed by a fresh `run()`) never reports the
        # previous bracket's outcome. `None` is correct for a run that closed
        # nothing.
        self._closed_outcome = None
        self._compaction_consumed_cancel = False
        self._compaction_ran = False
        self._active_run = run

    def _end_run(self, run: AgentRun) -> None:
        if self._active_run is run:
            self._active_run = None

    def _refresh_status(self) -> None:
        self._set_status(self.ledger.derive_status())

    def _ensure_open_turn(self) -> None:
        """Open a new bracket unless one is already open (resume). Called at
        the engine's top for lazy runs and synchronously at `start()` time for
        eager runs — a started run is cancellable before its first tick."""
        if self.ledger.open_turn_index() is None:
            self._append(
                lambda entry_id, parent_id, ts: TurnStart(
                    id=entry_id,
                    parent_id=parent_id,
                    created_at=ts,
                )
            )

    def _open_bracket_for_start(self) -> None:
        """`start()`'s call-time bracket decision: a COMPACTION bracket when
        one is due, otherwise the ordinary `TurnStart`.

        This is why `should_compact` is sync. `AgentRun.__init__` opens a
        bracket synchronously, so without this an eager run would always
        present the drive with an open conversational turn — and the drive's
        compaction step skips when it finds one, so a policy-driven compaction
        would never fire for an eager run at all. An already-scheduled
        compaction needs nothing special: its bracket is open, so
        `_ensure_open_turn` is already a no-op.

        A `should_compact` that raises propagates from `start()` — swallowing
        it would make a broken policy indistinguishable from one that
        declines, and nothing durable exists yet to record the failure on."""
        if (
            self.compaction_policy is not None
            and self.ledger.open_turn_index() is None
            and self.compaction_policy.should_compact(self.session)
        ):
            self._open_compaction_bracket(CompactionSource.POLICY)
            return
        self._ensure_open_turn()

    def _build_run_result(self) -> RunResult:
        """Snapshot where the engine stopped: the DERIVED status, plus how the
        last bracket this run closed ended (None if it closed none).

        Neither can be read off the path any more. After a successful
        compaction the closing `TurnFinish` is on the ARCHIVED conversation,
        and after a compaction-only drive the active leaf may be the
        `CompactionEntry` itself or a carried assistant message — neither has
        an outcome to read."""
        if self.status == ConversationStatus.AWAITING_APPROVAL:
            return RunResult(
                status=ConversationStatus.AWAITING_APPROVAL,
                outcome=None,
                pending_approvals=self.pending_approvals(),
            )
        return RunResult(
            status=self.status,
            outcome=self._closed_outcome,
            pending_approvals=[],
        )

    # ── middleware machinery ─────────────────────────────────────────────────

    def _run_middlewares(
        self,
        method_name: str,
        value,
        *ctx_args,
        unpack_values: bool = False,
        **ctx_kwargs,
    ):
        """Thread `value` through each middleware's `method_name` hook in order.
        Context args/kwargs are forwarded unchanged to every call. With
        `unpack_values=True`, `value` is a tuple unpacked as positional args
        (used for `before_llm_call` which takes and returns a pair)."""
        for mw in self.middleware:
            if not hasattr(mw, method_name):
                continue
            if unpack_values:
                value = getattr(mw, method_name)(*value, *ctx_args, **ctx_kwargs)
            else:
                value = getattr(mw, method_name)(value, *ctx_args, **ctx_kwargs)
        return value

    def _complete_entry(self, entry: AnyEntry) -> AnyEntry:
        """The fallible half of preparing any entry: calculate its
        `context_tokens` (context calculation is part of preparing a complete
        entry — it runs BEFORE middleware, and never again after), then thread
        it through `before_entry_written`."""
        entry.context_tokens = self.context_manager.calculate_context(
            self.session,
            entry,
        )
        return self._run_middlewares("before_entry_written", entry)

    def _append(self, build_fn) -> AnyEntry:
        """Append one entry: build it, complete it, commit it to the ledger."""
        return self.ledger.append(
            lambda entry_id, parent_id, ts: self._complete_entry(
                build_fn(entry_id, parent_id, ts),
            )
        )

    def _complete_uncommitted(
        self,
        build_fn,
        parent_id: str | None,
        ts: int,
    ) -> AnyEntry:
        """A complete, fully-middlewared entry that is NOT committed — the
        other half of `_append`, for entries that belong to a conversation
        which does not exist yet (a compaction plan's). Identity comes from
        the same hooks; the caller supplies the parent, because the new
        conversation's leaf is not the active path's.

        (Named for what it does rather than `_prepare`, so `prepare` in this
        file means the tool-lifecycle phase and nothing else.)"""
        return self._complete_entry(build_fn(self.generate_id(), parent_id, ts))

    def recalculate_context_tokens(self) -> None:
        """Re-derive `context_tokens` for EVERY entry in `session.entries`,
        threading each through `before_entry_written`.

        `Entry.context_tokens` is stored, so a `ContextManager` that counts
        against the active model leaves every stored count stale the moment
        `session_config.llm_config` changes — and a context gauge that sums
        them then reports a number with no single basis. This is the way back.

        Every entry, not just the active path: the count is intrinsic to an
        entry and shared by every conversation that references it, so
        refreshing only the active path would leave archived conversations on
        the old basis. Through the middleware door, because middleware has the
        final say on context and nothing is recomputed behind it. It sets no
        other field.

        NOTHING IN THE FRAMEWORK CALLS THIS. There is no constructor keyword,
        no CLI flag, and no automatic invocation on a model switch (which
        would put an unbounded rewrite behind an innocuous-looking
        assignment). The shipped `ContextManager` is a character estimate that
        no model choice affects; this exists for the application that swaps in
        a real tokenizer, and that application calls it."""
        for entry_id in list(self.session.entries):
            entry = self.session.entries[entry_id]
            refreshed = self._complete_entry(entry.model_copy())
            self.ledger.refresh_entry(refreshed)

    def _persist_entry(
        self,
        entry: AnyEntry,
        *,
        recalculate: bool = True,
        **changes,
    ) -> AnyEntry:
        """The one in-place update path, for both mutable entry types: copy
        with `changes`, complete, store. `recalculate=False` runs middleware
        ONLY — used for a tool execution's final persist, where
        `_finalize_outcome` has already settled the context and middleware
        must keep the final say on it."""
        updated = entry.model_copy(update=changes)
        updated = (
            self._complete_entry(updated) if recalculate else self._run_middlewares("before_entry_written", updated)
        )
        return self.ledger.put_entry(updated)

    def _persist_execution(
        self,
        execution: ToolExecution,
        *,
        recalculate: bool = True,
        **changes,
    ) -> ToolExecution:
        """`_persist_entry` plus the `ToolExecution`-only `updated_at` stamp.
        Every execution persistence — approval updates, the RUNNING
        transition, cancellation stamps, terminal outcomes — lands here."""
        return self._persist_entry(
            execution,
            recalculate=recalculate,
            **changes,
            updated_at=self.now_ms(),
        )

    # ── tool-execution outcome machinery ─────────────────────────────────────

    def to_tool_execution_error(
        self,
        execution: ToolExecution,
        exception: Exception,
        *,
        phase: str,
    ) -> ToolExecutionError:
        """Convert a live exception into the durable `ToolExecutionError`.
        Override to redact secrets, preserve domain codes, or add a traceback
        to `details` — the live exception itself is never persisted.

        The default keeps the exception's type and message and records the
        failure phase, nesting structured validation errors under
        `details["errors"]` where they exist.

        `phase` is a FACT the caller knows — which of the runner's three
        observation points the raise came out of — not an inference from
        `started_at`: `"create_execution"`, `"prepare"`, or `"execution"`. It
        is populated on every registry- or tool-owned raise. Those three
        values are the RUNNER's vocabulary for raises it observed; a registry
        authoring a terminal-at-birth error owns its own `details` and may use
        its own phase vocabulary."""
        details: dict = {"phase": phase}
        if isinstance(exception, InvalidToolArguments):
            details["errors"] = exception.errors
        elif isinstance(exception, ValidationError):
            details["errors"] = json.loads(exception.json(include_url=False))
        return ToolExecutionError(
            error_type=type(exception).__name__,
            error_message=str(exception),
            details=details,
        )

    def _tool_executed_event(self, execution: ToolExecution) -> ToolExecuted:
        """Project the terminal execution once and derive the event's
        presentation fields from it. The next LLM request re-projects the same
        durable execution (projection is deterministic), so event and wire
        always agree."""
        message = self.conversation_projector.project_tool_execution(
            execution,
            self.session.entries,
        )
        return ToolExecuted(
            tool_call_id=execution.tool_call_id,
            execution=execution.model_copy(deep=True),
            result_text=tool_message_text(message),
            is_error=message.is_error,
        )

    def _finalize_outcome(
        self,
        execution: ToolExecution,
        exception: Exception | None = None,
    ) -> tuple[ToolExecution, ToolExecuted]:
        """The shared tail of every execution outcome: recalculate
        `context_tokens` from the final model-facing result or error (the
        birth count was 0 — no outcome existed; context always settles
        BEFORE middleware, never after) → `after_tool_execution` over the
        fully formed execution → final persistence through
        `before_entry_written` → the `ToolExecuted` event."""
        execution = execution.model_copy(
            update={
                "context_tokens": self.context_manager.calculate_context(
                    self.session,
                    execution,
                ),
            },
        )
        execution = self._run_middlewares(
            "after_tool_execution",
            execution,
            exception,
        )
        execution = self._persist_execution(execution, recalculate=False)
        return execution, self._tool_executed_event(execution)

    def _finalize_undispatched(
        self,
        execution: ToolExecution,
        exception: Exception | None = None,
    ) -> tuple[ToolExecution, ToolExecuted]:
        """Outcome pipeline for a call whose body will never run — terminal
        at birth, REJECTED, or CANCELLED before dispatch. These calls still
        pass through `before_tool_execution` (which sees the terminal status
        already set) before the shared outcome tail."""
        execution = self._run_middlewares("before_tool_execution", execution)
        return self._finalize_outcome(execution, exception)

    def _recover_orphans(self) -> list[AgentEvent]:
        """A persisted RUNNING execution without its live runtime task is an
        orphan (a crash, or a drive suspended mid-body). Transition it to
        INTERRUPTED: `after_tool_execution` runs with no exception,
        `before_tool_execution` is NOT re-invoked, and the tool is never
        automatically re-dispatched. Durable state records nothing
        crash-specific — an orphan is exactly another INTERRUPTED execution."""
        events: list[AgentEvent] = []
        for execution in self.ledger.open_turn_running_executions():
            terminal = execution.model_copy(
                update={
                    "status": ExecutionStatus.INTERRUPTED,
                    "ended_at": self.now_ms(),
                },
            )
            _, event = self._finalize_outcome(terminal)
            events.append(event)
        return events

    # ── per-call build methods ───────────────────────────────────────────────

    def build_model_string(self, llm_cfg: LLMConfig) -> str:
        """Build the model identifier for the LLM client, threading it through
        any `build_model_string` middleware. Called per LLM invocation."""
        model_string = f"{llm_cfg.provider}:{llm_cfg.model}"
        return self._run_middlewares("build_model_string", model_string, llm_cfg)

    async def build_tool_list(self) -> list[LucaTool]:
        """Return the wire tool list for this LLM call: query the registry
        fresh (`get_tools` is dynamic — the result may vary with session
        state; a toolless runner contributes none), convert each `ToolSpec`
        via the adapter, and thread the list through any `build_tool_list`
        middleware. Called per LLM invocation.

        Async because `get_tools` is — a registry that needs I/O to list its
        tools (a remote tool server, a plugin host, a permissions service)
        must not block the event loop. The `build_tool_list` MIDDLEWARE hook
        stays synchronous and unchanged: it runs on the converted WIRE list,
        after the await. The drive races this whole method against the
        cancellation token rather than the inner `get_tools`, so a subclass
        that overrides it is covered for free and the token never has to
        appear in this public signature."""
        specs = await self.tool_registry.get_tools(self.session) if self.tool_registry is not None else []
        tool_list = [adapter.tool_spec_to_luca_tool(spec) for spec in specs]
        return self._run_middlewares("build_tool_list", tool_list)

    def build_messages(self) -> list:
        """Project the active conversation to canonical client messages via
        the configured `ConversationProjector` — derived per call, never
        stored. History-shaping policy belongs on the projector itself (there
        is no projection middleware); `before_llm_call` remains downstream for
        last-mile request changes."""
        return self.conversation_projector.project(
            self.session.active_conversation,
            self.session.entries,
        )

    def build_system_message(self) -> str | None:
        """Assemble the system prompt for one LLM call: resolve the parts
        (a callable part is invoked with the live session config and runtime
        status, its return value coerced like a static part), sort them by
        priority, assemble. A blank result means no system message is sent
        at all."""
        parts = [
            coerce_system_prompt_part(
                part(
                    self.session.session_config,
                    self.session.session_runtime_status,
                )
            )
            if callable(part)
            else part
            for part in self.system_prompt_parts
        ]
        parts = sorted(parts, key=lambda part: part.priority)
        prompt = self.system_prompt_assembler.assemble_system_prompt(parts)
        return prompt if prompt.strip() else None

    def prepare_llm_call(self) -> tuple[list, str | None]:
        """Build the (messages, system_message) pair for the next LLM call.
        Calls `build_messages()` and `build_system_message()`, then threads
        the pair through any `before_llm_call` middleware."""
        messages = self.build_messages()
        system_message = self.build_system_message()
        return self._run_middlewares(
            "before_llm_call",
            (messages, system_message),
            unpack_values=True,
        )

    # ── compaction ───────────────────────────────────────────────────────────

    def _open_compaction_bracket(
        self,
        source: CompactionSource,
    ) -> CompactionEntry:
        """`TurnStart` then `CompactionEntry` — two plain appends, exactly as
        safe as recording a user message, and adjacent by construction (which
        is what makes the compaction-bracket predicate exact). Emits nothing;
        the drive emits `CompactionScheduled`."""
        self._append(
            lambda entry_id, parent_id, ts: TurnStart(
                id=entry_id,
                parent_id=parent_id,
                created_at=ts,
            )
        )
        return self._append(
            lambda entry_id, parent_id, ts: CompactionEntry(
                id=entry_id,
                parent_id=parent_id,
                created_at=ts,
                source=source,
            )
        )

    def _snapshot_conversation(self) -> ConversationSnapshot:
        """The full active path (G2's input) plus the view handed to the
        policy: the same path with THIS compaction's `TurnStart` removed.

        Exactly one element, positionally — the tail is `[…, ts_c, cmp]` by
        construction (the two appends are consecutive, `post_message` raises
        while the bracket is open, and a parked cancel flushes before the
        policy is ever reached), so this is a removal, not a filter over
        types. `cmp` deliberately STAYS in the view: stripping it too would
        make `plan.nodes = list(nodes)` illegal, trading one invisible
        requirement for another."""
        conversation = self.session.active_conversation
        nodes = tuple(conversation.nodes)
        bracket = self.ledger.open_turn_index()
        return ConversationSnapshot(
            id=conversation.id,
            nodes=nodes,
            offered=nodes[:bracket] + nodes[bracket + 1 :],
        )

    async def _compaction_step(
        self,
        token: CancellationToken,
    ) -> AsyncIterator[AgentEvent]:
        """The whole compaction operation, run once at the top of a drive,
        BEFORE the conversational bracket opens.

        Flush a parked cancel first; then resume an interrupted compaction,
        skip entirely (an open conversational turn — an approval pause, a
        crash mid-turn, a suspended run — is never carved up by a policy), or
        ask the policy whether one is due; then run it. The bracket never
        stays open past the end of a drive that did not crash.

        Everything expensive and failure-prone happens BEFORE the transition,
        which is what makes the operation safe by construction rather than by
        recovery logic: a failure here closes the bracket and leaves the
        conversation exactly as it was, and a crash leaves the bracket open —
        the resumable state."""
        if self.compaction_policy is None:
            return

        entry = self.ledger.open_compaction_entry()

        # 1) FLUSH FIRST. A parked cancel inside an open compaction bracket
        #    ends it now — no Scheduled, no Started, no policy call.
        if entry is not None:
            cancel = self.ledger.open_turn_cancel_requested()
            if cancel is not None:
                self._compaction_consumed_cancel = True
                yield self._close_compaction(entry, cancel.outcome, cancel.error)
                return

        # 2) RESUME, SKIP, or DECIDE. An open bracket around an entry that
        #    already has `parts` is NOT resumable (G6) — the ledger's read
        #    answers None for it, so it lands in the skip branch below and the
        #    drive treats it as the phantom conversational turn it is.
        if entry is None:
            if self.ledger.open_turn_index() is not None:
                return  # an open conversational turn → not this drive
            if not self.compaction_policy.should_compact(self.session):
                return
            entry = self._open_compaction_bracket(CompactionSource.POLICY)
        self._compaction_ran = True  # past every early return
        yield CompactionScheduled(entry=entry.model_copy(deep=True))

        # 3) RUN IT. `started_at` is stamped once, on the first attempt, so a
        #    resumed compaction keeps the original stamp and the previous
        #    attempt stays visible in the log.
        if entry.started_at is None:
            entry = self._persist_entry(entry, started_at=self.now_ms())
        yield CompactionStarted(entry=entry.model_copy(deep=True))

        snapshot = self._snapshot_conversation()
        try:
            plan = await self._invoke_policy(entry, snapshot.offered, token)
        except _CompactionEnded as stop:
            # A cancellation always wins over the deadline that raced it, and
            # always stops the drive: the cancel is against the drive, not
            # against the compaction alone. `cancel()` writes the durable
            # request BEFORE tripping the token, so a token-fired ending
            # always finds one here and only a deadline reaches the lines
            # below — which is why they raise the same timeout the
            # conversational LLM call raises.
            cancel = self.ledger.open_turn_cancel_requested()
            if cancel is not None:
                self._compaction_consumed_cancel = True
                yield self._close_compaction(entry, cancel.outcome, cancel.error)
                return
            yield self._close_compaction(entry, stop.outcome, stop.error)
            if entry.source == CompactionSource.USER:
                raise ClientTimeoutError(stop.error) from None
            return
        except Exception as exc:
            yield self._close_compaction(entry, _outcome_for(exc), str(exc))
            if entry.source == CompactionSource.USER:
                raise
            return  # a POLICY failure DEGRADES — the user's turn survives

        if plan is not None:
            # G2 first, then the usage write, both inside the failure
            # handling. G2 first because a policy that replaced the active
            # conversation would otherwise make `record_usage` raise its own
            # unrelated error and pre-empt the plan rejection; guarded because
            # an escape here would leave the bracket OPEN — the "resume me"
            # state — and the next drive would replay the same failing call
            # with `should_compact` never consulted.
            try:
                check_snapshot(session=self.session, snapshot=snapshot)
                self.ledger.record_usage(entry.id, **plan.usage.model_dump())
            except Exception as exc:
                yield self._close_compaction(entry, TurnOutcome.ERRORED, str(exc))
                if entry.source == CompactionSource.USER:
                    raise
                return

        # A cancel that arrived within the grace window DISCARDS the plan.
        # Unlike the LLM path, which records a within-grace answer: adding a
        # node is not rewriting what the conversation is.
        cancel = self.ledger.open_turn_cancel_requested()
        if cancel is not None:
            self._compaction_consumed_cancel = True
            yield self._close_compaction(entry, cancel.outcome, cancel.error)
            return

        if plan is None:  # the ONE "nothing to do" signal
            yield self._close_compaction(entry, TurnOutcome.COMPLETED)
            return

        try:
            conversation, final, created = self._commit(entry, plan, snapshot)
        except Exception as exc:
            yield self._close_compaction(entry, TurnOutcome.ERRORED, str(exc))
            if entry.source == CompactionSource.USER:
                raise
            return

        self._closed_outcome = TurnOutcome.COMPLETED  # _close_turn never ran
        yield CompactionFinished(
            entry=final.model_copy(deep=True),
            outcome=TurnOutcome.COMPLETED,
            created=[e.model_copy(deep=True) for e in created],
            conversation_id=conversation.id,
        )

    async def _invoke_policy(
        self,
        entry: CompactionEntry,
        offered: tuple[str, ...],
        token: CancellationToken,
    ) -> CompactionPlan | None:
        """Call `compact()` under the run's cancellation race and the session's
        wall-clock deadline.

        The runner races the call itself: `client_completion_timeout_in_ms` is
        a kwarg the runner passes to the client, and it cannot reach a request
        the policy makes on its own. The value is converted exactly as the LLM
        step converts it, so the default (`Inf`) means NO deadline at all —
        a hung policy hangs the drive until cancelled, identical to the
        conversational call's default.

        The policy is handed a DEEP COPY of the entry and the LIVE session.
        The copy is load-bearing: a policy that wrote `parts` onto the live
        entry and then failed would leave the bracket closed ERRORED, the path
        unchanged — and the entry projecting a summary of nothing."""
        config = self.session.session_config.runtime_config
        grace_ms = config.llm_completion_cancellation_grace_period
        deadline = _ms_to_seconds(config.client_completion_timeout_in_ms)
        task = asyncio.ensure_future(
            self.compaction_policy.compact(
                self.session,
                offered,
                entry.model_copy(deep=True),
            )
        )
        if deadline is None:
            completed, plan, _ = await _race_cancellation(
                task,
                token,
                grace_ms,
                None,
            )
        else:
            try:
                async with asyncio.timeout(deadline) as scope:
                    completed, plan, _ = await _race_cancellation(
                        task,
                        token,
                        grace_ms,
                        None,
                    )
            except TimeoutError:
                if not scope.expired():
                    raise  # the policy's OWN TimeoutError — a normal raise
                await _kill(task, detach=False)  # idempotent backstop
                raise _CompactionEnded(
                    TurnOutcome.TIMED_OUT,
                    f"compaction exceeded total_timeout={deadline}s",
                ) from None
        if not completed:  # the token fired and the grace expired
            raise _CompactionEnded(TurnOutcome.CANCELLED, None)
        return plan

    def _close_compaction(
        self,
        entry: CompactionEntry,
        outcome: TurnOutcome,
        error: str | None = None,
    ) -> CompactionFinished:
        """The single NON-TRANSITION ending: stamp `ended_at`, close the
        bracket on the pre-compaction path, return the event.

        `parts` and `compacted_nodes` stay None — nothing was committed, so
        nothing may project. Both writes run `before_entry_written` and are
        therefore fallible; a middleware raise here leaves the bracket open,
        which is the recoverable state, and propagates."""
        final = self._persist_entry(entry, ended_at=self.now_ms())
        self._close_turn(outcome, error)
        return CompactionFinished(
            entry=final.model_copy(deep=True),
            outcome=outcome,
            error=error,
            created=[],
            conversation_id=None,
        )

    def _commit(
        self,
        entry: CompactionEntry,
        plan: CompactionPlan,
        snapshot: ConversationSnapshot,
    ) -> tuple[Conversation, CompactionEntry, list[AnyEntry]]:
        """Everything fallible, then one infallible door.

        Validation, the context recalculation, entry middleware for the
        updated compaction entry and every entry the plan creates, and the
        closing `TurnFinish` are all PREPARED here and stored by
        `transition_conversation`. In particular the marker is built but not
        appended: `_close_turn` would put it on the active path and close the
        bracket COMPLETED ahead of a transition that then failed, leaving a
        summary projecting onto an unchanged conversation."""
        validate_plan(
            plan,
            entry_id=entry.id,
            session=self.session,
            snapshot=snapshot,
        )
        ts = self.now_ms()  # ONE timestamp for the whole transition
        # The bracket is still open — the closing marker is only BUILT below —
        # and G2 has just proven this is the same index `_snapshot_conversation`
        # stripped at, so the offered view and the compacted span can never
        # disagree about where the bracket is.
        bracket = self.ledger.open_turn_index()
        carried = {node for node in plan.nodes if isinstance(node, str)}
        # Over the path BEFORE the bracket, so the compaction's own `ts_c`
        # never lands in the list of ids this entry replaced.
        compacted = [node for node in snapshot.nodes[:bracket] if node not in carried]

        final = self._complete_entry(
            entry.model_copy(
                update={
                    "parts": plan.entry.parts,
                    "llm_config": plan.entry.llm_config,
                    "metadata": plan.entry.metadata,
                    "compacted_nodes": compacted,
                    "ended_at": ts,
                },
            )
        )
        nodes: list[str] = []
        created: list[AnyEntry] = []
        parent: str | None = None
        for node in plan.nodes:
            if isinstance(node, str):
                parent = node
            else:
                built = self._complete_uncommitted(
                    lambda entry_id, parent_id, stamp, template=node: template.model_copy(
                        update={
                            "id": entry_id,
                            "parent_id": parent_id,
                            "created_at": stamp,
                        },
                    ),
                    parent,
                    ts,
                )
                created.append(built)
                parent = built.id
            nodes.append(parent)
        closing = self._complete_uncommitted(
            lambda entry_id, parent_id, stamp: TurnFinish(
                id=entry_id,
                parent_id=parent_id,
                created_at=stamp,
                outcome=TurnOutcome.COMPLETED,
            ),
            self.session.active_conversation.nodes[-1],
            ts,
        )
        # ─────────────────────────── THE TRANSITION ──────────────────────────
        conversation = self.ledger.transition_conversation(
            updates=[final],
            created=created,
            closing=closing,
            nodes=nodes,
            ts=ts,
        )
        return conversation, final, created

    # ── the engine ───────────────────────────────────────────────────────────

    async def _drive(
        self,
        streaming: bool,
        token: CancellationToken,
    ) -> AsyncIterator[AgentEvent]:
        """The single engine behind both methods; `AgentRun` is its only
        consumer (lazy pulls it directly, eager drains it from the background
        task)."""
        self._set_status(ConversationStatus.RUNNING)

        # Compaction runs BEFORE the conversational bracket, at most once per
        # drive (structural: the step sits outside the loop below).
        async for event in self._compaction_step(token):
            yield event
        if self._compaction_consumed_cancel:
            # The cancel was against the DRIVE. Consuming it and then going on
            # to answer the queued turn would defy the instruction.
            return
        if self._compaction_ran and self.ledger.derive_status() == ConversationStatus.IDLE:
            # A compaction-only drive: a user-scheduled compaction on an
            # otherwise-finished session was PENDING only because its bracket
            # was open. With it closed there is nothing to drive, and opening
            # a turn would call the model with no user input. Gated on the
            # step having done something, so every other drive is provably
            # unchanged.
            return

        # A committed transition installed a NEW conversation whose status
        # came from derivation, so RUNNING has to be set again — an
        # application polling status mid-drive would otherwise see PENDING.
        self._set_status(ConversationStatus.RUNNING)
        # Resume the open turn if one exists; otherwise open a new bracket
        # (an eager run already opened it at start() time).
        self._ensure_open_turn()

        # Crash recovery: any persisted RUNNING execution has no live task on
        # a fresh drive — terminalize it as INTERRUPTED before anything else
        # (before the flush too, so a parked cancel never CANCELLs a call
        # whose body actually started).
        for event in self._recover_orphans():
            yield event

        while True:
            # 0) Cancel check — every step boundary funnels back here. An
            # unconsumed CancelRequested ends the turn NOW; the same path is
            # the parked-cancel FLUSH (run()/start() on a CANCELLING session),
            # which may legitimately emit zero events.
            cancel_entry = self.ledger.open_turn_cancel_requested()
            if cancel_entry is not None:
                for event in self._wind_down(cancel_entry):
                    yield event
                return

            # 1) Undecided executions → THE decide() call site. Serves the
            # fresh path (created one iteration ago) and every resume path (a
            # re-entered run, a reloaded session) identically. A registry
            # response updates approval_status directly and lands in the
            # audit log; a DENY is terminal on the spot. All decision writes
            # land before any denial event is yielded.
            undecided = self.ledger.open_turn_undecided_executions()
            if undecided:
                pairs = await asyncio.gather(*(self._decide_one(ex, token) for ex in undecided))
                denial_events: list[AgentEvent] = []
                for pair in pairs:
                    if pair is None:
                        continue  # the token won this decide — nothing decided
                    modified, decision = pair
                    denied = decision.decision == ApprovalOption.DENY
                    changes: dict = {
                        "approval_decisions": [
                            *modified.approval_decisions,
                            decision,
                        ],
                        "approval_status": _APPROVAL_STATUS[decision.decision],
                    }
                    if denied:
                        changes["status"] = ExecutionStatus.REJECTED
                        changes["ended_at"] = self.now_ms()
                    persisted = self._persist_execution(modified, **changes)
                    if denied:
                        _, event = self._finalize_undispatched(persisted)
                        denial_events.append(event)
                for event in denial_events:
                    yield event

            # 2) Dispatch every ALLOWED-and-unrun execution. An allowed
            # sibling proceeds even while another call sits deferred — the
            # runner parks only after all currently runnable work advanced.
            ready = self.ledger.open_turn_ready_executions()
            if ready:
                async for event in self._dispatch_batch(ready, token):
                    yield event

            # 3) Park while any approval remains explicitly deferred (a
            # cancel that landed mid-decide or mid-dispatch wins instead:
            # wind down rather than pausing at the gate).
            awaiting = self.ledger.open_turn_awaiting_executions()
            if awaiting:
                cancel_entry = self.ledger.open_turn_cancel_requested()
                if cancel_entry is not None:
                    for event in self._wind_down(cancel_entry):
                        yield event
                    return
                self._set_status(ConversationStatus.AWAITING_APPROVAL)
                yield ApprovalRequired(
                    executions=[ex.model_copy(deep=True) for ex in self.pending_approvals()],
                )
                return

            if undecided or ready:
                continue  # re-run the cancel check before calling the model

            # 4) Step-limit and doom-loop checks, then call the model — only
            # reached when every execution in the open turn is terminal.
            #
            # Hard max: if the open turn already has step_count LLM responses
            # and step_count >= hard_max_steps, close the turn now.
            # Soft max / doom loop: restrict tool_choice to "none" so the LLM
            # can only reply with text, ending the turn gracefully.
            config = self.session.session_config.runtime_config
            step_count = self.session.session_runtime_status.step_count
            if config.hard_max_steps > 0 and step_count >= config.hard_max_steps:
                self._close_turn(
                    TurnOutcome.ERRORED,
                    error=f"Hard max steps limit reached: {step_count}",
                )
                return

            tool_choice: str | None = None
            if (
                config.soft_max_steps > 0
                and step_count >= config.soft_max_steps
                and config.limit_tool_choice_on_soft_max_steps_reached
            ):
                tool_choice = "none"
            if config.limit_tool_choice_on_doom_loop_flagged and self.ledger.open_turn_has_doom_loop_flagged():
                tool_choice = "none"

            # Call the model, racing the run's token (§R4): on cancel the
            # call is torn down (httpx closes the connection) and NOTHING from
            # the aborted attempt is recorded — control returns to the loop
            # top, which winds down. A non-zero grace period waits that long
            # for a natural finish first; an answer landing in time is
            # recorded, but a pending cancel still controls the close. A
            # FAILED call (timeout / provider error) closes the turn
            # (TIMED_OUT / ERRORED, status PENDING — retry-ready) and
            # re-raises — unless a cancel is pending, which wins; safe here
            # and only here, because the model call runs only after every
            # execution is terminal.
            llm_cfg = self.session.session_config.llm_config
            model_string = self.build_model_string(llm_cfg)
            messages, system_message = self.prepare_llm_call()
            # `get_tools` is application code and may block indefinitely, so
            # the whole step is raced. A lost race produces no tool list and
            # makes no LLM call: control returns to the loop top, which winds
            # the turn down — exactly the aborted-LLM-call path. A RAISE is
            # deliberately not caught here: it propagates and aborts the run
            # with the turn left open and resumable, and the next run() asks
            # again. Substituting an empty tool list would silently change the
            # model's answer.
            tools_task = asyncio.ensure_future(self.build_tool_list())
            tools_done, tool_list, _ = await _race_cancellation(
                tools_task,
                token,
                0,
                None,
            )
            if not tools_done:
                continue
            grace_ms = config.llm_completion_cancellation_grace_period
            request_timeout = _ms_to_seconds(
                config.builtin_client_completion_timeout_in_ms,
            )
            total_timeout = _ms_to_seconds(config.client_completion_timeout_in_ms)

            try:
                if streaming:
                    message = None
                    finish_reason = None
                    aborted = False
                    grace_deadline = None
                    stream = acompletion_stream(
                        model=model_string,
                        messages=messages,
                        system_message=system_message,
                        tools=tool_list or None,
                        tool_choice=tool_choice,
                        reasoning=llm_cfg.reasoning,
                        provider=self.provider,
                        timeout=request_timeout,
                        total_timeout=total_timeout,
                    )
                    async with stream as s:
                        iterator = s.__aiter__()
                        try:
                            while True:
                                step = asyncio.ensure_future(iterator.__anext__())
                                try:
                                    completed, stream_event, grace_deadline = await _race_cancellation(
                                        step,
                                        token,
                                        grace_ms,
                                        grace_deadline,
                                    )
                                except StopAsyncIteration:
                                    break
                                if not completed:
                                    aborted = True
                                    break
                                if stream_event.type == "finish":
                                    message = stream_event.message
                                    finish_reason = stream_event.finish_reason
                                elif stream_event.type == "error":
                                    raise stream_event.error
                                elif (delta := _to_delta_event(stream_event)) is not None:
                                    yield delta
                        finally:
                            await iterator.aclose()
                    if aborted:
                        continue  # partial dropped; the loop top winds down
                    if message is None:
                        raise RuntimeError("stream ended without a FinishEvent")
                else:
                    llm_task = asyncio.ensure_future(
                        acompletion(
                            model=model_string,
                            messages=messages,
                            system_message=system_message,
                            tools=tool_list or None,
                            tool_choice=tool_choice,
                            reasoning=llm_cfg.reasoning,
                            provider=self.provider,
                            timeout=request_timeout,
                            total_timeout=total_timeout,
                        )
                    )
                    completed, response, _ = await _race_cancellation(
                        llm_task,
                        token,
                        grace_ms,
                        None,
                    )
                    if not completed:
                        continue  # nothing recorded; the loop top winds down
                    message = response.message
                    finish_reason = response.finish_reason
            except Exception as exc:
                # asyncio.CancelledError passes through untouched (crash
                # semantics); the client TimeoutError covers both tiers. An
                # unconsumed cancel controls this close too: the call was
                # being torn down anyway, so the requested outcome stands and
                # the run returns normally (the failure is discarded).
                cancel_entry = self.ledger.open_turn_cancel_requested()
                if cancel_entry is not None:
                    for event in self._wind_down(cancel_entry):
                        yield event
                    return
                outcome = TurnOutcome.TIMED_OUT if isinstance(exc, ClientTimeoutError) else TurnOutcome.ERRORED
                self._close_turn(outcome, error=str(exc))
                raise

            # Run after_llm_response middleware before recording: the message
            # is fully assembled (streaming or non-streaming) so all content
            # blocks are present.
            message = self._run_middlewares("after_llm_response", message)

            # Record the assistant message, create its executions, and (for a
            # final answer) close the bracket ATOMICALLY — every session write
            # for this round lands before the first yield, so a suspend can
            # never strand a tool_use request without its ToolExecutions, nor
            # leave a fully-answered bracket open to a duplicate LLM call.
            # The round keys off the tool_calls themselves, not finish_reason:
            # a misclassifying provider can neither wedge the conversation
            # ("stop" + calls) nor loop it ("tool_use" + none).
            events = self._record_assistant(message, finish_reason, llm_cfg)
            if message.tool_calls:
                events.extend(await self._create_executions(message, token))
                for event in events:
                    yield event
                continue  # → step 1 hands the fresh executions to decide()

            # Final answer: an unconsumed cancel controls the close — the
            # within-grace message stays recorded, the requested outcome wins.
            cancel_entry = self.ledger.open_turn_cancel_requested()
            if cancel_entry is not None:
                events.extend(self._wind_down(cancel_entry))
            else:
                self._close_turn(TurnOutcome.COMPLETED)
            for event in events:
                yield event
            return

    # ── per-step machinery ─────────────────────────────────────────────────

    def _wind_down(self, cancel_entry: CancelRequested) -> list[AgentEvent]:
        """Consume a `CancelRequested`: every still-PENDING execution in the
        open turn is stamped `cancel_signalled_at` and becomes CANCELLED —
        resultless, errorless, approval state untouched. (A denied call was
        already terminal REJECTED at decision time; an in-flight one was
        settled by the grace machinery; an orphaned RUNNING one was recovered
        at drive start.) Each cancelled execution passes through the outcome
        middleware pair, the turn closes with the requested outcome, and the
        `ToolExecuted` events return to the caller. All session writes happen
        before any event is yielded."""
        events: list[AgentEvent] = []
        for execution in self.ledger.open_turn_pending_executions():
            ts = self.now_ms()
            stamped = execution.model_copy(
                update={
                    "cancel_signalled_at": ts,
                    "status": ExecutionStatus.CANCELLED,
                    "result": None,
                    "error": None,
                    "ended_at": ts,
                },
            )
            _, event = self._finalize_undispatched(stamped)
            events.append(event)
        self._close_turn(cancel_entry.outcome, cancel_entry.error)
        return events

    def _record_assistant(
        self,
        message,
        finish_reason,
        llm_cfg: LLMConfig,
    ) -> list[AgentEvent]:
        """Append the assistant message and write its provider-usage record
        to `AgentSession.usages` (usage is accessory conversation-entry data,
        never embedded in the entry — this is the only place the runner
        creates one); return its block events (block-level events fire in
        both modes)."""
        parts = adapter.message_to_parts(message)
        entry = self._append(
            lambda entry_id, parent_id, ts: AssistantMessage(
                id=entry_id,
                parent_id=parent_id,
                created_at=ts,
                parts=parts,
                llm_config=llm_cfg.model_copy(),
                stop_reason=finish_reason or "stop",
            )
        )
        self.ledger.record_usage(entry.id, **_to_usage_counters(message.usage))
        events: list[AgentEvent] = []
        for part in parts:
            if isinstance(part, ThinkingContent):
                events.append(
                    ReasoningBlock(
                        text=part.thinking,
                        redacted=part.redacted,
                    ),
                )
            elif isinstance(part, TextContent):
                events.append(TextBlock(text=part.text))
        events.append(FinishReason(finish_reason=finish_reason))
        return events

    async def _create_executions(
        self,
        message,
        token: CancellationToken,
    ) -> list[AgentEvent]:
        """Set-oriented birth: ask the registry for one draft per call in the
        assistant response (concurrently — each call gets a deep-copied
        `ToolCall`, so a draft can never alias the assistant message part),
        then eagerly persist them in model-request order, always before any
        decision. The registry owns the call-scoped facts (`raw_tool_call`,
        `tool_spec`, the birth `status` — PENDING or terminal-at-birth —
        `error`, `extras`); the runner re-stamps identity (`id`, `parent_id`,
        `created_at`), `ended_at` for a terminal birth, `context_tokens`
        (via `_append`), and `is_doom_loop_flagged`, and the ledger files the
        spec and stamps `tool_spec_id`. Failures are isolated per call: a
        raising `create_execution` (or a toolless runner) never breaks the
        set — the runner synthesizes the draft itself, FAILED for a raise and
        NOT_FOUND for the toolless case — preserving the invariant that every
        tool call produces exactly one tool output. Terminal births
        immediately run the outcome middleware pair.

        The cancellation race is PER CALL, inside `_birth_draft`, never around
        this gather: killing the gather would lose every draft and break
        one-output-per-call. A response with N tool calls yields N tool
        executions even when a cancellation lands mid-batch."""
        drafts = await asyncio.gather(*(self._birth_draft(tc, token) for tc in message.tool_calls))
        events: list[AgentEvent] = []
        for tc, (draft, exception) in zip(message.tool_calls, drafts, strict=False):
            # Doom-loop check runs before the append so it only sees
            # previously-appended executions; parallel tool calls are
            # evaluated in append order.
            doom_flagged = self._is_doom_loop(tc)

            def build(
                entry_id,
                parent_id,
                ts,
                _draft=draft,
                _d=doom_flagged,
            ) -> ToolExecution:
                return _draft.model_copy(
                    update={
                        "id": entry_id,
                        "parent_id": parent_id,
                        "created_at": ts,
                        "ended_at": (ts if _draft.status != ExecutionStatus.PENDING else None),
                        "is_doom_loop_flagged": _d,
                    },
                )

            execution = self._append(build)
            events.append(
                ToolCallReceived(
                    tool_call_id=execution.tool_call_id,
                    execution=execution.model_copy(deep=True),
                )
            )
            if execution.status != ExecutionStatus.PENDING:  # terminal birth
                _, event = self._finalize_undispatched(execution, exception)
                events.append(event)
        return events

    async def _birth_draft(
        self,
        tc,
        token: CancellationToken,
    ) -> tuple[ToolExecution, Exception | None]:
        """One call's guarded birth: delegate to
        `tool_registry.create_execution` with a deep-copied `ToolCall`, raced
        against the run's cancellation token. The draft comes back with no
        identity (`id` / `created_at` are `None`) for `_create_executions` to
        stamp. A raise is caught and becomes a runner-synthesized FAILED draft
        (the live exception is returned for the outcome middleware); a
        toolless runner synthesizes NOT_FOUND.

        A LOST RACE is not a failure: it synthesizes a plain PENDING draft so
        the call still gets its one execution, and the loop-top wind-down
        records it CANCELLED. A cancelled birth is CANCELLED, never FAILED —
        a cancellation is not a tool failure. (No `except CancelledError`
        clause is needed for that: `asyncio.CancelledError` derives from
        `BaseException`, so the broad `except Exception` below never sees it,
        and the race helper absorbs the kill it issued and reports the outcome
        as a boolean rather than re-raising.)"""
        raw = ToolCall(
            id=tc.id,
            name=tc.name,
            arguments=copy.deepcopy(tc.arguments),
        )
        if self.tool_registry is None:
            exc: Exception = ToolNotFound(f"Unknown tool: {tc.name!r}.")
            draft = ToolExecution(
                tool_call_id=raw.id,
                raw_tool_call=raw,
                status=ExecutionStatus.NOT_FOUND,
            )
            draft.error = self.to_tool_execution_error(
                draft,
                exc,
                phase="create_execution",
            )
            return draft, exc
        task = asyncio.ensure_future(
            self.tool_registry.create_execution(self.session, raw),
        )
        try:
            completed, draft, _ = await _race_cancellation(task, token, 0, None)
        except Exception as exc:
            failed = ToolExecution(
                tool_call_id=raw.id,
                raw_tool_call=raw,
                status=ExecutionStatus.FAILED,
            )
            failed.error = self.to_tool_execution_error(
                failed,
                exc,
                phase="create_execution",
            )
            return failed, exc
        if not completed:
            return ToolExecution(
                tool_call_id=raw.id,
                raw_tool_call=raw,
                status=ExecutionStatus.PENDING,
            ), None
        return draft, None

    async def _decide_one(
        self,
        execution: ToolExecution,
        token: CancellationToken,
    ) -> tuple[ToolExecution, ApprovalDecision] | None:
        """`_decide_with_middleware` under the cancellation race — `None` when
        the token won, and then the caller records NOTHING for this execution.

        The race wraps the whole middleware pair, not `registry.decide` alone,
        so a token already tripped when this is entered fires no hook at all —
        the inner task is killed before it ever runs. (In `_drive` the loop-top
        cancel check normally reaches a parked cancellation first; this is what
        makes the guarantee hold anyway, and it is why the race goes around the
        pair rather than inside it.)

        A token tripping DURING `decide` is the reachable case, and it still
        leaves `before_permission_check` fired with its returned execution
        discarded: the execution has to stay PENDING for the wind-down, so
        there is nowhere to put it, and a decision that never happened has
        nothing to apply. In particular `after_permission_decision` must not
        fire. Siblings decided in the same batch have all been scheduled by
        then, so their hooks fire too — the guarantee is about entry, not about
        interrupting a batch already under way."""
        task = asyncio.ensure_future(self._decide_with_middleware(execution))
        completed, pair, _ = await _race_cancellation(task, token, 0, None)
        return pair if completed else None

    async def _decide_with_middleware(
        self,
        execution: ToolExecution,
    ) -> tuple[ToolExecution, ApprovalDecision]:
        """Apply `before_permission_check` middleware, call the registry's
        `decide()`, then apply `after_permission_decision` middleware.
        Returns `(modified_execution, decision)` — the modified execution is
        what the registry saw AND the execution the decision is applied to
        and persisted (its changes are not restricted to the decide call).
        A toolless runner allows — the prepare step then produces the honest
        NOT_FOUND terminal rather than recording a false REJECTED."""
        modified = self._run_middlewares("before_permission_check", execution)
        if self.tool_registry is None:
            decision = ApprovalDecision(
                decision=ApprovalOption.ALLOW,
                created_at=self.now_ms(),
            )
        else:
            decision = await self.tool_registry.decide(self.session, modified)
        return modified, self._run_middlewares(
            "after_permission_decision",
            decision,
            modified,
        )

    async def _dispatch_batch(
        self,
        ready: list[ToolExecution],
        token: CancellationToken,
    ) -> AsyncIterator[AgentEvent]:
        """Dispatch every ready (PENDING + ALLOWED) execution with the
        implementation-chosen sequential scheduler — any scheduler conforms
        as long as every execution stays independent: one call's return,
        raise, or timeout never assigns an outcome to a sibling, and each
        call has its own deadline.

        A cancellation observed before an execution's turn comes up stops the
        batch here: this execution and every one after it are untouched —
        still PENDING, no middleware fired — and the loop-top wind-down
        terminalizes them. Only the executions the batch actually reached are
        the dispatch path's to finish."""
        for execution in ready:
            if token.cancelled:
                return
            async for event in self._dispatch_one(execution, token):
                yield event

    async def _dispatch_one(
        self,
        execution: ToolExecution,
        token: CancellationToken,
    ) -> AsyncIterator[AgentEvent]:
        """Prepare, then run, for one allowed execution.

        1. `before_tool_execution` fires — its returned `raw_tool_call` is the
           effective call, which is why the hook stays AHEAD of `prepare()`.
        2. `prepare()` runs, raced against the token.
        3. A raise, or a return that is not callable, terminalizes the
           execution WITHOUT it ever being marked RUNNING: `started_at` stays
           None, `dispatched` stays False, no `ToolExecutionStarted` is
           emitted, and `details["phase"]` is `"prepare"`.
        4. A cancellation observed at any point up to and including
           `prepare()` settling means the body is NOT dispatched — even when
           `prepare()` returned successfully. The grace window exists to let
           in-flight work finish, not to start new work after a cancellation
           was requested.
        5. Otherwise RUNNING + `started_at` are persisted, the birth
           `tool_spec` standing (there is NO dispatch-time re-snapshot),
           `ToolExecutionStarted` is emitted, and the callable is invoked
           under the cancellation race and the deadline.

        Every path from step 3 on finalizes through `_finalize_outcome`, NEVER
        `_finalize_undispatched`: `before_tool_execution` has already fired for
        this call and the undispatched pipeline would fire it a second time.
        The hook is the boundary — an execution whose hook has not fired
        belongs to the loop-top wind-down, one whose hook has fired belongs
        here."""
        execution = self._run_middlewares("before_tool_execution", execution)

        prepare_task = asyncio.ensure_future(self._prepare_tool(execution))
        try:
            prepared_ok, prepared, _ = await _race_cancellation(
                prepare_task,
                token,
                0,
                None,
            )
        except Exception as exc:
            _, event = self._finalize_outcome(
                self._terminal_for_prepare_failure(execution, exc),
                exc,
            )
            yield event
            return
        if not prepared_ok:
            _, event = self._finalize_outcome(self._cancelled_in_place(execution))
            yield event
            return
        if not callable(prepared):
            # A registry that returned None, a plain value, or anything else
            # that cannot be invoked. The runner synthesizes the failure
            # rather than letting it blow up later: nothing has been
            # dispatched, so this is a preparation failure and must record
            # like one. (A bare coroutine also lands here — and, being
            # un-awaited, additionally warns; §5.2 is why the contract asks
            # for a callable.)
            exc = AgentError(
                f"prepare() for tool {execution.raw_tool_call.name!r} returned "
                f"{type(prepared).__name__}, which is not callable."
            )
            _, event = self._finalize_outcome(
                self._terminal_for_prepare_failure(execution, exc),
                exc,
            )
            yield event
            return
        if token.cancelled:
            _, event = self._finalize_outcome(self._cancelled_in_place(execution))
            yield event
            return

        execution = self._persist_execution(
            execution,
            status=ExecutionStatus.RUNNING,
            started_at=self.now_ms(),
        )
        yield ToolExecutionStarted(
            tool_call_id=execution.tool_call_id,
            execution=execution.model_copy(deep=True),
        )
        terminal, exception = await self._run_tool_body(execution, prepared, token)
        _, event = self._finalize_outcome(terminal, exception)
        yield event

    def _terminal_for_prepare_failure(
        self,
        execution: ToolExecution,
        exception: Exception,
    ) -> ToolExecution:
        """The terminal (not yet persisted) execution for a `prepare()` that
        raised or returned a non-callable.

        The exception-type-to-status mapping did not disappear with `execute`
        — it MOVED here, where it is accurate, because the only work done at
        this point is resolution and validation. Once the callable has been
        invoked every raise is FAILED."""
        if isinstance(exception, ToolNotFound):
            status = ExecutionStatus.NOT_FOUND
        elif isinstance(exception, (InvalidToolArguments, ValidationError)):
            status = ExecutionStatus.INVALID
        else:
            status = ExecutionStatus.FAILED
        terminal = execution.model_copy(
            update={
                "status": status,
                "ended_at": self.now_ms(),
            },
        )
        terminal.error = self.to_tool_execution_error(
            terminal,
            exception,
            phase="prepare",
        )
        return terminal

    def _cancelled_in_place(self, execution: ToolExecution) -> ToolExecution:
        """The terminal execution for a cancellation during (or right after)
        `prepare()` — the one asymmetry in the cancellation rules.

        Everywhere else a cancelled registry phase leaves the execution
        PENDING for the loop-top wind-down. Here it cannot:
        `before_tool_execution` has already fired, and the wind-down would
        fire it again. The durable shape is identical to a wind-down
        cancellation — resultless, errorless, `cancel_signalled_at` and
        `ended_at` stamped, `started_at` unset, `dispatched` False."""
        ts = self.now_ms()
        return execution.model_copy(
            update={
                "cancel_signalled_at": ts,
                "status": ExecutionStatus.CANCELLED,
                "result": None,
                "error": None,
                "ended_at": ts,
            },
        )

    async def _prepare_tool(self, execution: ToolExecution) -> PreparedTool:
        """The single `registry.prepare` call site. A toolless runner raises
        `ToolNotFound` so a loaded ready execution still terminalizes
        honestly (NOT_FOUND) instead of crashing the run."""
        if self.tool_registry is None:
            raise ToolNotFound(f"Unknown tool: {execution.raw_tool_call.name!r}.")
        return await self.tool_registry.prepare(self.session, execution)

    async def _run_tool_body(
        self,
        execution: ToolExecution,
        prepared: PreparedTool,
        token: CancellationToken,
    ) -> tuple[ToolExecution, Exception | None]:
        """Invoke the prepared callable under the cancellation race and the
        outside deadline; return the terminal (not yet persisted) execution
        and the live exception, if one exists. Outcomes: the body *returned*
        (even early, cooperatively, within the cancel grace) → COMPLETED with
        its real result, whatever `is_error` says; it *raised* → FAILED for
        EVERY exception type, because resolution and validation already
        happened in `prepare()` and a body that raises `ToolNotFound` looking
        up a sub-resource is a tool failure, not a resolution failure; the
        deadline expired → hard-cancelled, TIMED_OUT; the cancel grace
        expired → hard-cancelled, INTERRUPTED. When run cancellation is
        signalled while the body is in flight, the RUNNING execution is
        persisted with `cancel_signalled_at` BEFORE the grace window runs, so
        the signal is durable whatever settles the call. The deadline is
        outside enforcement only — the birth `tool_spec.timeout_in_ms` beats
        `RuntimeConfig.tool_execution_timeout_in_ms`; it never touches the
        shared token (one call's deadline must not cancel siblings) and does
        not populate `cancel_signalled_at`.

        The INVOCATION sits inside the failure handling. A callable that
        returns a plain value rather than an awaitable makes
        `asyncio.ensure_future` raise `TypeError` synchronously, and that
        callable has ALREADY been invoked — so the honest record is a
        post-dispatch failure like any other, not a crashed run. Testing
        awaitability any earlier would mean invoking the body before RUNNING
        is durable, which is exactly what the prepare split exists to
        prevent."""
        config = self.session.session_config.runtime_config
        grace_ms = config.tool_cancellation_grace_period
        spec_timeout = execution.tool_spec.timeout_in_ms if execution.tool_spec is not None else None
        deadline_ms = spec_timeout if spec_timeout is not None else config.tool_execution_timeout_in_ms
        current = execution

        def note_cancel_signalled() -> None:
            nonlocal current
            current = self._persist_execution(
                current,
                cancel_signalled_at=self.now_ms(),
            )

        try:
            tool_task = asyncio.ensure_future(prepared(cancellation_token=token))
            if deadline_ms == Inf:
                completed, result, _ = await _race_cancellation(
                    tool_task,
                    token,
                    grace_ms,
                    None,
                    detach=True,
                    on_cancel_signalled=note_cancel_signalled,
                )
            else:
                try:
                    async with asyncio.timeout(deadline_ms / 1000.0) as scope:
                        completed, result, _ = await _race_cancellation(
                            tool_task,
                            token,
                            grace_ms,
                            None,
                            detach=True,
                            on_cancel_signalled=note_cancel_signalled,
                        )
                except TimeoutError:
                    if not scope.expired():
                        raise  # the tool's own TimeoutError — a normal raise
                    await _kill(tool_task, detach=True)  # idempotent backstop
                    return current.model_copy(
                        update={
                            "status": ExecutionStatus.TIMED_OUT,
                            "ended_at": self.now_ms(),
                        },
                    ), None
        except Exception as exc:
            terminal = current.model_copy(
                update={
                    "status": ExecutionStatus.FAILED,
                    "ended_at": self.now_ms(),
                },
            )
            terminal.error = self.to_tool_execution_error(
                terminal,
                exc,
                phase="execution",
            )
            return terminal, exc
        if not completed:
            return current.model_copy(
                update={
                    "status": ExecutionStatus.INTERRUPTED,
                    "ended_at": self.now_ms(),
                },
            ), None
        # The returned result passes through the context manager BEFORE the
        # terminal execution is constructed (and thus before any middleware):
        # what persists, projects, and feeds the ToolExecuted event is the
        # processed output. The execution it sees is IN TRANSITION — still
        # RUNNING, no result attached — and is there to be read for identity.
        return current.model_copy(
            update={
                "status": ExecutionStatus.COMPLETED,
                "result": self.context_manager.process_tool_output(
                    self.session,
                    current,
                    result,
                ),
                "ended_at": self.now_ms(),
            },
        ), None

    def _is_doom_loop(self, tc) -> bool:
        """True if the current tool call would be the Nth consecutive identical
        call (same name + arguments, compared on `raw_tool_call`) in the open
        turn (where N = doom_loop_threshold). Checks the already-appended
        ToolExecution entries, so parallel tool calls are evaluated in append
        order."""
        threshold = self.session.session_config.runtime_config.doom_loop_threshold
        if threshold <= 0:
            return False
        lookback = threshold - 1
        current_turn_executions = self.ledger.open_turn_executions()
        subset = current_turn_executions[-lookback:]
        if len(subset) != lookback:
            return False
        return all(te.raw_tool_call.name == tc.name and te.raw_tool_call.arguments == tc.arguments for te in subset)

    def _close_turn(self, outcome: TurnOutcome, error: str | None = None) -> None:
        """The only TurnFinish writer that APPENDS — normal close, cancel
        wind-down, failure close, and every non-transition compaction ending
        land here. The status re-derives from the entries (IDLE for
        COMPLETED/CANCELLED; retry-ready PENDING for a failure), and the
        outcome is recorded for `RunResult`. No usage rollup: turn usage is
        derived from `AgentSession.usages`, never duplicated on the marker.

        (A committed compaction is the one close that does NOT come through
        here: its marker belongs to the outgoing conversation and is written
        inside the transition — see `_commit`.)"""
        self._append(
            lambda entry_id, parent_id, ts: TurnFinish(
                id=entry_id,
                parent_id=parent_id,
                created_at=ts,
                outcome=outcome,
                error=error,
            )
        )
        self._closed_outcome = outcome
        self._refresh_status()

    def _set_status(self, status: ConversationStatus) -> None:
        self.session.active_conversation.status = status


# ── helpers ─────────────────────────────────────────────────────────────────


class _CompactionEnded(Exception):
    """Internal: the compaction ended before `compact()` returned — the run's
    cancellation token fired and its grace expired, or the wall-clock deadline
    did. Carries the outcome the bracket should close with; never escapes the
    step (a user-scheduled timeout re-raises as a `ClientTimeoutError`, the
    same type the conversational LLM call's timeout produces)."""

    def __init__(self, outcome: TurnOutcome, error: str | None = None) -> None:
        super().__init__(error or outcome.value)
        self.outcome = outcome
        self.error = error


def _outcome_for(exception: Exception) -> TurnOutcome:
    """How a compaction bracket closes for a live failure — the same
    TIMED_OUT / ERRORED split the conversational LLM call uses."""
    return TurnOutcome.TIMED_OUT if isinstance(exception, ClientTimeoutError) else TurnOutcome.ERRORED


def _equivalent(a: object, b: object) -> bool:
    """Collaborator equivalence for runner equality: the same object, or two
    instances of the same class with equal instance state. Covers the plain
    classes used for tools / strategies / middleware, which rarely define
    `__eq__`; objects without a `__dict__` fall back to `==`."""
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    state_a = getattr(a, "__dict__", None)
    state_b = getattr(b, "__dict__", None)
    if state_a is None or state_b is None:
        return a == b
    return state_a == state_b


def _all_equivalent(xs: list, ys: list) -> bool:
    return len(xs) == len(ys) and all(_equivalent(x, y) for x, y in zip(xs, ys, strict=False))


async def _race_cancellation(
    task: asyncio.Task,
    token: CancellationToken,
    grace_ms: int,
    grace_deadline: float | None,
    *,
    detach: bool = False,
    on_cancel_signalled: Callable[[], None] | None = None,
):
    """Await `task` racing the run's cancellation token, honoring a grace
    window. Returns `(completed, value, grace_deadline)`:

    - the task finished (before the token, or within grace) →
      `(True, value, ...)`; its exception (including `StopAsyncIteration`
      from a stream step) propagates instead.
    - the token fired and grace expired → the task is killed and
      `(False, None, ...)` returns.

    `grace_deadline` threads the window across calls (a streaming cancel's
    grace spans the REST of the stream, not each chunk). An external
    `asyncio.CancelledError` re-raises with the task killed — no orphans.
    `detach=True` (tools) gives a kill two ticks to land, then detaches with a
    result-swallowing callback (thread-backed work can't be interrupted);
    `detach=False` (LLM steps) awaits the teardown so the wire is closed
    before control returns. `on_cancel_signalled` fires once, as soon as the
    token is observed to have fired while the task was in flight — BEFORE any
    grace handling, and also when the task settled in the same tick (a
    cooperative early return must still record that cancellation reached it).
    The tool path persists `cancel_signalled_at` there."""
    if not token.cancelled:
        waiter = asyncio.ensure_future(token.wait_cancelled())
        try:
            done, _ = await asyncio.wait(
                {task, waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except BaseException:
            await _kill(task, detach)
            await _cancel_quietly(waiter)
            raise
        await _cancel_quietly(waiter)
        if task in done:
            if token.cancelled and on_cancel_signalled is not None:
                on_cancel_signalled()
            return True, task.result(), grace_deadline
    elif task.done():  # finished in the same tick the token fired — it wins
        if on_cancel_signalled is not None:
            on_cancel_signalled()
        return True, task.result(), grace_deadline
    if on_cancel_signalled is not None:
        on_cancel_signalled()
    if grace_ms == 0:
        await _kill(task, detach)
        return False, None, grace_deadline
    if grace_deadline is None:
        grace_deadline = float("inf") if grace_ms == Inf else asyncio.get_running_loop().time() + grace_ms / 1000.0
    remaining = grace_deadline - asyncio.get_running_loop().time()
    try:
        if remaining == float("inf"):
            value = await asyncio.shield(task)
        else:
            value = await asyncio.wait_for(
                asyncio.shield(task),
                max(remaining, 0.0),
            )
        return True, value, grace_deadline
    except TimeoutError:
        if task.done() and not task.cancelled():
            raise  # the task's OWN TimeoutError — its result, not grace expiry
        await _kill(task, detach)
        return False, None, grace_deadline
    except BaseException:
        if not task.done():  # external cancel mid-grace — no orphans
            await _kill(task, detach)
        raise


async def _kill(task: asyncio.Task, detach: bool) -> None:
    """Hard-cancel `task` leak-free. `detach=False`: await the teardown out.
    `detach=True`: give the cancel two ticks; a task that ignores it (detached
    thread work) is left to finish on its own, its outcome swallowed by
    callback — the warnings-as-errors suite would flag a true leak."""
    if not detach:
        await _cancel_quietly(task)
        return
    task.cancel()
    for _ in range(2):
        if task.done():
            break
        await asyncio.sleep(0)
    if task.done():
        with contextlib.suppress(BaseException):
            task.result()
    else:
        task.add_done_callback(_swallow_result)


async def _cancel_quietly(task: asyncio.Task) -> None:
    task.cancel()
    with contextlib.suppress(BaseException):
        await task


def _swallow_result(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()  # retrieved — no unretrieved-exception noise


_POST_PARTS = TypeAdapter(list[ContentPart])


def _normalize_post_parts(content: str | list[ContentPart]) -> list[ContentPart]:
    """`post_message` input → the part list persisted on the `UserMessage`.

    Shape is checked against `ContentPart` itself, so a new part type needs no
    change here; a part the union does not admit raises `ValidationError`."""
    if not content:
        raise AgentError("post_message requires a non-empty string or list of content parts.")
    if isinstance(content, str):
        return [TextContent(text=content)]
    if isinstance(content, BaseModel):
        raise AgentError(f"post_message takes a list of content parts; wrap the {type(content).__name__} in a list.")
    return _POST_PARTS.validate_python(content)


def _to_delta_event(event) -> AgentEvent | None:
    """Translate one client stream event into its agent delta/`*Start` mirror
    (None for raw/usage events with no agent-level equivalent)."""
    if event.type == "text_start":
        return TextStart()
    if event.type == "text_delta":
        return TextDelta(text=event.delta)
    if event.type == "thinking_start":
        return ReasoningStart()
    if event.type == "thinking_delta":
        return ReasoningDelta(text=event.delta)
    if event.type == "tool_call_start":
        return ToolCallStart(tool_call_id=event.id, name=event.name)
    return None


_APPROVAL_STATUS = {
    ApprovalOption.ALLOW: ApprovalStatus.ALLOWED,
    ApprovalOption.DENY: ApprovalStatus.REJECTED,
    ApprovalOption.PENDING: ApprovalStatus.PENDING,
}


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _ms_to_seconds(ms: int) -> float | None:
    """RuntimeConfig duration → the float-seconds kwarg (Inf → not passed)."""
    return None if ms == Inf else ms / 1000.0


def _to_usage_counters(usage) -> dict[str, int]:
    """Client usage → the counter kwargs for `SessionLedger.record_usage()`
    (which owns building the id-carrying `Usage` record)."""
    if usage is None:
        return {}
    return {
        "input": usage.input_tokens or 0,
        "output": usage.output_tokens or 0,
        "cache_read": usage.cached_input_tokens or 0,
        "cache_write": usage.cache_write_tokens or 0,
        "total_tokens": usage.total_tokens or 0,
    }
