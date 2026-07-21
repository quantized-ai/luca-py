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

The whole tool lifecycle is delegated to the `ToolRegistry` the runner is
constructed with (`tool_registry.py`; `None` = toolless agent). The runner
touches tools through exactly four registry methods: `get_tools` (queried
fresh per LLM call), `create_execution` (the birth draft — the runner stamps
identity and appends), `decide` (approval), and `execute` (the body). The
loop has exactly ONE decide() call site — its top: "any undecided
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
from .context import CancellationToken, ToolContext
from .context_manager import ContextManager
from .models import (
    AgentSession,
    AnyEntry,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    CancelRequested,
    Conversation,
    ConversationStatus,
    ExecutionResult,
    ExecutionStatus,
    Inf,
    LLMConfig,
    RuntimeConfig,
    SessionConfig,
    SessionRuntimeStatus,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    UserMessage,
    ContentPart,
)
from .events import (
    AgentEvent,
    ApprovalRequired,
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
from .projection import ConversationProjector, tool_message_text
from .system_prompt import (
    DefaultSystemPromptAssembler,
    SystemPromptAssembler,
    SystemPromptPartInput,
    coerce_system_prompt_part,
)
from .tool_registry import ToolRegistry

EventCallback = Callable[[AgentEvent], "None | Awaitable[None]"]


class RunResult(BaseModel):
    """Where a run stopped.

    - Turn completed → `status=IDLE`, `outcome` from the closing TurnFinish
      (COMPLETED, or CANCELLED for a wind-down).
    - Approval pause → `status=AWAITING_APPROVAL`, `outcome=None`,
      `pending_approvals` non-empty.

    Carries no usage: provider consumption lives in
    `AgentSession.usages[conversation_id][entry_id]`, one record per
    assistant entry — aggregate as needed.

    A timeout / LLM failure does NOT produce a result: the turn is closed
    (`TurnFinish(TIMED_OUT | ERRORED)`) and the exception re-raises through
    `await` / iteration — `run.result` stays None on the raise path. With a
    cancel pending, the wind-down consumes the failure instead and the run
    returns normally (outcome from the `CancelRequested`)."""

    model_config = ConfigDict(extra="forbid")

    status: ConversationStatus  # IDLE | AWAITING_APPROVAL
    outcome: TurnOutcome | None  # set iff the bracket closed during this run
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
        self._context: ToolContext | None = None
        self._finished = False  # the engine produced its last event
        self._entered = False
        self._exited = False
        if eager:
            # Validates state synchronously at call time and spawns the
            # background task. The loop is resolved FIRST so a sync-context
            # start() fails before taking the one-engine guard; the bracket
            # opens durably at call time so an immediate cancel() has an open
            # turn to attach to (the first drive is then the flush).
            loop = asyncio.get_running_loop()
            self._runner._begin_run(self)
            self._runner._ensure_open_turn()
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
                "iterate inside 'async with' (e.g. async with runner.run() as "
                "run: async for event in run: ...)"
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
            raise AgentError(
                "this run was suspended and finalized; resume with a fresh "
                "runner.run()"
            )
        if self._finished:
            raise StopAsyncIteration
        if self._engine is None:
            self._runner._begin_run(self)  # raises on IDLE / concurrent run
            self._engine = self._runner._drive(
                streaming=self._streaming, context=self._context,
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
            streaming=self._streaming, context=self._context,
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
                try:
                    await self._task
                except BaseException:
                    pass
            raise

    async def _finalize_eager(self, swallow_failure: bool) -> None:
        try:
            await self._task
        except asyncio.CancelledError:
            if not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except BaseException:
                    pass
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
                nodes=[], created_at=ts, updated_at=ts,
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
        middleware: list | None = None,
    ) -> None:
        self.session = session
        self.tool_registry = tool_registry
        # A single projector OBJECT (never a class to instantiate, never
        # stacked by plugins): lives on the runner, is never serialized, and
        # is invoked fresh whenever messages are prepared for an LLM call.
        self.conversation_projector = (
            conversation_projector or ConversationProjector()
        )
        # The context-accounting strategy (context_manager.py): calculates
        # every new entry's `context_tokens`, processes returned tool output,
        # and builds pruned replacements. Same collaborator pattern as the
        # projector — one object, never serialized, defaults to the simple
        # built-in policy.
        self.context_manager = context_manager or ContextManager()
        # Static parts (str / dict / SystemPromptPart) coerce eagerly — a bad
        # part fails at construction, not mid-turn. Callables resolve per call.
        self.system_prompt_parts = [
            part if callable(part) else coerce_system_prompt_part(part)
            for part in (system_prompt_parts or [])
        ]
        self.system_prompt_assembler = (
            system_prompt_assembler or DefaultSystemPromptAssembler()
        )
        self.provider = provider
        self.middleware = list(middleware or [])
        self.ledger = SessionLedger(session, self.now_ms, self.generate_id)
        self._active_run: AgentRun | None = None  # first-drive → finalization

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
        tool registry, prompt parts, assembler, provider, and middleware.
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
                self.system_prompt_assembler, other.system_prompt_assembler,
            )
            and _equivalent(
                self.conversation_projector, other.conversation_projector,
            )
            and _equivalent(self.context_manager, other.context_manager)
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
        resumable bracket — always rejects.

        `content` is a bare string (the common case) or an ordered list of
        parts mixing text and images; `before_post_message` sees that list and
        returns the one that is persisted."""
        if (
            self.status not in (ConversationStatus.IDLE, ConversationStatus.PENDING)
            or self.ledger.open_turn_index() is not None
        ):
            raise AgentError(
                f"post_message requires a closed turn and IDLE/PENDING status "
                f"(status={self.status.value})."
            )
        parts = self._run_middlewares(
            "before_post_message", _normalize_post_parts(content),
        )
        message = self._append(
            lambda entry_id, parent_id, ts: UserMessage(
                id=entry_id, parent_id=parent_id, created_at=ts, parts=parts,
            )
        )
        self._set_status(ConversationStatus.PENDING)
        return message.id

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
                "a cancellation is already pending for the open turn; the "
                "first request's outcome/error stand"
            )
        self._append(
            lambda entry_id, parent_id, ts: CancelRequested(
                id=entry_id, parent_id=parent_id, created_at=ts,
                outcome=outcome, error=error,
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
        run's own CancellationToken + ToolContext."""
        if self._active_run is not None:
            raise AgentError(
                "another run is already active on this runner; finish or "
                "finalize it first"
            )
        if self.idle():
            raise AgentError("Nothing to run; call post_message() first.")
        run._token = CancellationToken()
        run._context = ToolContext(
            session_id=self.session.id,
            model=self.session.session_config.llm_config,
        )
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
                    id=entry_id, parent_id=parent_id, created_at=ts,
                )
            )

    def _build_run_result(self) -> RunResult:
        """Snapshot where the engine stopped. IDLE → the bracket closed this
        run (the trailing TurnFinish carries the outcome);
        AWAITING_APPROVAL → an approval pause (outcome None)."""
        if self.status == ConversationStatus.AWAITING_APPROVAL:
            return RunResult(
                status=ConversationStatus.AWAITING_APPROVAL,
                outcome=None,
                pending_approvals=self.pending_approvals(),
            )
        nodes = self.session.active_conversation.nodes
        finish = self.session.entries[nodes[-1]]
        return RunResult(
            status=ConversationStatus.IDLE,
            outcome=finish.outcome,
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

    def _append(self, build_fn) -> AnyEntry:
        """Append one entry: build it, calculate its `context_tokens`
        (context calculation is part of preparing a complete entry — it runs
        BEFORE middleware, and never again after), run it through
        `before_entry_written` middleware, then commit to the ledger."""
        def wrapped(entry_id: str, parent_id: str | None, ts: int) -> AnyEntry:
            entry = build_fn(entry_id, parent_id, ts)
            entry.context_tokens = self.context_manager.calculate_context(entry)
            return self._run_middlewares("before_entry_written", entry)
        return self.ledger.append(wrapped)

    def _persist_execution(
        self, execution: ToolExecution, **changes,
    ) -> ToolExecution:
        """The one update door for a `ToolExecution`: build the replacement,
        stamp `updated_at`, thread `before_entry_written`, store. Every
        persistence — approval updates, the RUNNING transition, cancellation
        stamps, terminal outcomes — lands here."""
        updated = execution.model_copy(
            update={**changes, "updated_at": self.now_ms()},
        )
        updated = self._run_middlewares("before_entry_written", updated)
        return self.ledger.put_execution(updated)

    # ── tool-execution outcome machinery ─────────────────────────────────────

    def to_tool_execution_error(
        self,
        execution: ToolExecution,
        exception: Exception,
    ) -> ToolExecutionError:
        """Convert a live exception into the durable `ToolExecutionError`.
        Override to redact secrets, preserve domain codes, or add a traceback
        to `details` — the live exception itself is never persisted.

        The default keeps the exception's type and message, nests structured
        validation errors under `details["errors"]`, and records the failure
        phase for registry/tool-owned raises (`create_execution` before
        dispatch, `execution` after — derived from `started_at`)."""
        if isinstance(exception, InvalidToolArguments):
            details: dict = {"errors": exception.errors}
        elif isinstance(exception, ToolNotFound):
            details = {}
        elif isinstance(exception, ValidationError):
            details = {"errors": json.loads(exception.json(include_url=False))}
        else:
            details = {
                "phase": (
                    "execution"
                    if execution.started_at is not None
                    else "create_execution"
                ),
            }
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
            execution, self.session.entries,
        )
        return ToolExecuted(
            tool_call_id=execution.tool_call_id,
            execution=execution.model_copy(deep=True),
            result_text=tool_message_text(message),
            is_error=message.is_error,
        )

    def _finalize_outcome(
        self, execution: ToolExecution, exception: Exception | None = None,
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
                    execution,
                ),
            },
        )
        execution = self._run_middlewares(
            "after_tool_execution", execution, exception,
        )
        execution = self._persist_execution(execution)
        return execution, self._tool_executed_event(execution)

    def _finalize_undispatched(
        self, execution: ToolExecution, exception: Exception | None = None,
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

    def build_tool_list(self) -> list[LucaTool]:
        """Return the wire tool list for this LLM call: query the registry
        fresh (`get_tools` is dynamic — the result may vary with session
        state; a toolless runner contributes none), convert each tool via the
        adapter, and thread the list through any `build_tool_list`
        middleware. Called per LLM invocation."""
        tools = (
            self.tool_registry.get_tools(self.session)
            if self.tool_registry is not None
            else []
        )
        tool_list = [adapter.tool_to_luca_tool(tool) for tool in tools]
        return self._run_middlewares("build_tool_list", tool_list)

    def build_messages(self) -> list:
        """Project the active conversation to canonical client messages via
        the configured `ConversationProjector` — derived per call, never
        stored. History-shaping policy belongs on the projector itself (there
        is no projection middleware); `before_llm_call` remains downstream for
        last-mile request changes."""
        return self.conversation_projector.project(
            self.session.active_conversation, self.session.entries,
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
            "before_llm_call", (messages, system_message), unpack_values=True,
        )

    # ── the engine ───────────────────────────────────────────────────────────

    async def _drive(
        self, streaming: bool, context: ToolContext, token: CancellationToken,
    ) -> AsyncIterator[AgentEvent]:
        """The single engine behind both methods; `AgentRun` is its only
        consumer (lazy pulls it directly, eager drains it from the background
        task)."""
        # Resume the open turn if one exists; otherwise open a new bracket
        # (an eager run already opened it at start() time).
        self._ensure_open_turn()
        self._set_status(ConversationStatus.RUNNING)

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
                pairs = await asyncio.gather(
                    *(
                        self._decide_with_middleware(ex, context)
                        for ex in undecided
                    )
                )
                denial_events: list[AgentEvent] = []
                for modified, decision in pairs:
                    denied = decision.decision == ApprovalOption.DENY
                    changes: dict = {
                        "approval_decisions": [
                            *modified.approval_decisions, decision,
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
                async for event in self._dispatch_batch(ready, context, token):
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
                    executions=[
                        ex.model_copy(deep=True)
                        for ex in self.pending_approvals()
                    ],
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
            if (
                config.limit_tool_choice_on_doom_loop_flagged
                and self.ledger.open_turn_has_doom_loop_flagged()
            ):
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
            tool_list = self.build_tool_list()
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
                        reasoning_effort=llm_cfg.reasoning_effort,
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
                                    completed, stream_event, grace_deadline = (
                                        await _race_cancellation(
                                            step, token, grace_ms, grace_deadline,
                                        )
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
                                elif (
                                    delta := _to_delta_event(stream_event)
                                ) is not None:
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
                            reasoning_effort=llm_cfg.reasoning_effort,
                            provider=self.provider,
                            timeout=request_timeout,
                            total_timeout=total_timeout,
                        )
                    )
                    completed, response, _ = await _race_cancellation(
                        llm_task, token, grace_ms, None,
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
                outcome = (
                    TurnOutcome.TIMED_OUT
                    if isinstance(exc, ClientTimeoutError)
                    else TurnOutcome.ERRORED
                )
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
                events.extend(await self._create_executions(message, context))
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
        self, message, finish_reason, llm_cfg: LLMConfig,
    ) -> list[AgentEvent]:
        """Append the assistant message and write its provider-usage record
        to `AgentSession.usages` (usage is accessory conversation-entry data,
        never embedded in the entry — this is the only place the runner
        creates one); return its block events (block-level events fire in
        both modes)."""
        parts = adapter.message_to_parts(message)
        entry = self._append(
            lambda entry_id, parent_id, ts: AssistantMessage(
                id=entry_id, parent_id=parent_id, created_at=ts,
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
                        text=part.thinking, redacted=part.redacted,
                    ),
                )
            elif isinstance(part, TextContent):
                events.append(TextBlock(text=part.text))
        events.append(FinishReason(finish_reason=finish_reason))
        return events

    async def _create_executions(self, message, ctx: ToolContext) -> list[AgentEvent]:
        """Set-oriented birth: ask the registry for one draft per call in the
        assistant response (concurrently — each call gets a deep-copied
        `ToolCall`, so a draft can never alias the assistant message part),
        then eagerly persist them in model-request order, always before any
        decision. The registry owns the call-scoped facts (`raw_tool_call`,
        `tool_spec`, the birth `status` — PENDING or terminal-at-birth —
        `error`, `extras`); the runner re-stamps identity (`id`, `parent_id`,
        `created_at`), `ended_at` for a terminal birth, `context_tokens`
        (via `_append`), and `is_doom_loop_flagged`. Failures are isolated
        per call: a raising `create_execution` (or a toolless runner) never
        breaks the set — the runner synthesizes the draft itself, FAILED for
        a raise and NOT_FOUND for the toolless case — preserving the
        invariant that every tool call produces exactly one tool output.
        Terminal births immediately run the outcome middleware pair."""
        drafts = await asyncio.gather(
            *(self._birth_draft(tc, ctx) for tc in message.tool_calls)
        )
        events: list[AgentEvent] = []
        for tc, (draft, exception) in zip(message.tool_calls, drafts):
            # Doom-loop check runs before the append so it only sees
            # previously-appended executions; parallel tool calls are
            # evaluated in append order.
            doom_flagged = self._is_doom_loop(tc)

            def build(
                entry_id, parent_id, ts, _draft=draft, _d=doom_flagged,
            ) -> ToolExecution:
                return _draft.model_copy(
                    update={
                        "id": entry_id,
                        "parent_id": parent_id,
                        "created_at": ts,
                        "ended_at": (
                            ts
                            if _draft.status != ExecutionStatus.PENDING
                            else None
                        ),
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
        self, tc, ctx: ToolContext,
    ) -> tuple[ToolExecution, Exception | None]:
        """One call's guarded birth: delegate to
        `tool_registry.create_execution` with a deep-copied `ToolCall`. The
        draft comes back with placeholder identity (`id=""`, `created_at=0`)
        for `_create_executions` to stamp. A raise is caught and becomes a
        runner-synthesized FAILED draft (the live exception is returned for
        the outcome middleware); a toolless runner synthesizes NOT_FOUND."""
        raw = ToolCall(
            id=tc.id, name=tc.name, arguments=copy.deepcopy(tc.arguments),
        )
        if self.tool_registry is None:
            exc: Exception = ToolNotFound(f"Unknown tool: {tc.name!r}.")
            draft = ToolExecution(
                id="", created_at=0,
                tool_call_id=raw.id, raw_tool_call=raw,
                status=ExecutionStatus.NOT_FOUND,
            )
            draft.error = self.to_tool_execution_error(draft, exc)
            return draft, exc
        try:
            return await self.tool_registry.create_execution(raw, ctx), None
        except Exception as exc:
            draft = ToolExecution(
                id="", created_at=0,
                tool_call_id=raw.id, raw_tool_call=raw,
                status=ExecutionStatus.FAILED,
            )
            draft.error = self.to_tool_execution_error(draft, exc)
            return draft, exc

    async def _decide_with_middleware(
        self, execution: ToolExecution, ctx: ToolContext,
    ) -> tuple[ToolExecution, ApprovalDecision]:
        """Apply `before_permission_check` middleware, call the registry's
        `decide()`, then apply `after_permission_decision` middleware.
        Returns `(modified_execution, decision)` — the modified execution is
        what the registry saw AND the execution the decision is applied to
        and persisted (its changes are not restricted to the decide call).
        A toolless runner allows — `execute` then produces the honest
        NOT_FOUND terminal rather than recording a false REJECTED."""
        modified = self._run_middlewares("before_permission_check", execution)
        if self.tool_registry is None:
            decision = ApprovalDecision(
                decision=ApprovalOption.ALLOW, created_at=self.now_ms(),
            )
        else:
            decision = await self.tool_registry.decide(modified, ctx)
        return modified, self._run_middlewares(
            "after_permission_decision", decision, modified,
        )

    async def _dispatch_batch(
        self, ready: list[ToolExecution], ctx: ToolContext,
        token: CancellationToken,
    ) -> AsyncIterator[AgentEvent]:
        """Dispatch every ready (PENDING + ALLOWED) execution with the
        implementation-chosen sequential scheduler — any scheduler conforms
        as long as every execution stays independent: one call's return,
        raise, or timeout never assigns an outcome to a sibling, and each
        call has its own deadline. A token tripped mid-batch finishes the
        in-flight call (per its grace) and skips the rest — the loop-top
        wind-down cancels them."""
        for execution in ready:
            if token.cancelled:
                return
            async for event in self._dispatch_one(execution, ctx, token):
                yield event

    async def _dispatch_one(
        self, execution: ToolExecution, ctx: ToolContext,
        token: CancellationToken,
    ) -> AsyncIterator[AgentEvent]:
        """Dispatch preparation + body invocation for one allowed execution:
        apply `before_tool_execution` (its returned `raw_tool_call` is the
        effective call), persist RUNNING with `started_at` (the birth
        `tool_spec` stands — there is NO dispatch-time re-snapshot), emit
        `ToolExecutionStarted`, and run the body via `registry.execute`.
        Resolution and validation live inside the registry, so their
        failures land AFTER `started_at` is set, with `ToolExecutionStarted`
        emitted — RUNNING must be durably persisted before the body runs."""
        execution = self._run_middlewares("before_tool_execution", execution)
        execution = self._persist_execution(
            execution,
            status=ExecutionStatus.RUNNING,
            started_at=self.now_ms(),
        )
        yield ToolExecutionStarted(
            tool_call_id=execution.tool_call_id,
            execution=execution.model_copy(deep=True),
        )
        terminal, exception = await self._run_tool_body(execution, ctx, token)
        _, event = self._finalize_outcome(terminal, exception)
        yield event

    async def _execute_body(
        self, execution: ToolExecution, ctx: ToolContext,
        token: CancellationToken,
    ) -> ExecutionResult:
        """The single `registry.execute` call site. A toolless runner raises
        `ToolNotFound` so a loaded ready execution still terminalizes
        honestly (NOT_FOUND) instead of crashing the run."""
        if self.tool_registry is None:
            raise ToolNotFound(
                f"Unknown tool: {execution.raw_tool_call.name!r}."
            )
        return await self.tool_registry.execute(
            execution, ctx, cancellation_token=token,
        )

    async def _run_tool_body(
        self,
        execution: ToolExecution,
        ctx: ToolContext,
        token: CancellationToken,
    ) -> tuple[ToolExecution, Exception | None]:
        """Invoke `registry.execute` under the cancellation race and the
        outside deadline; return the terminal (not yet persisted) execution
        and the live exception, if one exists. Outcomes: the body *returned*
        (even early, cooperatively, within the cancel grace) → COMPLETED with
        its real result, whatever `is_error` says; it *raised* → the contract
        mapping (`ToolNotFound` → NOT_FOUND, `InvalidToolArguments` /
        `ValidationError` → INVALID, anything else → FAILED) with a
        structured error; the deadline expired → hard-cancelled, TIMED_OUT;
        the cancel grace expired → hard-cancelled, INTERRUPTED. When run
        cancellation is signalled while the body is in flight, the RUNNING
        execution is persisted with `cancel_signalled_at` BEFORE the grace
        window runs, so the signal is durable whatever settles the call. The
        deadline is outside enforcement only — the birth
        `tool_spec.timeout_in_ms` beats
        `RuntimeConfig.tool_execution_timeout_in_ms`; it never touches the
        shared token (one call's deadline must not cancel siblings) and does
        not populate `cancel_signalled_at`."""
        config = self.session.session_config.runtime_config
        grace_ms = config.tool_cancellation_grace_period
        spec_timeout = (
            execution.tool_spec.timeout_in_ms
            if execution.tool_spec is not None
            else None
        )
        deadline_ms = (
            spec_timeout
            if spec_timeout is not None
            else config.tool_execution_timeout_in_ms
        )
        current = execution

        def note_cancel_signalled() -> None:
            nonlocal current
            current = self._persist_execution(
                current, cancel_signalled_at=self.now_ms(),
            )

        tool_task = asyncio.ensure_future(
            self._execute_body(current, ctx, token)
        )
        try:
            if deadline_ms == Inf:
                completed, result, _ = await _race_cancellation(
                    tool_task, token, grace_ms, None,
                    detach=True, on_cancel_signalled=note_cancel_signalled,
                )
            else:
                try:
                    async with asyncio.timeout(deadline_ms / 1000.0) as scope:
                        completed, result, _ = await _race_cancellation(
                            tool_task, token, grace_ms, None,
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
            if isinstance(exc, ToolNotFound):
                status = ExecutionStatus.NOT_FOUND
            elif isinstance(exc, (InvalidToolArguments, ValidationError)):
                status = ExecutionStatus.INVALID
            else:
                status = ExecutionStatus.FAILED
            terminal = current.model_copy(
                update={
                    "status": status,
                    "ended_at": self.now_ms(),
                },
            )
            terminal.error = self.to_tool_execution_error(terminal, exc)
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
        # processed output.
        return current.model_copy(
            update={
                "status": ExecutionStatus.COMPLETED,
                "result": self.context_manager.process_tool_output(result),
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
        return all(
            te.raw_tool_call.name == tc.name
            and te.raw_tool_call.arguments == tc.arguments
            for te in subset
        )

    def _close_turn(self, outcome: TurnOutcome, error: str | None = None) -> None:
        """The only TurnFinish writer — normal close, cancel wind-down, and
        failure close all land here. The status re-derives from the entries
        (IDLE for COMPLETED/CANCELLED; retry-ready PENDING for a failure).
        No usage rollup: turn usage is derived from `AgentSession.usages`,
        never duplicated on the marker."""
        self._append(
            lambda entry_id, parent_id, ts: TurnFinish(
                id=entry_id, parent_id=parent_id, created_at=ts,
                outcome=outcome, error=error,
            )
        )
        self._refresh_status()

    def _set_status(self, status: ConversationStatus) -> None:
        self.session.active_conversation.status = status


# ── helpers ─────────────────────────────────────────────────────────────────


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
    return len(xs) == len(ys) and all(_equivalent(x, y) for x, y in zip(xs, ys))


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
                {task, waiter}, return_when=asyncio.FIRST_COMPLETED,
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
        grace_deadline = (
            float("inf")
            if grace_ms == Inf
            else asyncio.get_running_loop().time() + grace_ms / 1000.0
        )
    remaining = grace_deadline - asyncio.get_running_loop().time()
    try:
        if remaining == float("inf"):
            value = await asyncio.shield(task)
        else:
            value = await asyncio.wait_for(
                asyncio.shield(task), max(remaining, 0.0),
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
        try:
            task.result()
        except BaseException:
            pass
    else:
        task.add_done_callback(_swallow_result)


async def _cancel_quietly(task: asyncio.Task) -> None:
    task.cancel()
    try:
        await task
    except BaseException:
        pass


def _swallow_result(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()  # retrieved — no unretrieved-exception noise


_POST_PARTS = TypeAdapter(list[ContentPart])


def _normalize_post_parts(content: str | list[ContentPart]) -> list[ContentPart]:
    """`post_message` input → the part list persisted on the `UserMessage`.

    Shape is checked against `ContentPart` itself, so a new part type needs no
    change here; a part the union does not admit raises `ValidationError`."""
    if not content:
        raise AgentError(
            "post_message requires a non-empty string or list of content parts."
        )
    if isinstance(content, str):
        return [TextContent(text=content)]
    if isinstance(content, BaseModel):
        raise AgentError(
            f"post_message takes a list of content parts; wrap the "
            f"{type(content).__name__} in a list."
        )
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
