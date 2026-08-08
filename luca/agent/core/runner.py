"""AgentSessionRunner: a resumable async agent loop over the core data model.

The runner is a small state machine wrapped around an `AgentSession`. A caller
drives it by polling its status and supplying input:

    runner = AgentSessionRunner(session, tool_registry=REGISTRY)
    while True:
        if runner.idle():
            runner.post_message(input("> "))
        else:                                          # BUSY / BLOCKED / CANCELLING
            async with runner.run() as run:
                async for event in run:                # render the event
                    ...
            if runner.blocked():
                ...                                    # resolve on the registry

The sketch polls, but `post_message` is not IDLE-only: posting while the agent
works (BUSY or BLOCKED) appends into the open turn — subagents running
included; a parked parent is woken so the model can steer — and the turn
answers the message before it closes COMPLETED. See `post_message` for the
full acceptance matrix (CANCELLING and an open compaction bracket reject).

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
- the registry left a call's approval PENDING        → status BLOCKED.

Status is DERIVED from the entries on every read (`AgentSession.
get_conversation_status`), never stored, and it is per CONVERSATION: a session
holds a catalog of them (`AgentSession.conversations`), one of which the user is
talking to (`main_conversation_id`) and the rest of which are subagents running
in parallel beneath it. Everything below that names a conversation means that
one; the runner's own predicates are about the main one.

One logical *turn* (the agent's full response to a user message) is bracketed
by a single `TurnStart` / `TurnFinish` (the finish carries the `TurnOutcome`),
even when it pauses for approval across several runs (or a process restart): a
`TurnStart` with no later `TurnFinish` means the open turn is resumed rather
than re-opened. Provider usage is recorded per assistant entry in
`AgentSession.usages[conversation_id][entry_id]` — accessory
conversation-entry data, never embedded in entries or rolled up on markers.

Compaction — replacing the older span of a conversation with a summary of it —
is delegated the same way, to the `ContextManager`'s `should_compact()` /
`compact()` pair (`context_manager.py`; the shipped default declines, so
compaction never happens unless the manager implements it). It runs as a step at
the top of a drive, before the conversational bracket opens, inside a turn
bracket of its own; a successful one archives the conversation and installs a
new one over the path the manager chose, in a single atomic swap. See
`schedule_compaction()` and `_compaction_step`.

The whole tool lifecycle is delegated to the `ToolRegistry` the runner is
constructed with (`tool_registry.py`; `None` = toolless agent). The runner
touches tools through exactly four registry methods: `get_tools` (queried
fresh per LLM call), `create_execution` (the birth draft — the runner stamps
identity and appends), `decide` (approval), and `prepare` (resolution +
validation, returning the callable that runs the body). All four are async,
all four take the live session and the conversation they are answering for, and
all four are raced against the run's
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
deferred — the runner parks (`BLOCKED`) only after all currently
runnable work has advanced, and it never calls the model again until every
tool call in the preceding assistant response has a terminal execution and a
correlated tool output, with one exception (0008): an unseen user post drives
one round past a gate, the gated execution projecting as an
awaiting-approval placeholder, and the drive re-parks at the same gate after
the round.

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

LOGGING. The runner converts exceptions into durable state — a raise becomes a
`ToolExecutionError` or a `TurnFinish(ERRORED)`, and only `str(exc)` survives.
Every one of those conversion points logs at ERROR with `exc_info=True` first,
because this is the only place the traceback still exists. Records carry the
conversation in the message (`conv=<id>`); nothing is configured here — see
`luca/__init__.py`.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import time
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import ClassVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from luca.client import acompletion, acompletion_stream
from luca.client.exceptions import TimeoutError as ClientTimeoutError
from luca.client.types import Tool as LucaTool

from . import adapter
from .compaction import (
    CompactionPlan,
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
    SubagentFinished,
    SubagentPaused,
    SubagentsSpawned,
    SubagentStarted,
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
    ConversationCancellingError,
    IncompleteResponseError,
    InvalidToolArguments,
    ToolNotFound,
)
from .ledger import SessionLedger
from .models import (
    SPAWN_MARKER,
    SPAWN_REQUIRED_KEYS,
    AgentSession,
    AnyEntry,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    CancelRequested,
    ChildConversation,
    CompactionEntry,
    CompactionSource,
    ContentPart,
    Conversation,
    ConversationStatus,
    ExecutionResult,
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
    ToolSpec,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    UserMessage,
    declares_spawn,
    is_compaction_bracket,
    open_turn_unresolved_children,
    open_turn_unseen_material,
    open_turn_unseen_post,
    spawn_payload,
    spawns_committed,
    stop_payload,
)
from .projection import ConversationProjector, tool_message_text
from .system_prompt import (
    DefaultSystemPromptAssembler,
    SystemPromptAssembler,
    SystemPromptPartInput,
    coerce_system_prompt_part,
)
from .tool_registry import PreparedTool, ToolRegistry

logger = logging.getLogger(__name__)

EventCallback = Callable[[AgentEvent], "Awaitable[None] | None"]

# Canonical finish reasons that are NOT an answer, so a round carrying no tool
# calls and one of these must not close the turn COMPLETED. The client's
# vocabulary is already normalized across transports — OpenAI's `length` and
# Anthropic's / Bedrock's `max_tokens` both arrive as "length", and every
# refusal, safety filter and guardrail arrives as "error" with an
# `error_message` — so this set is provider-agnostic by construction. See
# `IncompleteResponseError`.
NON_ANSWER_FINISH_REASONS = frozenset({"error", "length"})


class RunResult(BaseModel):
    """Where a run stopped: the DERIVED status there, plus how the last
    bracket this run closed ended (`None` if it closed none).

    - Turn completed → `status=IDLE`, `outcome` COMPLETED (or CANCELLED for a
      wind-down).
    - Approval pause → `status=BLOCKED`, `outcome=None`, `pending_approvals`
      non-empty. Subtree-scoped: a run whose SUBAGENT gated stops here too, and
      `pending_approvals` names the conversation each gate came from.
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
    pending_approvals: list[ToolExecution]  # non-empty iff BLOCKED


class _SlotWaiter:
    """One request for a worker-pool slot, queued FIFO.

    `grant` is `handle._start_eager` for a never-started child,
    `handle._redrive` for a parked child being restarted, and `Event.set` for
    a live drive re-acquiring after a park (`resume=True`). All three are
    synchronous. `announce` publishes `SubagentStarted` through that handle on
    grant — set for every start and restart, None only where resuming is not
    a start (a parked drive waking)."""

    __slots__ = ("announce", "conversation_id", "grant", "handle", "resume")

    def __init__(
        self,
        conversation_id: str,
        grant: Callable[[], None],
        *,
        handle: AgentRun | None = None,
        announce: AgentRun | None = None,
        resume: bool = False,
    ) -> None:
        self.conversation_id = conversation_id
        self.grant = grant
        self.handle = handle
        self.announce = announce
        self.resume = resume


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
        conversation_id: str,
        streaming: bool,
        on_event: EventCallback | None,
        eager: bool,
        defer_start: bool = False,
        autostart_subagents: bool = True,
        parent: AgentRun | None = None,
    ) -> None:
        self._runner = runner
        # Which conversation this handle drives. A run is per conversation, not
        # per session: the engine, the cancellation token and the one-run guard
        # are all scoped to it.
        self.conversation_id = conversation_id
        self._streaming = streaming
        self._on_event = on_event
        self._eager = eager
        # Who drives the subagents this run spawns. `True` (the default) hands
        # the framework their lifecycle: each starts immediately, advances on
        # its own, and forwards its events here. `False` hands the application
        # a lazy handle per child and the obligation to drive or cancel every
        # one — the parent's turn is blocked on them, so a child that is never
        # driven blocks it forever.
        self.autostart_subagents = autostart_subagents
        self._parent = parent
        self._children: dict[str, AgentRun] = {}
        # Events forwarded from framework-driven children. Only populated when
        # this run OWNS them: under `autostart_subagents=False` the application
        # consumes each child handle directly, and forwarding here too would
        # render every child event twice.
        self._inbox: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self._own_step: asyncio.Task | None = None  # a held, not-yet-consumed pull
        # Gates raised anywhere in this run's subtree, for `run.approvals`.
        self._approvals: asyncio.Queue[ToolExecution | None] = asyncio.Queue()
        self._approvals_closed = False
        # Set whenever a child of this run ENDS. Distinct from `_wake` (the
        # eager buffer's signal) because they answer different questions: this
        # one says "the subtree changed shape", and without it a fan-in wait
        # could block forever on an inbox nothing will fill again.
        self._subtree_wake = asyncio.Event()
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
        # A deferred handle the pool decided never to start (its subtree was
        # suspended, or its admission failed). Terminal: it will never drive.
        self._abandoned = False
        # A per-drive latch for the SubagentPaused / SubagentFinished
        # announcement, so the extra `_end_run` call sites cannot publish the
        # same drive's ending twice. Armed by `_start_eager` / `_redrive`.
        self._ended_published = True
        if eager and not defer_start:
            self._start_eager()

    def _start_eager(self) -> None:
        """Begin this handle's drive — the one door through which an eager
        drive ever starts.

        Validates state synchronously at call time and spawns the background
        task. The loop is resolved FIRST so a sync-context start() fails
        before taking the one-engine guard; the bracket opens durably at call
        time so an immediate cancel() has an open turn to attach to (the first
        drive is then the flush) — and which bracket that is has to be decided
        here too, or a policy-driven compaction could never fire for an eager
        run. A no-op if the drive already started; `_wake` is set at the end
        so a consumer blocked on a not-yet-started handle re-checks."""
        if self._task is not None:
            return
        loop = asyncio.get_running_loop()
        self._runner._begin_run(self)
        try:
            self._runner._open_bracket_for_start(self.conversation_id)
        except BaseException:
            # `_begin_run` has already taken the one-run guard, and the
            # three sites that release it are all downstream of
            # `create_task`, which never runs. Without this the runner is
            # permanently unusable after one raise from application code
            # (`should_compact`, or `before_entry_written`).
            self._runner._end_run(self)
            raise
        self._ended_published = False
        self._task = loop.create_task(self._consume())
        self._wake.set()

    # ── the subtree ─────────────────────────────────────────────────────────

    @property
    def _framework_owned(self) -> bool:
        """True when the FRAMEWORK, not the application, owns this run's
        lifecycle — a child spawned under `autostart_subagents=True`. Only such
        a run may be restarted out of band by `notify()`."""
        return self._parent is not None and self._parent.autostart_subagents

    @property
    def children(self) -> dict[str, AgentRun]:
        """The subagents this run spawned, by conversation id. Grows as spawns
        land; a copy, so mutating it changes nothing."""
        return dict(self._children)

    def child(self, conversation_id: str) -> AgentRun | None:
        """The handle for one conversation anywhere in this run's subtree, or
        None. Handles come from the parent run — never from an event, which is
        a serializable snapshot meant to be forwarded.

        Under `autostart_subagents=False` this also MINTS: a run handle is
        single-use, and a subagent's drive ends the moment its gate defers (see
        `_await_subtree`), so a handle that already ran is replaced by a fresh
        one, and an unresolved child of this conversation that this run never
        spawned itself gets one too. Without that there is no door back to a
        parked subagent — and the parent's turn stays open until every child
        resolves, so the conversation would wedge. Under `True` the framework
        owns those restarts (`notify()`, and the next `run()`), and this stays a
        pure lookup."""
        found = self._lookup(conversation_id)
        if found is not None and not (found._exited or found._finished):
            return found
        if not self.autostart_subagents and conversation_id in self._runner._unresolved_child_ids(
            self.conversation_id,
        ):
            fresh = AgentRun(
                self._runner,
                conversation_id=conversation_id,
                streaming=self._streaming,
                on_event=None,  # the application consumes this handle directly
                eager=False,
                autostart_subagents=False,
                parent=self,
            )
            self._adopt(fresh)
            return fresh
        return found

    def _lookup(self, conversation_id: str) -> AgentRun | None:
        """Pure subtree lookup, finalized handles included. `child()` is the
        door that may replace one."""
        found = self._children.get(conversation_id)
        if found is not None:
            return found
        for child in self._children.values():
            found = child._lookup(conversation_id)
            if found is not None:
                return found
        return None

    def _adopt(self, child: AgentRun) -> None:
        self._children[child.conversation_id] = child

    def _redrive(self) -> None:
        """Start a NEW drive on a handle whose previous one already ended.

        For a parked child this is a restart, not a wake: that drive is gone,
        so there is nothing to signal. It is legitimate precisely because
        `autostart_subagents=True` handed the framework this child's lifecycle
        — and it is why `False` needs no equivalent, since there the
        application's own `drive(child)` returned and it already has control.
        The buffer is cumulative across drives, so a consumer reading this
        handle sees one continuous stream."""
        if self._task is None or not self._task.done():
            # None: never started — a deferred handle's first start belongs to
            # the pool (`_start_eager`), and a drive spun up here would run
            # with no token and no bracket. Not done: already driving.
            return
        self._finished = False
        self._exception = None
        # The previous drive closed this handle's approvals stream; a new drive
        # can raise new gates, so it reopens.
        self._approvals_closed = False
        self._ended_published = False
        # RE-REGISTER. `_runs` is what every liveness check keys on — the
        # cancel door's token trip, `_flush_cancelled_children`'s "no drive
        # left", `_settle_children`'s task wait — and a redriven drive that is
        # absent from it reads as dead: a `stop_subagent` (or any cancel)
        # against it would skip the token AND let the parent's flush close a
        # turn this drive keeps writing to. Safe to claim unconditionally: the
        # redrive paths (`_ensure_driven`, the pool's stale-waiter check) only
        # fire when no live run holds the conversation.
        self._runner._runs[self.conversation_id] = self
        self._task = asyncio.get_running_loop().create_task(self._consume())

    # ── the three consumption forms ─────────────────────────────────────────

    def cancel(
        self,
        outcome: TurnOutcome = TurnOutcome.CANCELLED,
        error: str | None = None,
    ) -> None:
        """Cancel THIS run's conversation and cascade to every live subagent
        beneath it. Cancelling one subagent leaves its siblings and its parent
        running: the parent's `ChildConversation` still resolves, with a result
        that reflects the cancellation."""
        self._runner.cancel(outcome, error, conversation_id=self.conversation_id)

    def cancel_drive(self) -> None:
        """Park this run's SUBTREE at its next step boundaries WITHOUT writing
        a `CancelRequested`.

        This is suspension, not cancellation: nothing durable is recorded and
        the bracket stays open and resumable. `cancel()` is the other thing —
        it writes the durable request and closes the turn. A child that never
        started (queued behind the worker pool) is ABANDONED instead: there is
        no drive to park, but the pool must not start one for a suspended
        tree. The cascade is recursive so a nested tree suspends whole."""
        if self._eager and self._task is None:
            self._abandoned = True
            self._wake.set()  # a joiner parked in _await_started re-checks
        if self._task is not None and not self._task.done():
            self._task.cancel()
        # A pending slot request for this conversation — a queued start, or a
        # gated child's already-answered redrive — must not fire after the
        # suspension: the pool would otherwise run real model/tool work
        # inside a tree the application believes is fully parked.
        self._runner._drop_waiter(self.conversation_id)
        for child in self._children.values():
            child.cancel_drive()

    def notify(self, execution: ToolExecution) -> None:
        """Something changed out of band for this execution's conversation —
        look again NOW.

        NO DECISION TRAVELS THROUGH THIS. The answer still reaches the runner
        through `decide()`, the engine's single call site; this says only
        "something changed, re-ask". It is SYNC for the same reason `cancel()`
        is: it is a signal, not work. Making it async would put the re-decision
        on the CALLER's task, which would escape the cancellation race, race
        the engine's own decide step for the same execution, and have nowhere
        to emit the `ToolExecuted` a DENY produces.

        `execution` is an ADDRESS: it resolves to `execution.conversation_id`,
        which is exactly why that field exists and why this needs no
        conversation argument. Ids, not object identity — events carry deep
        snapshots, so the runner re-reads the live entry.

        There is deliberately no `runner.notify()`: outside a run, the next
        `run()` already re-asks every undecided execution."""
        conversation_id = execution.conversation_id
        if conversation_id is None:
            raise AgentError(
                "notify() needs an execution that carries a conversation_id; "
                "the runner stamps one on every execution it creates."
            )
        self._runner._recheck.add(conversation_id)
        self._runner._ensure_driven(conversation_id)

    @property
    def approvals(self) -> AsyncIterator[ToolExecution]:
        """Gates raised DURING this run, as they are raised — subtree-scoped,
        closing when the run ends.

        It exists for one situation the polling loop cannot reach on its own: a
        subagent gates while its siblings keep working, so the run does not
        return and there is no between-drives moment to check
        `pending_approvals()` in. The top-level run notifies its own gates plus
        every subagent's; `run.child(cid).approvals` narrows to that subtree.

        The element type is `ToolExecution` — the same objects
        `pending_approvals()` returns, attributed by `execution.conversation_id`.
        There is no request wrapper.

        AT-LEAST-ONCE. `strategy.apply_answer` writes to the strategy, not to
        the execution, so `approval_status` stays PENDING until a drive re-asks
        `decide()` — and `notify()` deliberately causes another pass. Consumers
        must dedup."""
        return self._drain_approvals()

    async def _drain_approvals(self) -> AsyncIterator[ToolExecution]:
        while True:
            execution = await self._approvals.get()
            if execution is None:  # the run ended
                return
            yield execution

    def _publish_approval(self, execution: ToolExecution) -> None:
        """Put a gate on this run's stream and on every ancestor's, so a root
        handle sees its whole subtree's gates."""
        run: AgentRun | None = self
        while run is not None:
            if not run._approvals_closed:
                run._approvals.put_nowait(execution)
            run = run._parent

    def _close_approvals(self) -> None:
        if not self._approvals_closed:
            self._approvals_closed = True
            self._approvals.put_nowait(None)

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
        # it) and finalize the handle. No entry is written, and there is no
        # status to re-derive: status is a pure function of the entries.
        if self._engine is not None and not self._finished:
            self._finished = True
            await self._close_engine()
            self._runner._end_run(self)
            self._close_approvals()
        # Suspension cascades: a parked parent parks its subagents too, each at
        # its own next step boundary.
        for child in self._children.values():
            child.cancel_drive()
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
        """Advance the lazy run one event, delivering it to `on_event`.

        THE STREAM IS THE TREE'S. This run's own drive and its framework-driven
        subagents produce events concurrently, so a pull races the two: one
        step of the own engine against whatever a child has already forwarded.
        Whichever lands first is yielded, and a pending own-step is HELD for
        the next pull rather than discarded — an engine step is a whole LLM
        call, and dropping one would lose its event.

        Iteration still gates this run's own drive, exactly as before. What it
        no longer gates is the subagents: they are background tasks and advance
        whether or not anyone pulls."""
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
                self.conversation_id,
                streaming=self._streaming,
                token=self._token,
            )
        try:
            event = await self._next_own_or_forwarded()
        except StopAsyncIteration:
            self._finished = True
            self.result = self._runner._build_run_result(self.conversation_id)
            self._runner._end_run(self)
            self._close_approvals()
            raise
        except BaseException as exc:  # engine raised (incl. external cancel)
            self._finished = True
            self._exception = exc
            self._runner._end_run(self)
            self._close_approvals()
            raise
        try:
            await self._deliver(event)
        except BaseException as exc:
            # on_event is app code: crash semantics — tear the engine down,
            # leave the bracket open (resumable), propagate.
            self._finished = True
            self._exception = exc
            await self._close_engine()
            self._runner._end_run(self)
            self._close_approvals()
            raise
        return event

    async def _next_own_or_forwarded(self) -> AgentEvent:
        """One event from this run's own engine or from a child, whichever is
        ready first.

        Terminates only when the whole SUBTREE is settled: the own engine
        exhausted AND no child still running AND the inbox drained. That is
        what makes `ApprovalRequired` non-terminal on a parent's stream — a
        subagent gating cannot end a run while a sibling is still working.

        A pending own-step is HELD, never discarded: an engine step is a whole
        LLM call, and cancelling one to take a child's event would lose it.
        `asyncio.Queue.get()` is the opposite — cancelling a pending get
        consumes nothing — so the inbox getter is the one that is torn down."""
        while True:
            if not self._inbox.empty():
                return self._inbox.get_nowait()
            if self._own_step is None and self._engine is not None:
                self._own_step = asyncio.ensure_future(self._engine.__anext__())

            waiters: set[asyncio.Future] = set()
            if self._own_step is not None:
                waiters.add(self._own_step)
            forwarded: asyncio.Future | None = None
            settled: asyncio.Future | None = None
            # The inbox is armed while the OWN ENGINE is live too, not only
            # while children are: an engine step can publish onto its own
            # run's inbox mid-step (the spawn announcement and the batch's
            # `SubagentStarted`s), and an unarmed inbox would sit undelivered
            # until the step's next yield. Termination is unaffected — with
            # the engine exhausted and no live children nothing arms, and the
            # loop top already drained the queue.
            if self._live_children() or self._engine is not None:
                self._subtree_wake.clear()
                if not self._inbox.empty():
                    return self._inbox.get_nowait()
                forwarded = asyncio.ensure_future(self._inbox.get())
                # A child ENDING is also progress: without this the wait could
                # block forever on an inbox nothing will ever fill again.
                settled = asyncio.ensure_future(self._subtree_wake.wait())
                waiters |= {forwarded, settled}
            if not waiters:
                raise StopAsyncIteration

            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            for helper in (forwarded, settled):
                if helper is not None and helper not in done:
                    await _cancel_quietly(helper)

            if forwarded is not None and forwarded in done:
                if settled is not None and settled in done:
                    await _cancel_quietly(settled)
                return forwarded.result()
            if self._own_step is not None and self._own_step in done:
                if not self._inbox.empty():
                    # Events published DURING the step that just completed go
                    # first — the same inbox-before-engine rule the loop top
                    # applies, made consistent WITHIN a pull. The completed
                    # step is held; a later iteration consumes its result.
                    return self._inbox.get_nowait()
                step, self._own_step = self._own_step, None
                try:
                    return step.result()
                except StopAsyncIteration:
                    self._engine = None
                    if self._live_children() or not self._inbox.empty():
                        continue  # the subtree is still going
                    raise
            # only `settled` fired: a child ended. Loop and re-evaluate.

    @property
    def _queued(self) -> bool:
        """Created for an eager start the worker pool has not granted yet.

        Derived from the conversation rather than latched on the handle, so a
        queued child that is cancelled and flushed by its parent's drive
        (turn closed → IDLE) stops counting on its own — no bookkeeping call
        has to remember to clear it."""
        if not self._eager or self._task is not None or self._abandoned:
            return False
        status = self._runner.session.get_conversation_status(self.conversation_id).status
        return status is not ConversationStatus.IDLE

    def _live_children(self) -> bool:
        # A queued child counts: it has not produced events yet, but it will —
        # ending the parent's stream on it would truncate the run (§ the fan-in
        # terminates only when the whole subtree is settled).
        return any(
            (child._task is not None and not child._task.done()) or child._queued for child in self._children.values()
        )

    async def _close_engine(self) -> None:
        """Tear the engine down, held step first.

        The fan-in may be holding a started-but-unconsumed `__anext__`, and
        `aclose()` on a generator with a pending step raises. Killing the step
        is also what unwinds the drive: the generator's `finally` blocks run,
        so nothing is left half-open."""
        if self._own_step is not None:
            step, self._own_step = self._own_step, None
            await _cancel_quietly(step)
        if self._engine is not None:
            engine, self._engine = self._engine, None
            await engine.aclose()

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
        self._engine = self._runner._drive(
            self.conversation_id,
            streaming=self._streaming,
            token=self._token,
        )
        try:
            while True:
                try:
                    event = await self._next_own_or_forwarded()
                except StopAsyncIteration:
                    break
                self._buffer.append(event)
                self._wake.set()
                self._forward(event)
                await self._deliver(event)
            self.result = self._runner._build_run_result(self.conversation_id)
            return self.result
        except BaseException as exc:
            self._exception = exc
            raise
        finally:
            self._finished = True
            self._wake.set()
            try:
                await self._close_engine()  # no-op unless on_event/cancel left it open
            finally:
                self._runner._end_run(self)  # guard released after teardown
                self._close_approvals()
                if self._parent is not None:
                    # Two different sleepers, and both have to be woken. The
                    # parent's FAN-IN may be waiting for another event; the
                    # parent's DRIVE may be blocked on this very child.
                    self._parent._subtree_wake.set()
                    self._parent._wake.set()
                    self._runner._ensure_driven(self._parent.conversation_id)

    def _forward(self, event: AgentEvent) -> None:
        """Push one of THIS run's events onto its parent's inbox.

        Forwarding follows OWNERSHIP: only a framework-driven child forwards.
        Under `autostart_subagents=False` the application consumes each child
        handle itself, and forwarding here as well would deliver every child
        event twice — once on the parent's stream and once on the child's."""
        if self._framework_owned:
            self._parent._inbox.put_nowait(event)

    async def _next_buffered(self) -> AgentEvent:
        # `_task is None` is a deferred handle the pool has not admitted yet:
        # not done, nothing buffered — wait on `_wake`, which `_start_eager`
        # sets on admission (and abandonment sets, ending the empty stream).
        while True:
            if self._cursor < len(self._buffer):
                event = self._buffer[self._cursor]
                self._cursor += 1
                return event
            if self._task is None:
                if self._abandoned:
                    raise StopAsyncIteration  # never started — an empty stream
            elif self._task.done():
                if self._task.cancelled():
                    raise asyncio.CancelledError()
                exc = self._task.exception()
                if exc is not None:
                    raise exc  # buffer drained → surface the failure
                raise StopAsyncIteration
            self._wake.clear()
            if self._cursor < len(self._buffer) or self._abandoned or (self._task is not None and self._task.done()):
                continue  # produced between the check and the clear
            await self._wake.wait()

    async def _await_started(self) -> None:
        """Joining a deferred handle waits for the pool to admit it first —
        that is the honest answer to "join this child". An ABANDONED handle
        (cancelled or suspended before admission) will never start, and
        consumers return with whatever the session records for it instead of
        blocking forever."""
        while self._task is None and not self._abandoned:
            self._wake.clear()
            if self._task is not None or self._abandoned:
                break
            await self._wake.wait()

    async def _join(self) -> RunResult:
        await self._await_started()
        if self._task is None:
            # abandoned before it ever started — no drive ever ran, so the
            # session-derived result is the whole answer
            return self._runner._build_run_result(self.conversation_id)
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
        await self._await_started()
        if self._task is None:
            return  # abandoned before it ever started — nothing to finalize
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
        try:
            result = self._on_event(event)
            if inspect.isawaitable(result):
                await result
        except Exception:
            # Application code with crash semantics: the caller sees the raise,
            # but only here does its traceback still exist. `Exception`, not
            # `BaseException`, so a cancellation passes through unlogged.
            logger.error(
                "conv=%s on_event callback raised on %s",
                self.conversation_id,
                type(event).__name__,
                exc_info=True,
            )
            raise


class AgentSessionRunner:
    """Stateful driver over an `AgentSession`. Owns the `ToolRegistry` (the
    single tool touch point; `None` = toolless agent), the system-prompt
    parts + assembler (see `system_prompt.py`), and the id/clock hooks;
    mutates the session in place (through its `SessionLedger`)."""

    #: What an unresolved subagent's result says when its parent was cancelled.
    #: Class-level so an application can change it without touching the runner.
    CANCELLED_SUBAGENT_TEXT: ClassVar[str] = "[subagent cancelled]"

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
        conversation_id = conversation_id or uuid4().hex[:8]
        ts = _now_ms()
        return AgentSession(
            id=session_id,
            conversations={
                conversation_id: Conversation(
                    id=conversation_id,
                    nodes=[],
                    created_at=ts,
                    updated_at=ts,
                    depth=0,
                ),
            },
            main_conversation_id=conversation_id,
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
        self.conversation_projector = conversation_projector or ConversationProjector()
        # The context strategy (context_manager.py): calculates every new
        # entry's `context_tokens`, processes returned tool output, builds
        # pruned replacements, and owns compaction — when, what, with which
        # model, which nodes survive, and whether to try again; the runner only
        # triggers, stamps and archives. Same collaborator pattern as the
        # projector — one object, never serialized, defaults to the simple
        # built-in. That default declines to compact, so compaction never
        # happens unless the manager implements it.
        self.context_manager = context_manager or ContextManager()
        # Static parts (str / dict / SystemPromptPart) coerce eagerly — a bad
        # part fails at construction, not mid-turn. Callables resolve per call.
        self.system_prompt_parts = [
            part if callable(part) else coerce_system_prompt_part(part) for part in (system_prompt_parts or [])
        ]
        self.system_prompt_assembler = system_prompt_assembler or DefaultSystemPromptAssembler()
        self.provider = provider
        self.middleware = list(middleware or [])
        self.ledger = SessionLedger(session, self.now_ms, self.generate_id)
        # ── runtime-only, never serialized (the same class of state as a
        # `CancellationToken`) ────────────────────────────────────────────────
        # One live run per CONVERSATION, not one per session: the main
        # conversation and every subagent advance at the same time.
        self._runs: dict[str, AgentRun] = {}
        # A live drive's wake, per conversation. Present iff a drive is
        # currently inside its loop and able to be woken.
        self._wakes: dict[str, asyncio.Event] = {}
        # Conversations with an unconsumed "look again" from `run.notify()`.
        # CONVERSATION ids, not execution ids: once a drive is woken its decide
        # step re-asks every undecided execution in the open turn anyway, and
        # `decide()` is contractually an idempotent query, so gated siblings are
        # re-asked for free. A per-execution dirty set would be a second,
        # narrower decide path alongside the one the loop already has.
        # Never serialized: on a cold load the next `run()` re-asks everything
        # undecided, so there is nothing to persist, and a stale id left behind
        # is harmless — waking a conversation with nothing undecided is a no-op.
        self._recheck: set[str] = set()
        # ── the subagent worker pool (`subagents_max_workers`) ──────────────
        # Conversations currently holding a slot: a strict subset of the
        # subagent conversations, and NOT derivable from `_runs` — a drive
        # parked on its subtree or winding a cancelled turn down is live but
        # holds no slot, and a just-granted child holds one before its task
        # exists. With the default `Inf` both stay empty and every request is
        # granted inline.
        self._working: set[str] = set()
        # Everything that wants a slot, in FIFO order: queued children waiting
        # to start, gated children waiting to be re-driven, parked drives
        # waiting to resume. One queue, so ordering is a single testable
        # policy.
        self._waiters: list[_SlotWaiter] = []
        # A pool-granted start that failed (`before_entry_written` raising on
        # the child's `TurnStart`, say) has no caller to raise into — the
        # exception is recorded against the CHILD conversation here and
        # re-raised by the PARENT's drive at its loop top. Fail-loud,
        # preserved for deferred starts.
        self._admission_errors: dict[str, BaseException] = {}
        # Per-DRIVE state, keyed by conversation and reset in `_begin_run`.
        # Keyed rather than plain attributes because several drives run at
        # once: a plain attribute would belong to whichever conversation wrote
        # it last, which is the silent failure §3.4's rule 1 is about.
        #
        # `_closed_outcomes` is how the LAST bracket a run closed ended —
        # carried rather than re-read from the path, because after a compaction
        # the closing marker is on the archived conversation. The two flag sets
        # belong to the compaction step: whether it consumed a cancellation
        # (the drive must then stop without answering the queued turn) and
        # whether it did anything at all (only then may a re-derived IDLE end
        # the drive).
        self._closed_outcomes: dict[str, TurnOutcome | None] = {}
        self._compaction_cancelled: set[str] = set()
        self._compaction_did_run: set[str] = set()

        # Nothing to normalize on load: status is derived from the entries on
        # every read, so a reloaded session cannot carry a stale one.
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
        tool registry, prompt parts, assembler, provider, context manager,
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

    # ── status predicates (the MAIN conversation) ──────────────────────────
    #
    # All four answer for the main conversation, which is what a caller driving
    # the session asks about. A subagent's status is
    # `session.get_conversation_status(conversation_id)` — the one door — and
    # the runner deliberately offers no second way to reach it.

    @property
    def main_conversation_id(self) -> str:
        return self.session.main_conversation_id

    @property
    def status(self) -> ConversationStatus:
        """The main conversation's DERIVED status. Recomputed on every read;
        nothing caches it."""
        return self.session.get_conversation_status(self.main_conversation_id).status

    def idle(self) -> bool:
        """Nothing to do — post a message to give the next `run()` work."""
        return self.status == ConversationStatus.IDLE

    def busy(self) -> bool:
        """There is work; a run can still be exhausted."""
        return self.status == ConversationStatus.BUSY

    def blocked(self) -> bool:
        """Nothing can advance — you must act first (usually: answer a gate)."""
        return self.status == ConversationStatus.BLOCKED

    def cancelling(self) -> bool:
        """An unconsumed cancel; the next drive is the flush."""
        return self.status == ConversationStatus.CANCELLING

    # ── caller-facing mutations / queries ────────────────────────────────────

    def post_message(
        self,
        content: str | list[ContentPart],
        conversation_id: str | None = None,
    ) -> str:
        """Append a user message — to the MAIN conversation by default
        (`conversation_id=None`, resolved at call time), or to any
        conversation named. Returns the persisted entry's id.

        A post is accepted or rejected from the target's durable state at the
        instant of the call; the method is synchronous, so that picture is
        exact — nothing can change between the check and the append. The
        rule: a conversation accepts a message whenever something will
        eventually answer it.

        ACCEPTED:

        - IDLE main conversation — the classic case; the next `run()` opens a
          turn.
        - BUSY with no open turn — a trailing message is already queued (the
          next turn answers both), or a freshly-spawned subagent still
          holding its seed (the post projects right behind it).
        - Any OPEN conversational turn, BUSY or BLOCKED — the mid-turn
          append, subagents included. The message lands inside the turn, the
          model sees it on its next LLM call, and the turn does not close
          COMPLETED until it has (the drive's close-site check loops for one
          more round instead). Posting into a BLOCKED turn reaches the model
          too (0008): the conversation derives BUSY, the next drive projects
          the gated execution as an awaiting-approval placeholder and calls
          the model — one round, answering the post while the approval prompt
          is still up — then re-parks at the same gate and derives BLOCKED
          again. The gate itself is untouched: a post is not an approval and
          never becomes one; `pending_approvals()` returns it before and
          after the round.
        - An open turn with unresolved subagents — the mid-ORCHESTRATION
          append. The children never see it, but the parent does: a parked
          drive is woken and calls the model with the message (still-running
          tasks project nothing until they resolve — the model tracks them
          through the spawn confirmations and `list_subagents`), so the model
          can steer — answer, spawn more, or stop a subagent. Any
          conversation, main or not.

        REJECTED:

        - CANCELLING → `ConversationCancellingError`. The turn is being
          flushed; an append would be buried in the cancelled bracket. Catch,
          keep the draft, retry after the flush. Durable: a reloaded
          CANCELLING session refuses input the same way.
        - An open COMPACTION bracket, scheduled or in flight → `AgentError`.
          The compaction snapshot is built on "nothing is appended while the
          bracket is open"; drive the compaction first.
        - An archived conversation (a compaction replaced it) or a finished
          subagent → `AgentError`. Nothing will ever drive either again;
          accepting would wedge the message forever.
        - An unknown `conversation_id` → `AgentError`.

        A message caught inside a turn that then closes CANCELLED / ERRORED /
        TIMED_OUT is BURIED: that turn never answers it. Buried is not lost —
        projection walks failed brackets, so it reaches the model with the
        next engagement; late, not never.

        `content` is a bare string (the common case) or an ordered list of
        parts mixing text and images; `before_post_message` sees that list and
        returns the one that is persisted."""
        target = self.main_conversation_id if conversation_id is None else conversation_id
        conversation = self.ledger.conversation(target)  # unknown id raises
        entries = self.session.entries
        if self.ledger.open_turn_cancel_requested(target) is not None:
            raise ConversationCancellingError(
                f"conversation {target!r} is cancelling; the open turn is being flushed. Retry after the flush."
            )
        open_index = self.ledger.open_turn_index(target)
        if open_index is not None and is_compaction_bracket(conversation.nodes, entries, open_index):
            raise AgentError(
                f"conversation {target!r} has a compaction scheduled or in flight; drive it before posting."
            )
        # Archived-ness is identity, not status: a queued message compacted
        # behind leaves the archived path deriving BUSY, and a status check
        # would accept into a conversation nothing will ever drive.
        if any(c.previous_conversation_id == target for c in self.session.conversations.values()):
            raise AgentError(f"conversation {target!r} is archived (a compaction replaced it); post to its successor.")
        if target != self.main_conversation_id and open_index is None:
            status = self.session.get_conversation_status(target).status
            if status is not ConversationStatus.BUSY:
                raise AgentError(
                    f"conversation {target!r} is a finished subagent; its result is "
                    f"already resolved and nothing will ever drive it again."
                )
        parts = self._run_middlewares(
            "before_post_message",
            target,
            _normalize_post_parts(content),
        )
        message = self._append(
            target,
            lambda entry_id, parent_id, ts: UserMessage(
                id=entry_id,
                parent_id=parent_id,
                created_at=ts,
                parts=parts,
            ),
        )
        # A drive parked on its subagents must look again NOW — the message is
        # material and the model can steer. The full notify door (`_recheck` +
        # `_ensure_driven`), the same pair `run.notify()` uses: the recheck set
        # is what survives a drive caught mid-teardown, where the wake event
        # alone would be set on something that never loops again. A no-op for
        # everything else — an idle conversation has no drive to wake, and the
        # next `run()` reads the message off the path anyway.
        self._recheck.add(target)
        self._ensure_driven(target)
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
        open bracket already derives BUSY ("work is queued, call run()").
        The price is that `post_message` — which refuses an open compaction
        bracket — raises until that drive has run, across a reload; the
        compaction snapshot is built on nothing being appended while its
        bracket is open.

        Requires a CLOSED bracket. An open conversational turn (a resumable
        bracket, BLOCKED, CANCELLING) raises `AgentError`: the
        appended `TurnStart` would nest inside it, and `open_turn_index()`
        walks back to the NEAREST one, so the open turn's eventual
        `TurnFinish` would silently close the wrong bracket.

        It does NOT check that the `ContextManager` implements compaction —
        there is nothing to check, since a manager always exists. A manager
        that only accounts reaches the base `compact()` on the next drive, and
        its `NotImplementedError` closes the bracket ERRORED and propagates out
        of `run()` (this is a USER-sourced compaction; see `_run_compaction`)."""
        main = self.main_conversation_id
        existing = self.ledger.open_compaction_entry(main)
        if existing is not None:
            return existing.id  # idempotent — nothing is written
        if self.ledger.open_turn_index(main) is not None:
            raise AgentError(f"schedule_compaction requires a closed turn (status={self.status.value}).")
        entry = self._open_compaction_bracket(main, CompactionSource.USER)
        return entry.id

    def pending_approvals(self, conversation_id: str | None = None) -> list[ToolExecution]:
        """The open turn's executions awaiting an out-of-band approval — those
        whose `approval_status` is PENDING. Each is self-contained
        (`raw_tool_call` + whatever the registry recorded in `extras`);
        resolve them on the registry's own state, then call `run()` again (it
        asks the registry again — no posting back through the runner).

        SUBTREE-SCOPED: asking a conversation returns every gated execution
        BENEATH it, which is how a subagent's request reaches the main
        conversation's caller while the parent is still BUSY and the subagent's
        siblings keep working. Everything you ask a conversation is about its
        subtree.

        The element type does not change — a flat `list[ToolExecution]`
        spanning several conversations is still attributable, because each
        carries its own `conversation_id`. That is what lets an interactive
        application say "subagent B is asking" without any wrapper type."""
        root = conversation_id or self.main_conversation_id
        awaiting = list(self.ledger.open_turn_awaiting_executions(root))
        for child_id in self._unresolved_child_ids(root):
            if child_id in self.session.conversations:
                awaiting.extend(self.pending_approvals(child_id))
        return awaiting

    def cancel(
        self,
        outcome: TurnOutcome = TurnOutcome.CANCELLED,
        error: str | None = None,
        *,
        conversation_id: str | None = None,
    ) -> None:
        """The universal cancellation door — synchronous, conversation-scoped
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
           this branch is an undriven lazy handle, or no run at all).

        ONE EXCEPTION to 3, and it is what makes "cancel a subagent" mean
        something: an unresolved SUBAGENT with no bracket (spawned, never
        driven) gets one opened here, so the cancellation has something to
        close. Without it the call would be a silent no-op and the parent —
        whose turn cannot end until that child resolves — would wait forever
        for a conversation nobody is ever going to drive."""
        target = conversation_id or self.main_conversation_id
        if self.ledger.open_turn_index(target) is None:
            if self._link_for(target) is None:
                return
            self._ensure_open_turn(target)
        if self.ledger.open_turn_cancel_requested(target) is not None:
            raise AlreadyCancellingError(
                "a cancellation is already pending for the open turn; the first request's outcome/error stand"
            )
        self._append(
            target,
            lambda entry_id, parent_id, ts: CancelRequested(
                id=entry_id,
                parent_id=parent_id,
                created_at=ts,
                outcome=outcome,
                error=error,
            ),
        )
        run = self._runs.get(target)
        if run is not None and run._token is not None:
            run._token.cancel()
        if run is None:
            # No live drive will consume this request, so nothing would wake
            # on its own — a real hazard since the worker pool made "spawned
            # but never started" an ordinary state. A QUEUED handle is
            # abandoned (the pool must never start it now, and a joiner
            # blocked on it must unblock), and the PARENT's drive is woken:
            # it is the only conversation that will flush this child
            # (`_flush_cancelled_children`) and resolve its link.
            handle = self._parked_handle(target)
            if handle is not None and handle._eager and handle._task is None:
                handle._abandoned = True
                handle._wake.set()
            parent_id = self._parent_of(target)
            if parent_id is not None:
                self._ensure_driven(parent_id)
        self._cascade_cancel(target, outcome, error)

    def _cascade_cancel(
        self,
        conversation_id: str,
        outcome: TurnOutcome,
        error: str | None,
    ) -> None:
        """Cancel every live descendant of `conversation_id`.

        Cancelling a conversation cancels the WORK it is waiting on, and a
        parent blocked on its children is waiting on them. The cascade must
        tolerate a child that is already cancelling — cancelling one subagent
        and then the whole tree is a normal sequence, not an error — so
        `AlreadyCancellingError` is swallowed here, and only the caller's own
        direct request can raise it."""
        for child_id in self._unresolved_child_ids(conversation_id):
            with contextlib.suppress(AlreadyCancellingError):
                self.cancel(outcome, error, conversation_id=child_id)

    def _link_for(self, child_id: str) -> ChildConversation | None:
        """The unresolved `ChildConversation` naming this conversation, if it is
        somebody's subagent. Child → parent is not a link the data model
        carries (parent → child is the only direction anything traverses), so
        this scans; it is only ever called on a cancellation."""
        for conversation in self.session.conversations.values():
            for node_id in conversation.nodes:
                entry = self.session.entries.get(node_id)
                if (
                    isinstance(entry, ChildConversation)
                    and entry.conversation_id == child_id
                    and entry.execution_result is None
                ):
                    return entry
        return None

    def _parent_of(self, child_id: str) -> str | None:
        """The conversation whose path holds the unresolved link to
        `child_id`, if any — the one whose drive flushes it. Same scan as
        `_link_for`, cancellation-only."""
        for conversation_id, conversation in self.session.conversations.items():
            for node_id in conversation.nodes:
                entry = self.session.entries.get(node_id)
                if (
                    isinstance(entry, ChildConversation)
                    and entry.conversation_id == child_id
                    and entry.execution_result is None
                ):
                    return conversation_id
        return None

    def _unresolved_child_ids(self, conversation_id: str) -> list[str]:
        """The conversations spawned by `conversation_id`'s open turn that have
        not yet produced a result."""
        entries = self.session.entries
        return [
            entry.conversation_id
            for node_id in self.ledger.conversation(conversation_id).nodes
            if isinstance(entry := entries.get(node_id), ChildConversation) and entry.execution_result is None
        ]

    # ── the two run methods ──────────────────────────────────────────────────

    def run(
        self,
        *,
        streaming: bool = False,
        on_event: EventCallback | None = None,
        autostart_subagents: bool = True,
    ) -> AgentRun:
        """Lazy: nothing happens until awaited or iterated; stopping iteration
        stops the agent.

        `autostart_subagents=True` (the default) has the FRAMEWORK drive every
        subagent this run spawns: each begins immediately, advances on its own
        whatever mode the parent is in, and its events arrive on this stream
        tagged with their conversation. Iteration therefore stops gating all
        work — your pull rate controls this conversation's drive and the
        delivery of events, but the subagents progress regardless. Suspension
        still cascades: leaving the context manager parks every child at its
        next step boundary.

        `False` hands you a lazy, unstarted handle per child (`run.child(cid)`,
        announced by `SubagentsSpawned`) and the obligation that comes with it:
        the turn is blocked on its children, so every spawn must be driven or
        cancelled or it blocks forever.

        Creating (and discarding) the handle is harmless — no work, no
        validation; the IDLE/concurrent-run guards fire at first drive.
        `streaming` selects the event vocabulary only (block events vs block +
        delta events). Events reach `on_event` even when the run is only
        awaited; without it, an awaited run discards them."""
        if not autostart_subagents and self.session.session_config.runtime_config.subagents_max_workers != Inf:
            raise AgentError(
                "subagents_max_workers is framework-owned: it works by withholding a "
                "subagent's start, which is only the framework's to withhold under "
                "autostart_subagents=True. Drive the children yourself and pace them "
                "yourself, or let the framework drive them and cap them."
            )
        return AgentRun(
            self,
            conversation_id=self.session.main_conversation_id,
            streaming=streaming,
            on_event=on_event,
            eager=False,
            autostart_subagents=autostart_subagents,
        )

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
        history from event 0 — and also delivered to `on_event`.

        Implies `autostart_subagents=True`: an eager run completes regardless
        of observation, and a subagent nobody drives would stop it doing so."""
        return AgentRun(
            self,
            conversation_id=self.session.main_conversation_id,
            streaming=streaming,
            on_event=on_event,
            eager=True,
            autostart_subagents=True,
        )

    # ── run lifecycle plumbing (used by AgentRun) ────────────────────────────

    def _begin_run(self, run: AgentRun) -> None:
        """First-drive gate: one engine per CONVERSATION, runnable status, and
        the run's own CancellationToken. There is no per-run context object:
        registries and tools receive the live session (`context.py`).

        The guard is per conversation rather than per session because that is
        what parallel subagents are: several conversations advancing at once,
        each with exactly one engine driving it."""
        if self._runs.get(run.conversation_id) is not None:
            raise AgentError(
                f"another run is already active on conversation {run.conversation_id!r}; finish or finalize it first"
            )
        if self.session.get_conversation_status(run.conversation_id).status == ConversationStatus.IDLE:
            raise AgentError("Nothing to run; call post_message() first.")
        run._token = CancellationToken()
        # Per-run, so a handle re-driven after an approval pause (or a
        # suspended lazy run resumed by a fresh `run()`) never reports the
        # previous bracket's outcome. `None` is correct for a run that closed
        # nothing.
        self._closed_outcomes[run.conversation_id] = None
        self._compaction_cancelled.discard(run.conversation_id)
        self._compaction_did_run.discard(run.conversation_id)
        self._runs[run.conversation_id] = run
        if run.autostart_subagents:
            self._restart_unresolved_children(run)

    def _restart_unresolved_children(self, run: AgentRun) -> None:
        """Re-drive the subtree a previous run left parked.

        A subagent's drive ENDS as soon as its gate defers — nothing in its own
        subtree can advance — so answering that gate has to restart it. Inside a
        live run `notify()` does that; once every drive has ended there is
        nothing left to signal, and this is what makes the documented recovery
        ("answer, then run again") mean the same thing for a subagent as for the
        main conversation. Without it a parent blocked on a gated child could
        never be unblocked, since the parent's own drive returns immediately
        while its only child is BLOCKED.

        Skipped under `autostart_subagents=False`: there the application owns
        these lifecycles and `run.child()` hands the handles back."""
        for child_id in self._unresolved_child_ids(run.conversation_id):
            if child_id not in self.session.conversations:
                continue
            live = self._runs.get(child_id)
            if live is not None:
                # A child still DRIVING from a previous run — a depth-0
                # wake-round failure leaves the turn open and re-raises while
                # the children keep working, so the next run() finds them
                # live. Adopt the existing handle: without the re-parent its
                # forwarded events, gate publications and lifecycle endings
                # would land on the dead handle's queues, and `run.child()`
                # would answer None for a conversation that is plainly part
                # of this run's subtree.
                live._parent = run
                run._adopt(live)
                continue
            if self.session.get_conversation_status(child_id).status is ConversationStatus.IDLE:
                continue  # finished — this run resolves it; there is nothing to drive
            child = AgentRun(
                self,
                conversation_id=child_id,
                streaming=run._streaming,
                on_event=None,  # the parent's callback sees it through the fan-in
                eager=True,
                defer_start=True,  # the pool decides when — a resume must not stampede past the cap
                autostart_subagents=True,
                parent=run,
            )
            run._adopt(child)
            self._request_slot(child_id, child._start_eager, handle=child, announce=child)

    def _compaction_ran(self, conversation_id: str) -> bool:
        return conversation_id in self._compaction_did_run

    def _rebind_run(self, previous: str, conversation_id: str) -> None:
        """Follow a conversation that a transition replaced: the live run, its
        wake and its recorded outcome all move to the successor's id."""
        run = self._runs.pop(previous, None)
        if run is not None:
            run.conversation_id = conversation_id
            self._runs[conversation_id] = run
        wake = self._wakes.pop(previous, None)
        if wake is not None:
            self._wakes[conversation_id] = wake
        self._closed_outcomes[conversation_id] = self._closed_outcomes.pop(previous, None)
        if previous in self._compaction_did_run:
            self._compaction_did_run.add(conversation_id)
        if previous in self._compaction_cancelled:
            self._compaction_cancelled.add(conversation_id)

    def _end_run(self, run: AgentRun) -> None:
        if self._runs.get(run.conversation_id) is run:
            del self._runs[run.conversation_id]
        # ANNOUNCE BEFORE RELEASING: releasing pumps the pool, and the pump
        # may start a queued sibling that announces its own start — "S1
        # paused, so S4 started" must arrive in that order on the stream.
        # Deliberately OUTSIDE the `_runs` guard — belt and suspenders: every
        # drive (a `_redrive`n one included) registers there now, but an
        # ending must announce and release even if the registration was
        # superseded (`_publish_run_ended` is latched per drive, so the extra
        # call sites cannot double-publish).
        self._publish_run_ended(run)
        self._release_slot(run.conversation_id)

    # ── waking a conversation out of band ───────────────────────────────────

    def _ensure_driven(self, conversation_id: str) -> None:
        """Something changed out of band for `conversation_id`; make sure a
        drive looks again.

        A live drive is WOKEN. A conversation whose drive already ended needs a
        RESTART rather than a signal — see `AgentRun._redrive`, which applies
        only where the FRAMEWORK owns that conversation's lifecycle. Where the
        application owns it, the recheck flag alone is enough: the app's next
        pull re-enters the loop and the flag is waiting there.

        `_runs` holds only DRIVING runs, so a subagent parked at its gate is
        reachable nowhere but its parent's subtree — which is exactly the handle
        this has to restart."""
        wake = self._wakes.get(conversation_id)
        if wake is not None:
            wake.set()
            return
        run = self._runs.get(conversation_id) or self._parked_handle(conversation_id)
        if run is None or not run._framework_owned:
            return
        if run._task is None or not run._task.done():
            # Never started — its FIRST start belongs to the pool, and a
            # `_redrive` here would spin a drive up with no token and no
            # bracket. Or still driving — nothing to restart.
            return
        # A restart is a start a consumer has to see, so it announces.
        self._request_slot(conversation_id, run._redrive, handle=run, announce=run)

    def _parked_handle(self, conversation_id: str) -> AgentRun | None:
        """A finished handle for `conversation_id`, found through the live
        runs' subtrees."""
        for run in list(self._runs.values()):
            found = run._lookup(conversation_id)
            if found is not None:
                return found
        return None

    # ── the subagent worker pool ─────────────────────────────────────────────
    #
    # `subagents_max_workers` bounds how many subagent conversations are DOING
    # WORK at the same time. The rule everything below follows: a slot is held
    # only while a conversation does its own productive work — never while it
    # waits for another conversation, waits for a human, or winds a cancelled
    # turn down. That rule is what makes a nested tree unable to deadlock: a
    # parent releases on park, so its children can always be admitted.

    def _slot_limit(self) -> int:
        return self.session.session_config.runtime_config.subagents_max_workers

    def _needs_slot(self, conversation_id: str) -> bool:
        """The main conversation never competes — the cap bounds subagent
        work and the main conversation must always be able to advance — and
        an unlimited pool never accounts."""
        return self._slot_limit() != Inf and self.session.conversations[conversation_id].depth > 0

    def _request_slot(
        self,
        conversation_id: str,
        grant: Callable[[], None],
        *,
        handle: AgentRun | None = None,
        announce: AgentRun | None = None,
        resume: bool = False,
    ) -> None:
        """Ask for a slot; granted synchronously if one is free."""
        waiter = _SlotWaiter(conversation_id, grant, handle=handle, announce=announce, resume=resume)
        if not self._needs_slot(conversation_id):
            self._grant(waiter)
            return
        self._waiters.append(waiter)
        self._pump_slots()

    def _grant(self, waiter: _SlotWaiter) -> None:
        """The ONE place a slot is handed over, so accounting, announcement
        and failure routing cannot drift apart."""
        if self._needs_slot(waiter.conversation_id):
            self._working.add(waiter.conversation_id)
        try:
            waiter.grant()
        except BaseException as exc:
            # A pool-granted start has no caller to raise into — the sibling
            # whose `finally` pumped the queue must not inherit the failure,
            # and the parent would otherwise park forever on a child whose
            # seed message still derives BUSY. Route it to the parent's drive.
            logger.error(
                "conv=%s subagent failed to start",
                waiter.conversation_id,
                exc_info=exc,
            )
            self._working.discard(waiter.conversation_id)
            if waiter.handle is not None:
                waiter.handle._abandoned = True
                waiter.handle._wake.set()  # a joiner parked on it re-checks
            self._admission_errors[waiter.conversation_id] = exc
            parent = waiter.handle._parent if waiter.handle is not None else None
            if parent is not None:
                parent._subtree_wake.set()
                parent._wake.set()
                self._ensure_driven(parent.conversation_id)
            return
        if waiter.announce is not None:
            self._publish_subagent_event(
                waiter.announce,
                SubagentStarted(conversation_id=waiter.conversation_id),
            )

    def _release_slot(self, conversation_id: str) -> None:
        """Give a slot back (a no-op for a conversation holding none) and
        hand it on."""
        self._working.discard(conversation_id)
        self._pump_slots()

    def _pump_slots(self) -> None:
        while self._waiters and (self._slot_limit() == Inf or len(self._working) < self._slot_limit()):
            waiter = self._waiters.pop(0)
            if not self._still_wants_slot(waiter):
                continue  # stale — drop it and take the next
            self._grant(waiter)

    def _still_wants_slot(self, waiter: _SlotWaiter) -> bool:
        """A waiter can go stale between enqueue and grant."""
        conversation_id = waiter.conversation_id
        if waiter.handle is not None and waiter.handle._abandoned:
            return False  # its tree was suspended before admission
        if waiter.handle is not None and waiter.handle._task is not None and not waiter.handle._task.done():
            return False  # already driving — a duplicate request
        if conversation_id not in self.session.conversations:
            return False
        if waiter.resume:
            # A parked drive re-acquiring: it is live by definition, so the
            # start-oriented checks below do not apply. It is stale only when
            # that drive is gone — ended or suspended mid-wait — in which
            # case granting would strand a slot nobody will ever release.
            return conversation_id in self._runs or self._wakes.get(conversation_id) is not None
        if conversation_id in self._runs:
            return False  # already driving — a restart path got there first
        if self.session.get_conversation_status(conversation_id).status is ConversationStatus.IDLE:
            return False  # already finished, or already flushed
        # A CANCELLED CONVERSATION IS NEVER GRANTED A SLOT. Its parent's
        # drive flushes it (`_flush_cancelled_children`) — the path that
        # already exists for a child with no drive. Starting one purely so it
        # could wind itself down would burn a slot on bookkeeping AND race
        # the parent for the same flush.
        return self.ledger.open_turn_cancel_requested(conversation_id) is None

    def _drop_waiter(self, conversation_id: str) -> None:
        self._waiters = [w for w in self._waiters if w.conversation_id != conversation_id]

    async def _acquire_slot(self, conversation_id: str, token: CancellationToken) -> bool:
        """Take (or retake) a slot before doing work. False only when the
        cancellation token won the race — the caller then proceeds without a
        slot, straight into a wind-down that needs none."""
        if not self._needs_slot(conversation_id):
            return True
        granted = asyncio.Event()
        # announce=None: resuming a parked drive is not a start.
        self._request_slot(conversation_id, granted.set, resume=True)
        if granted.is_set():
            return True
        waiter = asyncio.ensure_future(granted.wait())
        cancelled = asyncio.ensure_future(token.wait_cancelled())
        try:
            await asyncio.wait({waiter, cancelled}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            await _cancel_quietly(waiter)
            await _cancel_quietly(cancelled)
        if granted.is_set():
            return True
        self._drop_waiter(conversation_id)  # cancelled while queued
        return False

    def _pop_admission_error(self, conversation_id: str) -> BaseException | None:
        """The recorded failure of a pool-granted start among this
        conversation's unresolved children, if any — consumed by its drive."""
        for child_id in self._unresolved_child_ids(conversation_id):
            exc = self._admission_errors.pop(child_id, None)
            if exc is not None:
                return exc
        return None

    def _publish_subagent_event(self, run: AgentRun, event: AgentEvent) -> None:
        """Put a subagent lifecycle event on the stream that OWNS the
        subagent: the PARENT run's inbox — the same door the child's own
        events already ride (`AgentRun._forward`), so it bubbles to the root
        handle like any other event. `put_nowait` is synchronous and never
        blocks, which is what lets this be called from `_end_run` and
        `_grant`, neither of which runs inside the drive generator."""
        parent = run._parent
        if parent is None:
            return
        parent._inbox.put_nowait(event)

    def _publish_run_ended(self, run: AgentRun) -> None:
        """Announce a framework-driven subagent's drive ending — a pause (its
        turn is still open; it will run again) or a finish (its turn closed,
        with the outcome the close recorded). Latched per drive, so the extra
        `_end_run` call sites cannot publish one ending twice."""
        if run._ended_published:
            return
        run._ended_published = True
        if run._task is None:
            return  # never actually drove — the failed-start unwind path
        if run._parent is None or not run._framework_owned:
            return  # not a framework-driven subagent: no lifecycle events
        status = self.session.get_conversation_status(run.conversation_id).status
        if status is ConversationStatus.IDLE:
            outcome = self._closed_outcomes.get(run.conversation_id) or TurnOutcome.COMPLETED
            ended: AgentEvent = SubagentFinished(conversation_id=run.conversation_id, outcome=outcome)
        else:
            ended = SubagentPaused(conversation_id=run.conversation_id)
        self._publish_subagent_event(run, ended)

    def _ensure_open_turn(self, conversation_id: str) -> None:
        """Open a new bracket unless one is already open (resume). Called at
        the engine's top for lazy runs and synchronously at `start()` time for
        eager runs — a started run is cancellable before its first tick."""
        if self.ledger.open_turn_index(conversation_id) is None:
            self._append(
                conversation_id,
                lambda entry_id, parent_id, ts: TurnStart(
                    id=entry_id,
                    parent_id=parent_id,
                    created_at=ts,
                ),
            )

    def _open_bracket_for_start(self, conversation_id: str) -> None:
        """`start()`'s call-time bracket decision: a COMPACTION bracket when
        one is due, otherwise the ordinary `TurnStart`.

        Only the MAIN conversation is ever compaction-checked (a subagent's is
        not, in V0), so a child's eager start just opens its turn.

        This is why `should_compact` is sync. `AgentRun.__init__` opens a
        bracket synchronously, so without this an eager run would always
        present the drive with an open conversational turn — and the drive's
        compaction step skips when it finds one, so a policy-driven compaction
        would never fire for an eager run at all. An already-scheduled
        compaction needs nothing special: its bracket is open, so
        `_ensure_open_turn` is already a no-op.

        A `should_compact` that raises propagates from `start()` — swallowing
        it would make a broken manager indistinguishable from one that
        declines, and nothing durable exists yet to record the failure on."""
        due = (
            conversation_id == self.main_conversation_id
            and self.ledger.open_turn_index(conversation_id) is None
            and self.context_manager.should_compact(self.session, conversation_id)
        )
        if due:
            self._open_compaction_bracket(conversation_id, CompactionSource.POLICY)
            return
        self._ensure_open_turn(conversation_id)

    def _build_run_result(self, conversation_id: str) -> RunResult:
        """Snapshot where the engine stopped: the DERIVED status, plus how the
        last bracket this run closed ended (None if it closed none).

        The outcome cannot be read off the path any more. After a successful
        compaction the closing `TurnFinish` is on the ARCHIVED conversation,
        and after a compaction-only drive the leaf may be the `CompactionEntry`
        itself or a carried assistant message — neither has an outcome to read.

        `pending_approvals` is reported whenever it is non-empty, rather than
        only in one status: a gate no longer implies a status of its own, so
        the list IS the signal."""
        return RunResult(
            status=self.session.get_conversation_status(conversation_id).status,
            outcome=self._closed_outcomes.get(conversation_id),
            pending_approvals=self.pending_approvals(conversation_id),
        )

    # ── middleware machinery ─────────────────────────────────────────────────

    def _run_middlewares(
        self,
        method_name: str,
        conversation_id: str,
        value,
        *ctx_args,
        unpack_values: bool = False,
        **ctx_kwargs,
    ):
        """Thread `value` through each middleware's `method_name` hook in order.

        Every hook is called `(session, conversation_id, value, *ctx)`. The
        session comes off `self` — it is the LIVE object the runner and ledger
        write through, so no call site passes it. `conversation_id` is the
        scope of the OPERATION that invoked the hook, which is what makes one
        middleware instance safe to share across the main conversation and its
        subagents; it does not assert that `value` belongs exclusively to that
        conversation.

        Context args/kwargs are forwarded unchanged to every call. With
        `unpack_values=True`, `value` is a tuple unpacked as positional args
        (used for `before_llm_call` which takes and returns a pair)."""
        for mw in self.middleware:
            if not hasattr(mw, method_name):
                continue
            scope = (self.session, conversation_id)
            if unpack_values:
                value = getattr(mw, method_name)(*scope, *value, *ctx_args, **ctx_kwargs)
            else:
                value = getattr(mw, method_name)(*scope, value, *ctx_args, **ctx_kwargs)
        return value

    def _complete_entry(self, conversation_id: str, entry: AnyEntry) -> AnyEntry:
        """The fallible half of preparing any entry: calculate its
        `context_tokens` (context calculation is part of preparing a complete
        entry — it runs BEFORE middleware, and never again after), then thread
        it through `before_entry_written` under the conversation whose
        operation caused the write."""
        entry.context_tokens = self.context_manager.calculate_context(
            self.session,
            entry,
        )
        return self._run_middlewares("before_entry_written", conversation_id, entry)

    def _append(self, conversation_id: str, build_fn) -> AnyEntry:
        """Append one entry: build it, complete it, commit it to the ledger."""
        return self.ledger.append(
            conversation_id,
            lambda entry_id, parent_id, ts: self._complete_entry(
                conversation_id,
                build_fn(entry_id, parent_id, ts),
            ),
        )

    def _complete_uncommitted(
        self,
        conversation_id: str,
        build_fn,
        parent_id: str | None,
        ts: int,
    ) -> AnyEntry:
        """A complete, fully-middlewared entry that is NOT committed — the
        other half of `_append`, for entries that belong to a conversation
        which does not exist yet (a compaction plan's). Identity comes from
        the same hooks; the caller supplies the parent, because the new
        conversation's leaf is not the active path's.

        `conversation_id` is therefore the OUTGOING conversation — the one
        being compacted, whose drive caused the write. The destination's id is
        minted inside `ledger.transition_conversation` and does not exist yet,
        and the hook's contract is the operation's scope rather than the
        entry's eventual home.

        (Named for what it does rather than `_prepare`, so `prepare` in this
        file means the tool-lifecycle phase and nothing else.)"""
        return self._complete_entry(
            conversation_id,
            build_fn(self.generate_id(), parent_id, ts),
        )

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
        the old basis. It sets no other field.

        NO MIDDLEWARE RUNS HERE. `before_entry_written` is scoped to the
        conversation whose operation caused a write, and this method rewrites
        every entry across every conversation at once — there is no single id
        that honestly describes the operation, and inventing one (or admitting
        `None` into a contract built on concrete ids) costs more than the hook
        is worth. This is an operational refresh of a derived estimate, not a
        write with a scope; `ledger.refresh_entry` is the matching door, and
        it takes no conversation for the same reason.

        NOTHING IN THE FRAMEWORK CALLS THIS. There is no constructor keyword,
        no CLI flag, and no automatic invocation on a model switch (which
        would put an unbounded rewrite behind an innocuous-looking
        assignment). The shipped `ContextManager` is a character estimate that
        no model choice affects; this exists for the application that swaps in
        a real tokenizer, and that application calls it."""
        for entry_id in list(self.session.entries):
            refreshed = self.session.entries[entry_id].model_copy()
            refreshed.context_tokens = self.context_manager.calculate_context(
                self.session,
                refreshed,
            )
            self.ledger.refresh_entry(refreshed)

    def _persist_entry(
        self,
        conversation_id: str,
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
            self._complete_entry(conversation_id, updated)
            if recalculate
            else self._run_middlewares("before_entry_written", conversation_id, updated)
        )
        return self.ledger.put_entry(conversation_id, updated)

    def _persist_execution(
        self,
        conversation_id: str,
        execution: ToolExecution,
        *,
        recalculate: bool = True,
        **changes,
    ) -> ToolExecution:
        """`_persist_entry` plus the `ToolExecution`-only `updated_at` stamp.
        Every execution persistence — approval updates, the RUNNING
        transition, cancellation stamps, terminal outcomes — lands here."""
        return self._persist_entry(
            conversation_id,
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

    def _tool_executed_event(self, conversation_id: str, execution: ToolExecution) -> ToolExecuted:
        """Project the terminal execution once and derive the event's
        presentation fields from it. The next LLM request re-projects the same
        durable execution (projection is deterministic), so event and wire
        always agree."""
        message = self.conversation_projector.project_tool_execution(
            execution,
            self.session.entries,
        )
        return ToolExecuted(
            conversation_id=conversation_id,
            tool_call_id=execution.tool_call_id,
            execution=execution.model_copy(deep=True),
            result_text=tool_message_text(message),
            is_error=message.is_error,
        )

    def _finalize_outcome(
        self,
        conversation_id: str,
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
            conversation_id,
            execution,
            exception,
        )
        execution = self._persist_execution(conversation_id, execution, recalculate=False)
        return execution, self._tool_executed_event(conversation_id, execution)

    def _recover_orphans(self, conversation_id: str) -> list[AgentEvent]:
        """A persisted RUNNING execution without its live runtime task is an
        orphan (a crash, or a drive suspended mid-body). Transition it to
        INTERRUPTED: `after_tool_execution` runs with no exception,
        `before_tool_execution` is NOT re-invoked, and the tool is never
        automatically re-dispatched. Durable state records nothing
        crash-specific — an orphan is exactly another INTERRUPTED execution."""
        events: list[AgentEvent] = []
        for execution in self.ledger.open_turn_running_executions(conversation_id):
            terminal = execution.model_copy(
                update={
                    "status": ExecutionStatus.INTERRUPTED,
                    "ended_at": self.now_ms(),
                },
            )
            _, event = self._finalize_outcome(conversation_id, terminal)
            events.append(event)
        return events

    # ── per-call build methods ───────────────────────────────────────────────

    def build_model_string(self, conversation_id: str, llm_cfg: LLMConfig) -> str:
        """Build the model identifier for the LLM client, threading it through
        any `build_model_string` middleware. Called per LLM invocation.

        Conversation-scoped so a middleware can route per conversation — a
        cheap model for subagents, the configured one for the main
        conversation."""
        model_string = f"{llm_cfg.provider}:{llm_cfg.model}"
        return self._run_middlewares("build_model_string", conversation_id, model_string, llm_cfg)

    async def resolve_tool_specs(self, conversation_id: str) -> list[ToolSpec]:
        """The registry's answer for this conversation — the RUNTIME's view of
        what tools exist, private ones included.

        `get_tools` is dynamic: the runner calls it fresh per LLM call, the
        result may vary with session state, and it may legitimately vary BY
        CONVERSATION (a registry withholding a tool from a subagent). A
        toolless runner contributes none.

        Split from `build_tool_list` because the two answer different
        questions. This is what the runtime may resolve and dispatch; that is
        what the model is shown. A private spec appears here and not there, and
        the runner needs BOTH lists in the same iteration — the private names
        to refuse a model call that guesses one."""
        if self.tool_registry is None:
            return []
        return await self.tool_registry.get_tools(self.session, conversation_id)

    def build_tool_list(
        self,
        conversation_id: str,
        specs: list[ToolSpec],
    ) -> list[ToolSpec]:
        """The MODEL-VISIBLE catalog: private specs dropped, the rest threaded
        through any `build_tool_list` middleware. Called per LLM invocation.

        This is the ONE place `is_private` is enforced on the way out, and the
        filter stays AHEAD of the hook — a middleware never sees a private
        tool, which is the point. (A middleware that mints a spec with
        `is_private=True` therefore puts it on the wire: the filter has already
        run. That is the trust model working, not a gap to close.)

        Returns `ToolSpec`s, not wire tools. The spec is the core's own tool
        type and the one the registry answers with; the wire `luca.client.Tool`
        drops `tool_kind`, `namespace`, `is_private`, `output_schema` and
        `metadata`, so a middleware handed the adapted list could not filter on
        any of them. `_collect_tools` adapts what this returns.

        Synchronous: the awaiting moved to `resolve_tool_specs`. The drive
        races the pair together rather than either alone, so a subclass
        overriding either is covered for free and the cancellation token never
        has to appear in these public signatures."""
        visible = [spec for spec in specs if not spec.is_private]
        return self._run_middlewares("build_tool_list", conversation_id, visible)

    def _spawn_gate_open(self, conversation_id: str, *, exclude: str | None = None) -> bool:
        """May THIS conversation spawn subagents?

        The predicate the gate and the system prompt must both derive from —
        a static prompt telling a subagent at the depth cap that it can spawn,
        while the tool list withholds the tool, would have the model try.

        `exclude` drops one execution id from the per-turn count. `_spawn_one`
        needs it: converting a settled spawn re-checks this gate at a point
        where that execution is already counted, and its question is "was
        there room for this one?", not "is there room for another?"."""
        config = self.session.session_config.runtime_config
        if not config.subagents_enabled:
            return False
        depth_cap = config.subagents_max_depth
        if depth_cap != Inf and self.session.conversations[conversation_id].depth >= depth_cap:
            return False
        # the per-turn budget: Inf by default, so this clause never fires for
        # a session that did not ask for a limit
        limit = config.subagents_max_per_turn
        return limit == Inf or spawns_committed(self.session, conversation_id, exclude=exclude) < limit

    def _verify_gate(self, conversation_id: str, specs: list[ToolSpec]) -> None:
        """A registry that returned a spawn-declaring spec for a conversation
        at or past the cap has violated the contract. RAISE rather than quietly
        filter it out: that keeps `subagents_max_depth` a hard guarantee the
        implementation may rely on, and it surfaces here — before the model
        call — naming the spec. The alternative failure is invisible: the spec
        is offered, the model calls it, the handshake fires, and a grandchild
        appears with nothing in the log explaining why."""
        if self._spawn_gate_open(conversation_id):
            return
        leaked = [spec.name for spec in specs if declares_spawn(spec)]
        if leaked:
            raise AgentError(
                f"tool registry offered spawn-declaring tool(s) {leaked} to conversation "
                f"{conversation_id!r}, which may not spawn subagents "
                f"(subagents_enabled="
                f"{self.session.session_config.runtime_config.subagents_enabled}, "
                f"depth={self.session.conversations[conversation_id].depth}, "
                f"max_depth={self.session.session_config.runtime_config.subagents_max_depth}, "
                f"spawns_committed={spawns_committed(self.session, conversation_id)}, "
                f"max_per_turn={self.session.session_config.runtime_config.subagents_max_per_turn})."
            )

    async def _collect_tools(
        self,
        conversation_id: str,
    ) -> tuple[list[ToolSpec], list[LucaTool]]:
        """Both lists, resolved once. The result stays a LOCAL in the drive and
        is never stashed on `self`: per-call state on a runner driving several
        conversations belongs to whichever one wrote it last.

        The gate runs on the REGISTRY's answer, before the middleware hook: it
        checks a registry-contract violation, not an application one.
        Adaptation to client tool DTOs happens here, after the spec-level hook,
        so that contract stays in the core's own vocabulary — and the adapted
        list then runs through `adapt_tool_declarations`, the CLIENT-vocabulary
        slot, where a middleware swaps a function declaration for a
        provider-native declaration item (dropping belongs in
        `build_tool_list`, which sees the richer spec fields)."""
        specs = await self.resolve_tool_specs(conversation_id)
        self._verify_gate(conversation_id, specs)
        visible = self.build_tool_list(conversation_id, specs)
        tools = [adapter.tool_spec_to_luca_tool(spec) for spec in visible]
        return specs, self._run_middlewares("adapt_tool_declarations", conversation_id, tools)

    def build_messages(self, conversation_id: str) -> list:
        """Project the driven conversation's path to canonical client messages
        via the configured `ConversationProjector` — derived per call, never
        stored. History-shaping policy belongs on the projector itself (there
        is no projection middleware); `before_llm_call` remains downstream for
        last-mile request changes."""
        return self.conversation_projector.project(
            self.session.conversations[conversation_id].nodes,
            self.session.entries,
        )

    def build_system_message(self, conversation_id: str) -> str | None:
        """Assemble the system prompt for one LLM call: resolve the parts
        (a callable part is invoked with the live session and the id of the
        conversation the prompt is for, its return value coerced like a static
        part), drop the ones that contributed nothing, sort by priority,
        assemble. A blank result means no system message is sent at all."""
        resolved = [
            coerce_system_prompt_part(part(self.session, conversation_id)) if callable(part) else part
            for part in self.system_prompt_parts
        ]
        parts = sorted((part for part in resolved if part is not None), key=lambda part: part.priority)
        prompt = self.system_prompt_assembler.assemble_system_prompt(parts)
        return prompt if prompt.strip() else None

    def prepare_llm_call(self, conversation_id: str) -> tuple[list, str | None]:
        """Build the (messages, system_message) pair for the next LLM call.
        Calls `build_messages()` and `build_system_message()`, then threads
        the pair through any `before_llm_call` middleware."""
        messages = self.build_messages(conversation_id)
        system_message = self.build_system_message(conversation_id)
        return self._run_middlewares(
            "before_llm_call",
            conversation_id,
            (messages, system_message),
            unpack_values=True,
        )

    # ── compaction ───────────────────────────────────────────────────────────

    def _open_compaction_bracket(
        self,
        conversation_id: str,
        source: CompactionSource,
    ) -> CompactionEntry:
        """`TurnStart` then `CompactionEntry` — two plain appends, exactly as
        safe as recording a user message, and adjacent by construction (which
        is what makes the compaction-bracket predicate exact). Emits nothing;
        the drive emits `CompactionScheduled`."""
        self._append(
            conversation_id,
            lambda entry_id, parent_id, ts: TurnStart(
                id=entry_id,
                parent_id=parent_id,
                created_at=ts,
            ),
        )
        return self._append(
            conversation_id,
            lambda entry_id, parent_id, ts: CompactionEntry(
                id=entry_id,
                parent_id=parent_id,
                created_at=ts,
                source=source,
            ),
        )

    def _snapshot_conversation(self, conversation_id: str) -> ConversationSnapshot:
        """The full active path (G2's input) plus the view handed to the
        manager: the same path with THIS compaction's `TurnStart` removed.

        Exactly one element, positionally — the tail is `[…, ts_c, cmp]` by
        construction (the two appends are consecutive, `post_message` raises
        while the bracket is open, and a parked cancel flushes before
        `compact()` is ever reached), so this is a removal, not a filter over
        types. `cmp` deliberately STAYS in the view: stripping it too would
        make `plan.nodes = list(nodes)` illegal, trading one invisible
        requirement for another."""
        conversation = self.session.conversations[conversation_id]
        nodes = tuple(conversation.nodes)
        bracket = self.ledger.open_turn_index(conversation_id)
        return ConversationSnapshot(
            id=conversation.id,
            nodes=nodes,
            offered=nodes[:bracket] + nodes[bracket + 1 :],
        )

    async def _compaction_step(
        self,
        conversation_id: str,
        token: CancellationToken,
    ) -> AsyncIterator[AgentEvent]:
        """The whole compaction operation, run once at the top of a drive,
        BEFORE the conversational bracket opens.

        Flush a parked cancel first; then resume an interrupted compaction,
        skip entirely (an open conversational turn — an approval pause, a
        crash mid-turn, a suspended run — is never carved up by a manager), or
        ask the manager whether one is due; then run it. The bracket never
        stays open past the end of a drive that did not crash.

        Everything expensive and failure-prone happens BEFORE the transition,
        which is what makes the operation safe by construction rather than by
        recovery logic: a failure here closes the bracket and leaves the
        conversation exactly as it was, and a crash leaves the bracket open —
        the resumable state."""
        entry = self.ledger.open_compaction_entry(conversation_id)

        # 1) FLUSH FIRST. A parked cancel inside an open compaction bracket
        #    ends it now — no Scheduled, no Started, no `compact()` call.
        if entry is not None:
            cancel = self.ledger.open_turn_cancel_requested(conversation_id)
            if cancel is not None:
                self._compaction_cancelled.add(conversation_id)
                yield self._close_compaction(conversation_id, entry, cancel.outcome, cancel.error)
                return

        # 2) RESUME, SKIP, or DECIDE. An open bracket around an entry that
        #    already has `parts` is NOT resumable (G6) — the ledger's read
        #    answers None for it, so it lands in the skip branch below and the
        #    drive treats it as the phantom conversational turn it is.
        if entry is None:
            if self.ledger.open_turn_index(conversation_id) is not None:
                return  # an open conversational turn → not this drive
            if not self.context_manager.should_compact(self.session, conversation_id):
                return
            entry = self._open_compaction_bracket(conversation_id, CompactionSource.POLICY)
        self._compaction_did_run.add(conversation_id)  # past every early return
        yield CompactionScheduled(
            conversation_id=conversation_id,
            entry=entry.model_copy(deep=True),
        )

        # 3) RUN IT. `started_at` is stamped once, on the first attempt, so a
        #    resumed compaction keeps the original stamp and the previous
        #    attempt stays visible in the log.
        if entry.started_at is None:
            entry = self._persist_entry(conversation_id, entry, started_at=self.now_ms())
        yield CompactionStarted(
            conversation_id=conversation_id,
            entry=entry.model_copy(deep=True),
        )

        snapshot = self._snapshot_conversation(conversation_id)
        try:
            plan = await self._invoke_compaction(conversation_id, entry, snapshot.offered, token)
        except _CompactionEnded as stop:
            # A cancellation always wins over the deadline that raced it, and
            # always stops the drive: the cancel is against the drive, not
            # against the compaction alone. `cancel()` writes the durable
            # request BEFORE tripping the token, so a token-fired ending
            # always finds one here and only a deadline reaches the lines
            # below — which is why they raise the same timeout the
            # conversational LLM call raises.
            cancel = self.ledger.open_turn_cancel_requested(conversation_id)
            if cancel is not None:
                self._compaction_cancelled.add(conversation_id)
                yield self._close_compaction(conversation_id, entry, cancel.outcome, cancel.error)
                return
            yield self._close_compaction(conversation_id, entry, stop.outcome, stop.error)
            if entry.source == CompactionSource.USER:
                raise ClientTimeoutError(stop.error) from None
            return
        except Exception as exc:
            # A POLICY failure is swallowed below, so this log line is the only
            # record of it anywhere.
            logger.error(
                "conv=%s compaction failed during planning (source=%s)",
                conversation_id,
                entry.source.value,
                exc_info=exc,
            )
            yield self._close_compaction(conversation_id, entry, _outcome_for(exc), str(exc))
            if entry.source == CompactionSource.USER:
                raise
            return  # a POLICY failure DEGRADES — the user's turn survives

        if plan is not None:
            # G2 first, then the usage write, both inside the failure
            # handling. G2 first because a manager that replaced the active
            # conversation would otherwise make `record_usage` raise its own
            # unrelated error and pre-empt the plan rejection; guarded because
            # an escape here would leave the bracket OPEN — the "resume me"
            # state — and the next drive would replay the same failing call
            # with `should_compact` never consulted.
            try:
                # Re-resolve the NAME rather than trusting the id this drive
                # started with: G2 asks "is the conversation I compacted still
                # the one that is named?", and a manager that installed a
                # successor under us has to be caught here.
                check_snapshot(
                    session=self.session,
                    conversation_id=self.main_conversation_id,
                    snapshot=snapshot,
                )
                self.ledger.record_usage(conversation_id, entry.id, **plan.usage.model_dump())
            except Exception as exc:
                logger.error(
                    "conv=%s compaction plan rejected (source=%s)",
                    conversation_id,
                    entry.source.value,
                    exc_info=exc,
                )
                yield self._close_compaction(conversation_id, entry, TurnOutcome.ERRORED, str(exc))
                if entry.source == CompactionSource.USER:
                    raise
                return

        # A cancel that arrived within the grace window DISCARDS the plan.
        # Unlike the LLM path, which records a within-grace answer: adding a
        # node is not rewriting what the conversation is.
        cancel = self.ledger.open_turn_cancel_requested(conversation_id)
        if cancel is not None:
            self._compaction_cancelled.add(conversation_id)
            yield self._close_compaction(conversation_id, entry, cancel.outcome, cancel.error)
            return

        if plan is None:  # the ONE "nothing to do" signal
            yield self._close_compaction(conversation_id, entry, TurnOutcome.COMPLETED)
            return

        try:
            conversation, final, created = self._commit(conversation_id, entry, plan, snapshot)
        except Exception as exc:
            logger.error(
                "conv=%s compaction commit failed (source=%s)",
                conversation_id,
                entry.source.value,
                exc_info=exc,
            )
            yield self._close_compaction(conversation_id, entry, TurnOutcome.ERRORED, str(exc))
            if entry.source == CompactionSource.USER:
                raise
            return

        self._closed_outcomes[conversation_id] = TurnOutcome.COMPLETED  # _close_turn never ran
        yield CompactionFinished(
            conversation_id=conversation_id,
            entry=final.model_copy(deep=True),
            outcome=TurnOutcome.COMPLETED,
            created=[e.model_copy(deep=True) for e in created],
            new_conversation_id=conversation.id,
        )

    async def _invoke_compaction(
        self,
        conversation_id: str,
        entry: CompactionEntry,
        offered: tuple[str, ...],
        token: CancellationToken,
    ) -> CompactionPlan | None:
        """Call `ContextManager.compact()` under the run's cancellation race and
        the session's wall-clock deadline.

        The runner races the call itself: `client_completion_timeout_in_ms` is
        a kwarg the runner passes to the client, and it cannot reach a request
        the manager makes on its own. The value is converted exactly as the LLM
        step converts it, so the default (`Inf`) means NO deadline at all —
        a hung manager hangs the drive until cancelled, identical to the
        conversational call's default.

        The manager is handed a DEEP COPY of the entry and the LIVE session.
        The copy is load-bearing: a manager that wrote `parts` onto the live
        entry and then failed would leave the bracket closed ERRORED, the path
        unchanged — and the entry projecting a summary of nothing."""
        config = self.session.session_config.runtime_config
        grace_ms = config.llm_completion_cancellation_grace_period
        deadline = _ms_to_seconds(config.client_completion_timeout_in_ms)
        task = asyncio.ensure_future(
            self.context_manager.compact(
                self.session,
                conversation_id,
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
                    raise  # the manager's OWN TimeoutError — a normal raise
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
        conversation_id: str,
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
        final = self._persist_entry(conversation_id, entry, ended_at=self.now_ms())
        self._close_turn(conversation_id, outcome, error)
        return CompactionFinished(
            conversation_id=conversation_id,
            entry=final.model_copy(deep=True),
            outcome=outcome,
            error=error,
            created=[],
            new_conversation_id=None,
        )

    def _commit(
        self,
        conversation_id: str,
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
            conversation_id=self.main_conversation_id,
            snapshot=snapshot,
        )
        ts = self.now_ms()  # ONE timestamp for the whole transition
        # The bracket is still open — the closing marker is only BUILT below —
        # and G2 has just proven this is the same index `_snapshot_conversation`
        # stripped at, so the offered view and the compacted span can never
        # disagree about where the bracket is.
        bracket = self.ledger.open_turn_index(conversation_id)
        carried = {node for node in plan.nodes if isinstance(node, str)}
        # Over the path BEFORE the bracket, so the compaction's own `ts_c`
        # never lands in the list of ids this entry replaced.
        compacted = [node for node in snapshot.nodes[:bracket] if node not in carried]

        final = self._complete_entry(
            conversation_id,
            entry.model_copy(
                update={
                    "parts": plan.entry.parts,
                    "llm_config": plan.entry.llm_config,
                    "metadata": plan.entry.metadata,
                    "compacted_nodes": compacted,
                    "ended_at": ts,
                },
            ),
        )
        nodes: list[str] = []
        created: list[AnyEntry] = []
        parent: str | None = None
        for node in plan.nodes:
            if isinstance(node, str):
                parent = node
            else:
                built = self._complete_uncommitted(
                    conversation_id,
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
            conversation_id,
            lambda entry_id, parent_id, stamp: TurnFinish(
                id=entry_id,
                parent_id=parent_id,
                created_at=stamp,
                outcome=TurnOutcome.COMPLETED,
            ),
            self.session.conversations[conversation_id].nodes[-1],
            ts,
        )
        # ─────────────────────────── THE TRANSITION ──────────────────────────
        conversation = self.ledger.transition_conversation(
            conversation_id,
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
        conversation_id: str,
        streaming: bool,
        token: CancellationToken,
    ) -> AsyncIterator[AgentEvent]:
        """The single engine behind both methods, driving ONE conversation;
        `AgentRun` is its only consumer (lazy pulls it directly, eager drains
        it from the background task)."""
        # Compaction runs BEFORE the conversational bracket, at most once per
        # drive (structural: the step sits outside the loop below).
        async for event in self._compaction_step(conversation_id, token):
            yield event
        if conversation_id in self._compaction_cancelled:
            # The cancel was against the DRIVE. Consuming it and then going on
            # to answer the queued turn would defy the instruction.
            return
        compacted = self._compaction_ran(conversation_id)
        if compacted:
            # A committed transition installed a NEW conversation and
            # re-pointed whatever named the old one, so re-resolve — and move
            # the HANDLE with it, since a run drives a role ("the main
            # conversation") and a transition is precisely the moment that role
            # changes rows. ONLY here: a compaction is the one thing that
            # replaces a conversation, and only the main conversation is ever
            # compacted, so re-resolving unconditionally would silently switch
            # a subagent's drive onto the main conversation.
            previous, conversation_id = conversation_id, self.main_conversation_id
            if previous != conversation_id:
                self._rebind_run(previous, conversation_id)
        if compacted and self.status == ConversationStatus.IDLE:
            # A compaction-only drive: a user-scheduled compaction on an
            # otherwise-finished session was drivable only because its bracket
            # was open. With it closed there is nothing to drive, and opening
            # a turn would call the model with no user input. Gated on the
            # step having done something, so every other drive is provably
            # unchanged.
            return

        # Resume the open turn if one exists; otherwise open a new bracket
        # (an eager run already opened it at start() time).
        self._ensure_open_turn(conversation_id)

        # Crash recovery: any persisted RUNNING execution has no live task on
        # a fresh drive — terminalize it as INTERRUPTED before anything else
        # (before the flush too, so a parked cancel never CANCELLs a call
        # whose body actually started).
        for event in self._recover_orphans(conversation_id):
            yield event

        # A live drive can be woken out of band — by `run.notify()`, or by a
        # subagent of this conversation finishing. Registered for the whole
        # loop and removed in the `finally` below, so `_ensure_driven` can tell
        # "wake it" from "restart it".
        wake = self._wakes.setdefault(conversation_id, asyncio.Event())
        try:
            async for event in self._drive_loop(conversation_id, streaming, token, wake):
                yield event
        finally:
            for cid, registered in list(self._wakes.items()):
                if registered is wake:
                    del self._wakes[cid]

    async def _drive_loop(
        self,
        conversation_id: str,
        streaming: bool,
        token: CancellationToken,
        wake: asyncio.Event,
    ) -> AsyncIterator[AgentEvent]:
        """The conversational loop itself. Split from `_drive` only so the
        wake registration has a `finally` to be removed in."""
        # Execution ids already announced through `ApprovalRequired` in THIS
        # drive. A set rather than a bool since 0008: a fall-through round can
        # mint a NEW gated call while the first gate is still open, and the
        # new gate must be announced — while a re-park at the same gate must
        # not repeat the event. Cleared when the awaiting set empties, exactly
        # as the old boolean reset.
        announced_gates: set[str] = set()
        # The path length at the instant of THIS DRIVE's most recent
        # projection — v1's close-site fingerprint, now also consulted by the
        # subagent park (step 3c): a message posted while an LLM call is in
        # flight lands BEFORE the recorded assistant entry, so the durable
        # material predicate cannot see it and only this can. None until this
        # drive has projected once.
        last_seen: int | None = None
        while True:
            # 0) Cancel check — every step boundary funnels back here. An
            # unconsumed CancelRequested ends the turn NOW; the same path is
            # the parked-cancel FLUSH (run()/start() on a CANCELLING session),
            # which may legitimately emit zero events.
            cancel_entry = self.ledger.open_turn_cancel_requested(conversation_id)
            if cancel_entry is not None:
                for event in await self._wind_down_async(conversation_id, cancel_entry):
                    yield event
                return

            # 0b) A pool-granted start that failed after the spawn batch
            # returned has no caller of its own — the failure was recorded
            # against the child and this drive re-raises it, preserving the
            # fail-loud contract a synchronous start had.
            admission_error = self._pop_admission_error(conversation_id)
            if admission_error is not None:
                raise admission_error

            # 1) Undecided executions → THE decide() call site. Serves the
            # fresh path (created one iteration ago) and every resume path (a
            # re-entered run, a reloaded session) identically. A registry
            # response updates approval_status directly and lands in the
            # audit log; a DENY is terminal on the spot. All decision writes
            # land before any denial event is yielded.
            # CLEAR FIRST, THEN READ. A `notify()` landing while `decide()` is
            # in flight has to survive it and cause another pass; consuming the
            # flag after the fact would swallow exactly that signal and leave
            # an answered gate sitting inert.
            self._recheck.discard(conversation_id)
            wake.clear()

            # 1a) Unborn executions → THE create_execution() call site. They
            # were appended synchronously with the assistant message that asked
            # for them (`_receive_executions`), so the registry is consulted
            # here, one step later, against durable state — which is what makes
            # the birth resumable and what keeps the path projectable while it
            # is in flight.
            received = self.ledger.open_turn_received_executions(conversation_id)
            if received:
                for event in await self._birth_executions(conversation_id, received, token):
                    yield event
                continue  # → the cancel check, then decide() on the newly born

            undecided = self.ledger.open_turn_undecided_executions(conversation_id)
            if undecided:
                pairs = await asyncio.gather(*(self._decide_one(conversation_id, ex, token) for ex in undecided))
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
                    persisted = self._persist_execution(conversation_id, modified, **changes)
                    if decision.decision == ApprovalOption.PENDING:
                        self._publish_approval(conversation_id, persisted)
                    if denied:
                        _, event = self._finalize_outcome(conversation_id, persisted)
                        denial_events.append(event)
                for event in denial_events:
                    yield event

            # 2) Dispatch every ALLOWED-and-unrun execution. An allowed
            # sibling proceeds even while another call sits deferred — the
            # runner parks only after all currently runnable work advanced.
            ready = self.ledger.open_turn_ready_executions(conversation_id)
            if ready:
                async for event in self._dispatch_batch(conversation_id, ready, token):
                    yield event

            # 2b) THE SPAWN HANDSHAKE. A completed execution whose structured
            # payload says it spawned becomes a child conversation. The
            # `ChildConversation` entry is the durable record that the spawn was
            # handled, so this is idempotent across a reload for free.
            spawned = list(self._spawn_children(conversation_id))
            if spawned:
                announcement = SubagentsSpawned(
                    conversation_id=conversation_id,
                    conversation_ids=[entry.conversation_id for entry in spawned],
                )
                run = self._runs.get(conversation_id) or self._parked_handle(conversation_id)
                if run is not None and run.autostart_subagents:
                    # THE INBOX, NOT A YIELD, in framework mode. The fan-in
                    # drains the inbox before the engine on every pull, so a
                    # fast child's forwarded events would overtake an
                    # engine-yielded announcement (and its Started, parked in
                    # this generator, would arrive after the child's own
                    # Finished). Putting the announcement here, synchronously,
                    # BEFORE the starts — which publish onto the same queue
                    # before any child task has had a chance to run — is what
                    # guarantees Spawned ≺ Started(child) ≺ that child's own
                    # events, in every consumption mode.
                    run._inbox.put_nowait(announcement)
                    self._start_children(conversation_id, spawned)
                else:
                    # Application-driven children: the yield IS the transfer
                    # of control — the consumer drives every child inside
                    # this event's branch, and the handles have to exist by
                    # then or `run.child(cid)` returns None. No lifecycle
                    # events exist in this mode.
                    self._start_children(conversation_id, spawned)
                    yield announcement

            # 2b') THE STOP HANDSHAKE. A completed execution whose structured
            # payload declares `is_subagent_stop` cancels the named DIRECT
            # child — but only one spawned BEFORE the stop was issued (the
            # path-position bound), so a later child reusing the task id can
            # never be killed by an old signal, and a reload replays safely.
            # Cancelling is the whole action: the child's own machinery
            # (wind-down, the flush below, resolution) does the rest, and a
            # target already finished or already cancelling makes this a
            # no-op.
            self._stop_children(conversation_id)

            # 2c) RESOLVE FINISHED CHILDREN. A child is finished when its turn
            # bracket closes, whatever the outcome — a child that failed is a
            # finished child whose result says so, never an exception
            # travelling upward. A child that was CANCELLED with no drive left
            # is closed here first, since nothing else ever will — unless THIS
            # conversation is winding down too, in which case its own wind-down
            # owns every link and resolves them without a result tool at all.
            if self.ledger.open_turn_cancel_requested(conversation_id) is None:
                for event in self._flush_cancelled_children(conversation_id):
                    yield event
            resolved = False
            async for event in self._resolve_children(conversation_id, token):
                resolved = True
                yield event

            # 3) Park while any approval remains explicitly deferred (a
            # cancel that landed mid-decide or mid-dispatch wins instead:
            # wind down rather than pausing at the gate) — UNLESS an unseen
            # user post can reach the model past the gate (3b below).
            awaiting = self.ledger.open_turn_awaiting_executions(conversation_id)
            if awaiting:
                cancel_entry = self.ledger.open_turn_cancel_requested(conversation_id)
                if cancel_entry is not None:
                    for event in await self._wind_down_async(conversation_id, cancel_entry):
                        yield event
                    return
                if any(ex.id not in announced_gates for ex in awaiting):
                    announced_gates.update(ex.id for ex in awaiting)
                    yield ApprovalRequired(
                        conversation_id=conversation_id,
                        executions=[ex.model_copy(deep=True) for ex in awaiting],
                    )
                # 3b) A POST REACHES THE MODEL PAST THE GATE (0008). The gated
                # execution projects as a placeholder, so the path is
                # well-formed and the user gets an answer while the approval
                # prompt is still up. ONLY a post does this: an allowed sibling
                # that completed in this same round is not new material to the
                # model, and counting it would fire a round on every approval.
                if not self._has_unseen_post(conversation_id, last_seen):
                    # A DRIVE RETURNS ONLY WHEN NOTHING IN ITS SUBTREE CAN
                    # ADVANCE. One rule, and the whole asymmetry falls out of
                    # it: a gated child with no subtree of its own returns (its
                    # drive is gone, and `notify()` restarts it), while a gated
                    # PARENT whose children are still working waits — which is
                    # exactly why `ApprovalRequired` is not terminal on a
                    # parent's stream. The degenerate case (a gate with nothing
                    # else running) returns exactly as it always did.
                    if not await self._await_subtree(conversation_id, token, wake):
                        return
                    continue
                # fall through to the model call — and deliberately PAST the
                # progress-continue below: `undecided` holds this gate on every
                # pass (the ledger counts approval_status=PENDING as undecided
                # so a re-drive re-asks), so continuing here would re-ask
                # decide() forever. A cancel landing between here and the model
                # call trips the token, which `_race_cancellation` catches.
            else:
                announced_gates.clear()
                if undecided or ready or spawned or resolved:
                    continue  # re-run the cancel check before calling the model

            # 3c) UNRESOLVED CHILDREN. The parent's turn cannot CLOSE until
            # every subagent resolves (the close site guards that), but the
            # model is RE-ENGAGED whenever the open turn holds something it
            # has not seen: a resolved child's result (the runtime-minted
            # result execution is its durable, positional record), a mid-turn
            # user post, any non-spawn tool result. A pure spawn round is
            # deliberately not a wake — the confirmations travel with the
            # first real update. `open_turn_unseen_material` is the durable
            # half; `last_seen` covers the one state it cannot see (a post
            # that landed under this drive's own in-flight LLM call). Only
            # with neither does the drive park on the subtree.
            # `wake_parent_on_subagent_completion=False` narrows the durable
            # half only: a resolution alone is not material, so the parent
            # batches — resolving each child as it finishes but calling the
            # model once, when no unresolved child remains (this guard stops
            # applying) or when other material arrives.
            if self.ledger.open_turn_index(conversation_id) is not None and _unresolved_children(
                self.session, conversation_id
            ):
                conversation = self.session.conversations[conversation_id]
                fresh_material = open_turn_unseen_material(
                    conversation.nodes,
                    self.session.entries,
                    include_child_results=(
                        self.session.session_config.runtime_config.wake_parent_on_subagent_completion
                    ),
                ) or (last_seen is not None and self._has_unseen_user_message(conversation_id, last_seen))
                if not fresh_material:
                    if not await self._await_subtree(conversation_id, token, wake):
                        return
                    continue
                # fall through — step 4 calls the model with the update;
                # still-running children project as nothing new

            # 4) Step-limit and doom-loop checks, then call the model —
            # reached when every execution in the open turn is terminal, OR
            # through 3b's fall-through with a gated execution still live and
            # projecting its placeholder (0008).
            #
            # Hard max: if the open turn already has step_count LLM responses
            # and step_count >= hard_max_steps, close the turn now.
            # Soft max / doom loop: restrict tool_choice to "none" so the LLM
            # can only reply with text, ending the turn gracefully.
            config = self.session.session_config.runtime_config
            step_count = self.session.get_conversation_status(conversation_id).step_count
            soft_max_steps, hard_max_steps = self._step_limits(conversation_id)
            if hard_max_steps > 0 and step_count >= hard_max_steps:
                error = f"Hard max steps limit reached: {step_count}"
                if _unresolved_children(self.session, conversation_id):
                    # The hard limit is a COST stop and it wins over the
                    # children: parking until they resolve would leave a
                    # BUSY-deriving state whose run() does nothing (a hot
                    # loop for a polling app) while their work keeps
                    # billing. Cancel them, settle every link, then close.
                    self._cascade_cancel(conversation_id, TurnOutcome.CANCELLED, error)
                    for event in await self._settle_children(conversation_id):
                        yield event
                    # A cancel() that landed during the settle's awaits
                    # controls this close like every other: the children are
                    # settled either way, and burying the request inside an
                    # ERRORED bracket would leave it consumed by nothing.
                    cancel_entry = self.ledger.open_turn_cancel_requested(conversation_id)
                    if cancel_entry is not None:
                        for event in self._wind_down(conversation_id, cancel_entry):
                            yield event
                        return
                # A gated execution is reachable here since 0008 — each post
                # while blocked burns a real step — and no close leaves a
                # nonterminal execution behind.
                for event in self._settle_undispatched(conversation_id):
                    yield event
                self._close_turn(conversation_id, TurnOutcome.ERRORED, error=error)
                return

            tool_choice: str | None = None
            if (
                soft_max_steps > 0
                and step_count >= soft_max_steps
                and config.limit_tool_choice_on_soft_max_steps_reached
            ):
                tool_choice = "none"
            if config.limit_tool_choice_on_doom_loop_flagged and self.ledger.open_turn_has_doom_loop_flagged(
                conversation_id
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
            # and only here, because every execution is terminal at this point
            # except a gated one on a 3b fall-through round (0008), and both
            # the wind-down and the failure close settle those before closing.
            model_string = self.build_model_string(
                conversation_id,
                self.session.session_config.llm_config,
            )
            # Stamp the ACTIVE config + native flag on the session, derived
            # from the CONFIGURED values (a routing middleware must not drift
            # the session) — from here on, everything handed the session
            # (`get_tools`, every middleware, provenance recording) reads what
            # THIS call will use from it. Re-stamped every iteration, which is
            # what makes a mid-session model or native flip just work.
            self.session.update_llm_config(
                model_string,
                use_native_tools=self.session.session_config.use_native_tools,
            )
            # What this LLM call will have seen, fingerprinted by path length
            # — exact because the path is append-only and nothing yields
            # between this capture and the projection inside
            # `prepare_llm_call`. Anything user-posted at or past this index
            # when the final answer lands was NOT in the projection; the
            # close-site check below loops one more round instead of closing
            # over it.
            seen = len(self.session.conversations[conversation_id].nodes)
            last_seen = seen
            messages, system_message = self.prepare_llm_call(conversation_id)
            # `get_tools` is application code and may block indefinitely, so
            # the whole step is raced. A lost race produces no tool list and
            # makes no LLM call: control returns to the loop top, which winds
            # the turn down — exactly the aborted-LLM-call path. A RAISE is
            # deliberately not caught here: it propagates and aborts the run
            # with the turn left open and resumable, and the next run() asks
            # again. Substituting an empty tool list would silently change the
            # model's answer.
            tools_task = asyncio.ensure_future(self._collect_tools(conversation_id))
            tools_done, collected, _ = await _race_cancellation(
                tools_task,
                token,
                0,
                None,
            )
            if not tools_done:
                continue
            # Only the list the model is shown is needed here; the private
            # names it excludes are re-resolved by the birth step, which is the
            # one that has to refuse a call naming them.
            _, tool_list = collected
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
                        reasoning=self.session.llm_config.reasoning,
                        provider=self.provider,
                        timeout=request_timeout,
                        total_timeout=total_timeout,
                        **completion_options(self.session.llm_config),
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
                                elif (delta := _to_delta_event(conversation_id, stream_event)) is not None:
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
                            reasoning=self.session.llm_config.reasoning,
                            provider=self.provider,
                            timeout=request_timeout,
                            total_timeout=total_timeout,
                            **completion_options(self.session.llm_config),
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
                logger.error(
                    "conv=%s LLM call failed (model=%s)",
                    conversation_id,
                    self.session.llm_config.model,
                    exc_info=exc,
                )
                cancel_entry = self.ledger.open_turn_cancel_requested(conversation_id)
                if cancel_entry is not None:
                    for event in await self._wind_down_async(conversation_id, cancel_entry):
                        yield event
                    return
                if _unresolved_children(self.session, conversation_id):
                    # A wake-round call failed while subagents are mid-work.
                    if self.session.conversations[conversation_id].depth == 0:
                        # The MAIN conversation's failure must not destroy
                        # the children's work over a provider hiccup: the
                        # turn stays OPEN and the failure re-raises. The
                        # children keep driving; the next run() resumes the
                        # bracket (the unchanged material means the model
                        # call is retried — by the caller's explicit run(),
                        # never automatically).
                        raise
                    # A SUBAGENT parent must terminalize instead: nothing
                    # outside it ever retries a subagent mid-run, so an open
                    # failed turn would strand the whole tree. Settle its
                    # children and close — a finished child whose result says
                    # so, the established doctrine.
                    self._cascade_cancel(conversation_id, TurnOutcome.CANCELLED, str(exc))
                    for event in await self._settle_children(conversation_id):
                        yield event
                    cancel_entry = self.ledger.open_turn_cancel_requested(conversation_id)
                    if cancel_entry is not None:
                        for event in self._wind_down(conversation_id, cancel_entry):
                            yield event
                        return
                # A gated execution is reachable here since 0008 — a post can
                # drive a fall-through round over a live gate — and no close
                # leaves a nonterminal execution behind. Yielding then raising
                # is fine: the consumer pulls the events, and the next pull
                # raises.
                for event in self._settle_undispatched(conversation_id):
                    yield event
                outcome = TurnOutcome.TIMED_OUT if isinstance(exc, ClientTimeoutError) else TurnOutcome.ERRORED
                self._close_turn(conversation_id, outcome, error=str(exc))
                raise

            # Run after_llm_response middleware before recording: the message
            # is fully assembled (streaming or non-streaming) so all content
            # blocks are present.
            message = self._run_middlewares("after_llm_response", conversation_id, message)

            # Record the assistant message, receive its executions, and (for a
            # final answer) close the bracket ATOMICALLY — and atomically means
            # NO AWAIT, not merely no yield: every session write for this round
            # lands in one synchronous block, so nothing — a suspend, a crash,
            # or a `post_message` arriving from an application's UI on this same
            # event loop — can strand a tool_use request without its
            # ToolExecutions or leave a fully-answered bracket open to a
            # duplicate LLM call. Birth is deliberately NOT here: asking the
            # registry is async, so it runs as its own step against the durable
            # RECEIVED entries this wrote.
            # The round keys off the tool_calls themselves, not finish_reason:
            # a misclassifying provider can neither wedge the conversation
            # ("stop" + calls) nor loop it ("tool_use" + none).
            events = self._record_assistant(conversation_id, message, finish_reason, self.session.llm_config)
            if message.tool_calls:
                self._receive_executions(conversation_id, message)
                for event in events:
                    yield event
                continue  # → step 1a births them, then decide()

            # Final answer. Precedence at the close site: an unconsumed
            # cancel controls the close (the within-grace message stays
            # recorded, the requested outcome wins and buries any unseen
            # post); next, a user message posted after `seen` OR an
            # unresolved subagent means the turn cannot close COMPLETED yet —
            # record the premature final answer and loop instead of closing
            # (an unseen message runs another round straight away via
            # `last_seen`; unresolved children park until the next material);
            # next, a finish reason that is NOT AN ANSWER closes ERRORED
            # instead (see below); else close COMPLETED. All checks are
            # synchronous inside the same no-yield window as the record and
            # the close, so nothing can land between a check and the
            # `TurnFinish`.
            cancel_entry = self.ledger.open_turn_cancel_requested(conversation_id)
            if cancel_entry is not None:
                events.extend(await self._wind_down_async(conversation_id, cancel_entry))
            elif (
                self._has_unseen_user_message(conversation_id, seen)
                or _unresolved_children(self.session, conversation_id)
                # A LIVE GATE KEEPS THE TURN OPEN (0008). Reachable only since
                # a post can now drive a round past the gate; closing here
                # would strand the approval outside any open turn, where
                # `pending_approvals()` cannot see it and `notify()` has
                # nothing to re-decide, and would freeze the placeholder into
                # the history for a call that never ran. The next pass parks at
                # step 3 instead.
                or self.ledger.has_awaiting_approval(conversation_id)
            ):
                for event in events:
                    yield event
                continue  # → the next projection carries answer + message
            elif finish_reason in NON_ANSWER_FINISH_REASONS:
                # NO TOOL CALLS AND NOT AN ANSWER. The round above keys off
                # `tool_calls` so a misclassifying provider cannot wedge or
                # loop the conversation; that rule settles "stop" vs
                # "tool_use" and says nothing about a model that stopped
                # WITHOUT answering. Reaching here means the turn was about to
                # be declared a complete answer when it is not one: `length`
                # cut it off mid-sentence, or `error` — every transport's
                # canonical value for a refusal / safety filter / guardrail,
                # always with an `error_message` — refused it.
                #
                # The partial STAYS recorded (`_record_assistant` ran above).
                # That is the difference from a transport failure, which drops
                # its partial: those tokens were really produced, and on a
                # truncation they are the useful half. Settling first because
                # no close may leave a nonterminal execution behind.
                events.extend(self._settle_undispatched(conversation_id))
                failure = message.error_message or f"the model stopped with finish_reason {finish_reason!r}"
                self._close_turn(conversation_id, TurnOutcome.ERRORED, error=failure)
                for event in events:
                    yield event
                raise IncompleteResponseError(failure)
            else:
                self._close_turn(conversation_id, TurnOutcome.COMPLETED)
            for event in events:
                yield event
            return

    # ── subagents ───────────────────────────────────────────────────────────

    def _spawn_children(self, conversation_id: str) -> list[ChildConversation]:
        """Create one child conversation per unhandled spawn in the open turn.

        Handled-ness is DURABLE: a spawn has been handled iff a
        `ChildConversation` in the open turn names its execution. Nothing is
        remembered in memory, so a reload mid-tree re-enters here and skips
        exactly the spawns that already have children."""
        handled = {entry.tool_execution_id for entry in self._child_entries(conversation_id)}
        created: list[ChildConversation] = []
        for execution in self.ledger.open_turn_executions(conversation_id):
            if execution.id in handled:
                continue
            payload = spawn_payload(execution)
            if payload is None:
                continue
            created.append(self._spawn_one(conversation_id, execution, payload))
        return created

    def _spawn_one(
        self,
        conversation_id: str,
        execution: ToolExecution,
        payload: dict,
    ) -> ChildConversation:
        """One child: validated, created, seeded, and linked into the parent."""
        # VIOLATION 2 — a spawn that was never declared. This payload went
        # around the gate entirely: the child would be spawned at any depth and
        # the cap would silently not exist. Same invisible failure a name match
        # would have had, closed from the other side.
        if execution.tool_spec is None or not declares_spawn(execution.tool_spec):
            raise AgentError(
                f"execution {execution.id!r} returned a {SPAWN_MARKER!r} payload but its "
                f"tool spec never declared one; a spawn that bypasses the gate is a "
                f"contract violation, not a subagent."
            )
        # VIOLATION 3 — a spawn from a conversation at or past the cap. The
        # gate lives in `get_tools`, but a registry resolves by NAME at
        # dispatch, so anything reaching dispatch with that name runs. This
        # check puts the cap where it actually means something: child creation.
        # `exclude` because THIS execution is settled and already in the count
        # — without it the Nth legitimate spawn would read as the (N+1)th.
        if not self._spawn_gate_open(conversation_id, exclude=execution.id):
            raise AgentError(
                f"conversation {conversation_id!r} may not spawn subagents "
                f"(depth={self.session.conversations[conversation_id].depth}, "
                f"max_depth={self.session.session_config.runtime_config.subagents_max_depth})."
            )
        # The runner validates the keys it is about to ACT on, at the moment it
        # acts on them, because nothing else does: the core never checks
        # `structured_content` against `output_schema`, so a tool that
        # contradicts its own declaration still records COMPLETED.
        missing = [key for key in SPAWN_REQUIRED_KEYS if not payload.get(key)]
        if missing:
            raise AgentError(
                f"spawn payload from execution {execution.id!r} is missing required "
                f"field(s) {missing}; that is a contract violation, not a subagent "
                f"with empty fields."
            )

        parent = self.session.conversations[conversation_id]
        ts = self.now_ms()
        child_id = self.generate_id()
        self.session.conversations[child_id] = Conversation(
            id=child_id,
            nodes=[],
            created_at=ts,
            updated_at=ts,
            depth=parent.depth + 1,
        )
        # The seed — the child's FIRST user message, not necessarily its only
        # one: a live child accepts `post_message` like any conversation with
        # an open turn (its own subagents running included). The
        # `TurnStart` is opened by the child's own drive, exactly like any
        # other conversation's.
        self._append(
            child_id,
            lambda entry_id, parent_id, stamp: UserMessage(
                id=entry_id,
                parent_id=parent_id,
                created_at=stamp,
                parts=[TextContent(text=payload["prompt"])],
            ),
        )
        return self._append(
            conversation_id,
            lambda entry_id, parent_id, stamp: ChildConversation(
                id=entry_id,
                parent_id=parent_id,
                created_at=stamp,
                conversation_id=child_id,
                tool_execution_id=execution.id,
            ),
        )

    def _start_children(
        self,
        conversation_id: str,
        spawned: list[ChildConversation],
    ) -> None:
        """Give every fresh child a handle on the run that spawned it, started
        or queued according to who owns them and what the worker pool allows.

        Every admission — inline or later — announces `SubagentStarted` from
        `_grant`, synchronously, before the child's task first runs; with the
        batch's `SubagentsSpawned` already on the same inbox, the stream
        carries them in order."""
        run = self._runs.get(conversation_id) or self._parked_handle(conversation_id)
        if run is None:
            return
        for entry in spawned:
            child = AgentRun(
                self,
                conversation_id=entry.conversation_id,
                streaming=run._streaming,
                on_event=None,  # the parent's callback sees it through the fan-in
                eager=run.autostart_subagents,
                defer_start=True,  # ← the pool decides when
                autostart_subagents=run.autostart_subagents,
                parent=run,
            )
            run._adopt(child)
            if run.autostart_subagents:
                self._request_slot(
                    entry.conversation_id,
                    child._start_eager,
                    handle=child,
                    announce=child,
                )

    def _child_entries(self, conversation_id: str) -> list[ChildConversation]:
        entries = self.session.entries
        index = self.ledger.open_turn_index(conversation_id)
        if index is None:
            return []
        return [
            entry
            for node_id in self.ledger.conversation(conversation_id).nodes[index:]
            if isinstance(entry := entries.get(node_id), ChildConversation)
        ]

    def _stop_children(self, conversation_id: str) -> None:
        """THE STOP HANDSHAKE (drive step 2b'): act on every stop payload in
        the open turn. Value-side only — there is no gate half, because
        stopping bypasses no cap.

        Idempotent by construction, from two facts. The POSITION BOUND: a
        stop only matches a `ChildConversation` link that PRECEDES it on the
        path — children that existed when the stop was issued — so a later
        spawn reusing the same task id (task ids are model-authored) can
        never be killed by an old signal, on a later loop pass or a reload
        alike. And the TARGET-STATE no-ops: a resolved, missing, finished
        (IDLE — its resolution is imminent) or already-CANCELLING match means
        the signal is already consumed."""
        index = self.ledger.open_turn_index(conversation_id)
        if index is None:
            return
        nodes = self.ledger.conversation(conversation_id).nodes
        entries = self.session.entries
        positions = {node_id: position for position, node_id in enumerate(nodes[index:], start=index)}
        links: list[tuple[int, ChildConversation]] = []
        stops: list[tuple[int, ToolExecution, dict]] = []
        for position in range(index, len(nodes)):
            entry = entries.get(nodes[position])
            if isinstance(entry, ChildConversation):
                links.append((position, entry))
            elif isinstance(entry, ToolExecution) and (payload := stop_payload(entry)) is not None:
                stops.append((position, entry, payload))
        for position, execution, payload in stops:
            task_id = payload.get("task_id")
            if not task_id:
                # The runner validates the keys it is about to ACT on, at the
                # moment it acts on them — the same rule the spawn payload
                # gets, because nothing else checks a payload against its
                # tool's own declaration.
                raise AgentError(
                    f"stop payload from execution {execution.id!r} is missing its "
                    f"task_id; that is a contract violation, not a stop with an "
                    f"empty target."
                )
            target = self._stop_target(links, positions, position, task_id)
            if target is not None:
                with contextlib.suppress(AlreadyCancellingError):
                    self.cancel(conversation_id=target)

    def _stop_target(
        self,
        links: list[tuple[int, ChildConversation]],
        positions: dict[str, int],
        stop_position: int,
        task_id: str,
    ) -> str | None:
        """The conversation one stop payload cancels, or None when the signal
        is already consumed (or never matched anything live).

        Task ids are model-authored and nothing enforces uniqueness, so with
        duplicates the target is "the first match that was still UNRESOLVED
        when the stop was issued" — and issue-time state is read off PATH
        POSITIONS, because this handler replays every loop pass and live state
        alone cannot tell "resolved before the stop" (skip it — the model was
        naming its sibling) from "resolved after it" (the stop caused or raced
        that resolution — consumed; falling through here would turn one
        consumed stop into a standing kill order for the next same-id
        sibling). A link's resolution position is its result execution's; a
        link resolved WITHOUT one (a wind-down settle) counts as consumed —
        the whole turn is closing."""
        for position, link in links:
            if position >= stop_position:
                return None  # a stop only names children that predate it
            spawn = self.session.entries.get(link.tool_execution_id)
            if not isinstance(spawn, ToolExecution):
                continue
            payload = spawn_payload(spawn)
            if payload is None or payload.get("task_id") != task_id:
                continue
            if link.execution_result is not None:
                resolved_at = positions.get(link.result_execution_id or "")
                if resolved_at is not None and resolved_at < stop_position:
                    continue  # already resolved when the stop was issued — not the target
                return None  # resolved after (or by) the stop — consumed
            child_id = link.conversation_id
            if child_id not in self.session.conversations:
                return None
            if self.session.get_conversation_status(child_id).status is ConversationStatus.IDLE:
                return None  # already finished; there is nothing to stop
            if self.ledger.open_turn_cancel_requested(child_id) is not None:
                return None  # already cancelling — the signal is consumed
            return child_id
        return None

    async def _resolve_children(
        self,
        conversation_id: str,
        token: CancellationToken,
    ) -> AsyncIterator[AgentEvent]:
        """Derive a result for every subagent of this conversation whose turn
        bracket has closed.

        "Its turn bracket closes" is the WHOLE signal — a finished child
        refuses further input (`post_message` rejects a finished subagent), so
        a closed bracket means finished, and a child that failed or timed out
        is a finished child, not an exception travelling upward. Whichever way
        it ended, the `ChildConversation` resolves and the parent continues.

        The result is derived by a TOOL — the one the spawn payload named — run
        in the PARENT conversation through the ordinary lifecycle: birth,
        approvals, middleware, timeouts, cancellation, events, usage. It is a
        private tool, so it never projects as a `ToolMessage`; its output
        reaches the model through `ChildConversation.execution_result`, which
        is what the parent's projection renders."""
        for entry in self._child_entries(conversation_id):
            if entry.execution_result is not None:
                continue
            child_id = entry.conversation_id
            if child_id not in self.session.conversations:
                continue
            if self.session.get_conversation_status(child_id).status != ConversationStatus.IDLE:
                continue  # still working, blocked, or winding down
            async for event in self._derive_child_result(conversation_id, entry, token):
                yield event

    async def _derive_child_result(
        self,
        conversation_id: str,
        entry: ChildConversation,
        token: CancellationToken,
    ) -> AsyncIterator[AgentEvent]:
        payload = spawn_payload(self.session.entries[entry.tool_execution_id])
        if payload is None:  # unreachable: the child only exists because it spawned
            raise AgentError(f"ChildConversation {entry.id!r} has no spawn payload to derive a result from.")
        # A runtime-minted call. The tool_call_id is the RUNNER's, not a
        # provider's — no `ToolCall` for it exists in any assistant message,
        # which is precisely why the execution never projects as a ToolMessage.
        call = ToolCall(
            id=self.generate_id(),
            name=payload["process_subagent_result_tool_name"],
            arguments={
                "task_id": payload["task_id"],
                "prompt": payload["prompt"],
                "description": payload["description"],
                # NOTE the two ids. The tool's `conversation_id` PARAMETER is
                # the parent — where it runs. This ARGUMENT is the child being
                # summarized. They are never the same conversation.
                "conversation_id": entry.conversation_id,
            },
        )
        async for event in self._invoke_runtime_tool(conversation_id, call, token):
            yield event
            if isinstance(event, ToolExecuted):
                # `result_execution_id` lands in the same write as the result:
                # the execution's path position is where the projector renders
                # this resolution — after every assistant message that
                # predates it — so the projected history stays append-only
                # while this link mutates in place.
                self._persist_entry(
                    conversation_id,
                    entry,
                    execution_result=self._child_result_from(event),
                    result_execution_id=event.execution.id,
                )
                self._retire_child_failure(entry.conversation_id)

    def _retire_child_failure(self, child_id: str) -> None:
        """Retrieve a resolved child's stored task exception so asyncio never
        reports it unretrieved. A subagent's failure never propagates — its
        drive stores the exception on the handle and closes the turn ERRORED,
        and the failure reaches the model through the link's result — so once
        the link is resolved, nobody will ever await that task."""
        handle = self._parked_handle(child_id)
        if handle is not None and handle._task is not None and handle._task.done() and not handle._task.cancelled():
            handle._task.exception()

    def _child_result_from(self, event: ToolExecuted) -> ExecutionResult:
        """The `ExecutionResult` that becomes `ChildConversation.execution_result`.

        A COMPLETED result tool gives its own. Anything else — FAILED,
        NOT_FOUND, TIMED_OUT, REJECTED, CANCELLED — still has to resolve the
        child, or the parent blocks forever on a subagent that already
        finished. The derived text comes from the projector, the same
        derivation the wire and the `ToolExecuted` event already share, so the
        three never disagree."""
        execution = event.execution
        if execution.status == ExecutionStatus.COMPLETED and execution.result is not None:
            return execution.result
        return ExecutionResult(
            content=[TextContent(text=event.result_text)],
            is_error=True,
        )

    async def _invoke_runtime_tool(
        self,
        conversation_id: str,
        call: ToolCall,
        token: CancellationToken,
    ) -> AsyncIterator[AgentEvent]:
        """Run ONE runner-originated tool call through the ordinary lifecycle.

        Two rules do not apply to a call the runtime minted. The private-name
        refusal is about a MODEL naming a tool it was never shown, and the
        doom-loop check is a heuristic about model behavior; neither means
        anything here. Everything else is identical to a model's call — the
        creation middleware pair included, which is why `after_tool_creation`
        runs inside the build below rather than being skipped as a
        model-behavior concern. `_birth_draft` already fired its partner, and
        the two must stay paired on every creation path."""
        draft, exception = await self._birth_draft(conversation_id, call, set(), token)
        execution = self._append(
            conversation_id,
            lambda entry_id, parent_id, ts: self._run_middlewares(
                "after_tool_creation",
                conversation_id,
                draft.model_copy(
                    update={
                        "id": entry_id,
                        "parent_id": parent_id,
                        "created_at": ts,
                        "conversation_id": conversation_id,
                        "ended_at": (ts if draft.status != ExecutionStatus.PENDING else None),
                    },
                ),
                exception,
            ),
        )
        yield ToolCallReceived(
            conversation_id=conversation_id,
            tool_call_id=execution.tool_call_id,
            execution=execution.model_copy(deep=True),
        )
        if execution.status != ExecutionStatus.PENDING:  # terminal at birth
            _, event = self._finalize_outcome(conversation_id, execution, exception)
            yield event
            return
        pair = await self._decide_one(conversation_id, execution, token)
        if pair is None:  # the token won; the wind-down records CANCELLED
            return
        modified, decision = pair
        changes: dict = {
            "approval_decisions": [*modified.approval_decisions, decision],
            "approval_status": _APPROVAL_STATUS[decision.decision],
        }
        denied = decision.decision == ApprovalOption.DENY
        if denied:
            changes["status"] = ExecutionStatus.REJECTED
            changes["ended_at"] = self.now_ms()
        persisted = self._persist_execution(conversation_id, modified, **changes)
        if denied:
            _, event = self._finalize_outcome(conversation_id, persisted)
            yield event
            return
        if decision.decision == ApprovalOption.PENDING:
            # A policy that gates the result tool parks the parent exactly like
            # any other gate. The child stays unresolved until it is answered.
            self._publish_approval(conversation_id, persisted)
            return
        async for event in self._dispatch_one(conversation_id, persisted, token):
            yield event

    # ── waiting on the subtree ──────────────────────────────────────────────

    async def _await_subtree(
        self,
        conversation_id: str,
        token: CancellationToken,
        wake: asyncio.Event,
    ) -> bool:
        """Wait for something in this conversation's subtree to change. True if
        the drive should loop again; False if it should return.

        THE TEARDOWN WINDOW. A `notify()` can land on a drive that is already
        returning, where a wake would go to something that will never loop
        again. That is why the recheck SET is the source of truth and the event
        is only a wake: this re-checks the set immediately before deciding to
        return, and loops instead if the id is back in it."""
        if conversation_id in self._recheck:
            return True
        if not self._can_subtree_advance(conversation_id):
            return False
        if token.cancelled:
            return True  # loop; the cancel check at the top winds down
        # PARKED IS NOT WORKING. Releasing here is what lets this
        # conversation's own children be admitted, and it is the whole reason
        # a nested tree cannot deadlock under `subagents_max_workers`.
        self._release_slot(conversation_id)
        waiter = asyncio.ensure_future(wake.wait())
        cancelled = asyncio.ensure_future(token.wait_cancelled())
        try:
            await asyncio.wait({waiter, cancelled}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            await _cancel_quietly(waiter)
            await _cancel_quietly(cancelled)
            # Re-acquire, raced against the token: a drive blocked on a slot
            # must stay cancellable. When the token wins this returns without
            # a slot, and the loop top winds the turn down — which needs none.
            await self._acquire_slot(conversation_id, token)
        return True

    def _can_subtree_advance(self, conversation_id: str) -> bool:
        """Is any unresolved subagent of this conversation still able to make
        progress? A BUSY child is working; an IDLE one has finished and this
        conversation can resolve it; a CANCELLING one is winding down. A
        BLOCKED child contributes nothing — it is waiting on the application
        exactly as this conversation is.

        A child with a LIVE DRIVE can advance whatever its entries currently
        say. That term is what makes a restart safe: a drive that has just been
        handed back a child (`_restart_unresolved_children`, `notify()`) has not
        re-asked `decide()` yet, so the child still derives BLOCKED — and
        without this the parent would give up in the window before its own
        subagent's first step."""
        return any(
            self._runs.get(child_id) is not None
            or self.session.get_conversation_status(child_id).status
            in (
                ConversationStatus.BUSY,
                ConversationStatus.IDLE,
                ConversationStatus.CANCELLING,
            )
            for child_id in self._unresolved_child_ids(conversation_id)
        )

    def _publish_approval(self, conversation_id: str, execution: ToolExecution) -> None:
        """Hand a freshly-deferred gate to the run watching this conversation,
        which forwards it to every ancestor's `approvals` stream."""
        run = self._runs.get(conversation_id)
        if run is not None:
            run._publish_approval(execution.model_copy(deep=True))

    # ── per-step machinery ─────────────────────────────────────────────────

    async def _wind_down_async(
        self,
        conversation_id: str,
        cancel_entry: CancelRequested,
    ) -> list[AgentEvent]:
        """`_wind_down` plus the subagent half.

        A cancel cascades, so every live child already has its own
        `CancelRequested` and is winding down. Wait for those drives to settle,
        then resolve each still-unresolved `ChildConversation` with a
        cancellation result — WITHOUT running the result tool, exactly as a
        PENDING execution is terminalized without being dispatched.

        Resolving them is not tidiness: an unresolved child behind a CLOSED
        turn fails projection loudly (the same fail-loud rule a nonterminal
        execution gets), so leaving one would wedge the conversation
        permanently. This also covers the `autostart_subagents=False` child
        that was never driven at all."""
        events = await self._settle_children(conversation_id)
        return [*events, *self._wind_down(conversation_id, cancel_entry)]

    async def _settle_children(self, conversation_id: str) -> list[AgentEvent]:
        """Wait out, flush, and resolve every unresolved subagent of this
        conversation — the child half of every close that must not leave one
        behind: the cancel wind-down, the hard-step-limit close, and a
        subagent parent's failure close. The caller has already cascaded a
        cancel to the children; the links resolve WITHOUT the result tool,
        exactly as a PENDING execution is terminalized without being
        dispatched (`result_execution_id` stays None, which is also what
        makes the LINK itself render the outcome)."""
        # SETTLING IS BOOKKEEPING, NOT WORK: no model call, no tool body.
        # The slot goes back first — waiting below for cancelled children to
        # settle while holding one would be a slot-holder waiting on other
        # conversations, the one thing the pool's rule forbids — and the
        # freed capacity goes to an unrelated queued subagent immediately
        # rather than a grace window later.
        self._release_slot(conversation_id)
        for child_id in self._unresolved_child_ids(conversation_id):
            # A child whose pool-granted start failed carries a recorded
            # admission error; resolving its link below makes it unreachable
            # for the loop-top re-raise, so this close deliberately wins and
            # the record is discarded rather than leaked.
            self._admission_errors.pop(child_id, None)
            run = self._runs.get(child_id)
            if run is not None and run._task is not None:
                with contextlib.suppress(BaseException):
                    await run._task
        # A child cancelled with no drive left has nobody else to close its
        # bracket, and a conversation stuck CANCELLING forever is exactly what
        # the resolution below exists to prevent one level up.
        events = list(self._flush_cancelled_children(conversation_id))
        for entry in self._child_entries(conversation_id):
            if entry.execution_result is None:
                self._persist_entry(
                    conversation_id,
                    entry,
                    execution_result=ExecutionResult(
                        content=[TextContent(text=self.CANCELLED_SUBAGENT_TEXT)],
                        is_error=True,
                    ),
                )
        return events

    def _flush_cancelled_children(self, conversation_id: str) -> list[AgentEvent]:
        """Wind down every unresolved subagent that was cancelled and has no
        drive left to do it itself.

        `cancel()` is a signal: it writes the `CancelRequested` and stops,
        because consuming one is a DRIVE's job. A subagent's drive is often
        already gone by then — it parked at a gate, or the application never
        started it — and nothing else will ever look at that conversation. The
        parent is the only conversation that knows the child exists, and it is
        the one that would otherwise wait forever, so the parent's drive does
        the flush.

        The child's events surface on the parent's stream. They cannot arrive
        twice: the only handle that could have delivered them is the one whose
        drive already ended."""
        events: list[AgentEvent] = []
        # `SubagentFinished` follows ownership like every lifecycle event: a
        # framework-driven tree announces the flush's close (this is how a
        # never-started child's cancellation becomes visible — `Spawned` with
        # a `Finished` and no `Started` reads "cancelled before admission");
        # under `autostart_subagents=False` no lifecycle events exist. A child
        # with a live drive is skipped here, and its own `_end_run` announces.
        run = self._runs.get(conversation_id) or self._parked_handle(conversation_id)
        announces = run is not None and run.autostart_subagents
        for child_id in self._unresolved_child_ids(conversation_id):
            if self._runs.get(child_id) is not None:
                continue  # a live drive winds itself down
            cancel_entry = self.ledger.open_turn_cancel_requested(child_id)
            if cancel_entry is None:
                continue
            events.extend(self._wind_down(child_id, cancel_entry))
            if announces:
                events.append(SubagentFinished(conversation_id=child_id, outcome=cancel_entry.outcome))
        return events

    def _settle_undispatched(self, conversation_id: str) -> list[AgentEvent]:
        """Terminalize every undispatched execution in the open turn as
        CANCELLED — stamped `cancel_signalled_at`, resultless, errorless,
        approval state untouched — and return their `ToolExecuted` events.
        RECEIVED ones included: a close landing while the registry is being
        consulted must settle the unborn too, or the turn closes over an
        execution no drive will ever finish. (A denied call was already
        terminal REJECTED at decision time; an in-flight one was settled by
        the grace machinery; an orphaned RUNNING one was recovered at drive
        start.) Each settled execution passes through the outcome middleware
        pair.

        The close-side half of "no close leaves a nonterminal execution
        behind". The cancel wind-down has always done this; the failure closes
        need it too since 0008, because a post can now drive rounds while a
        gate is open, which puts `hard_max_steps` and an LLM failure within
        reach of a live gate. Closing over one strands the approval outside any
        open turn and freezes its placeholder into the projected history."""
        events: list[AgentEvent] = []
        for execution in self.ledger.open_turn_undispatched_executions(conversation_id):
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
            _, event = self._finalize_outcome(conversation_id, stamped)
            events.append(event)
        return events

    def _wind_down(self, conversation_id: str, cancel_entry: CancelRequested) -> list[AgentEvent]:
        """Consume a `CancelRequested`: settle every undispatched execution
        (`_settle_undispatched`), then close the turn with the requested
        outcome. The `ToolExecuted` events return to the caller. All session
        writes happen before any event is yielded."""
        events = self._settle_undispatched(conversation_id)
        self._close_turn(conversation_id, cancel_entry.outcome, cancel_entry.error)
        return events

    def _record_assistant(
        self,
        conversation_id: str,
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
            conversation_id,
            lambda entry_id, parent_id, ts: AssistantMessage(
                id=entry_id,
                parent_id=parent_id,
                created_at=ts,
                parts=parts,
                llm_config=llm_cfg.model_copy(),
                stop_reason=finish_reason or "stop",
            ),
        )
        self.ledger.record_usage(conversation_id, entry.id, **_to_usage_counters(message.usage))
        events: list[AgentEvent] = []
        for part in parts:
            if isinstance(part, ThinkingContent):
                events.append(
                    ReasoningBlock(
                        conversation_id=conversation_id,
                        text=part.thinking,
                        redacted=part.redacted,
                    ),
                )
            elif isinstance(part, TextContent):
                events.append(TextBlock(conversation_id=conversation_id, text=part.text))
        events.append(FinishReason(conversation_id=conversation_id, finish_reason=finish_reason))
        return events

    def _receive_executions(self, conversation_id: str, message) -> None:
        """Append one RECEIVED `ToolExecution` per tool call in the assistant
        response — SYNCHRONOUSLY, in model-request order, inside the same
        no-await block that appended the assistant message itself.

        THIS IS THE TRANSACTION that keeps the path always projectable. An
        assistant message carrying tool calls and the execution nodes that
        answer them are written together, so nothing can be appended between a
        `tool_call` and its output — not a `post_message` landing from an
        application's UI, not a crash. Without it, any await between the two
        appends leaves a path whose next projection puts a user message between
        an assistant tool call and its tool result, which every provider
        rejects.

        Birth — the registry's `create_execution` — is async and
        application-owned, so it CANNOT be part of this block. It runs as its
        own drive step (`_birth_executions`) and folds its answer into these
        entries in place.

        Only what the runner already knows is recorded: identity, the
        deep-copied `raw_tool_call` (so an entry can never alias the assistant
        message part — `extras` and any future `ToolCall` field ride along),
        and `is_doom_loop_flagged` — evaluated in append order, seeing only
        previously-appended executions."""
        calls = [part for part in adapter.message_to_parts(message) if isinstance(part, ToolCall)]
        for tc in calls:
            # Runs before the append so it only sees previously-appended
            # executions; parallel tool calls are evaluated in append order.
            doom_flagged = self._is_doom_loop(conversation_id, tc)
            raw = tc.model_copy(deep=True)

            def build(
                entry_id,
                parent_id,
                ts,
                _raw=raw,
                _d=doom_flagged,
            ) -> ToolExecution:
                return ToolExecution(
                    id=entry_id,
                    parent_id=parent_id,
                    created_at=ts,
                    # Provenance, stamped with the rest of the identity set the
                    # runner owns. A registry must not set it.
                    conversation_id=conversation_id,
                    tool_call_id=_raw.id,
                    raw_tool_call=_raw,
                    status=ExecutionStatus.RECEIVED,
                    is_doom_loop_flagged=_d,
                )

            self._append(conversation_id, build)

    async def _birth_executions(
        self,
        conversation_id: str,
        received: list[ToolExecution],
        token: CancellationToken,
    ) -> list[AgentEvent]:
        """THE `create_execution` CALL SITE: ask the registry for one draft per
        RECEIVED execution (concurrently), then fold each answer into the entry
        already on the path.

        A DRIVE STEP rather than straight-line code, for the same reason
        `decide` is one: its input is derived from durable state, so a run that
        returned mid-round — or a reloaded session, or a process that died here
        — re-enters and births exactly the executions still unborn. That is
        what makes RECEIVED safe to persist rather than merely transient.

        The registry owns the call-scoped facts (`raw_tool_call`, `tool_spec`,
        the birth `status` — PENDING or terminal-at-birth — `error`, `extras`,
        any approval state); the runner keeps the identity set and
        `is_doom_loop_flagged`, stamps `ended_at` for a terminal birth, and the
        ledger files the spec and stamps `tool_spec_id`. Failures are isolated
        per call: a raising `create_execution` (or a toolless runner) never
        breaks the set — the runner synthesizes the draft itself, FAILED for a
        raise and NOT_FOUND for the toolless case — preserving the invariant
        that every tool call produces exactly one tool output. Terminal births
        immediately run the outcome middleware pair.

        The specs are re-resolved here rather than carried over from the model
        round: a resumed birth has no such round behind it, and the private
        names are the only thing this step needs from them. Losing that race to
        a cancel leaves every execution RECEIVED for the loop top to wind down.

        The FOLD IS SEQUENTIAL even though the births are concurrent — the
        spawn budget the tool list could not enforce is applied in it, and has
        to see calls 1…k-1 of the same message as in-flight reservations.

        The cancellation race is PER CALL, inside `_birth_draft`, never around
        the gather: killing the gather would lose every draft and break
        one-output-per-call. N unborn executions yield N born ones even when a
        cancellation lands mid-batch."""
        specs_task = asyncio.ensure_future(self.resolve_tool_specs(conversation_id))
        resolved, specs, _ = await _race_cancellation(specs_task, token, 0, None)
        if not resolved:
            return []
        # The names the model was NOT shown. A call naming one is refused
        # rather than dispatched — see `_birth_draft`.
        private_names = {spec.name for spec in specs if spec.is_private}
        drafts = await asyncio.gather(
            *(
                self._birth_draft(conversation_id, execution.raw_tool_call, private_names, token)
                for execution in received
            )
        )
        events: list[AgentEvent] = []
        for execution, (draft, exception) in zip(received, drafts, strict=False):
            # The spawn budget the tool list could not enforce: the list was
            # fixed before the model call, so several spawn calls in ONE
            # response can overrun it.
            refusal = self._spawn_budget_refusal(conversation_id, draft)
            changes: dict = {
                "raw_tool_call": draft.raw_tool_call,
                "tool_spec": draft.tool_spec,
                "extras": draft.extras,
                "approval_status": draft.approval_status,
                "approval_decisions": draft.approval_decisions,
                "status": draft.status,
                "error": draft.error,
            }
            if refusal is not None:
                changes |= {"status": ExecutionStatus.REFUSED, "error": refusal}
            if changes["status"] != ExecutionStatus.PENDING:  # terminal birth
                changes["ended_at"] = self.now_ms()
            # `after_tool_creation` sees the EFFECTIVE birth state — the
            # registry's draft folded into the entry, with every runner-owned
            # birth fact already applied — and its return value is what gets
            # persisted, so the lifecycle branch chosen below reads the
            # post-hook status. A middleware may terminalize a PENDING birth
            # (straight to the outcome tail, never reaching `decide`) or
            # revive a terminal one.
            effective = execution.model_copy(update=changes)
            effective = self._run_middlewares(
                "after_tool_creation",
                conversation_id,
                effective,
                exception,
            )
            # `_persist_entry`, NOT `_persist_execution`: birth completes an
            # entry's creation rather than mutating it afterwards, so it leaves
            # `updated_at` alone. A born execution is durably identical to the
            # one a single append used to produce, and `updated_at` keeps
            # meaning "changed after it was created".
            born = self._persist_entry(conversation_id, effective)
            events.append(
                ToolCallReceived(
                    conversation_id=conversation_id,
                    tool_call_id=born.tool_call_id,
                    execution=born.model_copy(deep=True),
                )
            )
            if born.status != ExecutionStatus.PENDING:
                _, event = self._finalize_outcome(conversation_id, born, exception)
                events.append(event)
        return events

    def _spawn_budget_refusal(
        self,
        conversation_id: str,
        draft: ToolExecution,
    ) -> ToolExecutionError | None:
        """None if this call may spawn; the durable error to be born with if
        the open turn has spent its `subagents_max_per_turn` budget.

        Decided BEFORE either body runs, so dispatch order, concurrency and
        tool duration are all irrelevant — the only ordering it relies on is
        `_create_executions`' append loop, which is inherent to persisting
        executions in model-request order. A draft already terminal at birth
        keeps its own status: it will never dispatch, so it neither consumes
        the budget nor needs refusing."""
        if draft.status != ExecutionStatus.PENDING:
            return None
        if draft.tool_spec is None or not declares_spawn(draft.tool_spec):
            return None
        limit = self.session.session_config.runtime_config.subagents_max_per_turn
        if limit == Inf:
            return None
        committed = spawns_committed(self.session, conversation_id)
        if committed < limit:
            return None
        return ToolExecutionError(
            error_type="SpawnLimitReached",
            error_message=(
                f"Spawn limit reached ({committed}/{limit} subagents this turn). "
                f"Complete the remaining work yourself; do not retry."
            ),
            details={"limit": limit, "committed": committed},
        )

    async def _birth_draft(
        self,
        conversation_id: str,
        tc,
        private_names: set[str],
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
        # A model_copy, not a field-by-field rebuild: `extras` (and any future
        # `ToolCall` field) must ride into the draft's `raw_tool_call`.
        raw = tc.model_copy(deep=True)
        # `before_tool_creation` runs on the deep COPY, ahead of the private
        # check: a middleware that renames a call must be checked against the
        # effective name, or "private" would be bypassable by rewriting into
        # one. The returned call is what the registry sees and what the
        # execution carries.
        raw = self._run_middlewares("before_tool_creation", conversation_id, raw)
        # A PRIVATE tool was never offered, but a model can still emit a name it
        # was never given. Refuse it exactly as if it did not exist —
        # deliberately indistinguishable from the toolless case, because from
        # the model's point of view that tool does not exist. Without this
        # "private" would be advisory and any model that guesses the name gets
        # to call it. (A runtime-originated invocation does not come through
        # here at all.)
        if self.tool_registry is None or raw.name in private_names:
            exc: Exception = ToolNotFound(f"Unknown tool: {raw.name!r}.")
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
            self.tool_registry.create_execution(self.session, conversation_id, raw),
        )
        try:
            completed, draft, _ = await _race_cancellation(task, token, 0, None)
        except Exception as exc:
            logger.error(
                "conv=%s create_execution raised for tool=%s",
                conversation_id,
                raw.name,
                exc_info=exc,
            )
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
        conversation_id: str,
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
        task = asyncio.ensure_future(self._decide_with_middleware(conversation_id, execution))
        completed, pair, _ = await _race_cancellation(task, token, 0, None)
        return pair if completed else None

    async def _decide_with_middleware(
        self,
        conversation_id: str,
        execution: ToolExecution,
    ) -> tuple[ToolExecution, ApprovalDecision]:
        """Apply `before_permission_check` middleware, call the registry's
        `decide()`, then apply `after_permission_decision` middleware.
        Returns `(modified_execution, decision)` — the modified execution is
        what the registry saw AND the execution the decision is applied to
        and persisted (its changes are not restricted to the decide call).
        A toolless runner allows — the prepare step then produces the honest
        NOT_FOUND terminal rather than recording a false REJECTED."""
        modified = self._run_middlewares("before_permission_check", conversation_id, execution)
        if self.tool_registry is None:
            decision = ApprovalDecision(
                decision=ApprovalOption.ALLOW,
                metadata={"via": "toolless_runner"},
                created_at=self.now_ms(),
            )
        else:
            decision = await self.tool_registry.decide(
                self.session,
                conversation_id,
                modified,
            )
        return modified, self._run_middlewares(
            "after_permission_decision",
            conversation_id,
            decision,
            modified,
        )

    async def _dispatch_batch(
        self,
        conversation_id: str,
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
            async for event in self._dispatch_one(conversation_id, execution, token):
                yield event

    async def _dispatch_one(
        self,
        conversation_id: str,
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

        This is the ONLY `before_tool_execution` call site. The hook means "a
        dispatch attempt is starting" and nothing else: a call that is terminal
        at birth, rejected, refused, cancelled before dispatch, or recovered
        from an orphaned RUNNING never reaches here and never fires it. Every
        such outcome still runs `after_tool_execution`, which is the universal
        terminal transformation point."""
        execution = self._run_middlewares("before_tool_execution", conversation_id, execution)

        prepare_task = asyncio.ensure_future(self._prepare_tool(conversation_id, execution))
        try:
            prepared_ok, prepared, _ = await _race_cancellation(
                prepare_task,
                token,
                0,
                None,
            )
        except Exception as exc:
            logger.error(
                "conv=%s prepare() raised for tool=%s",
                conversation_id,
                execution.raw_tool_call.name,
                exc_info=exc,
            )
            _, event = self._finalize_outcome(
                conversation_id,
                self._terminal_for_prepare_failure(execution, exc),
                exc,
            )
            yield event
            return
        if not prepared_ok:
            _, event = self._finalize_outcome(conversation_id, self._cancelled_in_place(execution))
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
                conversation_id,
                self._terminal_for_prepare_failure(execution, exc),
                exc,
            )
            yield event
            return
        if token.cancelled:
            _, event = self._finalize_outcome(conversation_id, self._cancelled_in_place(execution))
            yield event
            return

        execution = self._persist_execution(
            conversation_id,
            execution,
            status=ExecutionStatus.RUNNING,
            started_at=self.now_ms(),
        )
        yield ToolExecutionStarted(
            conversation_id=conversation_id,
            tool_call_id=execution.tool_call_id,
            execution=execution.model_copy(deep=True),
        )
        terminal, exception = await self._run_tool_body(conversation_id, execution, prepared, token)
        _, event = self._finalize_outcome(conversation_id, terminal, exception)
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

    async def _prepare_tool(self, conversation_id: str, execution: ToolExecution) -> PreparedTool:
        """The single `registry.prepare` call site. A toolless runner raises
        `ToolNotFound` so a loaded ready execution still terminalizes
        honestly (NOT_FOUND) instead of crashing the run."""
        if self.tool_registry is None:
            raise ToolNotFound(f"Unknown tool: {execution.raw_tool_call.name!r}.")
        return await self.tool_registry.prepare(
            self.session,
            conversation_id,
            execution,
        )

    async def _run_tool_body(
        self,
        conversation_id: str,
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
                conversation_id,
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
                    logger.warning(
                        "conv=%s tool=%s timed out after %sms",
                        conversation_id,
                        current.raw_tool_call.name,
                        deadline_ms,
                    )
                    await _kill(tool_task, detach=True)  # idempotent backstop
                    return current.model_copy(
                        update={
                            "status": ExecutionStatus.TIMED_OUT,
                            "ended_at": self.now_ms(),
                        },
                    ), None
        except Exception as exc:
            logger.error(
                "conv=%s tool=%s raised",
                conversation_id,
                current.raw_tool_call.name,
                exc_info=exc,
            )
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

    def _step_limits(self, conversation_id: str) -> tuple[int, int]:
        """`(soft, hard)` for this conversation, resolved from its DEPTH.

        A subagent gets `subagent_*_max_steps` when set, and falls back to the
        main limits when they are `None`. That fallback is why `None` and `Inf`
        are different values: `None` means "same as the main conversation",
        `Inf` means "no limit at all". The bound matters because a child is
        never compaction-checked in V0 — the step limit is what stops it
        growing without one."""
        config = self.session.session_config.runtime_config
        if self.session.conversations[conversation_id].depth == 0:
            return config.soft_max_steps, config.hard_max_steps
        soft = config.subagent_soft_max_steps
        hard = config.subagent_hard_max_steps
        return (
            config.soft_max_steps if soft is None else soft,
            config.hard_max_steps if hard is None else hard,
        )

    def _is_doom_loop(self, conversation_id: str, tc) -> bool:
        """True if the current tool call would be the Nth consecutive identical
        call (same name + arguments, compared on `raw_tool_call`) in the open
        turn (where N = doom_loop_threshold). Checks the already-appended
        ToolExecution entries, so parallel tool calls are evaluated in append
        order."""
        threshold = self.session.session_config.runtime_config.doom_loop_threshold
        if threshold <= 0:
            return False
        lookback = threshold - 1
        current_turn_executions = self.ledger.open_turn_executions(conversation_id)
        subset = current_turn_executions[-lookback:]
        if len(subset) != lookback:
            return False
        return all(te.raw_tool_call.name == tc.name and te.raw_tool_call.arguments == tc.arguments for te in subset)

    def _has_unseen_user_message(self, conversation_id: str, seen: int) -> bool:
        """A `UserMessage` appended past `seen` — the path length captured in
        the same sync region as the most recent projection. One loop-local
        integer fingerprints exactly what that projection contained, because
        the path is append-only; anything user-authored past it has never been
        shown to the model.

        Only the COMPLETED close consults this: a turn must not close
        COMPLETED while its open span holds a message the model has not seen
        (an in-flight final call is indistinguishable from any other in-flight
        call, so the close site is the only place the decision can live). The
        failure closes — CANCELLED, ERRORED, TIMED_OUT, the hard-step limit —
        deliberately do not: they bury the message, and projection surfaces it
        with the next engagement."""
        entries = self.session.entries
        nodes = self.session.conversations[conversation_id].nodes
        return any(isinstance(entries.get(node_id), UserMessage) for node_id in nodes[seen:])

    def _has_unseen_post(self, conversation_id: str, last_seen: int | None) -> bool:
        """Is there a user post the model has not been shown? Both halves, the
        same pair step 3c uses for the subagent park: the DURABLE predicate
        (`open_turn_unseen_post`) plus this drive's `last_seen` fingerprint,
        which covers the one state the durable half cannot see — a post that
        landed under this drive's own in-flight LLM call, and so sits BEFORE
        the assistant entry that answered the previous round."""
        conversation = self.session.conversations[conversation_id]
        if open_turn_unseen_post(conversation.nodes, self.session.entries):
            return True
        return last_seen is not None and self._has_unseen_user_message(conversation_id, last_seen)

    def _close_turn(self, conversation_id: str, outcome: TurnOutcome, error: str | None = None) -> None:
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
            conversation_id,
            lambda entry_id, parent_id, ts: TurnFinish(
                id=entry_id,
                parent_id=parent_id,
                created_at=ts,
                outcome=outcome,
                error=error,
            ),
        )
        self._closed_outcomes[conversation_id] = outcome


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


# The spawn/stop conventions themselves — the marker names, `declares_spawn`,
# `spawn_payload`, `stop_payload`, `spawns_committed` — are pure reads over the
# data model and live in `models.py` (the projector needs `spawn_payload` too,
# and it cannot import this module).


def _unresolved_children(session: AgentSession, conversation_id: str) -> list[ChildConversation]:
    """The open turn's subagents that have not produced a result yet."""
    conversation = session.conversations[conversation_id]
    return open_turn_unresolved_children(conversation.nodes, session.entries)


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


def _to_delta_event(conversation_id: str, event) -> AgentEvent | None:
    """Translate one client stream event into its agent delta/`*Start` mirror
    (None for raw/usage events with no agent-level equivalent)."""
    if event.type == "text_start":
        return TextStart(conversation_id=conversation_id)
    if event.type == "text_delta":
        return TextDelta(conversation_id=conversation_id, text=event.delta)
    if event.type == "thinking_start":
        return ReasoningStart(conversation_id=conversation_id)
    if event.type == "thinking_delta":
        return ReasoningDelta(conversation_id=conversation_id, text=event.delta)
    if event.type == "tool_call_start":
        return ToolCallStart(
            conversation_id=conversation_id,
            tool_call_id=event.id,
            name=event.name,
        )
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


def completion_options(llm_config: LLMConfig) -> dict:
    """`LLMConfig.options` → the client completion kwargs it sets.

    Only fields the application actually configured appear, so an unset knob
    stays absent from the request and the provider's own default stands.
    Public because the same translation is what a `ContextManager` making its
    own model call needs."""
    options = llm_config.options
    if options is None:
        return {}
    return options.model_dump(exclude_none=True, exclude={"extras"})


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
