"""ToolRegistry — the runner's single contract with the tool world.

The core framework does not resolve tools, validate arguments, or decide
approvals. The `AgentSessionRunner` is constructed with one `ToolRegistry`
(or none, for a toolless agent) and touches tools through exactly four
methods — everything else (permission policies, registries with behavior,
plugins) is application space; `luca.agent.contrib.simple_tool_registry` is
the batteries-included implementation.

All four methods are async and take the live `AgentSession` first and the
`conversation_id` they are answering for second. None of them receives the
cancellation token: the runner races each call against it instead (see
`context.py`).

WHY THE CONVERSATION IS AN ARGUMENT. A session holds several conversations —
the main one and one per subagent — and they advance CONCURRENTLY. There is no
"active" conversation to read behind the caller's back, so the scope is passed.
It is always the ID, never the `Conversation` object: a conversation is LIVE
and MUTABLE (the runner appends to its `nodes` while application code holds
it), and every other cross-reference in the data model is an id too. Resolve it
when you need the object — `session.conversations[conversation_id]` — which is
one lookup, because the session is always the first argument.

That is what lets a registry answer DIFFERENTLY per conversation: withholding a
tool from a subagent, deciding approvals differently inside one, or keying its
own state by conversation.

Contract semantics:

- `get_tools` is DYNAMIC: the runner calls it fresh per LLM call, and the
  result may vary with session state AND with the conversation asked about. It
  is a query — never a lifecycle hook.
  It returns `ToolSpec`s — plain JSON-serializable data, not Python classes —
  so a registry backed by a remote tool server (MCP, an HTTP tool service,
  another agent) can hand back the JSON Schema it already has. An exception
  propagates and aborts the run: the turn is left open and resumable and the
  next `run()` asks again. The runner never substitutes an empty tool list.
- `create_execution` returns a birth DRAFT. The registry owns the call-scoped
  facts: `tool_call_id`, `raw_tool_call`, `tool_spec` (including
  `timeout_in_ms`; `None` if unresolved), `status` (PENDING, or a
  terminal-at-birth NOT_FOUND / INVALID / FAILED), `error` (a
  `ToolExecutionError` for terminal births — the registry authors it), and
  `extras`. It carries no identity — `id` / `created_at` stay `None`; the
  RUNNER stamps `id`, `parent_id`, `created_at`, `finished_at` (when terminal at
  birth), `context_tokens`, and `is_doom_loop_flagged`, so ids and timestamps
  keep flowing through `generate_id()` / `now_ms()`. Registries stay unaware
  of the storage scheme: never compute a `spec_id` and never touch
  `session.tool_specs` — the framework's write doors do both.
- `decide` exceptions propagate and abort the run; the executions stay
  unresolved and the next `run()` asks again. Implementations must be
  idempotent queries of their own state — record answers out-of-band, return
  them when asked. A PENDING decision parks only that execution.
- `prepare` resolves the tool and validates the arguments, then returns a
  CALLABLE that runs the body. Raising means the body never runs, and the
  execution is never marked RUNNING: `ToolNotFound` → NOT_FOUND;
  `InvalidToolArguments` or a pydantic `ValidationError` → INVALID; anything
  else → FAILED — all with no `ExecutionAttempt` appended and
  `dispatched=False`. Once the callable is invoked, every raise is FAILED. A
  returned `ExecutionResult` → COMPLETED (after
  `ContextManager.process_tool_output`). A returned `ExecutionDeferred` →
  AWAITING_RESULT: the call is RE-DISPATCHED from scratch on a later drive,
  through this same `prepare()`, which is exactly why the contract already
  requires it to be re-callable and idempotent. `prepare` is called once per
  dispatch attempt and only for an already-approved execution: an
  approval-deferred or denied call is never prepared, and a parked one is never
  re-decided.

The prepared callable:

    PreparedTool = Callable[..., Awaitable[ExecutionResult | ExecutionDeferred]]

    async def run(
        *, cancellation_token: CancellationToken
    ) -> ExecutionResult | ExecutionDeferred:

It takes the run's cancellation token and nothing else. It must be a
CALLABLE, not a coroutine object: the runner may never invoke it (a
cancellation landing between the return and the invocation), and a bare
coroutine would emit `RuntimeWarning: coroutine was never awaited`. Calling it
must produce an awaitable; a callable that returns a plain value has already
been invoked, so it records as a post-dispatch failure (FAILED, with its
`ExecutionAttempt` appended) rather than a preparation failure.

WHAT THE ARGUMENTS MEAN. The `AgentSession` is the LIVE object — the same
instance the runner and ledger write through, not a copy. A registry may hold
the reference and re-read current state from it later, including from inside a
prepared callable, where `session.entries[execution.id]` is the correct way to
see the execution's current durable state. The `ToolExecution` is a SNAPSHOT:
a detached copy as of the moment the call was made. The runner persists
RUNNING and the new `ExecutionAttempt` AFTER `prepare()` returns, and
persistence stores a copy, so a reference captured during `prepare()` is stale by the time the
callable runs. Capture what the callable needs during `prepare()`; do not read
execution state through the captured object.

THE SESSION IS READ-ONLY to every implementation. The runner owns every write
to session state; a strategy that mutates what it is handed corrupts the
ledger.

RULES FOR REGISTRY AUTHORS. Contract requirements, not suggestions. Every one
of them fails silently when violated.

 1. `prepare()` must be safe to call more than once for the same execution.
    If the process dies during preparation the execution is still PENDING and
    the next drive calls `prepare()` again. The constraint is on CALL-SCOPED
    side effects: do not consume a token, advance a cursor, or write a record
    keyed to this call. Registry-lifecycle work is fine and idempotent by
    nature — lazy connect-on-first-use behind an already-connected check,
    warming a catalog, refreshing a shared credential.
 2. `prepare()` must not return while holding anything that needs releasing.
    `async with` INSIDE `prepare()` is fine: a cancellation raises inside it
    and its `finally` runs. What is forbidden is handing back a callable while
    a lock, lease, sandbox, connection or concurrency slot is held — the
    runner may never invoke that callable and nothing runs a cleanup path for
    it. Acquire inside the returned callable, under `try/finally`.
 3. `prepare()` must not block. No deadline applies to it — `timeout_in_ms`
    bounds the BODY only. It is raced against the run's cancellation token, so
    a stuck preparation is not unkillable, but that race is a safety net, not
    a license: the only thing that ends a hung `prepare()` is an explicit
    `cancel()` from the application. Resolve from local state — a static dict,
    a cached catalog. A registry fronting a remote tool server keeps a cached
    tool list refreshed out of band and does all of its network work inside
    the callable.
 4. The returned callable is where wrapping belongs. Retries, rate limiting,
    metrics, tracing, exception translation and result post-processing all go
    in a closure the registry owns. Returning a tool's bound method directly
    is valid but gives all of that up.
 5. Capture what the callable needs during `prepare()`. It receives only
    `cancellation_token`.
 6. Any of the four methods may be cancelled at any `await`. Anything acquired
    inside them must be released in a `try/finally` or `async with` —
    cancellation raises `asyncio.CancelledError` at the current await point
    (not luca's own `CancelledError`, which only a cooperative tool body ever
    sees) and normal unwinding applies, but nothing is released automatically.
 7. Never swallow `asyncio.CancelledError`. The runner waits for a cancelled
    registry call to finish unwinding before it proceeds, precisely so that
    cleanup completes. A method that catches the cancellation and keeps
    working makes that wait unbounded.
 8. Blocking synchronous work must run in `asyncio.to_thread`. A cancellation
    cannot interrupt a blocking syscall, so synchronous I/O inside these
    methods blocks the event loop regardless of cancellation handling.
 9. Nothing runs on a hard process kill. A registry holding something that
    must not leak — a distributed lock, a remote lease — needs a TTL or a
    startup reconciliation pass, not a `finally`.
10. `decide` must be an idempotent query of your own state. Record answers out
    of band and return them when asked.
11. THE CORE NEVER VALIDATES ARGUMENTS. It now knows each tool's input schema
    but will never check a call against it, because a registry may delegate to
    a remote server that validates on its own side and double validation would
    break it. Validation is yours. Deliberate, not an oversight.
12. Do not put volatile data in `ToolSpec.metadata`. A spec must be a pure
    function of the tool definition; anything call-scoped mints a new stored
    row on every call and silently defeats normalization.
13. BE CONCURRENCY-SAFE. One registry instance serves every conversation, and
    the framework does NOT serialize these calls — serializing them would gate
    every sibling subagent behind the slowest one, which is exactly what
    parallel subagents exist to avoid. `get_tools`, `prepare` and the prepared
    callable are now concurrent ACROSS conversations (`create_execution` and
    `decide` already were, across sibling calls in one response). Six rules:
    a. NO PER-CALL STATE ON `self`. `self` is for immutable configuration.
       Anything stashed there and read back after an `await` may belong to
       another conversation by then. This is the failure mode, and it is
       silent — a tool that returns another subagent's answer, intermittently,
       with nothing in the log.
    b. State keyed by `conversation_id` needs NO lock. Tool dispatch within one
       conversation is sequential, so exactly one body ever touches a given
       conversation's slot at a time. (Sequential dispatch is a runner CHOICE,
       documented as such; if it ever becomes parallel this line is revisited.)
    c. State deliberately shared ACROSS conversations needs an `asyncio.Lock`
       around the MUTATION, never around the I/O. Holding a lock across a slow
       await re-serializes the tree. Do the I/O unlocked, take the lock for the
       read-modify-write.
    d. `asyncio.to_thread` is REAL parallelism. Two conversations' bodies then
       genuinely run on two OS threads, and nothing the single-threaded
       argument buys you survives that boundary: shared state touched inside
       needs a `threading.Lock`, on both sides.
    e. Cancellation can kill a body at any await, and the LOCK is not what is
       at risk — `async with` releases correctly. The DATA can be left
       half-written and a sibling may read it. Make the mutation the last
       await-free step, or make it idempotent.
    f. Process-global and external resources are not a locking problem. The
       current working directory, relative paths, temp files, subprocesses, a
       git worktree, a connection that cannot have two queries in flight —
       scope them PER CONVERSATION instead. A mutex around a shared cwd is the
       wrong shape.

COMPOSING REGISTRIES. A registry that routes to children must gate a call
through its owning child's permission policy on every path where it can
resolve it — including a cold cross-process resume, where no tool listing has
happened yet and the drive reaches `decide` and dispatch before any LLM call.
A call that resolves must never run unpermissioned.

NOTHING MAY CRASH THE RUN. Every failure path terminalizes one execution and
leaves its siblings alone.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from .models import (
    AgentSession,
    ApprovalDecision,
    ExecutionDeferred,
    ExecutionResult,
    ToolCall,
    ToolExecution,
    ToolSpec,
)

# What `prepare()` hands back: a callable taking only the run's cancellation
# token, whose invocation produces the awaitable that runs the tool body.
# `ExecutionDeferred` is the tool saying "I cannot return the final result
# yet"; the call parks and is re-dispatched from scratch on a later drive.
PreparedTool = Callable[..., Awaitable[ExecutionResult | ExecutionDeferred]]


class ToolRegistry:
    """The four-method tool-lifecycle contract. A duck-typed concrete base
    (no ABC), matching the house strategy style — subclass and override."""

    async def get_tools(
        self,
        session: AgentSession,
        conversation_id: str,
    ) -> list[ToolSpec]:
        raise NotImplementedError

    async def create_execution(
        self,
        session: AgentSession,
        conversation_id: str,
        call: ToolCall,
    ) -> ToolExecution:
        raise NotImplementedError

    async def decide(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> ApprovalDecision:
        raise NotImplementedError

    async def prepare(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> PreparedTool:
        raise NotImplementedError
