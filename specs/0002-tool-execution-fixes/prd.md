# Tool Execution Refactor

Status: ready to implement
Scope: `luca/agent/core`, `luca/agent/contrib`, `tests/agent`, `docs/agent`, `AGENTS.agent.md`

This document is the complete and only specification for this refactor. It
assumes no prior context: §2 describes the system as it stands today, §3 what is
wrong with it, and everything after that what to build. Nothing else needs to be
read to implement it.

**How to read it.** §1–§14 are binding: they define the contracts and the
observable behavior, and they are sufficient on their own to implement the
change. §15 is implementation notes — one way to get there, useful but not
authoritative; if a note contradicts the behavior sections, the behavior
sections win. §16 records what stays out of scope and why.

---

## 1. Summary

Six changes to how the agent runs tools. They ship as ONE commit: one
submodule, one sweep of contrib, tests and docs, and no released consumers to
stage for.

They are not all interdependent, and the document should not pretend otherwise.
Changes 1–2 (`ToolSpec` + normalization) and changes 3–6 (session signatures +
cancellability + `prepare` + the context refresh) are two independent stacks
that can be built in either order — §15.6's build order rests on exactly that.
What forces the single commit is the churn, not the dependency graph.

1. **`ToolSpec` replaces `Tool` as the core's tool contract.** The core stops
   reaching into a Python class for a tool's input schema. `ToolSpec` carries
   the schema as JSON, `get_tools` returns `list[ToolSpec]`, and `Tool` moves
   to contrib as an ergonomic helper for people writing tools in Python.

2. **Tool specs are stored once per session.** A content-addressed
   `session.tool_specs` store holds each distinct spec; executions reference it
   by id instead of embedding a copy.

3. **`AgentSession` replaces `ToolContext` everywhere.** Every method on
   `ToolRegistry` and `ContextManager` receives the live session as its first
   argument. `ToolContext` is deleted. `get_tools` becomes async.

4. **`ToolRegistry.execute` is replaced by `ToolRegistry.prepare`.** Not added
   alongside it — `execute` is removed from the contract entirely. `prepare`
   resolves the tool and validates the arguments, then returns a callable that
   runs the body; invoking that callable is what `execute` used to do. The
   runner writes the durable `RUNNING` row only once `prepare` has returned, so
   a tool call that never resolved is no longer recorded as one that ran.

5. **Every registry call is cancellable.** All four registry awaits are raced
   against the run's cancellation token, so a hung registry or tool-owned
   preflight can no longer make `cancel()` a no-op.

6. **`ContextManager` gets a refresh path.** Change 3 lets it count tokens
   against the active model, which makes every stored count stale the moment
   that model changes. The runner gains one public method,
   `recalculate_context_tokens()`, to re-derive them. Nothing calls it — no
   constructor flag, no CLI flag, no automatic invocation (§6.8).

The registry contract goes from
`get_tools` / `create_execution` / `decide` / `execute` to
`get_tools` / `create_execution` / `decide` / `prepare`, with a uniform
signature shape across all four.

---

## 2. Background

Read this even if you know the codebase — the rest of the document assumes
these specific facts.

### 2.1 The runner never touches tools directly

`AgentSessionRunner` is constructed with one `ToolRegistry` (or `None`, for a
toolless agent) and drives the entire tool lifecycle through it:

```python
class ToolRegistry:                                       # TODAY
    def get_tools(self, agent_session) -> list[Tool]: ...
    async def create_execution(self, call, context) -> ToolExecution: ...
    async def decide(self, tool_execution, context) -> ApprovalDecision: ...
    async def execute(self, tool_execution, context, *, cancellation_token) -> ExecutionResult: ...
```

Tool resolution, argument validation and permission policy all live behind that
contract, in `contrib` or application space. The core owns none of it. The
batteries-included implementations are `SimpleToolRegistry` and
`ProxyToolRegistry` in `luca/agent/contrib/simple_tool_registry/`.

### 2.2 A tool call becomes a durable entry

Every tool call the model emits produces exactly one `ToolExecution` entry in
the session's append-only entry store. It is born at `create_execution`
(`PENDING`, or terminal at birth — `NOT_FOUND` / `INVALID` / `FAILED`), gated by
`decide`, and run by `execute`.

The registry returns a *draft*; the runner stamps identity (`id`, `parent_id`,
`created_at`) and the derived fields. `ToolExecution.started_at` is documented
as "set iff the body was dispatched", and the derived `dispatched` property
reads exactly that. `tool_spec: ToolSpec | None` is a snapshot of the resolved
tool's identity taken at birth, so a saved conversation stays interpretable even
if the registry later changes or drops the tool; `None` means the tool never
resolved.

Before invoking `execute`, the runner persists `status=RUNNING` and
`started_at`. That durable row is what makes crash recovery safe: on the next
drive, a persisted-`RUNNING` execution with no live task is terminalized as
`INTERRUPTED` and never re-dispatched. Without the row, a crash mid-body would
leave the execution `PENDING` and the next drive would run a side-effecting tool
twice.

**Framework invariant: every tool call produces exactly one tool output,
always.** Whatever happens to a call or its siblings, the model sees one result
per call it requested.

### 2.3 Cancellation is a pure signal

`runner.cancel()` is synchronous. It appends a durable `CancelRequested` entry,
trips a flag on the run's `CancellationToken`, sets status to `CANCELLING`, and
returns. It interrupts nothing by itself.

What gives the flag effect is `_race_cancellation`: it races an awaited task
against the flag, honors a grace window, and kills the task if the flag wins.
Today it wraps the LLM call, the streaming steps, compaction and the tool body —
and nothing else.

Wind-down happens at the drive loop's step boundaries: every still-`PENDING`
execution in the open turn is stamped `cancel_signalled_at` and becomes
`CANCELLED` — resultless, errorless, approval state untouched — and the turn
closes with the requested outcome.

### 2.4 The two collaborator contracts

`ToolRegistry` (above) and `ContextManager`, the context-accounting strategy:

```python
class ContextManager:                                     # TODAY
    def calculate_context(self, entry: Entry) -> int
    def prune_entry(self, entry: Entry) -> PrunedEntry
    def process_tool_output(self, execution_result: ExecutionResult) -> ExecutionResult
```

`ToolContext` is a transient Pydantic model built once per run and passed to
registry and tool calls. It holds two fields: `session_id: str` and
`model: LLMConfig`.

---

## 3. Problems

### P1 — The core depends on a Python class

`luca` is meant to be an implementation-agnostic specification: someone should
be able to write `luca-ts` against the same data model and the same contracts.
Today they cannot, in one place. The core obtains a tool's input schema by
reading a `ClassVar` off a Python class (`tool.Args`) and hands a Pydantic model
class to the transport layer (`core/adapter.py:56`).

This makes the `ToolRegistry` contract dishonest about itself. It claims the
runner touches tools through four methods, but one of them returns Python
objects the core then reaches into for a field the contract never mentions.

The practical cost is immediate. A registry backed by a remote tool server
(MCP, an HTTP tool service, another agent) has JSON Schema and no Pydantic
class. Today it must fabricate a `Tool` subclass with a synthetic `Args` model
purely to get past the adapter. And the core is stricter than the client it
wraps: `luca.client.Tool.parameters` already accepts `dict | type |
TypeAdapter`.

The coupling itself is small. In the entire core, `Tool` appears in three
places: the return type of `get_tools`, the adapter, and the re-export. The
runner reads `tool_spec` in exactly one place — `tool_spec.timeout_in_ms` at
dispatch. Projection, events, the ledger, the context manager, compaction and
middleware never mention `Tool` at all.

### P2 — Tool specs are duplicated into every execution

Each `ToolExecution` embeds a full `ToolSpec` copy. A session that calls the
same tool fifty times stores fifty identical specs. Measured on the `shell`
contrib tools:

| Tool | description | JSON schema |
|---|---|---|
| `bash` | 4067 bytes | 566 bytes |
| `edit` | 1361 bytes | 627 bytes |
| `read` | 1158 bytes | 540 bytes |

A 200-call session carries on the order of 900 KB of duplicated spec text. This
is disk and transport cost only — context accounting never reads `tool_spec`, so
token counts and compaction thresholds are unaffected. Adding the input schema
(required by P1) makes it worse, which is why both ship together.

### P3 — The contracts hand implementations less than they need

**`ToolRegistry` speaks four vocabularies.**

| method | session | ToolContext | token | async |
|---|---|---|---|---|
| `get_tools` | yes | — | — | no |
| `create_execution` | — | yes | — | yes |
| `decide` | — | yes | — | yes |
| `execute` | — | yes | yes | yes |

`get_tools` receives the full `AgentSession` and is documented as dynamic — its
answer may vary with session state. The other three cannot see the state
`get_tools` is licensed to vary on. A registry that hides a tool based on
session state has no way to reason about the resulting call, or to explain the
`NOT_FOUND` it produces.

**`get_tools` is synchronous.** Any registry that needs I/O to list its tools —
a remote tool server, a plugin host, a permissions service — cannot implement it
without blocking the event loop.

**`ToolContext` adds nothing.** Both fields are already reachable from
`AgentSession` (`session.id`, `session.session_config.llm_config`). It is a
lossy copy of the object callers actually want. It is also un-extensible
(`extra="forbid"`, constructed inline in `_begin_run`, no `build_tool_context`
middleware while `build_model_string` and `build_tool_list` both have one), so
an application has nowhere to pass ambient state to its tools.

**`ContextManager` is internally inconsistent.** `calculate_context` and
`prune_entry` receive a whole `Entry`; `process_tool_output` receives only the
`ExecutionResult` — no tool name, no `tool_spec`, no arguments. The one policy
in the class that most obviously needs to vary by tool is the only one that
structurally cannot. "Truncate `bash` output at 30k characters but never
truncate `read`" is not expressible.

**Token counting cannot be model-aware.** `calculate_context` sees no model
configuration, so an implementation using a real tokenizer has nothing to select
on. The default character estimate works without it; anything accurate does not.

### P4 — `execute()` fuses resolution with execution

`execute()` does three separate things behind one call: resolve the tool name,
validate the stored arguments, and run the body. The runner persists `RUNNING`
and `started_at` before calling it — so it records that a tool is running before
the registry has established that there is anything to run.

**Concrete failure.** A tool call is created while its tool exists. The
permission policy defers; the run parks awaiting approval. The application
resumes later, and the registry — which may vary its tool list with session
state, and may be running a newer build of the host application — no longer
offers that tool. The runner persists `RUNNING` with `started_at=1000`, calls
`execute()`, which raises `ToolNotFound`. The durable record reads
`status=NOT_FOUND`, `started_at=1000`, `dispatched=True`, `duration_ms=5` — a
complete description of a body that never existed. The same failure detected one
phase earlier, at `create_execution` time, records `started_at=None` and
`dispatched=False`. One outcome, two durable shapes, and no consumer can
reliably answer "did this tool actually run?"

Two workarounds exist in the runner only because of this fusion, and both are
unreliable:

- **The failure phase is guessed from the exception type.** After `execute()`
  raises, the runner maps `ToolNotFound` → `NOT_FOUND`, `InvalidToolArguments` /
  pydantic `ValidationError` → `INVALID`, anything else → `FAILED`. A tool body
  that looks up a sub-resource and raises `ToolNotFound` is indistinguishable
  from a registry that could not resolve the tool.
- **The recorded failure phase is inferred from `started_at`.** The durable
  error's `details["phase"]` is `"execution"` when `started_at` is set and
  `"create_execution"` when it is not — which, given the above, is a tautology
  rather than a fact.

Neither is fixable inside the runner: it cannot know which of `execute()`'s
three jobs raised, because the contract hands it one call.

A secondary cost: `SimpleToolRegistry` resolves and validates at birth, then
again in `execute`, discarding the validated arguments both times.

### P5 — Cancellation coverage is asymmetric

Only `execute` is raced against the token. `create_execution` and `decide` are
awaited with a plain `asyncio.gather`, and `get_tools` is called outside any
race. All three run arbitrary application code — in the shipped registry,
`create_execution` awaits the tool's own `get_approval_context`, which may do
network or filesystem I/O.

**Concrete failure.** A tool's `get_approval_context` hangs on a network lookup.
The user cancels. `cancel()` writes its record, trips the flag, returns. Nobody
is watching. The runner stays inside `gather()`, session status reads
`CANCELLING` indefinitely, and the only way out is killing the process.
`Tool.timeout_in_ms` does not help — it is read at dispatch, after both calls
have already completed.

---

## 4. Goals and non-goals

### Goals

- `ToolSpec` is the only tool-related type the core knows about, and it is fully
  JSON-serializable and language-neutral.
- `Tool` survives as an ergonomic convenience for Python tool authors; nothing
  in the core depends on it.
- Each distinct `ToolSpec` is stored once per session. A spec that changes
  between runs produces a new stored spec; historical executions keep pointing
  at the version that was live when they ran.
- Both collaborator contracts speak one vocabulary and see the same state.
- `started_at` / `dispatched` mean what they say, for every outcome.
- `NOT_FOUND` and `INVALID` mean resolution and validation failed, never that a
  tool body raised a similarly-named exception.
- No registry or tool-owned code can make `cancel()` a no-op.
- No change to the set of execution statuses, the approval model, the event
  types, or crash-recovery semantics.

### Non-goals

- **The core never validates arguments.** It will now *know* each tool's input
  schema but must still never validate a call against it. Validation stays the
  registry's job, because a registry may delegate to a remote server that
  validates on its own side; double validation would break it. Deliberate, not
  an oversight, and it belongs in the `ToolRegistry` docstring.
- **No per-turn tool manifest.** `session.tool_specs` holds specs referenced by
  an execution — tools that were actually *called*. Recording the full set
  *advertised* on each turn ("was tool Y even offered?") is a separate feature.
- **No migration of existing session files.** Pre-v0; regenerate them. No
  compatibility path. They will not silently half-work — see §8.
- **No garbage collection of `tool_specs`.** Append-only, like the entry store.
- **No new `ExecutionStatus` value.** Nothing can observe the interval between
  `prepare()` returning and `started_at` being written, so it has no durable
  representation.
- **No parallel tool dispatch.** Dispatch stays sequential. §16 covers what this
  change constrains about a future concurrent scheduler.
- **No deadline on the non-body registry calls.** See §6.6.
- **No `build_tool_list` middleware over `ToolSpec`.** The hook keeps receiving
  the post-adapter wire list. See §16.
- **No new contract method for resolution** (`owns(name)` / `resolve(name)`).
  §6.4 states the composing-registry requirement behaviorally; meeting it is the
  composing registry's problem.
- **No `build_tool_context` middleware.** `ToolContext` is being deleted, not
  extended; per-run application state is not the framework's concern (§10).

### The accepted trade-off

Naming it up front, because it is the first thing a reviewer will raise.

After this change `ToolSpec` serves two roles with different natural lifetimes:
it is both the **advertisement** sent to the model (name, description, input
schema) and the **historical identity snapshot** attached to a past execution
(kind, namespace, version, timeout). Those are not obviously the same object,
and unifying them means every persisted execution references the full
advertisement forever.

We accept it for two reasons. Normalization removes the storage cost that made
the conflation expensive — one stored copy per distinct spec, not one per call.
And having the exact schema the model was shown attached to the execution is
what makes an old session auditable and replayable, which the current design
cannot do at all.

The alternative — two types, a thin spec for history and a fat one for the wire
— was rejected: it reintroduces the same split this change exists to remove, and
forces every registry to produce both.

---

## 5. The contracts after this change

### 5.1 `ToolRegistry`

```python
class ToolRegistry:
    async def get_tools(self, session: AgentSession) -> list[ToolSpec]: ...
    async def create_execution(self, session: AgentSession, call: ToolCall) -> ToolExecution: ...
    async def decide(self, session: AgentSession, tool_execution: ToolExecution) -> ApprovalDecision: ...
    async def prepare(self, session: AgentSession, tool_execution: ToolExecution) -> PreparedTool: ...
```

All four are async, all four take the session first, none of them receives the
cancellation token.

`prepare()` resolves the tool and validates the arguments, then returns a
callable that runs the body. Raising means the body never runs.

### 5.2 The prepared callable

```python
PreparedTool = Callable[..., Awaitable[ExecutionResult]]

async def run(*, cancellation_token: CancellationToken) -> ExecutionResult: ...
```

It takes the run's cancellation token and nothing else.

It must be a **callable, not a coroutine object**. If the runner never invokes
it — a cancellation landing between the return and the invocation — a bare
coroutine emits `RuntimeWarning: coroutine was never awaited`, and
`pyproject.toml` sets `filterwarnings = ["error"]`, so that fails the build. A
callable is inert until called.

**Calling it must produce an awaitable.** The runner hands the invocation to
`asyncio.ensure_future`, which raises `TypeError` synchronously for a plain
value. Today that call sits *outside* the body's failure handling
(`runner.py`, `_run_tool_body`), so a registry returning a plain `def` would
crash the whole run instead of failing one execution. Move it inside. A
callable that returns a non-awaitable has already been invoked, so the honest
record is a post-dispatch failure, not a preparation failure — `FAILED` with
`started_at` set (§6.4).

`prepare()` is called once per dispatch attempt, and only for an execution that
has already been approved. A deferred or denied call is never prepared.

### 5.3 `ContextManager`

```python
class ContextManager:
    def calculate_context(self, session: AgentSession, entry: Entry) -> int
    def prune_entry(self, session: AgentSession, entry: Entry) -> PrunedEntry
    def process_tool_output(
        self,
        session: AgentSession,
        execution: ToolExecution,
        result: ExecutionResult,
    ) -> ExecutionResult
```

`prune_entry` has **no framework call site** — the runner never calls it. An
application calls it and hands the template to `SessionLedger.prune()` (see
`docs/agent/11-context-and-usage.md`), so the caller already holds the session.
Its session argument is uniformity, not need; it changes anyway so that one
argument order describes all three methods. Do not go looking for the runner
call site.

### 5.4 `Tool` (contrib)

`Tool` moves out of the core into `luca/agent/contrib/tools.py`, unchanged in
spirit. Two adjustments:

- `get_tool_spec()` now also stamps `input_schema` from the tool's `Args`
  model: `Args.model_json_schema()`, verbatim. That is byte-identical to what
  the transport derives from `Args` today
  (`client/types/tools.py::tool_parameters_to_json_schema`), so the wire payload
  does not change — the schema is computed at snapshot time instead of at send
  time.
- `execute` / `_execute` / the duck-typed `get_approval_context` take
  `session: AgentSession` where they took `ToolContext`.

It does not move into the registry package: `resource_permissions`, `shell` and
`memory` all build on `Tool` without going through any particular registry.

### 5.5 What the two arguments mean

Both arguments are easy to misuse, in opposite directions.

**The `AgentSession` is the live object.** It is the same instance the runner
and ledger write through, not a copy. A registry may hold the reference and
re-read current state from it later — including from inside a prepared callable,
where re-reading `session.entries[execution.id]` is the correct way to see the
execution's *current* durable state.

**The session is read-only to every implementation.** The runner owns every
write to session state. A strategy that mutates what it is handed corrupts the
ledger. This includes `session.tool_specs` — filing specs there is the framework's job
(§6.9), and a registry that writes there is out of contract.

**The `ToolExecution` is a snapshot.** It is a detached copy of the entry as of
the moment the call was made. The runner persists `RUNNING` and `started_at`
*after* `prepare()` returns, and persistence stores a copy, so a reference
captured during `prepare()` is stale by the time the callable runs. Capture what
the callable needs during `prepare()`; do not read execution state through the
captured object.

---

## 6. Behavior

### 6.1 Advertising tools to the LLM

Before each LLM request the runner asks the registry for the tools to
advertise and receives `ToolSpec` objects. It converts each into the client's
wire tool definition using `name`, `description` and `input_schema`. The
client's wire type already accepts a raw JSON Schema dict for `parameters`, so
nothing changes in the client layer.

The runner's `build_tool_list` becomes async, because `get_tools` is. The
`build_tool_list` **middleware hook stays synchronous** and unchanged: it runs
on the converted wire list, after the await.

The whole `build_tool_list` step is raced against the cancellation token. When
the token wins, no tool list is produced and no LLM call is made; control
returns to the drive loop's top, which winds the turn down.

`get_tools` raising is treated like `decide` raising: the exception propagates
and aborts the run, the turn is left open and resumable, and the next `run()`
asks again. The runner does not substitute an empty tool list — calling the
model with no tools when the registry meant to offer some would silently change
the answer.

### 6.2 Birth — `create_execution`

Unchanged from the caller's perspective, except for the signature. The registry
returns a draft `ToolExecution` carrying the resolved tool's `ToolSpec` (or
`None` if it did not resolve), the birth `status`, the `error` for a terminal
birth, and `extras`. Registries remain unaware of the storage scheme: they never
compute a spec id and never touch `session.tool_specs`.

Drafts for the calls in one assistant response are still produced concurrently,
one per call, and persisted in model-request order. Failures stay isolated per
call: a raising `create_execution` becomes a runner-synthesized `FAILED` draft;
a toolless runner synthesizes `NOT_FOUND`.

### 6.3 Approval — `decide`

Unchanged, except for the signature and cancellability. Exceptions still
propagate and abort the run, leaving the executions unresolved so the next
`run()` asks again; implementations must be idempotent queries of their own
state. A `PENDING` decision still defers only that execution.

### 6.4 Dispatch — prepare, then run

For each ready (`PENDING` + `ALLOWED`) execution, in order:

0. If a cancellation was already observed before this execution's turn came up,
   the batch stops here. This execution and every one after it are untouched —
   still `PENDING`, no middleware fired — and the loop-top wind-down terminalizes
   them. Only the executions the batch actually reached are the dispatch path's
   to finish.
1. `before_tool_execution` middleware runs. Its returned `raw_tool_call` is the
   effective call — this is what `prepare()` resolves and validates from, which
   is why the hook stays ahead of `prepare()`.
2. `prepare()` is called, raced against the cancellation token.
3. If `prepare()` raised, or returned something that is not callable, the
   execution terminalizes without ever being marked `RUNNING`. For the
   not-callable case the runner synthesizes the failure itself: an `AgentError`
   naming the tool and what came back, `details["phase"] = "prepare"`.
4. If a cancellation was observed at any point up to and including `prepare()`
   settling, the body is **not** dispatched — even when `prepare()` returned
   successfully. The grace window exists to let in-flight work finish, not to
   start new work after a cancellation was requested.
5. Otherwise the runner persists `status=RUNNING` and `started_at`, emits
   `ToolExecutionStarted`, and invokes the callable under the cancellation race
   and the deadline. The invocation itself must sit inside the failure
   handling: a callable that returns a plain value rather than an awaitable has
   *already been invoked*, so it is recorded like any other post-dispatch
   failure — `FAILED`, `started_at` set, `phase="execution"` — and the
   `TypeError` must not escape (§5.2). Testing awaitability any earlier would
   mean invoking the body before `RUNNING` is durable, which is exactly what
   this change exists to prevent.

The full outcome table:

| What happened | status | `started_at` | `dispatched` | `details["phase"]` |
|---|---|---|---|---|
| `create_execution` raised | FAILED | `None` | False | `create_execution` |
| toolless runner, at birth | NOT_FOUND | `None` | False | `create_execution` |
| registry authored a terminal draft | NOT_FOUND / INVALID / FAILED | `None` | False | registry's own |
| `decide` returned DENY | REJECTED | `None` | False | — |
| `prepare` raised `ToolNotFound` | NOT_FOUND | `None` | False | `prepare` |
| `prepare` raised `InvalidToolArguments` / pydantic `ValidationError` | INVALID | `None` | False | `prepare` |
| `prepare` raised anything else | FAILED | `None` | False | `prepare` |
| `prepare` returned a non-callable | FAILED | `None` | False | `prepare` |
| callable raised (any exception type) | FAILED | set | True | `execution` |
| callable returned a non-awaitable | FAILED | set | True | `execution` |
| callable returned | COMPLETED | set | True | — |
| deadline expired on the callable | TIMED_OUT | set | True | — |
| cancel grace expired on the callable | INTERRUPTED | set | True | — |
| cancelled up to and including `prepare()` settling | CANCELLED | `None` | False | — |
| crash after `RUNNING` was persisted | INTERRUPTED | set | True | — |

Three consequences worth stating separately:

- **The exception-type-to-status mapping does not disappear — it moves.** It
  applies only to `prepare()`, where it is accurate, because the only work done
  at that point is resolution and validation. Once the callable is invoked,
  every raise is `FAILED`.
- **`ToolExecutionStarted` is emitted iff the body was dispatched.** It fires
  strictly less often than today: a call that fails to resolve after approval no
  longer produces one.
- **`before_tool_execution` fires exactly once per dispatch attempt.** Anything
  that terminalizes an execution after that hook has run — a `prepare()`
  failure, a cancellation during or after `prepare()` — must go through the
  shared outcome tail and must not re-run it. A crash during `prepare()` starts
  a fresh attempt on the next drive and fires it again (§8).

**Composing registries.** Giving the dispatch path a resolution that no longer
depends on a warmed routing cache removes the accident that currently makes a
cache miss harmless. Any registry that routes to children must gate a call
through its owning child's permission policy on every path where it can now
resolve it — including a cold cross-process resume, where no tool listing has
happened yet and the drive reaches `decide` and dispatch before any LLM call. A
call that resolves must never run unpermissioned.

**Toolless runner.** With no registry, the prepare step raises `ToolNotFound`,
so a loaded ready execution terminalizes honestly as `NOT_FOUND` instead of
crashing the run — the same treatment the toolless case already gets at
`create_execution` time.

### 6.5 Cancellation — one rule

All four registry awaits — `get_tools`, `create_execution`, `decide`,
`prepare` — plus the prepared callable are raced against the run's cancellation
token. When the token wins, the runner kills the await and continues; the
registry never learns that a token exists.

**Cancellation never writes a terminal status of its own inside a registry
phase.** It stops the phase and leaves the call in a state the existing
machinery already knows how to finish:

| Cancelled during | What the runner does | Durable outcome |
|---|---|---|
| `get_tools` | no LLM call; return to the loop top | turn winds down; executions unaffected |
| `create_execution` | synthesize a `PENDING` draft so the call still gets its one execution | wind-down records CANCELLED |
| `decide` | record no decision; approval state untouched | wind-down records CANCELLED |
| before a ready execution's dispatch begins | leave it `PENDING`, stop the batch | wind-down records CANCELLED |
| `prepare` | no `RUNNING` row, no `ToolExecutionStarted` | dispatch path records CANCELLED **in place** |
| the callable | unchanged grace machinery | COMPLETED / FAILED / INTERRUPTED |

The `prepare` row is the one asymmetry, and it is deliberate:
`before_tool_execution` has already fired for that call, so it cannot be handed
back to the loop-top wind-down, which fires it again. The dispatch path stamps
`cancel_signalled_at`, sets `CANCELLED`, and finalizes through the shared
outcome tail — same durable shape as a wind-down cancellation, one hook
invocation. **The hook is the boundary**: an execution whose hook has not fired
belongs to the wind-down, one whose hook has fired belongs to the dispatch path.

**The `decide` race wraps the middleware pair with it.** The runner races
`before_permission_check` → `decide` → `after_permission_decision` as one unit,
not the registry call alone, so a token already tripped when the batch starts
fires no hook at all. A token tripping *during* `decide` still leaves
`before_permission_check` fired with its returned execution discarded: the
execution has to stay `PENDING` for the wind-down, so there is nowhere to put
it, and a decision that never happened has nothing to apply.

**`before_tool_execution` is the only hook with an exactly-once, paired
guarantee** (§8). Every other hook may fire without its result being persisted.
Say so in the middleware docs rather than letting each hook's behavior be
discovered.

Every execution that ends up `CANCELLED` is resultless and errorless, with
`cancel_signalled_at` and `ended_at` stamped, `started_at` unset and
`dispatched` False — whichever of the two paths recorded it. For two calls that
reached the SAME lifecycle point, the two paths must produce byte-identical
records apart from the timestamps; §14 asserts exactly that, on a two-call batch
where both were born normally and both were allowed. Calls that reached
different points legitimately differ in the fields they never got to fill: a
cancelled birth carries `tool_spec=None`, `approval_status=None` and no
decisions, where a call cancelled at `prepare()` carries a `tool_spec` and
`approval_status=ALLOWED`. (An execution cancelled while its body
was already in flight does not end up `CANCELLED` at all: it is `COMPLETED`,
`FAILED` or `INTERRUPTED` per the grace machinery, with `started_at` set and
`cancel_signalled_at` stamped.)

**A cancelled birth is CANCELLED, never FAILED.** A cancellation is not a tool
failure and must not be recorded as one.

**A response containing N tool calls yields N tool executions**, even when
cancellation lands mid-batch. A cancelled batch may not return a short list.

**Cancellation is not delivered as a token to the four non-body calls.** The
token exists so a cooperative tool body can return partial output within the
grace window. There is no partial answer worth having from listing tools,
creating an execution record, deciding an approval, or preparing a dispatch.
Racing at the runner level kills the task without the callee needing to know a
token exists.

**All four non-body races use no grace window and wait for unwinding.** That
includes `prepare` — it is a registry call, not the body, and takes the same
treatment as the other three. Grace is zero — there is nothing to salvage — and
the runner awaits the killed task's teardown before proceeding, so a registry
releasing a resource in a `finally` or `async with` finishes releasing it. (Only
the prepared callable keeps a grace window and a detached kill, because
thread-backed work cannot be interrupted; that reason applies to none of the
four.)

### 6.6 Deadlines

Deadlines bound the **prepared callable only**. `ToolSpec.timeout_in_ms`
(snapshotted at birth, no dispatch-time re-snapshot) beats
`RuntimeConfig.tool_execution_timeout_in_ms`. Expiry hard-cancels the call and
records `TIMED_OUT`, resultless; it never touches the shared token — one call's
deadline must not cancel its siblings — and it never populates
`cancel_signalled_at`.

`get_tools`, `create_execution`, `decide` and `prepare` have **no deadline**.
This is a deliberate trade: those calls are contractually local and
non-blocking, so bounding them buys little, and a per-phase deadline would add a
second timeout tier to reason about. Cancellability is what stops a user's
cancel from being ignored; a deadline would stop an *unattended* run from
hanging forever, and that is a separate decision we are not taking here.

The consequence must be documented plainly wherever `timeout_in_ms` is
documented, not only where `prepare()` is: **a tool configured with
`timeout_in_ms=5000` is not bounded end to end — the 5s bounds its body.**
Registry authors who do I/O in `prepare()` in spite of the rules in §9 are
responsible for their own timeouts.

### 6.7 Crash recovery

- **A crash after `prepare()` returned and before the callable completed**
  recovers as `INTERRUPTED` and is never re-dispatched — the body may have had
  side effects. Unchanged.
- **A crash during `prepare()` is fully recoverable.** No session write happens
  during `prepare()`, so the execution is still `PENDING` and the next drive
  prepares it again. This is why `prepare()` must be safe to call more than once
  (§9). Two consequences of writing nothing: `before_tool_execution` fires again
  on the next drive (§8), and a `raw_tool_call` that hook rewrote is gone —
  today that rewrite lands with the `RUNNING` transition and survives the crash.
  Both are correct for an attempt that produced no outcome; neither is silent,
  because the hook simply runs again over the original call.
- **An execution born before a crash resumes against its original `ToolSpec`**,
  even if the tool's schema or timeout changed meanwhile. The runner reads
  `timeout_in_ms` only at dispatch and deliberately does not re-snapshot. Do not
  introduce a dispatch-time re-snapshot.

### 6.8 Context accounting and tool output

`calculate_context(session, entry)` runs on every new entry before
`before_entry_written`, and again on a `ToolExecution`'s terminal transition
before `after_tool_execution`. Middleware still has the final say — nothing is
recalculated, validated or repaired after it.

`process_tool_output(session, execution, result)` transforms a returned
`ExecutionResult` before the terminal `ToolExecution` is constructed and before
any middleware runs. The durable session, the `ToolExecuted` event and the wire
all see the processed output. The `execution` it receives is **in transition**:
`status` is still `RUNNING` and `result` is not yet attached. It is there to be
read for identity — `tool_spec`, `raw_tool_call.name`,
`raw_tool_call.arguments` — not inspected for outcome.

**`calculate_context` runs on every new entry.** Scanning `session.entries`
inside it makes a turn quadratic. Cross-entry work belongs in `prune_entry` and
`process_tool_output`, which run rarely. On an append it also runs *inside* the
ledger's build callback, so the entry already has its `id` but is not yet a
member of `session.entries` — an implementation that looks itself up there
raises `KeyError` on every append.

**Model-aware counts need a refresh path.** `Entry.context_tokens` is stored on
the entry. `llm_config` is session-wide, so only one model is ever active, but
switching it mid-session leaves every stored count computed under the old model
— and `contrib/compaction`'s context gauge, which sums stored `context_tokens`
over the active path, then reports a number with no single basis.

The runner gains one public method, `recalculate_context_tokens()`, that
re-derives `context_tokens` for every entry in `session.entries` — not just the
active path, because the count is intrinsic to an entry and shared by every
conversation that references it — threading each through `before_entry_written`.
It sets no other field itself.

**Nothing calls it.** No constructor keyword, no CLI flag, no automatic
invocation on a model switch; the TUI's `/model` is untouched. The shipped
`ContextManager` is a character estimate that no model choice affects, so
nothing in this repo would ever trigger it, and the project does not add knobs
before a second real case exists. It is there for the application that swaps in
a real tokenizer, and that application calls it.

### 6.9 Tool spec storage

**Writing.** When an execution is written to the session, the framework stores
its spec in `session.tool_specs` under a content-derived id and records that id
on the execution. Writing the same spec again is a no-op: identical content
produces an identical id.

This happens on the framework's own write paths, on every write. There are
**three** of them, all on `SessionLedger`: `append` (an execution's birth),
`put_entry` (every later update to one), and `transition_conversation` (the
compaction install — `CompactionPlan.nodes` admits any `AnyEntry`, and that
method already carries an `isinstance(entry, ToolExecution)` branch that indexes
plan-created executions, so a spec can arrive there and nowhere else). `prune`
is not one of them: it only ever writes a `PrunedEntry`.

At the third door the helper must run over **`updates`, `created` and `closing`
alike**, not just `created`. `transition_conversation` takes
`updates: list[AnyEntry]` and is a public door; a `ToolExecution` can arrive on
either list even though the runner's own compaction commit currently sends only
the `CompactionEntry` as an update. Covering only the list the shipped caller
happens to use is how the missing door got missed the first time.

All three must store the spec. One that does not is silent: the execution
carries its spec in memory for the rest of the process, and then loses it the
first time the session is saved, because the session serializer strips inline
specs and has no id to write in their place. The load-time guards in §8 cannot
catch that — the stripping already destroyed what they look for. Registries are
on none of these paths.

**Reading.** `execution.tool_spec` continues to return the full `ToolSpec`
object, for every consumer, in memory and after a reload. Application code that
reads it today keeps working unchanged. This matters specifically because the
three tool lifecycle events (`ToolCallReceived`, `ToolExecutionStarted`,
`ToolExecuted`) carry a deep snapshot of the `ToolExecution` and are consumed by
code — terminal UIs, custom renderers — that has no session object in hand.

Nothing in the restored spec references a live `Tool` class, an `Args` model, or
anything importable: it is plain data, so a session whose tools were deleted
from the codebase years ago still renders its `name`, `description`,
`input_schema` and `tool_kind`. That is the whole point of P1. After a load,
every execution referencing the same spec holds the *same* `ToolSpec` instance,
shared with the one in `session.tool_specs` — a value object held by reference,
not a per-execution copy.

**Serializing.** Serializing a *session* omits the per-execution spec copies and
writes the shared `tool_specs` dict once. Serializing a *`ToolExecution` on its
own* — what an event consumer forwarding to a web UI or a log sink does —
includes the full spec inline, so it stays self-describing.

**Loading.** Loading a session restores `execution.tool_spec` on every execution
from the shared dict. A session cannot exist in a partially-restored state, and
an unresolvable reference fails loudly (§8).

**The id function.** The id must be identical across processes, across machines,
and across independent implementations of `luca` in other languages. It is
therefore **not** a runner hook and **not** overridable — unlike `generate_id()`
and `now_ms()`, which exist for test determinism. Here determinism is a
data-integrity requirement; a subclass that changes it corrupts the store.

1. Render the spec to its JSON representation (enums as their string values,
   `None`-valued fields included).
2. Serialize with **recursively sorted keys**, no whitespace, non-ASCII emitted
   literally, encoded UTF-8.
3. SHA-256 the bytes.
4. Hex-encode. **Full 64 characters, no truncation.**

Not MD5: `hashlib.md5()` raises on FIPS-enabled builds, which is a real
deployment failure for a library. Not truncated: the saving is roughly 1% of
what normalization already buys, and every free parameter in the rule is a place
a second implementation can silently diverge and stop producing portable
sessions. The four numbered points exist to close those parameters — step 2 in
particular, because key sorting, whitespace and ASCII-escaping are exactly where
two JSON encoders disagree by default.

One accepted limitation, in three forms. JSON arrays are order-sensitive, so a
schema whose `required: [...]` list gets reordered mints a new id. A Pydantic
upgrade that changes `model_json_schema()` output does the same. And a field
added to `ToolSpec` later makes an old session's stored specs hash differently
from the keys they are already filed under. All three cost the same thing and
heal the same way: the next write of that execution mints one redundant row and
the old row stays, still resolvable by the executions that point at it. A
content hash's only failure mode is a redundant row, never a wrong lookup.

### 6.10 Errors and the failure phase

The durable error's `details["phase"]` becomes a fact, known from which call
raised, rather than an inference from `started_at`. It is populated on **every**
registry- or tool-owned raise, including the ones that previously carried only
structured `errors`:

```
details = {"phase": "create_execution" | "prepare" | "execution", ...structured}
```

where `...structured` is `{"errors": [...]}` for `InvalidToolArguments` and
pydantic `ValidationError`, and nothing otherwise.

Those three values are the runner's vocabulary for raises it observed. A
registry authoring a terminal-at-birth error owns its own `details` and may use
its own phase vocabulary — `SimpleToolRegistry` writes
`{"phase": "approval_context"}` when a tool's `get_approval_context` raises, and
that stays.

---

## 7. Data model changes

### `ToolSpec`

| Field | Change |
|---|---|
| `description: str \| None` | becomes `description: str`, **required** |
| `input_schema: dict` | **new**, required — the tool's arguments as JSON Schema |

`description` must become mandatory because the client's wire tool type requires
a non-null string; today the adapter reads it off a required `Tool` class
variable, so nothing regresses.

`input_schema` is a plain `dict` holding JSON Schema. Not a Pydantic model
class, not a `TypeAdapter` — the whole point is that it survives a round trip
through JSON. It is required and never `None`, including for a tool that takes
no arguments; that case is the empty object schema,
`{"type": "object", "properties": {}}`. An absent schema and an empty schema
mean different things to a model provider, and only one of them is ever correct.

`ToolSpec` gains one method, `spec_id()`, returning its content-derived id
(§6.9). A **method, not a field** — a stored `spec_id` field would be
self-referential, and given `extra="forbid"` a stray serialized `spec_id` key
would make loading fail.

`metadata` **stays**, unchanged: a free-form dict the core never interprets, for
registries that carry their own vocabulary on a spec. Nothing in this repo
writes it and `Tool` has no `metadata` ClassVar — that is not a reason to remove
an extension point from a framework. It picks up one hard rule under
normalization: what goes in it must be a pure function of the tool definition
(§8).

`ToolSpec` stays a mutable Pydantic model like every other model here. The
framework never mutates a spec in place, and what application code does to one
is application code's business — the standing trust model in `middleware.py`.

### `AgentSession`

New field, alongside the existing `entries` / `tool_executions` / `usages`
stores:

```
tool_specs: dict[str, ToolSpec]   # spec_id → spec; append-only
```

### `ToolExecution`

| Field | Change |
|---|---|
| `tool_spec: ToolSpec \| None` | kept, unchanged shape; now a restorable cache rather than the stored copy |
| `tool_spec_id: str \| None` | **new** — the durable reference; `None` when the tool never resolved |

`tool_spec_id` is authoritative. `tool_spec` is derived from it on load and must
never be the source of truth for anything durable. This is a deliberate
exception to the project's rule that nothing on the session is transient, and it
needs to be as loudly documented as the existing exception
(`AgentSession.session_runtime_status`, a recomputed property).

### Deleted

`ToolContext` — the model and its export.

### On disk, before and after

Before — two calls to the same tool, two full copies:

```jsonc
{
  "entries": {
    "e1": { "type": "tool_execution", "tool_spec": { "name": "bash", "description": "…4 KB…" } },
    "e2": { "type": "tool_execution", "tool_spec": { "name": "bash", "description": "…4 KB…" } }
  }
}
```

After — one copy, two references:

```jsonc
{
  "tool_specs": {
    "9f2c…64 hex chars…": {
      "name": "bash",
      "description": "…4 KB…",
      "input_schema": { "type": "object", "properties": { "…": "…" } }
    }
  },
  "entries": {
    "e1": { "type": "tool_execution", "tool_spec_id": "9f2c…" },
    "e2": { "type": "tool_execution", "tool_spec_id": "9f2c…" }
  }
}
```

`tool_spec` does not appear in the serialized entries at all, and both
executions are restored to a live `tool_spec` object on load.

---

## 8. Invariants

**Every tool call produces exactly one tool output.** Pre-existing, and now
load-bearing in three new places: a cancelled birth, a cancelled decide, and a
`prepare()` failure each still produce their one execution and one output.

**`before_tool_execution` fires exactly once per DISPATCH ATTEMPT, and never
twice for one outcome.** Every path that terminalizes an execution after
dispatch preparation began must use the shared outcome tail, never the
undispatched pipeline that re-runs the hook.

It is deliberately NOT once per call for all time, and that is a change. Today
the hook fires only after `RUNNING` is persisted, so a crash recovers as
`INTERRUPTED` without re-firing it. After this change a crash during `prepare()`
leaves the execution `PENDING` (§6.7) and the next drive fires the hook again
for the same call — correct, because the first attempt produced no outcome and
wrote nothing, but worth knowing before a hook is written that assumes
once-forever.

**`ToolSpec` is a pure function of the tool definition, never of the call.**
This was always true informally; normalization makes it load-bearing. A registry
that puts anything volatile in `ToolSpec.metadata` — a timestamp, a request id —
mints a new row on every single call and silently defeats the normalization,
with no error and no warning. Document it on `ToolSpec` and pin it with a test
that calls one tool twice and asserts `len(session.tool_specs) == 1`.

**At every write, `tool_spec_id` equals `spec_id(tool_spec)`; when `tool_spec`
is `None`, `tool_spec_id` is `None`.** "Every write" means all three doors in
§6.9 — `append`, `put_entry`, `transition_conversation` — not just the two an
ordinary tool call travels through. Do not short-circuit when the id is already
set: an execution's spec can be replaced between writes, and a skipped recompute
leaves a stale id pointing at the previous version. Hashing a few KB costs on
the order of ten microseconds; correctness wins.

**A dangling `tool_spec_id` raises at load.** If an execution references an id
absent from `tool_specs`, refuse to construct the session. The tolerant
alternative is silent and lands on two untraceable behavior changes: the
dispatch deadline falls back to `RuntimeConfig.tool_execution_timeout_in_ms`,
which defaults to `Inf`, so a tool that declared a 5s bound quietly runs
unbounded; and `resource_permissions` loses `tool_kind`, so a permission rule
written for a given kind quietly stops matching and the call takes a different
approval path.

In normal operation this cannot happen — the write door writes both sides
together. It happens when a session is assembled by something other than that
door: entries copied between sessions without their specs (the realistic case,
since this is a library and callers will slice and merge sessions), a
hand-edited or crash-truncated file, or a test literal that sets the id and
forgets the store.

**The mirror case also raises at load: a serialized execution carrying a
`tool_spec` but no `tool_spec_id`.** That combination can only come from a
pre-normalization file. Without the guard such a file loads and runs fine, then
loses every spec the first time it is saved, because the session serializer
strips inline specs and there is no id to replace them with — silent data loss
with no failure anywhere near the cause. This guard is what makes "no migration"
a safe decision rather than a lossy one.

It fires only on session construction. Registries build draft executions in
memory with `tool_spec` set and no id yet, which is correct and untouched — the
write door assigns the id. A session *literal* in a test or application needs
`tool_spec_id` plus its `tool_specs` row; `tool_spec` itself is optional there,
because the load validator restores it from the id. It is `tool_spec` WITHOUT an
id that raises.

**The session is read-only to registries, tools and context managers.**

### Verified safe — no work required

| Area | Why |
|---|---|
| **Fork** | `fork_session` is `model_copy(deep=True)`; `tool_specs` rides along, and `model_copy` does not re-run validators. |
| **Compaction** | `ConversationSnapshot` carries only entry ids, and the transition installs a new `Conversation` over the same session. |
| **Pruning** | `SessionLedger.prune()` never deletes the original entry, so a spec can never be orphaned. |
| **Context accounting** | `_model_facing_text` ignores `tool_spec` entirely; token counts and compaction thresholds are unaffected by normalization. |
| **Event snapshots** | Deep copies do not run serializers, so the specs carried on tool lifecycle events survive regardless. |

---

## 9. Rules for registry authors

These are contract requirements, not suggestions. Every one of them fails
silently when violated. They belong in the `ToolRegistry` module docstring.

**1. `prepare()` must be safe to call more than once for the same execution.**
If the process dies during preparation, the execution is still `PENDING`, and
the next drive calls `prepare()` again. The constraint is on *call-scoped* side
effects: do not consume a token, advance a cursor, or write a record keyed to
this call. Registry-lifecycle work is fine and is idempotent by nature — lazy
connect-on-first-use behind an already-connected check, warming a catalog,
refreshing a shared credential.

**2. `prepare()` must not return while holding anything that needs releasing.**
Using `async with` *inside* `prepare()` is fine: a cancellation raises inside it
and its `finally` runs. What is forbidden is handing back a callable while a
lock, lease, sandbox, connection or concurrency slot is held, because the runner
may never invoke that callable and nothing runs a cleanup path for it. Acquire
inside the returned callable, under `try/finally`.

**3. `prepare()` must not block.** No deadline applies to it. It is raced
against the run's cancellation token, so a stuck preparation is not unkillable,
but that race is a safety net, not a license: the only thing that ends a hung
`prepare()` is an explicit `cancel()` from the application. Resolve from local
state — a static dict, a cached catalog. A registry fronting a remote tool
server keeps a cached tool list refreshed out of band and does all of its
network work inside the callable.

**4. The returned callable is where wrapping belongs.** Retries, rate limiting,
metrics, tracing, exception translation and result post-processing all go in a
closure the registry owns. Returning a tool's bound method directly is valid but
gives all of that up.

**5. Capture what the callable needs during `prepare()`.** The callable receives
only `cancellation_token`. See §5.5 for what is live and what is a snapshot.

**6. Any of the four methods may be cancelled at any `await`.** Anything
acquired inside them must be released in a `try/finally` or `async with` —
cancellation raises `asyncio.CancelledError` at the current await point (not
luca's own `CancelledError`, which only a cooperative tool body ever sees) and
normal unwinding applies, but nothing is released automatically.

**7. Never swallow `asyncio.CancelledError`.** The runner waits for a cancelled
registry call to finish unwinding before it proceeds, precisely so that cleanup
completes. A method that catches the cancellation and keeps working makes that
wait unbounded and re-creates the hang this change exists to remove.

**8. Blocking synchronous work must run in `asyncio.to_thread`.** A cancellation
cannot interrupt a blocking syscall, so synchronous I/O inside these methods
blocks the event loop regardless of cancellation handling. This is not
hypothetical here: the shell tools' permission preflight calls `path.is_dir()`,
a synchronous filesystem stat — on a hung network mount, no amount of racing
helps.

**9. Nothing runs on a hard process kill.** A registry holding something that
must not leak — a distributed lock, a remote lease — needs a TTL or a startup
reconciliation pass, not a `finally`.

**10. `decide` must be an idempotent query of your own state.** Record answers
out of band and return them when asked. Its exceptions propagate and abort the
run; the executions stay unresolved and the next `run()` asks again.

**11. The core never validates arguments.** It knows each tool's input schema
but will never check a call against it, because a registry may delegate to a
remote server that validates on its own side. Validation is yours.

**12. Do not put volatile data in `ToolSpec.metadata`.** See §8.

### Edge behaviors registries choose

**Tool no longer exists** (stale session, upgraded host application). The
registry chooses the outcome: raise `ToolNotFound` for a hard `NOT_FOUND`, or
return a callable that produces `ExecutionResult(content=[…], is_error=True)`
saying the tool is no longer available — a normal completed result the model can
read and work around. Same choice when stored arguments no longer validate
against a changed schema.

**Nothing may crash the run.** Every failure path terminalizes one execution and
leaves its siblings alone.

---

## 10. Rules for tool authors

Unchanged in substance; restated because the signature moved.

- `Tool` now lives in contrib. `execute` / `_execute` / `get_approval_context`
  receive the live `AgentSession` instead of a `ToolContext`. Read
  `session.id` and `session.session_config.llm_config` where you read
  `context.session_id` and `context.model`. Do not write to the session.
- The cancellation and deadline contract on the tool body is unchanged: a
  cancelled run hard-cancels the task once the grace period expires
  (`INTERRUPTED`, resultless); a cooperative tool may watch the token and return
  early within the grace window, and whatever it returns is its real result.
  Deadline expiry hard-cancels the same way (`TIMED_OUT`).
- Tools that spawn processes must kill their process group on
  `asyncio.CancelledError`. Blocking sync work belongs in `asyncio.to_thread`.
- **Per-run application state is not the framework's concern.** Registries and
  tools are application code; they can hold references to application services
  or read a `contextvars.ContextVar`. Nothing about that needs threading through
  these signatures — which is the other half of why `ToolContext` is being
  deleted rather than extended.

---

## 11. Behavior changes callers will see

- A tool **body** that raises `ToolNotFound` (looking up a sub-resource, say)
  now records `FAILED` instead of `NOT_FOUND`. Same for a body raising a
  pydantic `ValidationError`: `FAILED`, not `INVALID`. This is the point of the
  change — those statuses now mean what they say.
- An execution that fails to resolve after approval records `started_at=None` /
  `dispatched=False`, where it previously recorded a timestamp and a duration.
- `ToolExecutionStarted` is no longer emitted for a call that fails to resolve
  at dispatch.
- `details["phase"]` is populated on every registry- or tool-owned raise,
  including ones that previously carried only `errors`.
- `to_tool_execution_error` takes an explicit phase. Applications that override
  it must update the signature.
- `AgentSessionRunner` gains a public `recalculate_context_tokens()` method.
  Nothing calls it, so behavior is unchanged unless an application does (§6.8).
- `before_tool_execution` is exactly-once per dispatch attempt rather than per
  call: a crash during `prepare()` re-fires it on the next drive (§8).
- Session files written before this change do not load. Regenerate them.
- `AgentSession` construction now raises on an execution with a `tool_spec` and
  no `tool_spec_id`, or with a `tool_spec_id` absent from `tool_specs`. A
  session literal needs `tool_spec_id` plus its `tool_specs` row; `tool_spec` is
  optional there, since the load validator restores it from the id.
- Every `ToolRegistry`, `ContextManager` and `Tool` implementation must be
  updated: signatures changed on all of them, and `execute` was replaced by
  `prepare`. There is no compatibility shim; this is pre-v0.

---

## 12. Judgement calls, and why they went this way

Twelve decisions where a competent implementer could reasonably have chosen
otherwise. Each is binding; each is here so it can be argued with specifically
rather than discovered halfway through the work.

1. **All four methods take the session first, and none takes anything else
   ambient.** `prepare(session, tool_execution)`, not
   `prepare(tool_execution, session)` or a signature carrying leftover context.
   One shape across the contract is worth more than per-method convenience: a
   reader who learns one method's argument order knows all four.

2. **`Tool` moves to contrib *and* changes signature in the same step.** It
   would be tidier to move the module untouched and re-sign it later. But
   `ToolContext` is being deleted, and a `Tool` that still takes it cannot
   compile after the deletion. Move and re-sign together.

3. **Cancellation during `prepare()` is terminalized by the dispatch path, not
   handed back to the loop-top wind-down.** The tempting implementation is to
   leave the execution `PENDING` and let the generic wind-down finish it, which
   is exactly right for a cancellation during `create_execution` or `decide`.
   It is wrong here: `before_tool_execution` has already fired for this call,
   and the wind-down path fires it again. One call, two hook invocations,
   silently. §6.5 makes the hook the boundary for precisely this reason.

4. **No `except CancelledError` clause is needed anywhere in the runner.** This
   is worth stating because the opposite looks necessary: `create_execution` is
   wrapped in a broad `except Exception` that records a `FAILED` draft, so it
   seems a cancellation could be recorded as a tool failure. It cannot, twice
   over. `asyncio.CancelledError` derives from `BaseException`, so `except
   Exception` never catches it; and the race helper absorbs the kill it issued
   and reports the outcome as a boolean rather than re-raising, so nothing
   reaches the handler at all. (Luca's own `CancelledError` *is* an `Exception`
   — but registries are never handed a token to check, and one that raises it
   regardless is treated like any other exception.) Do not add defensive
   clauses for a path that does not exist; they will look load-bearing to the
   next reader.

5. **All four non-body races use a zero grace window and await the unwinding.**
   Grace exists so in-flight work can finish and hand back a partial result.
   Listing tools, minting an execution record, deciding an approval and
   preparing a dispatch have no partial result worth having — the same reason
   they are not handed the token (§6.5) — so waiting is pure latency. `prepare`
   is grouped with the other three and not with the body: a prepared callable
   the runner never invokes is worth nothing, so there is no case for letting
   preparation finish. Awaiting the unwinding is not optional though: it is what
   lets a registry's `finally` complete before the run moves on, and it is what
   makes rule 7 in §9 a hard requirement rather than advice.

6. **`build_tool_list` is raced as a whole, not `get_tools` inside it.** Racing
   the inner call would force the token into `build_tool_list`'s signature — a
   public, overridable method — and would leave a subclass that overrides it
   uncovered. Racing the outer call keeps the signature clean and covers
   overrides for free.

7. **The context refresh walks every entry, runs `before_entry_written` on
   each, and ships as a bare method with no caller.** Every entry rather than
   the active path, because `context_tokens` is intrinsic to an entry and shared
   by every conversation that references it; refreshing only the active path
   would leave the archived ones on the old basis. Through the middleware door,
   because the framework's standing rule is that middleware has the final say on
   context and nothing is recomputed behind it. No constructor keyword and no
   CLI flag: the shipped `ContextManager` is a character estimate that no model
   choice affects, so nothing in this repo would ever set them, and the project
   does not add knobs before a second real case exists. Both were specified in
   an earlier round and cut for that reason — add them alongside the first
   tokenizer-backed `ContextManager`, not before. Automatic invalidation on a
   model switch was considered and rejected too: it puts an unbounded rewrite
   behind an innocuous-looking assignment.

8. **`PermissionPolicy.decide` gains the session too.** This is contrib, one
   layer below the contract being changed, and it would be defensible to leave
   it alone. But `SimpleToolRegistry.decide` now receives the session and would
   drop it on the floor when delegating — recreating one layer down the exact
   information asymmetry this refactor exists to remove. Cheap while the package
   is already open; awkward to retrofit later.

9. **The id function pins more than "sorted keys, no whitespace, UTF-8".** Enum
   rendering, `None`-field inclusion and ASCII escaping are left unspecified by
   that phrase, and they are precisely where two JSON encoders differ by
   default. Every unpinned parameter is a place a second implementation of
   `luca` silently produces different ids for the same spec and stops being able
   to read the first one's sessions. §6.9 closes all of them.

10. **The cancellation race around `decide` wraps the middleware pair with
    it.** Racing `registry.decide` alone would leave `before_permission_check`
    firing on a run that is already cancelling; racing the whole step means an
    already-tripped token fires no hook at all. It deliberately does NOT make
    the hook's result durable when the token trips mid-decide: the execution
    must stay `PENDING` for the wind-down, so there is nowhere to put it, and
    persisting it before the decision would add a ledger write per undecided
    execution per drive and break the ordering the drive loop relies on ("all
    decision writes land before any denial event is yielded").

11. **`transition_conversation` stores specs too.** It is the third write door
    (§6.9) and the easiest to miss, because no ordinary tool call travels
    through it — only a compaction plan that carries or creates a
    `ToolExecution` does. Covering it is one call to the same helper the other
    two doors use; not covering it means a spec that survives in memory and
    vanishes on the next save, in framework code, with the load-time guards
    unable to see it. Rejecting plan-created executions instead would cost the
    same and take away a capability the ledger already supports on purpose.

12. **The `tool()` / `tool_class()` factories get fixed here.** They are broken
    today: the factory builds `_execute(self, args, context)` while
    `Tool.execute` calls it with `cancellation_token=`, so both public helpers
    raise `TypeError` on every call, and `tests/agent/test_tools.py` skips the
    entire factory block rather than the API being removed. This refactor
    rewrites that exact signature and moves that exact module, so the fix is
    nearly free now and awkward later. Fix the factory to match `Tool.execute`'s
    call and un-skip the tests. Deleting the factories instead is a legitimate
    alternative — leaving a broken public API with its tests turned off is not.

---

## 13. Scope

### `luca/agent/core`

| File | Change |
|---|---|
| `models.py` | `ToolSpec`: required `description`, new `input_schema`, `spec_id()`. `ToolExecution`: new `tool_spec_id`. `AgentSession`: new `tool_specs`, load restoration, session-dump stripping. |
| `context.py` | delete `ToolContext`; keep `CancellationToken`. Module docstring rewritten. |
| `tool_registry.py` | the four new signatures; contract semantics and the twelve author rules in the docstring. |
| `context_manager.py` | three new signatures. |
| `tools.py` | **deleted** (moves to contrib). |
| `adapter.py` | `tool_to_luca_tool(tool)` → `tool_spec_to_luca_tool(spec)`. |
| `ledger.py` | store specs and stamp `tool_spec_id` at all three write doors — `append`, `put_entry`, `transition_conversation` (§6.9); a ledger method for the context refresh. |
| `runner.py` | async `build_tool_list`; the prepare step and its cancellation handling (including moving the body's `ensure_future` inside the failure handling, §5.2); the moved exception mapping; phase-aware `to_tool_execution_error`; the four races, with the `decide` race wrapping its middleware pair; `recalculate_context_tokens()`; drop the per-run `ToolContext`. |
| `middleware.py` | `before_tool_execution` docstring (resolution now happens in `prepare()`). |
| `core/__init__.py` | drop `ToolContext`, `Tool`, `tool`. |
| `luca/agent/__init__.py` | module docstring names `ToolContext` and the `Tool` base class. |

### `luca/agent/contrib`

| Package | Change |
|---|---|
| `tools.py` (new) | `Tool`, `tool`, `tool_class` moved from core; `get_tool_spec()` adds `input_schema`; session-taking signatures; factory fix (§12.12). |
| `simple_tool_registry` | `get_tools` → `list[ToolSpec]`; `execute` → `prepare`; session signatures; `PermissionPolicy.decide` gains the session. `ProxyToolRegistry`: cache-independent resolution for `decide` and `prepare`; ALLOW-on-cache-miss removed, ALLOW-on-unresolvable kept (§15.5). |
| `resource_permissions` | mixin and strategy signatures. |
| `shell`, `memory` | tool signatures and imports; `contrib/shell/AGENTS.md` documents `ToolContext`. |
| `tui/wiring` | tool signatures and imports. |
| `tui/cli`, `main.py` | **no change.** No `--refresh-context` flag, `/model` untouched (§6.8, §12.7). Listed so nobody adds one. |

### Docs

`docs/agent/01-quickstart.md`, `02-data-model.md` (the `tool_specs` store and
the `tool_spec_id` reference), `03-tools.md` (splits: the `ToolSpec` contract
stays in core docs, the `Tool` ergonomics move to a new
`docs/agent/contrib/tools/README.md`), `04-runner.md`, `05-permissions.md`,
`07-middleware.md` (the signature changes AND the exactly-once note from §6.5),
`08-runtime-config.md` (the deadline scope, §6.6), `09-plugins.md` (its
`get_tools` mentions), `11-context-and-usage.md` (the three signatures + the
refresh method), `README.md`, `docs/agent/contrib/README.md` (its package table
needs a `tools/` row), `docs/agent/contrib/simple_tool_registry/README.md`,
`docs/agent/contrib/resource_permissions/README.md` (it documents the
`get_approval_context(args, context)` convention), and
`docs/agent/contrib/plugins/README.md` (one `get_tools` mention). Then
`AGENTS.agent.md` (the registry section, "Tool identity", "Timeouts and step
limits", principle 12). Follow `docs/llm.txt`.

### Tests

Fourteen files touch `ToolSpec` or `ToolExecution` literals — 70 `ToolSpec(...)`
(25 of them with no `description`), 101 `ToolExecution(...)`, 202
`AgentSession(...)`. `description` becoming required, `input_schema` becoming
required, and session literals needing `tool_specs` + `tool_spec_id` on both
sides make this the largest single block of work in the change, and most of it
is mechanical.

**Add a small factory to `tests/agent/scenarios.py`** — a `spec(name, **over)`
returning a complete `ToolSpec`, and a session builder that fills `tool_specs`
from the executions it is given — rather than hand-editing 70 literals into
correctness. Module-level factories are not "logic in the test body"; they keep
the precondition declarative, which is the point of the house rule.

Three files carry work beyond that:

- `tests/agent/scenarios.py` — `FakeToolRegistry` and the tool doubles are built
  on `Tool`. Core tests must not import contrib, so rewrite them to return
  `ToolSpec` literals from `get_tools` and prepare via closures. This is the
  right kind of churn: it *proves* the decoupling rather than asserting it. The
  mid-state session literals there need `tool_specs` populated and
  `tool_spec_id` set.
- `tests/agent/test_runner_tool_output.py` — `InlineToolRegistry`, same
  treatment.
- `tests/agent/test_tools.py` moves to `tests/agent/contrib/` essentially
  verbatim, with the factory block un-skipped.
- `tests/agent/test_adapter.py` — one test for the renamed function.
- `tests/agent/contrib/test_simple_tool_registry.py` needs *new* coverage, not
  just signature updates: `prepare` returns a callable without running the body,
  raises `ToolNotFound` / `InvalidToolArguments` otherwise, and
  `ProxyToolRegistry` routes both `prepare` and `decide` to the owning child on
  a never-warmed route.

Everything else is import, signature and literal churn:
`test_runner.py`, `test_runner_lifecycle.py`, `test_runner_cancellation.py`,
`test_runner_approvals.py`, `test_runner_failures.py`, `test_ledger.py`,
`test_projection.py`, `test_models.py`, `test_utils.py`,
`contrib/test_resource_permissions.py`, `contrib/test_memory.py`,
`contrib/test_plugins.py`, `contrib/shell/conftest.py`,
`contrib/shell/test_plugin.py`. One special case:
`contrib/tui/test_wiring.py` calls `runner.build_tool_list()` synchronously and
has to await it.

Per the project's test style, assert on the whole `ToolExecution` and the whole
event list, not on individual fields.

---

## 14. Acceptance criteria

**Decoupling and normalization**

- No `Tool` import remains anywhere in `luca/agent/core`; no `ToolContext`
  remains in the codebase.
- A registry that returns hand-written `ToolSpec` literals with raw JSON Schema,
  and never defines a `Tool` subclass, drives a full tool call end to end.
- The same tool called twice produces exactly one row in `tool_specs`.
- A session round-trips through dump/load with `tool_spec` restored on every
  execution.
- A standalone `ToolExecution` dump still contains its spec inline.
- Middleware rewriting a spec via `before_entry_written` updates
  `tool_spec_id`.
- A dangling `tool_spec_id` raises at load; an execution with `tool_spec` and no
  `tool_spec_id` raises at load.
- Two specs differing only in `description` get distinct ids and both persist,
  and older executions still resolve to the older spec.
- A tool taking no arguments round-trips with an empty object schema.
- Key insertion order in `input_schema` does not affect the id.
- A compaction plan that creates a `ToolExecution` carrying a `tool_spec`
  round-trips through dump/load with the spec intact and one row in
  `tool_specs` — the `transition_conversation` door (§6.9). Same for one passed
  on that door's `updates` list rather than `created`.

**Contracts**

- All four `ToolRegistry` methods are async and receive `AgentSession`; all
  three `ContextManager` methods receive `AgentSession`.
- A `ContextManager` produces different token counts for the same entry under
  two different `llm_config` models.
- A `ContextManager` truncates one tool's output and leaves another's untouched,
  selecting on `tool_spec.name`.
- Changing the session's model and calling the refresh method updates
  `context_tokens` on every existing entry — including entries on an archived
  conversation and entries on no path at all.
- Constructing an `AgentSessionRunner` changes no `context_tokens`, even with a
  `ContextManager` that would produce different counts — nothing calls the
  refresh but the application.
- `calculate_context` receives an entry that is not yet in `session.entries`
  when it runs on an append.

**Prepare**

- An execution whose `prepare()` raises has `started_at is None` and
  `dispatched is False`, for every exception type, with
  `details["phase"] == "prepare"`.
- An execution that reaches the callable has `started_at` set, whatever its
  terminal status.
- A resolution failure at `create_execution` time and the same failure at
  `prepare()` time produce the same `status`, `error.error_type`, `result`,
  `started_at` and `dispatched`. (Approval fields legitimately differ: a
  terminal-at-birth execution has `approval_status=None` and no decisions; one
  that failed at `prepare()` was allowed first.)
- A tool body raising `ToolNotFound` records `FAILED` with
  `details["phase"] == "execution"`, not `NOT_FOUND`.
- A registry returning `None` or a non-callable records that one execution
  `FAILED` with `started_at is None`, and leaves the run and its siblings
  intact.
- A registry returning a plain `def` that returns a string records that one
  execution `FAILED` with `started_at` set and `details["phase"] ==
  "execution"`; the `TypeError` does not escape and the run survives (§5.2).
- The prepared callable is not invoked as a side effect of `prepare()` — the
  tool body must not have run.
- `before_tool_execution` fires exactly once for an execution whose `prepare()`
  raised, and exactly once for one cancelled during `prepare()`.
- A `prepare()` taking measurably longer than a small `timeout_in_ms` (50 ms
  against a 10 ms setting) still completes and the call succeeds; the same 10 ms
  against a slow *callable* records `TIMED_OUT`.
- A crash after `prepare()` returns and before the callable completes recovers
  as `INTERRUPTED` and is never re-dispatched.
- A crash during `prepare()` leaves the execution `PENDING`, and the next drive
  prepares it again and completes — firing `before_tool_execution` a second
  time for that call (§8).
- A composing registry resolves and dispatches a pending call on a freshly
  loaded session with no prior tool listing, and that call is gated by the
  owning child's permission policy.
- A composing registry asked to decide a name no child owns returns ALLOW, and
  the call records `NOT_FOUND` rather than `REJECTED` (§15.5).
- A toolless runner terminalizes a loaded ready execution as `NOT_FOUND` without
  crashing the run.

**Cancellation**

- `cancel()` during a hung `get_tools`, `create_execution`, `decide` or
  `prepare` unblocks the run and closes the turn with the requested outcome.
- A cancelled `create_execution` records `CANCELLED`, never `FAILED`.
- A cancellation that lands while `prepare()` is in flight *and* one where
  `prepare()` returns successfully inside the window both record `CANCELLED`
  with `started_at is None`, and the tool body did not run.
- A response containing N tool calls yields N tool executions even when
  cancellation lands mid-batch.
- In a two-call batch cancelled after the first call's dispatch began, both
  executions end `CANCELLED` with identical durable shape, and
  `before_tool_execution` fired exactly once per execution — for the first from
  the dispatch path, for the second from the wind-down.
- A registry that acquires a resource in `async with` inside `create_execution`
  has it released before the run proceeds.
- A cancellation already pending when the decide step begins fires no
  `before_permission_check` at all (§6.5).
- `get_tools` raising leaves the turn open and the next `run()` calls it again.
- The `build_tool_list` middleware hook still receives wire tool objects, not
  `ToolSpec`s.

---

## 15. Implementation notes

Non-binding. Python- and repo-specific; everything above is language-neutral.

### 15.1 Restoring and stripping specs

Restore with a `@model_validator(mode="after")` on `AgentSession`: walk
`entries`, and for each `ToolExecution` with a `tool_spec_id`, set `tool_spec`
from `tool_specs`, raising on a miss and on the mirror case. Put it on the
model, not on `AgentSessionRunner.__init__` — sessions are loaded without ever
building a runner (`pretty_print(session)`, the TUI's session list, a direct
`model_validate_json`), and those must not see a half-restored session. On the
model, restoration is part of what constructing an `AgentSession` *is*.

Strip with a `@model_serializer(mode="wrap")` on `AgentSession` that removes
`tool_spec` from each serialized entry. Do **not** use `Field(exclude=True)` on
`ToolExecution.tool_spec`: one line instead of fifteen, but it strips the field
from *every* serialization including a standalone `ToolExecution`, which breaks
the event consumers in §6.9. Normalization is a session-storage concern, so it
belongs in the session's serializer.

The codebase already uses both validator forms (`models.py`,
`contrib/tui/config.py`).

### 15.2 Storing the specs

`SessionLedger` holds `self.session` and owns every path that puts an entry into
it, so this goes there: one private helper, called from three places.

- `append()` — an execution's birth.
- `put_entry()` — every later update to one.
- `transition_conversation()` — the compaction install. Easy to miss, because
  no ordinary tool call travels through it; a plan-created `ToolExecution`
  does, and the method already has an `isinstance(entry, ToolExecution)` branch
  for indexing one. Run the helper over `updates`, `created` and `closing`, not
  just `created` (§6.9).

`prune()` is not a fourth: it only ever writes a `PrunedEntry`.

Registries stay session-blind about storage and keep returning drafts with
`tool_spec=` populated. Apply the recompute-every-time rule from §8.

### 15.3 The runner's tool path

- `build_tool_list` becomes `async`, awaits `get_tools(self.session)`, converts
  via `adapter.tool_spec_to_luca_tool`, then threads the wire list through the
  sync middleware hook. In `_drive`, wrap the whole `build_tool_list()` call in
  the cancellation race; a lost race is `continue` — the loop top winds down,
  exactly like the aborted-LLM-call path already does.
- Race `create_execution` **per call**, inside the per-call birth helper, not
  around the `asyncio.gather`. Racing the gather and killing it loses every
  draft and breaks one-output-per-call. A lost race synthesizes a `PENDING`
  draft; the loop-top wind-down does the rest.
- Race `decide` **per execution**, and race the whole `_decide_with_middleware`
  call rather than the `registry.decide` inside it, so an already-tripped token
  fires no hook at all (§6.5, §12.10). A lost race returns "no decision" and
  the caller records nothing for that execution — in particular
  `after_permission_decision` must not fire.
- `_dispatch_one` gains: the `prepare` call (raced with grace 0 and
  `detach=False`, exactly like the other three registry calls — §6.5) before
  the `RUNNING` persist, the post-prepare cancellation check, the `callable()`
  guard, and the invocation. A `prepare()` failure and a prepare-window
  cancellation both finalize through `_finalize_outcome`, **not**
  `_finalize_undispatched` — the latter re-runs `before_tool_execution` (§8).
- `_run_tool_body` currently calls `asyncio.ensure_future(...)` on the line
  BEFORE its `try`. Move it inside: otherwise a prepared callable that returns
  a plain value raises `TypeError` outside the failure handling and takes the
  run down with it. Inside, it lands in the same `except` as any other body
  failure and needs no special case (§5.2).
- `_execute_body` becomes the single `registry.prepare` call site; the toolless
  runner raises `ToolNotFound` there.
- `_run_tool_body` keeps the cancellation race and the `asyncio.timeout`
  deadline but now wraps the prepared callable. Its post-body exception mapping
  collapses to `FAILED`.
- `to_tool_execution_error(execution, exception, *, phase)` sets
  `details["phase"]` from the call site for every exception type, and keeps
  nesting structured `errors` where they exist.
- The runner already has a private `_prepare` (an entry-completion helper).
  Rename it — `_complete_uncommitted`, say — so the tool-lifecycle `prepare` is
  the only thing that word means in this file.
- `recalculate_context_tokens()` should not go through `put_entry`, which is
  documented as the mutation door for the two mutable entry types; a dedicated
  ledger method for a derived-field refresh is cleaner. Nothing in the runner
  calls it (§12.7).

### 15.4 `SimpleToolRegistry`

Today's `execute()` body becomes `prepare()`, minus the final invocation: look
up the tool by name, `Args.model_validate` the arguments, and return a closure
binding the validated arguments and the session, which calls
`tool.execute(args, session, cancellation_token=...)`. That also removes the
double-validation waste — `create_execution` and `prepare` each validate once,
for different reasons (birth status vs. dispatch), and neither result is thrown
away and re-derived inside the same phase.

`get_tools` returns `[t.get_tool_spec() for t in self.tools]`.

### 15.5 `ProxyToolRegistry`

`get_tools` only ever reads `.name` from what its children return, so the
`list[Tool]` → `list[ToolSpec]` change is a type change and nothing else,
including the duplicate-name detection.

The real work is resolution. Today the `{name → child}` route is a side effect
of `get_tools()` — which the contract documents as a dynamic query, not a
lifecycle hook — and `decide()` returns a blanket `ALLOW` on a cache miss. That
is only safe today because `execute()` then raises `ToolNotFound` so nothing
runs. Once `prepare()` can resolve without the cache, the backstop is gone and
the blanket ALLOW becomes a permission bypass on a cold resume. Route `decide()`
and `prepare()` through the same cache-independent resolution, in this change.

The straightforward implementation: on a cold route, await each child's
`get_tools(session)` once and memoize. That is I/O inside `prepare()`, which
rule 3 discourages — but it is bounded, cached after the first hit, and raced
against the token. If it proves unsatisfying, the honest fix is a resolution
method on the contract, which §4 puts out of scope.

**What "blanket-ALLOW removed" does and does not mean.** What goes is
ALLOW-on-*cache-miss*: today `decide()` allows a name it cannot route, and that
is only safe because `execute()` then raises `ToolNotFound` so nothing runs.
Once `prepare()` resolves independently of the cache, that backstop is gone and
the cache-miss ALLOW is a permission bypass. What STAYS is ALLOW on a name that
is genuinely unresolvable after cache-independent resolution — `prepare()` will
raise `ToolNotFound` for it and the call records `NOT_FOUND`, which is the
honest outcome. Returning DENY there instead would record `REJECTED` for a tool
that never existed, contradicting §6.4's table. Resolution must be the same
question in `decide` and `prepare`; only then is one answer safe.

**This also closes the cold-resume degradation, not just its permission half.**
Once `decide` and `prepare` both resolve without the cache, a call left pending
approval by a previous process resolves on a fresh one, is gated by its owning
child, and dispatches. The only path still needing a warm route is
`create_execution`, and it is unreachable in practice: a tool call only arrives
after an LLM call, which warmed the route on its way out.

### 15.6 Suggested build order

One commit (§1), so this is a build order, not a landing order. Each step should
still leave the suite green: a green checkpoint is the only cheap way to
localize a regression in a change this wide, and the checkpoints are free.

1. **`ToolSpec` + normalization.** Add `input_schema` and `spec_id()`, the
   `tool_specs` store, `tool_spec_id`, the three ledger doors (§15.2), the load
   validator and the dump serializer. `get_tools` still returns `Tool`. `input_schema` is
   required from the moment it exists, so `Tool.get_tool_spec()` must start
   stamping it in this step — it is still in core here — and every `ToolSpec`
   literal in the suite needs it. Skipping that leaves step 1 red.
2. **`get_tools` → `list[ToolSpec]`.** Adapter rename, move `core/tools.py` to
   contrib, drop the core exports, rewrite the core test doubles. This is the
   step that proves the decoupling.
3. **`AgentSession` for `ToolContext`.** Signature sweep across both contracts,
   `Tool`, and all of contrib; delete `ToolContext`; make `get_tools` async;
   add `recalculate_context_tokens()`; fix the tool factories.
4. **Cancellation races.** All four call sites, with the per-call/per-execution
   placement from §15.3.
5. **`execute` → `prepare`.** The dispatch reordering, the moved exception
   mapping, the phase field, the proxy's resolution.

Within each pair the order is fixed — 2 needs 1, and 4 needs 3 because racing
`get_tools` requires it to be async first — but the two pairs are independent of
each other and can land in either order. Step 5 needs both 3 and 4: the
session-taking signature and the race machinery must be in place before
`prepare` replaces `execute`.

Run `uv run py.test tests/` throughout. `pyproject.toml` sets
`filterwarnings = ["error"]` — any warning fails the build, which is exactly why
§5.2 requires a callable rather than a coroutine.

---

## 16. Adjacent, explicitly not in this change

- **A `ToolSpec`-level `build_tool_list` middleware hook.** The obvious use —
  filtering by `tool_kind` or `namespace` — is impossible today because the
  adapter has already dropped those fields by the time the hook runs. This
  change makes the fix nearly free (the pre-adapter list is now plain data), but
  it is a new hook with a new contract and belongs in its own ticket.
- **Deadlines on the non-body registry calls.** §6.6 explains the trade. If
  unattended runs turn out to hang on registry preflight, this is the follow-up.
- **A resolution method on `ToolRegistry`** (`owns(name)` / `resolve(name)`).
  §15.5 is a workaround for its absence, and the proxy's routing is the standing
  argument for adding it.
- **Populating `ToolSpec.metadata` from `Tool`.** The field stays and is
  documented (§7, §8); giving `Tool` a `metadata` ClassVar so the shipped base
  class can fill it is a separate, additive change.
- **`ToolKind` is a closed enum the core never reads.** It exists purely for
  app-space policy, yet applications cannot extend it — `OTHER` is the only
  escape and it erases the classification.
- **Doom-loop detection and batch scheduling** are the only tool-path behaviors
  with no override point, in a submodule where permissions, projection, context
  and compaction are all swappable strategies.
- **`Tool.execute` receives a plain args dict** while the registry receives the
  whole `ToolExecution`, so a tool cannot see its own approval decision,
  `extras`, spec or execution id. Contrib-level, and a real gap.
- **`spec_version` on `AgentSession`.** Once sessions are portable across
  implementations, they need a format version. Not yet.

### If dispatch is made concurrent later

This change is neutral toward parallel dispatch — it neither enables nor
requires it — but it constrains the shape of any future scheduler:

- **Chain per execution, not phase per batch.** Each execution needs its own
  prepare → write `RUNNING` → invoke chain, with the chains running
  concurrently. Not a batch that prepares everything, then writes `RUNNING` for
  everything, then runs everything: the batched form marks one call `RUNNING`
  while another is still preparing, and a crash there records a body that never
  started — reintroducing exactly the defect this change removes.
- **Never let one chain cancel its siblings.** A bare `asyncio.gather` cancels
  the rest on the first exception. Collect exceptions instead: every tool call
  must produce exactly one tool output regardless of what its siblings do.
- **A concurrency cap can sit between `prepare()` and the `RUNNING` write.**
  Because preparation is separate and contractually cheap, a future parallelism
  knob can meter only the bodies, so `started_at` marks when the body actually
  began and `duration_ms` excludes queue time. (A registry holding its own
  semaphore inside the callable per rule 4 reintroduces queue time there; the
  runner can only account for its own cap.)
