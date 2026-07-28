Guidance for the `luca.agent` layer. Read this file whenever you're working in `luca/agent/` or `tests/agent/`.

## What this layer is

`luca.agent` is the primary product: a full-featured, durable agent framework. Its central artifact is a single serializable `AgentSession` that captures a complete conversation history — messages, tool executions, reasoning, turn boundaries, compaction — and can be reloaded to resume exactly where it stopped.

## Goals

- One canonical, JSON-serializable `AgentSession` that round-trips through `model_dump_json` / `model_validate_json` losslessly.
- A flat, append-only entry store (`AgentSession.entries`) addressed by id, with `Conversation.nodes` (an ordered list of ids) as the traversal path. Forking is cheap and explicit.
- A resumable async agent loop exposed as `runner.run()` (lazy) and `runner.start()` (eager), both returning an `AgentRun` handle. The engine projects the active conversation to LLM messages, calls the model, records the assistant turn, executes tool calls, and loops — bracketed by `TurnStart` / `TurnFinish(outcome)`. One logical turn can span multiple runs.
- The whole tool lifecycle delegated to a `ToolRegistry` — the core has no built-in tool resolution or approval engine; permission policies are contrib (`contrib/simple_tool_registry`). The core's only tool type is `ToolSpec`, plain JSON-serializable data: it depends on no Python tool class, so a registry fronting a remote tool server (MCP, an HTTP tool service, another agent) is a first-class implementation, and `luca` stays an implementation-agnostic specification another language could implement against.
- Durable cancellation via `runner.cancel()` / `run.cancel()`, recorded as a `CancelRequested` entry and wound down at engine step boundaries.
- Compaction as a step inside a drive: a `CompactionPolicy` decides and summarizes, the runner archives the conversation and installs a new one over the path the policy chose — atomically, losing nothing.
- Configurable timeouts and step limits that ride on the persisted `RuntimeConfig`, not the constructor.
- A purely observational event stream (`luca.agent.core.events`) consumed by iterating the handle or via `on_event`.

When the code disagrees with a doc, the code wins — fix the doc.

## File layout

```
luca/agent/
├── __init__.py          # docstring only — NO imports; all public surface lives in core/
├── contrib/             # optional packages built on core's public surface — core never imports it
│   ├── simple_tool_registry/
│   │   ├── __init__.py  # package surface: SimpleToolRegistry, ProxyToolRegistry,
│   │   │                #   PermissionPolicy, YoloPermissionPolicy
│   │   ├── permissions.py # PermissionPolicy (async decide()), YoloPermissionPolicy
│   │   └── registry.py  # SimpleToolRegistry (tools + policy), ProxyToolRegistry (composition)
│   ├── plugins/
│   │   ├── __init__.py  # package surface: BasePlugin, PluginAgentSessionRunner
│   │   ├── plugin.py    # BasePlugin — duck-typed hooks (registry / parts / middleware)
│   │   └── runner.py    # PluginAgentSessionRunner — composes plugins over a ProxyToolRegistry
│   ├── tools.py         # Tool base class + tool()/tool_class() factories — the ERGONOMIC
│   │                    #   Python tool surface (name/description/Args ClassVars,
│   │                    #   get_tool_spec() incl. input_schema, _execute/execute with the
│   │                    #   keyword-only cancellation_token, timeout_in_ms). Contrib, not
│   │                    #   core: nothing in core/ depends on it. Sits at the package root
│   │                    #   because resource_permissions, shell and memory all build on it
│   │                    #   without going through any particular registry.
│   ├── memory/
│   │   ├── __init__.py  # package surface: MemoryPlugin + the scratchpad / todo tools
│   │   └── plugin.py    # scratchpad + todo-list tools, MemoryPlugin
│   ├── compaction/     # a concrete CompactionPolicy (LLM summary + context gauge)
│   │   ├── __init__.py  # package surface: SummarizingCompactionPolicy, gauge
│   │   ├── context.py   # the context gauge (used vs the model's window)
│   │   └── policy.py    # SummarizingCompactionPolicy (should_compact + compact; keep_turns)
│   ├── mcp/             # external MCP servers → tools (optional `mcp` extra; SDK-free config, lazy SDK)
│   │   ├── config.py    # McpServerDef (stdio/http) — no SDK import, so LucaConfig stays SDK-free
│   │   ├── session.py   # open_session/list_tools/call_tool — one short-lived session per call
│   │   └── registry.py/plugin.py/factory.py/oauth.py # ToolRegistry, McpPlugin, build_mcp_plugin, OAuth
│   ├── resource_permissions/
│   │   ├── __init__.py  # package surface: PermissionStrategy, rules, answers, the mixin
│   │   ├── strategy.py  # PermissionMode, ToolRule/ToolKindRule, ApprovalAnswer, PermissionStrategy
│   │   └── mixin.py     # ResourcePermission, AnswerOption, PermissionRequest, ResourcePermissionToolMixin
│   ├── shell/           # the 7 shell tools + ShellAccessPlugin — see AGENTS.md there
│   └── tui/             # the Textual terminal UI (AgentApp + wiring + approval modal);
│       │                #   Textual-free logic in approvals.py / render.py / sessions.py / wiring.py
│       │                #   — needs the `tui` dependency group (a uv default group)
└── core/
    ├── __init__.py      # external surface: AgentSessionRunner, ToolRegistry, PreparedTool,
    │                    #   SystemPromptAssembler, all entry types, exceptions.
    │                    #   NO Tool / tool / ToolContext — those left the core.
    ├── models.py        # AgentSession (incl. the usages + tool_specs stores), all entry
    │                    #   classes (incl. CancelRequested, PrunedEntry),
    │                    #   ToolSpec (+ spec_id()),
    │                    #   ExecutionStatus/ApprovalStatus/ToolExecutionError,
    │                    #   TurnOutcome, RuntimeConfig, SessionConfig, Usage,
    │                    #   SessionRuntimeStatus, ConversationStatus — pure Pydantic v2
    ├── tool_registry.py # ToolRegistry — the 4-method contract the runner drives tools
    │                    #   through (all async, session first) + PreparedTool + the
    │                    #   registry-author rules
    ├── context.py       # CancellationToken (runtime-only; never persisted)
    ├── context_manager.py # ContextManager — context-accounting strategy: per-entry
    │                    #   context_tokens estimation, tool-output processing,
    │                    #   PrunedEntry templates (concrete class; runner default).
    │                    #   All three methods take the live session first.
    ├── compaction.py    # CompactionPolicy contract + CompactionPlan / UsageCounters /
    │                    #   ConversationSnapshot + validate_plan (pure — no session
    │                    #   mutation, no ledger, no asyncio)
    ├── exceptions.py    # AgentError, CancelledError, AlreadyCancellingError,
    │                    #   ToolNotFound, InvalidToolArguments, ProjectionError,
    │                    #   CompactionPlanError
    ├── events.py        # AgentEvent union (block-level + streaming-delta + the
    │                    #   lifecycle events: ApprovalRequired + the three
    │                    #   Compaction* ones); tool events carry deep snapshots
    ├── projection.py    # ConversationProjector — the PUBLIC conversation → LLM-message
    │                    #   strategy (subclass to customize history/tool-output policy)
    ├── adapter.py       # message_to_parts() (inbound response conversion) +
    │                    #   tool_spec_to_luca_tool() (tool-definition conversion)
    ├── middleware.py    # AgentMiddlewareMixin — the 10 duck-typed middleware hooks
    ├── ledger.py        # SessionLedger — the single append/read door onto the entry log
    ├── system_prompt.py # coerce_system_prompt_part, SystemPromptAssembler,
    │                    #   DefaultSystemPromptAssembler, part-input type aliases
    ├── runner.py        # AgentSessionRunner, AgentRun handle, RunResult
    └── utils.py         # pretty_print(session) — the read-only text transcript
                         #   of a session (debugging view; reads the durable
                         #   entries, never the projection)

tests/agent/             # all agent tests; mirrors core/ layout; contrib tests under tests/agent/contrib/
main.py                  # runnable agent demo — launches the contrib TUI
```

`contrib/` packages are library code, but deliberately *outside* the core
contract: each one consumes only the public `luca.agent.core` surface, exactly
like application code would (contrib→contrib dependencies ARE allowed — e.g.
`plugins` builds on `simple_tool_registry`). When adding functionality, decide
first whether it belongs to the core (data model, runner, strategy contracts)
or to a contrib package (everything else). Each contrib package gets its own
docs folder under `docs/agent/contrib/<package>/` and self-scoped tests under
`tests/agent/contrib/`.

## Design principles

Internalize all of these before touching `luca.agent`.

### 1. One serializable session

`AgentSession` and everything it holds is pure Pydantic v2 with `extra="forbid"`. It must round-trip losslessly. Runtime collaborators — the tool registry, the system-prompt strategy — live on the **runner**, never in the session. Nothing on the session is transient.

Two loudly-documented exceptions, and only two:

- `AgentSession.session_runtime_status` is a plain `@property` (not a Pydantic field) that recomputes `SessionRuntimeStatus` from the live entries on every access. It is never serialized and never trusted from disk.
- `ToolExecution.tool_spec` is a restorable CACHE of the spec its `tool_spec_id` names. The id is authoritative; `tool_spec` is stripped from a serialized session and restored on construction, and must never be the source of truth for anything durable.

Both are exceptions because a consumer reads them like ordinary state and a writer must not treat them as such.

### 2. Storage and traversal are separate

`entries: dict[str, AnyEntry]` is the durable, append-only, uniformly-addressable node space.
`Conversation.nodes: list[str]` is the path — an ordered list of entry ids. Walk the path; resolve ids in the store.
`parent_id` is a recovery backstop and is **never traversed**.

### 3. Messages are entries

`UserMessage` and `AssistantMessage` live in `entries` alongside `ToolExecution`, `TurnStart`, `TurnFinish`, and `CompactionEntry`. One `Entry` base class, one `type` discriminator field, one `AnyEntry` discriminated union.

### 4. A tool call is two things

- The request block: a `ToolCall` object inside `AssistantMessage.parts`.
- A separate, mutable `ToolExecution` entry — the durable source of truth about that call's whole lifecycle — correlated by `tool_call_id`.

`ToolExecution` is one of the **two** mutable entry types (`CompactionEntry` is the other — see below).

`tool_executions: dict[str, list[str]]` is a denormalized index from `tool_call_id` → execution-entry ids.

Each `ToolExecution` carries three orthogonal facts plus its provenance:
- `status: ExecutionStatus` — the framework's execution lifecycle and ONLY that: `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `NOT_FOUND`, `INVALID`, `REJECTED`, `CANCELLED`, `INTERRUPTED`, `TIMED_OUT`. `COMPLETED` means "the framework received a result", not "the tool succeeded".
- `approval_status: ApprovalStatus | None` — the CURRENT approval state (`None` = the policy never processed it; `PENDING`/`ALLOWED`/`REJECTED`). Always read approval from this field; `approval_decisions` is the append-only audit log of policy responses (only PENDING may repeat), never the source of current state.
- The outcome payload: exactly one of `result: ExecutionResult` (the body returned; `result.is_error` is the tool's OWN verdict — an `is_error=True` result is still `COMPLETED`) or `error: ToolExecutionError` (structured `error_type`/`error_message`/`details`, populated for `FAILED`/`NOT_FOUND`/`INVALID`) — or neither, for the status-only terminals.
- `raw_tool_call: ToolCall` — the (possibly middleware-effective) request; makes the execution self-contained. `tool_spec_id: str | None` — the durable reference into `AgentSession.tool_specs`, with `tool_spec: ToolSpec | None` as its restorable cache (`name`, `description`, `input_schema`, `metadata`, `tool_kind`, `namespace`, `version`, `timeout_in_ms` — NO arguments; both `None` when the tool never resolved). `extras` — a free-form dict written by registries/middleware, stored verbatim, never interpreted by the core (`SimpleToolRegistry` stores the tool's approval context under `extras["approval_context"]`).
- Lifecycle timestamps: `started_at` (set iff the body was dispatched — true for EVERY outcome, because the runner persists it only after `prepare()` has returned), `ended_at` (every terminal transition), `cancel_signalled_at` (run cancellation only — a deadline never sets it). `updated_at` is ledger bookkeeping, not timing.

**Tool specs are normalized.** Each distinct `ToolSpec` is stored once per session in `AgentSession.tool_specs` under `spec_id()` — a full 64-char SHA-256 hex over the spec's JSON with recursively sorted keys, no whitespace, non-ASCII literal, UTF-8. The rule is pinned rather than left to the implementation and is deliberately NOT an overridable hook like `generate_id()` / `now_ms()`: it has to be identical across processes, machines and other-language implementations, so determinism here is data integrity, not test convenience. A spec must be a pure function of the tool DEFINITION — anything call-scoped in `ToolSpec.metadata` mints a row per call and silently defeats the whole thing. The store is append-only and never garbage-collected; a content hash's only failure mode is a redundant row, never a wrong lookup.

Constructing an `AgentSession` restores every `tool_spec` from its id (shared BY REFERENCE with the `tool_specs` row) and refuses two shapes: a `tool_spec_id` absent from `tool_specs`, and a `tool_spec` with no `tool_spec_id` (which can only come from a pre-normalization file, which would otherwise load fine and then lose every spec on the first save). There is no migration — regenerate pre-refactor session files.

These combinations are framework conventions, not Pydantic validators — middleware is trusted and may author unusual state; the application owns the consequences.

### 5. The wire payload is derived, never stored

The **`ConversationProjector`** (`projection.py`) is the public strategy that recomputes the LLM message list from the path on every call: it drops `TurnStart`/`CancelRequested`, projects a CANCELLED `TurnFinish` as the synthetic `[Request interrupted by user]` marker, unwraps messages into client content blocks, projects each terminal `ToolExecution` to its correlated `ToolMessage` (COMPLETED → its result verbatim; every other terminal → derived error text with `is_error=True`), and renders a `CompactionEntry` as a synthetic user message — with one POSITIONAL rule that lives on `project()` itself: a whole compaction bracket (`ts_c cmp [cr] tf_c`) projects as nothing, whatever its outcome, while a `CompactionEntry` outside a bracket projects its `parts`. It is a concrete class — pass a subclass as `conversation_projector=` to change any policy (history shaping, redaction, tool-output wording); ALL default derived wording lives on the class. The same `project_tool_execution` output feeds the `ToolExecuted` event's `result_text`/`is_error`, so event and wire never disagree. There is no projection middleware; `before_llm_call` stays as the downstream last-mile hook.

### 6. Fail loud on a mid-execution projection

Projecting a conversation that contains a `PENDING` or `RUNNING` `ToolExecution` raises `ProjectionError` — the runtime must never call the model while an execution is nonterminal. Missing entry ids, unknown entry types, and a COMPLETED execution without a result fail the same way; projection never invents fallback content.

### 7. Turn boundaries are markers, not objects

Each loop iteration is bracketed by a `TurnStart` entry and a `TurnFinish` entry (a boundary + outcome record only — no usage rollup). There is no "Turn" object.

### 8. Determinism by extension, not injection

The runner carries no test parameters. Every id and timestamp flows through two overridable hook methods:
- `generate_id()` — returns a uuid by default.
- `now_ms()` — returns wall-clock milliseconds by default.

Tests layer determinism from outside by subclassing (`DeterministicRunner` in `tests/agent/scenarios.py`): `ids` scripts the id hook across every call (`post_message`, `run`, resume, `cancel()`), and `now` freezes the clock.

`provider=` is not test scaffolding — it is a zero-logic passthrough of the client's `provider=` kwarg, which is also how tests inject a `FauxProvider`.

### 9. One handle, three consumption forms

Both `run()` and `start()` return an `AgentRun`. That handle supports three patterns:

1. `await run` — drives/joins to the next stopping point and returns a `RunResult` (status + outcome + pending_approvals; no usage — read `AgentSession.usages`).
2. `async with run: async for event in run` — iterates events. **Iteration requires the context manager.** For a lazy run, iteration IS the engine. For an eager run, iteration reads the background task's buffer at the consumer's pace.
3. `run.cancel()` — delegates to `runner.cancel()`.

Additional semantics:
- `start()` opens the `TurnStart` bracket synchronously at call time, so an immediate cancel parks the flush.
- One cursor per handle; a second `await` returns the cached result.
- Exiting a lazy `async with` block always **suspends** (closes the provider stream, re-derives status, finalizes) — it never advances the engine. A later `run()` resumes the same bracket.
- Cancel returns normally with `outcome=CANCELLED`.
- TIMED_OUT or ERRORED closes the turn and re-raises (`run.result` stays None), unless a cancel is pending — then the wind-down consumes the failure and the run returns normally.
- `on_event` (sync or async) receives every event even when the run is only awaited (not iterated).

### 10. Two-tier events

The engine yields `AgentEvent` union members in two tiers:

- **Block events** (always): `ReasoningBlock`, `TextBlock`, `ToolCallReceived`, `ToolExecutionStarted`, `ToolExecuted`, `FinishReason`.
- **Delta events** (`streaming=True` only): `ReasoningStart`/`Delta`, `TextStart`/`Delta`, `ToolCallStart`. Session behavior is identical regardless of streaming.
- **Lifecycle events** (always): `ApprovalRequired` fires as the last event before a gate, carrying the awaiting `ToolExecution` entries (equivalent to `runner.pending_approvals()`); `CompactionScheduled` / `CompactionStarted` / `CompactionFinished` carry a deep `CompactionEntry` snapshot through one compaction's lifecycle, `Finished` firing whatever the outcome.

The three tool-lifecycle events carry a deep `ToolExecution` SNAPSHOT (plus a denormalized `tool_call_id`), never a live ledger reference — tool name and arguments come from `execution.raw_tool_call`. Per execution: `ToolCallReceived` fires once at the persisted birth state (PENDING, or a preflight-terminal NOT_FOUND/INVALID/FAILED); `ToolExecutionStarted` fires iff the body dispatches, after RUNNING is persisted and immediately before invocation; `ToolExecuted` fires once at the terminal outcome, with `result_text`/`is_error` copied from the projector's `project_tool_execution` output. Every event follows the persistence of the state its snapshot shows — the stream never leads the durable session.

Every text-bearing event exposes its content on `text`, so one `match` statement serves both vocabularies.

There is no `TurnFinished` event — `RunResult` is the completion signal. A flush run may emit zero events.

### 11. Resumable status machine

`Conversation.status` (`ConversationStatus`: IDLE / PENDING / RUNNING / AWAITING_APPROVAL / CANCELLING) is set by the runner and persisted, but treated as a denormalized cache. When `AgentSessionRunner.__init__` takes a session, status is re-derived from the entries (so a stale RUNNING self-heals).

Status derivation rules:
- `AWAITING_APPROVAL` — an open-turn execution has `approval_status=PENDING`; resolve out-of-band, then call `run()`.
- An execution with `approval_status=None` (crash mid-decide) → plain PENDING — `run()` self-heals by asking the registry again.
- An orphaned `RUNNING` execution (crash mid-body) → plain PENDING — the next drive recovers it to `INTERRUPTED` (no re-dispatch) before doing anything else.
- Open turn with an unconsumed `CancelRequested` → `CANCELLING` — the next drive is the flush.
- Closed turn whose trailing `TurnFinish` is TIMED_OUT or ERRORED → PENDING (retry-ready).
- A closed COMPACTION bracket (`ts_c cmp tf_c`) is transparent — skipped, repeatedly if several stack, and the leaf before it derives. Without it a failed compaction reads as retry-ready (a spin) and a completed one buries a queued user message. An *open* compaction bracket derives PENDING like any open turn.

A logical turn spans one `TurnStart`/`TurnFinish` bracket even across an approval pause. A `TurnStart` with no later `TurnFinish` means resume, not re-open.

`post_message` requires a closed bracket and IDLE or PENDING status. It always rejects CANCELLING and AWAITING_APPROVAL — and an open compaction bracket, durably, until the compaction has been driven.

### 12. Cancel is a pure signal

`cancel(outcome, error)` (callable in any state):
- Appends a durable `CancelRequested` to the open turn.
- Trips the live run's cancellation token.
- Sets status to CANCELLING.
- Returns immediately — performs NO bookkeeping.

**Every collaborator await is raced against the token** — all four `ToolRegistry` calls, the prepared callable, the LLM call and its streaming steps, and compaction — so no registry or tool-owned code can make `cancel()` a no-op. The four registry calls use a ZERO grace window and AWAIT the killed task's unwinding, so a registry's `finally` / `async with` completes before the run moves on (which is what makes "never swallow `asyncio.CancelledError`" a hard rule rather than advice). Only the tool body keeps a grace window and a detached kill, because thread-backed work cannot be interrupted. The registry is never handed the token: there is no partial answer worth having from listing tools, minting a record, deciding an approval, or preparing a dispatch.

Cancellation never writes a terminal status of its own inside a registry phase — it stops the phase and leaves the call in a state the existing machinery already knows how to finish:

| Cancelled during | Durable outcome |
|---|---|
| `get_tools` | no LLM call; back to the loop top; the turn winds down, executions unaffected |
| `create_execution` | a PENDING draft is synthesized so the call still gets its one execution; wind-down records CANCELLED |
| `decide` | no decision recorded, approval state untouched; wind-down records CANCELLED |
| before a ready execution's dispatch begins | left PENDING, the batch stops; wind-down records CANCELLED |
| `prepare` | no RUNNING row, no `ToolExecutionStarted`; the DISPATCH path records CANCELLED **in place** |
| the prepared callable | unchanged grace machinery — COMPLETED / FAILED / INTERRUPTED |

The `prepare` row is the one asymmetry, and it is deliberate: `before_tool_execution` has already fired for that call, so handing it back to the loop-top wind-down would fire the hook twice. **The hook is the boundary** — an execution whose hook has not fired belongs to the wind-down, one whose hook has fired belongs to the dispatch path. Both produce the same durable shape.

Wind-down itself happens at the engine's step boundaries and turn-close sites:
- Still-PENDING executions → stamped `cancel_signalled_at`, then `CANCELLED` (approval state untouched; a DENIED call was already terminal `REJECTED` at decision time). **A cancelled birth is CANCELLED, never FAILED** — a cancellation is not a tool failure.
- In-flight executions → persisted with `cancel_signalled_at` FIRST, then the grace period: a within-grace return is `COMPLETED` with its real result (keeping the stamp), a raise is `FAILED`, expiry is `INTERRUPTED`.
- Already-terminal executions are untouched.
- Closes the turn with `TurnFinish(outcome)`.

A response containing N tool calls yields N tool executions, even when cancellation lands mid-batch. This is why the `create_execution` race is per call, inside the birth helper, and never around the `asyncio.gather`: killing the gather would lose every draft.

No `except CancelledError` clause is needed anywhere in the runner, and adding one would mislead the next reader. `asyncio.CancelledError` derives from `BaseException`, so the broad `except Exception` around `create_execution` never sees it; and the race helper absorbs the kill it issued and reports the outcome as a boolean rather than re-raising.

An unconsumed cancel controls every close: an LLM answer landing within the grace window is recorded but the turn still closes with the cancel outcome; an LLM failure within the grace window is discarded and the run returns normally.

A parked cancel survives save/reload — the next `run()` or `start()` is the flush (instant, no LLM call).

A second `cancel()` while one is unconsumed raises `AlreadyCancellingError` (first call wins).

No open turn → no-op (only possible on an undriven lazy handle or before any run; `start()` opens the bracket at call time, so a started run is always cancellable).

Wire projection: a cancelled turn becomes a synthetic user message `[Request interrupted by user]`. Failed turns project nothing.

## Key facts

### Context vs usage

Two different measurements, never conflated:

- **`Entry.context_tokens`** — the intrinsic estimated size of that entry's model-facing content, shared with the entry across every conversation that references it. Calculated by the runner's **`ContextManager`** collaborator (`context_manager.py`, passed as `context_manager=`, defaults to the simple built-in: one token per 4 characters) on every NEW entry before `before_entry_written`, and recalculated on a `ToolExecution`'s terminal transition before `after_tool_execution`. Middleware has the final say — nothing is recalculated, validated, or repaired after it. Never derived from provider usage.
- **`AgentSession.usages[conversation_id][entry_id]` → `Usage`** — the provider-reported consumption for one entry in one conversation (the same assistant entry in two conversations can have different usage: input covers the whole request context). A self-describing association record (`conversation_id` + `entry_id` are required fields), written only through `SessionLedger.record_usage()` when an assistant message is recorded. Entries carry NO usage field; `TurnFinish` carries no rollup; `RunResult` carries no usage — aggregate from the store.

All three `ContextManager` methods take the live session first, so one argument order describes the whole contract and every policy sees the same state — the active model included, which is what makes a real tokenizer implementable at all:

```python
def calculate_context(self, session, entry) -> int
def prune_entry(self, session, entry) -> PrunedEntry          # NO framework call site
def process_tool_output(self, session, execution, result) -> ExecutionResult
```

`process_tool_output()` transforms a returned `ExecutionResult` before the terminal execution is constructed (identity by default) — the durable session, the `ToolExecuted` event, and the wire all see the processed output. It receives the `ToolExecution` IN TRANSITION (status still RUNNING, `result` not yet attached): read it for identity — `tool_spec`, `raw_tool_call.name`/`arguments` — never for outcome. That is what makes "truncate `bash` output at 30k characters but never truncate `read`" expressible.

Two `calculate_context` gotchas, both silent when violated: it runs on EVERY new entry, so scanning `session.entries` inside it makes a turn quadratic (cross-entry work belongs in `prune_entry` / `process_tool_output`, which run rarely); and on an append it runs INSIDE the ledger's build callback, so the entry already has its `id` but is not yet a member of `session.entries` — an implementation that looks itself up there raises `KeyError` on every append.

Because counts are STORED on the entry, a model-aware `ContextManager` goes stale the moment `llm_config` changes. `AgentSessionRunner.recalculate_context_tokens()` re-derives `context_tokens` for every entry in `session.entries` — every entry, not just the active path, because the count is intrinsic to an entry and shared by every conversation referencing it — threading each through `before_entry_written` and setting no other field. **Nothing in the framework calls it**: no constructor keyword, no CLI flag, no automatic invalidation on a model switch (which would put an unbounded rewrite behind an innocuous assignment). The shipped `ContextManager` is a character estimate no model choice affects; the method is there for the application that swaps in a real tokenizer, and that application calls it.

**Pruning** replaces an entry's contribution to the path without touching the original: `ContextManager.prune_entry()` builds a `PrunedEntry` TEMPLATE (placeholder identity; v1 supports terminal tool executions only, replacement text `"[tool output has been pruned to reduce context]"`), and `SessionLedger.prune(original_id, build)` stamps identity (the original's `parent_id`), verifies the referent/type/terminality invariants, and swaps the node id in place. The original stays in `entries` and in the `tool_executions` index. `ConversationProjector.project_pruned` resolves the referent and re-emits the replacement content under the original's role and `tool_call_id` (a missing referent, type mismatch, or unprojectable source raises). The runner deliberately exposes NO public prune/context-total methods yet.

### The tool registry

The runner is constructed with one `ToolRegistry` (`luca/agent/core/tool_registry.py`; `None` = toolless agent). The core touches tools through exactly four methods — all async, all taking the live `AgentSession` first, none receiving the cancellation token:

```python
class ToolRegistry:
    async def get_tools(self, session) -> list[ToolSpec]: ...
    async def create_execution(self, session, call) -> ToolExecution: ...
    async def decide(self, session, tool_execution) -> ApprovalDecision: ...
    async def prepare(self, session, tool_execution) -> PreparedTool: ...

PreparedTool = Callable[..., Awaitable[ExecutionResult]]
async def run(*, cancellation_token: CancellationToken) -> ExecutionResult: ...
```

- **`get_tools` is dynamic** — queried fresh per LLM call (the result may vary with session state); never a lifecycle hook. It returns `ToolSpec`s, so a registry backed by JSON Schema needs no Python class. An exception propagates and aborts the run; the runner never substitutes an empty tool list, because calling the model with no tools when the registry meant to offer some silently changes the answer.
- **`create_execution` returns a birth DRAFT** with no identity (`id`/`created_at` are `None`). The registry owns the call-scoped facts: `raw_tool_call`, `tool_spec` (incl. `timeout_in_ms`; `None` if unresolved), the birth `status` (PENDING, or terminal-at-birth NOT_FOUND/INVALID/FAILED), the `error` for a terminal birth (the registry authors it), and `extras`. The RUNNER stamps `id`/`parent_id`/`created_at`/`ended_at`-if-terminal/`context_tokens`/`is_doom_loop_flagged`, so determinism principle 8 holds, and the LEDGER files the spec and stamps `tool_spec_id` — registries stay unaware of the storage scheme. If `create_execution` raises (or the registry is `None`), the runner synthesizes the draft itself (FAILED / NOT_FOUND) — failures stay isolated per call.
- **`decide`** returns ALLOW / DENY / PENDING; exceptions propagate and abort the run (the executions stay unresolved; the next `run()` asks again), so implementations must be idempotent queries of their own state.
- **`prepare`** resolves the tool and validates the arguments, then returns a CALLABLE that runs the body — it must not run it. Raising means the body never runs and the execution is never marked RUNNING: `ToolNotFound` → NOT_FOUND, `InvalidToolArguments`/pydantic `ValidationError` → INVALID, anything else → FAILED, all with `started_at=None`. Once the callable is invoked, every raise is FAILED. A returned `ExecutionResult` → COMPLETED (after `ContextManager.process_tool_output`). It is called once per dispatch attempt and only for an already-approved execution.

The arguments are easy to misuse in opposite directions. The **`AgentSession` is the LIVE object** — the same instance the runner and ledger write through — so a registry may hold it and re-read current state later, including from inside a prepared callable. The **`ToolExecution` is a SNAPSHOT**, detached as of the moment the call was made; the runner persists RUNNING and `started_at` AFTER `prepare()` returns, so a reference captured during `prepare()` is already stale. Capture what the callable needs during `prepare()`. And the session is **read-only to every implementation** — the runner owns every write, `session.tool_specs` included.

The registry-author rules (`prepare()` must be re-callable and non-blocking, must not return holding a lock/lease/slot, must never swallow `asyncio.CancelledError`, blocking sync work belongs in `asyncio.to_thread`, and **the core never validates arguments** even though it now knows every schema) live in full in `tool_registry.py`'s module docstring. Every one of them fails silently when violated; read them before writing a registry.

There is **no global permission gate anywhere**: each registry answers `decide()` for its own tools. Cross-cutting approval is an application composition pattern (share one strategy across registries), never a framework or plugin API concern.

The engine has exactly **one** `decide()` call site — the top of its loop. Any open-turn execution that is undecided (`approval_status` is `None` or `PENDING`) is handed to the registry. Sibling undecided executions are decided concurrently via `asyncio.gather`. Each response both updates `approval_status` directly and appends to the `approval_decisions` audit log; a DENY is terminal on the spot (`status=REJECTED`, `ended_at` stamped, outcome middleware runs, `ToolExecuted(REJECTED)` emitted).

A PENDING decision defers only THAT execution: every ALLOWED sibling proceeds to dispatch, and the runner parks (status → `AWAITING_APPROVAL`, `ApprovalRequired` as the final event) only after all currently runnable work has advanced. The model is never called again until every tool call in the assistant response has a terminal execution and a correlated tool output. Re-entering `run()` does not raise — it simply asks the registry again. `runner.pending_approvals()` returns the awaiting `ToolExecution` objects.

**Dispatch is PREPARE, then run** (per ready `PENDING`+`ALLOWED` execution, in order):

1. `before_tool_execution` middleware — its returned `raw_tool_call` is the effective call, which is why the hook stays AHEAD of `prepare()`.
2. `registry.prepare(...)`, raced against the token.
3. A raise, or a return that is not callable, terminalizes the execution WITHOUT it ever being marked RUNNING — `started_at` stays `None`, no `ToolExecutionStarted`, `details["phase"] == "prepare"`. For the not-callable case the runner synthesizes the failure itself.
4. A cancellation observed at any point up to and including `prepare()` settling means the body is NOT dispatched, even when `prepare()` returned successfully: grace exists to let in-flight work finish, not to start new work after a cancel was requested.
5. Otherwise persist `RUNNING` + `started_at` (the birth `tool_spec` stands — no dispatch-time re-snapshot), emit `ToolExecutionStarted`, and invoke the callable under the cancellation race + deadline (`tool_spec.timeout_in_ms`, else `RuntimeConfig.tool_execution_timeout_in_ms`). The INVOCATION sits inside the failure handling: a callable returning a plain value has already been invoked, so it records as a post-dispatch failure rather than crashing the run.

A returned result is `COMPLETED` (whatever `result.is_error` says); every post-dispatch raise is `FAILED`, with a structured `ToolExecutionError` built by the overridable `runner.to_tool_execution_error(execution, exception, *, phase)`; deadline expiry is `TIMED_OUT`; grace expiry is `INTERRUPTED`. `after_tool_execution(execution, exception)` observes EVERY outcome before the final persist (registry-authored terminal births carry no live exception).

Three consequences worth stating separately:

- **The exception-type-to-status mapping did not disappear — it MOVED to `prepare()`**, where it is accurate, because the only work done at that point is resolution and validation. A tool BODY that raises `ToolNotFound` looking up a sub-resource records FAILED, not NOT_FOUND; same for a body raising a pydantic `ValidationError`.
- **`ToolExecutionStarted` is emitted iff the body was dispatched** — strictly less often than before.
- **`before_tool_execution` fires exactly once per DISPATCH ATTEMPT**, never twice for one outcome. Everything that terminalizes an execution after the hook has run goes through `_finalize_outcome`, never `_finalize_undispatched` (which re-runs it). A crash during `prepare()` writes nothing, so the next drive fires it again over the ORIGINAL call — deliberately NOT once-per-call-forever.

**Invariant: every tool call produces exactly one tool output, always.** Now load-bearing in three new places: a cancelled birth, a cancelled decide, and a `prepare()` failure each still produce their one execution and one output.

A call that is terminal at birth is persisted with a structured `error` and never reaches `decide()` (`approval_status` stays `None`); it still passes through the tool middleware pair.

The batteries-included registries live in `luca/agent/contrib/simple_tool_registry/`: `SimpleToolRegistry(tools, permission_policy)` reproduces the classic preflight (resolve → validate → duck-typed `get_approval_context`, stored under `extras["approval_context"]`), delegates `decide()` to a `PermissionPolicy` (async `decide(session, tool_execution)`; `YoloPermissionPolicy` allows everything), and returns from `prepare()` a closure binding the validated arguments and the session; `ProxyToolRegistry(*registries)` composes registries (`get_tools` recomputes and caches a `{name → child}` route, duplicate names raise). The proxy's `decide` and `prepare` resolve INDEPENDENTLY of that cache, warming it once from the children on a miss — a call left pending approval by a previous process now resolves on a fresh one, is gated by its owning child, and dispatches. That is a correctness requirement, not a convenience: once `prepare()` can resolve without the cache, the old blanket ALLOW-on-cache-miss becomes a permission bypass on a cold resume. ALLOW on a genuinely unresolvable name STAYS — `prepare()` then raises `ToolNotFound` and the call records the honest NOT_FOUND, where DENY would record REJECTED for a tool that never existed. Everything richer — modes, rules, resource globs, answer-decoupled interactive approval, the `ResourcePermissionToolMixin` — lives in `luca/agent/contrib/resource_permissions/` and is driven interactively by `main.py`.

### Compaction

Replacing the older span of a conversation with a summary of it. **Compaction opens a new conversation inside the same session** — it does not create a new session: add one entry, archive the current view, open a new one over the ids that survive. `conversation_history` keeps the exact pre-compaction path, every compacted entry stays in `entries`, and `CompactionEntry.compacted_nodes` records precisely which ids the summary replaced.

The runner is constructed with one `CompactionPolicy` (`luca/agent/core/compaction.py`; `None` = compaction never happens). Two methods, and they own the whole decision:

```python
class CompactionPolicy:
    def should_compact(self, session) -> bool: ...                     # SYNC — start() consults it at call time
    async def compact(self, session, nodes, entry) -> CompactionPlan | None: ...
```

- **`nodes`** is the path the policy may rewrite: the active path minus this compaction's own `TurnStart`, ending with `entry`. `plan.nodes` may carry any of those ids in any order with new entries interleaved, and NOTHING else — an id outside the tuple is a plan rejection, so a policy never has to route around framework markers. `plan.nodes = list(nodes)` is a legal full carry.
- **`entry`** is a DEEP COPY of the committed `CompactionEntry`; the runner applies exactly `parts`, `llm_config` and `metadata` from the returned plan and discards the rest. The copy is load-bearing: a policy that wrote `parts` onto the live entry and then failed would leave a summary projecting onto an unchanged path.
- **`plan.nodes`** is the new conversation before ids exist: a `str` carries an existing node over, an entry object is created there (the runner stamps `id`/`parent_id`/`created_at`, one timestamp, parents threaded left to right).
- **`plan.usage`** is a typed `UsageCounters` recorded for the ATTEMPT, against the pre-compaction conversation.

`CompactionEntry` is the second mutable entry type: written when the intent exists (`schedule_compaction()` or `should_compact`), mutated as it progresses, left in its terminal state whichever way it ended. It has NO `status` field — the surrounding turn bracket owns how the attempt ended and the entry owns what it produced.

**Runner integration.** The compaction step runs at the top of `_drive`, BEFORE the conversational bracket: flush a parked cancel → resume / skip / decide → run it → then drive the turn. At most one per drive (structural — the step sits outside the loop), and never while a conversational turn is open. `start()` decides at call time so an eager run opens a compaction bracket instead of a `TurnStart`.

**Safety.** Everything fallible happens before the transition; `SessionLedger.transition_conversation` is the atomic region and contains only plain assignments. `validate_plan` refuses a plan that references an unknown or un-offered id, references one twice, is empty, omits the compaction entry, was computed against a conversation that has since moved, or carries no content — STRUCTURE only, never meaning. A `source=USER` failure raises; a `source=POLICY` failure degrades so the user's turn survives; a cancel always stops the drive. An interrupted compaction resumes in place (the test is `entry.parts is None`, not the bracket's shape); a closed bracket is never retried.

Details in `docs/agent/12-compaction.md`. No default policy ships yet — the summarization strategy is planned as contrib.

### Tool identity

`ToolSpec` (in `models.py`) is the core's ONLY tool type, and it serves two roles at once: the **advertisement** sent to the model (`name`, `description`, `input_schema`) and the **historical identity snapshot** attached to a past execution (`tool_kind`, `namespace`, `version`, `timeout_in_ms`). That conflation is a deliberate, accepted trade-off — normalization removed the storage cost that made it expensive, and having the exact schema the model was shown attached to the execution is what makes an old session auditable and replayable at all. The alternative (a thin history type and a fat wire type) was rejected: it reintroduces the split this design exists to remove and forces every registry to produce both.

Nothing in a restored spec references a live class, an `Args` model, or anything importable — it is plain data, so a session whose tools were deleted from the codebase years ago still renders its `name`, `description`, `input_schema` and `tool_kind`. `input_schema` is required and never `None`, including for a tool that takes no arguments (that case is the empty object schema `{"type": "object", "properties": {}}` — an absent schema and an empty schema mean different things to a provider). `tool_kind` defaults to `OTHER`; the network-egress kind is `WEB_FETCH`. Invocation arguments are never here — they live on `ToolExecution.raw_tool_call`.

`Tool` (in `luca/agent/contrib/tools.py`) is contrib: the ergonomic way to write a tool in Python, and the EXECUTION contract only. It declares `tool_kind`, `namespace`, `version` and `timeout_in_ms` as `ClassVar`s, and `get_tool_spec()` stamps them plus `input_schema=Args.model_json_schema()` into the snapshot. `execute` / `_execute` receive the live `AgentSession` (read-only) and the keyword-only `CancellationToken`.

There is no `get_approval_context` on the base class — it is a duck-typed convention read by `SimpleToolRegistry` (`async get_approval_context(args, session) -> dict`, receiving the **validated** args; `resource_permissions.ResourcePermissionToolMixin` provides it). The core never mentions it. It is awaited inside `create_execution`, on the event loop and under no deadline, so blocking work belongs in `asyncio.to_thread` — which is exactly what the mixin does with its synchronous `build_permission_requests` override point, whose path stats would otherwise stall the run on a hung mount.

### Timeouts and step limits

All config rides on `SessionConfig.runtime_config` (a `RuntimeConfig`), which persists with the session. The runner reads it live — not from its constructor.

**Timeout fields:**

| Field | Effect |
|-------|--------|
| `builtin_client_completion_timeout_in_ms` | Client per-phase `timeout=`. INERT when the runner is built with a provider instance. |
| `client_completion_timeout_in_ms` | Client wall-clock `total_timeout=` (async helpers only). |
| `tool_execution_timeout_in_ms` | Outside deadline on the PREPARED CALLABLE only; beaten by the birth `ToolSpec.timeout_in_ms` (stamped from the tool's `timeout_in_ms` ClassVar). Expiry → `TIMED_OUT`, resultless. |
| `*_cancellation_grace_period` | Grace window for cancel races. 0 = immediate hard cancel; a tool returning within grace records its real result. Applies to the tool BODY and the LLM call — the four registry calls are raced with grace 0. |

> **Deadlines bound the body, not the call.** `get_tools`, `create_execution`, `decide` and `prepare` have NO deadline — they are contractually local and non-blocking, so bounding them buys little and would add a second timeout tier. A tool configured with `timeout_in_ms=5000` is therefore **not bounded end to end**: the 5s bounds its body. Cancellability is what stops a user's `cancel()` from being ignored during those four; stopping an *unattended* run from hanging on registry preflight is a separate decision, deliberately not taken. A registry that does I/O in `prepare()` in spite of the contract owns its own timeout.

**Step-limit fields:**

| Field | Effect |
|-------|--------|
| `hard_max_steps` | If `AssistantMessage` count in the open turn reaches this limit, the engine closes the turn `TurnOutcome.ERRORED` and returns (status → PENDING, retry-ready; no raise). |
| `soft_max_steps` | When reached and `limit_tool_choice_on_soft_max_steps_reached` is True, the next LLM call gets `tool_choice="none"`, forcing a text-only response. |
| `doom_loop_threshold` | When the same tool call (name + parameters) appears consecutively this many times in the current turn, `ToolExecution.is_doom_loop_flagged` is set True on the Nth occurrence. If `limit_tool_choice_on_doom_loop_flagged` is True, subsequent LLM calls in the same turn get `tool_choice="none"`. |

All int fields use -1 (Inf) or 0 to disable. Constructing a runner where `soft_max_steps == hard_max_steps > 0` emits a `UserWarning` (hard prevails). Timeout fields are milliseconds; limit fields are plain ints. `Seconds()` / `MilliSeconds()` convert durations. Seed config via `AgentSessionRunner.new_session(..., runtime_config=)`.

### System prompt parts

The runner takes no `system_prompt` string. Instead it takes `system_prompt_parts` — a list whose items are any of (machinery in `luca/agent/core/system_prompt.py`):

- a `SystemPromptPart` (fields: `text`, `source`, `priority`);
- a `str` → `SystemPromptPart(text=...)`;
- a dict with `text` + optional `priority` / `source` → validated strictly;
- a callable `(session_config, runtime_status) -> ` any of the above, invoked fresh before every LLM call.

Static parts are coerced eagerly at construction (`coerce_system_prompt_part` — a bad part raises `TypeError`/`ValidationError` at `__init__`); callables resolve per call and their return value is coerced the same way.

Before every LLM call, `build_system_message()`:
1. Resolves the parts (callables get the live `SessionConfig` and the freshly computed `SessionRuntimeStatus`).
2. Sorts parts by `priority` (ascending).
3. Assembles via `SystemPromptAssembler.assemble_system_prompt(parts) -> str`; if blank, sends no system message.

The assembler is optional and duck-typed (concrete base, no ABC — override the one hook); `DefaultSystemPromptAssembler` newline-joins the part texts. A runner with no parts sends no system message.

`SessionRuntimeStatus` (in `models.py`) carries:
- `step_count` — `AssistantMessage` count in the open turn.
- `turn_count` — total `TurnStart` entry count.
- `status` — current `ConversationStatus`.

It is always recomputed via `AgentSession.session_runtime_status` (a `@property`, not a serialized field).

### Reasoning durability

`AssistantMessage.parts` retains `ThinkingContent`, so reasoning is durable in the saved session and survives reload — text, `signature` and `redacted` alike.

Whether it goes back on the wire is the transport's call: OpenAI-compatible hosts have no replay surface and drop it, Anthropic requires the signature during tool use and replays the block verbatim. A `ThinkingContent` is therefore immutable once persisted; rewriting `thinking` in middleware invalidates the signature and the provider will reject the turn.

## Common tasks

### Add an entry type

1. Add the Pydantic class to `luca/agent/core/models.py`: subclass `Entry`, give it a `Literal[...]` `type` discriminator.
2. Add it to the `AnyEntry` union.
3. Re-export it from `luca/agent/core/__init__.py`.
4. Handle it in `ConversationProjector.project_entry` (add a `project_<entry>(entry, entries)` method that projects or returns `None`) — an unknown entry type raises.
5. Decide its `ContextManager.calculate_context` ownership (the default counts nothing for unknown types).
6. If the loop emits it, handle it in `AgentSessionRunner`.
7. Tests go in `tests/agent/` (projection cases in `test_projection.py`).

### Add a tool

Subclass `Tool` from `luca.agent.contrib.tools` (in your application, or a contrib package). It is contrib, not core — a registry that already has JSON Schema never needs it and can hand back hand-written `ToolSpec`s instead.

- Set `name`, `description`, and `Args` (a Pydantic model — `get_tool_spec()` turns it into the `input_schema` the model is shown) as class vars.
- Set `tool_kind` (a `ToolKind`; carried on the `ToolSpec` snapshot), and optionally `namespace` / `version` / `timeout_in_ms` (also snapshotted).
- Override `async _execute(args, session, *, cancellation_token) -> str` for simple tools, or `async execute(args, session, *, cancellation_token) -> ExecutionResult` for rich output (`is_error`, `metadata`, multi-block). An `is_error=True` result still records `COMPLETED` — `is_error` is the tool's own verdict, not a lifecycle fact.
- Both receive the LIVE `AgentSession` (read `session.id` / `session.session_config.llm_config`; treat it as READ-ONLY — the runner owns every write) plus the keyword-only `CancellationToken`. Per-run application state is not the framework's concern: a tool is application code and can close over its own references or read a `contextvars.ContextVar`.
- `SimpleToolRegistry` validates LLM-produced arguments through `Args` at birth (for the birth status) and again in `prepare()` (for dispatch). Malformed args become a terminal `INVALID` execution with a structured `ToolExecutionError` and never reach `decide()`.
- Timing is recorded by the runner on the execution (`started_at`/`ended_at`), never on the result. `timeout_in_ms` bounds the BODY only.
- Define `async get_approval_context(args, session) -> dict` (the duck-typed convention) to describe the call for `SimpleToolRegistry`'s permission strategy.
- `tool()` / `tool_class()` build one from plain callables for runtime-constructed tools; subclassing stays the recommended mechanism.

Wrap **instances** in a registry and pass that to the runner:

```python
from luca.agent.contrib.tools import Tool
from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry, YoloPermissionPolicy

registry = SimpleToolRegistry(tools=[MyTool()], permission_policy=YoloPermissionPolicy())
runner = AgentSessionRunner(session, tool_registry=registry)
```

The runner projects `get_tools()`'s `ToolSpec`s to wire tools via `adapter.tool_spec_to_luca_tool` (`input_schema` straight through as `parameters`); the registry dispatches by name.

### Add middleware

Write a plain Python class that implements any of the 10 hook methods defined in `luca/agent/core/middleware.py` (`AgentMiddlewareMixin` — its hooks are identity pass-throughs, so subclassing is safe for partial overrides, but plain classes are the recommended style; the runner dispatches via `hasattr`). Pass instances as `middleware=[mw1, mw2]` to `AgentSessionRunner`. Hooks run in list order; there is no reverse ordering. The tool pair works on the whole execution: `before_tool_execution(execution) -> execution` (pre-dispatch for allowed calls — its `raw_tool_call` is the effective call, which is what `prepare()` then resolves and validates from — and once for terminal-at-birth / rejected / cancelled-before-dispatch calls) and `after_tool_execution(execution, exception=None) -> execution` (every outcome; the returned execution is what gets persisted). There is deliberately NO `build_messages` hook — history policy belongs on the `ConversationProjector`.

**Only `before_tool_execution` has an exactly-once, paired guarantee**, and it is per DISPATCH ATTEMPT, not per call for all time — a crash during `prepare()` writes nothing, so the next drive fires it again over the original call. Every other hook may fire without its result being persisted; the visible case is `before_permission_check`, whose returned execution is discarded when a cancellation lands while `decide()` is in flight (the execution must stay PENDING for the wind-down, so there is nowhere to put it). `build_tool_list` is unchanged: the runner's method around it is now `async`, but the HOOK stays synchronous and still receives the post-adapter WIRE list, never `ToolSpec`s.

Tests go in `tests/agent/test_runner_middleware.py`. See `docs/agent/07-middleware.md` for the full hook catalogue. The doc embeds the mixin's full source — keep it in sync when the mixin changes.

### Add a plugin

Plugins are a CONTRIB concept — the core runner knows nothing about them. A plugin bundles a tool registry + system-prompt parts + middleware behind one object (`luca/agent/contrib/plugins/`). Write a plain class implementing any of `get_tool_registry(agent_session)` / `get_system_prompt_parts(agent_session)` / `get_middleware(agent_session)` — duck-typed via `hasattr` like middleware; `BasePlugin` is an optional base. Pass instances as `plugins=[...]` to `PluginAgentSessionRunner`, which composes them at construction: the directly-passed registry and every plugin registry become children of one `ProxyToolRegistry`, and each hook's result extends the matching list (after the directly-passed items, in plugin order) — pure construction-time sugar, equivalent to composing the same objects directly. `AgentSessionRunner.__eq__` compares that effective configuration, which is how the tests assert equivalence.

Plugin tests are scoped to `tests/agent/contrib/test_plugins.py` ONLY. `luca/agent/contrib/memory/` (scratchpad + todo tools bundled by `MemoryPlugin` in its own auto-allowing registry) is the reference plugin; docs in `docs/agent/09-plugins.md` and `docs/agent/contrib/plugins/`.

### Change the agent loop

The engine lives in `luca/agent/core/runner.py`.

`run()` / `start()` construct `AgentRun` handles over the single `_drive(streaming, token)` generator. Lazy runs pull the generator directly (`_pump`); eager runs drain it from a background task into a grow-only buffer (`_consume`).

The handle owns lifecycle plumbing: `_begin_run`'s one-engine-at-a-time guard, the per-run `CancellationToken`, suspend finalization in `__aexit__`, and `RunResult` construction via `_build_run_result`. There is no per-run context object — registries and tools receive the live session.

**Engine order — once per drive, then each loop iteration in sequence:**

-1. (drive start) `_recover_orphans`: any persisted `RUNNING` execution → `INTERRUPTED` (`after_tool_execution` runs, no re-dispatch), before the flush too.
0. Unconsumed `CancelRequested` → `_wind_down` (also handles the parked flush).
1. Undecided executions (`approval_status` None/PENDING) → `asyncio.gather` `_decide_one` over them; each races the WHOLE `_decide_with_middleware` step (hook → `tool_registry.decide()` → hook) against the token, and a lost race records nothing for that execution. Each surviving response updates `approval_status` + appends to the audit log; DENY → terminal `REJECTED` now (outcome pipeline + `ToolExecuted`).
2. Ready executions (`PENDING`+`ALLOWED`) → `_dispatch_batch` (sequential by choice — the state model is parallel-ready; a token already tripped when an execution's turn comes up stops the batch, leaving it and its successors untouched for the wind-down).
3. Any execution still `approval_status=PENDING` → cancel check, then park (`ApprovalRequired` last).
4. Step-limit / doom-loop checks → `build_tool_list()` (raced; a lost race skips the LLM call and returns to the loop top) → model call (reached only when every execution is terminal).
   - `hard_max_steps` reached → `_close_turn(ERRORED)` and return (no raise).
   - `soft_max_steps` reached, or a doom-loop-flagged execution exists → `tool_choice="none"`.
   - Race the cancellation token via `_race_cancellation` (grace window, hard-kill via `_kill`), wired to `RuntimeConfig`'s `timeout=`/`total_timeout=`.
   - `TimeoutError` → `TurnFinish(TIMED_OUT)` and re-raise; any other failure → `ERRORED` (status PENDING, retry-ready); cancel pending → wins (wind-down, normal return).
   - Recording the assistant message, creating its executions, and closing a final-answer bracket is **atomic** (no yield between).

**Per-step methods:**

| Method | Responsibility |
|--------|---------------|
| `build_system_message` | Resolve `system_prompt_parts` (callables get `(session_config, runtime_status)`) → sort by priority → assemble; blank → no system message |
| `build_messages` | Delegates to `conversation_projector.project()` — no middleware stage |
| `_record_assistant` | Converts message to parts via `adapter.message_to_parts` |
| `_create_executions` / `_birth_draft` | Set-oriented birth: gather `tool_registry.create_execution(session, deep-copied call)` per call (concurrent, each raced against the token INDIVIDUALLY — never the gather), then eager appends in call order — the runner re-stamps identity (`id`/`parent_id`/`created_at`/`ended_at`-if-terminal/`context_tokens`/`is_doom_loop_flagged`); a raising `create_execution` (or `None` registry) synthesizes the draft (`FAILED`/`NOT_FOUND`), a lost race synthesizes a PENDING one, isolated per call; terminal births run the outcome middleware pair |
| `_dispatch_one` / `_prepare_tool` / `_run_tool_body` | `before_tool_execution` (its `raw_tool_call` is the effective call) → `tool_registry.prepare(session, execution)` raced (grace 0) → on raise/non-callable: terminal with `phase="prepare"`, NO `RUNNING`, NO `ToolExecutionStarted` → on cancel: `CANCELLED` in place → else persist `RUNNING`+`started_at` (birth `tool_spec` stands) → `ToolExecutionStarted` → invoke the callable (the `ensure_future` INSIDE the failure handling) under token race + outside deadline (`tool_spec.timeout_in_ms`, else config) → `COMPLETED`/`FAILED`/`INTERRUPTED`/`TIMED_OUT`; a mid-body cancel persists `cancel_signalled_at` before the grace wait |
| `_finalize_outcome` / `_finalize_undispatched` | The shared outcome tail: (`before_tool_execution` for never-dispatched calls →) `after_tool_execution(execution, exception)` → persist → `ToolExecuted` built from the projector. Everything past the hook in `_dispatch_one` uses `_finalize_outcome`; using `_finalize_undispatched` there would fire the hook twice |
| `to_tool_execution_error` | PUBLIC override point: live exception → durable `ToolExecutionError` (type + message; pydantic errors under `details.errors`; `details.phase` is a FACT passed by the call site — `create_execution` / `prepare` / `execution` — not inferred from `started_at`) |
| `recalculate_context_tokens` | PUBLIC: re-derive `context_tokens` for every entry through `before_entry_written`. Nothing in the framework calls it |
| `_is_doom_loop(tc)` | Compares last `threshold-1` `ToolExecution`s in the open turn against the incoming call's `raw_tool_call` name + arguments |
| `_close_turn(outcome, error)` | The only `TurnFinish` writer |

All entry appends and entry-derived queries (open turn, pending/undecided/awaiting/ready/running executions, unconsumed cancel, doom-loop flag, derived status) are delegated to `SessionLedger` (`ledger.py`) — one append path so parent links, path, `updated_at`, and `tool_executions` indexing cannot drift. The ledger is also the only door onto the usage store (`record_usage`), the only path-replacement write (`prune`), and the only writer of `tool_specs`. Every `ToolExecution` persistence — creation AND every update — passes through `before_entry_written` (updates via the runner's `_persist_execution`, stored by `ledger.put_entry`).

**The tool-spec write doors.** `_store_tool_spec` hashes an execution's spec, files it under `spec_id()` and stamps `tool_spec_id`, and it runs on every door that puts a `ToolExecution` into the store: `append` (birth), `put_entry` (every later update), `transition_conversation` (a compaction plan carrying or creating one — over `updates`, `created` AND `closing`, not just the list the shipped caller happens to use), and `refresh_entry` (the derived-field door `recalculate_context_tokens` uses). `prune` is NOT one: it only ever writes a `PrunedEntry`. The id is recomputed on EVERY write and never short-circuited when one is already set — a spec can be replaced between writes (middleware may rewrite it) and a skipped recompute leaves a stale reference. A door that skips this is silent: the execution carries its spec in memory for the rest of the process and loses it on the first save, because the session serializer strips inline specs and would have no id to write in their place.

### Write tests

Tests are declarative: precondition → one action → postcondition. No logic, no helper functions in the test body. Never race two timed things.

**Precondition:** a known session — either an inline literal or one of the shared mid-state constants in `tests/agent/scenarios.py`:
- `GATED_SESSION`, `CLEARED_SESSION`, `UNDECIDED_SESSION`, `STALE_RUNNING_SESSION`, `CANCEL_PARKED_SESSION`, `POST_FAILURE_SESSION`, `RUNNING_ORPHAN_SESSION`
- Always `model_copy(deep=True)` before use.

**Literal factories (`scenarios.py`).** `ToolSpec` requires `description` + `input_schema`, and a session literal has to carry `tool_specs` plus a `tool_spec_id` on every execution — none of which any test is about. Use `spec(name, **over)` for a complete `ToolSpec`, `make_session(**fields)` for an `AgentSession` literal (it hashes each execution's spec, stamps the id and fills the store, exactly like the ledger's write doors; explicitly-passed `tool_specs` rows and already-stamped ids are left alone, so a dangling reference can still be authored on purpose), and the `ADD_SPEC` / `MULTIPLY_SPEC` / `READ_FILE_SPEC` / … constants when a precondition must be byte-identical to what the registry double produces. Module-level factories like these are not "logic in the test body" — they are what keeps the precondition declarative.

Load the session cold into a fresh runner to exercise the persisted-resume path.

**Consuming runs:**
- Drain a lazy run for event-list asserts: `async with runner.run() as run: events = [e async for e in run]`
- Await for `RunResult` asserts: `await runner.run()`

**Assertions:** the project-wide full-object rule (see `AGENTS.md`), applied here — both `runner.session == AgentSession(...)` (status included) and the complete `events == [...]` list.

**Providers:** use `FauxProvider` via `provider=`. `faux_hang()` scripts a hang for cancellation/timeout scenarios.

**Runner:** drive scenarios through `DeterministicRunner` (`tests/agent/scenarios.py`). Its `ids`/`now` overrides span `post_message`, every run, and `cancel()` (the `CancelRequested` entry and the closing `TurnFinish` consume ids too), including resume across an approval pause.

**Registries:** core tests must NOT import contrib — wire tools through `FakeToolRegistry` (in `scenarios.py`), the core-only deterministic registry double: `get_tools` answering `ToolSpec`s, a preflight-faithful `create_execution`, a resolve-validate-then-close-over-the-body `prepare` (recording the names it resolved in `prepared`), and a scripted `decide` — with no `decisions` script it ALLOWs everything with frozen `created_at`; with one, each decide() pops the next decision (unresolved-path order) and its `seen` list records which execution snapshots the runner asked about. The tool doubles are built on `FakeTool`, a core-only base — deliberately NOT contrib's `Tool`, since building the doubles out of core types is what PROVES the core needs no Python tool class.

**Test files and their scope:**

| File | Covers |
|------|--------|
| `tests/agent/test_runner.py` | Turns, streaming, birth failure modes, event snapshots, serialization round-trip |
| `tests/agent/test_runner_tool_output.py` | Fully-inlined decision-support stories: the full session + event shape of one tool round per outcome — do NOT factor helpers out of it |
| `tests/agent/test_runner_lifecycle.py` | `AgentRun` handle: lazy/eager, suspend, `RunResult`, `on_event` |
| `tests/agent/test_runner_approvals.py` | Gate / re-ask / allowed-sibling-dispatch / cold-resume / decide-failure scenarios |
| `tests/agent/test_runner_cancellation.py` | Cancel / wind-down / flush / grace / `cancel_signalled_at`, plus the four registry-phase races (a hung `get_tools` / `create_execution` / `decide` / `prepare` each unblocked by `cancel()`), the CANCELLED-not-FAILED birth rule, and the identical durable shape of a dispatch-path vs wind-down cancellation |
| `tests/agent/test_runner_failures.py` | The prepare/dispatch outcome table (every `prepare()` failure mode, the non-callable and non-awaitable guards, `started_at`/`dispatched`/`details["phase"]`), tool deadlines and their scope, crash recovery (orphaned RUNNING), LLM failure closes, `post_message` matrix |
| `tests/agent/test_runner_projector.py` | The runner ↔ `ConversationProjector` seam: wire history, event/wire agreement, equality |
| `tests/agent/test_runner_context.py` | The runner ↔ `ContextManager` seam: context stamping, middleware final say, processed tool output (session/event/wire agreement), prune-machinery composition, and `recalculate_context_tokens()` (every entry, archived and off-path included; construction changes nothing) |
| `tests/agent/test_context_manager.py` | Default `ContextManager`: per-type context ownership, prune templates + refusals, identity tool-output, subclass overrides, model-aware counting, and the non-membership of an entry during an append |
| `tests/agent/test_runner_system_prompt.py` | `system_prompt_parts` forms (str / dict / part / callable) + assembler (callable parts receive `(session_config, runtime_status)`) |
| `tests/agent/test_runner_limits.py` | Hard/soft `max_steps`, doom-loop flagging, `tool_choice` restriction |
| `tests/agent/test_models.py` | The data model as pure Pydantic: entry shapes and defaults, `ToolSpec.spec_id()` (stability, key-order independence, distinct ids for distinct content), tool-spec normalization end to end (one row per distinct spec, dump strips / load restores by reference, standalone execution dumps stay inline, both load guards raise) |
| `tests/agent/test_ledger.py` | Entry-derived query matrix (status × approval subsets), the `record_usage` / `put_entry` / `transition_conversation` / `refresh_entry` doors, tool-spec filing at every door (one row per distinct spec, recompute-always, all three `transition_conversation` lists), `open_compaction_entry`, the derive-status skip matrix |
| `tests/agent/test_compaction.py` | The compaction CONTRACT as a unit: `validate_plan` (every rejection), `has_content`, `check_snapshot`, the plan value objects — no runner, no ledger, nothing async |
| `tests/agent/test_runner_compaction.py` | The drive: brackets, events, the transition, failures by source, cancels, resumes (G6), statuses, `RunResult` |
| `tests/agent/test_projection.py` | `ConversationProjector`: every entry type, every terminal tool status, fail-loud rules, subclass override points |
| `tests/agent/test_adapter.py` | Inbound message parts + `tool_spec_to_luca_tool` (the `input_schema` → `parameters` pass-through) |
| `tests/agent/test_utils.py` | `pretty_print`: whole-transcript assertions per session shape (answered turn, tool tree, failure, open turn, compaction/pruning, clipping) |
| `tests/agent/test_runner_middleware.py` | Middleware hook dispatch (incl. the tool pair across every outcome) |
| `tests/agent/contrib/test_tools.py` | Self-scoped contrib tests: the `Tool` base contract (spec stamping incl. `input_schema` and `timeout_in_ms`, session + token pass-through) and the working `tool()` / `tool_class()` factories |
| `tests/agent/contrib/test_simple_tool_registry.py` | Self-scoped contrib tests: birth drafts per preflight outcome, decide delegation, `prepare` returning a callable WITHOUT running the body and its two raise paths, `ProxyToolRegistry` routing/nesting and cache-independent `decide`/`prepare` on a never-warmed route — no runner |
| `tests/agent/contrib/test_plugins.py` | Self-scoped contrib tests: `PluginAgentSessionRunner` composition (one proxy, parts/middleware flattening, equality with a directly-configured runner) |
| `tests/agent/contrib/test_resource_permissions.py` | Self-scoped contrib tests: `PermissionStrategy` decide / apply_answer / pending_requests / grant + the tool mixin — no runner, no session |
| `tests/agent/contrib/shell/` | Self-scoped contrib tests: one file per shell tool (`tools/test_<name>.py`) + `test_plugin.py` (`ShellAccessPlugin` wiring, seeded rules, decide/pending flows) — no runner |
| `tests/agent/contrib/test_memory.py` | Self-scoped contrib tests: `MemoryPlugin` surface + scratchpad / todo-list behavior — no runner |
| `tests/agent/contrib/test_compaction_policy.py` | Self-scoped contrib tests: `SummarizingCompactionPolicy` — the context gauge, the split strategies, and the `CompactionPlan` it returns (via `FauxProvider`); no runner |
| `tests/agent/contrib/tui/` | Self-scoped contrib tests: pure modules (`test_approvals.py`, `test_render.py`, `test_sessions.py`, `test_wiring.py`, `test_cli.py`, `test_config.py`, `test_context_bar.py`) + headless Pilot tests driving `AgentApp` with a scripted `FauxProvider` (`test_app*.py`); the directory skips itself when textual is missing |
| `tests/agent/contrib/mcp/` | Self-scoped contrib tests: config validation, result/schema mapping (pure), the per-call registry against a live stdio FastMCP fixture, permissions, and OAuth token-store/redirect units; skips when the `mcp` extra is absent |

## When in doubt

| Question | Go to |
|----------|-------|
| Tool-execution lifecycle / approval state / errors / events | `luca/agent/core/models.py` + design principles 4 and 10 above |
| LLM projection / tool-output derivation | `luca/agent/core/projection.py` + design principles 5 and 6 above |
| Agent data model / session invariants | `luca/agent/core/models.py` + design principles 1–3 above |
| The tool-registry contract (incl. the registry-author rules) | `luca/agent/core/tool_registry.py` + `luca/agent/contrib/simple_tool_registry/` |
| Writing a tool in Python (`Tool`, the factories, the body's cancellation contract) | `luca/agent/contrib/tools.py` + `docs/agent/contrib/tools/README.md` |
| Tool-spec storage (`spec_id`, the write doors, the load guards) | `luca/agent/core/models.py` + `luca/agent/core/ledger.py` + the "Tool identity" section above |
| Run lifecycle (run/start, AgentRun, cancel, timeouts, outcomes) | `luca/agent/core/runner.py` + design principles 9, 11, and 12 above |
| Compaction (policy contract, the plan, the transition, the guarantees) | `luca/agent/core/compaction.py` + the Compaction section above + `docs/agent/12-compaction.md` |
| Where does this responsibility belong | `runner.py`, `projection.py`, `adapter.py`, and their tests under `tests/agent/` |
