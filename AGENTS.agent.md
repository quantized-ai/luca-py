Guidance for the `luca.agent` layer. Read this file whenever you're working in `luca/agent/` or `tests/agent/`.

## What this layer is

`luca.agent` is the primary product: a full-featured, durable agent framework. Its central artifact is a single serializable `AgentSession` that captures a complete conversation history — messages, tool executions, reasoning, turn boundaries, compaction — and can be reloaded to resume exactly where it stopped.

## Goals

- One canonical, JSON-serializable `AgentSession` that round-trips through `model_dump_json` / `model_validate_json` losslessly.
- A flat, append-only entry store (`AgentSession.entries`) addressed by id, with `Conversation.nodes` (an ordered list of ids) as the traversal path. Forking is cheap and explicit.
- A resumable async agent loop exposed as `runner.run()` (lazy) and `runner.start()` (eager), both returning an `AgentRun` handle. The engine projects the active conversation to LLM messages, calls the model, records the assistant turn, executes tool calls, and loops — bracketed by `TurnStart` / `TurnFinish(outcome)`. One logical turn can span multiple runs.
- The whole tool lifecycle delegated to a `ToolRegistry` — the core has no built-in tool resolution or approval engine; permission policies are contrib (`contrib/simple_tool_registry`).
- Durable cancellation via `runner.cancel()` / `run.cancel()`, recorded as a `CancelRequested` entry and wound down at engine step boundaries.
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
│   ├── memory/
│   │   ├── __init__.py  # package surface: MemoryPlugin + the scratchpad / todo tools
│   │   └── plugin.py    # scratchpad + todo-list tools, MemoryPlugin
│   ├── compaction/
│   │   ├── __init__.py  # package surface: CompactionStrategy, RecentTurnsStrategy, Compactor
│   │   ├── strategy.py  # split policies (what to keep verbatim) — concrete base, no ABC
│   │   └── compactor.py # the gauge + summarize + build_compacted_session (new session, source untouched)
│   ├── resource_permissions/
│   │   ├── __init__.py  # package surface: PermissionStrategy, rules, answers, the mixin
│   │   ├── strategy.py  # PermissionMode, ToolRule/ToolKindRule, ApprovalAnswer, PermissionStrategy
│   │   └── mixin.py     # ResourcePermission, AnswerOption, PermissionRequest, ResourcePermissionToolMixin
│   ├── shell/           # the 7 shell tools + ShellAccessPlugin — see AGENTS.md there
│   └── tui/             # the Textual terminal UI (AgentApp + wiring + approval modal);
│       │                #   Textual-free logic in approvals.py / render.py / sessions.py / wiring.py
│       │                #   — needs the `tui` dependency group (a uv default group)
└── core/
    ├── __init__.py      # external surface: AgentSessionRunner, ToolRegistry, Tool,
    │                    #   ToolContext, SystemPromptAssembler,
    │                    #   all entry types, exceptions
    ├── models.py        # AgentSession (incl. the usages store), all entry classes
    │                    #   (incl. CancelRequested, PrunedEntry),
    │                    #   ExecutionStatus/ApprovalStatus/ToolExecutionError,
    │                    #   TurnOutcome, RuntimeConfig, SessionConfig, Usage,
    │                    #   SessionRuntimeStatus, ConversationStatus — pure Pydantic v2
    ├── tool_registry.py # ToolRegistry — the 4-method contract the runner drives tools through
    ├── tools.py         # Tool base class: name/description/Args ClassVars,
    │                    #   get_tool_spec()/_execute/execute (keyword-only cancellation_token),
    │                    #   timeout_in_ms ClassVar (snapshotted into ToolSpec)
    ├── context.py       # ToolContext + CancellationToken (runtime-only; never persisted)
    ├── context_manager.py # ContextManager — context-accounting strategy: per-entry
    │                    #   context_tokens estimation, tool-output processing,
    │                    #   PrunedEntry templates (concrete class; runner default)
    ├── exceptions.py    # AgentError, CancelledError, AlreadyCancellingError,
    │                    #   ToolNotFound, InvalidToolArguments, ProjectionError
    ├── events.py        # AgentEvent union (block-level + streaming-delta + ApprovalRequired);
    │                    #   tool events carry deep ToolExecution snapshots
    ├── projection.py    # ConversationProjector — the PUBLIC conversation → LLM-message
    │                    #   strategy (subclass to customize history/tool-output policy)
    ├── adapter.py       # message_to_parts() (inbound response conversion) +
    │                    #   tool_to_luca_tool() (tool-definition conversion)
    ├── middleware.py    # AgentMiddlewareMixin — the 10 duck-typed middleware hooks
    ├── ledger.py        # SessionLedger — the single append/read door onto the entry log
    ├── system_prompt.py # coerce_system_prompt_part, SystemPromptAssembler,
    │                    #   DefaultSystemPromptAssembler, part-input type aliases
    └── runner.py        # AgentSessionRunner, AgentRun handle, RunResult

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

Exception: `AgentSession.session_runtime_status` is a plain `@property` (not a Pydantic field) that recomputes `SessionRuntimeStatus` from the live entries on every access. It is never serialized and never trusted from disk.

### 2. Storage and traversal are separate

`entries: dict[str, AnyEntry]` is the durable, append-only, uniformly-addressable node space.
`Conversation.nodes: list[str]` is the path — an ordered list of entry ids. Walk the path; resolve ids in the store.
`parent_id` is a recovery backstop and is **never traversed**.

### 3. Messages are entries

`UserMessage` and `AssistantMessage` live in `entries` alongside `ToolExecution`, `TurnStart`, `TurnFinish`, and `CompactionEntry`. One `Entry` base class, one `type` discriminator field, one `AnyEntry` discriminated union.

### 4. A tool call is two things

- The request block: a `ToolCall` object inside `AssistantMessage.parts`.
- A separate, mutable `ToolExecution` entry — the durable source of truth about that call's whole lifecycle — correlated by `tool_call_id`.

`ToolExecution` is the **only** mutable entry type.

`tool_executions: dict[str, list[str]]` is a denormalized index from `tool_call_id` → execution-entry ids.

Each `ToolExecution` carries three orthogonal facts plus its provenance:
- `status: ExecutionStatus` — the framework's execution lifecycle and ONLY that: `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `NOT_FOUND`, `INVALID`, `REJECTED`, `CANCELLED`, `INTERRUPTED`, `TIMED_OUT`. `COMPLETED` means "the framework received a result", not "the tool succeeded".
- `approval_status: ApprovalStatus | None` — the CURRENT approval state (`None` = the policy never processed it; `PENDING`/`ALLOWED`/`REJECTED`). Always read approval from this field; `approval_decisions` is the append-only audit log of policy responses (only PENDING may repeat), never the source of current state.
- The outcome payload: exactly one of `result: ExecutionResult` (the body returned; `result.is_error` is the tool's OWN verdict — an `is_error=True` result is still `COMPLETED`) or `error: ToolExecutionError` (structured `error_type`/`error_message`/`details`, populated for `FAILED`/`NOT_FOUND`/`INVALID`) — or neither, for the status-only terminals.
- `raw_tool_call: ToolCall` — the (possibly middleware-effective) request; makes the execution self-contained. `tool_spec: ToolSpec | None` — the resolved tool's identity snapshot (`name`, `description`, `metadata`, `tool_kind`, `namespace`, `version`, `timeout_in_ms` — NO arguments; `None` when the tool never resolved). `extras` — a free-form dict written by registries/middleware, stored verbatim, never interpreted by the core (`SimpleToolRegistry` stores the tool's approval context under `extras["approval_context"]`).
- Lifecycle timestamps: `started_at` (set iff the body dispatched), `ended_at` (every terminal transition), `cancel_signalled_at` (run cancellation only — a deadline never sets it). `updated_at` is ledger bookkeeping, not timing.

These combinations are framework conventions, not Pydantic validators — middleware is trusted and may author unusual state; the application owns the consequences.

### 5. The wire payload is derived, never stored

The **`ConversationProjector`** (`projection.py`) is the public strategy that recomputes the LLM message list from the path on every call: it drops `TurnStart`/`CancelRequested`, projects a CANCELLED `TurnFinish` as the synthetic `[Request interrupted by user]` marker, unwraps messages into client content blocks, projects each terminal `ToolExecution` to its correlated `ToolMessage` (COMPLETED → its result verbatim; every other terminal → derived error text with `is_error=True`), and renders `CompactionEntry` as a synthetic user message. It is a concrete class — pass a subclass as `conversation_projector=` to change any policy (history shaping, redaction, tool-output wording); ALL default derived wording lives on the class. The same `project_tool_execution` output feeds the `ToolExecuted` event's `result_text`/`is_error`, so event and wire never disagree. There is no projection middleware; `before_llm_call` stays as the downstream last-mile hook.

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
- **`ApprovalRequired`** fires as the last event before a gate, carrying the awaiting `ToolExecution` entries (equivalent to `runner.pending_approvals()`).

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

A logical turn spans one `TurnStart`/`TurnFinish` bracket even across an approval pause. A `TurnStart` with no later `TurnFinish` means resume, not re-open.

`post_message` requires a closed bracket and IDLE or PENDING status. It always rejects CANCELLING and AWAITING_APPROVAL.

### 12. Cancel is a pure signal

`cancel(outcome, error)` (callable in any state):
- Appends a durable `CancelRequested` to the open turn.
- Trips the live run's cancellation token.
- Sets status to CANCELLING.
- Returns immediately — performs NO bookkeeping.

All wind-down happens at the engine's step boundaries and turn-close sites:
- Still-PENDING executions → stamped `cancel_signalled_at`, then `CANCELLED` (approval state untouched; a DENIED call was already terminal `REJECTED` at decision time).
- In-flight executions → persisted with `cancel_signalled_at` FIRST, then the grace period: a within-grace return is `COMPLETED` with its real result (keeping the stamp), a raise is `FAILED`, expiry is `INTERRUPTED`.
- Already-terminal executions are untouched.
- Closes the turn with `TurnFinish(outcome)`.

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

`ContextManager.process_tool_output()` transforms a returned `ExecutionResult` before the terminal execution is constructed (identity by default) — the durable session, the `ToolExecuted` event, and the wire all see the processed output.

**Pruning** replaces an entry's contribution to the path without touching the original: `ContextManager.prune_entry()` builds a `PrunedEntry` TEMPLATE (placeholder identity; v1 supports terminal tool executions only, replacement text `"[tool output has been pruned to reduce context]"`), and `SessionLedger.prune(original_id, build)` stamps identity (the original's `parent_id`), verifies the referent/type/terminality invariants, and swaps the node id in place. The original stays in `entries` and in the `tool_executions` index. `ConversationProjector.project_pruned` resolves the referent and re-emits the replacement content under the original's role and `tool_call_id` (a missing referent, type mismatch, or unprojectable source raises). The runner deliberately exposes NO public prune/context-total methods yet.

### The tool registry

The runner is constructed with one `ToolRegistry` (`luca/agent/core/tool_registry.py`; `None` = toolless agent). The core touches tools through exactly four methods:

```python
class ToolRegistry:
    def get_tools(self, agent_session) -> list[Tool]: ...
    async def create_execution(self, call, context) -> ToolExecution: ...
    async def decide(self, tool_execution, context) -> ApprovalDecision: ...
    async def execute(self, tool_execution, context, *, cancellation_token) -> ExecutionResult: ...
```

- **`get_tools` is dynamic** — queried fresh per LLM call (the result may vary with session state); never a lifecycle hook.
- **`create_execution` returns a birth DRAFT** with placeholder identity (`id=""`, `created_at=0`). The registry owns the call-scoped facts: `raw_tool_call`, `tool_spec` (incl. `timeout_in_ms`; `None` if unresolved), the birth `status` (PENDING, or terminal-at-birth NOT_FOUND/INVALID/FAILED), the `error` for a terminal birth (the registry authors it), and `extras`. The RUNNER stamps `id`/`parent_id`/`created_at`/`ended_at`-if-terminal/`context_tokens`/`is_doom_loop_flagged`, so determinism principle 8 holds. If `create_execution` raises (or the registry is `None`), the runner synthesizes the draft itself (FAILED / NOT_FOUND) — failures stay isolated per call.
- **`decide`** returns ALLOW / DENY / PENDING; exceptions propagate and abort the run (the executions stay unresolved; the next `run()` asks again), so implementations must be idempotent queries of their own state.
- **`execute`** raises map to statuses at the runner: `ToolNotFound` → NOT_FOUND, `InvalidToolArguments`/pydantic `ValidationError` → INVALID, anything else → FAILED; a returned `ExecutionResult` → COMPLETED (after `ContextManager.process_tool_output`).

There is **no global permission gate anywhere**: each registry answers `decide()` for its own tools. Cross-cutting approval is an application composition pattern (share one strategy across registries), never a framework or plugin API concern.

The engine has exactly **one** `decide()` call site — the top of its loop. Any open-turn execution that is undecided (`approval_status` is `None` or `PENDING`) is handed to the registry. Sibling undecided executions are decided concurrently via `asyncio.gather`. Each response both updates `approval_status` directly and appends to the `approval_decisions` audit log; a DENY is terminal on the spot (`status=REJECTED`, `ended_at` stamped, outcome middleware runs, `ToolExecuted(REJECTED)` emitted).

A PENDING decision defers only THAT execution: every ALLOWED sibling proceeds to dispatch, and the runner parks (status → `AWAITING_APPROVAL`, `ApprovalRequired` as the final event) only after all currently runnable work has advanced. The model is never called again until every tool call in the assistant response has a terminal execution and a correlated tool output. Re-entering `run()` does not raise — it simply asks the registry again. `runner.pending_approvals()` returns the awaiting `ToolExecution` objects.

Dispatch (per ready `PENDING`+`ALLOWED` execution): `before_tool_execution` middleware (its returned `raw_tool_call` is the effective call) → persist `RUNNING` + `started_at` (the birth `tool_spec` stands — no dispatch-time re-snapshot) → emit `ToolExecutionStarted` → `registry.execute(...)` under the cancellation race + deadline (deadline = `tool_spec.timeout_in_ms`, else `RuntimeConfig.tool_execution_timeout_in_ms`). A returned result is `COMPLETED` (whatever `result.is_error` says); a raise maps per the contract above with a structured `ToolExecutionError` built by the overridable `runner.to_tool_execution_error(execution, exception)`; deadline expiry is `TIMED_OUT`; grace expiry is `INTERRUPTED`. `after_tool_execution(execution, exception)` observes EVERY outcome before the final persist (registry-authored terminal births carry no live exception). Because resolution/validation live inside `registry.execute`, a dispatch-time NOT_FOUND/INVALID lands AFTER `started_at` is set, with `ToolExecutionStarted` emitted.

**Invariant: every tool call produces exactly one tool output, always.**

A call that is terminal at birth is persisted with a structured `error` and never reaches `decide()` (`approval_status` stays `None`); it still passes through the tool middleware pair.

The batteries-included registries live in `luca/agent/contrib/simple_tool_registry/`: `SimpleToolRegistry(tools, permission_policy)` reproduces the classic preflight (resolve → validate → duck-typed `get_approval_context`, stored under `extras["approval_context"]`) and delegates `decide()` to a `PermissionPolicy` (one-arg async `decide(tool_execution)`; `YoloPermissionPolicy` allows everything); `ProxyToolRegistry(*registries)` composes registries (`get_tools` recomputes and caches a `{name → child}` route, duplicate names raise; the other methods route from the cache and degrade to NOT_FOUND on a miss). Everything richer — modes, rules, resource globs, answer-decoupled interactive approval, the `ResourcePermissionToolMixin` — lives in `luca/agent/contrib/resource_permissions/` and is driven interactively by `main.py`.

### Tool identity

`Tool` (in `luca/agent/core/tools.py`) is the EXECUTION contract only. It declares `tool_kind`, `namespace`, `version`, and `timeout_in_ms` as `ClassVar`s; `get_tool_spec()` stamps them into a `ToolSpec` snapshot — the tool's identity only; invocation arguments live on `ToolExecution.raw_tool_call`. `tool_kind` defaults to `OTHER`; the network-egress kind is `WEB_FETCH`.

There is no `get_approval_context` on the base class — it is a duck-typed convention read by `SimpleToolRegistry` (`async get_approval_context(args, context) -> dict`, receiving the **validated** args; `resource_permissions.ResourcePermissionToolMixin` provides it). The core never mentions it.

### Timeouts and step limits

All config rides on `SessionConfig.runtime_config` (a `RuntimeConfig`), which persists with the session. The runner reads it live — not from its constructor.

**Timeout fields:**

| Field | Effect |
|-------|--------|
| `builtin_client_completion_timeout_in_ms` | Client per-phase `timeout=`. INERT when the runner is built with a provider instance. |
| `client_completion_timeout_in_ms` | Client wall-clock `total_timeout=` (async helpers only). |
| `tool_execution_timeout_in_ms` | Outside tool deadline; beaten by the birth `ToolSpec.timeout_in_ms` (stamped from the tool's `timeout_in_ms` ClassVar). Expiry → `TIMED_OUT`, resultless. |
| `*_cancellation_grace_period` | Grace window for cancel races. 0 = immediate hard cancel; a tool returning within grace records its real result. |

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

Subclass `Tool` (in your application, or a contrib package):

- Set `name`, `description`, and `Args` (a Pydantic model — becomes the wire JSON schema) as class vars.
- Set `tool_kind` (a `ToolKind`; carried on the `ToolSpec` snapshot), and optionally `timeout_in_ms` (also snapshotted).
- Override `async _execute(args, context, *, cancellation_token) -> str` for simple tools, or `async execute(args, context, *, cancellation_token) -> ExecutionResult` for rich output (`is_error`, `metadata`, multi-block). An `is_error=True` result still records `COMPLETED` — `is_error` is the tool's own verdict, not a lifecycle fact.
- Both receive a `ToolContext` (session id, model) plus the keyword-only `CancellationToken`.
- `SimpleToolRegistry` validates LLM-produced arguments through `Args` at birth (and again at dispatch). Malformed args become a terminal `INVALID` execution with a structured `ToolExecutionError` and never reach `decide()`.
- Timing is recorded by the runner on the execution (`started_at`/`ended_at`), never on the result.
- Define `async get_approval_context(args, context) -> dict` (the duck-typed convention) to describe the call for `SimpleToolRegistry`'s permission strategy.

Wrap **instances** in a registry and pass that to the runner:

```python
registry = SimpleToolRegistry(tools=[MyTool()], permission_policy=YoloPermissionPolicy())
runner = AgentSessionRunner(session, tool_registry=registry)
```

The runner projects `get_tools()`'s result to wire tools via `adapter.tool_to_luca_tool`; the registry dispatches by name.

### Add middleware

Write a plain Python class that implements any of the 10 hook methods defined in `luca/agent/core/middleware.py` (`AgentMiddlewareMixin` — its hooks are identity pass-throughs, so subclassing is safe for partial overrides, but plain classes are the recommended style; the runner dispatches via `hasattr`). Pass instances as `middleware=[mw1, mw2]` to `AgentSessionRunner`. Hooks run in list order; there is no reverse ordering. The tool pair works on the whole execution: `before_tool_execution(execution) -> execution` (pre-dispatch for allowed calls — its `raw_tool_call` is the effective call — and once for terminal-at-birth / rejected / cancelled-before-dispatch calls) and `after_tool_execution(execution, exception=None) -> execution` (every outcome; the returned execution is what gets persisted). There is deliberately NO `build_messages` hook — history policy belongs on the `ConversationProjector`.

Tests go in `tests/agent/test_runner_middleware.py`. See `docs/agent/07-middleware.md` for the full hook catalogue. The doc embeds the mixin's full source — keep it in sync when the mixin changes.

### Add a plugin

Plugins are a CONTRIB concept — the core runner knows nothing about them. A plugin bundles a tool registry + system-prompt parts + middleware behind one object (`luca/agent/contrib/plugins/`). Write a plain class implementing any of `get_tool_registry(agent_session)` / `get_system_prompt_parts(agent_session)` / `get_middleware(agent_session)` — duck-typed via `hasattr` like middleware; `BasePlugin` is an optional base. Pass instances as `plugins=[...]` to `PluginAgentSessionRunner`, which composes them at construction: the directly-passed registry and every plugin registry become children of one `ProxyToolRegistry`, and each hook's result extends the matching list (after the directly-passed items, in plugin order) — pure construction-time sugar, equivalent to composing the same objects directly. `AgentSessionRunner.__eq__` compares that effective configuration, which is how the tests assert equivalence.

Plugin tests are scoped to `tests/agent/contrib/test_plugins.py` ONLY. `luca/agent/contrib/memory/` (scratchpad + todo tools bundled by `MemoryPlugin` in its own auto-allowing registry) is the reference plugin; docs in `docs/agent/09-plugins.md` and `docs/agent/contrib/plugins/`.

### Change the agent loop

The engine lives in `luca/agent/core/runner.py`.

`run()` / `start()` construct `AgentRun` handles over the single `_drive(streaming, context)` generator. Lazy runs pull the generator directly (`_pump`); eager runs drain it from a background task into a grow-only buffer (`_consume`).

The handle owns lifecycle plumbing: `_begin_run`'s one-engine-at-a-time guard, per-run `CancellationToken`/`ToolContext`, suspend finalization in `__aexit__`, and `RunResult` construction via `_build_run_result`.

**Engine order — once per drive, then each loop iteration in sequence:**

-1. (drive start) `_recover_orphans`: any persisted `RUNNING` execution → `INTERRUPTED` (`after_tool_execution` runs, no re-dispatch), before the flush too.
0. Unconsumed `CancelRequested` → `_wind_down` (also handles the parked flush).
1. Undecided executions (`approval_status` None/PENDING) → `asyncio.gather` `tool_registry.decide()` over them; each response updates `approval_status` + appends to the audit log; DENY → terminal `REJECTED` now (outcome pipeline + `ToolExecuted`).
2. Ready executions (`PENDING`+`ALLOWED`) → `_dispatch_batch` (sequential by choice — the state model is parallel-ready; a tripped token skips the rest).
3. Any execution still `approval_status=PENDING` → cancel check, then park (`ApprovalRequired` last).
4. Step-limit / doom-loop checks → model call (reached only when every execution is terminal).
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
| `_create_executions` / `_birth_draft` | Set-oriented birth: gather `tool_registry.create_execution(deep-copied call, ctx)` per call (concurrent), then eager appends in call order — the runner re-stamps identity (`id`/`parent_id`/`created_at`/`ended_at`-if-terminal/`context_tokens`/`is_doom_loop_flagged`); a raising `create_execution` (or `None` registry) synthesizes the draft (`FAILED`/`NOT_FOUND`), isolated per call; terminal births run the outcome middleware pair |
| `_dispatch_one` / `_run_tool_body` | `before_tool_execution` (its `raw_tool_call` is the effective call) → persist `RUNNING`+`started_at` (birth `tool_spec` stands) → `ToolExecutionStarted` → `tool_registry.execute(..., cancellation_token=)` under token race + outside deadline (`tool_spec.timeout_in_ms`, else config) → `COMPLETED`/`NOT_FOUND`/`INVALID`/`FAILED`/`INTERRUPTED`/`TIMED_OUT`; a mid-body cancel persists `cancel_signalled_at` before the grace wait |
| `_finalize_outcome` / `_finalize_undispatched` | The shared outcome tail: (`before_tool_execution` for never-dispatched calls →) `after_tool_execution(execution, exception)` → persist → `ToolExecuted` built from the projector |
| `to_tool_execution_error` | PUBLIC override point: live exception → durable `ToolExecutionError` (type + message; pydantic errors under `details.errors`; `details.phase` from `started_at`) |
| `_is_doom_loop(tc)` | Compares last `threshold-1` `ToolExecution`s in the open turn against the incoming call's `raw_tool_call` name + arguments |
| `_close_turn(outcome, error)` | The only `TurnFinish` writer |

All entry appends and entry-derived queries (open turn, pending/undecided/awaiting/ready/running executions, unconsumed cancel, doom-loop flag, derived status) are delegated to `SessionLedger` (`ledger.py`) — one append path so parent links, path, `updated_at`, and `tool_executions` indexing cannot drift. The ledger is also the only door onto the usage store (`record_usage`) and the only path-replacement write (`prune`). Every `ToolExecution` persistence — creation AND every update — passes through `before_entry_written` (updates via the runner's `_persist_execution`, stored by `ledger.put_execution`).

### Write tests

Tests are declarative: precondition → one action → postcondition. No logic, no helper functions in the test body. Never race two timed things.

**Precondition:** a known session — either an inline literal or one of the shared mid-state constants in `tests/agent/scenarios.py`:
- `GATED_SESSION`, `CLEARED_SESSION`, `UNDECIDED_SESSION`, `STALE_RUNNING_SESSION`, `CANCEL_PARKED_SESSION`, `POST_FAILURE_SESSION`, `RUNNING_ORPHAN_SESSION`
- Always `model_copy(deep=True)` before use.

Load the session cold into a fresh runner to exercise the persisted-resume path.

**Consuming runs:**
- Drain a lazy run for event-list asserts: `async with runner.run() as run: events = [e async for e in run]`
- Await for `RunResult` asserts: `await runner.run()`

**Assertions:** the project-wide full-object rule (see `AGENTS.md`), applied here — both `runner.session == AgentSession(...)` (status included) and the complete `events == [...]` list.

**Providers:** use `FauxProvider` via `provider=`. `faux_hang()` scripts a hang for cancellation/timeout scenarios.

**Runner:** drive scenarios through `DeterministicRunner` (`tests/agent/scenarios.py`). Its `ids`/`now` overrides span `post_message`, every run, and `cancel()` (the `CancelRequested` entry and the closing `TurnFinish` consume ids too), including resume across an approval pause.

**Registries:** core tests must NOT import contrib — wire tools through `FakeToolRegistry` (in `scenarios.py`), the core-only deterministic registry double: static `get_tools`, a preflight-faithful `create_execution`, a resolve-validate-invoke `execute`, and a scripted `decide` — with no `decisions` script it ALLOWs everything with frozen `created_at`; with one, each decide() pops the next decision (unresolved-path order) and its `seen` list records which execution snapshots the runner asked about.

**Test files and their scope:**

| File | Covers |
|------|--------|
| `tests/agent/test_runner.py` | Turns, streaming, birth failure modes, event snapshots, serialization round-trip |
| `tests/agent/test_runner_tool_output.py` | Fully-inlined decision-support stories: the full session + event shape of one tool round per outcome — do NOT factor helpers out of it |
| `tests/agent/test_runner_lifecycle.py` | `AgentRun` handle: lazy/eager, suspend, `RunResult`, `on_event` |
| `tests/agent/test_runner_approvals.py` | Gate / re-ask / allowed-sibling-dispatch / cold-resume / decide-failure scenarios |
| `tests/agent/test_runner_cancellation.py` | Cancel / wind-down / flush / grace / `cancel_signalled_at` |
| `tests/agent/test_runner_failures.py` | Tool deadlines, crash recovery (orphaned RUNNING), LLM failure closes, `post_message` matrix |
| `tests/agent/test_runner_projector.py` | The runner ↔ `ConversationProjector` seam: wire history, event/wire agreement, equality |
| `tests/agent/test_runner_context.py` | The runner ↔ `ContextManager` seam: context stamping, middleware final say, processed tool output (session/event/wire agreement), prune-machinery composition |
| `tests/agent/test_context_manager.py` | Default `ContextManager`: per-type context ownership, prune templates + refusals, identity tool-output, subclass overrides |
| `tests/agent/test_runner_system_prompt.py` | `system_prompt_parts` forms (str / dict / part / callable) + assembler (callable parts receive `(session_config, runtime_status)`) |
| `tests/agent/test_runner_limits.py` | Hard/soft `max_steps`, doom-loop flagging, `tool_choice` restriction |
| `tests/agent/test_ledger.py` | Entry-derived query matrix (status × approval subsets), the `record_usage` door, the `prune` door |
| `tests/agent/test_projection.py` | `ConversationProjector`: every entry type, every terminal tool status, fail-loud rules, subclass override points |
| `tests/agent/test_adapter.py` | Inbound message parts + tool wire format |
| `tests/agent/test_runner_middleware.py` | Middleware hook dispatch (incl. the tool pair across every outcome) |
| `tests/agent/test_tools.py` | `Tool` base contract (spec stamping incl. `timeout_in_ms`, token pass-through); the `tool()`/`tool_class()` factory tests are skipped pending their redesign |
| `tests/agent/contrib/test_simple_tool_registry.py` | Self-scoped contrib tests: birth drafts per preflight outcome, decide delegation, execute resolution, `ProxyToolRegistry` routing/miss degradations/nesting — no runner |
| `tests/agent/contrib/test_plugins.py` | Self-scoped contrib tests: `PluginAgentSessionRunner` composition (one proxy, parts/middleware flattening, equality with a directly-configured runner) |
| `tests/agent/contrib/test_resource_permissions.py` | Self-scoped contrib tests: `PermissionStrategy` decide / apply_answer / pending_requests / grant + the tool mixin — no runner, no session |
| `tests/agent/contrib/shell/` | Self-scoped contrib tests: one file per shell tool (`tools/test_<name>.py`) + `test_plugin.py` (`ShellAccessPlugin` wiring, seeded rules, decide/pending flows) — no runner |
| `tests/agent/contrib/test_memory.py` | Self-scoped contrib tests: `MemoryPlugin` surface + scratchpad / todo-list behavior — no runner |
| `tests/agent/contrib/test_compaction.py` | Self-scoped contrib tests: split strategies, the context gauge, `build_compacted_session` (both strategies, tool-index rebuild, source untouched), and `compact()` against a `FauxProvider` — no TUI |
| `tests/agent/contrib/tui/` | Self-scoped contrib tests: pure modules (`test_approvals.py`, `test_render.py`, `test_sessions.py`, `test_wiring.py`, `test_cli.py`, `test_config.py`, `test_context_bar.py`) + headless Pilot tests driving `AgentApp` with a scripted `FauxProvider` (`test_app*.py`); the directory skips itself when textual is missing |

## When in doubt

| Question | Go to |
|----------|-------|
| Tool-execution lifecycle / approval state / errors / events | `luca/agent/core/models.py` + design principles 4 and 10 above |
| LLM projection / tool-output derivation | `luca/agent/core/projection.py` + design principles 5 and 6 above |
| Agent data model / session invariants | `luca/agent/core/models.py` + design principles 1–3 above |
| The tool-registry contract | `luca/agent/core/tool_registry.py` + `luca/agent/contrib/simple_tool_registry/` |
| Run lifecycle (run/start, AgentRun, cancel, timeouts, outcomes) | `luca/agent/core/runner.py` + design principles 9, 11, and 12 above |
| Where does this responsibility belong | `runner.py`, `projection.py`, `adapter.py`, and their tests under `tests/agent/` |
