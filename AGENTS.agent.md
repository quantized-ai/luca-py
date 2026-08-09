Guidance for the `luca.agent` layer. Read this file whenever you're working in `luca/agent/` or `tests/agent/`.

If you're working with the TUI, also read `luca/agent/contrib/tui/AGENTS.tui.md`.

## What this layer is

`luca.agent` is the primary product: a full-featured, durable agent framework. Its central artifact is a single serializable `AgentSession` that captures a complete conversation history — messages, tool executions, reasoning, turn boundaries, compaction — and can be reloaded to resume exactly where it stopped.

## Goals

- One canonical, JSON-serializable `AgentSession` that round-trips through `model_dump_json` / `model_validate_json` losslessly.
- A flat, append-only entry store (`AgentSession.entries`) addressed by id, with `Conversation.nodes` (an ordered list of ids) as the traversal path. Forking is cheap and explicit. Conversations are a STORE too — `AgentSession.conversations`, keyed by id, holding the main conversation and its archived predecessors — with `main_conversation_id` as the pointer into it. There is no "active" conversation.
- A resumable async agent loop exposed as `runner.run()` (lazy) and `runner.start()` (eager), both returning an `AgentRun` handle. The engine projects the active conversation to LLM messages, calls the model, records the assistant turn, executes tool calls, and loops — bracketed by `TurnStart` / `TurnFinish(outcome)`. One logical turn can span multiple runs.
- The whole tool lifecycle delegated to a `ToolRegistry` — the core has no built-in tool resolution or approval engine; permission policies are contrib (`contrib/simple_tool_registry`). The core's only tool type is `ToolSpec`, plain JSON-serializable data: it depends on no Python tool class, so a registry fronting a remote tool server (MCP, an HTTP tool service, another agent) is a first-class implementation, and `luca` stays an implementation-agnostic specification another language could implement against.
- Durable cancellation via `runner.cancel()` / `run.cancel()`, recorded as a `CancelRequested` entry and wound down at engine step boundaries.
- Compaction as a step inside a drive: the `ContextManager` decides and summarizes, the runner archives the conversation and installs a new one over the path it chose — atomically, losing nothing. The archived conversation stays in `conversations`; only the NAME moves (`main_conversation_id`), and the successor points back through `previous_conversation_id`.
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
│   ├── simple_context_manager/  # a concrete ContextManager that also compacts
│   │   ├── __init__.py  # package surface: SummarizingContextManager, gauge
│   │   └── manager.py   # SummarizingContextManager (should_compact + compact;
│   │                    #   keep_turns) + the context gauge (used vs the window)
│   ├── resource_permissions/
│   │   ├── __init__.py  # package surface: PermissionStrategy, rules, answers, the mixin
│   │   ├── strategy.py  # PermissionMode, ToolRule/ToolKindRule, ApprovalAnswer, PermissionStrategy
│   │   └── mixin.py     # ResourcePermission, AnswerOption, PermissionRequest, ResourcePermissionToolMixin
│   ├── subagents/       # parallel subagents: the four tools + the plugin
│   │   ├── __init__.py  # package surface: SubagentsPlugin, the tools, spawn_gate_open
│   │   ├── tools.py     # SpawnSubagent (declares is_subagent_spawn) +
│   │   │                #   CreateConversationResult (PRIVATE, runtime-invoked) +
│   │   │                #   StopSubagent (declares is_subagent_stop) + ListSubagents,
│   │   │                #   open_turn_children (the roster read)
│   │   └── plugin.py    # spawn_gate_open, spawning_prompt_part + control_prompt_part
│   │                    #   (callables), SubagentToolRegistry (the depth gate + the
│   │                    #   control-tool withholding), SubagentsPlugin
│   ├── skills/          # SKILL.md instruction sets read from disk
│   │   ├── __init__.py  # package surface: SkillsPlugin, SkillTool, Skill, the
│   │   │                #   discovery functions
│   │   ├── discovery.py # frontmatter parsing, the location precedence list,
│   │   │                #   discover_skills (nothing here raises for a bad skill)
│   │   └── plugin.py    # SkillTool (the body, on demand) + SkillsPlugin
│   │                    #   — needs the `skills` dependency group (PyYAML)
│   ├── prompts/         # the base system prompt + the project's instruction files
│   │   ├── __init__.py  # package surface: the two plugins + the pure helpers
│   │   ├── selection.py # FAMILIES, select_family (model id only), load_prompt
│   │   ├── text/        # base.md + anthropic/gpt/gemini/generic addenda
│   │   ├── environment.py # format_environment — pure, every input injected
│   │   ├── instructions.py # LUCA.md/AGENTS.md/CLAUDE.md, per-directory name
│   │   │                #   precedence, git-root-bounded walk, byte budget
│   │   └── plugin.py    # SystemPromptPlugin (callable PUBLIC part builders —
│   │                    #   subclass and override one; /model moves the prompt)
│   │                    #   + InstructionsPlugin
│   └── shell/           # the 8 generic shell tools + native/ (the 4
│                        #   provider-native ones + ShellNativeMiddleware) +
│                        #   ShellAccessPlugin — see AGENTS.md there
└── core/
    ├── __init__.py      # external surface: AgentSessionRunner, ToolRegistry, PreparedTool,
    │                    #   SystemPromptAssembler, all entry types, exceptions.
    │                    #   NO Tool / tool / ToolContext — those left the core.
    ├── models.py        # AgentSession (incl. the usages + tool_specs stores), all entry
    │                    #   classes (incl. CancelRequested, PrunedEntry),
    │                    #   the spawn/stop handshake reads (declares_spawn,
    │                    #   spawn_payload, stop_payload, spawns_committed),
    │                    #   ToolSpec (+ spec_id()),
    │                    #   ExecutionStatus/ApprovalStatus/ToolExecutionError,
    │                    #   TurnOutcome, RuntimeConfig, SessionConfig, Usage,
    │                    #   ConversationRuntimeStatus, ConversationStatus, and the
    │                    #   pure (nodes, entries) path derivations — pure Pydantic v2
    ├── tool_registry.py # ToolRegistry — the 4-method contract the runner drives tools
    │                    #   through (all async, session + conversation_id first) +
    │                    #   PreparedTool + the registry-author rules
    ├── context.py       # CancellationToken (runtime-only; never persisted)
    ├── context_manager.py # ContextManager — the context strategy: per-entry
    │                    #   context_tokens estimation, tool-output processing,
    │                    #   PrunedEntry templates, AND the compaction pair
    │                    #   (should_compact + compact) — concrete class, runner
    │                    #   default. Every method takes the live session first.
    ├── compaction.py    # CompactionPlan / UsageCounters / ConversationSnapshot +
    │                    #   validate_plan — the vocabulary the compaction pair
    │                    #   exchanges (pure — no session mutation, no ledger,
    │                    #   no asyncio). The extension point is ContextManager.
    ├── exceptions.py    # AgentError, CancelledError, AlreadyCancellingError,
    │                    #   ToolNotFound, InvalidToolArguments, ProjectionError,
    │                    #   CompactionPlanError
    ├── events.py        # AgentEvent union (block-level + streaming-delta + the
    │                    #   lifecycle events: ApprovalRequired, SubagentsSpawned,
    │                    #   SubagentStarted/Paused/Finished + the three
    │                    #   Compaction* ones); tool events carry deep snapshots
    ├── projection.py    # ConversationProjector — the PUBLIC conversation → LLM-message
    │                    #   strategy (subclass to customize history/tool-output policy)
    ├── adapter.py       # message_to_parts() (inbound response conversion) +
    │                    #   tool_spec_to_luca_tool() (tool-definition conversion)
    ├── middleware.py    # AgentMiddlewareMixin — the 13 duck-typed middleware
                         #   hooks, every one (session, conversation_id, …)
    ├── ledger.py        # SessionLedger — the single append/read door onto the entry
    │                    #   log; every door takes a conversation_id
    ├── system_prompt.py # coerce_system_prompt_part, SystemPromptAssembler,
    │                    #   DefaultSystemPromptAssembler, part-input type aliases
    ├── runner.py        # AgentSessionRunner, AgentRun handle, RunResult
    └── utils.py         # pretty_print(session) — the read-only text transcript
                         #   of a session (debugging view; reads the durable
                         #   entries, never the projection)

tests/agent/             # all agent tests; mirrors core/ layout; contrib tests under tests/agent/contrib/
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

One loudly-documented exception, and only one:

- `ToolExecution.tool_spec` is a restorable CACHE of the spec its `tool_spec_id` names. The id is authoritative; `tool_spec` is stripped from a serialized session and restored on construction, and must never be the source of truth for anything durable.

It is an exception because a consumer reads it like ordinary state and a writer must not treat it as such.

**Status is not stored at all.** `Conversation` has no `status` field: `AgentSession.get_conversation_status(conversation_id)` recomputes a `ConversationRuntimeStatus` from the nodes on every call, and it is the only door. A field nobody trusts is a rendering, not state — and with several conversations a persisted one would need a rule for which get refreshed and when, while archived conversations must stay frozen.

### 2. Storage and traversal are separate

`entries: dict[str, AnyEntry]` is the durable, append-only, uniformly-addressable node space.
`Conversation.nodes: list[str]` is the path — an ordered list of entry ids. Walk the path; resolve ids in the store.
`parent_id` is a recovery backstop and is **never traversed**.

### 3. Messages are entries

`UserMessage` and `AssistantMessage` live in `entries` alongside `ToolExecution`, `TurnStart`, `TurnFinish`, and `CompactionEntry`. One `Entry` base class, one `type` discriminator field, one `AnyEntry` discriminated union.

### 4. A tool call is two things

- The request block: a `ToolCall` object inside `AssistantMessage.parts`.
- A separate, mutable `ToolExecution` entry — the durable source of truth about that call's whole lifecycle — correlated by `tool_call_id`.

`ToolExecution` is one of the **three** mutable entry types (`CompactionEntry` and `ChildConversation` are the others — see below).

Each `ToolExecution` carries three orthogonal facts plus its provenance (`conversation_id` — the conversation it was BORN in, stamped by the runner; it is on this entry and no other because a `ToolExecution` is the only entry a consumer ever receives DETACHED from a path, and it is never traversed):
- `status: ExecutionStatus` — the framework's execution lifecycle and ONLY that: `RECEIVED`, `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `NOT_FOUND`, `INVALID`, `REJECTED`, `REFUSED`, `CANCELLED`, `INTERRUPTED`, `TIMED_OUT`. `RECEIVED` is the pre-birth state: the model asked for the call and the runner appended the entry, but the registry has not answered yet. `COMPLETED` means "the framework received a result", not "the tool succeeded". `NONTERMINAL_STATUSES` (RECEIVED/PENDING/RUNNING) is the single source of truth for every "still advanceable" rule — projection, pruning, context accounting.
- `approval_status: ApprovalStatus | None` — the CURRENT approval state (`None` = the policy never processed it; `PENDING`/`ALLOWED`/`REJECTED`). Always read approval from this field; `approval_decisions` is the append-only audit log of policy responses (only PENDING may repeat), never the source of current state.
- The outcome payload: exactly one of `result: ExecutionResult` (the body returned; `result.is_error` is the tool's OWN verdict — an `is_error=True` result is still `COMPLETED`) or `error: ToolExecutionError` (structured `error_type`/`error_message`/`details`, populated for `FAILED`/`NOT_FOUND`/`INVALID`/`REFUSED`) — or neither, for the status-only terminals.
- `raw_tool_call: ToolCall` — the (possibly middleware-effective) request; makes the execution self-contained. `tool_spec_id: str | None` — the durable reference into `AgentSession.tool_specs`, with `tool_spec: ToolSpec | None` as its restorable cache (`name`, `description`, `input_schema`, `metadata`, `tool_kind`, `namespace`, `version`, `timeout_in_ms` — NO arguments; both `None` when the tool never resolved). `extras` — a free-form dict written by registries/middleware, stored verbatim, never interpreted by the core (`SimpleToolRegistry` stores the tool's approval context under `extras["approval_context"]`).
- Lifecycle timestamps: `started_at` (set iff the body was dispatched — true for EVERY outcome, because the runner persists it only after `prepare()` has returned), `ended_at` (every terminal transition), `cancel_signalled_at` (run cancellation only — a deadline never sets it). `updated_at` is ledger bookkeeping, not timing.

**Tool specs are normalized.** Each distinct `ToolSpec` is stored once per session in `AgentSession.tool_specs` under `spec_id()` — a full 64-char SHA-256 hex over the spec's JSON with recursively sorted keys, no whitespace, non-ASCII literal, UTF-8. The rule is pinned rather than left to the implementation and is deliberately NOT an overridable hook like `generate_id()` / `now_ms()`: it has to be identical across processes, machines and other-language implementations, so determinism here is data integrity, not test convenience. A spec must be a pure function of the tool DEFINITION — anything call-scoped in `ToolSpec.metadata` mints a row per call and silently defeats the whole thing. The store is append-only and never garbage-collected; a content hash's only failure mode is a redundant row, never a wrong lookup.

Constructing an `AgentSession` restores every `tool_spec` from its id (shared BY REFERENCE with the `tool_specs` row) and refuses two shapes: a `tool_spec_id` absent from `tool_specs`, and a `tool_spec` with no `tool_spec_id` (which can only come from a pre-normalization file, which would otherwise load fine and then lose every spec on the first save). There is no migration — regenerate pre-refactor session files.

These combinations are framework conventions, not Pydantic validators — middleware is trusted and may author unusual state; the application owns the consequences.

**`AgentSession.extras`** is free-form application state, stored verbatim and never interpreted by the core — the session-level twin of `ToolExecution.extras`. It is how a tool, a registry or a plugin keeps state that outlives the process WITHOUT the application inventing a second file: whoever composes the runner hands the dict in and it rides along on every save. `MemoryPlugin` is the reference case (`wiring.build_runner` passes `session.extras.setdefault("todos", {})` as the todo store, so a resumed session keeps its plan and its numbering); the plugin itself only ever sees a dict. Keys should be namespaced and values JSON-serializable. Note that a compaction installs a NEW conversation id and does NOT re-key anything under `extras` — state keyed by conversation is the application's to move (`CompactionFinished.new_conversation_id`).

### 5. The wire payload is derived, never stored

The **`ConversationProjector`** (`projection.py`) is the public strategy that recomputes the LLM message list from a path — `project(nodes, entries)`, taking the ordered id list rather than a `Conversation` object — on every call: it drops `TurnStart`/`CancelRequested`, projects a CANCELLED `TurnFinish` as the synthetic `[Request interrupted by user]` marker, unwraps messages into client content blocks, projects each terminal `ToolExecution` to its correlated `ToolMessage` (COMPLETED → its result verbatim; every other terminal → derived error text with `is_error=True`), and renders a `CompactionEntry` as a synthetic user message — with POSITIONAL rules that live on `project()` itself: a whole compaction bracket (`ts_c cmp [cr] tf_c`) projects as nothing, whatever its outcome, while a `CompactionEntry` outside a bracket projects its `parts`; and a subagent's result renders as a task-update user message at its RESULT EXECUTION's path position (the execution `ChildConversation.result_execution_id` names — the link itself renders nothing), so the projected history stays append-only while the link mutates in place and a re-awakened model always finds the update below its own last reply. It is a concrete class — pass a subclass as `conversation_projector=` to change any policy (history shaping, redaction, tool-output wording); ALL default derived wording lives on the class. The same `project_tool_execution` output feeds the `ToolExecuted` event's `result_text`/`is_error`, so event and wire never disagree. There is no projection middleware; `before_llm_call` stays as the downstream last-mile hook.

### 6. Fail loud on a mid-execution projection

Projecting a conversation that contains a nonterminal (`RECEIVED`, `PENDING` or `RUNNING`) `ToolExecution` raises `ProjectionError` — the runtime must never call the model while an execution is nonterminal — with EXACTLY ONE carve-out (0008): a GATED execution (`PENDING` with `approval_status=PENDING`) projects a placeholder `ToolMessage` carrying `ConversationProjector.AWAITING_APPROVAL_OUTPUT` (`is_error=False` — the call has not failed, and an error result is what makes a model retry). A gate is a durable resting state only the application can move, not a runtime in flight, and the placeholder is what lets a message posted while blocked reach the model at all; the real result replaces it at the same path position once the approval is answered, making it the one projected tool message that is not final. `PENDING` with approval `None`/`ALLOWED` still raises. Missing entry ids, unknown entry types, and a COMPLETED execution without a result fail the same way; projection never invents fallback content. An UNRESOLVED `ChildConversation` is legal inside the OPEN turn (it renders nothing — the model tracks its tasks through the spawn confirmations and the updates) and fails loud anywhere else: no close may leave an unresolved child behind, so that state is a framework bug or hand-authored corruption.

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

A handle is also the root of a TREE once a turn spawns subagents: `run.children` / `run.child(cid)` reach the subtree, `run.approvals` streams the gates raised during the run, and `run.notify(execution)` says "look again NOW" for that execution's conversation. `autostart_subagents=` (default True, and implied by `start()`) decides whether the FRAMEWORK drives the children (their events arrive on this stream) or the application does (`False` — the handles are lazy and unstarted, forwarding is suppressed so nothing is delivered twice, and driving or cancelling every spawn becomes the caller's obligation). A run handle is single-use, so `child()` MINTS a fresh one under `False` — that is the only door back to a subagent parked at a gate.

### 10. Two-tier events

The engine yields `AgentEvent` union members in two tiers:

- **Block events** (always): `ReasoningBlock`, `TextBlock`, `ToolCallReceived`, `ToolExecutionStarted`, `ToolExecuted`, `FinishReason`.
- **Delta events** (`streaming=True` only): `ReasoningStart`/`Delta`, `TextStart`/`Delta`, `ToolCallStart`. Session behavior is identical regardless of streaming.
- **Lifecycle events** (always): `ApprovalRequired` fires as the last event before a gate, carrying the awaiting `ToolExecution` entries (equivalent to `runner.pending_approvals()`); `SubagentsSpawned` announces one response's children as a batch (parent-attributed) and `SubagentStarted` / `SubagentPaused` / `SubagentFinished` track each subagent's drive (subagent-attributed; `Finished` carries the closing `TurnOutcome`) — announced-but-never-Started means queued behind `subagents_max_workers`, and a `Finished(CANCELLED)` with no `Started` means cancelled before admission (the parent's flush announces it, framework mode only); `CompactionScheduled` / `CompactionStarted` / `CompactionFinished` carry a deep `CompactionEntry` snapshot through one compaction's lifecycle, `Finished` firing whatever the outcome.

The three tool-lifecycle events carry a deep `ToolExecution` SNAPSHOT (plus a denormalized `tool_call_id`), never a live ledger reference — tool name and arguments come from `execution.raw_tool_call`. Per execution: `ToolCallReceived` fires once at the persisted birth state (PENDING, or a preflight-terminal NOT_FOUND/INVALID/FAILED/REFUSED); `ToolExecutionStarted` fires iff the body dispatches, after RUNNING is persisted and immediately before invocation; `ToolExecuted` fires once at the terminal outcome, with `result_text`/`is_error` copied from the projector's `project_tool_execution` output. Every event follows the persistence of the state its snapshot shows — the stream never leads the durable session.

Every text-bearing event exposes its content on `text`, so one `match` statement serves both vocabularies.

There is no `TurnFinished` event — `RunResult` is the completion signal. A flush run may emit zero events.

### 11. Derived status machine

`ConversationStatus` has FOUR values and answers exactly one question — **what will the next `run()` do?** It is derived from the entries on every read (`session.get_conversation_status(id)`) and never stored.

| | the next `run()` | post a message? |
|---|---|---|
| `IDLE` | nothing — there is no work | yes |
| `BUSY` | work — the run can still be exhausted | yes — into the open turn (or queued behind a trailing message) |
| `BLOCKED` | stop again immediately; you must act first | yes — the conversation derives `BUSY` and the next drive answers it past the gate (0008), then re-parks |
| `CANCELLING` | flush the turn, not answer it | no |

The post column is the common case, not the whole rule — `post_message` owns
the full acceptance matrix (an open compaction bracket rejects, archived and
finished conversations reject whatever they derive; an open turn with
unresolved subagents ACCEPTS — the post is material and wakes the parent).

Derivation, in precedence order:
- open turn with an unconsumed `CancelRequested` → `CANCELLING` — the next drive is the flush.
- open turn with UNRESOLVED CHILDREN → its own sub-order, because a parent can advance for more reasons than its own executions: (1) an advanceable execution (`RECEIVED` / `RUNNING` / `PENDING` with approval `None`/`ALLOWED`) → `BUSY`; (2) any child that can advance (BUSY / IDLE / CANCELLING) → `BUSY`; (3) a gated execution with no unseen user post → `BLOCKED` — ranked ABOVE the material term, because the next drive can only re-park at the gate and `BUSY` would hot-loop a poll-for-BLOCKED consumer; an unseen POST (`open_turn_unseen_post` — a `UserMessage` after the open turn's last `AssistantMessage`, deliberately narrower than the material predicate) lets the gate term yield (0008), since the placeholder makes the next drive able to answer it; (4) unseen material (`open_turn_unseen_material` — a posted message or a terminal non-spawn-declaring execution after the last assistant message, the resolved child's result execution included; EXCLUDED under `wake_parent_on_subagent_completion=False`, which the derivation passes as the predicate's `include_child_results=` so status and drive stay in agreement) → `BUSY`: the next run() calls the model with it; (5) else `BLOCKED`.
- open turn with something runnable → `BUSY`. Runnable means: a `RECEIVED` execution (the next drive births it), or an orphaned `RUNNING` execution (the next drive recovers it to `INTERRUPTED`, no re-dispatch), or a `PENDING` execution whose `approval_status` is `None` (crash mid-decide — `run()` self-heals by asking again) or `ALLOWED` (dispatchable), or no nonterminal execution at all (the model can be called).
- open turn with nothing runnable but an unseen user post (`open_turn_unseen_post`) → `BUSY` (0008): only gated executions remain, but the placeholder lets the next drive answer the post and re-park.
- open turn with nothing runnable → `BLOCKED`. That is: only gated executions remain.
- trailing `UserMessage` → `BUSY` (queued work).
- anything else, INCLUDING a closed `TurnFinish` whatever its outcome → `IDLE`.
- A closed COMPACTION bracket (`ts_c cmp tf_c`) is transparent — skipped, repeatedly if several stack, and the leaf before it derives; otherwise a completed one buries a queued user message. An *open* compaction bracket derives `BUSY` like any open turn.

**The status says nothing about approvals.** A gate can belong to a subagent whose siblings are still working, so the conversation stays `BUSY` while `pending_approvals()` already returns that gate; only when nothing else can advance does it become `BLOCKED`. `BUSY` is also still true after a crash — "can be advanced" survives the process dying, so unlike the old `RUNNING` it needs no self-healing rule.

A logical turn spans one `TurnStart`/`TurnFinish` bracket even across an approval pause. A `TurnStart` with no later `TurnFinish` means resume, not re-open.

`post_message(content, conversation_id=None)` accepts far more than IDLE. The rule: **a conversation accepts a message whenever something will eventually answer it** — an open conversational turn (BUSY or BLOCKED) takes the mid-turn append (a post into a GATED turn is answered past the gate, 0008: the conversation derives BUSY, the next drive projects the gated call as the awaiting-approval placeholder and runs exactly one model round, then re-parks — the gate itself untouched, `pending_approvals()` returning it before and after), a trailing message queues more behind it (one turn answers them all), a live subagent accepts posts too (`conversation_id=`; its seed prompt is merely its first user message), and an open turn with **unresolved subagents** accepts as well: the children never see the message, but the PARENT does — the post is material, a parked drive is woken through the notify door (`_recheck` + `_ensure_driven`), and the model can steer (answer, spawn more, stop a task). Rejections: CANCELLING → dedicated `ConversationCancellingError`; an open COMPACTION bracket (bracket-shape check, never status) → `AgentError`; a finished subagent or an archived predecessor (identity-checked — an archived path can derive BUSY) → `AgentError`. Two guarantees ride with the mid-turn append: **a turn never closes COMPLETED while its open span holds a user message the model has not seen** — the drive's close-site check records the premature final answer and runs one extra round instead (`hard_max_steps` still bounds it) — and a failure close (CANCELLED / ERRORED / TIMED_OUT, the hard step limit) **buries** the message: unanswered in that turn, but projection's flat walk carries it into the next request. **A failed turn still does not auto-retry** — a closed bracket is IDLE whatever its outcome, so recovery means posting a new message rather than silently re-sending the identical request. One documented blind spot: a message posted while an LLM call is in flight lands BEFORE the recorded assistant entry, where the durable material predicate cannot see it — the live drive covers that window with its `seen` fingerprint; a session reloaded from exactly that window answers the message on the next material instead of immediately.

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

The `prepare` row is the one asymmetry, and it is deliberate: the call has already been SELECTED for dispatch, so it is the dispatch path's to finish. Both routes produce the same durable shape, and the choice is now purely structural — the dispatch path holds the state and the wind-down does not have to re-derive it. (Historically the reason was hook-firing: `before_tool_execution` used to run in the wind-down too, so handing the call back would have fired it twice. It no longer does — the hook is dispatch-only — so that justification is retired, not the behavior.)

Wind-down itself happens at the engine's step boundaries and turn-close sites:
- Undispatched executions — `RECEIVED` (never born) or `PENDING` — → stamped `cancel_signalled_at`, then `CANCELLED` (approval state untouched; a DENIED call was already terminal `REJECTED` at decision time). **A cancelled birth is CANCELLED, never FAILED** — a cancellation is not a tool failure.
- In-flight executions → persisted with `cancel_signalled_at` FIRST, then the grace period: a within-grace return is `COMPLETED` with its real result (keeping the stamp), a raise is `FAILED`, expiry is `INTERRUPTED`.
- Already-terminal executions are untouched.
- Closes the turn with `TurnFinish(outcome)`.

A response containing N tool calls yields N tool executions, even when cancellation lands mid-batch — and now regardless of when the cancel lands, because the N entries are appended before any birth is attempted. This is why the `create_execution` race is per call, inside the birth helper, and never around the `asyncio.gather`: killing the gather would lose every draft.

No `except CancelledError` clause is needed anywhere in the runner, and adding one would mislead the next reader. `asyncio.CancelledError` derives from `BaseException`, so the broad `except Exception` around `create_execution` never sees it; and the race helper absorbs the kill it issued and reports the outcome as a boolean rather than re-raising.

An unconsumed cancel controls every close: an LLM answer landing within the grace window is recorded but the turn still closes with the cancel outcome; an LLM failure within the grace window is discarded and the run returns normally.

A parked cancel survives save/reload — the next `run()` or `start()` is the flush (instant, no LLM call).

A second `cancel()` while one is unconsumed raises `AlreadyCancellingError` (first call wins).

No open turn → no-op (only possible on an undriven lazy handle or before any run; `start()` opens the bracket at call time, so a started run is always cancellable).

Wire projection: a cancelled turn becomes a synthetic user message `[Request interrupted by user]`. Failed turns project nothing.

## Key facts

### Context vs usage

Two different measurements, never conflated:

- **`Entry.context_tokens`** — the intrinsic estimated size of that entry's model-facing content, shared with the entry across every conversation that references it. Calculated by the runner's **`ContextManager`** collaborator (`context_manager.py`, passed as `context_manager=`, defaults to the simple built-in: one token per 4 characters) on every NEW entry before `before_entry_written`, and recalculated on a `ToolExecution`'s terminal transition before `after_tool_execution`. Middleware has the final say on every WRITE path — nothing is recalculated, validated, or repaired after it (the one exception is `recalculate_context_tokens()`, which runs no middleware at all; see below). Never derived from provider usage.
- **`AgentSession.usages[conversation_id][entry_id]` → `Usage`** — the provider-reported consumption for one entry in one conversation (the same assistant entry in two conversations can have different usage: input covers the whole request context). A self-describing association record (`conversation_id` + `entry_id` are required fields), written only through `SessionLedger.record_usage()` when an assistant message is recorded. Entries carry NO usage field; `TurnFinish` carries no rollup; `RunResult` carries no usage — aggregate from the store.

Every `ContextManager` method takes the live session first, so one argument order describes the whole contract and every policy sees the same state — the active model included, which is what makes a real tokenizer implementable at all:

```python
def calculate_context(self, session, entry) -> int
def prune_entry(self, session, entry) -> PrunedEntry          # NO framework call site
def process_tool_output(self, session, execution, result) -> ExecutionResult
def should_compact(self, session, conversation_id) -> bool    # compaction — see below
async def compact(self, session, conversation_id, nodes, entry) -> CompactionPlan | None
```

Only the compaction pair takes a `conversation_id`, and the split is a decision: `context_tokens` is intrinsic to the ENTRY and shared by every conversation referencing it, so the first two do not need one; `process_tool_output` does not either, because the execution it already receives carries `conversation_id`.

`process_tool_output()` transforms a returned `ExecutionResult` before the terminal execution is constructed (identity by default) — the durable session, the `ToolExecuted` event, and the wire all see the processed output. It receives the `ToolExecution` IN TRANSITION (status still RUNNING, `result` not yet attached): read it for identity — `tool_spec`, `raw_tool_call.name`/`arguments` — never for outcome. That is what makes "truncate `bash` output at 30k characters but never truncate `read`" expressible.

Two `calculate_context` gotchas, both silent when violated: it runs on EVERY new entry, so scanning `session.entries` inside it makes a turn quadratic (cross-entry work belongs in `prune_entry` / `process_tool_output`, which run rarely); and on an append it runs INSIDE the ledger's build callback, so the entry already has its `id` but is not yet a member of `session.entries` — an implementation that looks itself up there raises `KeyError` on every append.

Because counts are STORED on the entry, a model-aware `ContextManager` goes stale the moment `llm_config` changes. `AgentSessionRunner.recalculate_context_tokens()` re-derives `context_tokens` for every entry in `session.entries` — every entry, not just the active path, because the count is intrinsic to an entry and shared by every conversation referencing it — setting no other field. It runs NO middleware: `before_entry_written` is scoped to the conversation whose operation caused a write, and this method rewrites every entry across every conversation at once, so no single id would be honest. It is an operational refresh of a derived estimate, not a write with a scope. **Nothing in the framework calls it**: no constructor keyword, no CLI flag, no automatic invalidation on a model switch (which would put an unbounded rewrite behind an innocuous assignment). The shipped `ContextManager` is a character estimate no model choice affects; the method is there for the application that swaps in a real tokenizer, and that application calls it.

**Pruning** replaces an entry's contribution to the path without touching the original: `ContextManager.prune_entry()` builds a `PrunedEntry` TEMPLATE (placeholder identity; v1 supports terminal tool executions only, replacement text `"[tool output has been pruned to reduce context]"`), and `SessionLedger.prune(original_id, build)` stamps identity (the original's `parent_id`), verifies the referent/type/terminality invariants, and swaps the node id in place. The original stays in `entries`. `ConversationProjector.project_pruned` resolves the referent and re-emits the replacement content under the original's role and `tool_call_id` (a missing referent, type mismatch, or unprojectable source raises). The runner deliberately exposes NO public prune/context-total methods yet.

### The tool registry

The runner is constructed with one `ToolRegistry` (`luca/agent/core/tool_registry.py`; `None` = toolless agent). The core touches tools through exactly four methods — all async, all taking the live `AgentSession` first, none receiving the cancellation token:

```python
class ToolRegistry:
    async def get_tools(self, session, conversation_id) -> list[ToolSpec]: ...
    async def create_execution(self, session, conversation_id, call) -> ToolExecution: ...
    async def decide(self, session, conversation_id, tool_execution) -> ApprovalDecision: ...
    async def prepare(self, session, conversation_id, tool_execution) -> PreparedTool: ...

PreparedTool = Callable[..., Awaitable[ExecutionResult]]
async def run(*, cancellation_token: CancellationToken) -> ExecutionResult: ...
```

**Every seam takes the conversation as an ID, never the `Conversation` object.** A session holds several conversations advancing concurrently, so there is no "active" one to read behind the caller's back. The id is passed because `Conversation` is live and mutable (the runner appends to its `nodes` while application code holds it) and because every other cross-reference in the data model is already an id; resolve it with `session.conversations[conversation_id]` when the object is needed. This is what lets a registry answer differently per conversation — withholding a tool from a subagent, or keying its own state.

- **`get_tools` is dynamic** — queried fresh per LLM call (the result may vary with session state); never a lifecycle hook. It returns `ToolSpec`s, so a registry backed by JSON Schema needs no Python class. An exception propagates and aborts the run; the runner never substitutes an empty tool list, because calling the model with no tools when the registry meant to offer some silently changes the answer.
- **`create_execution` returns a birth DRAFT** with no identity (`id`/`created_at` are `None`). The registry owns the call-scoped facts: `raw_tool_call`, `tool_spec` (incl. `timeout_in_ms`; `None` if unresolved), the birth `status` (PENDING, or terminal-at-birth NOT_FOUND/INVALID/FAILED), the `error` for a terminal birth (the registry authors it), and `extras`. The RUNNER owns identity and `is_doom_loop_flagged` and stamps `ended_at`-if-terminal/`context_tokens`, so determinism principle 8 holds, and the LEDGER files the spec and stamps `tool_spec_id` — registries stay unaware of the storage scheme. If `create_execution` raises (or the registry is `None`), the runner synthesizes the draft itself (FAILED / NOT_FOUND) — failures stay isolated per call.
  **The entry exists before the call is made.** The runner appends a `RECEIVED` execution per tool call synchronously with the assistant message, then folds this draft into it one drive step later; the draft is not what creates the entry. Two consequences a registry author can rely on: `create_execution` sees a path that ALREADY contains its execution (and the assistant message that asked for it), and the call is retried from durable state after a crash or a suspend, so it must stay an idempotent query of the registry's own state — the same rule `decide` has.
- **`decide`** returns ALLOW / DENY / PENDING; exceptions propagate and abort the run (the executions stay unresolved; the next `run()` asks again), so implementations must be idempotent queries of their own state.
- **`prepare`** resolves the tool and validates the arguments, then returns a CALLABLE that runs the body — it must not run it. Raising means the body never runs and the execution is never marked RUNNING: `ToolNotFound` → NOT_FOUND, `InvalidToolArguments`/pydantic `ValidationError` → INVALID, anything else → FAILED, all with `started_at=None`. Once the callable is invoked, every raise is FAILED. A returned `ExecutionResult` → COMPLETED (after `ContextManager.process_tool_output`). It is called once per dispatch attempt and only for an already-approved execution.

The arguments are easy to misuse in opposite directions. The **`AgentSession` is the LIVE object** — the same instance the runner and ledger write through — so a registry may hold it and re-read current state later, including from inside a prepared callable. The **`ToolExecution` is a SNAPSHOT**, detached as of the moment the call was made; the runner persists RUNNING and `started_at` AFTER `prepare()` returns, so a reference captured during `prepare()` is already stale. Capture what the callable needs during `prepare()`. And the session is **read-only to every implementation** — the runner owns every write, `session.tool_specs` included.

The registry-author rules (`prepare()` must be re-callable and non-blocking, must not return holding a lock/lease/slot, must never swallow `asyncio.CancelledError`, blocking sync work belongs in `asyncio.to_thread`, **the core never validates arguments** even though it now knows every schema, and **rule 13: be concurrency-safe** — one registry instance serves every conversation and the framework does NOT serialize these calls, so no per-call state on `self`, state keyed by `conversation_id` needs no lock, deliberately-shared state needs an `asyncio.Lock` around the mutation and never around the I/O) live in full in `tool_registry.py`'s module docstring. Every one of them fails silently when violated; read them before writing a registry.

There is **no global permission gate anywhere**: each registry answers `decide()` for its own tools. Cross-cutting approval is an application composition pattern (share one strategy across registries), never a framework or plugin API concern.

The engine has exactly **one** `decide()` call site — the top of its loop. Any open-turn execution that is undecided (`approval_status` is `None` or `PENDING`) is handed to the registry. Sibling undecided executions are decided concurrently via `asyncio.gather`. Each response both updates `approval_status` directly and appends to the `approval_decisions` audit log; a DENY is terminal on the spot (`status=REJECTED`, `ended_at` stamped, outcome middleware runs, `ToolExecuted(REJECTED)` emitted).

A PENDING decision defers only THAT execution: every ALLOWED sibling proceeds to dispatch, and the runner parks (`ApprovalRequired` as the final event; the conversation then derives `BLOCKED`) only after all currently runnable work has advanced. The model is never called again until every tool call in the assistant response has a terminal execution and a correlated tool output — with one exception (0008): an unseen user post drives one round past the gate, the gated call projecting as the awaiting-approval placeholder; the drive then re-parks at the same gate. Re-entering `run()` does not raise — it simply asks the registry again. `runner.pending_approvals()` returns the awaiting `ToolExecution` objects.

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
- **`before_tool_execution` fires exactly once per DISPATCH ATTEMPT**, never twice for one outcome, and never at all for a call that does not dispatch. Every terminal outcome — dispatched or not — goes through the single `_finalize_outcome` tail. A crash during `prepare()` writes nothing, so the next drive fires it again over the ORIGINAL call — deliberately NOT once-per-call-forever.

**Invariant: every tool call produces exactly one tool output, always.** Now load-bearing in three new places: a cancelled birth, a cancelled decide, and a `prepare()` failure each still produce their one execution and one output.

A call that is terminal at birth is persisted with a structured `error` and never reaches `decide()` (`approval_status` stays `None`); it still passes through the tool middleware pair.

The batteries-included registries live in `luca/agent/contrib/simple_tool_registry/`: `SimpleToolRegistry(tools, permission_policy)` reproduces the classic preflight (resolve → validate → duck-typed `get_approval_context`, stored under `extras["approval_context"]`), delegates `decide()` to a `PermissionPolicy` (async `decide(session, tool_execution)`; `YoloPermissionPolicy` allows everything), and returns from `prepare()` a closure binding the validated arguments and the session; `ProxyToolRegistry(*registries)` composes registries (`get_tools` recomputes and caches a `{name → child}` route PER CONVERSATION — a child may withhold a tool from a subagent, so one shared route would be wrong routing in both directions; duplicate names raise). The proxy's `decide` and `prepare` resolve INDEPENDENTLY of that cache, warming it once from the children on a miss — a call left pending approval by a previous process now resolves on a fresh one, is gated by its owning child, and dispatches. That is a correctness requirement, not a convenience: once `prepare()` can resolve without the cache, the old blanket ALLOW-on-cache-miss becomes a permission bypass on a cold resume. ALLOW on a genuinely unresolvable name STAYS — `prepare()` then raises `ToolNotFound` and the call records the honest NOT_FOUND, where DENY would record REJECTED for a tool that never existed. Everything richer — modes, rules, resource globs, answer-decoupled interactive approval, the `ResourcePermissionToolMixin` — lives in `luca/agent/contrib/resource_permissions/`.

### Compaction

Replacing the older span of a conversation with a summary of it. **Compaction opens a new conversation inside the same session** — it does not create a new session: add one entry, archive the current view, open a new one over the ids that survive. The archived conversation keeps the exact pre-compaction path as its own row in `conversations`, every compacted entry stays in `entries`, and `CompactionEntry.compacted_nodes` records precisely which ids the summary replaced.

Compaction is the `ContextManager`'s other half (`luca/agent/core/context_manager.py`) — the same collaborator that counts context decides when there is too much of it. Two methods, and they own the whole decision:

```python
class ContextManager:                                                  # …plus the three per-entry methods
    def should_compact(self, session, conversation_id) -> bool: ...    # SYNC — start() consults it at call time
    async def compact(self, session, conversation_id, nodes, entry) -> CompactionPlan | None: ...
```

The shipped default declines and raises, so compaction never happens unless the manager implements it. There is no "no policy configured" state to check — a manager always exists; `schedule_compaction()` therefore always writes its bracket, and a manager that only accounts surfaces `NotImplementedError` on the next drive (ERRORED bracket, raised to the caller like any USER-sourced failure).

- **`nodes`** is the path the manager may rewrite: the active path minus this compaction's own `TurnStart`, ending with `entry`. `plan.nodes` may carry any of those ids in any order with new entries interleaved, and NOTHING else — an id outside the tuple is a plan rejection, so a manager never has to route around framework markers. `plan.nodes = list(nodes)` is a legal full carry.
- **`entry`** is a DEEP COPY of the committed `CompactionEntry`; the runner applies exactly `parts`, `llm_config` and `metadata` from the returned plan and discards the rest. The copy is load-bearing: a manager that wrote `parts` onto the live entry and then failed would leave a summary projecting onto an unchanged path.
- **`plan.nodes`** is the new conversation before ids exist: a `str` carries an existing node over, an entry object is created there (the runner stamps `id`/`parent_id`/`created_at`, one timestamp, parents threaded left to right).
- **`plan.usage`** is a typed `UsageCounters` recorded for the ATTEMPT, against the pre-compaction conversation.

`CompactionEntry` is the second mutable entry type: written when the intent exists (`schedule_compaction()` or `should_compact`), mutated as it progresses, left in its terminal state whichever way it ended. It has NO `status` field — the surrounding turn bracket owns how the attempt ended and the entry owns what it produced.

**Runner integration.** The compaction step runs at the top of `_drive`, BEFORE the conversational bracket: flush a parked cancel → resume / skip / decide → run it → then drive the turn. At most one per drive (structural — the step sits outside the loop), and never while a conversational turn is open. `start()` decides at call time so an eager run opens a compaction bracket instead of a `TurnStart`.

**Safety.** Everything fallible happens before the transition; `SessionLedger.transition_conversation` is the atomic region and contains only plain assignments. `validate_plan` refuses a plan that references an unknown or un-offered id, references one twice, is empty, omits the compaction entry, was computed against a conversation that has since moved, or carries no content — STRUCTURE only, never meaning. A `source=USER` failure raises; a `source=POLICY` failure degrades so the user's turn survives; a cancel always stops the drive. An interrupted compaction resumes in place (the test is `entry.parts is None`, not the bracket's shape); a closed bracket is never retried.

Details in `docs/agent/12-compaction.md`. A ready-made implementation ships in `luca/agent/contrib/simple_context_manager/`: `SummarizingContextManager` (the context gauge + an LLM summary, `keep_turns` knob).

### Subagents

A subagent is a SECOND CONVERSATION in the same session, advancing at the same time as the main one and linked into its parent's path by a `ChildConversation` entry (the third mutable entry type: appended unresolved, gaining `execution_result` plus `result_execution_id` — the runtime-minted result execution that produced it — when the child's turn closes). The catalog is flat and holds no parent pointer — parent → child is the only direction anything traverses, which is exactly why `Conversation.depth` is stored.

**The parent is re-engaged per resolution, and its turn cannot CLOSE until every child resolves.** A parent parked on its subagents wakes whenever the open turn holds something the model has not seen — `open_turn_unseen_material`: a mid-turn post, or a terminal execution of a non-spawn-declaring tool after the last assistant message, which the resolved child's private result execution always is. A pure spawn round deliberately does not wake (its confirmations travel with the first real update), and a spawn-declaring call's outcome — spawned, declined, refused — never wakes by itself. `wake_parent_on_subagent_completion=False` (RuntimeConfig, default True) narrows the material to everything BUT result executions (the ones an open-turn link's `result_execution_id` names, via the predicate's `include_child_results=` keyword): children still resolve the moment they finish, posts and ordinary tool results still wake, but the model is called once — when no unresolved child remains and the park guard stops applying. Projection is untouched, so a model woken for any reason still sees every update it has not seen. Each resolution projects as a task-update user message at the RESULT EXECUTION's path position (`<task id={task_id} status=… completed_at="…">` — the spawn payload's task id, absolute-UTC timestamp), so history stays append-only while links mutate in place. The COMPLETED close site refuses to close over an unresolved child; the drive parks again once the model answers text-only with children still out. A wake round is a real assistant step: `hard_max_steps` counts it, and a hard-limit trip with children still running CANCELS them, settles every link, and closes ERRORED (a cost stop wins). A MAIN-conversation LLM failure mid-orchestration leaves the turn OPEN and re-raises (children keep working; the next `run()` resumes); a SUBAGENT parent's failure settles its own children and closes ERRORED — nothing outside a subagent ever retries it mid-run.

**The stop handshake** mirrors the spawn handshake, value-side only (no gate — stopping bypasses no cap): a completed execution whose `structured_content` carries `is_subagent_stop=True` makes the drive cancel the named DIRECT child — matched by spawn-payload `task_id` among links that PRECEDE the stop execution on the path, so a later task reusing the id can never be killed by an old signal, and reload replays are no-ops (a resolved, finished, or already-cancelling target consumes the signal). Contrib ships `stop_subagent` (declares `SubagentStop`; validates the target against the live session and returns the payload with the flag down for a dead target) and the read-only `list_subagents`; `SubagentToolRegistry` withholds both until the open turn has children, and `control_prompt_part` teaches them on exactly that predicate — a second prompt/tool-list identity beside the spawn gate's, separate because a spent spawn budget must not silence it.

**The contract is a DECLARATION, never a tool name.** A tool whose `ToolSpec.output_schema` declares `is_subagent_spawn` is a spawn tool; a completed execution whose `structured_content["is_subagent_spawn"]` is True created one. The gate reads the declaration (before the model call, from the spec alone) and the handshake reads the value (after the call), which is what lets a spawn tool decide at runtime NOT to spawn while staying gated. A name match would have let a `delegate_work` tool spawn through the handshake and never be filtered — the depth cap would quietly stop existing.

Three violations are refused loudly at child-creation time, all `AgentError`: a payload from a spec that never declared one, a spawn from a conversation at or past `subagents_max_depth`, and a payload missing a required field. The registry's `get_tools` gate is the first line and the runner's check is the second, because a registry resolves by NAME at dispatch: the cap has to mean something at the moment a child would appear, not only where tools are advertised. The gate is one predicate written twice (`spawn_gate_open` in contrib, `_spawn_gate_open` in core; `_verify_gate` catches them drifting apart) with three clauses: enabled, `depth < subagents_max_depth`, and the per-turn spawn budget `spawns_committed(...) < subagents_max_per_turn`. The budget's overflow is NOT a violation: the tool list is fixed before the model call, so several spawn calls in one response can overrun it — the overflow execution is born `REFUSED` with a `SpawnLimitReached` error the model reads verbatim, its body never runs, and no child is created. Only calls that commit a subagent count (a declining, denied or failed spawn call consumes nothing); each conversation's open turn has its own budget, counted from durable entries (`spawns_committed`), so a reload changes nothing.

**The worker pool** (`subagents_max_workers`) bounds how many subagents are DOING WORK at once, session-wide. Spawning always succeeds; a child past the cap waits, holding its seed message, until a slot frees — admission is FIFO in spawn order, announced by `SubagentStarted`. One rule: a slot is held only during a subagent's own productive work — a conversation parked on its children, gated on a human, or winding a cancelled turn down holds none (release-on-park is what makes a nested tree unable to deadlock at any cap; `subagents_max_workers=1` is legal at any depth). The pool is runtime state, rebuilt from the session on the next `run()`; it never interrupts work in progress, only decides what starts next; the main conversation never competes; the model is never told. Incompatible with `run(autostart_subagents=False)` — the cap works by withholding starts the framework owns, so that combination raises.

**Approvals are subtree-scoped.** `runner.pending_approvals(conversation_id=None)` returns every gated execution BENEATH that conversation, as a flat `list[ToolExecution]` — attribution is `execution.conversation_id`, and there is deliberately no wrapper type. Two ways to make a drive re-ask `decide()`: the next `run()` (which also restarts any framework-owned child parked at a gate) or `run.notify(execution)` from inside a live run. `run.approvals` is the stream that makes the second usable — a subagent can gate while its siblings keep working, so there is no between-drives moment to poll in. It is AT-LEAST-ONCE; consumers dedup.

**Cancellation cascades downward and resolves upward.** `cancel()` on a conversation writes a `CancelRequested` for it and for every unresolved descendant, waits for the live drives to settle, and then resolves each still-unresolved link with a cancellation result WITHOUT running the result tool — an unresolved child on a CLOSED turn is unprojectable, so the parent would wedge otherwise. Cancelling one subagent leaves its siblings and its parent running.

Two rules keep that from stranding anything, and both exist because `cancel()` is a SIGNAL that a DRIVE has to consume. **Cancelling a conversation always ends it**: an unresolved subagent with no bracket (spawned, never driven) gets one opened so the cancellation has something to close — otherwise the call is a silent no-op and the parent waits forever on a conversation nobody will drive. And **the parent's drive flushes a cancelled child whose own drive is gone** (`_flush_cancelled_children`, before `_resolve_children` and inside `_settle_children`, the child half `_wind_down_async`, the hard-limit close and a subagent parent's failure close all share) — it is the only conversation that knows the child exists. That flush is skipped while the parent itself has a pending cancel, because its own wind-down owns every link and resolves them without a result tool at all. In framework mode the flush also announces each close — `SubagentFinished(CANCELLED)`; a `Finished` with no `Started` reads "cancelled before admission" — while `autostart_subagents=False` stays lifecycle-event-free.

**A child's failure never propagates.** Its turn closing is the whole signal, whatever the outcome: a child that errored, timed out or ran out of steps is a FINISHED child whose result says so.

**What is shared is keyed, not locked.** One registry, one plugin, one permission strategy serve the whole tree. State keyed by `conversation_id` needs no lock (dispatch within one conversation is sequential); deliberately-shared state locks the mutation and never the I/O. `MemoryPlugin`'s stores and `shell`'s `FileReadTracker` are both keyed — for the tracker that is a safety property, since unkeyed, one subagent's read would satisfy another's read-before-write guard.

**Middleware is conversation-aware.** Every hook receives `(session, conversation_id)`, so one instance safely serves the whole tree: a hook can route a subagent to a cheaper model, withhold tools by depth, attribute cost per conversation, or key its own state. The id is the OPERATION's scope, not exclusive ownership — a subagent's model round is the child's, the `ChildConversation` link into the parent's path is the parent's, and the runtime-minted result execution that resolves a child is the parent's too.

Limits the implementation may rely on: a subagent is never compaction-checked (bound it with `subagent_hard_max_steps` instead). `post_message` reaches subagents too: a live child accepts posts into its open turn (the spawn prompt is its FIRST user message, not its only one), a finished child rejects them, and a mid-orchestration parent accepts as well — the post wakes it, and steering (including `stop_subagent`) is exactly what the wake is for. Nesting is a real knob: `subagents_max_depth=N` means main plus N levels of subagents (default 1).

### Tool identity

`ToolSpec` (in `models.py`) is the core's ONLY tool type, and it serves two roles at once: the **advertisement** sent to the model (`name`, `description`, `input_schema`) and the **historical identity snapshot** attached to a past execution (`tool_kind`, `namespace`, `version`, `timeout_in_ms`). That conflation is a deliberate, accepted trade-off — normalization removed the storage cost that made it expensive, and having the exact schema the model was shown attached to the execution is what makes an old session auditable and replayable at all. The alternative (a thin history type and a fat wire type) was rejected: it reintroduces the split this design exists to remove and forces every registry to produce both.

Nothing in a restored spec references a live class, an `Args` model, or anything importable — it is plain data, so a session whose tools were deleted from the codebase years ago still renders its `name`, `description`, `input_schema` and `tool_kind`. `input_schema` is required and never `None`, including for a tool that takes no arguments (that case is the empty object schema `{"type": "object", "properties": {}}` — an absent schema and an empty schema mean different things to a provider). `tool_kind` defaults to `OTHER`; the network-egress kind is `WEB_FETCH`. Invocation arguments are never here — they live on `ToolExecution.raw_tool_call`.

`title: str | None` is the one PRESENTATION field, reached through
`display_name` (`title` when set, `name` otherwise). It exists so a tool whose
internal identity is a mouthful — `openai_apply_patch` — can render as "Apply
patch" without anything keying on the label: `name` stays the identity for
resolution, approvals, doom-loop and middleware. It is the only field excluded
from `spec_id()`, so giving a tool a title (or rewording one) never invalidates
a stored spec id — with the deliberate consequence that a title-only relabel
does not mint a new row and the FIRST-filed spec keeps standing.

`is_private: bool` is the ONE exception to "the advertisement sent to the model": a private spec is returned by `get_tools()` and resolved, prepared and dispatched exactly like any other, but the runner omits it from the wire list and a model tool call naming it records `NOT_FOUND` rather than resolving. Its execution never projects as a `ToolMessage` (forced — no `ToolCall` for it exists on the path), and V0's projector renders it as nothing at all. It participates in `spec_id()` like every other field.

`output_schema: dict | None` is a THIRD role, and neither of the two above: an optional declaration of the shape the tool can return in `ExecutionResult.structured_content`. It is application-facing — and, for a spawn tool, read by the runner's depth gate — but no provider accepts an output schema on a function tool, so `tool_spec_to_luca_tool` drops it like every other non-wire field, and nothing in the framework validates a payload against it. `structured_content` is likewise never projected (`content` stays the sole model-facing channel) and never counted by `calculate_context`. Both fields exist so an MCP-backed registry has somewhere to map `outputSchema` / `structuredContent`; the core reads neither. Note that `output_schema` participates in `spec_id()` like every other field — a tool that gains one is a new row.

`Tool` (in `luca/agent/contrib/tools.py`) is contrib: the ergonomic way to write a tool in Python, and the EXECUTION contract only. It declares `tool_kind`, `namespace`, `version`, `timeout_in_ms` and `is_private` as `ClassVar`s, and `get_tool_spec()` stamps them plus `input_schema=Args.model_json_schema()` into the snapshot. The optional `output_schema` ClassVar is derived the same way (`output_schema.model_json_schema()`, `None` when undeclared) — mind that the ClassVar is a model CLASS while `ToolSpec.output_schema` is the DICT, deliberately unlike `Args → input_schema`, which differ in name because they differ in type. Declaring it produces nothing: a tool that returns a payload overrides `execute()` and sets `structured_content` itself, since `_execute` is the `-> str` path. `tool()` / `tool_class()` take `output=` with the same two forms as `arguments=`, but only wire the text path — a factory-built tool can declare a schema and cannot populate a payload. `execute` / `_execute` receive the live `AgentSession` (read-only), the `conversation_id`, and the keyword-only `CancellationToken`.

There is no `get_approval_context` on the base class — it is a duck-typed convention read by `SimpleToolRegistry` (`async get_approval_context(args, session, conversation_id) -> dict`, receiving the **validated** args; `resource_permissions.ResourcePermissionToolMixin` provides it). The core never mentions it. It is awaited inside `create_execution`, on the event loop and under no deadline, so blocking work belongs in `asyncio.to_thread` — which is exactly what the mixin does with its synchronous `build_permission_requests` override point, whose path stats would otherwise stall the run on a hung mount.

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
| `subagents_enabled` | Master switch, default **False**. When off the registry withholds the spawn tool and the runner raises if one is returned anyway. |
| `subagents_max_depth` | The depth cap, default **1**: `N` lets depths `0..N-1` spawn, so the deepest subagent sits at depth `N` — main plus `N` levels. |
| `subagents_max_per_turn` | Per-conversation, per-open-turn spawn budget, default **Inf**. Spent budget withholds the spawn tool + prompt part; an overflow call inside one response is born `REFUSED`. `Inf` or `>= 1` — `0` is invalid. |
| `subagents_max_workers` | Session-wide cap on subagents doing work at once, default **Inf**. Spawns past it queue FIFO for a slot; release-on-park keeps nested trees deadlock-free. `Inf` or `>= 1` — `0` is invalid. Refused under `run(autostart_subagents=False)`. |
| `subagent_soft_max_steps` / `subagent_hard_max_steps` | Step limits for subagent conversations. `None` (the default) falls back to the main fields below — a different fact from `Inf` ("no limit"). These are what make an uncompacted subagent safe, since a child is never compaction-checked in V0. |
| `wake_parent_on_subagent_completion` | Default **True**: each resolution wakes the parent's model (a wake round). **False** batches — children still resolve as they finish, but the model runs once, after the last one; posts and ordinary tool results wake either way. |
| `hard_max_steps` | If `AssistantMessage` count in the open turn reaches this limit, the engine closes the turn `TurnOutcome.ERRORED` and returns (status → PENDING, retry-ready; no raise). |
| `soft_max_steps` | When reached and `limit_tool_choice_on_soft_max_steps_reached` is True, the next LLM call gets `tool_choice="none"`, forcing a text-only response. |
| `doom_loop_threshold` | When the same tool call (name + parameters) appears consecutively this many times in the current turn, `ToolExecution.is_doom_loop_flagged` is set True on the Nth occurrence. If `limit_tool_choice_on_doom_loop_flagged` is True, subsequent LLM calls in the same turn get `tool_choice="none"`. |

All int fields use -1 (Inf) or 0 to disable, except `subagents_max_per_turn` / `subagents_max_workers`, which reject 0 (that is `subagents_enabled=False` spelled incorrectly). Constructing a runner where `soft_max_steps == hard_max_steps > 0` emits a `UserWarning` (hard prevails). Timeout fields are milliseconds; limit fields are plain ints. `Seconds()` / `MilliSeconds()` convert durations. Seed config via `AgentSessionRunner.new_session(..., runtime_config=)`.

### Model options and credentials

`LLMConfig` says WHICH model runs (`model`, `provider`) and carries two FLAT, OPAQUE dicts saying how:

| Field | Holds | Reaches the client as |
|---|---|---|
| `model_options` | `luca.client.acompletion` keyword arguments — `max_tokens`, `temperature`, `reasoning`, `seed`, … | splatted verbatim |
| `provider_options` | `base_url`, `transport` (a dotted path to a transport class), plus any raw wire fields the provider documents | `base_url=` / `transport_class=`, remainder as `provider_options={<provider>: {...}}` |

Core never reads a key out of either. The single exception is `runner.completion_options()`, which lifts `base_url` and `transport` out of `provider_options` because those say how the client is REACHED rather than what it is asked for, and the client takes them as named parameters. `transport` is a dotted path rather than a class because an `LLMConfig` has to survive a round trip through JSON. Everything else is forwarded, so an unknown key is a `TypeError` from `acompletion`, not a validation error here — assembling a valid pair of dicts is the application's job.

**No credentials in the data model.** An `LLMConfig` is persisted with the session AND copied onto every assistant entry as provenance (see **Reasoning durability**), so a key stored there would be written to disk once per message. `api_key` is a runner constructor argument, in the same runtime-only class as `provider` — never serialized. `SummarizingContextManager` takes its own, because it makes its own model call and the application builds both.

`AgentSessionRunner(model_options=…, provider_options=…)` are runtime OVERRIDES that win per key over the session's, so a process can bound or reroute its own calls without rewriting what the session records.

`update_llm_config` derives the ACTIVE config from the CONFIGURED one and preserves both dicts untouched, and `completion_options` keys the raw block by the ACTIVE provider — so a `build_model_string` middleware that routes elsewhere carries the configured model's settings to the provider it chose. Routing to a provider that does not understand them is the routing middleware's business.

Core reads no config file. Resolving a per-provider / per-model table into those two dicts is contrib's job (`contrib/tui/config.py`: `resolve_model_options` / `apply_model_options`), and reading credentials is `contrib/tui/auth.py`.

### System prompt parts

The runner takes no `system_prompt` string. Instead it takes `system_prompt_parts` — a list whose items are any of (machinery in `luca/agent/core/system_prompt.py`):

- a `SystemPromptPart` (fields: `text`, `source`, `priority`);
- a `str` → `SystemPromptPart(text=...)`;
- a dict with `text` + optional `priority` / `source` → validated strictly;
- a callable `(session, conversation_id) -> ` any of the above **or `None`** (meaning "contribute nothing"), invoked fresh before every LLM call.

Static parts are coerced eagerly at construction (`coerce_system_prompt_part` — a bad part raises `TypeError`/`ValidationError` at `__init__`); callables resolve per call and their return value is coerced the same way.

Before every LLM call, `build_system_message()`:
1. Resolves the parts (callables get the live `AgentSession` and the id of the conversation the prompt is for; a `None` return is dropped).
2. Sorts parts by `priority` (ascending).
3. Assembles via `SystemPromptAssembler.assemble_system_prompt(parts) -> str`; if blank, sends no system message.

The assembler is optional and duck-typed (concrete base, no ABC — override the one hook); `DefaultSystemPromptAssembler` newline-joins the part texts. A runner with no parts sends no system message.

`ConversationRuntimeStatus` (in `models.py`) carries:
- `step_count` — `AssistantMessage` count in the open turn.
- `turn_count` — conversational `TurnStart` count on that path (compaction brackets excluded).
- `status` — the derived `ConversationStatus`.

It is always recomputed via `AgentSession.get_conversation_status(conversation_id)` — the only door, and never a stored field.

### Reasoning durability

`AssistantMessage.parts` retains `ThinkingContent`, so reasoning is durable in the saved session and survives reload — text, `id`, `signature` and `redacted` alike. `id` is the provider's identity for the reasoning ITEM (OpenAI's `rs_…`); `signature` its attestation over the content (Anthropic's signature, OpenAI's `encrypted_content`). The Responses API needs both to replay an item.

Whether it goes back on the wire is the transport's call, and it turns on TWO facts. First the protocol: chat-completions hosts have no replay surface and drop reasoning outright, while Anthropic and the OpenAI Responses API replay the block verbatim. Second the producing model: an attestation is minted by one (provider, model) pair and refused by every other, so `ConversationProjector.project_assistant_message` copies `entry.llm_config`'s provider and model onto the projected client message and the transport drops any block whose provenance disagrees with the model being called (`BaseTransport._attestation_is_replayable`). That path is live whenever an application switches models mid-session and replays the whole history to the new one. For the same reason `AssistantMessage.llm_config` records the ACTIVE model (`session.llm_config`, stamped by the drive via `session.update_llm_config()` after `build_model_string` middleware ran — the CONFIGURED value stays on `session.session_config.llm_config`), not the configured one: a `build_model_string` middleware may route a turn elsewhere, and recording the configured value would make provenance lie.

A `ThinkingContent` is therefore immutable once persisted; rewriting `thinking` in middleware invalidates the signature and the provider will reject the turn.

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
- Override `async _execute(args, session, conversation_id, *, cancellation_token) -> str` for simple tools, or `async execute(args, session, conversation_id, *, cancellation_token) -> ExecutionResult` for rich output (`is_error`, `metadata`, multi-block). An `is_error=True` result still records `COMPLETED` — `is_error` is the tool's own verdict, not a lifecycle fact.
- Both receive the LIVE `AgentSession` (read `session.id` / `session.session_config.llm_config`; treat it as READ-ONLY — the runner owns every write), the `conversation_id` the call belongs to (key any per-conversation state by it — a tool instance is shared by every conversation), and the keyword-only `CancellationToken`. Per-run application state is not the framework's concern: a tool is application code and can close over its own references or read a `contextvars.ContextVar`.
- `SimpleToolRegistry` validates LLM-produced arguments through `Args` at birth (for the birth status) and again in `prepare()` (for dispatch). Malformed args become a terminal `INVALID` execution with a structured `ToolExecutionError` and never reach `decide()`.
- Timing is recorded by the runner on the execution (`started_at`/`ended_at`), never on the result. `timeout_in_ms` bounds the BODY only.
- Define `async get_approval_context(args, session, conversation_id) -> dict` (the duck-typed convention) to describe the call for `SimpleToolRegistry`'s permission strategy — "a subagent is asking for this" belongs here.
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

Write a plain Python class that implements any of the 13 hook methods defined in `luca/agent/core/middleware.py` (`AgentMiddlewareMixin` — its hooks are identity pass-throughs, so subclassing is safe for partial overrides, but plain classes are the recommended style; the runner dispatches via `hasattr`). Pass instances as `middleware=[mw1, mw2]` to `AgentSessionRunner`. Hooks run in list order; there is no reverse ordering. There is deliberately NO `build_messages` hook — history policy belongs on the `ConversationProjector`.

**Every hook starts with `(session, conversation_id)`** — the live session the runner writes through, plus the conversation whose OPERATION invoked the hook. One instance serves the main conversation and every subagent concurrently, so that id is what makes per-conversation routing, state and attribution possible at all; it does NOT assert the value belongs exclusively to that conversation. Same prefix as `ToolRegistry`, `ContextManager.compact`, `Tool.execute` and system-prompt callables.

The tool lifecycle has FOUR hooks answering four different questions. Creation: `before_tool_creation(call) -> call` (immediately before `ToolRegistry.create_execution`; the returned call is what the registry sees, and the private-name check reads the EFFECTIVE name) and `after_tool_creation(execution, exception=None) -> execution` (the finished birth state before it is committed — the returned status decides the next lifecycle branch, so terminalizing here skips `decide()` entirely; also fires for every framework-synthesized draft, with the live exception where one exists). Dispatch: `before_tool_execution(execution) -> execution` — **dispatch attempts ONLY**; its `raw_tool_call` is the effective call, which is what `prepare()` then resolves and validates from. Outcome: `after_tool_execution(execution, exception=None) -> execution` — every terminal outcome, dispatched or not; the returned execution is what gets persisted.

`before_tool_execution` does NOT fire for terminal-at-birth, REFUSED, REJECTED, cancelled-before-dispatch, or orphan recovery. That asymmetry is the point: the before hook describes dispatch, the after hook is the universal outcome point. There is consequently ONE outcome tail (`_finalize_outcome`) rather than a dispatched/undispatched pair.

**Only `before_tool_execution` has an exactly-once, paired guarantee**, and it is per DISPATCH ATTEMPT, not per call for all time — a crash during `prepare()` writes nothing, so the next drive fires it again over the original call. Every other hook may fire without its result being persisted; the visible case is `before_permission_check`, whose returned execution is discarded when a cancellation lands while `decide()` is in flight (the execution must stay PENDING for the wind-down, so there is nowhere to put it). `build_tool_list` receives and returns `ToolSpec`s — private specs already dropped, adapter conversion after the hook, in `_collect_tools`.

Two write scopes are not what you would guess: a COMPACTION transition writes its plan's new entries under the OUTGOING conversation (the destination id is minted inside `transition_conversation` and does not exist yet), and `recalculate_context_tokens()` runs NO middleware at all — it rewrites every entry across every conversation, so no single id would be honest.

Tests go in `tests/agent/test_runner_middleware.py`, plus `tests/agent/subagents/test_middleware_scope.py` for anything about conversation scope across a tree. See `docs/agent/07-middleware.md` for the full hook catalogue. The doc embeds the mixin's full source — keep it in sync when the mixin changes.

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
1a. `RECEIVED` executions → `_birth_executions`: THE `create_execution` call site. Re-resolves the tool specs for the private-name check (a resumed birth has no model round behind it to inherit them from), gathers one draft per unborn execution, then folds each into its entry SEQUENTIALLY — the spawn-budget refusal is applied in that fold and has to see calls 1…k-1 as in-flight reservations. Emits `ToolCallReceived` per execution; a terminal birth runs the outcome pipeline on the spot.
1. Undecided executions (`approval_status` None/PENDING) → `asyncio.gather` `_decide_one` over them; each races the WHOLE `_decide_with_middleware` step (hook → `tool_registry.decide()` → hook) against the token, and a lost race records nothing for that execution. Each surviving response updates `approval_status` + appends to the audit log; DENY → terminal `REJECTED` now (outcome pipeline + `ToolExecuted`).
2. Ready executions (`PENDING`+`ALLOWED`) → `_dispatch_batch` (sequential by choice — the state model is parallel-ready; a token already tripped when an execution's turn comes up stops the batch, leaving it and its successors untouched for the wind-down).
2b. `_spawn_children`: a completed execution whose `structured_content` says `is_subagent_spawn` becomes a child conversation (seeded with the prompt) plus a `ChildConversation` link. Idempotent across a reload for free — the link IS the record that the spawn was handled. `_start_children` runs BEFORE the `SubagentsSpawned` yield, because a yield suspends the generator and `run.child(cid)` must resolve inside that event's branch; it adopts every child deferred and asks the worker pool for a slot per child. In framework mode (`autostart_subagents=True`) the announcement and every admission's `SubagentStarted` are published onto the parent run's OWN inbox, synchronously and in order, before any child task first runs — the fan-in drains the inbox before the engine on every pull, so this is what guarantees `Spawned ≺ Started(child) ≺ that child's own events` on the stream. Under `False` the announcement stays an engine yield: it is the transfer of control to the application, and no lifecycle events exist.
2b'. `_stop_children` — THE STOP HANDSHAKE: every completed open-turn execution carrying an `is_subagent_stop` payload cancels the matching DIRECT child. Matched by spawn-payload `task_id` among links that PRECEDE the stop execution on the path (an old signal can never kill a later task reusing the id); a resolved / finished / already-cancelling target is a consumed signal, which is what makes reload replays no-ops. A payload with no `task_id` raises — the runner validates the keys it acts on, like the spawn payload's.
2c. `_resolve_children`: every child whose turn bracket has CLOSED — whatever the outcome — is turned into a result by the private result tool named in the payload, which lands on its link together with `result_execution_id` (the execution's path position is where the projector renders the update).
3. Any execution still `approval_status=PENDING` → cancel check, then announce (`ApprovalRequired` — once per gate: a drive-local announced-id set, so a re-park at the same gate never repeats the event while a NEW gate minted by a 3b fall-through round is announced), then park via `_await_subtree` — unless 3b applies.
3b. A POST REACHES THE MODEL PAST THE GATE (0008): with an unseen user post (`_has_unseen_post` — the durable `open_turn_unseen_post` plus the drive-local `last_seen` fingerprint), the drive falls through to step 4 instead of parking, deliberately PAST the progress-`continue` (`undecided` holds the gate on every pass, so continuing would re-ask `decide()` in a tight loop). ONLY a post does this — an allowed sibling's result completed in the same round is not a wake. The close site refuses to close COMPLETED while the gate is live (`has_awaiting_approval` is its third term), and both failure closes (`hard_max_steps`, an LLM failure) settle gated executions first via `_settle_undispatched` — no close leaves a nonterminal execution behind.
3c. Otherwise, if the open turn still has an unresolved child AND no unseen material (`open_turn_unseen_material`, plus the drive-local `seen` fingerprint for a post that landed under this drive's own in-flight LLM call) → `_await_subtree`; with material the drive falls through to step 4 and calls the model — the wake round. Both park sites release the conversation's worker-pool slot on entry and re-acquire on wake (raced against the token) — parked is not working. Both obey ONE rule: **a drive returns only when nothing in its SUBTREE can advance**. A gated child with no subtree returns (its drive is gone; `notify()` or the next `run()` restarts it); a gated parent whose children still work waits — which is why `ApprovalRequired` is not terminal on a parent's stream.
4. Step-limit / doom-loop checks → `prepare_llm_call()` (`before_llm_call`) → `_collect_tools()` (raced; a lost race skips the LLM call and returns to the loop top) → model call (reached when every execution is terminal, or through 3b's fall-through with a gated execution still live and projecting its placeholder).
   - `hard_max_steps` reached → `_close_turn(ERRORED)` and return (no raise). With unresolved children: cascade-cancel them, `_settle_children`, re-check for a cancel that landed during the settle's awaits (it controls the close), then ERRORED — no close ever leaves an unresolved child.
   - `soft_max_steps` reached, or a doom-loop-flagged execution exists → `tool_choice="none"`.
   - Race the cancellation token via `_race_cancellation` (grace window, hard-kill via `_kill`), wired to `RuntimeConfig`'s `timeout=`/`total_timeout=`.
   - `TimeoutError` → `TurnFinish(TIMED_OUT)` and re-raise; any other failure → `ERRORED` (status PENDING, retry-ready); cancel pending → wins (wind-down, normal return). With unresolved children the close splits by depth: the MAIN conversation leaves the turn OPEN and re-raises (children keep working; the next `run()` resumes), a SUBAGENT parent cascade-cancels + settles its children and closes ERRORED (nothing outside it ever retries a subagent mid-run).
   - THE ROUND KEYS OFF `tool_calls`, NOT the finish reason — a misclassifying provider can neither wedge the conversation ("stop" + calls) nor loop it ("tool_use" + none). That settles "stop" vs "tool_use" and says nothing about a model that produced NEITHER: no calls plus a finish reason in `NON_ANSWER_FINISH_REASONS` (`{"error", "length"}` — the client's vocabulary is already normalized, so OpenAI's `length` and Anthropic's/Bedrock's `max_tokens` both arrive as `"length"`, and every refusal / safety filter / guardrail as `"error"` with an `error_message`) closes `ERRORED` and raises `IncompleteResponseError`, carrying `message.error_message` when the transport supplied one. It sits LAST in the close-site precedence chain, so it only ever converts what would otherwise have closed COMPLETED — a cancel, an unseen post, an unresolved child and a live gate all still win, unchanged. Unlike a transport failure it **keeps** the partial assistant message (`_record_assistant` already ran): those tokens were really produced, and on a truncation they are the useful half. Settles undispatched executions before closing, like every other failure close.
   - Recording the assistant message, RECEIVING its executions, and closing a final-answer bracket is **atomic**, and atomic here means **no `await` between**, not merely no yield. This is the invariant the whole `RECEIVED` state exists to serve: no append — a `post_message` from an application's UI on the same event loop, most of all — can ever land between a `tool_call` and the execution nodes that answer it, so the path is always projectable. Birth is deliberately outside that block (it is async and application-owned) and runs as step 1a against the durable entries this wrote.

**Per-step methods:**

| Method | Responsibility |
|--------|---------------|
| `resolve_tool_specs` | `get_tools()` for one conversation — the RUNTIME's view, private specs included |
| `build_tool_list` | The MODEL-VISIBLE catalog: private specs dropped, then the `build_tool_list` middleware hook. Returns `ToolSpec`s; `_collect_tools` adapts them onto the wire afterwards. Synchronous |
| `build_system_message` | Resolve `system_prompt_parts` (callables get `(session, conversation_id)`; `None` returns are dropped) → sort by priority → assemble; blank → no system message |
| `build_messages` | Delegates to `conversation_projector.project()` — no middleware stage |
| `_record_assistant` | Converts message to parts via `adapter.message_to_parts` |
| `_receive_executions` | SYNCHRONOUS, inside the assistant-message block: one `RECEIVED` execution appended per tool call in model-request order, carrying only what the runner knows then (identity, the deep-copied `raw_tool_call`, `is_doom_loop_flagged` evaluated in append order). No registry, no await — this is the transaction that keeps the path projectable |
| `_birth_executions` / `_birth_draft` | Set-oriented birth as drive step 1a: gather `tool_registry.create_execution(session, deep-copied call)` per unborn execution (concurrent, each raced against the token INDIVIDUALLY — never the gather), then a SEQUENTIAL fold of each draft into its existing entry (the spawn-budget refusal rides in that fold); a raising `create_execution` (or `None` registry) synthesizes the draft (`FAILED`/`NOT_FOUND`), a lost race synthesizes a PENDING one, isolated per call; terminal births run the outcome middleware pair. Folds through `_persist_entry`, not `_persist_execution`: a birth completes an entry's creation rather than mutating it, so `updated_at` stays `None` |
| `_dispatch_one` / `_prepare_tool` / `_run_tool_body` | `before_tool_execution` (its `raw_tool_call` is the effective call) → `tool_registry.prepare(session, execution)` raced (grace 0) → on raise/non-callable: terminal with `phase="prepare"`, NO `RUNNING`, NO `ToolExecutionStarted` → on cancel: `CANCELLED` in place → else persist `RUNNING`+`started_at` (birth `tool_spec` stands) → `ToolExecutionStarted` → invoke the callable (the `ensure_future` INSIDE the failure handling) under token race + outside deadline (`tool_spec.timeout_in_ms`, else config) → `COMPLETED`/`FAILED`/`INTERRUPTED`/`TIMED_OUT`; a mid-body cancel persists `cancel_signalled_at` before the grace wait |
| `_finalize_outcome` | The ONE outcome tail, for every terminal execution whether or not it dispatched: recalculate context → `after_tool_execution(execution, exception)` → persist → `ToolExecuted` built from the projector |
| `to_tool_execution_error` | PUBLIC override point: live exception → durable `ToolExecutionError` (type + message; pydantic errors under `details.errors`; `details.phase` is a FACT passed by the call site — `create_execution` / `prepare` / `execution` — not inferred from `started_at`) |
| `recalculate_context_tokens` | PUBLIC: re-derive `context_tokens` for every entry, straight to `ledger.refresh_entry`. Runs NO middleware — it spans every conversation, so no id would scope it. Nothing in the framework calls it |
| `_is_doom_loop(tc)` | Compares last `threshold-1` `ToolExecution`s in the open turn against the incoming call's `raw_tool_call` name + arguments |
| `_close_turn(outcome, error)` | The only `TurnFinish` writer |

All entry appends and entry-derived queries (open turn, pending/undecided/awaiting/ready/running executions, unconsumed cancel, doom-loop flag, derived status) are delegated to `SessionLedger` (`ledger.py`) — one append path so parent links, path, and `updated_at` cannot drift. The ledger is also the only door onto the usage store (`record_usage`), the only path-replacement write (`prune`), and the only writer of `tool_specs`. Every `ToolExecution` persistence — creation AND every update — passes through `before_entry_written` (updates via the runner's `_persist_execution`, stored by `ledger.put_entry`).

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
| `tests/agent/test_runner_failures.py` | The prepare/dispatch outcome table (every `prepare()` failure mode, the non-callable and non-awaitable guards, `started_at`/`dispatched`/`details["phase"]`), tool deadlines and their scope, crash recovery (orphaned RUNNING), LLM failure closes |
| `tests/agent/test_runner_post_message.py` | The `post_message` acceptance matrix (accept/reject per state — mid-orchestration accepts; the dedicated CANCELLING exception, the archived/finished rejections) + the mid-turn stories: the close-site unseen-message check and its extra round, burial and its projection, precedence (cancel, `hard_max_steps`), queued trailing messages, the birth-window regression (a post landing while the registry is being consulted lands BEHIND the whole tool round, never between a `tool_call` and its result), and the 0008 gate stories (a post while blocked answered past the gate then re-parked; the completed-sibling misfire guard; the hot-loop and close guards; `hard_max_steps` settling a live gate; cancel beating the unseen post at the gate) |
| `tests/agent/test_runner_projector.py` | The runner ↔ `ConversationProjector` seam: wire history, event/wire agreement, equality |
| `tests/agent/test_runner_context.py` | The runner ↔ `ContextManager` seam: context stamping, middleware final say on write paths, processed tool output (session/event/wire agreement), prune-machinery composition, and `recalculate_context_tokens()` (every entry, archived and off-path included; construction changes nothing) |
| `tests/agent/test_context_manager.py` | Default `ContextManager`: per-type context ownership, prune templates + refusals, identity tool-output, subclass overrides, model-aware counting, the non-membership of an entry during an append, and the compaction pair's base behavior (declines / raises) |
| `tests/agent/test_runner_system_prompt.py` | `system_prompt_parts` forms (str / dict / part / callable) + assembler (callable parts receive `(session, conversation_id)`) |
| `tests/agent/test_runner_limits.py` | Hard/soft `max_steps`, doom-loop flagging, `tool_choice` restriction |
| `tests/agent/test_private_tools.py` | `ToolSpec.is_private` end to end: the spec, the wire filter, the NOT_FOUND refusal of a guessed name, and the projection rules |
| `tests/agent/test_runner_tree.py` | The `AgentRun` TREE as machinery: `children`/`child()`, event fan-in, the approvals stream, `notify()`, suspend cascading — no subagent tools |
| `tests/agent/subagents/` | The end-to-end subagent stories (`test_spawn_handshake.py`, `test_parallel_and_control.py`, `test_post_messages.py`, `test_wake_rounds.py`, `test_stop_subagents.py`, `test_middleware_scope.py`): spawn, gate, parallelism, approvals, cancellation, resume, wake rounds per resolution (and their `wake_parent_on_subagent_completion=False` batching stories), mid-orchestration steering posts, the stop handshake and its position bound, and the middleware conversation-scope contract (one instance, distinct ids per conversation) — core + `contrib.subagents` only |
| `tests/agent/contrib/test_subagents.py` | Self-scoped contrib tests: the four tools' specs, payloads and verdicts, the gate predicate, the two prompt parts, the control-tool withholding, the plugin surface — no runner |
| `tests/agent/test_integration_full_stack.py` | ONE composed application (shell + memory + subagents + permissions, real tools, scripted transport) driven through every consumption form, ending in a declarative assertion over the whole catalog. A smoke test — it proves the pieces compose, not that any one is correct |
| `tests/agent/test_child_conversation.py` | `ChildConversation` as pure DATA: projection (the task update at the result execution's position; the link renders only a no-execution resolution; unresolved fails loud outside the open turn), context accounting, `pretty_print`, round-trip |
| `tests/agent/test_models.py` | The data model as pure Pydantic: entry shapes and defaults, `ToolSpec.spec_id()` (stability, key-order independence, distinct ids for distinct content), tool-spec normalization end to end (one row per distinct spec, dump strips / load restores by reference, standalone execution dumps stay inline, both load guards raise) |
| `tests/agent/test_ledger.py` | Entry-derived query matrix (status × approval subsets), the `record_usage` / `put_entry` / `transition_conversation` / `refresh_entry` doors, tool-spec filing at every door (one row per distinct spec, recompute-always, all three `transition_conversation` lists), `open_compaction_entry`, the derive-status skip matrix |
| `tests/agent/test_compaction.py` | The compaction VOCABULARY as a unit: `validate_plan` (every rejection), `has_content`, `check_snapshot`, the plan value objects — no runner, no ledger, nothing async |
| `tests/agent/test_runner_compaction.py` | The drive: brackets, events, the transition, failures by source, cancels, resumes (G6), statuses, `RunResult` |
| `tests/agent/test_projection.py` | `ConversationProjector`: every entry type, every terminal tool status, fail-loud rules, subclass override points |
| `tests/agent/test_adapter.py` | Inbound message parts + `tool_spec_to_luca_tool` (the `input_schema` → `parameters` pass-through) |
| `tests/agent/test_utils.py` | `pretty_print`: whole-transcript assertions per session shape (answered turn, tool tree, failure, open turn, compaction/pruning, clipping) |
| `tests/agent/test_runner_middleware.py` | Middleware hook dispatch: the thirteen hooks, the `(session, conversation_id)` prefix, the four-hook tool lifecycle (creation pair, dispatch-only `before_tool_execution`, universal `after_tool_execution`), chaining + exception context, `after_llm_response` exactly-once across streaming/non-streaming, and `recalculate_context_tokens()` firing nothing |
| `tests/agent/contrib/test_tools.py` | Self-scoped contrib tests: the `Tool` base contract (spec stamping incl. `input_schema`, `output_schema` and `timeout_in_ms`, session + token pass-through, a result carrying `structured_content`) and the working `tool()` / `tool_class()` factories (incl. `output=` in both forms and its `class_attrs` collision) |
| `tests/agent/contrib/test_simple_tool_registry.py` | Self-scoped contrib tests: birth drafts per preflight outcome, decide delegation, `prepare` returning a callable WITHOUT running the body and its two raise paths, `ProxyToolRegistry` routing/nesting and cache-independent `decide`/`prepare` on a never-warmed route — no runner |
| `tests/agent/contrib/test_plugins.py` | Self-scoped contrib tests: `PluginAgentSessionRunner` composition (one proxy, parts/middleware flattening, equality with a directly-configured runner) |
| `tests/agent/contrib/test_resource_permissions.py` | Self-scoped contrib tests: `PermissionStrategy` decide / apply_answer / pending_requests / grant + the tool mixin — no runner, no session |
| `tests/agent/contrib/shell/` | Self-scoped contrib tests: one file per shell tool (`tools/test_<name>.py`), `native/` (the per-model support table, the four native tools, the middleware's own tables) + `test_plugin.py` (`ShellAccessPlugin` wiring, seeded rules, decide/pending flows) — no runner |
| `tests/agent/test_native_tools/` | The provider-native BATTERY: given entries and an active config, the exact `(tools, messages)` `acompletion()` receives — advertisement per mode, adoption, projection across provider switches, the denied-shell synthesis, the no-plugin fail-safe. Real `ShellNativeMiddleware`, fixture tools |
| `tests/agent/contrib/test_memory.py` | Self-scoped contrib tests: `MemoryPlugin` surface + scratchpad / todo-list behavior — no runner |
| `tests/agent/contrib/test_skills.py` | Self-scoped contrib tests: frontmatter parsing (incl. the `>` / `\|` block scalars real skills use), the skip-don't-raise rules, location precedence, the `skill` tool, the plugin surface — no runner |
| `tests/agent/contrib/test_simple_context_manager.py` | Self-scoped contrib tests: `SummarizingContextManager` — the context gauge, the split strategies, and the `CompactionPlan` it returns (via `FauxProvider`); no runner |
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
| Subagents (the spawn contract, the tree of runs, the two loops) | `luca/agent/contrib/subagents/` + the Subagents section above + `docs/agent/13-subagents.md` |
| Where does this responsibility belong | `runner.py`, `projection.py`, `adapter.py`, and their tests under `tests/agent/` |
