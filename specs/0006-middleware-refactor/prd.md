# PRD: Conversation-aware middleware

Luca middleware is a set of explicit, synchronous transformation hooks at the
agent runtime's important lifecycle boundaries. It lets applications inspect
and replace model inputs, model outputs, messages, durable entries, tool
catalogs, tool calls, approval decisions, and tool outcomes.

Every hook receives the live `AgentSession` and the concrete
`conversation_id` whose operation invoked it. This makes the same middleware
usable for the main conversation, parallel subagents, and future conversation
types without relying on an implicit "active conversation."

The target API contains twelve hooks:

- Ten lifecycle hooks carried forward with conversation-aware signatures.
- Two tool-creation hooks around `ToolRegistry.create_execution()`.
- `before_tool_execution` is defined specifically as the start of a dispatch
  attempt.

Middleware is a trusted extension surface. Luca threads each returned value
directly into the next middleware and then into the runtime. It does not
validate, coerce, repair, or restrict middleware output.

## Why this change

The data model grew a second axis. An `AgentSession` now holds several
conversations — the main one plus parallel subagents — advancing
concurrently, and one middleware instance is shared by the whole tree. Every
other framework seam already adapted: `ToolRegistry`'s four methods,
`ContextManager`'s compaction pair, `SessionLedger`'s doors, and system-prompt
callables all take `(session, conversation_id)` or a `conversation_id`.

Middleware did not. No hook receives a conversation, so a hook cannot tell
which conversation an entry, a tool call, or a model round belongs to. Nothing
in `luca/` implements a middleware hook, so this is a missing capability
rather than a live bug — an application that ships one and assumed a single
conversation gets no error, just wrong behavior.

Two smaller boundaries went stale alongside it: the tool lifecycle grew a
registry-owned birth phase (`create_execution`) with no hook around it, and
`before_tool_execution` still fires for calls that will never be dispatched.

## Product context

A middleware instance is shared by the runner and may observe calls from the
main conversation and multiple subagents during the same run.

Middleware needs both pieces of scope on every invocation:

- `session` gives access to the complete, current framework state.
- `conversation_id` identifies the operation being performed and lets the
  middleware resolve `session.conversations[conversation_id]` when needed.

The identifier describes the **operation's scope**, not exclusive ownership of
an entry. An entry may be referenced by more than one conversation. Every
write path that runs middleware must propagate the conversation which caused
the write.

The `(session, conversation_id)` prefix is the framework-wide signature
convention, not a middleware-specific one. It matches `ToolRegistry`,
`ContextManager.compact`, `Tool.execute`, and system-prompt callables, so one
argument order describes every extension seam.

## Goals

1. Make every middleware hook safe to use in a multi-conversation session.
2. Expose the agent's meaningful transformation boundaries at the right type —
   the framework's own vocabulary where one exists.
3. Give application authors authority to replace values, not merely observe
   them.
4. Define exact invocation order and streaming behavior.
5. Cover the full tool lifecycle: creation, approval, dispatch, and outcome.
6. Keep model selection and tool-catalog construction independently
   extensible from the act of issuing a model call.
7. Preserve simple composition through optional, duck-typed methods.

## Non-goals

- **No provider-neutral model-message DTOs.** `before_llm_call` and
  `after_llm_response` keep `luca.client` message types. See "Type binding"
  below.
- **No new types of any kind.** This change adds arguments and two hooks; it
  introduces no class.
- **No compatibility shim.** No signature introspection, no dual dispatch, no
  deprecation path. V1 is unreleased — old signatures are replaced outright.
- **No projection middleware.** History shaping, redaction, and tool-output
  wording stay on `ConversationProjector`. `before_llm_call` remains the
  last-mile hook downstream of it.
- **No defensive programming around middleware.** Luca does not validate,
  clone, restore, or repair anything a hook returns.

## Middleware contract

### Shape

`AgentMiddlewareMixin` remains a reference implementation with identity
methods. Subclassing it is optional. Any object implementing one or more hook
names is valid middleware.

Hooks are synchronous. This keeps every persistence path synchronous and
preserves the runner's atomic write regions. The same rule applies to all
hooks so middleware composition has one execution model.

Every signature starts with:

```python
self,
session: AgentSession,
conversation_id: str,
```

The remaining arguments describe the value being transformed and any
read-only context for that transformation.

### Composition

Middleware instances run in the order supplied to `AgentSessionRunner`.
Before and after hooks both use that same order.

For a transformed value `v` and middleware `[a, b, c]`, Luca evaluates:

```text
v1 = a.hook(session, conversation_id, v, ...)
v2 = b.hook(session, conversation_id, v1, ...)
v3 = c.hook(session, conversation_id, v2, ...)
```

The final return value becomes the runtime's effective value. Context
arguments, including a live exception, are passed unchanged to each
middleware in the chain.

For hooks returning multiple values, the returned tuple is the transformed
value and is unpacked into the next middleware invocation.

### Trust and failure model

Middleware owns the consequences of what it returns.

- Luca performs no middleware-specific type check or same-class check.
- Luca does not restore ids, statuses, timestamps, tool arguments, results,
  approval data, or other fields changed by middleware.
- Luca does not clone a returned value to protect framework state.
- An exception raised by middleware propagates through the operation like any
  other application exception.
- Normal downstream code consumes the returned value as supplied. Any
  resulting behavior or failure belongs to the application.

Type annotations document the standard pipeline shape; they are not a
restriction on the power of middleware.

### Concurrency

One middleware instance may serve several conversations concurrently. The
runtime does not serialize middleware globally. Middleware which stores state
on itself is responsible for scoping that state and for any synchronization it
requires. State keyed by `conversation_id` needs no lock — the same rule the
tool registry and the shipped plugins already follow.

The `AgentSession` argument is the live session object. Middleware may inspect
the state present at invocation time. The runner remains the owner of durable
session writes; middleware changes state by returning transformed pipeline
values.

## Type binding

Middleware hooks are defined by **what a boundary carries**, not by a
particular Python class. Where Luca's own vocabulary describes the value, the
hook uses it; where the value simply *is* the client's, the hook uses the
client's type.

- **Tools** — `build_tool_list` operates on `ToolSpec`, the core's own tool
  type. `ToolSpec` is what `ToolRegistry.get_tools()` returns and what
  `ToolExecution` stores; the wire `luca.client.Tool` is a lossy projection of
  it (it drops `tool_kind`, `namespace`, `is_private`, `output_schema` and
  `metadata`, so a middleware handed the wire list cannot filter on any of
  them). Client tool DTOs are created by the adapter after the hook.
- **Model messages** — `before_llm_call` and `after_llm_response` operate on
  `luca.client` message types. Conceptually the boundary carries "the messages
  about to go to the model" and "the assistant message that came back"; in
  this implementation those are the client's messages, exactly as
  `ConversationProjector` already produces them (`projection.py`: *"The
  projector targets canonical `luca.client` DTOs and stops there"*). A Luca
  implementation in another language binds the same two concepts to whatever
  its client calls a message.

Introducing core-owned mirrors of the client message types was considered and
rejected — see "Historical design decisions".

Model call parameters remain explicit. Luca does not aggregate them into a
request object solely for middleware.

## Public hook specification

The reference mixin must expose the following signatures and identity
implementations:

```python
class AgentMiddlewareMixin:
    def build_model_string(
        self,
        session: AgentSession,
        conversation_id: str,
        model_string: str,
        llm_cfg: LLMConfig,
    ) -> str:
        return model_string

    def build_tool_list(
        self,
        session: AgentSession,
        conversation_id: str,
        tools: list[ToolSpec],
    ) -> list[ToolSpec]:
        return tools

    def before_post_message(
        self,
        session: AgentSession,
        conversation_id: str,
        parts: list[ContentPart],
    ) -> list[ContentPart]:
        return parts

    def before_entry_written(
        self,
        session: AgentSession,
        conversation_id: str,
        entry: AnyEntry,
    ) -> AnyEntry:
        return entry

    def before_llm_call(
        self,
        session: AgentSession,
        conversation_id: str,
        messages: list[Message],
        system_message: str | None,
    ) -> tuple[list[Message], str | None]:
        return messages, system_message

    def after_llm_response(
        self,
        session: AgentSession,
        conversation_id: str,
        message: ClientAssistantMessage,
    ) -> ClientAssistantMessage:
        return message

    def before_tool_creation(
        self,
        session: AgentSession,
        conversation_id: str,
        call: ToolCall,
    ) -> ToolCall:
        return call

    def after_tool_creation(
        self,
        session: AgentSession,
        conversation_id: str,
        execution: ToolExecution,
        exception: Exception | None = None,
    ) -> ToolExecution:
        return execution

    def before_permission_check(
        self,
        session: AgentSession,
        conversation_id: str,
        execution: ToolExecution,
    ) -> ToolExecution:
        return execution

    def after_permission_decision(
        self,
        session: AgentSession,
        conversation_id: str,
        decision: ApprovalDecision,
        execution: ToolExecution,
    ) -> ApprovalDecision:
        return decision

    def before_tool_execution(
        self,
        session: AgentSession,
        conversation_id: str,
        execution: ToolExecution,
    ) -> ToolExecution:
        return execution

    def after_tool_execution(
        self,
        session: AgentSession,
        conversation_id: str,
        execution: ToolExecution,
        exception: Exception | None = None,
    ) -> ToolExecution:
        return execution
```

`Message` and `ClientAssistantMessage` are `luca.client.types.messages.Message`
and `AssistantMessage`. Every other name is a core type.

## Hook semantics

### `build_model_string`

Builds the effective model identifier from the session's `LLMConfig`. It is a
general model-selection extension point and may be called independently of an
LLM invocation. Applications can use it for per-conversation routing,
fallback aliases, provider prefixes, or model policy.

The returned string is the value passed to the model adapter and is the basis
for the effective model provenance recorded on the assistant response
(`runner.effective_llm_config`).

### `build_tool_list`

Builds the tool catalog offered to a model, as `ToolSpec`s. It receives the
model-visible specs after private definitions (`ToolSpec.is_private`) have
been excluded and before any conversion to client tool DTOs.

The hook may add, remove, replace, or reorder specs. The returned list is
adapted for the model transport by `adapter.tool_spec_to_luca_tool`. Runtime
tool resolution remains the responsibility of `ToolRegistry`; Luca does not
require the returned catalog to match what that registry can execute.

Because the private filter runs *before* the hook, a middleware that adds a
spec with `is_private=True` puts it on the wire. That is the trust model
working as specified, not a gap to close.

Tool-catalog construction is an independently callable framework operation.
Invoking it does not mean an LLM call will be issued.

### `before_post_message`

Runs after Luca resolves the target conversation and validates that it accepts
the message, and before a `UserMessage` is appended. It receives the complete
ordered content-part list and may add, remove, replace, or reorder parts.

The returned parts are used to construct the durable user entry, which then
passes through `before_entry_written`.

The `conversation_id` is the RESOLVED target — the explicit
`post_message(conversation_id=...)` argument, or the main conversation when it
was omitted. It is never `None`.

### `before_entry_written`

Runs before every durable entry write through the session ledger, including:

- New messages and lifecycle markers.
- Initial `ToolExecution` records and every later execution update.
- Compaction and child-conversation records and their mutable updates.
- Entries built for a compaction plan's new conversation.

Context calculation and runner-owned field construction happen before this
hook. Its returned entry is the entry committed by the ledger. Luca does not
recalculate or restore fields after the hook.

The supplied `conversation_id` is the scope of the operation that caused the
write. It does not assert that the entry is referenced exclusively by that
conversation.

Two scope rules the implementation must follow:

- **A compaction transition writes under the OUTGOING conversation.** The
  entries a `CompactionPlan` creates belong to a conversation whose id is
  minted inside `SessionLedger.transition_conversation` and does not exist
  when the entries are built. The conversation that caused the write — the one
  being compacted — is the scope.
- **`recalculate_context_tokens()` runs no middleware at all.** It rewrites
  every entry in `session.entries` across every conversation, so no single id
  describes the operation. It is an operational refresh: recalculate, store,
  done. This is a change from today, where it threads each entry through
  `before_entry_written`.

### `before_llm_call`

Runs once for each model invocation attempt, after conversation projection and
system-prompt assembly and immediately before the client request is issued.

It retains the explicit `(messages, system_message)` interface. The hook may
rewrite either value and returns the pair consumed by the client. Tool catalog
and model selection stay in their dedicated build hooks.

The hook applies to model calls issued by `AgentSessionRunner`. A collaborator
which internally calls a model, such as a custom `ContextManager` summarizing
for compaction, owns that call and its extension surface.

### `after_llm_response`

Runs once after a successful complete model response, before the durable
assistant entry is written.

It runs for both tool-call responses and final-answer responses. Its return
value supplies the content and tool calls the runner records and acts upon.

For a streaming completion, the stream is consumed to its terminal assistant
message first. The hook then runs exactly once on that complete message. It is
not a per-delta hook. Streaming and non-streaming converge on one call site.

Streaming delta events may already have been delivered when this hook runs. A
middleware transformation of the final message changes the durable response
and subsequent runtime behavior; Luca does not replay or revise previously
emitted deltas.

An errored, cancelled, or incomplete model stream has no complete assistant
response, so it does not invoke this hook. A response which completes inside
the configured cancellation grace period is a complete response and follows
the normal hook and persistence path; cancellation still controls the turn's
eventual outcome.

### `before_tool_creation`

Runs at the start of each tool-creation attempt, immediately before
`ToolRegistry.create_execution(session, conversation_id, call)`.

The returned `ToolCall` is the effective call supplied to the registry and is
carried into the resulting execution. It may change the tool name, arguments,
id, or the entire call object.

This hook concerns creation of a call-scoped `ToolExecution`. It does not
concern construction of the model-visible tool catalog.

### `after_tool_creation`

Runs after the creation phase has produced a complete pre-persistence
`ToolExecution`. The runner first combines the registry's birth draft with the
existing `RECEIVED` execution and applies runner-owned birth facts. The hook
then receives that effective execution before it is committed as the birth
state.

The hook also runs for framework-produced creation outcomes, including a
draft synthesized when no registry is configured, when `create_execution`
raises, when a cancellation wins the race, and when the spawn budget refuses
the call. When a live `Exception` caused the synthesized outcome, the same
exception is supplied as context to every middleware.

The returned execution determines whether the call proceeds to approval or is
already terminal. Luca derives the next lifecycle step from that returned
state.

Creation may be retried from durable `RECEIVED` state after interruption or
process recovery. The before/after creation pair therefore applies per
creation attempt, not once per tool-call id for all time.

### `before_permission_check`

Runs immediately before `ToolRegistry.decide()`. The returned execution is
the execution the registry evaluates and the basis for the state updated with
the resulting decision.

Its result is discarded when a cancellation lands while `decide()` is in
flight — the execution must stay `PENDING` for the wind-down, so there is
nowhere to put it, and a decision that never happened has nothing to apply.

### `after_permission_decision`

Runs after `ToolRegistry.decide()` returns and before the decision is applied
to the execution. The returned decision is appended to the approval audit
trail and determines the current approval state and next lifecycle step.

The `execution` argument is context and is passed unchanged through the
decision middleware chain.

### `before_tool_execution`

Marks the beginning of an actual dispatch attempt. It runs only for an
execution selected for dispatch after creation and approval.

The hook runs before `ToolRegistry.prepare()`, so its returned execution — and
especially its `raw_tool_call` — is what the registry resolves and validates.
The dispatch attempt includes preparation and the tool body. A preparation
failure or cancellation is still part of that attempt even when the body is
never invoked.

It does not run for:

- Terminal-at-creation outcomes.
- Permission rejection.
- Runtime refusal before dispatch.
- Cancellation wind-down for a call never selected for dispatch.
- Recovery of a persisted orphaned `RUNNING` execution.

**This is a behavior change.** Today the hook also fires for every one of
those undispatched outcomes. "Before execution" for a call that will never
execute is the wrong contract, and `after_tool_execution` already covers every
terminal outcome.

The hook runs once per dispatch attempt. If the process dies before that
attempt produces durable state, a later drive may start a new attempt and
invoke it again — over the ORIGINAL `raw_tool_call`, since a rewrite from the
lost attempt is gone with it.

### `after_tool_execution`

Runs for every terminal tool outcome, whether or not dispatch was attempted.
This includes successful results, tool-reported error results, creation
failures, preparation failures, permission rejection, runtime refusal,
cancellation, timeouts, interruption, and orphan recovery.

It receives the fully formed terminal execution after context accounting and
before final persistence. Its return value passes through
`before_entry_written` and becomes the durable outcome.

`exception` is the live exception which produced the outcome in the current
process. It is `None` for outcomes without an exception and when no live
exception exists, such as persisted orphan recovery.

The asymmetry with `before_tool_execution` is intentional: the before hook
describes dispatch, while the after hook is the universal tool-outcome
transformation point.

## Lifecycle ordering

### User message

```text
resolve target conversation → validate acceptance
→ before_post_message
→ construct UserMessage
→ calculate context
→ before_entry_written
→ persist
```

### Model round

```text
build_model_string
→ project conversation to client messages
→ assemble system message
→ before_llm_call
→ resolve ToolSpecs (ToolRegistry.get_tools)
→ drop private specs
→ build_tool_list
→ adapt ToolSpecs to client tool DTOs
→ execute or consume the model completion
→ after_llm_response
→ construct the durable AssistantMessage
→ before_entry_written
→ persist
```

Message preparation and tool-catalog construction are INDEPENDENT: neither
reads the other's result, so their relative order is an implementation
detail, not a contract. (Today the runner projects first and collects tools
second, because the tool step is the one raced against the cancellation
token.) What IS normative is each hook's position against its own data: the
model string exists before the call, private specs are dropped before
`build_tool_list`, adaptation happens after it and before the client call, and
the complete response exists before `after_llm_response`.

### Tool call

```text
persist RECEIVED execution through before_entry_written
→ before_tool_creation
→ ToolRegistry.create_execution
→ merge the birth draft into the execution
→ after_tool_creation
→ persist birth state through before_entry_written

if terminal at birth:
    → after_tool_execution
    → before_entry_written
    → persist terminal outcome

if approval is required:
    → before_permission_check
    → ToolRegistry.decide
    → after_permission_decision
    → persist decision through before_entry_written

if the decision is terminal:
    → after_tool_execution
    → before_entry_written
    → persist terminal outcome

if selected for dispatch:
    → before_tool_execution
    → ToolRegistry.prepare
    → persist RUNNING through before_entry_written
    → invoke prepared tool body
    → form terminal outcome
    → after_tool_execution
    → before_entry_written
    → persist terminal outcome
```

Runtime refusal and cancellation wind-down enter the universal terminal tail
at `after_tool_execution`; they do not enter the dispatch-only
`before_tool_execution` path.

## Durability and invocation guarantees

Hook invocation and durable persistence are separate facts. A hook may run and
the operation may then be cancelled, fail, or lose the process before its
return value is committed.

The contract is:

- `before_entry_written` runs for each write attempt, not once per entry.
- Tool creation hooks run per creation attempt.
- Permission hooks run per decision attempt.
- `before_tool_execution` runs once per dispatch attempt.
- `after_tool_execution` runs whenever the runner forms a terminal outcome.
- Model hooks run per model invocation or completed response as defined above.

Middleware performing external side effects can use the supplied session,
conversation id, entry id, and tool-call id to define its own idempotency
policy.

## Events and other extension surfaces

Middleware transforms runtime values. `AgentEvent` remains the observational
surface for rendering, telemetry, and lifecycle consumption. Events are
emitted from the effective state after middleware and persistence, except for
streaming deltas which necessarily precede the completed response hook.

Conversation history policy remains the responsibility of
`ConversationProjector`. It owns traversal, omission, synthetic messages,
redaction policy, and tool-output projection. `before_llm_call` is the
last-mile transformation of the projector's result, not a replacement for the
projector.

`ToolRegistry` remains the execution and approval strategy. Middleware can
transform values at its boundaries without replacing its responsibility for
discovery, creation, decision, and preparation.

## Codebase findings

Grounding facts established while refining this PRD.

- **Nothing in `luca/` implements a middleware hook.** Only tests do. The
  blast radius is `runner.py`, `middleware.py`, `docs/agent/07-middleware.md`,
  `AGENTS.agent.md`, and a handful of test files. No contrib package changes.
- **`after_llm_response` already has one call site** (`runner.py:3158`).
  Streaming and non-streaming both converge on the assembled `message` before
  it; an aborted stream `continue`s and an errored one raises before reaching
  it. Requirement 7 is already satisfied by the implementation and needs
  tests, not code.
- **`SessionLedger` already takes a `conversation_id` on every door** except
  `refresh_entry`, which is deliberate. The missing scope is entirely in the
  runner's private write helpers (`_complete_entry`, `_append`,
  `_complete_uncommitted`, `_persist_entry`, `_persist_execution`), which do
  not currently receive one.
- **`_finalize_undispatched` becomes redundant.** It is exactly
  `before_tool_execution` followed by `_finalize_outcome`. Once the hook stops
  firing for undispatched calls, the two are identical; the method is deleted
  and its five call sites (`runner.py:2809, 3567, 3584, 3789, 3975`) call
  `_finalize_outcome`.
- **The "hook is the boundary" rule retires.** AGENTS.agent.md §12 documents a
  deliberate asymmetry — a cancellation during `prepare()` records `CANCELLED`
  in place rather than deferring to the loop-top wind-down — whose stated
  reason is that `before_tool_execution` has already fired and the wind-down
  would fire it twice. With the wind-down no longer firing the hook, that
  reason is gone. The durable shape is identical either way; the in-place
  behavior may stay, but the docs must stop justifying it that way.
- **`build_tool_list` is public and three tests call it directly**
  (`test_private_tools.py:98`, `test_runner.py:338`, `test_runner.py:365`),
  asserting wire tools. Its return type changes to `list[ToolSpec]` and the
  adapter conversion moves up into `_collect_tools`.

## Required implementation work

1. Add `session` and `conversation_id` to every middleware invocation and
   reference-mixin signature.
2. Thread `conversation_id` through the runner's write helpers
   (`_complete_entry`, `_append`, `_complete_uncommitted`, `_persist_entry`,
   `_persist_execution`) so every `before_entry_written` call carries an
   explicit scope. A compaction transition uses the OUTGOING conversation.
3. Stop running middleware in `recalculate_context_tokens()`: recalculate
   context and store through `ledger.refresh_entry`, no hook.
4. Change `build_tool_list` to transform `ToolSpec`s and return
   `list[ToolSpec]`; move `adapter.tool_spec_to_luca_tool` conversion into
   `_collect_tools`. The private filter stays ahead of the hook.
5. Remove the `luca.client.types.tools.Tool` import (and its `try/except
   ImportError` guard) from `luca/agent/core/middleware.py`. The two model
   message imports stay.
6. Add `before_tool_creation` and `after_tool_creation` around the execution
   birth phase, covering registry drafts and every synthesized draft.
7. Restrict `before_tool_execution` to dispatch attempts; delete
   `_finalize_undispatched` and route its five call sites through
   `_finalize_outcome`.
8. Update public exports, `docs/agent/07-middleware.md` (including the
   embedded mixin source), and the middleware sections of `AGENTS.agent.md`
   (the "Add middleware" recipe, design principle §12's hook-is-the-boundary
   paragraph, the subagents section's "Middleware stays conversation-blind"
   note, and the per-step method table).
9. Replace all tests and examples using the old signatures. No compatibility
   dispatch or signature introspection.

## Acceptance criteria

- The reference mixin exposes exactly the twelve hooks specified in this PRD.
- Every hook receives the same live `AgentSession` held by the runner and the
  exact conversation id responsible for the operation.
- Main-conversation and subagent tests demonstrate distinct conversation ids
  reaching the same middleware instance.
- Middleware chaining is tested in declared list order for before hooks, after
  hooks, tuple returns, and exception context.
- Returned values are passed through without middleware-specific validation or
  repair.
- `middleware.py` imports no tool type from `luca.client`; the two model
  message types remain.
- `build_tool_list` receives and returns `ToolSpec`s, never sees a private
  spec, and remains independently callable.
- `before_tool_creation` can change the call observed by
  `ToolRegistry.create_execution`.
- `after_tool_creation` can change the durable birth state and the lifecycle
  branch selected next, and receives the live exception for a synthesized
  failure draft.
- `before_tool_execution` runs for dispatch attempts, including preparation
  failures, and does not run for any undispatched terminal outcome.
- `after_tool_execution` runs for every terminal outcome.
- Streaming and non-streaming completions each invoke `after_llm_response`
  exactly once for a complete response and zero times for an incomplete one.
- A transformed complete streaming response is what is persisted and used for
  tool execution, without revising previously emitted delta events.
- Every entry write path that runs middleware passes an explicit conversation
  scope; a compaction transition passes the outgoing conversation.
- `recalculate_context_tokens()` invokes no middleware.
- The full agent test suite and lint checks pass.

## Historical design decisions

The following alternatives were considered during the design process:

- An `around_*` API was replaced by explicit `before_*` and `after_*` hooks so
  each lifecycle boundary has a direct, separately implementable contract.
- A middleware-only `LLMCall` request envelope was not adopted. Model inputs
  remain explicit arguments.
- **Core-owned provider-neutral `ModelMessage` / `ModelAssistantMessage` DTOs
  were considered and rejected.** The proposal was to mirror the client's
  message vocabulary inside `luca.agent.core` so `middleware.py` imported
  nothing from `luca.client`. Rejected because `projection.py` — core, next
  door — already targets client DTOs deliberately and documents it, so
  cleaning only `middleware.py` buys a cosmetic boundary; because doing it
  properly means duplicating three message classes and seven content-block
  types plus `MediaSource`, adding a bidirectional adapter, and rewriting the
  projector, `SummarizingContextManager` and every projection test; and
  because the motivation (another-language implementations) is served by
  specifying what the boundary CARRIES rather than by inventing a Python type
  for it. Middleware is a language-neutral contract at the level of concepts:
  "the messages about to go to the model" binds to `luca.client` messages in
  Python and to whatever the local client uses elsewhere.
- Model-string and tool-list builders remain independent extension points
  because applications may build or inspect them without issuing a model
  call.
- Per-delta response middleware was not added. Streaming converges on the same
  complete-response hook used by non-streaming execution.
- `recalculate_context_tokens()` originally threaded every entry through
  `before_entry_written` ("middleware has the final say on context"). It now
  runs no middleware: the method rewrites every entry across every
  conversation, so no single id honestly describes the operation, and
  inventing one (or admitting `None` into a contract built on concrete ids)
  costs more than the hook is worth there. It is an operational refresh.
