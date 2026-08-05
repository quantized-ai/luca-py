# Middleware

Middleware lets you intercept and modify the runner's pipeline at **12** points —
without subclassing the runner or touching the session model. Pass a list of
plain objects; each hook they implement is called, in list order.

```python
runner = AgentSessionRunner(
    session, tool_registry=registry,
    middleware=[LoggingMiddleware(), CostTracker(), ModelRouter()],
)
```

> ⚠️ **Projection is not middleware.** There is deliberately no
> `build_messages` hook: shaping the LLM message history (trimming, injecting,
> redacting, tool-output wording) belongs on the
> [`ConversationProjector`](10-projection.md). `before_llm_call` remains as the
> last-mile request hook downstream of projection.

## 1. Write a plain class with only the hooks you need

A middleware is any object that defines one or more hook methods. The runner
dispatches via `hasattr` and skips methods you don't define. Every hook is a
**pass-through transformer**: it receives a value and returns it, possibly
modified. Return it unchanged to observe without altering.

```python
class CostTracker:                       # plain class — no base
    def after_llm_response(self, session, conversation_id, message):
        record_usage(conversation_id, message.usage)
        return message
```

Every hook starts with `(session, conversation_id)`. `session` is the **live**
`AgentSession` the runner writes through; `conversation_id` is the conversation
whose operation invoked the hook — the main one, or a subagent
([13](13-subagents.md)). One middleware instance serves the whole tree, so that
id is how a hook keeps per-conversation state, routes a subagent to a cheaper
model, or attributes cost. It is the same prefix
[`ToolRegistry`](03-tools.md), `ContextManager.compact` and system-prompt
callables already take.

There's a mixin you can extend for your middlewares in
`luca.agent.core.middleware.AgentMiddlewareMixin`. Every hook on it is an
identity pass-through — it returns exactly what it receives — so the hooks you
don't override have no effect:

```python
class AgentMiddlewareMixin:
    def build_model_string(
        self,
        session: AgentSession,
        conversation_id: str,
        model_string: str,
        llm_cfg: LLMConfig,
    ) -> str:
        """Build the model identifier sent to the client.
        Override to route to a different model, add prefixes/suffixes,
        or implement per-conversation model selection — a cheap model for
        subagents, the configured one for the main conversation."""
        return model_string

    def build_tool_list(
        self,
        session: AgentSession,
        conversation_id: str,
        tools: list[ToolSpec],
    ) -> list[ToolSpec]:
        """Filter or modify the model-visible tool catalog on every call.
        Called per LLM invocation (not once at construction), so the result can
        vary by turn, conversation, session state, or any runtime condition.

        Receives `ToolSpec`s — the core's own tool type — AFTER private specs
        have been dropped and BEFORE the adapter converts them to client tool
        DTOs. A spec carries `tool_kind`, `namespace`, `is_private`,
        `output_schema` and `metadata`, none of which survive onto the wire, so
        this is the list a policy can actually filter on."""
        return tools

    def before_post_message(
        self,
        session: AgentSession,
        conversation_id: str,
        parts: list[ContentPart],
    ) -> list[ContentPart]:
        """Before a user message is appended to the session. Return the
        (possibly modified) content parts — sanitise, enrich, log. The whole
        ordered list is visible, text and images alike, so a hook can rewrite,
        drop, reorder or add parts.

        `conversation_id` is the RESOLVED target — the explicit
        `post_message(conversation_id=...)` argument, or the main conversation
        when it was omitted. Never None."""
        return parts

    def before_entry_written(
        self,
        session: AgentSession,
        conversation_id: str,
        entry: AnyEntry,
    ) -> AnyEntry:
        """Before any entry persistence — appends (UserMessage,
        AssistantMessage, ToolExecution, TurnStart, TurnFinish,
        CancelRequested, CompactionEntry, ChildConversation) AND every update
        to the three MUTABLE entry types: a `ToolExecution` (approval changes,
        the RUNNING transition, cancellation stamps, terminal outcomes), a
        `CompactionEntry` (the `started_at` stamp, and the summary landing at
        the commit point), and a `ChildConversation` (its `execution_result`
        and `result_execution_id` landing together once the subagent finishes
        — so this hook fires a second time for that entry). Return the
        (possibly modified) entry — add metadata, stamp external ids, mutate
        fields before persistence.

        `conversation_id` is the conversation whose operation caused the
        write, not a claim of ownership. Two scopes worth knowing: a COMPACTION
        transition writes its plan's new entries under the OUTGOING
        conversation (the destination's id does not exist yet), and
        `recalculate_context_tokens()` runs this hook NOT AT ALL — it rewrites
        every entry across every conversation, so no single id would be
        honest."""
        return entry

    def before_llm_call(
        self,
        session: AgentSession,
        conversation_id: str,
        messages: list[Message],
        system_message: str | None,
    ) -> tuple[list[Message], str | None]:
        """After conversation projection, before the HTTP request. Final
        chance to modify the message list or system prompt — inject context,
        redact PII, add a trailing reminder. Return the (possibly modified)
        pair. Rewriting a projected ToolMessage here can diverge from an
        already-emitted ToolExecuted event; the application owns that."""
        return messages, system_message

    def after_llm_response(
        self,
        session: AgentSession,
        conversation_id: str,
        message: ClientAssistantMessage,
    ) -> ClientAssistantMessage:
        """After the LLM responds, before the AssistantMessage is recorded.
        Fires on every round — both tool-call rounds and final answers. Return
        the (possibly modified) message — redact, enrich, track token usage.

        Exactly once per COMPLETE response, streaming or not: a stream is
        assembled to its terminal message first and this runs on that. It is
        not a per-delta hook, and an errored, cancelled or incomplete stream
        does not invoke it at all. Deltas already emitted are never revised, so
        a transformation here changes the durable response and everything
        downstream of it while the rendered stream stands as it was."""
        return message

    def before_tool_creation(
        self,
        session: AgentSession,
        conversation_id: str,
        call: ToolCall,
    ) -> ToolCall:
        """Before `ToolRegistry.create_execution()` — the start of one tool
        call's creation attempt. Return the (possibly modified) call: it is
        what the registry sees and what the resulting execution carries. Change
        the name, the arguments, the id, or the whole object.

        This is about creating a call-scoped `ToolExecution`, not about the
        model-visible catalog (that is `build_tool_list`). It runs per CREATION
        ATTEMPT: a birth retried from durable RECEIVED state after a crash
        fires it again for the same tool-call id."""
        return call

    def after_tool_creation(
        self,
        session: AgentSession,
        conversation_id: str,
        execution: ToolExecution,
        exception: Exception | None = None,
    ) -> ToolExecution:
        """After the creation phase has produced a complete pre-persistence
        execution — the registry's birth draft folded into the RECEIVED entry
        with every runner-owned birth fact applied — and before it is committed
        as the birth state. Return the (possibly modified) execution; the
        lifecycle branch taken next is derived from what you return, so
        terminalizing a PENDING birth here sends it straight to the outcome
        tail without ever reaching `decide()`.

        Also runs for framework-synthesized drafts: no registry configured, a
        raising `create_execution`, a cancellation losing the birth race, and a
        spawn-budget refusal. `exception` is the live exception behind a
        synthesized failure, None otherwise."""
        return execution

    def before_permission_check(
        self,
        session: AgentSession,
        conversation_id: str,
        execution: ToolExecution,
    ) -> ToolExecution:
        """Before the registry's decide() is asked about an execution.
        Return the (possibly modified) execution — it is what decide() sees
        AND the execution updated and persisted after the decision, so its
        changes are not restricted to the decide call."""
        return execution

    def after_permission_decision(
        self,
        session: AgentSession,
        conversation_id: str,
        decision: ApprovalDecision,
        execution: ToolExecution,
    ) -> ApprovalDecision:
        """After the registry's decide() returns, before the decision is
        recorded. Return the (possibly modified) decision — override DENY →
        ALLOW for trusted callers, log all decisions, escalate to a second
        reviewer. `execution` is read-only context, passed unchanged down the
        chain."""
        return decision

    def before_tool_execution(
        self,
        session: AgentSession,
        conversation_id: str,
        execution: ToolExecution,
    ) -> ToolExecution:
        """The start of an actual DISPATCH ATTEMPT, for an execution selected
        for dispatch after creation and approval. Change `raw_tool_call` here
        to alter the effective call, which is what the registry's `prepare()`
        then resolves and validates from (the hook deliberately runs AHEAD of
        it). Return the (possibly modified) execution.

        It does NOT run for an outcome that never dispatches: terminal at birth
        (NOT_FOUND / INVALID / FAILED), REFUSED, REJECTED, cancelled before
        dispatch, or an orphaned RUNNING recovered to INTERRUPTED. Every one of
        those still runs `after_tool_execution`. The attempt covers preparation
        as well as the body, so a `prepare()` failure is part of it.

        EXACTLY ONCE PER DISPATCH ATTEMPT — not once per call for all time. A
        crash during `prepare()` writes nothing, so the execution is still
        PENDING and the next drive fires this hook again for the same call,
        over the ORIGINAL `raw_tool_call` (a rewrite from the lost attempt is
        gone with it). Correct for an attempt that produced no outcome, but
        worth knowing before writing a hook that assumes once-forever."""
        return execution

    def after_tool_execution(
        self,
        session: AgentSession,
        conversation_id: str,
        execution: ToolExecution,
        exception: Exception | None = None,
    ) -> ToolExecution:
        """Runs for EVERY execution outcome, with the fully formed execution
        (status, result or error, lifecycle timestamps) — dispatched or not.
        Successes, tool-reported errors, creation failures, preparation
        failures, rejection, refusal, cancellation, timeouts, interruption and
        orphan recovery all land here; it is the universal tool-outcome
        transformation point, which is why it is deliberately not symmetric
        with the dispatch-only `before_tool_execution`.

        `exception` is the live exception behind a failure in the current
        process (the same one given to the runner's error converter); it is
        None for outcomes without one and when no live exception survives
        (crash recovery). Runs before the final persistence: the return value
        passes through `before_entry_written` and is stored."""
        return execution
```

> ⚠️ **Prefer a plain class.** We recommend you **not** extend/subclass
> `AgentMiddlewareMixin` — just create a normal class implementing the methods
> you wish to implement, as the base mixin might change. Reach for it only when
> strictly needed.

## 2. The pipeline — where each hook fires

Every signature below is shown *after* the `(session, conversation_id)` prefix.

| Stage | Hook | Signature → returns |
|---|---|---|
| User posts | `before_post_message` | `(parts: list[ContentPart])` → `list[ContentPart]` |
| **Any** entry persistence | `before_entry_written` | `(entry: AnyEntry)` → `AnyEntry` |
| Per model call | `build_model_string` | `(model_string: str, llm_cfg: LLMConfig)` → `str` |
| Per model call | `build_tool_list` | `(tools: list[ToolSpec])` → `list[ToolSpec]` |
| Per model call | `before_llm_call` | `(messages, system_message)` → `(messages, system_message)` |
| Model responded (complete) | `after_llm_response` | `(message)` → `message` |
| Per creation attempt | `before_tool_creation` | `(call: ToolCall)` → `ToolCall` |
| Per creation attempt | `after_tool_creation` | `(execution, exception=None)` → `ToolExecution` |
| Per undecided call | `before_permission_check` | `(execution: ToolExecution)` → `ToolExecution` |
| Per decision | `after_permission_decision` | `(decision, execution)` → `decision` |
| Per **dispatch attempt** | `before_tool_execution` | `(execution: ToolExecution)` → `ToolExecution` |
| Per execution outcome (exit) | `after_tool_execution` | `(execution, exception=None)` → `ToolExecution` |

> **What each boundary carries.** `message` in `before_llm_call` /
> `after_llm_response` is the **client** `Message` / `AssistantMessage` (wire
> types from `luca.client`), not the agent `AssistantMessage` *entry* — the
> entry is built afterward and passes through `before_entry_written`. `tools` in
> `build_tool_list`, by contrast, is the registry's own
> [`ToolSpec`](03-tools.md) list: the adapter converts it to wire tools *after*
> the hook, so a policy can filter on `tool_kind`, `namespace`, `metadata` and
> `output_schema`, none of which survive the conversion.

> ⚠️ **Only `before_tool_execution` is exactly-once and paired.** It fires once
> per dispatch attempt, and its returned execution is what the dispatch uses.
> Every other hook may fire without its result being persisted — most visibly
> `before_permission_check`, whose returned execution is discarded when a
> cancellation lands mid-`decide()` (the call stays PENDING for the wind-down,
> and `after_permission_decision` never fires).

> **`recalculate_context_tokens()` runs no middleware.** It re-derives
> `context_tokens` for every entry across every conversation, so no single
> `conversation_id` would honestly scope it. It is an operational refresh of a
> derived estimate, not a write ([11](11-context-and-usage.md)).

> ⚠️ **The four per-call hooks do NOT fire for a compaction's own LLM request.**
> `build_model_string`, `build_tool_list`, `before_llm_call` and
> `after_llm_response` are written in terms of the conversational turn, and a
> hook has no argument telling it which call it is in — a trailing reminder or
> a turn-count router would silently corrupt summarization requests. The
> `ContextManager.compact` owns that request end to end ([12](12-compaction.md)).
> `before_entry_written` *does* fire for everything a compaction writes,
> including the mutation that lands the summary.

## 3. The three big levers

**Route the model per call** — `build_model_string` runs on every call and knows
which conversation it is for, so you can switch model/provider by runtime
condition *or* by conversation:

```python
class CheapSubagents:
    def build_model_string(self, session, conversation_id, model_string, llm_cfg):
        if session.conversations[conversation_id].depth > 0:   # a subagent
            return "openrouter:anthropic/claude-haiku-4-5"
        return model_string
```

**Last-mile request changes** — `before_llm_call` sees the projected message
list and the assembled system prompt right before the HTTP request. (For
history *policy* — trimming, synthetic messages, tool-output wording — subclass
the [`ConversationProjector`](10-projection.md) instead.)

```python
class Reminder:
    def before_llm_call(self, session, conversation_id, messages, system_message):
        return messages, (system_message or "") + "\nAnswer in Spanish."
```

**Filter tools per call** — `build_tool_list` runs per call, so tool exposure can
vary by user, conversation or state. The hook is synchronous and sees
[`ToolSpec`](03-tools.md)s with the private ones already dropped; the runner's
`resolve_tool_specs()` is the `async` half, and the adapter converts what you
return into wire tools afterwards:

```python
class ReadOnlySubagents:
    def build_tool_list(self, session, conversation_id, tools):
        if session.conversations[conversation_id].depth == 0:
            return tools
        return [spec for spec in tools if spec.tool_kind is ToolKind.READ]
```

## 4. The tool lifecycle — four hooks, four different questions

All four work on the durable `ToolExecution` (or the `ToolCall` behind it).

**Creation.** `before_tool_creation(call)` runs immediately before
`ToolRegistry.create_execution()`; the returned call is what the registry sees
and what the execution carries. `after_tool_creation(execution, exception=None)`
runs on the finished birth state, before it is committed — and what you return
decides the next lifecycle step, so terminalizing a `PENDING` birth here sends
the call straight to the outcome tail without ever reaching `decide()`:

```python
class NoWritesForSubagents:
    def after_tool_creation(self, session, conversation_id, execution, exception=None):
        if session.conversations[conversation_id].depth == 0:
            return execution
        if execution.tool_spec is None or execution.tool_spec.tool_kind is not ToolKind.WRITE:
            return execution
        return execution.model_copy(update={
            "status": ExecutionStatus.REFUSED,
            "error": ToolExecutionError(
                error_type="PolicyRefusal",
                error_message="Subagents may not write.",
            ),
        })
```

`after_tool_creation` also fires for framework-synthesized drafts — no registry
configured, a raising `create_execution`, a cancellation losing the birth race,
a spawn-budget refusal — with the live exception where one exists.

**Dispatch.** `before_tool_execution(execution)` means *a dispatch attempt is
starting*, and nothing else. The call arrives still `PENDING`, and the returned
execution's `raw_tool_call` is the **effective call** — the hook runs *ahead* of
the registry's `prepare()`, which resolves the tool and validates the arguments
from it:

```python
class Args10x:
    def before_tool_execution(self, session, conversation_id, execution):
        args = execution.raw_tool_call.arguments
        return execution.model_copy(update={
            "raw_tool_call": execution.raw_tool_call.model_copy(
                update={"arguments": {k: v * 10 for k, v in args.items()}},
            ),
        })
```

> ⚠️ **It does not fire for a call that never dispatches.** Terminal at birth
> (`NOT_FOUND` / `INVALID` / `FAILED`), `REFUSED`, `REJECTED`, cancelled before
> dispatch, or an orphaned `RUNNING` recovered to `INTERRUPTED` — none of them
> reach it. Use `after_tool_creation` to intervene at birth and
> `after_tool_execution` to see every ending.

> ⚠️ **Once per dispatch attempt, not once per call forever.** A crash during
> `prepare()` persists nothing, so the call is still `PENDING` and the next
> drive fires the hook again — over the *original* `raw_tool_call`, since the
> lost attempt's rewrite went with it.

**Outcome.** `after_tool_execution(execution, exception=None)` observes **every**
outcome — `COMPLETED`, `FAILED` (with the live exception behind a `prepare()` or
body raise; registry-authored terminal births carry none), `NOT_FOUND`,
`INVALID`, `REJECTED`, `REFUSED`, `CANCELLED`, `INTERRUPTED`, `TIMED_OUT` —
dispatched or not, and its return value is what gets persisted:

```python
class RedactResults:
    def after_tool_execution(self, session, conversation_id, execution, exception=None):
        if execution.result is None:
            return execution
        return execution.model_copy(update={"result": redact(execution.result)})
```

The asymmetry between the two is deliberate: the *before* hook describes
dispatch, the *after* hook is the universal outcome point.

> ⚠️ **Trusted, not validated.** The runner persists whatever your hooks
> return — statuses, results, errors, timestamps included. It performs no
> defensive repair; unusual authored state is yours to own.

## 5. One instance, many conversations

A middleware instance is shared by the runner and serves the main conversation
**and** every subagent ([13](13-subagents.md)), concurrently. The
`conversation_id` on every hook is what makes that safe — it is the scope of the
*operation*, not a claim that the value belongs only to that conversation (an
entry can be referenced by more than one).

| Concern | The rule |
|---|---|
| per-conversation state | key it by `conversation_id` — no lock needed, dispatch within one conversation is sequential |
| deliberately shared state | lock the mutation, never the I/O |
| per-*call* state on `self` | don't — a field written in `before_llm_call` and read in `after_llm_response` is a race across conversations |
| attributing an entry | use the supplied `conversation_id`; a `ToolExecution` also carries `execution.conversation_id`, the conversation it was BORN in |

Two scopes are worth knowing because they are not what you might guess:

- A **compaction** writes its plan's new entries under the **outgoing**
  conversation — the destination's id does not exist yet ([12](12-compaction.md)).
- `recalculate_context_tokens()` runs **no** middleware at all.

## 6. Ordering

Every hook runs through the whole list in order; `middleware[n]`'s output is
`middleware[n+1]`'s input. There is **no** reverse ordering, even for
before/after pairs — `middleware[0]` always runs first. Context arguments (the
`llm_cfg`, the `execution` beside a decision, a live `exception`) are passed
**unchanged** to every middleware in the chain.

```python
class AddSuffix:
    def __init__(self, s): self.s = s
    def build_model_string(self, session, conversation_id, model_string, llm_cfg):
        return model_string + self.s

middleware=[AddSuffix("-preview"), AddSuffix("-2025")]
# model string sent to the client: "openrouter:openai/gpt-4o-mini-preview-2025"
```

## Calling the build methods directly

The per-call hooks are driven by public runner methods you can also call in
tests or subclasses. Each names the conversation it builds for:
`build_model_string(conversation_id, llm_cfg)`,
`await resolve_tool_specs(conversation_id)` then
`build_tool_list(conversation_id, specs)` (the split is what lets `get_tools`
be async while the hook stays synchronous; the private filter lives in the
second, and the adapter converts what it returns),
`build_messages(conversation_id)` *(no hook — delegates to the projector)*,
`build_system_message(conversation_id)` *(no hook — assembler only)*, and
`prepare_llm_call(conversation_id)` (runs `before_llm_call` after the builders).
Next: [`08-runtime-config.md`](08-runtime-config.md).
