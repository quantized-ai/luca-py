# Middleware

Middleware lets you intercept and modify the runner's pipeline at **10** points —
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
    def after_llm_response(self, message):
        record_usage(message.usage)
        return message
```

There's a mixin you can extend for your middlewares in
`luca.agent.core.middleware.AgentMiddlewareMixin`. Every hook on it is an
identity pass-through — it returns exactly what it receives — so the hooks you
don't override have no effect:

```python
class AgentMiddlewareMixin:
    def build_model_string(self, model_string: str, llm_cfg: LLMConfig) -> str:
        """Build the model identifier sent to the client.
        Override to route to a different model, add prefixes/suffixes,
        or implement per-turn model selection."""
        return model_string

    def build_tool_list(self, tools: list) -> list:
        """Filter or modify the wire tool list sent to the LLM on every call.
        Called per LLM invocation (not once at construction), so the result can
        vary by turn, session state, or any runtime condition."""
        return tools

    def before_post_message(self, parts: list[ContentPart]) -> list[ContentPart]:
        """Before a user message is appended to the session. Return the
        (possibly modified) content parts — sanitise, enrich, log. The whole
        ordered list is visible, text and images alike, so a hook can rewrite,
        drop, reorder or add parts."""
        return parts

    def before_entry_written(self, entry: AnyEntry) -> AnyEntry:
        """Before any entry persistence — appends (UserMessage,
        AssistantMessage, ToolExecution, TurnStart, TurnFinish,
        CancelRequested, CompactionEntry, ChildConversation) AND every update
        to the three MUTABLE entry types: a `ToolExecution` (approval changes,
        the RUNNING transition, cancellation stamps, terminal outcomes), a
        `CompactionEntry` (the `started_at` stamp, and the summary landing at
        the commit point), and a `ChildConversation` (its `execution_result`
        landing once the subagent finishes). Return the (possibly modified)
        entry — add metadata, stamp external ids, mutate fields before
        persistence."""
        return entry

    def before_llm_call(
        self,
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
        message: ClientAssistantMessage,
    ) -> ClientAssistantMessage:
        """After the LLM responds, before the AssistantMessage is recorded.
        Fires on every round — both tool-call rounds and final answers. Return
        the (possibly modified) message — redact, enrich, track token usage."""
        return message

    def before_permission_check(
        self,
        execution: ToolExecution,
    ) -> ToolExecution:
        """Before the registry's decide() is asked about an execution.
        Return the (possibly modified) execution — it is what decide() sees
        AND the execution updated and persisted after the decision, so its
        changes are not restricted to the decide call."""
        return execution

    def after_permission_decision(
        self,
        decision: ApprovalDecision,
        execution: ToolExecution,
    ) -> ApprovalDecision:
        """After the registry's decide() returns, before the decision is
        recorded. Return the (possibly modified) decision — override DENY →
        ALLOW for trusted callers, log all decisions, escalate to a second
        reviewer."""
        return decision

    def before_tool_execution(
        self,
        execution: ToolExecution,
    ) -> ToolExecution:
        """When the runtime is about to handle an execution's outcome. An
        allowed call receives it before dispatch, still PENDING — change
        `raw_tool_call` here to alter the effective call, which is what the
        registry's `prepare()` then resolves and validates from (the hook
        deliberately runs AHEAD of it). A terminal-at-birth call arrives with
        NOT_FOUND / INVALID / FAILED already set, a budget-refused call with
        REFUSED, a denied call with REJECTED,
        a call cancelled before dispatch with CANCELLED. Not invoked again
        when a RUNNING call later reaches its terminal status. Return the
        (possibly modified) execution.

        EXACTLY ONCE PER DISPATCH ATTEMPT — not once per call for all time. A
        crash during `prepare()` writes nothing, so the execution is still
        PENDING and the next drive fires this hook again for the same call,
        over the ORIGINAL `raw_tool_call` (a rewrite from the lost attempt is
        gone with it). Correct for an attempt that produced no outcome, but
        worth knowing before writing a hook that assumes once-forever."""
        return execution

    def after_tool_execution(
        self,
        execution: ToolExecution,
        exception: Exception | None = None,
    ) -> ToolExecution:
        """Runs for EVERY execution outcome, with the fully formed execution
        (status, result or error, lifecycle timestamps). `exception` is the
        live exception behind a failure in the current process (the same one
        given to the runner's error converter); it is None for outcomes
        without one and when no live exception survives (crash recovery).
        Runs before the final persistence: the return value passes through
        `before_entry_written` and is stored."""
        return execution
```

> ⚠️ **Prefer a plain class.** We recommend you **not** extend/subclass
> `AgentMiddlewareMixin` — just create a normal class implementing the methods
> you wish to implement, as the base mixin might change. Reach for it only when
> strictly needed.

## 2. The pipeline — where each hook fires

| Stage | Hook | Signature → returns |
|---|---|---|
| User posts | `before_post_message` | `(parts: list[ContentPart])` → `list[ContentPart]` |
| **Any** entry persistence | `before_entry_written` | `(entry: AnyEntry)` → `AnyEntry` |
| Per model call | `build_model_string` | `(model_string: str, llm_cfg: LLMConfig)` → `str` |
| Per model call | `build_tool_list` | `(tools: list[client Tool])` → `list` |
| Per model call | `before_llm_call` | `(messages, system_message)` → `(messages, system_message)` |
| Model responded | `after_llm_response` | `(message)` → `message` |
| Per undecided call | `before_permission_check` | `(execution: ToolExecution)` → `ToolExecution` |
| Per decision | `after_permission_decision` | `(decision, execution)` → `decision` |
| Per dispatch attempt / non-dispatch outcome | `before_tool_execution` | `(execution: ToolExecution)` → `ToolExecution` |
| Per execution outcome (exit) | `after_tool_execution` | `(execution, exception=None)` → `ToolExecution` |

> `message` in `before_llm_call` / `after_llm_response` is the **client**
> `AssistantMessage` / `Message` (wire types from `luca.client`), not the agent
> `AssistantMessage` *entry* — the entry is built afterward and passes through
> `before_entry_written`. Likewise `tools` in `build_tool_list` is the
> post-adapter **wire** list (`luca.client.types.Tool`), never the registry's
> [`ToolSpec`](03-tools.md)s.

> ⚠️ **Only `before_tool_execution` is exactly-once and paired.** It fires once
> per dispatch attempt, and its returned execution is what the dispatch uses.
> Every other hook may fire without its result being persisted — most visibly
> `before_permission_check`, whose returned execution is discarded when a
> cancellation lands mid-`decide()` (the call stays PENDING for the wind-down,
> and `after_permission_decision` never fires).

> ⚠️ **The four per-call hooks do NOT fire for a compaction's own LLM request.**
> `build_model_string`, `build_tool_list`, `before_llm_call` and
> `after_llm_response` are written in terms of the conversational turn, and a
> hook has no argument telling it which call it is in — a trailing reminder or
> a turn-count router would silently corrupt summarization requests. The
> `ContextManager.compact` owns that request end to end ([12](12-compaction.md)).
> `before_entry_written` *does* fire for everything a compaction writes,
> including the mutation that lands the summary.

## 3. The three big levers

**Route the model per turn** — `build_model_string` runs on every call, so you
can switch model/provider by runtime condition:

```python
class ModelRouter:
    def build_model_string(self, model_string, llm_cfg):
        return "openrouter:anthropic/claude-opus-4-8" if self.hard_task else model_string
```

**Last-mile request changes** — `before_llm_call` sees the projected message
list and the assembled system prompt right before the HTTP request. (For
history *policy* — trimming, synthetic messages, tool-output wording — subclass
the [`ConversationProjector`](10-projection.md) instead.)

```python
class Reminder:
    def before_llm_call(self, messages, system_message):
        return messages, (system_message or "") + "\nAnswer in Spanish."
```

**Filter tools per call** — `build_tool_list` runs per call, so tool exposure can
vary by user or state. The hook is synchronous and sees the converted wire
objects (`luca.client.types.Tool`: `name`, `description`, `parameters`); the
runner's own `build_tool_list()` is `async` because the registry's `get_tools`
is, and the hook runs after that await:

```python
class ScopeTools:
    def __init__(self, allowed): self.allowed = allowed
    def build_tool_list(self, tools):
        return [t for t in tools if t.name in self.allowed]
```

## 4. The tool pair — the whole execution, every outcome

Both tool hooks work on the durable `ToolExecution` itself.

`before_tool_execution(execution)` fires when the runtime is about to handle an
execution's outcome: an **allowed** call arrives still `PENDING`, and the
returned execution's `raw_tool_call` is the **effective call** — the hook runs
*ahead* of the registry's `prepare()`, which resolves the tool and validates the
arguments from it:

```python
class Args10x:
    def before_tool_execution(self, execution):
        args = execution.raw_tool_call.arguments
        return execution.model_copy(update={
            "raw_tool_call": execution.raw_tool_call.model_copy(
                update={"arguments": {k: v * 10 for k, v in args.items()}},
            ),
        })
```

A call that never dispatches also passes through — with `NOT_FOUND` / `INVALID`
/ `FAILED` (terminal at birth), `REFUSED` (a runtime limit, [13](13-subagents.md)),
`REJECTED` (denied), or `CANCELLED` already set.

> ⚠️ **Once per dispatch attempt, not once per call forever.** A crash during
> `prepare()` persists nothing, so the call is still `PENDING` and the next
> drive fires the hook again — over the *original* `raw_tool_call`, since the
> lost attempt's rewrite went with it.

`after_tool_execution(execution, exception=None)` observes **every** outcome —
`COMPLETED`, `FAILED` (with the live exception behind a `prepare()` or body
raise; registry-authored terminal births carry none), `NOT_FOUND`, `INVALID`,
`REJECTED`, `REFUSED`, `CANCELLED`, `INTERRUPTED`, `TIMED_OUT` — and its return
value is
what gets persisted:

```python
class RedactResults:
    def after_tool_execution(self, execution, exception=None):
        if execution.result is None:
            return execution
        return execution.model_copy(update={"result": redact(execution.result)})
```

> ⚠️ **Trusted, not validated.** The runner persists whatever your hooks
> return — statuses, results, errors, timestamps included. It performs no
> defensive repair; unusual authored state is yours to own.

## 5. Middleware is conversation-blind

No hook receives a `conversation_id`. With subagents running
([13](13-subagents.md)) that means a hook cannot tell which conversation it is
firing for, and hooks from several conversations interleave on one event loop.

| What that costs you | What to do |
|---|---|
| `before_entry_written` cannot attribute an entry to a conversation | a `ToolExecution` carries `execution.conversation_id`; nothing else does |
| the per-LLM-call hooks fire once per call in EVERY conversation | a turn-count router or a trailing reminder written for "the turn" now also rewrites a subagent's request |
| hook instances are shared | keep no per-call state on `self`; a field written in `before_llm_call` and read in `after_llm_response` is a race |

> ⚠️ **No error tells you.** Nothing in `luca/` implements a middleware hook, so
> this is a missing capability rather than a broken one — but an application
> that shipped a hook assuming one conversation gets wrong behavior silently
> once subagents are switched on.

## 6. Ordering

Every hook runs through the whole list in order; `middleware[n]`'s output is
`middleware[n+1]`'s input. There is **no** reverse ordering, even for
before/after pairs — `middleware[0]` always runs first.

```python
class AddSuffix:
    def __init__(self, s): self.s = s
    def build_model_string(self, model_string, llm_cfg): return model_string + self.s

middleware=[AddSuffix("-preview"), AddSuffix("-2025")]
# model string sent to the client: "openrouter:openai/gpt-4o-mini-preview-2025"
```

## Calling the build methods directly

The per-call hooks are driven by public runner methods you can also call in
tests or subclasses. Each names the conversation it builds for:
`build_model_string(llm_cfg)`, `await resolve_tool_specs(conversation_id)` then
`build_tool_list(conversation_id, specs)` (the split is what lets the private
filter and the wire conversion happen in one place),
`build_messages(conversation_id)` *(no hook — delegates to the projector)*,
`build_system_message(conversation_id)` *(no hook — assembler only)*, and
`prepare_llm_call(conversation_id)` (runs `before_llm_call` after the builders).
Next: [`08-runtime-config.md`](08-runtime-config.md).
