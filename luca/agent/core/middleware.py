"""Agent middleware interface for the AgentSessionRunner.

`AgentMiddlewareMixin` is a reference base class — subclassing it is optional
and discouraged. The runner uses duck typing: if a middleware object implements
a method, it will be called; if not, `hasattr` skips it. Any plain Python class
that defines the methods it needs is a valid middleware, and a plain class is
the recommended way to write one.

Every hook on the mixin is an identity pass-through — it returns exactly what
it receives — so a subclass that overrides only some hooks leaves every other
pipeline stage untouched. The one departure is `synthesize_executions`, which
CONTRIBUTES rather than transforms (each middleware returns only its own
templates, defaulting to none) — identity there is the empty list.

CONVERSATION SCOPE. Every hook starts with `(session, conversation_id)`. An
`AgentSession` holds several conversations — the main one plus parallel
subagents — advancing concurrently, and ONE middleware instance serves the
whole tree. `session` is the live session object the runner and ledger write
through; `conversation_id` is the scope of the OPERATION that invoked the
hook, and lets the hook resolve `session.conversations[conversation_id]` when
it needs the object. It does NOT assert that the transformed value belongs
exclusively to that conversation — an entry may be referenced by more than one.

This is the framework-wide signature convention, not a middleware-specific
one: `ToolRegistry`, `ContextManager.compact`, `Tool.execute` and
system-prompt callables all take the same prefix.

Concurrency follows from the same fact: the runtime does not serialize
middleware globally, so a middleware holding state on itself owns its own
scoping. State keyed by `conversation_id` needs no lock; deliberately shared
state locks the mutation, never the I/O.

Middlewares are applied in list order for all methods — there is no reverse
ordering for any method, including before/after pairs. Each middleware
receives the value returned by the previous one; context arguments (including
a live exception) are passed unchanged to every middleware in the chain.

Trust model: the runtime does not validate or repair what a hook returns. A
middleware may replace statuses, results, errors, ids, call data, approval
data, or timestamps on a `ToolExecution` — the framework persists and
proceeds from the returned value, and the application owns the consequences.

Durability of a hook's RESULT. `before_tool_execution` is the only hook with
an exactly-once, paired guarantee: it fires exactly once per dispatch attempt
and never twice for one outcome. Every other hook may fire without its result
being persisted — most visibly `before_permission_check`, whose returned
execution is discarded when a cancellation lands while `decide()` is in
flight (the execution has to stay PENDING for the wind-down, so there is
nowhere to put it, and a decision that never happened has nothing to apply).

TYPE BINDING. A hook is defined by what its boundary CARRIES, not by a
particular Python class. Where Luca has its own vocabulary the hook uses it —
`build_tool_list` transforms `ToolSpec`s, the core's tool type, which carry
strictly more than the wire tool the adapter later builds from them. Where the
value simply IS the client's, the hook uses the client's type: `before_llm_call`
and `after_llm_response` carry "the messages going to the model" and "the
assistant message that came back", which in this implementation are
`luca.client` messages, exactly as `ConversationProjector` already produces
them. A Luca implementation in another language binds the same two concepts to
whatever its own client calls a message.

Conversation projection is NOT middleware: history shaping, redaction, and
tool-output wording belong on the `ConversationProjector` collaborator
(`projection.py`). `before_llm_call` remains as the last-mile, downstream
request hook.
"""

from __future__ import annotations

from luca.client.types.messages import AssistantMessage as ClientAssistantMessage, Message
from luca.client.types.tools import BaseTool as ClientBaseTool

from .models import (
    AgentSession,
    AnyEntry,
    ApprovalDecision,
    AssistantMessage,
    ContentPart,
    LLMConfig,
    ToolCall,
    ToolExecution,
    ToolSpec,
)


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

    def adapt_tool_declarations(
        self,
        session: AgentSession,
        conversation_id: str,
        tools: list[ClientBaseTool],
    ) -> list[ClientBaseTool]:
        """After the spec → client-declaration conversion, immediately before
        the list is handed to the client as `acompletion(tools=…)`. CLIENT
        vocabulary — the counterpart of `build_tool_list`, one conversion
        later: that hook DECIDES which tools this request advertises (spec
        vocabulary, the richer fields to filter on), this one RESHAPES how a
        surviving tool is declared. The place to swap a function declaration
        for a provider-native declaration item (`ApplyPatchTool()`,
        `TextEditorTool()`, …) — and the place to APPEND a provider-HOSTED
        declaration item (a web-search tool the provider executes
        server-side): a hosted declaration has no `ToolSpec` behind it, so
        the spec-vocabulary door upstream cannot carry it. Dropping tools,
        and adding tools that DO have specs, still belong upstream."""
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

    def synthesize_executions(
        self,
        session: AgentSession,
        conversation_id: str,
        entry: AssistantMessage,
    ) -> list[ToolExecution]:
        """Executions to append after `entry` — the COMMITTED assistant entry,
        so its durable id is available for linking — derived from what the
        response carried (a hosted web operation, an MCP server-side action,
        anything the model has already seen that deserves a durable structural
        record).

        UNLIKE every other hook this one CONTRIBUTES rather than transforms:
        each middleware returns only its OWN templates and the runner
        concatenates them in middleware order — no value is threaded through
        the chain. Templates carry no identity (`id` / `created_at` /
        `conversation_id` are the runner's to stamp; a missing
        `tool_call_id` / `raw_tool_call.id` is backfilled via `generate_id()`)
        and MUST be TERMINAL and resolve to a PRIVATE spec — no `ToolCall`
        block for them exists on the path, so private is the only wire-legal
        shape. No tool lifecycle events fire for them.

        SYNC ONLY — it runs inside the drive's documented no-await atomic
        block, between the assistant commit and the RECEIVED births.

        A template that violates the terminal/private contract raises
        `AgentError` OUT of the drive with the turn left OPEN — the assistant
        entry is already committed and, for a tool-calling response, the
        RECEIVED births have not run yet, which is not a cleanly resumable
        state. Acceptable because the refusals guard against middleware BUGS,
        not runtime conditions (middleware is trusted); the recovery door is
        `cancel()`, whose wind-down closes the turn from exactly this state.

        Default: no synthetics."""
        return []

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
