"""Agent middleware interface for the AgentSessionRunner.

`AgentMiddlewareMixin` is a reference base class — subclassing it is optional
and discouraged. The runner uses duck typing: if a middleware object implements
a method, it will be called; if not, `hasattr` skips it. Any plain Python class
that defines the methods it needs is a valid middleware, and a plain class is
the recommended way to write one.

Every hook on the mixin is an identity pass-through — it returns exactly what
it receives — so a subclass that overrides only some hooks leaves every other
pipeline stage untouched.

Middlewares are applied in list order for all methods — there is no reverse
ordering for any method, including before/after pairs. Each middleware
receives the value returned by the previous one; context arguments (including
a live exception) are passed unchanged to every middleware in the chain.

Trust model: the runtime does not validate or repair what a hook returns. A
middleware may replace statuses, results, errors, ids, call data, approval
data, or timestamps on a `ToolExecution` — the framework persists and
proceeds from the returned value, and the application owns the consequences.

Conversation projection is NOT middleware: history shaping, redaction, and
tool-output wording belong on the `ConversationProjector` collaborator
(`projection.py`). `before_llm_call` remains as the last-mile, downstream
request hook.
"""

from __future__ import annotations

from luca.client.types.messages import AssistantMessage as ClientAssistantMessage
from luca.client.types.messages import Message

from .models import (
    AnyEntry,
    ApprovalDecision,
    LLMConfig,
    ToolExecution,
    ContentPart,
)

try:
    from luca.client.types.tools import Tool as LucaTool
except ImportError:
    LucaTool = object  # type: ignore[assignment,misc]


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
        CancelRequested) AND every `ToolExecution` update (approval changes,
        the RUNNING transition, cancellation stamps, terminal outcomes).
        Return the (possibly modified) entry — add metadata, stamp external
        ids, mutate fields before persistence."""
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
        self, message: ClientAssistantMessage,
    ) -> ClientAssistantMessage:
        """After the LLM responds, before the AssistantMessage is recorded.
        Fires on every round — both tool-call rounds and final answers. Return
        the (possibly modified) message — redact, enrich, track token usage."""
        return message

    def before_permission_check(
        self, execution: ToolExecution,
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
        `raw_tool_call` here to alter the effective call (the registry
        resolves and validates from it inside execute()). A terminal-at-birth
        call arrives with NOT_FOUND / INVALID / FAILED already set, a denied
        call with REJECTED, a call cancelled before dispatch with CANCELLED.
        Not invoked again when a RUNNING call later reaches its terminal
        status. Return the (possibly modified) execution."""
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
