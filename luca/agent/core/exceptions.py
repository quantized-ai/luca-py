"""Agent-level exceptions.

A single home for every exception the agent framework raises. `AgentError` is the
base; `CancelledError` is raised by a `CancellationToken` (see
`luca.agent.core.context`); `CompactionPlanError` is the one rejection the
compaction step can produce (see `luca.agent.core.compaction`).

`ConversationCancellingError` is the one `post_message` rejection an
interactive application hits in normal operation and must handle gracefully
(keep the draft, retry after the flush); it is a flat subclass of
`AgentError`, which stays correct for every other rejection.

An approval pause is NOT an exception: when the permission strategy returns a
PENDING decision the generator simply ends at the gate, and a later `run()`
asks the strategy again.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for all luca.agent errors."""


class CancelledError(AgentError):
    """Raised when a cancelled `CancellationToken` is checked via
    `raise_if_cancelled()`. Distinct from `asyncio.CancelledError`."""


class AlreadyCancellingError(AgentError):
    """cancel() while an unconsumed CancelRequested exists. The first call's
    outcome/error stand; this raise is diagnostic only."""


class ConversationCancellingError(AgentError):
    """post_message() on a CANCELLING conversation. The open turn is being
    flushed, so an append would be buried in the cancelled bracket, silently
    unanswered. Transient: catch it, keep the draft, retry after the flush.
    Distinct from `CancelledError` (the token's raise), from
    `AlreadyCancellingError` (the cancel()-twice diagnostic), and from
    `asyncio.CancelledError` (task semantics)."""


class ToolNotFound(AgentError):
    """The requested (or middleware-effective) tool could not be resolved.
    Raised internally by the runner so the durable `ToolExecutionError` gets a
    meaningful type via `to_tool_execution_error`; never escapes a run."""


class InvalidToolArguments(AgentError):
    """The requested (or middleware-effective) arguments failed validation.
    Carries the JSON-clean pydantic error list so the durable error's
    `details.errors` preserves the structured validation output."""

    def __init__(self, message: str, errors: list | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


class CompactionPlanError(AgentError):
    """A `CompactionPlan` the runner refuses to commit: it references an id
    that does not exist or was not offered to the policy, references one
    twice, is empty, omits the compaction entry itself, carries no content, or
    was computed against a conversation that has since moved. Structure only —
    the runner never judges whether a compaction was worth doing.

    Closes the compaction bracket ERRORED. A user-scheduled compaction raises
    it to the caller; a policy-initiated one degrades and lets the turn
    proceed."""


class ProjectionError(AgentError):
    """Conversation projection failed — a missing entry id, an unknown entry
    type, a nonterminal tool execution, or a COMPLETED execution without a
    result. Projection errors propagate; they are never converted into
    synthetic conversation content."""
