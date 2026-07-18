"""Agent-level exceptions.

A single home for every exception the agent framework raises. `AgentError` is the
base; `CancelledError` is raised by a `CancellationToken` (see
`luca.agent.core.context`).

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


class ProjectionError(AgentError):
    """Conversation projection failed — a missing entry id, an unknown entry
    type, a nonterminal tool execution, or a COMPLETED execution without a
    result. Projection errors propagate; they are never converted into
    synthetic conversation content."""
