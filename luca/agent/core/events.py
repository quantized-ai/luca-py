"""Agent events — the informational stream an `AgentRun` yields (iterate the
handle inside `async with`, or receive them through `on_event`).

Purely observational: nothing here controls behavior (approval gating, status,
etc. are resolved on the session *before* the corresponding event is emitted).
This replaces the old `EventListener` observer.

Two tiers, selected by the run's `streaming=` flag:

- BLOCK events fire in BOTH modes, once a block is complete:
  `ReasoningBlock`, `TextBlock`, `ToolCallReceived`, `ToolExecutionStarted`,
  `ToolExecuted`, `FinishReason`.
- DELTA / `*Start` events fire ONLY under `streaming=True`, as tokens arrive:
  `ReasoningStart`/`ReasoningDelta`, `TextStart`/`TextDelta`, `ToolCallStart`.

Plus one lifecycle event, `ApprovalRequired`, emitted as the last event before
the engine parks for external approval.

The tool-lifecycle events carry the durable `ToolExecution` itself rather than
a hand-picked subset of its fields. Every carried execution is a DEEP SNAPSHOT
taken at emission time: an event represents state at one moment, and later
approval decisions, status transitions, or middleware changes must never
retroactively change an event already emitted or buffered. Events follow the
persistence of the state their snapshot represents — the stream never leads
the durable session.

It is one shared discriminated union: a non-streaming run yields the block +
lifecycle subset, `streaming=True` adds the delta events — so a `match`
written for one also works for the other. The textual payload is `text` on
every text-bearing event (block and delta alike) for ergonomic pattern
matching. There is deliberately no `TurnFinished` event — `RunResult` is the
completion signal (a cancel flush may emit zero events).
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from .models import ToolExecution


# ── block events (fire in both modes) ──────────────────────────────────────────


class ReasoningBlock(BaseModel):
    type: Literal["reasoning_block"] = "reasoning_block"
    text: str

    model_config = ConfigDict(extra="forbid")


class TextBlock(BaseModel):
    type: Literal["text_block"] = "text_block"
    text: str

    model_config = ConfigDict(extra="forbid")


class ToolCallReceived(BaseModel):
    """Emitted once per model `ToolCall`, after the newborn `ToolExecution`
    has been persisted and before approval or dispatch begins. The snapshot
    shows the birth state: PENDING with `approval_status=None`, or a
    preflight-terminal NOT_FOUND / INVALID / FAILED. `tool_call_id` is
    denormalized on purpose — it is the correlation key most consumers use.
    Tool name and arguments live on `execution.raw_tool_call`."""

    type: Literal["tool_call_received"] = "tool_call_received"
    tool_call_id: str
    execution: ToolExecution  # deep snapshot — never a live ledger reference

    model_config = ConfigDict(extra="forbid")


class ToolExecutionStarted(BaseModel):
    """Emitted if and only if the tool body is dispatched: after the execution
    has been persisted as RUNNING (with `started_at`), immediately before the
    body is invoked. `execution.raw_tool_call` reflects the effective call
    after `before_tool_execution`, which may differ from the earlier
    `ToolCallReceived` snapshot."""

    type: Literal["tool_execution_started"] = "tool_execution_started"
    tool_call_id: str
    execution: ToolExecution  # deep snapshot — never a live ledger reference

    model_config = ConfigDict(extra="forbid")


class ToolExecuted(BaseModel):
    """Emitted once per execution when it reaches the terminal outcome that
    will be projected as the correlated tool output — after outcome middleware
    and the final persistence. `execution` answers what happened;
    `result_text` / `is_error` answer what the model is told (both derive from
    the same `ConversationProjector.project_tool_execution` output used for
    the next LLM request, so `is_error` is a projection flag, not a substitute
    for `execution.status`)."""

    type: Literal["tool_executed"] = "tool_executed"
    tool_call_id: str
    execution: ToolExecution  # deep snapshot — never a live ledger reference
    result_text: str
    is_error: bool

    model_config = ConfigDict(extra="forbid")


class FinishReason(BaseModel):
    type: Literal["finish_reason"] = "finish_reason"
    finish_reason: str | None

    model_config = ConfigDict(extra="forbid")


# ── delta / start events (fire only under streaming=True) ──────────────────────


class ReasoningStart(BaseModel):
    type: Literal["reasoning_start"] = "reasoning_start"

    model_config = ConfigDict(extra="forbid")


class ReasoningDelta(BaseModel):
    type: Literal["reasoning_delta"] = "reasoning_delta"
    text: str

    model_config = ConfigDict(extra="forbid")


class TextStart(BaseModel):
    type: Literal["text_start"] = "text_start"

    model_config = ConfigDict(extra="forbid")


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str

    model_config = ConfigDict(extra="forbid")


class ToolCallStart(BaseModel):
    type: Literal["tool_call_start"] = "tool_call_start"
    tool_call_id: str
    name: str

    model_config = ConfigDict(extra="forbid")


# ── lifecycle ──────────────────────────────────────────────────────────────────


class ApprovalRequired(BaseModel):
    """Emitted as the final event before the run parks for external approval,
    only after every currently runnable sibling has advanced. Carries deep
    snapshots of the executions whose `approval_status=PENDING` — the same
    list `runner.pending_approvals()` returns. Each is self-contained:
    `raw_tool_call` identifies the call, `extras` carries whatever free-form
    vocabulary the registry recorded for the application's UI to work from."""

    type: Literal["approval_required"] = "approval_required"
    executions: list[ToolExecution]

    model_config = ConfigDict(extra="forbid")


AgentEvent = Annotated[
    Union[
        ReasoningBlock,
        TextBlock,
        ToolCallReceived,
        ToolExecutionStarted,
        ToolExecuted,
        FinishReason,
        ReasoningStart,
        ReasoningDelta,
        TextStart,
        TextDelta,
        ToolCallStart,
        ApprovalRequired,
    ],
    Field(discriminator="type"),
]
