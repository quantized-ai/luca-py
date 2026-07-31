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

Plus the LIFECYCLE events, which fire in both modes: `SubagentsSpawned`,
`ApprovalRequired`,
emitted as the last event before the engine parks for external approval, and
the three compaction events (`CompactionScheduled` / `CompactionStarted` /
`CompactionFinished`), which map one-to-one onto the tool events — same
lifecycle shape, same payload discipline.

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

EVERY event carries a `conversation_id` (see `AgentEventBase`). A run's stream
is the stream of its whole conversation subtree, so a consumer routes by that
field; without it the ordinary "render every event" loop could not tell the
main agent's text from a subagent's.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import AnyEntry, CompactionEntry, ToolExecution, TurnOutcome


class AgentEventBase(BaseModel):
    """What every agent event carries: the conversation it came from.

    A session drives several conversations at once — the main one and one per
    subagent — and they all arrive interleaved on the top-level run's stream,
    so an event that did not name its conversation could not be routed. This
    is a SEPARATE fact from `ToolExecution.conversation_id`, not a duplicate of
    it: the field here also has to serve text, reasoning and finish events,
    which carry no entry at all, while the field on the execution has to
    survive being handed to `decide()`, to middleware and to
    `pending_approvals()`, where there is no event. The three tool-lifecycle
    events carry both, and the two must always agree."""

    conversation_id: str

    model_config = ConfigDict(extra="forbid")


# ── block events (fire in both modes) ──────────────────────────────────────────


class ReasoningBlock(AgentEventBase):
    type: Literal["reasoning_block"] = "reasoning_block"
    text: str
    # A provider may withhold the reasoning body and return only its encrypted
    # attestation; `text` is then empty and there is nothing to render.
    redacted: bool = False


class TextBlock(AgentEventBase):
    type: Literal["text_block"] = "text_block"
    text: str


class ToolCallReceived(AgentEventBase):
    """Emitted once per model `ToolCall`, after the newborn `ToolExecution`
    has been persisted and before approval or dispatch begins. The snapshot
    shows the birth state: PENDING with `approval_status=None`, or a
    preflight-terminal NOT_FOUND / INVALID / FAILED. `tool_call_id` is
    denormalized on purpose — it is the correlation key most consumers use.
    Tool name and arguments live on `execution.raw_tool_call`."""

    type: Literal["tool_call_received"] = "tool_call_received"
    tool_call_id: str
    execution: ToolExecution  # deep snapshot — never a live ledger reference


class ToolExecutionStarted(AgentEventBase):
    """Emitted if and only if the tool body is dispatched: after the execution
    has been persisted as RUNNING (with `started_at`), immediately before the
    body is invoked. `execution.raw_tool_call` reflects the effective call
    after `before_tool_execution`, which may differ from the earlier
    `ToolCallReceived` snapshot."""

    type: Literal["tool_execution_started"] = "tool_execution_started"
    tool_call_id: str
    execution: ToolExecution  # deep snapshot — never a live ledger reference


class ToolExecuted(AgentEventBase):
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


class FinishReason(AgentEventBase):
    type: Literal["finish_reason"] = "finish_reason"
    finish_reason: str | None


# ── delta / start events (fire only under streaming=True) ──────────────────────


class ReasoningStart(AgentEventBase):
    type: Literal["reasoning_start"] = "reasoning_start"


class ReasoningDelta(AgentEventBase):
    type: Literal["reasoning_delta"] = "reasoning_delta"
    text: str


class TextStart(AgentEventBase):
    type: Literal["text_start"] = "text_start"


class TextDelta(AgentEventBase):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ToolCallStart(AgentEventBase):
    type: Literal["tool_call_start"] = "tool_call_start"
    tool_call_id: str
    name: str


# ── lifecycle ──────────────────────────────────────────────────────────────────


class ApprovalRequired(AgentEventBase):
    """Emitted as the final event before the run parks for external approval,
    only after every currently runnable sibling has advanced. Carries deep
    snapshots of the executions whose `approval_status=PENDING` — the same
    list `runner.pending_approvals()` returns. Each is self-contained:
    `raw_tool_call` identifies the call, `extras` carries whatever free-form
    vocabulary the registry recorded for the application's UI to work from."""

    type: Literal["approval_required"] = "approval_required"
    executions: list[ToolExecution]


class CompactionScheduled(AgentEventBase):
    """Emitted at the top of a drive, once the compaction bracket and its
    entry are on the path — for BOTH sources, since `schedule_compaction()`
    happens outside any run and has no stream to emit on. `entry.source` says
    who asked; `started_at` is None on a fresh open and carries the previous
    stamp on a resume.

    The gap between this and `CompactionStarted` is usually microseconds, but
    it is a real window a consumer may cancel in — which is why the two are
    separate events."""

    type: Literal["compaction_scheduled"] = "compaction_scheduled"
    entry: CompactionEntry  # deep snapshot — never a live ledger reference


class CompactionStarted(AgentEventBase):
    """Emitted once `started_at` is stamped, immediately before the policy's
    LLM call. `entry.llm_config` is still None here and cannot be otherwise:
    the policy chooses the model — it may be a cheaper one than the session's
    — and has not been called yet. The model that produced a summary is known
    at `CompactionFinished`."""

    type: Literal["compaction_started"] = "compaction_started"
    entry: CompactionEntry  # deep snapshot — never a live ledger reference


class CompactionFinished(AgentEventBase):
    """Emitted after the bracket closes or the transition commits — WHATEVER
    the outcome, exactly as a failed tool execution still emits `ToolExecuted`.
    For a policy-initiated failure, which degrades silently so the user's turn
    survives, this is the ONLY signal that anything went wrong.

    `entry.parts` set → summarized; None → nothing was done. `outcome` and
    `error` come from the closing `TurnFinish`, so no new vocabulary is
    introduced and there is no separate `CompactionFailed`.

    TWO conversation ids, and they are different facts. `conversation_id` (from
    `AgentEventBase`) is where the compaction RAN — the outgoing conversation,
    always set. `new_conversation_id` is the successor the transition
    installed, and is None whenever nothing was actually compacted."""

    type: Literal["compaction_finished"] = "compaction_finished"
    entry: CompactionEntry  # deep snapshot — never a live ledger reference
    outcome: TurnOutcome
    error: str | None = None  # detail for TIMED_OUT / ERRORED
    created: list[AnyEntry] = Field(default_factory=list)  # other plan entries
    new_conversation_id: str | None = None  # None if nothing changed


class SubagentsSpawned(AgentEventBase):
    """One assistant response's subagents, announced as ONE batch.

    A batch rather than one event per child, because one response can spawn
    several at once and an application driving them itself
    (`autostart_subagents=False`) would otherwise be forced to drive them
    sequentially — the whole point is `asyncio.gather` over
    `[run.child(cid) for cid in conversation_ids]`.

    Emitted BEFORE the children start, so a `False`-mode consumer can reach
    every handle inside the event's own branch. `conversation_id` (from the
    base) is the PARENT — the conversation that spawned them."""

    type: Literal["subagents_spawned"] = "subagents_spawned"
    conversation_ids: list[str]


AgentEvent = Annotated[
    ReasoningBlock
    | TextBlock
    | ToolCallReceived
    | ToolExecutionStarted
    | ToolExecuted
    | FinishReason
    | ReasoningStart
    | ReasoningDelta
    | TextStart
    | TextDelta
    | ToolCallStart
    | ApprovalRequired
    | CompactionScheduled
    | CompactionStarted
    | CompactionFinished
    | SubagentsSpawned,
    Field(discriminator="type"),
]
