"""Core conversation data model for the luca agent framework.

A *declarative* definition of the agent's foundation types — no runtime
behavior lives here. Implements `data_model_proposal_v0.md`:

- Storage is a flat dict (`AgentSession.entries`); traversal is an ordered
  id list (`Conversation.nodes`).
- Messages *are* entries: `UserMessage` / `AssistantMessage` sit in
  `entries` next to `ToolExecution`, `TurnStart`, `TurnFinish`,
  `CompactionEntry`, `PrunedEntry`. One `Entry` base, one `type`
  discriminator.
- A tool call is two things: the request *block* inside the assistant
  message (`ToolCall`) and a separate, mutable `ToolExecution` entry that
  references it by `tool_call_id`.
- Context vs usage: `Entry.context_tokens` is the intrinsic estimated size
  of an entry's model-facing content; provider-reported `Usage` is accessory
  data on the conversation-entry relationship, stored in
  `AgentSession.usages[conversation_id][entry_id]` — never on the entry.

Pydantic v2 idioms only; `extra="forbid"` on every model.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _now_ms() -> int:
    """Unix ms — the default stamp for self-timestamping value objects."""
    return time.time_ns() // 1_000_000


# ── content parts (live INSIDE messages; not entries) ──────────────────────
# Each carries a `type` discriminator so the AssistantMessage.parts union
# round-trips through JSON unambiguously (signatures → pass 2).


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

    model_config = ConfigDict(extra="forbid")


class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str

    model_config = ConfigDict(extra="forbid")


class ToolCall(BaseModel):
    """The tool-call REQUEST block, carried in AssistantMessage.parts."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ImageURL(BaseModel):
    kind: Literal["url"] = "url"
    url: str
    media_type: str | None = None

    model_config = ConfigDict(extra="forbid")


class ImageBase64(BaseModel):
    kind: Literal["base64"] = "base64"
    data: str  # base64-encoded bytes, no `data:` prefix
    media_type: str

    model_config = ConfigDict(extra="forbid")


class ImageFileId(BaseModel):
    kind: Literal["file"] = "file"
    file_id: str
    media_type: str | None = None

    model_config = ConfigDict(extra="forbid")


ImageSource = Annotated[
    Union[ImageURL, ImageBase64, ImageFileId],
    Field(discriminator="kind"),
]


class ImageContent(BaseModel):
    """An image carried by a `UserMessage`.

    `metadata` is application-owned and deliberately NOT projected — the
    client's `ImageBlock` carries only a source. It survives in the session,
    so a replayed transcript can still describe an image whose original file
    has since been deleted (`name`, `path`, `size_bytes`, …)."""

    type: Literal["image"] = "image"
    source: ImageSource
    metadata: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


# Tool-result content. Text-only by design: `ImageContent` is a user-message
# part. Widening this to carry tool-result images touches `ExecutionResult`,
# `PrunedEntry` and the tool-execution projection, and is a separate change.
Content = TextContent


# ── value objects (carried by entries) ─────────────────────────────────────


class BaseConfigModel(BaseModel):
    """Shared base for all config value objects: inherits `extra="forbid"` and
    provides a free-form `extras` dict the core never interprets."""

    extras: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class LLMConfig(BaseConfigModel):
    model: str  # e.g. "openai/gpt-5.4-mini"
    provider: str  # e.g. "openrouter"
    reasoning_effort: str | None = None


class Usage(BaseModel):
    """Provider-reported consumption for one entry in one conversation — a
    self-describing association record, not an intrinsic entry fact. The same
    assistant entry can appear in several conversations (a fork), and the
    input reported for it depends on the whole context sent with that request,
    so usage lives in `AgentSession.usages[conversation_id][entry_id]` rather
    than on the entry. The ids are part of the record because usage only has
    meaning together with the conversation-entry relationship it describes.

    Never a measure of the entry's own content — that is
    `Entry.context_tokens`."""

    conversation_id: str
    entry_id: str

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0

    model_config = ConfigDict(extra="forbid")


class ExecutionResult(BaseModel):
    """What the tool body RETURNED. Exists iff the framework received a return
    value (`status=COMPLETED` under framework-produced state). `is_error` is
    the tool's own verdict about its result — a file-reading tool may return a
    useful "file does not exist" result with `is_error=True` and still be
    COMPLETED. Execution timing lives on `ToolExecution`, not here."""

    content: list[Content]  # what the LLM sees
    metadata: dict = Field(default_factory=dict)  # e.g. {"exit_code": 0}
    is_error: bool = False  # the tool's verdict about the returned result

    model_config = ConfigDict(extra="forbid")


# ── system prompt parts (handed to the runner via system_prompt_parts) ──────


class SystemPromptPart(BaseModel):
    """One fragment of the system prompt. `source` records provenance
    ("model", "agents.md", "env", "skills", …); `priority` orders fragments
    before assembly (ascending; -1 = unranked, sorts first)."""

    text: str
    source: str = "model"
    priority: int = -1

    model_config = ConfigDict(extra="forbid")


# ── tool classification + approval vocabulary ───────────────────────────────


class ToolKind(str, Enum):
    READ = "read"
    SEARCH = "search"
    WEB_FETCH = "web_fetch"
    EDIT = "edit"
    MOVE = "move"
    DELETE = "delete"
    EXECUTE = "execute"
    SWITCH_MODE = "switch_mode"
    OTHER = "other"


class ApprovalOption(str, Enum):
    """The state of one ApprovalDecision: resolved (ALLOW / DENY) or punted
    back to the application (PENDING — decide again on the next run)."""

    ALLOW = "allow"
    DENY = "deny"
    PENDING = "pending"


# ── tool identity snapshot ──────────────────────────────────────────────────


class ToolSpec(BaseModel):
    """Historical snapshot of the RESOLVED tool's identity and classification,
    captured when the tool is resolved — so an old conversation can describe
    the tool selected at the time even if the registry later changes or drops
    it. Carries no invocation arguments: those belong to
    `ToolExecution.raw_tool_call`."""

    name: str
    description: str | None = None
    metadata: dict | None = None
    tool_kind: ToolKind = ToolKind.OTHER  # permission/classification kind
    namespace: str | None = None  # owning tool group, e.g. "builtin.shell_tools"
    version: str | None = None  # tool version at call time, e.g. "0.0.1"
    # The tool's declared execution deadline (ms), stamped by the registry at
    # birth; None when the tool declares none. The runner reads it at dispatch,
    # falling back to RuntimeConfig.tool_execution_timeout_in_ms.
    timeout_in_ms: int | None = None

    model_config = ConfigDict(extra="forbid")


# ── approval decision (produced by the tool registry's decide()) ────────────


class ApprovalDecision(BaseModel):
    """One approval verdict for a whole tool call (per-call, all-or-nothing).
    Produced by the runner's `ToolRegistry.decide()` and appended to
    `ToolExecution.approval_decisions`. How it was decided — modes, rules, a
    human, a web service — is registry/application logic; `metadata` is
    free-form provenance (e.g. {"via": "rule"}) the core never interprets."""

    decision: ApprovalOption
    metadata: dict | None = None
    created_at: int = Field(default_factory=_now_ms)  # unix ms

    model_config = ConfigDict(extra="forbid")


# ── tool execution ──────────────────────────────────────────────────────────


class ExecutionStatus(str, Enum):
    """The framework's execution lifecycle — and ONLY that. Approval state
    lives on `ToolExecution.approval_status`; the tool's own verdict about a
    returned result lives on `ExecutionResult.is_error` (COMPLETED means "the
    framework received a result", not "the tool considers it a success")."""

    PENDING = "pending"  # body not started, no terminal outcome
    RUNNING = "running"  # body started, no terminal outcome
    COMPLETED = "completed"  # the body returned an ExecutionResult
    FAILED = "failed"  # tool-owned code raised (preflight or after dispatch)
    NOT_FOUND = "not_found"  # the requested/effective tool could not be resolved
    INVALID = "invalid"  # the requested/effective arguments failed validation
    REJECTED = "rejected"  # the registry's decide() denied the execution
    CANCELLED = "cancelled"  # cancellation prevented the body from starting
    INTERRUPTED = "interrupted"  # a started body did not finish (grace/orphan)
    TIMED_OUT = "timed_out"  # the framework-enforced deadline expired


class ApprovalStatus(str, Enum):
    """Current approval state, orthogonal to the execution lifecycle. `None`
    on the execution means no decision has ever processed it."""

    PENDING = "pending"  # the policy explicitly deferred its decision
    ALLOWED = "allowed"  # the policy allowed dispatch
    REJECTED = "rejected"  # the policy denied dispatch


class ToolExecutionError(BaseModel):
    """Durable record of a framework or tool failure that did NOT produce an
    `ExecutionResult`. Populated for FAILED / NOT_FOUND / INVALID; the other
    resultless terminal statuses are complete lifecycle facts on their own.
    `details` must stay JSON-clean; structured validation errors nest under an
    `"errors"` key. No traceback is stored by default."""

    error_type: str
    error_message: str
    details: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


# ── shared entry base ───────────────────────────────────────────────────────


class Entry(BaseModel):
    id: str
    parent_id: str | None = None  # RECOVERY BACKSTOP ONLY — never traversed
    created_at: int  # unix ms (one clock everywhere)
    type: str  # discriminator: "user" | "assistant" | "tool_execution" | ...
    # Estimated size of THIS entry's model-facing content — intrinsic to the
    # entry and shared by every conversation that references it. Calculated by
    # the runner's ContextManager before entry middleware runs; never derived
    # from provider usage. Markers (turn_start/turn_finish/cancel_requested)
    # and nonterminal tool executions stay 0.
    context_tokens: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid")


# ── message entries ──────────────────────────────────────────────────────────


UserPart = Annotated[
    Union[TextContent, ImageContent],
    Field(discriminator="type"),
]


class UserMessage(Entry):
    type: Literal["user"] = "user"
    parts: list[UserPart]


class AssistantMessage(Entry):
    type: Literal["assistant"] = "assistant"
    parts: list[
        Annotated[
            Union[TextContent, ThinkingContent, ToolCall],
            Field(discriminator="type"),
        ]
    ]
    llm_config: LLMConfig  # provenance: the config that PRODUCED this message
    stop_reason: str  # "stop" | "tool_use"  (error/aborted → pass 2)
    # NO usage field: provider consumption is conversation-scoped accessory
    # data — see `AgentSession.usages`.


# ── tool execution entry (the ONLY mutable entry) ─────────────────────────────


class ToolExecution(Entry):
    """The mutable, durable record of how the framework handles one model
    `ToolCall` — the source of truth about that call's whole lifecycle.

    `raw_tool_call` is initialized from the model's request and makes the
    execution self-contained; middleware may replace or mutate it (the
    original assistant message stays in the session history). `tool_call_id`
    deliberately duplicates `raw_tool_call.id` — it is the durable correlation
    key for the execution index and the LLM wire protocol.

    Approval is read from `approval_status`, never reconstructed from the
    log: `approval_decisions` is an append-only audit trail of registry
    responses (repeated PENDING responses are valid and stay visible).
    `extras` is a free-form dict written by registries and middleware —
    opaque to the core, never interpreted by it. An execution that is born
    terminal keeps `approval_status=None` — no decision ever processed it.

    `updated_at` is ledger bookkeeping (any persisted mutation may touch it);
    it is NOT an execution-end timestamp — use `started_at` / `ended_at`.

    The Pydantic model enforces no cross-field invariants: the framework
    produces the documented combinations, and application middleware is
    trusted to own the consequences of any state it authors."""

    type: Literal["tool_execution"] = "tool_execution"
    tool_call_id: str  # → duplicates raw_tool_call.id (correlation key)
    raw_tool_call: ToolCall  # the (possibly middleware-effective) request
    tool_spec: ToolSpec | None = None  # resolved-tool snapshot; None if unresolved

    extras: dict = Field(default_factory=dict)
    approval_status: ApprovalStatus | None = None
    approval_decisions: list[ApprovalDecision] = Field(default_factory=list)

    status: ExecutionStatus = ExecutionStatus.PENDING
    result: ExecutionResult | None = None
    error: ToolExecutionError | None = None

    started_at: int | None = None  # unix ms; set iff the body was dispatched
    ended_at: int | None = None  # unix ms; set on every terminal transition
    cancel_signalled_at: int | None = None  # unix ms; run cancellation only

    updated_at: int | None = None
    is_doom_loop_flagged: bool = False

    @property
    def dispatched(self) -> bool:
        """True iff the tool body was committed for dispatch."""
        return self.started_at is not None

    @property
    def duration_ms(self) -> int | None:
        """Body wall-clock duration, when both endpoints exist."""
        if self.started_at is None or self.ended_at is None:
            return None
        return self.ended_at - self.started_at


# ── loop boundary markers (no "Turn" object) ──────────────────────────────────


class TurnOutcome(str, Enum):
    """How a turn bracket closed. A `TurnFinish` means "this turn is over",
    not "the agent answered"."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"  # user ended the turn
    TIMED_OUT = "timed_out"  # LLM total/httpx timeout ended the attempt
    ERRORED = "errored"  # any other completion failure


class TurnStart(Entry):
    type: Literal["turn_start"] = "turn_start"


class TurnFinish(Entry):
    """Boundary and outcome record only. Turn usage is derived from the
    conversation-scoped records in `AgentSession.usages`, never duplicated
    on the marker."""

    type: Literal["turn_finish"] = "turn_finish"
    outcome: TurnOutcome = TurnOutcome.COMPLETED
    error: str | None = None  # detail for TIMED_OUT / ERRORED


class CancelRequested(Entry):
    """A durable cancellation request, appended inside the open turn by
    `cancel()`. Consumed by the wind-down that writes the closing TurnFinish —
    which runs at the engine's step boundaries AND at the turn-close sites, so
    an unconsumed instance controls every close (a within-grace LLM answer is
    recorded but the requested outcome still wins). An open turn with a
    trailing-unconsumed instance derives CANCELLING. Consumed instances
    accumulate across turns as an audit trail. The wire projection drops it
    (bookkeeping)."""

    type: Literal["cancel_requested"] = "cancel_requested"
    outcome: TurnOutcome = TurnOutcome.CANCELLED
    error: str | None = None

    @field_validator("outcome")
    @classmethod
    def _not_completed(cls, value: TurnOutcome) -> TurnOutcome:
        if value == TurnOutcome.COMPLETED:
            raise ValueError("a cancellation cannot request outcome=COMPLETED")
        return value


# ── compaction ────────────────────────────────────────────────────────────────


class CompactionEntry(Entry):
    type: Literal["compaction"] = "compaction"
    summary: str
    summarized: list[str]  # ids this entry replaced — self-describing span
    details: dict = Field(default_factory=dict)


# ── pruning ───────────────────────────────────────────────────────────────────


class PrunedEntry(Entry):
    """Durable replacement for one original entry in a conversation path.

    Pruning replaces the entry's *contribution* to the path without mutating
    or deleting the original: `pruned_entry_id` references the untouched
    original in `AgentSession.entries` (an entry id, never an internal id like
    a tool-call id), `pruned_entry_type` records — and must agree with — the
    referenced entry's `type`, and `content` is the replacement the model sees
    instead. Produced by `ContextManager.prune_entry()`; persisted through the
    ledger's prune door, which swaps the node id in place so the path remains
    the traversal authority. Context aggregation therefore counts this entry's
    own `context_tokens`, not the replaced entry's."""

    type: Literal["pruned"] = "pruned"
    pruned_entry_type: str
    pruned_entry_id: str
    content: list[Content]


# The durable, uniformly-addressable node space. Discriminated on `type` so a
# dict[str, AnyEntry] deserializes each value to its concrete subclass.
AnyEntry = Annotated[
    Union[
        UserMessage,
        AssistantMessage,
        ToolExecution,
        TurnStart,
        TurnFinish,
        CancelRequested,
        CompactionEntry,
        PrunedEntry,
    ],
    Field(discriminator="type"),
]


# ── runtime context ───────────────────────────────────────────────────────────

Inf = -1  # "infinite / disabled" sentinel for RuntimeConfig's int fields


def Seconds(s: int | float) -> int:
    """Convenience: seconds → the int milliseconds RuntimeConfig fields take."""
    return int(s * 1000)


def MilliSeconds(ms: int) -> int:
    """Convenience: an explicit-unit identity for RuntimeConfig's ms fields."""
    return ms


class RuntimeConfig(BaseConfigModel):
    """Behavioral knobs for the runner, persisted with the session via
    `SessionConfig` and read live on every use (no constructor kwargs).
    Durations are int milliseconds, converted to float seconds at the
    asyncio/client boundary; `Inf` (-1) means infinite / disabled. The
    defaults reproduce the unconfigured behavior exactly."""

    # → client timeout= (httpx, per-phase). INERT when the runner is built
    # with a provider INSTANCE — the client leaves pre-built providers
    # untouched ("caller drives the lifecycle"); the wall-clock tier below
    # always applies.
    builtin_client_completion_timeout_in_ms: int = Inf
    client_completion_timeout_in_ms: int = Inf  # → client total_timeout= (wall clock)
    tool_execution_timeout_in_ms: int = Inf

    llm_completion_cancellation_grace_period: int = 0  # ms; 0 = immediate teardown
    tool_cancellation_grace_period: int = 0  # ms; 0 = straight to hard cancel

    # Step limits. A "step" is one AssistantMessage in the current turn.
    # Inf (-1) or 0 disables the limit.
    soft_max_steps: int = Inf
    hard_max_steps: int = Inf

    # Doom-loop detection: flag a ToolExecution when the same tool call has
    # been repeated this many consecutive times in the current turn.
    # Inf (-1) or 0 disables detection.
    doom_loop_threshold: int = Inf

    # When soft_max_steps is reached, pass tool_choice="none" to the LLM.
    limit_tool_choice_on_soft_max_steps_reached: bool = True
    # When a doom-loop-flagged execution exists in the current turn, pass
    # tool_choice="none" to the LLM.
    limit_tool_choice_on_doom_loop_flagged: bool = True

    @field_validator(
        "builtin_client_completion_timeout_in_ms",
        "client_completion_timeout_in_ms",
        "tool_execution_timeout_in_ms",
        "llm_completion_cancellation_grace_period",
        "tool_cancellation_grace_period",
        "soft_max_steps",
        "hard_max_steps",
        "doom_loop_threshold",
    )
    @classmethod
    def _inf_or_natural(cls, value: int) -> int:
        if value < Inf:
            raise ValueError(f"must be >= {Inf} ({Inf} = infinite / disabled)")
        return value


class SessionConfig(BaseConfigModel):
    """Global session configuration, persisted with the session. `llm_config`
    is the default for the NEXT turn. Tool approval lives entirely outside
    the session: the runner's `ToolRegistry` is runtime collaboration, not
    state."""

    llm_config: LLMConfig
    runtime_config: RuntimeConfig = Field(default_factory=RuntimeConfig)


# ── traversal + container ──────────────────────────────────────────────────────


class ConversationStatus(str, Enum):
    """Global status of a conversation, set by the runner. The data model only
    *supports* it (a plain, persisted field); it does NOT enforce the transitions
    (that is the runner's job). The runner re-derives/normalizes it from the
    entries on load, so it doubles as a denormalized cache of the entry state."""

    IDLE = "idle"  # nothing queued; awaiting a user message
    PENDING = "pending"  # work queued (message, resolved approvals, retry) — call run()
    RUNNING = "running"  # a run is actively driving (internal; crash-recovery only)
    AWAITING_APPROVAL = "awaiting_approval"  # paused at a tool-approval gate
    CANCELLING = "cancelling"  # unconsumed CancelRequested — the next drive flushes


class Conversation(BaseModel):
    id: str  # stable identity of this path/branch — usage records key on it
    nodes: list[str] = Field(default_factory=list)  # ordered entry ids = THE path
    created_at: int
    updated_at: int
    status: ConversationStatus = ConversationStatus.IDLE

    model_config = ConfigDict(extra="forbid")


class SessionRuntimeStatus(BaseModel):
    """Derived view of a session's current runtime state — always recomputed
    from the entry log, never trusted from serialized data. Accessed via the
    `AgentSession.session_runtime_status` computed field."""

    status: ConversationStatus = ConversationStatus.IDLE
    turn_count: int = 0  # number of TurnStart entries (includes the open turn)
    step_count: int = 0  # AssistantMessages in the currently open turn

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def get_runtime_status_from_agent_session(
        cls, session: AgentSession,
    ) -> SessionRuntimeStatus:
        turn_count = sum(
            1 for e in session.entries.values() if isinstance(e, TurnStart)
        )
        nodes = session.active_conversation.nodes
        entries = session.entries
        open_idx: int | None = None
        for i in range(len(nodes) - 1, -1, -1):
            entry = entries[nodes[i]]
            if isinstance(entry, TurnFinish):
                break
            if isinstance(entry, TurnStart):
                open_idx = i
                break
        step_count = 0
        if open_idx is not None:
            step_count = sum(
                1
                for node_id in nodes[open_idx:]
                if isinstance(entries[node_id], AssistantMessage)
            )
        return cls(
            status=session.active_conversation.status,
            turn_count=turn_count,
            step_count=step_count,
        )


class AgentSession(BaseModel):
    id: str
    entries: dict[str, AnyEntry] = Field(default_factory=dict)  # append-only store
    tool_executions: dict[str, list[str]] = Field(default_factory=dict)  # denorm index
    # Accessory provider-usage records, conversation-first so per-conversation
    # reads and cleanup are direct: usages[conversation_id][entry_id] → Usage.
    # Written only through `SessionLedger.record_usage()`.
    usages: dict[str, dict[str, Usage]] = Field(default_factory=dict)
    active_conversation: Conversation
    conversation_history: list[Conversation] = Field(default_factory=list)
    session_config: SessionConfig

    model_config = ConfigDict(extra="forbid")

    @property
    def session_runtime_status(self) -> SessionRuntimeStatus:
        """Always recomputed from entries — never trust serialized values."""
        return SessionRuntimeStatus.get_runtime_status_from_agent_session(self)

    @property
    def status(self) -> ConversationStatus:
        """The active conversation's status (convenience accessor)."""
        return self.active_conversation.status
