"""Core conversation data model for the luca agent framework.

A *declarative* definition of the agent's foundation types — no runtime
behavior lives here. Implements `data_model_proposal_v0.md`:

- Storage is a flat dict (`AgentSession.entries`); traversal is an ordered
  id list (`Conversation.nodes`). Conversations are themselves a store —
  `AgentSession.conversations`, keyed by id, holding the main conversation,
  its archived predecessors and (later) one per subagent — with
  `main_conversation_id` as the pointer into it. Status is never stored:
  `AgentSession.get_conversation_status(id)` recomputes it from the nodes.
- Messages *are* entries: `UserMessage` / `AssistantMessage` sit in
  `entries` next to `ToolExecution`, `TurnStart`, `TurnFinish`,
  `CompactionEntry`, `ChildConversation`, `PrunedEntry`. One `Entry` base, one
  `type` discriminator.
- A tool call is two things: the request *block* inside the assistant
  message (`ToolCall`) and a separate, mutable `ToolExecution` entry that
  references it by `tool_call_id`.
- Context vs usage: `Entry.context_tokens` is the intrinsic estimated size
  of an entry's model-facing content; provider-reported `Usage` is accessory
  data on the conversation-entry relationship, stored in
  `AgentSession.usages[conversation_id][entry_id]` — never on the entry.
- Tool specs are NORMALIZED: each distinct `ToolSpec` is stored once in
  `AgentSession.tool_specs` under its content-derived `spec_id()`, and a
  `ToolExecution` references it by `tool_spec_id`. `tool_spec` stays on the
  execution as a restorable cache — see `ToolExecution` and `AgentSession`.

Pydantic v2 idioms only; `extra="forbid"` on every model.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)


def _now_ms() -> int:
    """Unix ms — the default stamp for self-timestamping value objects."""
    return time.time_ns() // 1_000_000


# ── content parts (live INSIDE messages; not entries) ──────────────────────
# Each carries a `type` discriminator so the AssistantMessage.parts union
# round-trips through JSON unambiguously (signatures → pass 2).


class TextContent(BaseModel):
    """Text carried by a message, a tool result or a pruned replacement.

    `metadata` is application-owned and deliberately NOT projected — the
    client's `TextBlock` carries only the text. It survives in the session, so
    a replayed transcript can describe where the text came from when that is
    not evident from the text itself (an `@`-mention's inlined file, say).
    Same contract as `ImageContent.metadata`: the core never interprets it."""

    type: Literal["text"] = "text"
    text: str
    metadata: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ThinkingContent(BaseModel):
    """The model's reasoning. `signature` is the provider's opaque attestation
    over it, durable because some providers reject a replayed thinking block
    without one; `id` is the provider's opaque identity FOR it, durable because
    OpenAI's Responses API replays a reasoning item only when its `rs_…` id
    comes back alongside the encrypted payload; `redacted` marks a block whose
    reasoning the provider encrypted, leaving `thinking` empty and the payload
    in `signature`.

    All three are provider-owned and must round-trip byte for byte: rewriting
    `thinking` in middleware invalidates the signature. Whether they go back on
    the wire is the transport's decision, not the projector's — an attestation
    minted by one (provider, model) pair is refused by every other, and the
    transport is what knows that."""

    type: Literal["thinking"] = "thinking"
    thinking: str
    id: str | None = None
    signature: str | None = None
    redacted: bool = False

    model_config = ConfigDict(extra="forbid")


class ToolCall(BaseModel):
    """The tool-call REQUEST block, carried in AssistantMessage.parts."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict = Field(default_factory=dict)
    # The client's generic-form native payload (`custom_type`, `item_id`,
    # `status`, …), captured verbatim when a provider-native call arrives —
    # empty for the ordinary function call. Opaque: the core stores it and
    # copies it, never interprets it; only the middleware that owns the native
    # tool reads it back at projection time. Excluded from doom-loop
    # comparison for free — `_is_doom_loop` compares name + arguments only.
    extras: dict = Field(default_factory=dict)

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
    ImageURL | ImageBase64 | ImageFileId,
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


# what a user message, a tool result or a pruned replacement carries
ContentPart = Annotated[
    TextContent | ImageContent,
    Field(discriminator="type"),
]

# what an assistant message carries — a different set, so a separate union
AssistantContentPart = Annotated[
    TextContent | ThinkingContent | ToolCall,
    Field(discriminator="type"),
]


# ── value objects (carried by entries) ─────────────────────────────────────


class BaseConfigModel(BaseModel):
    """Shared base for all config value objects: inherits `extra="forbid"` and
    provides a free-form `extras` dict the core never interprets."""

    extras: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class LLMConfig(BaseConfigModel):
    model: str  # e.g. "openai/gpt-5.4-mini"
    provider: str  # e.g. "openrouter"
    reasoning: str | None = None


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
    COMPLETED. Execution timing lives on `ToolExecution`, not here.

    Two output channels, and only one of them is model-facing: `content` is
    what the LLM sees, `structured_content` is what the APPLICATION reads."""

    content: list[ContentPart]  # what the LLM sees
    # The tool's machine-readable payload, ideally conforming to the
    # `output_schema` its `ToolSpec` declares. BEST EFFORT: nothing validates
    # it against that schema, `ConversationProjector` never puts it on the
    # wire, and `ContextManager` never counts it. A tool that wants the model
    # to see this serializes it into `content` itself. Distinct from
    # `metadata`, which stays free-form bookkeeping no schema describes.
    structured_content: dict | None = None
    metadata: dict = Field(default_factory=dict)  # e.g. {"exit_code": 0}
    is_error: bool = False  # the tool's verdict about the returned result
    # The client's generic-form RESULT payload — the mirror of
    # `ToolCall.extras`, written once by the tool's `execute()` when the
    # result has a provider-native shape (`custom_type`, `results`, …) and
    # empty otherwise. Opaque to the core; read back only by the middleware
    # that owns the native tool. Distinct from `metadata` (free-form
    # bookkeeping): this is model-facing data a projection middleware puts
    # back on the wire.
    extras: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ExecutionDeferred(BaseModel):
    """The tool cannot produce its final result yet.

    An EMPTY MARKER: the core learns exactly one new fact — "not yet" — and
    nothing more. Whatever the tool is waiting on, and whatever state that
    requires, belongs to the tool (`AgentSession.extras` is where a tool keeps
    state that must outlive the process).

    Returned by the prepared callable in place of an `ExecutionResult`. The
    execution moves to `AWAITING_RESULT` and the drive parks; there is no
    resume — the next drive RE-DISPATCHES the same call from scratch, through
    `before_tool_execution` → `prepare()` → a brand-new callable."""

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
    """The core's ONLY tool type — plain, language-neutral, JSON-serializable
    data. It plays two roles at once:

    - the ADVERTISEMENT sent to the model (`name`, `description`,
      `input_schema`), and
    - the historical IDENTITY SNAPSHOT attached to a past execution
      (`tool_kind`, `namespace`, `version`, `timeout_in_ms`), so an old
      conversation still describes the tool selected at the time even if the
      registry later changes or drops it.

    `output_schema` is neither: it is a declaration to the APPLICATION — and,
    since subagents, to the runner's spawn gate — but never to the model, since
    no provider accepts an output schema on a function tool.

    "The advertisement sent to the model" has exactly ONE exception:
    `is_private=True` marks a spec the runtime can resolve and dispatch but
    that is never put on the wire.

    Nothing here references a Python class: a session whose tools were deleted
    from the codebase years ago still renders its name, description, schema and
    kind. Carries no invocation arguments — those belong to
    `ToolExecution.raw_tool_call`.

    A `ToolSpec` MUST be a pure function of the tool definition, never of the
    call. Specs are normalized (stored once per session under `spec_id()`), so
    a registry that puts anything volatile in `metadata` — a timestamp, a
    request id — mints a new stored row on every single call and silently
    defeats the normalization, with no error and no warning."""

    name: str
    # Presentation ONLY: the human-facing label a UI renders (reach it through
    # `display_name`). `name` stays the stable internal identity every
    # framework lookup keys on — resolution, approvals, doom-loop, middleware.
    # Excluded from `spec_id()`, so giving a tool a title (or rewording one)
    # never invalidates stored spec ids.
    title: str | None = None
    description: str  # required: the client's wire tool type rejects null
    # The tool's arguments as a JSON Schema dict. Never a Pydantic model class
    # and never a TypeAdapter — the whole point is that it survives a round
    # trip through JSON. Required and never None, including for a tool that
    # takes no arguments: that case is the empty object schema,
    # `{"type": "object", "properties": {}}`. An absent schema and an empty
    # schema mean different things to a model provider.
    input_schema: dict
    # The shape of the machine-readable result this tool CAN produce, as a
    # JSON Schema dict — the declaration `ExecutionResult.structured_content`
    # is the payload for. Optional and ADVISORY in both directions: nothing in
    # the framework validates a payload against it, and no provider luca talks
    # to accepts an output schema on a function tool, so it is never sent on
    # the wire. It advertises to the APPLICATION (a registry, a UI, an MCP
    # bridge mapping `outputSchema`), not to the model. None = the tool
    # declares nothing, which is not the same as declaring an empty shape.
    output_schema: dict | None = None
    # NEVER ADVERTISED TO THE MODEL. A private tool is one the RUNTIME invokes:
    # `get_tools()` still returns it — that is how it resolves, prepares and
    # dispatches — but the runner omits it when building the wire tool list, and
    # a model tool call naming one records NOT_FOUND rather than resolving (from
    # the model's point of view that tool does not exist). Everything else is
    # untouched: the spec is filed under `spec_id()`, and an ordinary
    # `ToolExecution` is created with approvals, middleware, timeouts, events
    # and usage. Only its relationship to the WIRE changes.
    #
    # Like every other field it is definition-scoped and participates in
    # `spec_id()`, so a tool that gains it is a new row.
    is_private: bool = False
    metadata: dict | None = None  # free-form, registry-owned; never interpreted
    tool_kind: ToolKind = ToolKind.OTHER  # permission/classification kind
    namespace: str | None = None  # owning tool group, e.g. "builtin.shell_tools"
    version: str | None = None  # tool version at call time, e.g. "0.0.1"
    # The tool's declared execution deadline (ms), stamped by the registry at
    # birth; None when the tool declares none. The runner reads it at dispatch,
    # falling back to RuntimeConfig.tool_execution_timeout_in_ms. It bounds the
    # tool BODY only — not resolution, validation or any other registry phase.
    timeout_in_ms: int | None = None

    model_config = ConfigDict(extra="forbid")

    def spec_id(self) -> str:
        """This spec's content-derived id — the key it is filed under in
        `AgentSession.tool_specs`.

        A METHOD, not a field: a stored `spec_id` would be self-referential,
        and `extra="forbid"` would make a stray serialized key fail to load.

        The rule is pinned so the id is identical across processes, machines
        and independent implementations of luca in other languages, which is
        why it is deliberately NOT an overridable runner hook like
        `generate_id()` / `now_ms()`: here determinism is a data-integrity
        requirement, and a subclass that changed it would corrupt the store.

        1. Render the spec to its JSON representation (enums as their string
           values, `None`-valued fields included).
        2. Serialize with recursively sorted keys, no whitespace, non-ASCII
           emitted literally, encoded UTF-8.
        3. SHA-256 the bytes.  4. Hex-encode, full 64 characters.

        Not MD5 (raises on FIPS-enabled builds) and not truncated (every free
        parameter is a place a second implementation can silently diverge).

        JSON arrays are order-sensitive, so a reordered `required: [...]`, a
        Pydantic upgrade that changes `model_json_schema()` output, or a field
        added to `ToolSpec` later all mint a NEW id. All three heal the same
        way: one redundant stored row, with the old row still resolvable by the
        executions that point at it. A content hash's only failure mode is a
        redundant row, never a wrong lookup.

        `title` is excluded: it is presentation, not identity, and a UI
        relabeling a tool must not mint a new row."""
        payload = json.dumps(
            self.model_dump(mode="json", exclude={"title"}),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def display_name(self) -> str:
        """What a UI shows for this tool: `title` when one exists, the
        internal `name` otherwise."""
        return self.title or self.name


# ── approval decision (produced by the tool registry's decide()) ────────────


class ApprovalDecision(BaseModel):
    """One approval verdict for a whole tool call (per-call, all-or-nothing).
    Produced by the runner's `ToolRegistry.decide()` and appended to
    `ToolExecution.approval_decisions` — except on a toolless runner, which
    synthesizes an ALLOW itself ({"via": "toolless_runner"}): there is no
    gatekeeper, so nothing is gated, and the dispatch step records the honest
    NOT_FOUND terminal. How it was decided — modes, rules, a human, a web
    service — is registry/application logic; `metadata` is free-form
    provenance (e.g. {"via": "rule"}) the core never interprets."""

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

    RECEIVED = "received"  # the model asked for it; the registry has not answered yet
    PENDING = "pending"  # body not started, no terminal outcome
    RUNNING = "running"  # body started, no terminal outcome
    AWAITING_RESULT = "awaiting_result"  # the last dispatch returned ExecutionDeferred
    COMPLETED = "completed"  # the body returned an ExecutionResult
    FAILED = "failed"  # tool-owned code raised (preflight or after dispatch)
    NOT_FOUND = "not_found"  # the requested/effective tool could not be resolved
    INVALID = "invalid"  # the requested/effective arguments failed validation
    REJECTED = "rejected"  # the registry's decide() denied the execution
    REFUSED = "refused"  # a framework runtime limit refused the call before dispatch
    CANCELLED = "cancelled"  # cancellation prevented the body from starting
    INTERRUPTED = "interrupted"  # a started body did not finish (grace/orphan)
    TIMED_OUT = "timed_out"  # the framework-enforced deadline expired


# The statuses a drive can still advance. The single source of truth for every
# "nonterminal" rule — projection, pruning, context accounting, the cancel
# wind-down — so a status added to the lifecycle cannot be honoured in one of
# them and forgotten in the next.
NONTERMINAL_STATUSES = frozenset(
    {
        ExecutionStatus.RECEIVED,
        ExecutionStatus.PENDING,
        ExecutionStatus.RUNNING,
        ExecutionStatus.AWAITING_RESULT,
    }
)


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


class ExecutionAttemptOutcome(str, Enum):
    """How ONE invocation of a tool body ended.

    `INTERRUPTED` and `DISCARDED` are different facts and the distinction is
    the point: `INTERRUPTED` means the framework killed the body and knows it
    did not finish; `DISCARDED` means the process died and nobody ever saw how
    that run went."""

    DEFERRED = "deferred"  # returned ExecutionDeferred — not yet
    COMPLETED = "completed"  # returned an ExecutionResult
    FAILED = "failed"  # raised
    INTERRUPTED = "interrupted"  # the framework killed the body
    TIMED_OUT = "timed_out"  # the deadline expired
    DISCARDED = "discarded"  # the process died; this run's outcome was never seen


class ExecutionAttempt(BaseModel):
    """One invocation of a tool body, appended when the execution is persisted
    as `RUNNING` — after `prepare()` returned, before the callable is invoked —
    and closed with its outcome when the call settles.

    That placement IS the invariant: **a `prepare()` that raises appends no
    attempt**, on the first dispatch and the fifth alike. `outcome is None`
    means the attempt is still in flight (or the process died mid-flight, which
    orphan recovery closes as `DISCARDED`)."""

    outcome: ExecutionAttemptOutcome | None = None  # None while in flight
    started_at: int  # unix ms; the body was invoked
    ended_at: int | None = None  # unix ms; the invocation settled

    model_config = ConfigDict(extra="forbid")


# ── shared entry base ───────────────────────────────────────────────────────


class Entry(BaseModel):
    # `id` and `created_at` are None until the entry is COMMITTED: a template a
    # strategy builds (a registry's birth draft, a `PrunedEntry` replacement, a
    # compaction plan's new entry) carries no identity, and the persisting door
    # stamps both. Every entry in `AgentSession.entries` has them.
    id: str | None = None
    parent_id: str | None = None  # RECOVERY BACKSTOP ONLY — never traversed
    created_at: int | None = None  # unix ms (one clock everywhere)
    type: str  # discriminator: "user" | "assistant" | "tool_execution" | ...
    # Estimated size of THIS entry's model-facing content — intrinsic to the
    # entry and shared by every conversation that references it. Calculated by
    # the runner's ContextManager before entry middleware runs; never derived
    # from provider usage. Markers (turn_start/turn_finish/cancel_requested)
    # and nonterminal tool executions stay 0.
    context_tokens: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid")


# ── message entries ──────────────────────────────────────────────────────────


class UserMessage(Entry):
    type: Literal["user"] = "user"
    parts: list[ContentPart]


class AssistantMessage(Entry):
    type: Literal["assistant"] = "assistant"
    parts: list[AssistantContentPart]
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
    key for the LLM wire protocol.

    `tool_spec_id` is the DURABLE reference into `AgentSession.tool_specs`;
    `tool_spec` is a cache of the same spec, restored from that id whenever a
    session is constructed and stripped from a serialized session. The id is
    authoritative and `tool_spec` must never be the source of truth for
    anything durable — a deliberate exception to the rule that nothing on the
    session is transient, and now the only one.
    Both are `None` when the tool never resolved. Reading `tool_spec` works
    unchanged for every consumer, in memory and after a reload, which is what
    keeps the tool lifecycle EVENTS (which carry a deep copy of the execution
    and no session) self-describing.

    Approval is read from `approval_status`, never reconstructed from the
    log: `approval_decisions` is an append-only audit trail of registry
    responses (repeated PENDING responses are valid and stay visible).
    `extras` is a free-form dict written by registries and middleware —
    opaque to the core, never interpreted by it. An execution that is born
    terminal keeps `approval_status=None` — no decision ever processed it.

    `updated_at` is ledger bookkeeping (any persisted mutation may touch it);
    it is NOT an execution-end timestamp — use `created_at` / `finished_at`,
    or `attempts[-1]` for the wall-clock of the last body invocation.

    ONE CALL CAN BE DISPATCHED MANY TIMES. A tool may return
    `ExecutionDeferred` instead of a result; the execution parks at
    `AWAITING_RESULT` and the next drive re-dispatches it from scratch. So
    there is no single dispatch to timestamp: `created_at` (inherited) is the
    birth stamp, `finished_at` is the terminal transition, and each body
    invocation carries its own pair on its `ExecutionAttempt`.

    The Pydantic model enforces no cross-field invariants: the framework
    produces the documented combinations, and application middleware is
    trusted to own the consequences of any state it authors."""

    type: Literal["tool_execution"] = "tool_execution"
    tool_call_id: str  # → duplicates raw_tool_call.id (correlation key)
    # PROVENANCE, never traversal: the conversation this execution was BORN in.
    #
    # It is on THIS entry and no other because a `ToolExecution` is the only
    # entry a consumer ever receives DETACHED from a path — handed to
    # `ToolRegistry.decide()`, to the four tool middleware hooks, and carried as
    # a deep snapshot inside the tool events, `ApprovalRequired` and
    # `RunResult.pending_approvals`, none of which carry a session. Every other
    # entry reaches a consumer by walking `Conversation.nodes`, where the
    # conversation is already in hand. Do NOT generalize it onto `Entry`.
    #
    # Nothing may resolve a path through it: an entry CAN be referenced by two
    # conversations (a compaction carries node ids into its successor), so for a
    # historical execution this names the PREDECESSOR — under-reporting, never
    # wrong. For every LIVE use the birth conversation IS the current one,
    # because compaction never runs while a conversational turn is open.
    #
    # The RUNNER stamps it, alongside `id` / `parent_id` / `created_at`; a
    # registry's birth draft leaves it None and must not set it, which is why
    # the type is `str | None`.
    conversation_id: str | None = None
    raw_tool_call: ToolCall  # the (possibly middleware-effective) request
    tool_spec: ToolSpec | None = None  # restorable cache; None if unresolved
    tool_spec_id: str | None = None  # → AgentSession.tool_specs; the durable ref

    extras: dict = Field(default_factory=dict)
    approval_status: ApprovalStatus | None = None
    approval_decisions: list[ApprovalDecision] = Field(default_factory=list)

    status: ExecutionStatus = ExecutionStatus.PENDING
    result: ExecutionResult | None = None
    error: ToolExecutionError | None = None

    # AUDIT ONLY, and DERIVES NOTHING — not status, not result, not a source of
    # truth for anything. One entry per body invocation, in dispatch order.
    # UNBOUNDED: no cap, no fold. The driver contract keeps this at ~2 (a
    # handler must block until something really changed, so each attempt
    # corresponds to a real out-of-band event rather than a tick), and silent
    # truncation inside an audit log is worse than a large one. An application
    # that wants to trim does it in `after_tool_execution`, which runs exactly
    # once, when the list is final.
    attempts: list[ExecutionAttempt] = Field(default_factory=list)

    finished_at: int | None = None  # unix ms; set on every terminal transition
    cancel_signalled_at: int | None = None  # unix ms; run cancellation only

    updated_at: int | None = None
    is_doom_loop_flagged: bool = False

    @property
    def dispatched(self) -> bool:
        """True iff the tool body was committed for dispatch — an
        `ExecutionAttempt` exists iff the body was invoked."""
        return bool(self.attempts)

    @property
    def duration_ms(self) -> int | None:
        """TOTAL time this call was outstanding — the approval wait and any
        parked time included, NOT body wall-clock. Body time is per-attempt
        (`attempts[-1]`, or the sum)."""
        if self.finished_at is None or self.created_at is None:
            return None
        return self.finished_at - self.created_at


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


class CompactionSource(str, Enum):
    """Who asked for a compaction — the only fact about one that nothing else
    in the log records, which is why it is stored rather than derived."""

    USER = "user"  # runner.schedule_compaction()
    POLICY = "policy"  # the policy's should_compact() said so


class CompactionEntry(Entry):
    """The durable, MUTABLE lifecycle record of one compaction attempt — the
    second mutable entry type alongside `ToolExecution`. Written the moment a
    compaction is intended, mutated in place as it progresses, and left in its
    terminal state whether it succeeded or not.

    There is deliberately NO `status` field: the surrounding turn bracket owns
    how the attempt ended (`TurnFinish.outcome` + `error`) and these fields own
    what it produced, so nothing can disagree. Every state is readable from the
    two together:

    | Log state                                | Means                        |
    |------------------------------------------|------------------------------|
    | open bracket, `started_at is None`       | scheduled, not yet started   |
    | open bracket, `started_at` set           | running (or crashed mid-run) |
    | closed COMPLETED, `parts` set            | succeeded                    |
    | closed COMPLETED, `parts is None`        | nothing to compact           |
    | closed ERRORED / TIMED_OUT / CANCELLED   | failed; never retried        |

    `parts` is content, the same `ContentPart` list a `UserMessage` carries, so
    a summary may be several parts or carry an image. `None` (not `[]`) means
    nothing has been produced yet — the two are different facts.

    The entry carries into the new conversation while its bracket stays behind,
    so it has to be self-describing wherever it is read — which is why
    `started_at` / `ended_at` live here even though the bracket records the
    same instants."""

    type: Literal["compaction"] = "compaction"

    source: CompactionSource  # who asked

    parts: list[ContentPart] | None = None  # None → nothing produced yet
    compacted_nodes: list[str] | None = None  # None → nothing replaced
    llm_config: LLMConfig | None = None  # what PRODUCED the content

    started_at: int | None = None  # None → scheduled, not yet started
    ended_at: int | None = None

    metadata: dict = Field(default_factory=dict)  # policy-owned; opaque to core


# ── subagents ─────────────────────────────────────────────────────────────────


class ChildConversation(Entry):
    """A subagent, as a node in its PARENT's path — the THIRD mutable entry
    type, alongside `ToolExecution` and `CompactionEntry`.

    A subagent is not a new session and not a new runtime: it is one more
    conversation over the same stores, linked into its parent's path by this
    entry. That position is what makes the child's outcome appear in the
    parent's projected history at the right point; the child conversation
    itself lives in `AgentSession.conversations` like every other.

    - `conversation_id` names the child. A compaction of that child would
      re-point it, exactly as `main_conversation_id` moves for the main one —
      the name follows the conversation.
    - `tool_execution_id` names the spawn execution that created it, which is
      what carries the handshake payload (task id, prompt, description, and the
      tool that derives the result).
    - `execution_result` is the child's outcome, derived once its turn bracket
      closes — WHATEVER the outcome. A child that failed or timed out is a
      finished child whose result says so, not an exception travelling upward.
      `None` means unresolved; the parent's turn cannot CLOSE until every
      `ChildConversation` in its open turn has one (the model may be called
      mid-orchestration — resolved results project as task updates while the
      rest keep running).
    - `result_execution_id` names the result-tool execution that produced
      `execution_result`, stamped in the same write. It is what lets the
      projector render the result at the POSITION the resolution happened —
      after the assistant messages that predate it — instead of rewriting an
      earlier message in place. `None` with `execution_result` set means the
      link was resolved WITHOUT the result tool (a cancel wind-down or a
      hard-limit settle); only then does the link itself render.

    `context_tokens` is the size of what this contributes to the PARENT — its
    result, and only when the link itself renders it (`result_execution_id is
    None`); otherwise the result execution's own count covers the content. The
    child's own conversation does not count against the parent's window; that
    separation is the main reason subagents are useful."""

    type: Literal["child_conversation"] = "child_conversation"
    conversation_id: str
    tool_execution_id: str
    execution_result: ExecutionResult | None = None
    result_execution_id: str | None = None


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
    content: list[ContentPart]


# The durable, uniformly-addressable node space. Discriminated on `type` so a
# dict[str, AnyEntry] deserializes each value to its concrete subclass.
AnyEntry = Annotated[
    UserMessage
    | AssistantMessage
    | ToolExecution
    | TurnStart
    | TurnFinish
    | CancelRequested
    | CompactionEntry
    | ChildConversation
    | PrunedEntry,
    Field(discriminator="type"),
]


def is_compaction_bracket(
    nodes: Sequence[str],
    entries: Mapping[str, AnyEntry],
    turn_start_index: int,
) -> bool:
    """Is the bracket opened by `nodes[turn_start_index]` a COMPACTION bracket?

    One definition, four consumers — the resumable-compaction read, the
    status derivation's skip rule, `conversational_turn_count`, and
    `ConversationProjector.project()` — which is why it lives here, upstream of
    all of them, as a plain function rather than a method on any of them.

    The test is ADJACENCY: the node immediately after the `TurnStart` is a
    `CompactionEntry`. That pair is written by two consecutive appends, so it
    is exact by construction. "Contains a `CompactionEntry` anywhere in the
    span" would be true today too, but it couples a framework classification to
    policy behavior — the entry outlives its bracket and lands wherever a
    compaction plan puts it, so a policy that carries a turn marker without its
    pair could drop a summary inside an ordinary bracket and have it misread as
    a compaction.

    A bare `TurnStart` (nothing after it) is conversational, and so is any
    bracket with no `TurnStart` to anchor it."""
    following = turn_start_index + 1
    if following >= len(nodes):
        return False
    return isinstance(entries.get(nodes[following]), CompactionEntry)


# ── path derivations (pure functions of one path + the entry store) ───────────
#
# Every one of these is a pure function of `(nodes, entries)` — no session, no
# conversation object, no runtime state. They live HERE, upstream of both
# consumers, for the same reason `is_compaction_bracket` does: `SessionLedger`
# reads them bound to a conversation id, and `AgentSession.get_conversation_
# status` reads them to derive status. Two doors, one implementation; and
# `models.py` cannot import the ledger, since the ledger imports models.
#
# They RESOLVE TOLERANTLY: a node id with no entry behind it contributes
# nothing rather than raising. These are QUERIES over possibly hand-edited
# durable data — `pretty_print` promises that a broken session still prints,
# and a status read is not the place to discover corruption. The fail-loud
# rule stays where it belongs: `ConversationProjector` raises on a missing
# node, because that one is about to be sent to a model.


def open_turn_index(
    nodes: Sequence[str],
    entries: Mapping[str, AnyEntry],
) -> int | None:
    """Index of the `TurnStart` opening an unclosed turn, or None. Walking back
    from the leaf, a `TurnFinish` means the last turn is closed; a `TurnStart`
    seen first means that turn is still open."""
    for i in range(len(nodes) - 1, -1, -1):
        entry = entries.get(nodes[i])
        if isinstance(entry, TurnFinish):
            return None
        if isinstance(entry, TurnStart):
            return i
    return None


def open_turn_executions(
    nodes: Sequence[str],
    entries: Mapping[str, AnyEntry],
) -> list[ToolExecution]:
    """Every `ToolExecution` in the open turn, in path order."""
    index = open_turn_index(nodes, entries)
    if index is None:
        return []
    return [entry for node_id in nodes[index:] if isinstance(entry := entries.get(node_id), ToolExecution)]


def open_turn_cancel_requested(
    nodes: Sequence[str],
    entries: Mapping[str, AnyEntry],
) -> CancelRequested | None:
    """The unconsumed `CancelRequested` inside the open turn, or None (no open
    turn, or none requested). At most one can exist — `cancel()` refuses to
    stack a second; instances in closed turns are consumed."""
    index = open_turn_index(nodes, entries)
    if index is None:
        return None
    for node_id in nodes[index:]:
        if isinstance(entry := entries.get(node_id), CancelRequested):
            return entry
    return None


def open_compaction_entry(
    nodes: Sequence[str],
    entries: Mapping[str, AnyEntry],
) -> CompactionEntry | None:
    """The RESUMABLE `CompactionEntry` inside the open bracket, or None.

    `None` means "there is no compaction to resume", for any of three inputs
    that mean the same thing to every caller: no open bracket at all; an open
    CONVERSATIONAL turn; or an open compaction-shaped bracket whose entry
    already has `parts` — a compaction that already committed and whose markers
    a plan carried, not an interrupted attempt.

    That last test is the one that matters (G6). An open bracket is the SIGNAL
    that a compaction was interrupted; the entry's own state is the TEST."""
    index = open_turn_index(nodes, entries)
    if index is None:
        return None
    if not is_compaction_bracket(nodes, entries, index):
        return None
    entry = entries[nodes[index + 1]]  # is_compaction_bracket proved it resolves
    return None if entry.parts is not None else entry


def end_before_closed_compaction_brackets(
    nodes: Sequence[str],
    entries: Mapping[str, AnyEntry],
) -> int:
    """The length of the path with every trailing CLOSED compaction bracket
    dropped — the slice status derives from.

    A compaction bracket is not a conversational turn, and leaving it as the
    leaf gives a wrong answer silently: a compaction that closed behind a
    queued user message would bury it (the message stops being the leaf, so it
    stops deriving BUSY and is silently never answered).

    A trailing `TurnFinish` with no `TurnStart` to anchor it — a policy carried
    one without its pair — is NOT a compaction bracket: stop skipping and let
    the ordinary closed-turn rules apply."""
    end = len(nodes)
    while end > 0 and isinstance(entries.get(nodes[end - 1]), TurnFinish):
        start = None
        for i in range(end - 2, -1, -1):
            entry = entries.get(nodes[i])
            if isinstance(entry, TurnFinish):
                break
            if isinstance(entry, TurnStart):
                start = i
                break
        if start is None or not is_compaction_bracket(nodes, entries, start):
            break
        end = start
    return end


def conversational_turn_count(
    nodes: Sequence[str],
    entries: Mapping[str, AnyEntry],
) -> int:
    """`TurnStart` entries on the path, excluding compaction brackets — a
    compaction is not a turn. Scoped to the path because entries outlive their
    conversation (a compaction archives one and opens another over the same
    store)."""
    return sum(
        1
        for index, node_id in enumerate(nodes)
        if isinstance(entries.get(node_id), TurnStart) and not is_compaction_bracket(nodes, entries, index)
    )


def open_turn_step_count(
    nodes: Sequence[str],
    entries: Mapping[str, AnyEntry],
) -> int:
    """`AssistantMessage` entries in the currently open turn."""
    index = open_turn_index(nodes, entries)
    if index is None:
        return 0
    return sum(1 for node_id in nodes[index:] if isinstance(entries.get(node_id), AssistantMessage))


def open_turn_unresolved_children(
    nodes: Sequence[str],
    entries: Mapping[str, AnyEntry],
) -> list[ChildConversation]:
    """The subagents spawned in the open turn that have not produced a result.

    These are what keeps a parent's turn OPEN: it cannot close until every one
    has resolved (the model may still be called mid-orchestration — see
    `open_turn_unseen_material`)."""
    index = open_turn_index(nodes, entries)
    if index is None:
        return []
    return [
        entry
        for node_id in nodes[index:]
        if isinstance(entry := entries.get(node_id), ChildConversation) and entry.execution_result is None
    ]


def open_turn_has_advanceable_executions(
    nodes: Sequence[str],
    entries: Mapping[str, AnyEntry],
) -> bool:
    """Is any execution in the open turn still the DRIVE's to move? A
    `RECEIVED` execution has not reached the registry yet, so the next drive
    births it; a persisted `RUNNING` execution is an orphan the next drive
    recovers; a `PENDING` execution the policy has not resolved (or has
    ALLOWED) is decidable or dispatchable. A gated (`approval_status` PENDING)
    execution is deliberately NOT advanceable — the application must act.

    Neither is a PARKED one (`AWAITING_RESULT`), for the identical reason and
    with no code of its own: it falls through every clause above. The tool
    said "not yet", and only the application can make it say otherwise."""
    for execution in open_turn_executions(nodes, entries):
        if execution.status in (ExecutionStatus.RECEIVED, ExecutionStatus.RUNNING):
            return True
        if execution.status == ExecutionStatus.PENDING and execution.approval_status in (
            None,
            ApprovalStatus.ALLOWED,
        ):
            return True
    return False


def open_turn_has_awaiting_executions(
    nodes: Sequence[str],
    entries: Mapping[str, AnyEntry],
) -> bool:
    """Is any execution in the open turn gated — PENDING with an explicitly
    deferred approval the application must answer out of band?"""
    return any(
        execution.status == ExecutionStatus.PENDING and execution.approval_status == ApprovalStatus.PENDING
        for execution in open_turn_executions(nodes, entries)
    )


def open_turn_has_parked_executions(
    nodes: Sequence[str],
    entries: Mapping[str, AnyEntry],
) -> bool:
    """Is any execution in the open turn parked at `AWAITING_RESULT` —
    dispatched, deferred, and only the application can move it? The gate's
    twin: a durable resting point, not a runtime in flight."""
    return any(
        execution.status == ExecutionStatus.AWAITING_RESULT for execution in open_turn_executions(nodes, entries)
    )


def open_turn_is_runnable(
    nodes: Sequence[str],
    entries: Mapping[str, AnyEntry],
) -> bool:
    """Can this open turn be advanced by a drive, LOOKING ONLY AT THIS PATH?

    Either an execution is still the drive's to move
    (`open_turn_has_advanceable_executions`), or every execution is terminal
    and the model can be called. Anything still `PENDING` past the first check
    is a gate the application must answer out of band, and with only those
    left nothing can advance.

    A PARKED execution (`AWAITING_RESULT`) is listed EXPLICITLY below rather
    than left to fall through, and that is the trap this predicate hides: the
    fallback is written as "nothing is left except gates", and a parked call is
    not a gate. Omitting it derives `BUSY` for a drive that can only park and
    return, which spins a `while not idle(): run()` driver.

    UNRESOLVED SUBAGENTS ARE NOT CONSIDERED HERE — a path cannot see another
    conversation's entries. `AgentSession.get_conversation_status` handles that
    term, because it is the thing that can reach the child."""
    if open_turn_has_advanceable_executions(nodes, entries):
        return True
    # Anything still PENDING here is gated, and an AWAITING_RESULT one is
    # parked: with only those left, nothing in this conversation can advance
    # until the application answers or resolves.
    return not any(
        execution.status in (ExecutionStatus.PENDING, ExecutionStatus.AWAITING_RESULT)
        for execution in open_turn_executions(nodes, entries)
    )


def _last_assistant_index(
    nodes: Sequence[str],
    entries: Mapping[str, AnyEntry],
    start: int,
) -> int | None:
    """Index of the last `AssistantMessage` at or after `start`, or None."""
    for i in range(len(nodes) - 1, start - 1, -1):
        if isinstance(entries.get(nodes[i]), AssistantMessage):
            return i
    return None


def open_turn_unseen_material(
    nodes: Sequence[str],
    entries: Mapping[str, AnyEntry],
    *,
    include_child_results: bool = True,
) -> bool:
    """Does the open turn hold model-facing content recorded AFTER its last
    `AssistantMessage` — the durable signal that a parent parked on its
    subagents must call the model again rather than keep waiting?

    Material is: a `UserMessage` (a mid-turn post), or a TERMINAL
    `ToolExecution` of a tool that does not declare spawning — an ordinary
    tool result, a runtime-minted subagent-result execution (the durable
    record that a child resolved), a stop execution, a REJECTED or failed
    call. A spawn-DECLARING call's outcome — spawned, declined, refused or
    failed — is deliberately NOT material, whatever it settled to: the spawn
    round's own answers must not wake the model by themselves (they travel
    with the first real update), and keying on the declaration rather than
    the settled payload keeps a mixed all-spawn round from waking on its
    refusals while its committed siblings are still starting. A nonterminal
    execution is the ordinary loop's business, never a reason to wake.

    `include_child_results=False` (the projection of
    `RuntimeConfig.wake_parent_on_subagent_completion` onto this pure path
    predicate) additionally excludes the subagent-result executions — the
    ones an open-turn `ChildConversation.result_execution_id` names — so a
    resolution alone is not material and the parent batches until every
    child has resolved. Posts and ordinary tool results stay material.

    One documented blind spot: a message posted while an LLM call is in
    flight lands BEFORE the recorded assistant entry, so this predicate cannot
    see it and the durable state is indistinguishable from an answered post.
    The live drive covers that window with its `seen` fingerprint; a session
    reloaded from exactly that window answers the message on the next
    material instead of immediately."""
    index = open_turn_index(nodes, entries)
    if index is None:
        return False
    last_assistant = _last_assistant_index(nodes, entries, index)
    if last_assistant is None:
        return False
    excluded: set[str] = set()
    if not include_child_results:
        excluded = {
            entry.result_execution_id
            for node_id in nodes[index:]
            if isinstance(entry := entries.get(node_id), ChildConversation) and entry.result_execution_id is not None
        }
    for node_id in nodes[last_assistant + 1 :]:
        entry = entries.get(node_id)
        if isinstance(entry, UserMessage):
            return True
        if (
            isinstance(entry, ToolExecution)
            and entry.status not in NONTERMINAL_STATUSES
            and entry.id not in excluded
            and (entry.tool_spec is None or not declares_spawn(entry.tool_spec))
        ):
            return True
    return False


def open_turn_unseen_post(
    nodes: Sequence[str],
    entries: Mapping[str, AnyEntry],
) -> bool:
    """Does the open turn hold a `UserMessage` recorded AFTER its last
    `AssistantMessage` — a post the model has not been shown?

    The GATE's wake term (0008), and deliberately narrower than
    `open_turn_unseen_material`: a terminal tool execution does not count. A
    gated round's ALLOWED siblings complete in the same round the gate was
    raised, so counting them would fire a model call — with the gate's
    placeholder on the wire — on every approval in the system, before anyone
    posted anything. Only a human's message opens this hatch.

    The same blind spot as its sibling: a message posted while an LLM call is
    in flight lands BEFORE the recorded assistant entry, where this cannot see
    it. The live drive covers that window with its `seen` fingerprint."""
    index = open_turn_index(nodes, entries)
    if index is None:
        return False
    last_assistant = _last_assistant_index(nodes, entries, index)
    if last_assistant is None:
        return False
    return any(isinstance(entries.get(node_id), UserMessage) for node_id in nodes[last_assistant + 1 :])


# ── the subagent handshakes (pure reads over the data model) ──────────────────
#
# The names the spawn and stop conventions settle. The core matches on the
# FIELD NAME, never on a tool name and never on a core type contrib has to
# import — that is what keeps "what a spawn looks like" owned by the plugin.
# `luca.agent` core never imports from contrib, so contrib's `SubagentSpawn` /
# `SubagentStop` are the plugin's guarantees about the payloads it emits, not
# something the core can or should validate against.

SPAWN_MARKER = "is_subagent_spawn"
SPAWN_REQUIRED_KEYS = ("task_id", "prompt", "description", "process_subagent_result_tool_name")
STOP_MARKER = "is_subagent_stop"


def declares_spawn(spec: ToolSpec) -> bool:
    """Does this SPEC declare that it can spawn? A shape check on the published
    output schema, read BEFORE the model call — there is no value to check yet.

    The asymmetry with the handshake is deliberate and is what the flag buys:
    `is_subagent_spawn` is a `bool`, not a constant, so a spawn tool that
    decides at runtime NOT to spawn (a rejected task, a failed validation, a
    request it answered inline) returns the payload with `False` and no child
    is created. It is still GATED, because the gate reads the declaration."""
    schema = spec.output_schema
    return isinstance(schema, dict) and SPAWN_MARKER in (schema.get("properties") or {})


def spawn_payload(execution: ToolExecution) -> dict | None:
    """The handshake payload of a COMPLETED execution that claims it spawned,
    or None. Value check, after the fact — the other half of `declares_spawn`."""
    if execution.status != ExecutionStatus.COMPLETED or execution.result is None:
        return None
    payload = execution.result.structured_content
    if not isinstance(payload, dict) or payload.get(SPAWN_MARKER) is not True:
        return None
    return payload


def stop_payload(execution: ToolExecution) -> dict | None:
    """The payload of a COMPLETED execution that asks for a subagent to be
    stopped, or None. Value-side only — unlike the spawn contract there is no
    declaration half, because stopping bypasses no cap: the runner acts on the
    payload wherever it appears."""
    if execution.status != ExecutionStatus.COMPLETED or execution.result is None:
        return None
    payload = execution.result.structured_content
    if not isinstance(payload, dict) or payload.get(STOP_MARKER) is not True:
        return None
    return payload


def spawns_committed(
    session: AgentSession,
    conversation_id: str,
    *,
    exclude: str | None = None,
) -> int:
    """Subagents this conversation's OPEN TURN has already committed to.

    The number both halves of the per-turn budget count — the gate that
    withholds the spawn tool and the runner's born-REFUSED fallback — so it is
    a pure read over durable entries and rebuilds correctly after a reload.

    Counting rules, and why:
    - only spawn-DECLARING executions count; the budget is about subagents,
      not tool calls
    - a settled call counts only if it actually spawned — a spawn tool may
      return `is_subagent_spawn=False`, and a refusal is not a subagent
    - a call still in flight (PENDING / RUNNING) counts as ONE, reserved: it
      is about to become a child, and the budget must hold before the fact.
      If it ends up not spawning, the reservation is released simply by it
      becoming settled-and-not-spawned.

    `exclude` drops one execution id from the count. `_spawn_one` needs it:
    converting a settled spawn re-checks the gate at a point where that
    execution is already counted, and the question there is "was there room
    for this one?", not "is there room for another?"."""
    conversation = session.conversations[conversation_id]
    count = 0
    for execution in open_turn_executions(conversation.nodes, session.entries):
        if execution.id == exclude:
            continue
        if execution.tool_spec is None or not declares_spawn(execution.tool_spec):
            continue
        if execution.status in (ExecutionStatus.PENDING, ExecutionStatus.RUNNING):
            count += 1  # in flight — reserved
        elif spawn_payload(execution) is not None:
            count += 1  # settled, and it spawned
    return count


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

    # ── subagents ──────────────────────────────────────────────────────────
    # Off by default, so existing sessions and tests are unaffected: with this
    # False the registry withholds the spawn tool and the runner raises if one
    # comes back anyway.
    subagents_enabled: bool = False
    # How deep the tree may go: N allows spawning from depths 0..N-1, so the
    # deepest subagent sits at depth N — the main conversation plus N levels
    # below it. Default 1: the main conversation spawns, a subagent does not.
    # Nesting multiplies cost (step limits are per conversation, not per tree),
    # so raising this is deliberate, not free.
    subagents_max_depth: int = 1
    # How many subagents one conversation may spawn in one OPEN TURN. Spent
    # budget withholds the spawn tool and its prompt part; a spawn call that
    # arrives after the tool list was fixed is born REFUSED. Every subagent has
    # its own turn and therefore its own budget; the count resets when the
    # turn closes. Inf (-1) = no limit.
    subagents_max_per_turn: int = Inf
    # How many subagent conversations may be DOING WORK at the same time,
    # across the whole session. Inf (-1, the default) = no limit. The main
    # conversation never counts against it: it is not a subagent, and it must
    # always be able to advance. A slot is held only while a subagent works —
    # a conversation waiting on its own children, on a human, or winding a
    # cancelled turn down holds none — so size this by FAN-OUT (how many
    # siblings should work at once), never by `subagents_max_depth`: depth
    # costs sequence, not concurrency.
    subagents_max_workers: int = Inf
    # Step limits for subagent conversations. `None` falls back to the main
    # `soft_max_steps` / `hard_max_steps` above. These are what make an
    # uncompacted subagent safe: a child is never compaction-checked in V0, so
    # a bound on its steps is what stops it growing without limit.
    subagent_soft_max_steps: int | None = None
    subagent_hard_max_steps: int | None = None
    # Whether a resolved subagent's result, BY ITSELF, re-engages the parent's
    # model mid-orchestration (a wake round per resolution). False batches:
    # the parent still resolves each child the moment it finishes — the result
    # execution lands durably either way — but the next model call waits until
    # every child has resolved, or until OTHER material arrives (a mid-turn
    # post and an ordinary tool result wake regardless of this flag).
    wake_parent_on_subagent_completion: bool = True

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
        "subagents_max_depth",
    )
    @classmethod
    def _inf_or_natural(cls, value: int) -> int:
        if value < Inf:
            raise ValueError(f"must be >= {Inf} ({Inf} = infinite / disabled)")
        return value

    @field_validator("subagents_max_per_turn", "subagents_max_workers")
    @classmethod
    def _inf_or_positive(cls, value: int) -> int:
        """`Inf` (-1) or at least 1. `0` is not "disabled" — it would mean no
        subagent may ever exist, which is `subagents_enabled=False` spelled
        incorrectly."""
        if value != Inf and value < 1:
            raise ValueError(f"must be >= 1 or {Inf} ({Inf} = no limit)")
        return value

    @field_validator("subagent_soft_max_steps", "subagent_hard_max_steps")
    @classmethod
    def _optional_inf_or_natural(cls, value: int | None) -> int | None:
        """Same rule as the fields above, but `None` is meaningful here: it
        means "fall back to the main conversation's limit", which is a
        different fact from `Inf` ("no limit")."""
        if value is not None and value < Inf:
            raise ValueError(f"must be >= {Inf} ({Inf} = infinite / disabled) or None")
        return value


class SessionConfig(BaseConfigModel):
    """Global session configuration, persisted with the session. `llm_config`
    is the default for the NEXT turn. Tool approval lives entirely outside
    the session: the runner's `ToolRegistry` is runtime collaboration, not
    state."""

    llm_config: LLMConfig
    runtime_config: RuntimeConfig = Field(default_factory=RuntimeConfig)
    # CONFIGURED native-tools switch: does the application want provider-native
    # tool declarations where the provider has them? Purely an adaptation input
    # read by application/plugin middleware — the core itself has no concept of
    # a native tool. The runner re-stamps the session's ACTIVE
    # `use_native_tools` from this at the top of every drive iteration, which
    # is what makes a mid-session flip take effect on the next LLM call.
    use_native_tools: bool = False


# ── traversal + container ──────────────────────────────────────────────────────


class ConversationStatus(str, Enum):
    """What the next `run()` on a conversation will do. DERIVED from the
    entries on every read and never stored — see
    `AgentSession.get_conversation_status`.

    | | the next `run()` | post a message? |
    |---|---|---|
    | `IDLE` | nothing — there is no work | yes |
    | `BUSY` | work — the run can still be exhausted | yes — into the open turn (or queued behind a trailing message) |
    | `BLOCKED` | stop again immediately; you must act first | yes — the conversation derives `BUSY` and the next drive answers it past the gate, then re-parks (0008) |
    | `CANCELLING` | flush the turn, not answer it | no |

    The post column is the COMMON case, not the whole rule:
    `AgentSessionRunner.post_message` owns the full acceptance matrix — an
    open compaction bracket rejects, and archived or finished conversations
    reject whatever they derive. An open turn with unresolved subagents
    ACCEPTS: the message is material, and a parent parked on its children is
    woken to show it to the model.

    **It says nothing about approvals**, deliberately. A gate can belong to a
    subagent, and its siblings may still be working — so the conversation is
    `BUSY` (the run can be exhausted) while `pending_approvals()` already
    returns that gate. Only when nothing else can advance does it become
    `BLOCKED`. An `AWAITING_APPROVAL` status could not express that.

    `BUSY` is still true after a crash: "can be advanced" survives the process
    dying, so unlike the old `RUNNING` it needs no self-healing rule."""

    IDLE = "idle"  # nothing to do
    BUSY = "busy"  # the run can still be exhausted
    BLOCKED = "blocked"  # nothing can advance; you must act first
    CANCELLING = "cancelling"  # unconsumed CancelRequested — the next drive flushes


class Conversation(BaseModel):
    """One ordered path over the shared entry store.

    A conversation owns its own history, BY REFERENCE: compaction archives one
    conversation and installs another over a rewritten path, so the predecessor
    belongs to its successor — `previous_conversation_id`. An archived
    conversation stays a first-class row in `AgentSession.conversations`; it has
    to, because usage records key on conversation id. One home per
    conversation, pointers everywhere else.

    There is deliberately NO `parent_conversation_id`. A parent mints its
    child's handle and re-mints it after a reload from its own unresolved
    entries, so parent → child is the only direction anything traverses — which
    is exactly why `depth` has to be a stored field: with no link upwards it is
    not derivable from the conversation alone. It is stamped at creation and
    copied when a compaction installs a successor.

    There is also NO `status`. Status is a pure function of the entries, so a
    stored copy is a rendering rather than state, and a persisted one lets a
    stale file disagree with the log — read
    `AgentSession.get_conversation_status(id)`, which is the only door."""

    id: str  # stable identity of this path/branch — usage records key on it
    nodes: list[str] = Field(default_factory=list)  # ordered entry ids = THE path
    created_at: int
    updated_at: int
    # What this conversation replaced: the conversation a compaction archived
    # when it installed this one. None for a conversation nothing preceded.
    previous_conversation_id: str | None = None
    # 0 = the main conversation; a subagent's is its parent's + 1. Stored
    # because nothing links upwards, and read on every tool-list decision.
    depth: int = 0

    model_config = ConfigDict(extra="forbid")


class ConversationRuntimeStatus(BaseModel):
    """Derived view of ONE conversation's runtime state — always recomputed
    from the entry log, never stored and never trusted from serialized data.
    Obtained through `AgentSession.get_conversation_status(conversation_id)`,
    which is the only door."""

    status: ConversationStatus = ConversationStatus.IDLE
    # Conversational turns on this path, including the open one. Scoped to the
    # path because entries outlive their conversation (a compaction archives
    # one and opens another over the same store), and excluding compaction
    # brackets because a compaction is not a turn.
    turn_count: int = 0
    step_count: int = 0  # AssistantMessages in the currently open turn

    model_config = ConfigDict(extra="forbid")


# The version of the SERIALIZED session shape. Stamped on every session as
# `AgentSession.spec_version` and bumped only when a change to the data model
# breaks reading an older file. A loader reads the field to decide whether the
# session needs migrating before it can be constructed.
SPEC_VERSION = "0.0.1"


class AgentSession(BaseModel):
    """The whole durable session: one flat entry store, the CATALOG of
    conversations over it, and the normalized tool-spec store.

    `conversations` is a store keyed by id, exactly like `entries`,
    `tool_specs` and `usages`, and `main_conversation_id` is a pointer into it.
    That mirrors the entries/nodes split one level up: conversations are stored
    in one place and referenced from several. There is no "active"
    conversation — with several advancing at once there is no single one.

    ARCHIVED conversations live in the same dict as live ones. They have to:
    usage records key on conversation id, and a session total sums across the
    catalog. Nesting them under their successor would either put the same
    conversation in two places or leave `usages[cid]` with nothing to resolve
    against. Lineage is a pointer — `Conversation.previous_conversation_id`.

    Constructing one RESTORES `ToolExecution.tool_spec` from `tool_spec_id`
    and refuses a session it cannot restore — see `_restore_tool_specs`.
    Serializing one STRIPS the per-execution spec copies and writes the shared
    `tool_specs` dict once; serializing a `ToolExecution` on its own still
    includes its spec inline, so an event forwarded to a web UI or a log sink
    stays self-describing."""

    id: str
    # Which revision of the session shape this file was written against. A
    # session built in-process gets the current `SPEC_VERSION`; one loaded from
    # disk keeps whatever it was written with, so a loader can tell an old file
    # from a current one and migrate it.
    spec_version: str = SPEC_VERSION
    entries: dict[str, AnyEntry] = Field(default_factory=dict)  # append-only store
    # spec_id → ToolSpec. Append-only (never garbage-collected), written only
    # through `SessionLedger`'s write doors. Holds the specs referenced by an
    # execution — tools that were actually CALLED, not the set advertised on
    # any given turn.
    tool_specs: dict[str, ToolSpec] = Field(default_factory=dict)
    # Accessory provider-usage records, conversation-first so per-conversation
    # reads and cleanup are direct: usages[conversation_id][entry_id] → Usage.
    # Written only through `SessionLedger.record_usage()`.
    usages: dict[str, dict[str, Usage]] = Field(default_factory=dict)
    # conversation_id → Conversation. Every conversation this session has ever
    # held: the main one, its archived predecessors, and (later) one per
    # subagent. Written only through `SessionLedger`.
    conversations: dict[str, Conversation] = Field(default_factory=dict)
    # Which row in `conversations` is the one a user talks to. A compaction
    # re-points it at the successor it installed.
    main_conversation_id: str
    session_config: SessionConfig
    # ── the ACTIVE llm config — what the NEXT call will use ─────────────────
    # `session_config.llm_config` is the CONFIGURED value (durable, what the
    # user set); this pair is the ACTIVE one, stamped by the runner via
    # `update_llm_config()` at the top of every drive iteration — after any
    # `build_model_string` middleware routed the turn. Everything handed the
    # session (`get_tools`, every middleware, provenance recording) reads what
    # the next call will use from here, so no `llm_cfg` parameter threading
    # exists anywhere. Derived state: recomputed every iteration from the
    # configured values, so persistence carries no authority — construction
    # defaults both from `session_config` when absent.
    llm_config: LLMConfig | None = None
    use_native_tools: bool | None = None
    # Free-form application state, stored verbatim and never interpreted by
    # the core — the session-level twin of `ToolExecution.extras`. It exists so
    # a tool, a registry or a plugin can keep state that outlives the process
    # without the application inventing a second file to put it in: whoever
    # composes the runner hands the state in and it rides along on every save.
    # Namespace your key (the tool's `namespace` is the obvious choice) and
    # keep the value JSON-serializable — this is dumped with the session.
    extras: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _default_active_llm_config(self) -> AgentSession:
        """Default the ACTIVE pair from the CONFIGURED values. Before any
        drive stamps them, "what the next call will use" IS the configured
        config — so a freshly built or freshly loaded session always answers
        `session.llm_config` / `session.use_native_tools` honestly."""
        if self.llm_config is None:
            self.llm_config = self.session_config.llm_config.model_copy(deep=True)
        if self.use_native_tools is None:
            self.use_native_tools = self.session_config.use_native_tools
        return self

    @model_validator(mode="after")
    def _check_main_conversation(self) -> AgentSession:
        """`main_conversation_id` must name a row in `conversations`.

        The same class of load guard as `_restore_tool_specs`: a session whose
        pointer dangles loads fine and then fails on the first drive, deep
        inside the ledger, with nothing saying the file was wrong."""
        if self.main_conversation_id not in self.conversations:
            raise ValueError(
                f"main_conversation_id {self.main_conversation_id!r} is absent from session.conversations."
            )
        return self

    @model_validator(mode="after")
    def _restore_tool_specs(self) -> AgentSession:
        """Point every execution's `tool_spec` at the shared spec its
        `tool_spec_id` names, and refuse a session that cannot be fully
        restored. A session never exists half-restored.

        Every execution referencing the same id ends up holding the SAME
        `ToolSpec` instance — the one in `tool_specs` — a value object held by
        reference, not a per-execution copy.

        Two shapes raise:

        - a `tool_spec_id` absent from `tool_specs`. In normal operation this
          cannot happen (the ledger writes both sides together); it happens
          when a session is assembled by something else — entries copied
          between sessions without their specs, a hand-edited or truncated
          file, a literal that sets the id and forgets the store. Tolerating
          it is silent: the dispatch deadline would fall back to
          `RuntimeConfig.tool_execution_timeout_in_ms` (default: unbounded)
          and `tool_kind` would vanish, changing the approval path.
        - a `tool_spec` with NO `tool_spec_id`. That can only come from a
          pre-normalization file, which would otherwise load and run fine and
          then lose every spec on the first save — the serializer strips
          inline specs and would have no id to write in their place.

        It fires on CONSTRUCTION only. A registry's in-memory birth draft
        legitimately carries `tool_spec` with no id yet; the ledger's write
        door assigns it. A session literal needs `tool_spec_id` plus its
        `tool_specs` row — `tool_spec` itself is optional there, since this
        validator restores it."""
        for entry in self.entries.values():
            if not isinstance(entry, ToolExecution):
                continue
            if entry.tool_spec_id is None:
                if entry.tool_spec is not None:
                    raise ValueError(
                        f"ToolExecution {entry.id!r} carries a tool_spec with no "
                        f"tool_spec_id. Sessions written before tool-spec "
                        f"normalization do not load; regenerate them."
                    )
                continue
            spec = self.tool_specs.get(entry.tool_spec_id)
            if spec is None:
                raise ValueError(
                    f"ToolExecution {entry.id!r} references tool_spec_id "
                    f"{entry.tool_spec_id!r}, which is absent from "
                    f"session.tool_specs."
                )
            entry.tool_spec = spec
        return self

    @model_serializer(mode="wrap")
    def _strip_inline_tool_specs(self, handler):
        """Drop the per-execution `tool_spec` copies from a serialized
        SESSION: the shared `tool_specs` dict already holds each distinct spec
        once, and `_restore_tool_specs` puts them back on load.

        Deliberately here and not `Field(exclude=True)` on
        `ToolExecution.tool_spec`: that one line would strip the field from
        EVERY serialization, including a standalone `ToolExecution`, which is
        exactly what the tool lifecycle events forward to consumers holding no
        session. Normalization is a session-storage concern."""
        data = handler(self)
        entries = data.get("entries")
        if isinstance(entries, dict):
            for entry in entries.values():
                if isinstance(entry, dict) and entry.get("type") == "tool_execution":
                    entry.pop("tool_spec", None)
        return data

    def get_conversation_status(
        self,
        conversation_id: str,
    ) -> ConversationRuntimeStatus:
        """One conversation's runtime state, RECOMPUTED FROM THE NODES on every
        call. The only door — nothing caches this and nothing stores it.

        Derivation, in precedence order:

        - open turn with an unconsumed cancel  → `CANCELLING`
        - open turn with something runnable    → `BUSY`
        - open turn with nothing runnable (only gated or PARKED executions
          remain) but an unseen user post → `BUSY` — both project a
          placeholder, so the next drive answers the post and re-parks (0008)
        - open turn with nothing runnable      → `BLOCKED`
        - trailing `UserMessage` (queued work) → `BUSY`
        - anything else, INCLUDING a closed `TurnFinish` whatever its outcome
          → `IDLE`

        Two consequences, both deliberate. A failed turn derives `IDLE`, so
        recovering from one means posting a new message rather than re-driving
        the identical request. And a trailing `UserMessage` derives `BUSY`
        (queued work) — further messages may still be posted behind it, and
        the next turn answers them all; the status only says the next `run()`
        has work to do, never whether input is accepted (that is
        `AgentSessionRunner.post_message`'s matrix).

        Trailing CLOSED compaction brackets are transparent and skipped
        (repeatedly, if several stack), so a compaction that closed behind a
        queued question does not bury it."""
        conversation = self.conversations.get(conversation_id)
        if conversation is None:
            raise KeyError(f"no conversation {conversation_id!r} in session {self.id!r}")
        nodes = conversation.nodes
        entries = self.entries
        return ConversationRuntimeStatus(
            status=self._derive_status(nodes, entries),
            turn_count=conversational_turn_count(nodes, entries),
            step_count=open_turn_step_count(nodes, entries),
        )

    def update_llm_config(self, model_string: str, use_native_tools: bool) -> None:
        """Stamp the ACTIVE llm config + native-tools flag for the next call.

        The runner calls this at the top of every drive iteration, with the
        model string any `build_model_string` middleware produced. The
        derivation always starts from the CONFIGURED config
        (`session_config.llm_config`) — a middleware routing one turn to a
        cheaper model must not drift the session — splitting at the FIRST
        colon, the rule the client uses, so a model id containing one
        survives; `reasoning` and `extras` are preserved. A string with no
        prefix is a bare model id: the configured provider stands.

        Recording the active config here rather than in a drive-local is what
        makes it visible to everything handed the session; the transports
        compare an assistant message's recorded provenance against the model
        being called to decide whether a thinking signature may be replayed,
        so a routed turn recording the configured config would make
        provenance lie."""
        configured = self.session_config.llm_config
        provider, _, model = model_string.partition(":")
        if not model:  # no prefix — middleware returned a bare model id
            self.llm_config = configured.model_copy(update={"model": model_string})
        else:
            self.llm_config = configured.model_copy(update={"provider": provider, "model": model})
        self.use_native_tools = use_native_tools

    def get_tool_execution(self, tool_call_id: str) -> ToolExecution | None:
        """The `ToolExecution` correlated to one tool call id (its
        `tool_call_id` field), or None when no execution answers it.

        The read an adaptation middleware uses to reach the stored native
        payload back from a projected message: `raw_tool_call.extras` for a
        call, `result.extras` for a result (`result is None` is the
        derived-outcome signal — REJECTED / FAILED / awaiting approval). A
        linear scan over `entries`; index later if projection-time scans ever
        show up in profiles."""
        for entry in self.entries.values():
            if isinstance(entry, ToolExecution) and entry.tool_call_id == tool_call_id:
                return entry
        return None

    def _derive_status(
        self,
        nodes: Sequence[str],
        entries: Mapping[str, AnyEntry],
    ) -> ConversationStatus:
        if open_turn_index(nodes, entries) is not None:
            if open_turn_cancel_requested(nodes, entries) is not None:
                return ConversationStatus.CANCELLING
            children = open_turn_unresolved_children(nodes, entries)
            if children:
                # SUBTREE-AWARE, and it has to be: a parent is BUSY while its
                # children work, and a `Conversation` on its own cannot see
                # another conversation's entries. In precedence order:
                #
                # 1. An execution the drive itself can still move → BUSY. A
                #    wake round's calls advance regardless of what the
                #    children are doing.
                # 2. A BUSY child is working; an IDLE one has finished and
                #    this conversation can resolve it; a CANCELLING one is
                #    winding down → BUSY.
                # 3. A gated OR PARKED execution with nothing above it →
                #    BLOCKED, UNLESS an unseen post can reach the model past it
                #    (0008): with the gate (or the parked call) projecting a
                #    placeholder the next drive CAN do something, so the
                #    material term below is allowed to speak. Without a post the
                #    old reasoning stands — the drive would only re-park, and
                #    BUSY would hot-loop every poll-for-BLOCKED consumer.
                #    A parked call has to be named HERE and not left to the
                #    material term: any completed sibling satisfies that term,
                #    so a drive that only parks would derive BUSY.
                # 4. Unseen material (a resolved child's result, a mid-turn
                #    post, a non-spawn tool result) → BUSY: the next run()
                #    calls the model with it. With
                #    `wake_parent_on_subagent_completion=False` a resolved
                #    child's result is NOT material — the next run() parks on
                #    it — so the same flag gates the same predicate here,
                #    keeping the status and the drive in agreement.
                # 5. Otherwise every child is BLOCKED and there is nothing new
                #    to show the model → BLOCKED.
                if open_turn_has_advanceable_executions(nodes, entries):
                    return ConversationStatus.BUSY
                if any(
                    self.get_conversation_status(child.conversation_id).status
                    in (
                        ConversationStatus.BUSY,
                        ConversationStatus.IDLE,
                        ConversationStatus.CANCELLING,
                    )
                    for child in children
                    if child.conversation_id in self.conversations
                ):
                    return ConversationStatus.BUSY
                if (
                    open_turn_has_awaiting_executions(nodes, entries) or open_turn_has_parked_executions(nodes, entries)
                ) and not open_turn_unseen_post(nodes, entries):
                    return ConversationStatus.BLOCKED
                if open_turn_unseen_material(
                    nodes,
                    entries,
                    include_child_results=self.session_config.runtime_config.wake_parent_on_subagent_completion,
                ):
                    return ConversationStatus.BUSY
                return ConversationStatus.BLOCKED
            if open_turn_is_runnable(nodes, entries):
                return ConversationStatus.BUSY
            # Not runnable means only gated or PARKED executions remain. An
            # unseen post still reaches the model past them (0008): both
            # project a placeholder, so the next drive answers the post and
            # re-parks.
            if open_turn_unseen_post(nodes, entries):
                return ConversationStatus.BUSY
            return ConversationStatus.BLOCKED
        end = end_before_closed_compaction_brackets(nodes, entries)
        if end == 0:
            return ConversationStatus.IDLE
        if isinstance(entries.get(nodes[end - 1]), UserMessage):
            return ConversationStatus.BUSY
        return ConversationStatus.IDLE
