"""Shared declarative preconditions for the runner tests.

Six kinds of building blocks:

- **`DeterministicRunner`**: the test-side extension of `AgentSessionRunner`.
  Determinism is the TEST'S concern, layered on by overriding the production
  hooks — `generate_id()` is scripted by `ids` (consumed in entry-creation
  order) and `now_ms()` is frozen to `now`. The production class carries no
  test parameters.
- **`FakeToolRegistry`**: the core-only deterministic `ToolRegistry` double
  (core tests must NOT import contrib). `get_tools` returns `ToolSpec`s, a
  preflight-faithful `create_execution` (NOT_FOUND / INVALID /
  FAILED-from-approval-context / PENDING births with the classic error
  payloads), a scripted `decide` that records every execution it was asked
  about in `seen` (no script → deterministic allow-all with `created_at`
  frozen to `now`), and a resolve-validate-then-close-over-the-body `prepare`
  that records the names it resolved in `prepared`.
- **`FakeContextManager`**: the core-only scripted compaction half of
  `ContextManager` — a scripted `should_compact`, a scripted plan (or `None`, a
  raise, or a cooperative hang), and `seen` / `should_calls` records. Core's
  per-entry accounting is inherited untouched.
- **Tool doubles**: minimal `FakeTool` subclasses covering each behavior the
  registry/runner must handle (plain success, a duck-typed approval context,
  a captured session, a raised exception, a rich is_error result, a DEFERRING
  tool). `FakeTool` is deliberately NOT contrib's `Tool`: the core knows only
  `ToolSpec`, and building the doubles out of core types is what proves it.
- **`spec()` / `make_session()`**: the two literal factories. `ToolSpec` now
  requires `description` and `input_schema`, and a session literal has to
  carry `tool_specs` plus a `tool_spec_id` on every execution — neither is
  what any test is about, so the factories fill both and the precondition
  stays declarative.
- **Mid-state session literals**: known `AgentSession`s exactly as they would
  sit on disk (a turn paused at the approval gate, an approved-but-unrun call,
  a crash mid-decide, a stale RUNNING status, an orphaned RUNNING execution),
  plus the compaction set built on `RICH_SESSION` (scheduled, interrupted,
  cancel-parked, failed, buried, and already committed).
  Tests load these *cold* — into a fresh runner — so every scenario reads as:
  GIVEN this persisted state, WHEN one action, THEN this outcome. Always take
  a `model_copy(deep=True)` before mutating.

All literals use `created_at=500` so entries written by the test's frozen
clock (`now=1000`) are visually distinct from the precondition.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError

from luca.agent.core.compaction import CompactionPlan
from luca.agent.core.context import CancellationToken
from luca.agent.core.context_manager import ContextManager
from luca.agent.core.exceptions import InvalidToolArguments, ToolNotFound
from luca.agent.core.models import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    CancelRequested,
    CompactionEntry,
    CompactionSource,
    Conversation,
    ExecutionAttempt,
    ExecutionAttemptOutcome,
    ExecutionDeferred,
    ExecutionResult,
    ExecutionStatus,
    ImageContent,
    LLMConfig,
    MediaBase64,
    PrunedEntry,
    SessionConfig,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    ToolKind,
    ToolSpec,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    Usage,
    UserMessage,
)
from luca.agent.core.runner import AgentSessionRunner
from luca.agent.core.tool_registry import PreparedTool, ToolRegistry

MODEL = LLMConfig(model="test-model", provider="faux")
# What a compacting manager summarizes with — deliberately NOT the session's
# model, so a stamped `llm_config` proves whose choice it was.
CHEAP = LLMConfig(model="cheap-model", provider="faux")

# The schema of a tool that takes no arguments — required and never None, so
# `spec()` needs a default that means "no arguments" rather than "unknown".
EMPTY_SCHEMA = {"type": "object", "properties": {}}


# ── literal factories ──────────────────────────────────────────────────────────


def spec(name: str, **over) -> ToolSpec:
    """A complete `ToolSpec` for a precondition literal.

    `description` and `input_schema` are required on `ToolSpec` and neither is
    ever what a test is about, so they get filled here and a test overrides
    only the field it actually cares about: `spec("bash", timeout_in_ms=10)`.
    """
    fields: dict = {
        "name": name,
        "description": f"{name} tool.",
        "input_schema": EMPTY_SCHEMA,
    }
    fields.update(over)
    return ToolSpec(**fields)


def conversation(
    conversation_id: str,
    nodes: list[str] | None = None,
    *,
    created_at: int = 500,
    updated_at: int = 500,
    **over,
) -> Conversation:
    """A `Conversation` for a precondition literal.

    `created_at` / `updated_at` default to the 500 every literal here uses, and
    `previous_conversation_id` / `depth` stay at their defaults unless a test
    is about them — so an ordinary literal reads
    `conversation("c1", ["u1", "ts"])` and says only what it means."""
    return Conversation(
        id=conversation_id,
        nodes=list(nodes or []),
        created_at=created_at,
        updated_at=updated_at,
        **over,
    )


def main_conversation(session: AgentSession) -> Conversation:
    """A session's main conversation — `session.conversations[
    session.main_conversation_id]`, which is what an assertion about "the
    conversation" means now that a session holds several."""
    return session.conversations[session.main_conversation_id]


def make_session(**fields) -> AgentSession:
    """Build an `AgentSession` literal with its tool-spec store filled in.

    A session refuses to construct with an execution carrying a `tool_spec`
    and no `tool_spec_id`, and refuses an id absent from `tool_specs` — so a
    hand-written literal would otherwise have to file every spec by hand.
    This does exactly what the ledger's write doors do: hash each execution's
    spec, stamp the id, store it once. Explicitly-passed `tool_specs` rows and
    already-stamped ids are left alone, so a test can still author a dangling
    reference on purpose."""
    tool_specs: dict[str, ToolSpec] = dict(fields.pop("tool_specs", None) or {})
    for entry in (fields.get("entries") or {}).values():
        if not isinstance(entry, ToolExecution):
            continue
        if entry.tool_spec is None or entry.tool_spec_id is not None:
            continue
        entry.tool_spec_id = entry.tool_spec.spec_id()
        tool_specs.setdefault(entry.tool_spec_id, entry.tool_spec)
    return AgentSession(tool_specs=tool_specs, **fields)


# ── registry double ────────────────────────────────────────────────────────────


class FakeToolRegistry(ToolRegistry):
    """Core-only deterministic registry double.

    `get_tools` answers `ToolSpec`s — the core's only tool type. Nothing here
    imports contrib, so the doubles double as proof that the runner drives a
    full tool call without ever seeing a Python tool class.

    `create_execution` is preflight-faithful: an unknown name births a
    NOT_FOUND draft (`tool_spec=None`), invalid `Args` an INVALID draft, a
    raising duck-typed `get_approval_context` a FAILED draft (its result
    otherwise lands in `extras["approval_context"]`), and a healthy call a
    PENDING draft — all with placeholder identity for the runner to stamp.
    `decide` records the execution snapshot it was asked about in `seen`,
    then pops the next scripted decision (calls arrive in unresolved-path
    order; exhaustion raises IndexError — the script no longer matches what
    the runner asked); with no script it ALLOWs everything, `created_at`
    frozen to `now`. `prepare` resolves by the effective call's name,
    validates, records the name in `prepared`, and returns a closure over the
    tool body — it never invokes it."""

    def __init__(
        self,
        tools: Iterable[FakeTool] = (),
        decisions: Iterable[ApprovalDecision] | None = None,
        now: int = 1000,
    ) -> None:
        self.tools = list(tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.decisions = list(decisions) if decisions is not None else None
        self.now = now
        self.seen: list[ToolExecution] = []
        self.prepared: list[str] = []

    async def get_tools(
        self,
        session: AgentSession,
        conversation_id: str,
    ) -> list[ToolSpec]:
        return [tool.get_tool_spec() for tool in self.tools]

    async def create_execution(
        self,
        session: AgentSession,
        conversation_id: str,
        call: ToolCall,
    ) -> ToolExecution:
        def draft(status, tool_spec=None, error=None, extras=None):
            return ToolExecution(
                tool_call_id=call.id,
                raw_tool_call=call,
                tool_spec=tool_spec,
                status=status,
                error=error,
                extras=extras or {},
            )

        tool = self.tools_by_name.get(call.name)
        if tool is None:
            return draft(
                ExecutionStatus.NOT_FOUND,
                error=ToolExecutionError(
                    error_type="ToolNotFound",
                    error_message=f"Unknown tool: {call.name!r}.",
                ),
            )
        tool_spec = tool.get_tool_spec()
        try:
            args = tool.Args.model_validate(call.arguments)
        except ValidationError as exc:
            return draft(
                ExecutionStatus.INVALID,
                tool_spec=tool_spec,
                error=ToolExecutionError(
                    error_type="InvalidToolArguments",
                    error_message=(f"Arguments for tool {call.name!r} are invalid."),
                    details={
                        "errors": json.loads(exc.json(include_url=False)),
                    },
                ),
            )
        extras: dict = {}
        if hasattr(tool, "get_approval_context"):
            try:
                extras["approval_context"] = await tool.get_approval_context(
                    args.model_dump(),
                    session,
                    conversation_id,
                )
            except Exception as exc:
                return draft(
                    ExecutionStatus.FAILED,
                    tool_spec=tool_spec,
                    error=ToolExecutionError(
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        details={"phase": "approval_context"},
                    ),
                )
        return draft(
            ExecutionStatus.PENDING,
            tool_spec=tool_spec,
            extras=extras,
        )

    async def decide(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> ApprovalDecision:
        self.seen.append(tool_execution)
        if self.decisions is None:
            return ApprovalDecision(
                decision=ApprovalOption.ALLOW,
                created_at=self.now,
            )
        return self.decisions.pop(0)

    async def prepare(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> PreparedTool:
        name = tool_execution.raw_tool_call.name
        tool = self.tools_by_name.get(name)
        if tool is None:
            raise ToolNotFound(f"Unknown tool: {name!r}.")
        try:
            args = tool.Args.model_validate(
                tool_execution.raw_tool_call.arguments,
            )
        except ValidationError as exc:
            raise InvalidToolArguments(
                f"Arguments for tool {name!r} are invalid.",
                errors=json.loads(exc.json(include_url=False)),
            ) from exc
        payload = args.model_dump()
        self.prepared.append(name)

        tool_call_id = tool_execution.tool_call_id

        async def run(
            *,
            cancellation_token: CancellationToken,
        ) -> ExecutionResult | ExecutionDeferred:
            return await tool.execute(
                payload,
                session,
                conversation_id,
                tool_name=name,
                tool_call_id=tool_call_id,
                cancellation_token=cancellation_token,
            )

        return run


class FakeContextManager(ContextManager):
    """Core-only scripted `ContextManager` double for compaction (core tests
    must NOT import contrib). Per-entry accounting is core's, unchanged.

    `should` scripts `should_compact`: a bool, or a list popped per call whose
    last value repeats once exhausted. `plan` is what `compact()` returns — a
    `CompactionPlan`, `None` for "nothing to do", or a callable
    `(session, nodes, entry) -> plan | None` for the common case where the
    plan needs the live entry id. `raises` raises instead of returning.
    `hang` waits forever on an event nobody sets, so cancellation and deadline
    scenarios never race two timed things — one side is infinite, the other is
    the only clock. `mutate` writes `parts` onto the entry it was handed
    before doing anything else, which is how the deep copy is proven
    load-bearing.

    `seen` records `(session, nodes, entry)` per `compact()` call and
    `should_calls` counts the sync consultations."""

    def __init__(
        self,
        should: bool | list[bool] = False,
        plan=None,
        raises: Exception | None = None,
        hang: bool = False,
        mutate: bool = False,
    ) -> None:
        self.should = should
        self.plan = plan
        self.raises = raises
        self.hang = hang
        self.mutate = mutate
        self.seen: list[tuple] = []
        self.should_calls = 0

    def should_compact(self, session: AgentSession, conversation_id: str) -> bool:
        self.should_calls += 1
        if isinstance(self.should, list):
            if len(self.should) > 1:
                return self.should.pop(0)
            return self.should[0]
        return self.should

    async def compact(
        self,
        session: AgentSession,
        conversation_id: str,
        nodes: tuple[str, ...],
        entry: CompactionEntry,
    ) -> CompactionPlan | None:
        self.seen.append((session, nodes, entry))
        if self.mutate:
            entry.parts = [TextContent(text="injected without a compaction")]
        if self.hang:
            await asyncio.Event().wait()
        if self.raises is not None:
            raise self.raises
        if callable(self.plan):
            return self.plan(session, nodes, entry)
        return self.plan


class DeterministicRunner(AgentSessionRunner):
    """`AgentSessionRunner` with the id/clock hooks overridden for tests:
    `ids` scripts `generate_id()` (one per created entry, in creation order;
    exhaustion raises StopIteration — the test's id script no longer matches
    what the runner did), `now` freezes `now_ms()`."""

    def __init__(
        self,
        session: AgentSession,
        tool_registry: ToolRegistry | None = None,
        system_prompt_parts=None,
        system_prompt_assembler=None,
        *,
        provider=None,
        api_key: str | None = None,
        model_options: dict | None = None,
        provider_options: dict | None = None,
        conversation_projector=None,
        context_manager=None,
        ids: Iterable[str] = (),
        now: int = 1000,
        middleware: list | None = None,
    ) -> None:
        self._ids = iter(ids)
        self._now = now
        super().__init__(
            session,
            tool_registry,
            system_prompt_parts,
            system_prompt_assembler,
            provider=provider,
            api_key=api_key,
            model_options=model_options,
            provider_options=provider_options,
            conversation_projector=conversation_projector,
            context_manager=context_manager,
            middleware=middleware,
        )

    def generate_id(self) -> str:
        return next(self._ids)

    def now_ms(self) -> int:
        return self._now


# ── tool doubles ───────────────────────────────────────────────────────────────


class BinaryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: int
    b: int


class PathArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str


class FakeTool:
    """Core-only tool double — deliberately NOT `luca.agent.contrib.tools.Tool`.

    Core tests must not import contrib, and the core's only tool type is
    `ToolSpec`, so the doubles carry their own `get_tool_spec()` (schema from
    `Args`, exactly as the contrib base does) and their own body. Nothing the
    runner touches knows this class exists: `FakeToolRegistry` is what turns
    one into specs and prepared closures."""

    name: ClassVar[str]
    description: ClassVar[str]
    Args: ClassVar[type[BaseModel]]
    tool_kind: ClassVar[ToolKind] = ToolKind.OTHER
    timeout_in_ms: ClassVar[int | None] = None
    is_private: ClassVar[bool] = False
    output_schema: ClassVar[dict | None] = None

    def get_tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.Args.model_json_schema(),
            output_schema=self.output_schema,
            tool_kind=self.tool_kind,
            timeout_in_ms=self.timeout_in_ms,
            is_private=self.is_private,
        )

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> str:
        raise NotImplementedError

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult | ExecutionDeferred:
        output = await self._execute(
            args,
            session,
            conversation_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            cancellation_token=cancellation_token,
        )
        return ExecutionResult(content=[TextContent(text=output)])


class AddTool(FakeTool):
    name = "add"
    description = "Add two numbers."
    Args = BinaryArgs

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> str:
        return str(args["a"] + args["b"])


class MultiplyTool(FakeTool):
    name = "multiply"
    description = "Multiply two numbers."
    Args = BinaryArgs

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> str:
        return str(args["a"] * args["b"])


class CapturingTool(FakeTool):
    name = "capture"
    description = "Records the AgentSession and token it received."
    Args = BinaryArgs

    def __init__(self) -> None:
        self.seen: list[AgentSession] = []
        self.tokens: list[CancellationToken] = []

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> str:
        self.seen.append(session)
        self.tokens.append(cancellation_token)
        return "captured"


class ReadFileTool(FakeTool):
    """A context-bearing tool: emits the resources/preview/remember_as dict
    (the user-land convention, via the duck-typed `get_approval_context`
    hook `FakeToolRegistry` reads) so approval-context plumbing is
    exercisable."""

    name = "read_file"
    description = "Read a file."
    Args = PathArgs
    tool_kind = ToolKind.READ

    async def get_approval_context(self, args: dict, session: AgentSession, conversation_id: str) -> dict:
        return {
            "resources": [args["path"]],
            "preview": f"Read {args['path']}",
            "remember_as": [{"resource": "/etc/*", "preview": "Allow /etc/*"}],
        }

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> str:
        return f"contents of {args['path']}"


class RaisingTool(FakeTool):
    name = "boom"
    description = "Always raises."
    Args = BinaryArgs

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> str:
        raise ValueError("kaboom")


class PrivateTool(FakeTool):
    """Runtime-only: the RUNTIME resolves and dispatches it, the model is never
    shown it, and a model call naming it is refused."""

    name = "secret"
    description = "Never advertised."
    Args = BinaryArgs
    is_private = True

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> str:
        return "ran privately"


class RichErrorTool(FakeTool):
    """Overrides the rich `execute` path and reports an execution failure."""

    name = "report"
    description = "Returns a rich is_error result."
    Args = BinaryArgs

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        return ExecutionResult(
            content=[TextContent(text="disk full")],
            is_error=True,
            metadata={"code": 28},
        )


class DeferringTool(FakeTool):
    """Defers until the test resolves the call by id.

    A PURE PREDICATE — it reads `answers` and returns; it never waits, which is
    what makes it safe for the runner to invoke once per drive. `answers` is
    the whole state, so a test resolves a specific parked call with
    `tool.answers["tc1"] = "..."` and drives again.

    `dispatches` records every body invocation by call id, which is how a test
    distinguishes "the drive parked" from "the drive re-selected the same
    execution and spun"."""

    name = "ask"
    description = "Asks and waits."
    Args = BinaryArgs

    def __init__(self, answers: dict[str, str] | None = None) -> None:
        self.answers: dict[str, str] = {} if answers is None else answers
        self.dispatches: list[str] = []

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult | ExecutionDeferred:
        self.dispatches.append(tool_call_id)
        answer = self.answers.get(tool_call_id)
        if answer is None:
            return ExecutionDeferred()
        return ExecutionResult(content=[TextContent(text=answer)])


# The specs the registry double produces for the doubles above. Session
# literals use these rather than a hand-written `ToolSpec` so a precondition
# and a freshly-created execution are byte-identical — including the
# `input_schema` derived from `Args`.
ADD_SPEC = AddTool().get_tool_spec()
MULTIPLY_SPEC = MultiplyTool().get_tool_spec()
READ_FILE_SPEC = ReadFileTool().get_tool_spec()
CAPTURE_SPEC = CapturingTool().get_tool_spec()
BOOM_SPEC = RaisingTool().get_tool_spec()
REPORT_SPEC = RichErrorTool().get_tool_spec()
SECRET_SPEC = PrivateTool().get_tool_spec()
ASK_SPEC = DeferringTool().get_tool_spec()


# ── mid-state session literals ─────────────────────────────────────────────────

# A turn paused at the approval gate: the assistant requested `add(1, 2)`, the
# eagerly-persisted execution was handed to the strategy, and the strategy
# punted — approval_status is PENDING and the audit log records the deferral.
GATED_SESSION = make_session(
    id="s_gated",
    entries={
        "u1": UserMessage(
            id="u1",
            created_at=500,
            parts=[TextContent(text="Add 1 and 2")],
        ),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
        "a1": AssistantMessage(
            id="a1",
            parent_id="ts",
            created_at=500,
            parts=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
            llm_config=MODEL,
            stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1",
            conversation_id="c1",
            parent_id="a1",
            created_at=500,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            tool_spec=ADD_SPEC,
            status=ExecutionStatus.PENDING,
            result=None,
            approval_status=ApprovalStatus.PENDING,
            approval_decisions=[
                ApprovalDecision(decision=ApprovalOption.PENDING, created_at=500),
            ],
            updated_at=500,
        ),
    },
    conversations={
        "c1": conversation(
            "c1",
            ["u1", "ts", "a1", "te1"],
            created_at=500,
            updated_at=500,
        )
    },
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)

# The same turn one step later: on a later run() the strategy resolved the call
# (approval_status ALLOWED; the ALLOW appended after the PENDING in the audit
# log) but the execution has not run yet — the resume path must dispatch it
# before anything else, without re-deciding.
CLEARED_SESSION = make_session(
    id="s_cleared",
    entries={
        "u1": UserMessage(
            id="u1",
            created_at=500,
            parts=[TextContent(text="Add 1 and 2")],
        ),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
        "a1": AssistantMessage(
            id="a1",
            parent_id="ts",
            created_at=500,
            parts=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
            llm_config=MODEL,
            stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1",
            conversation_id="c1",
            parent_id="a1",
            created_at=500,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            tool_spec=ADD_SPEC,
            status=ExecutionStatus.PENDING,
            result=None,
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[
                ApprovalDecision(decision=ApprovalOption.PENDING, created_at=500),
                ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=600),
            ],
            updated_at=600,
        ),
    },
    conversations={
        "c1": conversation(
            "c1",
            ["u1", "ts", "a1", "te1"],
            created_at=500,
            updated_at=600,
        )
    },
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)

# A crash mid-decide: the execution was persisted eagerly but the process died
# before the strategy returned — approval_status None, empty audit log, and a
# stale RUNNING conversation status. Construction must self-heal to PENDING
# (not AWAITING_APPROVAL: the strategy was never asked, so a plain run()
# re-asks it).
UNDECIDED_SESSION = GATED_SESSION.model_copy(deep=True, update={"id": "s_undecided"})
UNDECIDED_SESSION.entries["te1"] = UNDECIDED_SESSION.entries["te1"].model_copy(
    update={"approval_decisions": [], "approval_status": None, "updated_at": None},
)

# The cleared session as a crashed process would have left it: the persisted
# status is a stale RUNNING that construction must self-heal to PENDING.
STALE_RUNNING_SESSION = CLEARED_SESSION.model_copy(deep=True, update={"id": "s_stale"})

# A parked cancel: the user abandoned the turn at the approval gate (cancel()
# appended the durable CancelRequested after the gated execution) and the
# process stopped before any drive consumed it. The next run()/start() is a
# FLUSH: wind down the executions, close the turn CANCELLED — no LLM call.
CANCEL_PARKED_SESSION = GATED_SESSION.model_copy(deep=True, update={"id": "s_parked"})
CANCEL_PARKED_SESSION.entries["cr"] = CancelRequested(
    id="cr",
    parent_id="te1",
    created_at=600,
)
main_conversation(CANCEL_PARKED_SESSION).nodes.append("cr")
main_conversation(CANCEL_PARKED_SESSION).updated_at = 600

# A failed turn as the engine leaves it: the LLM call timed out, the bracket
# was closed (TurnFinish carries the outcome + error) and the exception
# re-raised. Status is retry-ready PENDING — a plain run() opens a NEW bracket
# and re-answers; post_message is also legal (clarify before the retry).
POST_FAILURE_SESSION = make_session(
    id="s_failed",
    entries={
        "u1": UserMessage(
            id="u1",
            created_at=500,
            parts=[TextContent(text="Add 1 and 2")],
        ),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
        "tf": TurnFinish(
            id="tf",
            parent_id="ts",
            created_at=500,
            outcome=TurnOutcome.TIMED_OUT,
            error="completion exceeded total_timeout=0.05s",
        ),
    },
    conversations={
        "c1": conversation(
            "c1",
            ["u1", "ts", "tf"],
            created_at=500,
            updated_at=500,
        )
    },
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)

# A crash mid-body: the execution was persisted RUNNING (approval ALLOWED,
# with an OPEN ExecutionAttempt) and the process died before the tool settled —
# the
# session carries an orphaned RUNNING execution and a stale RUNNING status.
# The next drive must recover it to INTERRUPTED (after_tool_execution runs,
# no re-dispatch) and close its dangling attempt as DISCARDED before doing
# anything else.
RUNNING_ORPHAN_SESSION = make_session(
    id="s_orphan",
    entries={
        "u1": UserMessage(
            id="u1",
            created_at=500,
            parts=[TextContent(text="Add 1 and 2")],
        ),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
        "a1": AssistantMessage(
            id="a1",
            parent_id="ts",
            created_at=500,
            parts=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
            llm_config=MODEL,
            stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1",
            conversation_id="c1",
            parent_id="a1",
            created_at=500,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            tool_spec=ADD_SPEC,
            status=ExecutionStatus.RUNNING,
            result=None,
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[
                ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=500),
            ],
            attempts=[ExecutionAttempt(started_at=500)],
            updated_at=500,
        ),
    },
    conversations={
        "c1": conversation(
            "c1",
            ["u1", "ts", "a1", "te1"],
            created_at=500,
            updated_at=500,
        )
    },
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)


# ── compaction literals ───────────────────────────────────────────────────────

# The canonical feature-rich precondition for the compaction scenarios: one
# conversation carrying every entry shape the framework produces, plus a
# conversation that was already compacted once (`c0`, summarized by `cmp0`)
# and a pruned node whose referent is in the store but on no path. Tests
# assert against it whole — that carried entries are untouched, that compacted
# ones survive, and that `c0` is never revisited.
#
# Path shape: a prior summary, an answered tool turn, a FAILED turn, a
# CANCELLED turn, and a trailing unanswered question. Derived status: PENDING.
RICH_SESSION = make_session(
    id="s_rich",
    entries={
        # on c0 only — replaced by cmp0, still in the store
        "u0": UserMessage(
            id="u0",
            created_at=400,
            parts=[TextContent(text="Where do I sit?")],
        ),
        "a0": AssistantMessage(
            id="a0",
            parent_id="u0",
            created_at=400,
            parts=[TextContent(text="Row 4.")],
            llm_config=MODEL,
            stop_reason="stop",
        ),
        # the pruned referent: in the store, on no path
        "te0": ToolExecution(
            id="te0",
            conversation_id="c1",
            created_at=400,
            tool_call_id="tc0",
            raw_tool_call=ToolCall(
                id="tc0",
                name="read_file",
                arguments={"path": "/etc/motd"},
            ),
            tool_spec=READ_FILE_SPEC,
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="a long listing")]),
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[
                ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=400),
            ],
            attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=400, ended_at=400)],
            finished_at=400,
            updated_at=400,
        ),
        "cmp0": CompactionEntry(
            id="cmp0",
            created_at=500,
            source=CompactionSource.POLICY,
            parts=[
                TextContent(text="Earlier: the user asked where to sit."),
            ],
            compacted_nodes=["u0", "a0"],
            llm_config=CHEAP,
            started_at=500,
            ended_at=500,
            metadata={"strategy": "turn-brackets"},
            context_tokens=9,
        ),
        "u1": UserMessage(
            id="u1",
            parent_id="cmp0",
            created_at=500,
            parts=[
                TextContent(text="Add 1 and 2, and look at this"),
                ImageContent(
                    source=MediaBase64(data="aGk=", media_type="image/png"),
                ),
            ],
        ),
        "ts1": TurnStart(id="ts1", parent_id="u1", created_at=500),
        "a1": AssistantMessage(
            id="a1",
            parent_id="ts1",
            created_at=500,
            parts=[
                ThinkingContent(thinking="Both tools.", signature="sig-1"),
                TextContent(text="On it."),
                ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
                ToolCall(
                    id="tc2",
                    name="read_file",
                    arguments={"path": "/etc/hosts"},
                ),
            ],
            llm_config=MODEL,
            stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1",
            conversation_id="c1",
            parent_id="a1",
            created_at=500,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            tool_spec=ADD_SPEC,
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="3")]),
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[
                ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=500),
            ],
            attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=500, ended_at=500)],
            finished_at=500,
            updated_at=500,
        ),
        "te2": ToolExecution(
            id="te2",
            conversation_id="c1",
            parent_id="te1",
            created_at=500,
            tool_call_id="tc2",
            raw_tool_call=ToolCall(
                id="tc2",
                name="read_file",
                arguments={"path": "/etc/hosts"},
            ),
            tool_spec=READ_FILE_SPEC,
            status=ExecutionStatus.FAILED,
            error=ToolExecutionError(
                error_type="PermissionError",
                error_message="permission denied: /etc/hosts",
                details={"phase": "execution"},
            ),
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[
                ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=500),
            ],
            attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.FAILED, started_at=500, ended_at=500)],
            finished_at=500,
            updated_at=500,
        ),
        "pr1": PrunedEntry(
            id="pr1",
            parent_id="te2",
            created_at=500,
            pruned_entry_type="tool_execution",
            pruned_entry_id="te0",
            content=[TextContent(text="[tool output has been pruned]")],
            context_tokens=7,
        ),
        "a2": AssistantMessage(
            id="a2",
            parent_id="pr1",
            created_at=500,
            parts=[TextContent(text="1 + 2 = 3.")],
            llm_config=MODEL,
            stop_reason="stop",
        ),
        "tf1": TurnFinish(id="tf1", parent_id="a2", created_at=500),
        "u2": UserMessage(
            id="u2",
            parent_id="tf1",
            created_at=500,
            parts=[TextContent(text="And 2 + 2?")],
        ),
        "ts2": TurnStart(id="ts2", parent_id="u2", created_at=500),
        "tf2": TurnFinish(
            id="tf2",
            parent_id="ts2",
            created_at=500,
            outcome=TurnOutcome.ERRORED,
            error="provider error",
        ),
        "u3": UserMessage(
            id="u3",
            parent_id="tf2",
            created_at=500,
            parts=[TextContent(text="Never mind — multiply 3 by 4.")],
        ),
        "ts3": TurnStart(id="ts3", parent_id="u3", created_at=500),
        "a3": AssistantMessage(
            id="a3",
            parent_id="ts3",
            created_at=500,
            parts=[
                ToolCall(id="tc3", name="multiply", arguments={"a": 3, "b": 4}),
            ],
            llm_config=MODEL,
            stop_reason="tool_use",
        ),
        "te3": ToolExecution(
            id="te3",
            conversation_id="c1",
            parent_id="a3",
            created_at=500,
            tool_call_id="tc3",
            raw_tool_call=ToolCall(
                id="tc3",
                name="multiply",
                arguments={"a": 3, "b": 4},
            ),
            tool_spec=MULTIPLY_SPEC,
            status=ExecutionStatus.REJECTED,
            approval_status=ApprovalStatus.REJECTED,
            approval_decisions=[
                ApprovalDecision(decision=ApprovalOption.DENY, created_at=500),
            ],
            finished_at=500,
            updated_at=500,
        ),
        "cr1": CancelRequested(id="cr1", parent_id="te3", created_at=500),
        "tf3": TurnFinish(
            id="tf3",
            parent_id="cr1",
            created_at=500,
            outcome=TurnOutcome.CANCELLED,
        ),
        "u4": UserMessage(
            id="u4",
            parent_id="tf3",
            created_at=500,
            parts=[TextContent(text="What is X?")],
        ),
    },
    usages={
        "c0": {
            "a0": Usage(
                conversation_id="c0",
                entry_id="a0",
                input=10,
                output=5,
                total_tokens=15,
            ),
            "cmp0": Usage(
                conversation_id="c0",
                entry_id="cmp0",
                input=20,
                output=8,
                total_tokens=28,
            ),
        },
        "c1": {
            "a1": Usage(
                conversation_id="c1",
                entry_id="a1",
                input=40,
                output=12,
                total_tokens=52,
            ),
            "a2": Usage(
                conversation_id="c1",
                entry_id="a2",
                input=60,
                output=6,
                total_tokens=66,
            ),
            "a3": Usage(
                conversation_id="c1",
                entry_id="a3",
                input=80,
                output=9,
                total_tokens=89,
            ),
        },
    },
    conversations={
        "c1": conversation(
            "c1",
            [
                "cmp0",
                "u1",
                "ts1",
                "a1",
                "te1",
                "te2",
                "pr1",
                "a2",
                "tf1",
                "u2",
                "ts2",
                "tf2",
                "u3",
                "ts3",
                "a3",
                "te3",
                "cr1",
                "tf3",
                "u4",
            ],
            created_at=500,
            updated_at=500,
        )
    },
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)
# `c0` is the conversation `cmp0` archived. It stays a first-class row — usage
# records key on it (see `usages` above) — reachable from `c1` backwards.
RICH_SESSION.conversations["c0"] = conversation("c0", ["u0", "a0"], created_at=400, updated_at=400)
main_conversation(RICH_SESSION).previous_conversation_id = "c0"

# The same conversation with the trailing question answered — nothing is
# queued, so a compaction here is the whole drive. Derived status: IDLE.
RICH_IDLE_SESSION = RICH_SESSION.model_copy(deep=True, update={"id": "s_rich_idle"})
RICH_IDLE_SESSION.entries.pop("u4")
main_conversation(RICH_IDLE_SESSION).nodes.remove("u4")

# `schedule_compaction()` as it lands on disk: the bracket and the entry are
# durable, nothing has started. Derived status: PENDING — an open bracket
# already means "work is queued, call run()".
COMPACTION_SCHEDULED_SESSION = RICH_SESSION.model_copy(
    deep=True,
    update={"id": "s_scheduled"},
)
COMPACTION_SCHEDULED_SESSION.entries["ts_c"] = TurnStart(
    id="ts_c",
    parent_id="u4",
    created_at=600,
)
COMPACTION_SCHEDULED_SESSION.entries["cmp"] = CompactionEntry(
    id="cmp",
    parent_id="ts_c",
    created_at=600,
    source=CompactionSource.USER,
)
main_conversation(COMPACTION_SCHEDULED_SESSION).nodes += ["ts_c", "cmp"]
main_conversation(COMPACTION_SCHEDULED_SESSION).updated_at = 600

# A crash MID-SUMMARY: `started_at` is stamped, `parts` never landed, and the
# persisted status is a stale RUNNING that construction self-heals. The next
# drive resumes THIS entry — no second bracket.
COMPACTION_INTERRUPTED_SESSION = COMPACTION_SCHEDULED_SESSION.model_copy(
    deep=True,
    update={"id": "s_interrupted"},
)
COMPACTION_INTERRUPTED_SESSION.entries["cmp"] = COMPACTION_INTERRUPTED_SESSION.entries["cmp"].model_copy(
    update={"started_at": 600},
)

# The user cancelled a scheduled compaction and the process stopped before any
# drive consumed it. The next run() is the FLUSH: close the bracket, no policy
# call. Derived status: CANCELLING.
COMPACTION_CANCEL_PARKED_SESSION = COMPACTION_SCHEDULED_SESSION.model_copy(
    deep=True,
    update={"id": "s_compaction_cancelled"},
)
COMPACTION_CANCEL_PARKED_SESSION.entries["cr"] = CancelRequested(
    id="cr",
    parent_id="cmp",
    created_at=700,
)
main_conversation(COMPACTION_CANCEL_PARKED_SESSION).nodes.append("cr")

# A compaction that FAILED and closed. The bracket is closed, so it is never
# retried — and the skip rule derives IDLE from the turn before it, which is
# what stops a `while not runner.idle()` loop from opening a fresh bracket
# every drive forever.
COMPACTION_FAILED_SESSION = RICH_IDLE_SESSION.model_copy(
    deep=True,
    update={"id": "s_compaction_failed"},
)
COMPACTION_FAILED_SESSION.entries["ts_c"] = TurnStart(
    id="ts_c",
    parent_id="tf3",
    created_at=600,
)
COMPACTION_FAILED_SESSION.entries["cmp"] = CompactionEntry(
    id="cmp",
    parent_id="ts_c",
    created_at=600,
    source=CompactionSource.POLICY,
    started_at=600,
    ended_at=700,
)
COMPACTION_FAILED_SESSION.entries["tf_c"] = TurnFinish(
    id="tf_c",
    parent_id="cmp",
    created_at=700,
    outcome=TurnOutcome.ERRORED,
    error="the policy raised",
)
main_conversation(COMPACTION_FAILED_SESSION).nodes += ["ts_c", "cmp", "tf_c"]
main_conversation(COMPACTION_FAILED_SESSION).updated_at = 700

# A no-op compaction closed behind a queued question, so `u4` is no longer the
# leaf. Without the skip rule this session derives IDLE and the question is
# silently never answered. Derived status: PENDING.
COMPACTION_BURIED_SESSION = COMPACTION_SCHEDULED_SESSION.model_copy(
    deep=True,
    update={"id": "s_buried"},
)
COMPACTION_BURIED_SESSION.entries["cmp"] = COMPACTION_BURIED_SESSION.entries["cmp"].model_copy(
    update={"started_at": 600, "ended_at": 700},
)
COMPACTION_BURIED_SESSION.entries["tf_c"] = TurnFinish(
    id="tf_c",
    parent_id="cmp",
    created_at=700,
)
main_conversation(COMPACTION_BURIED_SESSION).nodes.append("tf_c")
main_conversation(COMPACTION_BURIED_SESSION).updated_at = 700

# A COMMITTED transition as it lands on disk: `c1` archived with the bracket
# it ran in, `c2` active over the summary plus the question it kept. Every
# compacted entry is still in `entries`; nothing was destroyed.
POST_COMPACTION_SESSION = RICH_SESSION.model_copy(
    deep=True,
    update={"id": "s_post_compaction"},
)
POST_COMPACTION_SESSION.entries["ts_c"] = TurnStart(
    id="ts_c",
    parent_id="u4",
    created_at=600,
)
POST_COMPACTION_SESSION.entries["cmp"] = CompactionEntry(
    id="cmp",
    parent_id="ts_c",
    created_at=600,
    source=CompactionSource.POLICY,
    parts=[TextContent(text="The user added 1 and 2 and was answered 3.")],
    compacted_nodes=[
        "cmp0",
        "u1",
        "ts1",
        "a1",
        "te1",
        "te2",
        "pr1",
        "a2",
        "tf1",
        "u2",
        "ts2",
        "tf2",
        "u3",
        "ts3",
        "a3",
        "te3",
        "cr1",
        "tf3",
    ],
    llm_config=CHEAP,
    started_at=600,
    ended_at=700,
    context_tokens=10,
)
POST_COMPACTION_SESSION.entries["tf_c"] = TurnFinish(
    id="tf_c",
    parent_id="cmp",
    created_at=700,
)
POST_COMPACTION_SESSION.usages["c1"]["cmp"] = Usage(
    conversation_id="c1",
    entry_id="cmp",
    input=500,
    output=40,
    total_tokens=540,
)
# The transition: `c1` keeps the bracket it ran in and stays in the catalog;
# `c2` is installed over the summary plus the question it kept, names `c1` as
# its predecessor, and becomes the main conversation. Nothing is destroyed and
# nothing moves — only the NAME moves.
_OUTGOING = main_conversation(POST_COMPACTION_SESSION)
_OUTGOING.nodes += ["ts_c", "cmp", "tf_c"]
_OUTGOING.updated_at = 700
POST_COMPACTION_SESSION.conversations["c2"] = conversation(
    "c2",
    ["cmp", "u4"],
    created_at=700,
    updated_at=700,
    previous_conversation_id="c1",
)
POST_COMPACTION_SESSION.main_conversation_id = "c2"
