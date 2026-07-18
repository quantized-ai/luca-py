"""Shared declarative preconditions for the runner tests.

Four kinds of building blocks:

- **`DeterministicRunner`**: the test-side extension of `AgentSessionRunner`.
  Determinism is the TEST'S concern, layered on by overriding the production
  hooks — `generate_id()` is scripted by `ids` (consumed in entry-creation
  order) and `now_ms()` is frozen to `now`. The production class carries no
  test parameters.
- **`FakeToolRegistry`**: the core-only deterministic `ToolRegistry` double
  (core tests must NOT import contrib). Static `get_tools`, a
  preflight-faithful `create_execution` (NOT_FOUND / INVALID /
  FAILED-from-approval-context / PENDING births with the classic error
  payloads), a scripted `decide` that records every execution it was asked
  about in `seen` (no script → deterministic allow-all with `created_at`
  frozen to `now`), and a resolve-validate-invoke `execute`.
- **Tool doubles**: minimal `Tool` subclasses covering each behavior the
  registry/runner must handle (plain success, a duck-typed approval context,
  captured context, a raised exception, a rich is_error result).
- **Mid-state session literals**: known `AgentSession`s exactly as they would
  sit on disk (a turn paused at the approval gate, an approved-but-unrun call,
  a crash mid-decide, a stale RUNNING status, an orphaned RUNNING execution).
  Tests load these *cold* — into a fresh runner — so every scenario reads as:
  GIVEN this persisted state, WHEN one action, THEN this outcome. Always take
  a `model_copy(deep=True)` before mutating.

All literals use `created_at=500` so entries written by the test's frozen
clock (`now=1000`) are visually distinct from the precondition.
"""

import json
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, ValidationError

from luca.agent.core.context import CancellationToken, ToolContext
from luca.agent.core.exceptions import InvalidToolArguments, ToolNotFound
from luca.agent.core.models import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    CancelRequested,
    Conversation,
    ConversationStatus,
    ExecutionResult,
    ExecutionStatus,
    LLMConfig,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    ToolKind,
    ToolSpec,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    UserMessage,
)
from luca.agent.core.runner import AgentSessionRunner
from luca.agent.core.tool_registry import ToolRegistry
from luca.agent.core.tools import Tool

MODEL = LLMConfig(model="test-model", provider="faux")


# ── registry double ────────────────────────────────────────────────────────────


class FakeToolRegistry(ToolRegistry):
    """Core-only deterministic registry double.

    `create_execution` is preflight-faithful: an unknown name births a
    NOT_FOUND draft (`tool_spec=None`), invalid `Args` an INVALID draft, a
    raising duck-typed `get_approval_context` a FAILED draft (its result
    otherwise lands in `extras["approval_context"]`), and a healthy call a
    PENDING draft — all with placeholder identity for the runner to stamp.
    `decide` records the execution snapshot it was asked about in `seen`,
    then pops the next scripted decision (calls arrive in unresolved-path
    order; exhaustion raises IndexError — the script no longer matches what
    the runner asked); with no script it ALLOWs everything, `created_at`
    frozen to `now`. `execute` resolves by the effective call's name,
    re-validates, and invokes the tool body."""

    def __init__(
        self,
        tools: Iterable[Tool] = (),
        decisions: Iterable[ApprovalDecision] | None = None,
        now: int = 1000,
    ) -> None:
        self.tools = list(tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.decisions = list(decisions) if decisions is not None else None
        self.now = now
        self.seen: list[ToolExecution] = []

    def get_tools(self, agent_session: AgentSession) -> list[Tool]:
        return list(self.tools)

    async def create_execution(
        self, call: ToolCall, context: ToolContext,
    ) -> ToolExecution:
        def draft(status, tool_spec=None, error=None, extras=None):
            return ToolExecution(
                id="", created_at=0,
                tool_call_id=call.id, raw_tool_call=call,
                tool_spec=tool_spec, status=status, error=error,
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
                    error_message=(
                        f"Arguments for tool {call.name!r} are invalid."
                    ),
                    details={
                        "errors": json.loads(exc.json(include_url=False)),
                    },
                ),
            )
        extras: dict = {}
        if hasattr(tool, "get_approval_context"):
            try:
                extras["approval_context"] = await tool.get_approval_context(
                    args.model_dump(), context,
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
            ExecutionStatus.PENDING, tool_spec=tool_spec, extras=extras,
        )

    async def decide(
        self, tool_execution: ToolExecution, context: ToolContext,
    ) -> ApprovalDecision:
        self.seen.append(tool_execution)
        if self.decisions is None:
            return ApprovalDecision(
                decision=ApprovalOption.ALLOW, created_at=self.now,
            )
        return self.decisions.pop(0)

    async def execute(
        self,
        tool_execution: ToolExecution,
        context: ToolContext,
        *,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
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
        return await tool.execute(
            args.model_dump(), context, cancellation_token=cancellation_token,
        )


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
        conversation_projector=None,
        context_manager=None,
        ids: Iterable[str] = (),
        now: int = 1000,
        middleware: list | None = None,
    ) -> None:
        self._ids = iter(ids)
        self._now = now
        super().__init__(
            session, tool_registry,
            system_prompt_parts, system_prompt_assembler,
            provider=provider,
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


class AddTool(Tool):
    name = "add"
    description = "Add two numbers."
    Args = BinaryArgs

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        return str(args["a"] + args["b"])


class MultiplyTool(Tool):
    name = "multiply"
    description = "Multiply two numbers."
    Args = BinaryArgs

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        return str(args["a"] * args["b"])


class CapturingTool(Tool):
    name = "capture"
    description = "Records the ToolContext and token it received."
    Args = BinaryArgs

    def __init__(self) -> None:
        self.seen: list[ToolContext] = []
        self.tokens: list[CancellationToken] = []

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        self.seen.append(context)
        self.tokens.append(cancellation_token)
        return "captured"


class ReadFileTool(Tool):
    """A context-bearing tool: emits the resources/preview/remember_as dict
    (the user-land convention, via the duck-typed `get_approval_context`
    hook `FakeToolRegistry` reads) so approval-context plumbing is
    exercisable."""

    name = "read_file"
    description = "Read a file."
    Args = PathArgs
    tool_kind = ToolKind.READ

    async def get_approval_context(self, args: dict, context: ToolContext) -> dict:
        return {
            "resources": [args["path"]],
            "preview": f"Read {args['path']}",
            "remember_as": [{"resource": "/etc/*", "preview": "Allow /etc/*"}],
        }

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        return f"contents of {args['path']}"


class RaisingTool(Tool):
    name = "boom"
    description = "Always raises."
    Args = BinaryArgs

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        raise ValueError("kaboom")


class RichErrorTool(Tool):
    """Overrides the rich `execute` path and reports an execution failure."""

    name = "report"
    description = "Returns a rich is_error result."
    Args = BinaryArgs

    async def execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        return ExecutionResult(
            content=[TextContent(text="disk full")],
            is_error=True,
            metadata={"code": 28},
        )


# ── mid-state session literals ─────────────────────────────────────────────────

# A turn paused at the approval gate: the assistant requested `add(1, 2)`, the
# eagerly-persisted execution was handed to the strategy, and the strategy
# punted — approval_status is PENDING and the audit log records the deferral.
GATED_SESSION = AgentSession(
    id="s_gated",
    entries={
        "u1": UserMessage(
            id="u1", created_at=500, parts=[TextContent(text="Add 1 and 2")],
        ),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
        "a1": AssistantMessage(
            id="a1", parent_id="ts", created_at=500,
            parts=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
            llm_config=MODEL, stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1", parent_id="a1", created_at=500,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            tool_spec=ToolSpec(name="add", description="Add two numbers."),
            status=ExecutionStatus.PENDING,
            result=None,
            approval_status=ApprovalStatus.PENDING,
            approval_decisions=[
                ApprovalDecision(decision=ApprovalOption.PENDING, created_at=500),
            ],
            updated_at=500,
        ),
    },
    tool_executions={"tc1": ["te1"]},
    active_conversation=Conversation(
        id="c1", nodes=["u1", "ts", "a1", "te1"], created_at=500, updated_at=500,
        status=ConversationStatus.AWAITING_APPROVAL,
    ),
    session_config=SessionConfig(llm_config=MODEL),
)

# The same turn one step later: on a later run() the strategy resolved the call
# (approval_status ALLOWED; the ALLOW appended after the PENDING in the audit
# log) but the execution has not run yet — the resume path must dispatch it
# before anything else, without re-deciding.
CLEARED_SESSION = AgentSession(
    id="s_cleared",
    entries={
        "u1": UserMessage(
            id="u1", created_at=500, parts=[TextContent(text="Add 1 and 2")],
        ),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
        "a1": AssistantMessage(
            id="a1", parent_id="ts", created_at=500,
            parts=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
            llm_config=MODEL, stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1", parent_id="a1", created_at=500,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            tool_spec=ToolSpec(name="add", description="Add two numbers."),
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
    tool_executions={"tc1": ["te1"]},
    active_conversation=Conversation(
        id="c1", nodes=["u1", "ts", "a1", "te1"], created_at=500, updated_at=600,
        status=ConversationStatus.PENDING,
    ),
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
UNDECIDED_SESSION.active_conversation.status = ConversationStatus.RUNNING

# The cleared session as a crashed process would have left it: the persisted
# status is a stale RUNNING that construction must self-heal to PENDING.
STALE_RUNNING_SESSION = CLEARED_SESSION.model_copy(deep=True, update={"id": "s_stale"})
STALE_RUNNING_SESSION.active_conversation.status = ConversationStatus.RUNNING

# A parked cancel: the user abandoned the turn at the approval gate (cancel()
# appended the durable CancelRequested after the gated execution) and the
# process stopped before any drive consumed it. The next run()/start() is a
# FLUSH: wind down the executions, close the turn CANCELLED — no LLM call.
CANCEL_PARKED_SESSION = GATED_SESSION.model_copy(deep=True, update={"id": "s_parked"})
CANCEL_PARKED_SESSION.entries["cr"] = CancelRequested(
    id="cr", parent_id="te1", created_at=600,
)
CANCEL_PARKED_SESSION.active_conversation.nodes.append("cr")
CANCEL_PARKED_SESSION.active_conversation.updated_at = 600
CANCEL_PARKED_SESSION.active_conversation.status = ConversationStatus.CANCELLING

# A failed turn as the engine leaves it: the LLM call timed out, the bracket
# was closed (TurnFinish carries the outcome + error) and the exception
# re-raised. Status is retry-ready PENDING — a plain run() opens a NEW bracket
# and re-answers; post_message is also legal (clarify before the retry).
POST_FAILURE_SESSION = AgentSession(
    id="s_failed",
    entries={
        "u1": UserMessage(
            id="u1", created_at=500, parts=[TextContent(text="Add 1 and 2")],
        ),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
        "tf": TurnFinish(
            id="tf", parent_id="ts", created_at=500,
            outcome=TurnOutcome.TIMED_OUT,
            error="completion exceeded total_timeout=0.05s",
        ),
    },
    active_conversation=Conversation(
        id="c1", nodes=["u1", "ts", "tf"], created_at=500, updated_at=500,
        status=ConversationStatus.PENDING,
    ),
    session_config=SessionConfig(llm_config=MODEL),
)

# A crash mid-body: the execution was persisted RUNNING (approval ALLOWED,
# started_at stamped) and the process died before the tool settled — the
# session carries an orphaned RUNNING execution and a stale RUNNING status.
# The next drive must recover it to INTERRUPTED (after_tool_execution runs,
# no re-dispatch) before doing anything else.
RUNNING_ORPHAN_SESSION = AgentSession(
    id="s_orphan",
    entries={
        "u1": UserMessage(
            id="u1", created_at=500, parts=[TextContent(text="Add 1 and 2")],
        ),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
        "a1": AssistantMessage(
            id="a1", parent_id="ts", created_at=500,
            parts=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
            llm_config=MODEL, stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1", parent_id="a1", created_at=500,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            tool_spec=ToolSpec(name="add", description="Add two numbers."),
            status=ExecutionStatus.RUNNING,
            result=None,
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[
                ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=500),
            ],
            started_at=500,
            updated_at=500,
        ),
    },
    tool_executions={"tc1": ["te1"]},
    active_conversation=Conversation(
        id="c1", nodes=["u1", "ts", "a1", "te1"], created_at=500, updated_at=500,
        status=ConversationStatus.RUNNING,
    ),
    session_config=SessionConfig(llm_config=MODEL),
)
