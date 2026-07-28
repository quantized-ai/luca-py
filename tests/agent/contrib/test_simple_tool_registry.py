"""Self-scoped tests for `luca.agent.contrib.simple_tool_registry`: the
`ToolSpec` list `get_tools` advertises, the birth drafts `create_execution`
authors per preflight outcome, `decide` delegation to the `PermissionPolicy`,
`prepare`'s resolve/validate/bind (it returns a callable and never runs the
body), and `ProxyToolRegistry` routing — recompute+cache, duplicate-name
rejection, cache-independent `decide`/`prepare` resolution, ALLOW on a name no
child owns, and nesting.

No runner here — the four-method registry contract is exercised directly,
exactly as the runner would call it: every method is async and takes the live
`AgentSession` first.
"""

import pytest
from pydantic import BaseModel, ConfigDict

from luca.agent.contrib.simple_tool_registry import (
    PermissionPolicy,
    ProxyToolRegistry,
    SimpleToolRegistry,
    YoloPermissionPolicy,
)
from luca.agent.contrib.tools import Tool
from luca.agent.core import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    CancellationToken,
    Conversation,
    ExecutionResult,
    ExecutionStatus,
    InvalidToolArguments,
    LLMConfig,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    ToolNotFound,
    ToolSpec,
)

MODEL = LLMConfig(model="test-model", provider="faux")

SESSION = AgentSession(
    id="s_registry",
    active_conversation=Conversation(id="c1", nodes=[], created_at=500, updated_at=500),
    session_config=SessionConfig(llm_config=MODEL),
)


# ── tool doubles ──────────────────────────────────────────────────────────────


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
        self,
        args: dict,
        session: AgentSession,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        return str(args["a"] + args["b"])


class MultiplyTool(Tool):
    name = "multiply"
    description = "Multiply two numbers."
    Args = BinaryArgs
    timeout_in_ms = 5000

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        return str(args["a"] * args["b"])


class ReadFileTool(Tool):
    """Supplies the duck-typed `get_approval_context` convention. The echoed
    `session_id` is what proves the live session reaches the hook."""

    name = "read_file"
    description = "Read a file."
    Args = PathArgs

    async def get_approval_context(self, args: dict, session: AgentSession) -> dict:
        return {
            "resources": [args["path"]],
            "preview": f"Read {args['path']}",
            "session_id": session.id,
        }

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        return f"contents of {args['path']}"


class BrokenContextTool(ReadFileTool):
    name = "broken_context"

    async def get_approval_context(self, args: dict, session: AgentSession) -> dict:
        raise RuntimeError("context exploded")


class CapturingTool(Tool):
    """Records what the prepared callable handed the body."""

    name = "capture"
    description = "Records its arguments, session, and token."
    Args = BinaryArgs

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        self.calls.append((args, session, cancellation_token))
        return "captured"


# The specs the doubles above snapshot themselves into: `input_schema` is the
# `Args` model's JSON Schema verbatim. Written out as literals rather than
# taken from `get_tool_spec()` so the expectation is independent of the
# method under test.
ADD_SPEC = ToolSpec(
    name="add",
    description="Add two numbers.",
    input_schema=BinaryArgs.model_json_schema(),
)
MULTIPLY_SPEC = ToolSpec(
    name="multiply",
    description="Multiply two numbers.",
    input_schema=BinaryArgs.model_json_schema(),
    timeout_in_ms=5000,
)
READ_FILE_SPEC = ToolSpec(
    name="read_file",
    description="Read a file.",
    input_schema=PathArgs.model_json_schema(),
)
BROKEN_CONTEXT_SPEC = ToolSpec(
    name="broken_context",
    description="Read a file.",
    input_schema=PathArgs.model_json_schema(),
)
CAPTURE_SPEC = ToolSpec(
    name="capture",
    description="Records its arguments, session, and token.",
    input_schema=BinaryArgs.model_json_schema(),
)


class RecordingPolicy(PermissionPolicy):
    """Records every `(session, execution)` pair it decides; always ALLOW with
    a frozen stamp."""

    def __init__(self) -> None:
        self.seen: list[tuple[AgentSession, ToolExecution]] = []

    async def decide(
        self,
        session: AgentSession,
        tool_execution: ToolExecution,
    ) -> ApprovalDecision:
        self.seen.append((session, tool_execution))
        return ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)


def stamped(draft: ToolExecution) -> ToolExecution:
    """What the runner does to a draft before decide/prepare see it."""
    return draft.model_copy(update={"id": "te1", "created_at": 1000})


def cold_execution(call: ToolCall) -> ToolExecution:
    """An approved execution as a fresh process loads it: no `get_tools()` has
    run, so nothing warmed a proxy's route. `raw_tool_call.name` is the only
    thing `decide`/`prepare` resolve from."""
    return ToolExecution(
        id="te1",
        created_at=1000,
        tool_call_id=call.id,
        raw_tool_call=call,
        status=ExecutionStatus.PENDING,
    )


def child(*tools: Tool) -> SimpleToolRegistry:
    return SimpleToolRegistry(
        tools=list(tools),
        permission_policy=YoloPermissionPolicy(),
    )


# ── SimpleToolRegistry: get_tools ─────────────────────────────────────────────


async def test_get_tools_returns_a_spec_per_tool_in_order():
    registry = SimpleToolRegistry(
        tools=[AddTool(), MultiplyTool()],
        permission_policy=YoloPermissionPolicy(),
    )

    specs = await registry.get_tools(SESSION)

    assert specs == [ADD_SPEC, MULTIPLY_SPEC]


# ── SimpleToolRegistry: create_execution (the birth drafts) ───────────────────


async def test_unknown_tool_births_a_not_found_draft():
    registry = SimpleToolRegistry(
        tools=[AddTool()],
        permission_policy=YoloPermissionPolicy(),
    )
    call = ToolCall(id="tc1", name="nope", arguments={"x": 1})

    draft = await registry.create_execution(SESSION, call)

    assert draft == ToolExecution(
        id=None,
        created_at=None,
        tool_call_id="tc1",
        raw_tool_call=call,
        tool_spec=None,
        tool_spec_id=None,
        status=ExecutionStatus.NOT_FOUND,
        error=ToolExecutionError(
            error_type="ToolNotFound",
            error_message="Unknown tool: 'nope'.",
        ),
        extras={},
    )


async def test_invalid_arguments_birth_an_invalid_draft():
    registry = SimpleToolRegistry(
        tools=[AddTool()],
        permission_policy=YoloPermissionPolicy(),
    )
    call = ToolCall(id="tc1", name="add", arguments={"a": 1})

    draft = await registry.create_execution(SESSION, call)

    assert draft == ToolExecution(
        id=None,
        created_at=None,
        tool_call_id="tc1",
        raw_tool_call=call,
        tool_spec=ADD_SPEC,
        tool_spec_id=None,
        status=ExecutionStatus.INVALID,
        error=ToolExecutionError(
            error_type="InvalidToolArguments",
            error_message="Arguments for tool 'add' are invalid.",
            details={
                "errors": [
                    {
                        "type": "missing",
                        "loc": ["b"],
                        "msg": "Field required",
                        "input": {"a": 1},
                    },
                ],
            },
        ),
        extras={},
    )


async def test_raising_approval_context_births_a_failed_draft():
    registry = SimpleToolRegistry(
        tools=[BrokenContextTool()],
        permission_policy=YoloPermissionPolicy(),
    )
    call = ToolCall(id="tc1", name="broken_context", arguments={"path": "/etc"})

    draft = await registry.create_execution(SESSION, call)

    assert draft == ToolExecution(
        id=None,
        created_at=None,
        tool_call_id="tc1",
        raw_tool_call=call,
        tool_spec=BROKEN_CONTEXT_SPEC,
        tool_spec_id=None,
        status=ExecutionStatus.FAILED,
        error=ToolExecutionError(
            error_type="RuntimeError",
            error_message="context exploded",
            details={"phase": "approval_context"},
        ),
        extras={},
    )


async def test_healthy_call_births_a_pending_draft_with_approval_context():
    registry = SimpleToolRegistry(
        tools=[ReadFileTool()],
        permission_policy=YoloPermissionPolicy(),
    )
    call = ToolCall(id="tc1", name="read_file", arguments={"path": "/etc/hosts"})

    draft = await registry.create_execution(SESSION, call)

    assert draft == ToolExecution(
        id=None,
        created_at=None,
        tool_call_id="tc1",
        raw_tool_call=call,
        tool_spec=READ_FILE_SPEC,
        tool_spec_id=None,
        status=ExecutionStatus.PENDING,
        extras={
            "approval_context": {
                "resources": ["/etc/hosts"],
                "preview": "Read /etc/hosts",
                "session_id": "s_registry",
            },
        },
    )


async def test_plain_tool_births_a_pending_draft_with_empty_extras():
    registry = SimpleToolRegistry(
        tools=[MultiplyTool()],
        permission_policy=YoloPermissionPolicy(),
    )
    call = ToolCall(id="tc1", name="multiply", arguments={"a": 3, "b": 4})

    draft = await registry.create_execution(SESSION, call)

    # the declared deadline rides on the birth spec
    assert draft == ToolExecution(
        id=None,
        created_at=None,
        tool_call_id="tc1",
        raw_tool_call=call,
        tool_spec=MULTIPLY_SPEC,
        tool_spec_id=None,
        status=ExecutionStatus.PENDING,
        extras={},
    )


# ── SimpleToolRegistry: decide delegation ─────────────────────────────────────


async def test_decide_delegates_the_session_and_execution_to_the_policy():
    policy = RecordingPolicy()
    registry = SimpleToolRegistry(tools=[AddTool()], permission_policy=policy)
    execution = stamped(
        await registry.create_execution(
            SESSION,
            ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        )
    )

    decision = await registry.decide(SESSION, execution)

    assert decision == ApprovalDecision(
        decision=ApprovalOption.ALLOW,
        created_at=1000,
    )
    assert policy.seen == [(SESSION, execution)]


async def test_yolo_policy_allows_every_execution():
    execution = cold_execution(
        ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
    )

    decision = await YoloPermissionPolicy().decide(SESSION, execution)

    # `created_at` is `ApprovalDecision`'s wall-clock default — noise this
    # test does not own.
    assert decision == ApprovalDecision(
        decision=ApprovalOption.ALLOW,
        created_at=decision.created_at,
    )


# ── SimpleToolRegistry: prepare ───────────────────────────────────────────────


async def test_prepare_returns_a_callable_without_running_the_body():
    tool = CapturingTool()
    registry = SimpleToolRegistry(
        tools=[tool],
        permission_policy=YoloPermissionPolicy(),
    )
    execution = stamped(
        await registry.create_execution(
            SESSION,
            ToolCall(id="tc1", name="capture", arguments={"a": 1, "b": 2}),
        )
    )

    prepared = await registry.prepare(SESSION, execution)

    assert callable(prepared)
    assert tool.calls == []


async def test_the_prepared_callable_runs_the_body_with_the_bound_arguments():
    tool = CapturingTool()
    token = CancellationToken()
    registry = SimpleToolRegistry(
        tools=[tool],
        permission_policy=YoloPermissionPolicy(),
    )
    execution = stamped(
        await registry.create_execution(
            SESSION,
            ToolCall(id="tc1", name="capture", arguments={"a": 1, "b": 2}),
        )
    )
    prepared = await registry.prepare(SESSION, execution)

    result = await prepared(cancellation_token=token)

    assert result == ExecutionResult(content=[TextContent(text="captured")])
    assert tool.calls == [({"a": 1, "b": 2}, SESSION, token)]


async def test_prepare_resolves_by_the_effective_call_name():
    # the execution's raw_tool_call (possibly middleware-rewritten) is the
    # dispatch authority — an unknown effective name raises ToolNotFound
    registry = SimpleToolRegistry(
        tools=[AddTool()],
        permission_policy=YoloPermissionPolicy(),
    )
    execution = stamped(
        await registry.create_execution(
            SESSION,
            ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        )
    ).model_copy(
        update={
            "raw_tool_call": ToolCall(id="tc1", name="renamed", arguments={}),
        }
    )

    with pytest.raises(ToolNotFound, match="Unknown tool: 'renamed'"):
        await registry.prepare(SESSION, execution)


async def test_prepare_wraps_validation_failures_in_invalid_tool_arguments():
    registry = SimpleToolRegistry(
        tools=[AddTool()],
        permission_policy=YoloPermissionPolicy(),
    )
    execution = stamped(
        await registry.create_execution(
            SESSION,
            ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        )
    ).model_copy(
        update={
            "raw_tool_call": ToolCall(id="tc1", name="add", arguments={"a": 1}),
        }
    )

    with pytest.raises(InvalidToolArguments) as excinfo:
        await registry.prepare(SESSION, execution)

    assert excinfo.value.errors == [
        {
            "type": "missing",
            "loc": ["b"],
            "msg": "Field required",
            "input": {"a": 1},
        },
    ]


# ── ProxyToolRegistry: get_tools ──────────────────────────────────────────────


async def test_proxy_get_tools_concatenates_children_in_order():
    proxy = ProxyToolRegistry(child(AddTool(), MultiplyTool()), child(ReadFileTool()))

    specs = await proxy.get_tools(SESSION)

    assert specs == [ADD_SPEC, MULTIPLY_SPEC, READ_FILE_SPEC]


async def test_proxy_get_tools_rejects_duplicate_names():
    proxy = ProxyToolRegistry(child(AddTool()), child(AddTool()))

    with pytest.raises(ValueError, match="Duplicate tool name across registries"):
        await proxy.get_tools(SESSION)


async def test_proxy_get_tools_recomputes_the_route():
    # a dynamic child may change its answer between calls; each get_tools
    # rebuilds the cache from scratch
    proxy = ProxyToolRegistry(child(AddTool()))
    await proxy.get_tools(SESSION)
    proxy.add_registry(child(MultiplyTool()))

    specs = await proxy.get_tools(SESSION)

    assert specs == [ADD_SPEC, MULTIPLY_SPEC]


# ── ProxyToolRegistry: routing ────────────────────────────────────────────────


async def test_proxy_create_execution_routes_to_the_owning_child():
    proxy = ProxyToolRegistry(child(AddTool()), child(MultiplyTool()))
    await proxy.get_tools(SESSION)  # warm the route
    call = ToolCall(id="tc1", name="multiply", arguments={"a": 3, "b": 4})

    draft = await proxy.create_execution(SESSION, call)

    assert draft == ToolExecution(
        id=None,
        created_at=None,
        tool_call_id="tc1",
        raw_tool_call=call,
        tool_spec=MULTIPLY_SPEC,
        tool_spec_id=None,
        status=ExecutionStatus.PENDING,
        extras={},
    )


async def test_proxy_create_execution_births_not_found_on_a_cold_route():
    # a tool call only ever arrives after an LLM call, which warmed the route
    # on its way out — create_execution routes from the plain cache
    proxy = ProxyToolRegistry(child(AddTool()))
    call = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})

    draft = await proxy.create_execution(SESSION, call)

    assert draft == ToolExecution(
        id=None,
        created_at=None,
        tool_call_id="tc1",
        raw_tool_call=call,
        tool_spec=None,
        tool_spec_id=None,
        status=ExecutionStatus.NOT_FOUND,
        error=ToolExecutionError(
            error_type="ToolNotFound",
            error_message="Unknown tool: 'add'.",
        ),
        extras={},
    )


async def test_proxy_decide_gates_a_cold_route_through_the_owning_childs_policy():
    # no get_tools() has run: decide resolves independently of the cache, so a
    # call resumed in a fresh process is still gated by its owner
    policy_a, policy_b = RecordingPolicy(), RecordingPolicy()
    proxy = ProxyToolRegistry(
        SimpleToolRegistry(tools=[AddTool()], permission_policy=policy_a),
        SimpleToolRegistry(tools=[MultiplyTool()], permission_policy=policy_b),
    )
    execution = cold_execution(
        ToolCall(id="tc1", name="multiply", arguments={"a": 3, "b": 4}),
    )

    decision = await proxy.decide(SESSION, execution)

    assert decision == ApprovalDecision(
        decision=ApprovalOption.ALLOW,
        created_at=1000,
    )
    assert policy_a.seen == []
    assert policy_b.seen == [(SESSION, execution)]


async def test_proxy_prepare_binds_a_cold_route_to_the_owning_childs_tool():
    proxy = ProxyToolRegistry(child(AddTool()), child(MultiplyTool()))
    execution = cold_execution(
        ToolCall(id="tc1", name="multiply", arguments={"a": 3, "b": 4}),
    )
    prepared = await proxy.prepare(SESSION, execution)

    result = await prepared(cancellation_token=CancellationToken())

    assert result == ExecutionResult(content=[TextContent(text="12")])


async def test_proxy_decide_allows_a_name_no_child_owns():
    # ALLOW, not DENY: prepare() raises ToolNotFound for it and the call
    # records the honest NOT_FOUND rather than a false REJECTED
    policy = RecordingPolicy()
    proxy = ProxyToolRegistry(
        SimpleToolRegistry(tools=[AddTool()], permission_policy=policy),
    )
    execution = cold_execution(ToolCall(id="tc1", name="nope", arguments={}))

    decision = await proxy.decide(SESSION, execution)

    # `created_at` is `ApprovalDecision`'s wall-clock default — noise this
    # test does not own.
    assert decision == ApprovalDecision(
        decision=ApprovalOption.ALLOW,
        created_at=decision.created_at,
    )
    assert policy.seen == []


async def test_proxy_prepare_raises_tool_not_found_for_a_name_no_child_owns():
    proxy = ProxyToolRegistry(child(AddTool()))
    execution = cold_execution(ToolCall(id="tc1", name="nope", arguments={}))

    with pytest.raises(ToolNotFound, match="Unknown tool: 'nope'"):
        await proxy.prepare(SESSION, execution)


async def test_nested_proxies_list_transparently():
    outer = ProxyToolRegistry(
        ProxyToolRegistry(child(AddTool())),
        child(MultiplyTool()),
    )

    specs = await outer.get_tools(SESSION)

    assert specs == [ADD_SPEC, MULTIPLY_SPEC]


async def test_nested_proxies_prepare_a_cold_route_through_the_inner_proxy():
    outer = ProxyToolRegistry(
        ProxyToolRegistry(child(AddTool())),
        child(MultiplyTool()),
    )
    execution = cold_execution(
        ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
    )
    prepared = await outer.prepare(SESSION, execution)

    result = await prepared(cancellation_token=CancellationToken())

    assert result == ExecutionResult(content=[TextContent(text="3")])
