"""Self-scoped tests for `luca.agent.contrib.simple_tool_registry`: birth
drafts per preflight outcome, decide delegation to the `PermissionPolicy`,
execute resolution/validation/invocation, and `ProxyToolRegistry` routing
(recompute+cache, duplicate-name rejection, the three miss degradations,
nesting).

No runner here — the registry contract is exercised directly, exactly as the
runner would call it: `create_execution` with a `ToolCall` + `ToolContext`,
`decide`/`execute` with the (stamped) `ToolExecution`.
"""

import pytest
from pydantic import BaseModel, ConfigDict

from luca.agent.contrib.simple_tool_registry import (
    PermissionPolicy,
    ProxyToolRegistry,
    SimpleToolRegistry,
    YoloPermissionPolicy,
)
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
    Tool,
    ToolCall,
    ToolContext,
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

CONTEXT = ToolContext(session_id="s_registry", model=MODEL)


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
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        return str(args["a"] + args["b"])


class MultiplyTool(Tool):
    name = "multiply"
    description = "Multiply two numbers."
    Args = BinaryArgs
    timeout_in_ms = 5000

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        return str(args["a"] * args["b"])


class ReadFileTool(Tool):
    """Supplies the duck-typed `get_approval_context` convention."""

    name = "read_file"
    description = "Read a file."
    Args = PathArgs

    async def get_approval_context(self, args: dict, context: ToolContext) -> dict:
        return {"resources": [args["path"]], "preview": f"Read {args['path']}"}

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        return f"contents of {args['path']}"


class BrokenContextTool(ReadFileTool):
    name = "broken_context"

    async def get_approval_context(self, args: dict, context: ToolContext) -> dict:
        raise RuntimeError("context exploded")


class RecordingPolicy(PermissionPolicy):
    """Records every execution it decides; always ALLOW with a frozen stamp."""

    def __init__(self) -> None:
        self.seen: list[ToolExecution] = []

    async def decide(self, tool_execution: ToolExecution) -> ApprovalDecision:
        self.seen.append(tool_execution)
        return ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)


def stamped(draft: ToolExecution) -> ToolExecution:
    """What the runner does to a draft before decide/execute see it."""
    return draft.model_copy(update={"id": "te1", "created_at": 1000})


# ── SimpleToolRegistry: get_tools ─────────────────────────────────────────────


def test_get_tools_returns_the_static_list():
    add, multiply = AddTool(), MultiplyTool()
    registry = SimpleToolRegistry(
        tools=[add, multiply], permission_policy=YoloPermissionPolicy(),
    )

    assert registry.get_tools(SESSION) == [add, multiply]


# ── SimpleToolRegistry: create_execution (the birth drafts) ───────────────────


async def test_unknown_tool_births_a_not_found_draft():
    registry = SimpleToolRegistry(
        tools=[AddTool()], permission_policy=YoloPermissionPolicy(),
    )
    call = ToolCall(id="tc1", name="nope", arguments={"x": 1})

    draft = await registry.create_execution(call, CONTEXT)

    assert draft == ToolExecution(
        id="", created_at=0,
        tool_call_id="tc1",
        raw_tool_call=call,
        tool_spec=None,
        status=ExecutionStatus.NOT_FOUND,
        error=ToolExecutionError(
            error_type="ToolNotFound",
            error_message="Unknown tool: 'nope'.",
        ),
        extras={},
    )


async def test_invalid_arguments_birth_an_invalid_draft():
    registry = SimpleToolRegistry(
        tools=[AddTool()], permission_policy=YoloPermissionPolicy(),
    )
    call = ToolCall(id="tc1", name="add", arguments={"a": 1})

    draft = await registry.create_execution(call, CONTEXT)

    assert draft.status == ExecutionStatus.INVALID
    assert draft.tool_spec == ToolSpec(name="add", description="Add two numbers.")
    assert draft.error.error_type == "InvalidToolArguments"
    assert draft.error.error_message == "Arguments for tool 'add' are invalid."
    assert draft.error.details["errors"][0]["type"] == "missing"
    assert draft.error.details["errors"][0]["loc"] == ["b"]
    assert draft.extras == {}
    assert (draft.id, draft.created_at, draft.ended_at) == ("", 0, None)


async def test_raising_approval_context_births_a_failed_draft():
    registry = SimpleToolRegistry(
        tools=[BrokenContextTool()], permission_policy=YoloPermissionPolicy(),
    )
    call = ToolCall(id="tc1", name="broken_context", arguments={"path": "/etc"})

    draft = await registry.create_execution(call, CONTEXT)

    assert draft.status == ExecutionStatus.FAILED
    assert draft.error == ToolExecutionError(
        error_type="RuntimeError",
        error_message="context exploded",
        details={"phase": "approval_context"},
    )
    assert draft.extras == {}


async def test_healthy_call_births_a_pending_draft_with_approval_context():
    registry = SimpleToolRegistry(
        tools=[ReadFileTool()], permission_policy=YoloPermissionPolicy(),
    )
    call = ToolCall(id="tc1", name="read_file", arguments={"path": "/etc/hosts"})

    draft = await registry.create_execution(call, CONTEXT)

    assert draft == ToolExecution(
        id="", created_at=0,
        tool_call_id="tc1",
        raw_tool_call=call,
        tool_spec=ToolSpec(name="read_file", description="Read a file."),
        status=ExecutionStatus.PENDING,
        extras={
            "approval_context": {
                "resources": ["/etc/hosts"],
                "preview": "Read /etc/hosts",
            },
        },
    )


async def test_plain_tool_births_a_pending_draft_with_empty_extras():
    registry = SimpleToolRegistry(
        tools=[MultiplyTool()], permission_policy=YoloPermissionPolicy(),
    )
    call = ToolCall(id="tc1", name="multiply", arguments={"a": 3, "b": 4})

    draft = await registry.create_execution(call, CONTEXT)

    assert draft.status == ExecutionStatus.PENDING
    assert draft.extras == {}
    # the declared deadline rides on the birth spec
    assert draft.tool_spec == ToolSpec(
        name="multiply", description="Multiply two numbers.", timeout_in_ms=5000,
    )


# ── SimpleToolRegistry: decide delegation ─────────────────────────────────────


async def test_decide_delegates_to_the_permission_policy():
    policy = RecordingPolicy()
    registry = SimpleToolRegistry(tools=[AddTool()], permission_policy=policy)
    execution = stamped(await registry.create_execution(
        ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}), CONTEXT,
    ))

    decision = await registry.decide(execution, CONTEXT)

    assert decision == ApprovalDecision(
        decision=ApprovalOption.ALLOW, created_at=1000,
    )
    assert policy.seen == [execution]


async def test_yolo_policy_allows_with_wall_clock_stamp():
    execution = stamped(await SimpleToolRegistry(
        tools=[AddTool()], permission_policy=YoloPermissionPolicy(),
    ).create_execution(
        ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}), CONTEXT,
    ))

    decision = await YoloPermissionPolicy().decide(execution)

    assert decision.decision == ApprovalOption.ALLOW
    assert decision.metadata is None
    assert isinstance(decision.created_at, int) and decision.created_at > 0


# ── SimpleToolRegistry: execute ───────────────────────────────────────────────


async def test_execute_resolves_validates_and_invokes_the_tool():
    registry = SimpleToolRegistry(
        tools=[AddTool()], permission_policy=YoloPermissionPolicy(),
    )
    execution = stamped(await registry.create_execution(
        ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}), CONTEXT,
    ))

    result = await registry.execute(
        execution, CONTEXT, cancellation_token=CancellationToken(),
    )

    assert result == ExecutionResult(content=[TextContent(text="3")])


async def test_execute_resolves_by_the_effective_call_name():
    # the execution's raw_tool_call (possibly middleware-rewritten) is the
    # dispatch authority — an unknown effective name raises ToolNotFound
    registry = SimpleToolRegistry(
        tools=[AddTool()], permission_policy=YoloPermissionPolicy(),
    )
    execution = stamped(await registry.create_execution(
        ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}), CONTEXT,
    )).model_copy(update={
        "raw_tool_call": ToolCall(id="tc1", name="renamed", arguments={}),
    })

    with pytest.raises(ToolNotFound):
        await registry.execute(
            execution, CONTEXT, cancellation_token=CancellationToken(),
        )


async def test_execute_wraps_validation_failures_in_invalid_tool_arguments():
    registry = SimpleToolRegistry(
        tools=[AddTool()], permission_policy=YoloPermissionPolicy(),
    )
    execution = stamped(await registry.create_execution(
        ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}), CONTEXT,
    )).model_copy(update={
        "raw_tool_call": ToolCall(id="tc1", name="add", arguments={"a": 1}),
    })

    with pytest.raises(InvalidToolArguments) as excinfo:
        await registry.execute(
            execution, CONTEXT, cancellation_token=CancellationToken(),
        )
    assert excinfo.value.errors[0]["loc"] == ["b"]


# ── ProxyToolRegistry ─────────────────────────────────────────────────────────


def child(*tools: Tool) -> SimpleToolRegistry:
    return SimpleToolRegistry(
        tools=list(tools), permission_policy=YoloPermissionPolicy(),
    )


def test_proxy_get_tools_concatenates_children_in_order():
    add, multiply, read = AddTool(), MultiplyTool(), ReadFileTool()
    proxy = ProxyToolRegistry(child(add, multiply), child(read))

    assert proxy.get_tools(SESSION) == [add, multiply, read]


def test_proxy_get_tools_rejects_duplicate_names():
    proxy = ProxyToolRegistry(child(AddTool()), child(AddTool()))

    with pytest.raises(ValueError):
        proxy.get_tools(SESSION)


def test_proxy_get_tools_recomputes_the_route():
    # a dynamic child may change its answer between calls; each get_tools
    # rebuilds the cache from scratch
    first = child(AddTool())
    proxy = ProxyToolRegistry(first)
    proxy.get_tools(SESSION)

    proxy.add_registry(child(MultiplyTool()))

    assert [tool.name for tool in proxy.get_tools(SESSION)] == ["add", "multiply"]


async def test_proxy_routes_each_method_to_the_owning_child():
    policy_a, policy_b = RecordingPolicy(), RecordingPolicy()
    proxy = ProxyToolRegistry(
        SimpleToolRegistry(tools=[AddTool()], permission_policy=policy_a),
        SimpleToolRegistry(tools=[MultiplyTool()], permission_policy=policy_b),
    )
    proxy.get_tools(SESSION)  # warm the route

    draft = await proxy.create_execution(
        ToolCall(id="tc1", name="multiply", arguments={"a": 3, "b": 4}), CONTEXT,
    )
    execution = stamped(draft)
    decision = await proxy.decide(execution, CONTEXT)
    result = await proxy.execute(
        execution, CONTEXT, cancellation_token=CancellationToken(),
    )

    assert draft.tool_spec.name == "multiply"
    assert decision.decision == ApprovalOption.ALLOW
    assert policy_a.seen == []  # only the owning child decided
    assert policy_b.seen == [execution]
    assert result == ExecutionResult(content=[TextContent(text="12")])


async def test_proxy_create_execution_miss_births_a_not_found_draft():
    proxy = ProxyToolRegistry(child(AddTool()))  # route never warmed
    call = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})

    draft = await proxy.create_execution(call, CONTEXT)

    assert draft == ToolExecution(
        id="", created_at=0,
        tool_call_id="tc1",
        raw_tool_call=call,
        tool_spec=None,
        status=ExecutionStatus.NOT_FOUND,
        error=ToolExecutionError(
            error_type="ToolNotFound",
            error_message="Unknown tool: 'add'.",
        ),
    )


async def test_proxy_decide_miss_allows_so_execute_terminalizes_honestly():
    proxy = ProxyToolRegistry(child(AddTool()))  # route never warmed
    execution = ToolExecution(
        id="te1", created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        status=ExecutionStatus.PENDING,
    )

    decision = await proxy.decide(execution, CONTEXT)

    assert decision.decision == ApprovalOption.ALLOW


async def test_proxy_execute_miss_raises_tool_not_found():
    proxy = ProxyToolRegistry(child(AddTool()))  # route never warmed
    execution = ToolExecution(
        id="te1", created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        status=ExecutionStatus.PENDING,
    )

    with pytest.raises(ToolNotFound):
        await proxy.execute(
            execution, CONTEXT, cancellation_token=CancellationToken(),
        )


async def test_nested_proxies_route_transparently():
    inner = ProxyToolRegistry(child(AddTool()))
    outer = ProxyToolRegistry(inner, child(MultiplyTool()))

    assert [tool.name for tool in outer.get_tools(SESSION)] == ["add", "multiply"]
    execution = stamped(await outer.create_execution(
        ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}), CONTEXT,
    ))
    result = await outer.execute(
        execution, CONTEXT, cancellation_token=CancellationToken(),
    )

    assert result == ExecutionResult(content=[TextContent(text="3")])
