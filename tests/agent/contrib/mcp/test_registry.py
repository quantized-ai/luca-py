"""The manager, connection actor, and registry against a real stdio server."""

import asyncio
import pathlib
import sys

import pytest

from luca.agent.contrib.mcp.config import StdioServer
from luca.agent.contrib.mcp.connection import McpConnection
from luca.agent.contrib.mcp.manager import McpManager
from luca.agent.contrib.mcp.registry import McpToolRegistry
from luca.agent.contrib.simple_tool_registry import YoloPermissionPolicy
from luca.agent.core import (
    ApprovalOption,
    ExecutionResult,
    ExecutionStatus,
    LLMConfig,
    TextContent,
    ToolCall,
    ToolContext,
    ToolExecution,
    ToolKind,
    ToolNotFound,
    ToolSpec,
)
from luca.agent.core.context import CancellationToken

_FIXTURE = pathlib.Path(__file__).parent / "mcp_server_fixture.py"
_CONTEXT = ToolContext(session_id="s", model=LLMConfig(model="m", provider="p"))


@pytest.fixture
async def manager():
    mgr = McpManager({"t": StdioServer(command=sys.executable, args=[str(_FIXTURE)])})
    await mgr.start_all()
    yield mgr
    await mgr.aclose()


def _call(name, arguments):
    return ToolExecution(
        id="e",
        created_at=0,
        tool_call_id="c",
        raw_tool_call=ToolCall(id="c", name=name, arguments=arguments),
        status=ExecutionStatus.PENDING,
    )


async def test_the_manager_lists_tools_namespaced_by_server(manager):
    assert sorted(name for name, _ in manager.list_tools()) == ["t__add", "t__echo"]


async def test_get_tools_returns_mcp_tools_that_carry_the_server_schema(manager):
    registry = McpToolRegistry(manager, YoloPermissionPolicy())
    tools = {tool.name: tool for tool in registry.get_tools(None)}
    assert set(tools) == {"t__echo", "t__add"}
    assert tools["t__echo"].input_schema["properties"]["text"]["type"] == "string"


async def test_execute_forwards_to_the_server_and_maps_the_result(manager):
    registry = McpToolRegistry(manager, YoloPermissionPolicy())
    result = await registry.execute(
        _call("t__echo", {"text": "hi"}),
        _CONTEXT,
        cancellation_token=CancellationToken(),
    )
    assert result == ExecutionResult(content=[TextContent(text="echo: hi")], is_error=False)


async def test_a_known_tool_births_pending_as_kind_other(manager):
    registry = McpToolRegistry(manager, YoloPermissionPolicy())
    draft = await registry.create_execution(
        ToolCall(id="c", name="t__add", arguments={"a": 1, "b": 2}),
        _CONTEXT,
    )
    assert draft.status is ExecutionStatus.PENDING
    assert draft.tool_spec == ToolSpec(name="t__add", description="", tool_kind=ToolKind.OTHER)


async def test_an_unknown_tool_births_not_found(manager):
    registry = McpToolRegistry(manager, YoloPermissionPolicy())
    draft = await registry.create_execution(
        ToolCall(id="c", name="t__missing", arguments={}),
        _CONTEXT,
    )
    assert draft.status is ExecutionStatus.NOT_FOUND


async def test_execute_routed_to_an_unknown_server_raises_tool_not_found(manager):
    # an unknown server label has no connection to route to
    registry = McpToolRegistry(manager, YoloPermissionPolicy())
    with pytest.raises(ToolNotFound):
        await registry.execute(
            _call("gone__tool", {}),
            _CONTEXT,
            cancellation_token=CancellationToken(),
        )


async def test_yolo_allows_and_ask_defers_to_the_user(manager):
    from luca.agent.contrib.resource_permissions import PermissionMode, PermissionStrategy

    call = _call("t__echo", {"text": "x"})
    allowed = await McpToolRegistry(manager, YoloPermissionPolicy()).decide(call, _CONTEXT)
    assert allowed.decision is ApprovalOption.ALLOW

    ask = McpToolRegistry(manager, PermissionStrategy(mode=PermissionMode.ASK))
    deferred = await ask.decide(call, _CONTEXT)
    assert deferred.decision is ApprovalOption.PENDING


async def test_a_server_that_fails_to_start_yields_no_tools_and_is_recorded():
    mgr = McpManager({"bad": StdioServer(command="luca-no-such-command-xyz")})
    await mgr.start_all()
    assert mgr.connected_labels == []
    assert "bad" in mgr.failures
    assert mgr.list_tools() == []
    await mgr.aclose()


async def test_a_call_after_close_raises_promptly_instead_of_hanging():
    conn = McpConnection("t", StdioServer(command=sys.executable, args=[str(_FIXTURE)]))
    await conn.start()
    assert conn.is_connected
    await conn.aclose()
    assert not conn.is_connected
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(conn.call_tool("echo", {"text": "x"}), timeout=3)


def test_a_server_label_containing_the_separator_is_rejected():
    with pytest.raises(ValueError, match="may not contain"):
        McpManager({"a__b": StdioServer(command="x")})
