"""The per-call registry against a real stdio server."""

import asyncio
import contextlib
import pathlib
import sys

import pytest

from luca.agent.contrib.mcp.config import StdioServer
from luca.agent.contrib.mcp.registry import McpToolRegistry
from luca.agent.contrib.simple_tool_registry import YoloPermissionPolicy
from luca.agent.core import (
    AgentSession,
    ApprovalOption,
    CancellationToken,
    Conversation,
    ExecutionResult,
    ExecutionStatus,
    LLMConfig,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolKind,
    ToolNotFound,
)

_FIXTURE = pathlib.Path(__file__).parent / "mcp_server_fixture.py"
_SERVER = StdioServer(command=sys.executable, args=[str(_FIXTURE)])

SESSION = AgentSession(
    id="s_mcp",
    active_conversation=Conversation(id="c1", nodes=[], created_at=500, updated_at=500),
    session_config=SessionConfig(llm_config=LLMConfig(model="m", provider="faux")),
)


def _registry(policy=None) -> McpToolRegistry:
    return McpToolRegistry({"t": _SERVER}, policy or YoloPermissionPolicy())


def _execution(name: str, arguments: dict) -> ToolExecution:
    return ToolExecution(
        tool_call_id="c",
        raw_tool_call=ToolCall(id="c", name=name, arguments=arguments),
        status=ExecutionStatus.PENDING,
    )


async def test_get_tools_lists_namespaced_specs_that_carry_the_server_schema():
    registry = _registry()
    await registry.wait_listed()  # get_tools is non-blocking; the listing is out of band
    specs = {spec.name: spec for spec in await registry.get_tools(SESSION)}
    assert set(specs) == {"t__echo", "t__add", "t__slow"}
    echo = specs["t__echo"]
    assert echo.tool_kind is ToolKind.OTHER
    # partial on input_schema: it is FastMCP-generated JSON Schema this test does
    # not own, so a full-object ToolSpec assertion would couple to its output
    assert echo.input_schema["properties"]["text"]["type"] == "string"


async def test_a_known_tool_births_pending_with_the_server_schema():
    registry = _registry()
    await registry.wait_listed()
    draft = await registry.create_execution(
        SESSION,
        ToolCall(id="c", name="t__add", arguments={"a": 1, "b": 2}),
    )
    # partial on tool_spec for the same reason: the input_schema is the server's
    # own JSON Schema, which this test does not own
    assert draft.status is ExecutionStatus.PENDING
    assert draft.tool_spec.name == "t__add"
    assert draft.tool_spec.tool_kind is ToolKind.OTHER
    assert draft.tool_spec.input_schema["properties"]["a"]["type"] == "integer"


async def test_an_unknown_tool_births_not_found():
    registry = _registry()
    await registry.wait_listed()
    draft = await registry.create_execution(SESSION, ToolCall(id="c", name="t__missing", arguments={}))
    assert draft.status is ExecutionStatus.NOT_FOUND
    assert draft.tool_spec is None


async def test_prepare_then_run_forwards_to_the_server_and_maps_the_result():
    prepared = await _registry().prepare(SESSION, _execution("t__echo", {"text": "hi"}))
    result = await prepared(cancellation_token=CancellationToken())
    assert result == ExecutionResult(content=[TextContent(text="echo: hi")], is_error=False)


async def test_prepare_for_an_unknown_server_raises_tool_not_found():
    with pytest.raises(ToolNotFound):
        await _registry().prepare(SESSION, _execution("gone__tool", {}))


async def test_yolo_allows_and_ask_defers_to_the_user():
    from luca.agent.contrib.resource_permissions import PermissionMode, PermissionStrategy

    execution = _execution("t__echo", {"text": "x"})
    allowed = await _registry(YoloPermissionPolicy()).decide(SESSION, execution)
    assert allowed.decision is ApprovalOption.ALLOW

    ask = McpToolRegistry({"t": _SERVER}, PermissionStrategy(mode=PermissionMode.ASK))
    deferred = await ask.decide(SESSION, execution)
    assert deferred.decision is ApprovalOption.PENDING


async def test_a_server_that_fails_to_list_is_skipped_and_recorded():
    registry = McpToolRegistry({"bad": StdioServer(command="luca-no-such-command-xyz")}, YoloPermissionPolicy())
    await registry.wait_listed()
    assert await registry.get_tools(SESSION) == []
    assert registry.connected_labels == []
    assert "bad" in registry.failures


async def test_a_cancelled_call_tears_down_the_subprocess():
    prepared = await _registry().prepare(SESSION, _execution("t__slow", {"seconds": 30}))
    task = asyncio.ensure_future(prepared(cancellation_token=CancellationToken()))
    task.cancel()
    # the ResourceWarning-as-error rule fails the test if the stdio subprocess leaks
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


def test_a_server_label_containing_the_separator_is_rejected():
    with pytest.raises(ValueError, match="may not contain"):
        McpToolRegistry({"a__b": _SERVER}, YoloPermissionPolicy())
