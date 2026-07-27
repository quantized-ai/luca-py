"""McpTool projection through the adapter, and the CallToolResult → ExecutionResult
mapping. Pure, no server."""

from mcp import types

from luca.agent.contrib.mcp.registry import to_execution_result
from luca.agent.contrib.mcp.tool import McpTool
from luca.agent.core import ExecutionResult, ImageBase64, ImageContent, TextContent, ToolKind, ToolSpec
from luca.agent.core.adapter import tool_to_luca_tool
from luca.client.types import Tool as WireTool

_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


def test_an_mcp_tool_carries_the_server_schema_and_is_kind_other():
    tool = McpTool(name="s__echo", description="Echo it", input_schema=_SCHEMA)
    assert tool.input_schema == _SCHEMA
    assert tool.get_tool_spec() == ToolSpec(name="s__echo", description="Echo it", tool_kind=ToolKind.OTHER)


def test_the_adapter_passes_the_raw_schema_straight_to_the_wire_tool():
    tool = McpTool(name="s__echo", description="Echo it", input_schema=_SCHEMA)
    assert tool_to_luca_tool(tool) == WireTool(
        name="s__echo",
        description="Echo it",
        parameters=_SCHEMA,
    )


def test_a_text_result_maps_to_text_content():
    result = types.CallToolResult(content=[types.TextContent(type="text", text="hi")])
    assert to_execution_result(result) == ExecutionResult(content=[TextContent(text="hi")], is_error=False)


def test_an_image_result_maps_to_base64_image_content():
    result = types.CallToolResult(
        content=[types.ImageContent(type="image", data="AAAA", mimeType="image/png")],
    )
    assert to_execution_result(result) == ExecutionResult(
        content=[ImageContent(source=ImageBase64(data="AAAA", media_type="image/png"))],
        is_error=False,
    )


def test_is_error_flows_through():
    result = types.CallToolResult(
        content=[types.TextContent(type="text", text="bad")],
        isError=True,
    )
    assert to_execution_result(result) == ExecutionResult(content=[TextContent(text="bad")], is_error=True)


def test_empty_content_becomes_a_single_empty_text_block():
    assert to_execution_result(types.CallToolResult(content=[])) == ExecutionResult(
        content=[TextContent(text="")],
        is_error=False,
    )
