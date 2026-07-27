"""McpToolRegistry — the ToolRegistry that exposes live MCP tools.

`get_tools` reads the manager's current tool list (dynamic, so a server that
comes up or adds tools is picked up on the next LLM call). Execution forwards to
the server and maps the result to an `ExecutionResult`. Approval delegates to
the shared `PermissionStrategy`, so MCP tools pass the same gate as every other
tool; each is `ToolKind.OTHER` (the framework cannot know its behavior).
"""

from __future__ import annotations

from mcp import types

from luca.agent.core import (
    ApprovalDecision,
    ExecutionResult,
    ExecutionStatus,
    ImageBase64,
    ImageContent,
    TextContent,
    Tool,
    ToolContext,
    ToolExecution,
    ToolExecutionError,
    ToolKind,
    ToolNotFound,
    ToolRegistry,
    ToolSpec,
)
from luca.agent.core.context import CancellationToken

from .manager import McpManager
from .tool import McpTool


def _draft(call, *, tool_spec, status, error=None) -> ToolExecution:
    return ToolExecution(
        id="",
        created_at=0,
        tool_call_id=call.id,
        raw_tool_call=call,
        tool_spec=tool_spec,
        status=status,
        error=error,
    )


def _fallback_text(block) -> str:
    resource = getattr(block, "resource", None)
    text = getattr(resource, "text", None)
    if text is not None:
        return text
    return f"[unsupported MCP content: {getattr(block, 'type', 'unknown')}]"


def to_execution_result(result: types.CallToolResult) -> ExecutionResult:
    content: list = []
    for block in result.content:
        if isinstance(block, types.TextContent):
            content.append(TextContent(text=block.text))
        elif isinstance(block, types.ImageContent):
            content.append(
                ImageContent(
                    source=ImageBase64(data=block.data, media_type=block.mimeType),
                )
            )
        else:  # embedded resource / audio / other → best-effort text
            content.append(TextContent(text=_fallback_text(block)))
    if not content:
        content = [TextContent(text="")]
    return ExecutionResult(content=content, is_error=bool(result.isError))


class McpToolRegistry(ToolRegistry):
    def __init__(self, manager: McpManager, permission_policy) -> None:
        self._manager = manager
        self._policy = permission_policy

    def _names(self) -> set[str]:
        return {namespaced for namespaced, _ in self._manager.list_tools()}

    def get_tools(self, agent_session) -> list[Tool]:
        return [
            McpTool(
                name=namespaced,
                description=tool.description,
                input_schema=tool.inputSchema,
            )
            for namespaced, tool in self._manager.list_tools()
        ]

    async def create_execution(self, call, context: ToolContext) -> ToolExecution:
        if call.name not in self._names():
            return _draft(
                call,
                tool_spec=None,
                status=ExecutionStatus.NOT_FOUND,
                error=ToolExecutionError(
                    error_type="ToolNotFound",
                    error_message=f"Unknown MCP tool: {call.name!r}.",
                ),
            )
        spec = ToolSpec(name=call.name, description="", tool_kind=ToolKind.OTHER)
        return _draft(call, tool_spec=spec, status=ExecutionStatus.PENDING)

    async def decide(self, tool_execution: ToolExecution, context: ToolContext) -> ApprovalDecision:
        return await self._policy.decide(tool_execution)

    async def execute(
        self,
        tool_execution: ToolExecution,
        context: ToolContext,
        *,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        name = tool_execution.raw_tool_call.name
        try:
            result = await self._manager.call_tool(
                name,
                tool_execution.raw_tool_call.arguments,
            )
        except KeyError as exc:
            raise ToolNotFound(f"Unknown MCP tool: {name!r}.") from exc
        return to_execution_result(result)
