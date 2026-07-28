"""McpToolRegistry — exposes external MCP tools, connecting per call.

`get_tools` lists each configured server's tools once and caches the specs,
namespaced `label__tool`. Execution opens a fresh session inside the prepared
callable, calls the tool, and closes, so there is no long-lived connection and
no lifecycle for anyone to own. Approval delegates to the shared permission
policy, so MCP tools pass the same gate as every other tool; each is
`ToolKind.OTHER` (the framework cannot know an external tool's behavior) and
declares no resources.
"""

from __future__ import annotations

import asyncio

from mcp import types

from luca.agent.core import (
    AgentSession,
    ApprovalDecision,
    CancellationToken,
    ExecutionResult,
    ExecutionStatus,
    ImageBase64,
    ImageContent,
    PreparedTool,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    ToolKind,
    ToolNotFound,
    ToolRegistry,
    ToolSpec,
)

from . import session as mcp_session
from .config import HttpServer, McpServerDef

# Separator between a server label and a tool name on the wire.
SEP = "__"


def _to_tool_spec(namespaced: str, tool: types.Tool) -> ToolSpec:
    """The server's tool as a `ToolSpec`. `inputSchema` is a JSON Schema dict,
    which is exactly what `ToolSpec.input_schema` wants (the empty object schema
    for a no-argument tool)."""
    return ToolSpec(
        name=namespaced,
        description=tool.description or "",
        input_schema=tool.inputSchema or {"type": "object", "properties": {}},
        tool_kind=ToolKind.OTHER,
    )


def _draft(
    call: ToolCall,
    *,
    tool_spec: ToolSpec | None,
    status: ExecutionStatus,
    error: ToolExecutionError | None = None,
) -> ToolExecution:
    """A birth draft with no identity — the runner stamps `id`/`created_at`."""
    return ToolExecution(
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
    def __init__(
        self,
        servers: dict[str, McpServerDef],
        permission_policy,
        *,
        auth_factory=None,
    ) -> None:
        for label in servers:
            if SEP in label:
                raise ValueError(f"MCP server label {label!r} may not contain {SEP!r}")
        self._servers = servers
        self._policy = permission_policy
        self._auth_factory = auth_factory  # (label, HttpServer) -> httpx.Auth | None
        self._specs: dict[str, ToolSpec] | None = None  # cached once by _ensure_listed
        self._lock = asyncio.Lock()  # serialize listing (a startup warm-up may race the first turn)
        self._connected: set[str] = set()  # servers that listed successfully
        self.failures: dict[str, str] = {}  # label -> error, surfaced by the TUI notice

    @property
    def connected_labels(self) -> list[str]:
        return sorted(self._connected)

    def _auth_for(self, label: str, server: McpServerDef):
        if isinstance(server, HttpServer) and server.oauth and self._auth_factory is not None:
            return self._auth_factory(label, server)
        return None

    async def _ensure_listed(self) -> dict[str, ToolSpec]:
        """List every server's tools once, concurrently, and cache the specs. A
        server that fails to list contributes no tools and is recorded in
        `failures` (the TUI surfaces it), never raising (a raise in `get_tools`
        aborts the run). The lock keeps a startup warm-up from racing the first
        turn into a double listing (and a double OAuth prompt)."""
        async with self._lock:
            if self._specs is None:
                specs: dict[str, ToolSpec] = {}

                async def _list(label: str, server: McpServerDef) -> None:
                    auth = self._auth_for(label, server)
                    try:
                        tools = await mcp_session.list_tools(server, auth=auth)
                    except Exception as exc:  # any connect/list failure → skip, record
                        self.failures[label] = str(exc)
                        return
                    self._connected.add(label)
                    for tool in tools:
                        namespaced = f"{label}{SEP}{tool.name}"
                        specs[namespaced] = _to_tool_spec(namespaced, tool)

                await asyncio.gather(*(_list(label, server) for label, server in self._servers.items()))
                self._specs = specs
        return self._specs

    async def get_tools(self, session: AgentSession) -> list[ToolSpec]:
        return list((await self._ensure_listed()).values())

    async def create_execution(self, session: AgentSession, call: ToolCall) -> ToolExecution:
        specs = await self._ensure_listed()
        spec = specs.get(call.name)
        if spec is None:
            return _draft(
                call,
                tool_spec=None,
                status=ExecutionStatus.NOT_FOUND,
                error=ToolExecutionError(
                    error_type="ToolNotFound",
                    error_message=f"Unknown MCP tool: {call.name!r}.",
                ),
            )
        return _draft(call, tool_spec=spec, status=ExecutionStatus.PENDING)

    async def decide(self, session: AgentSession, tool_execution: ToolExecution) -> ApprovalDecision:
        return await self._policy.decide(session, tool_execution)

    async def prepare(self, session: AgentSession, tool_execution: ToolExecution) -> PreparedTool:
        # resolve by the label prefix, which is cache-independent (so a cold
        # resume still routes); when a listing is cached, use it for a precise
        # NOT_FOUND on a known server's unknown tool
        name = tool_execution.raw_tool_call.name
        label, sep, tool_name = name.partition(SEP)
        server = self._servers.get(label)
        if not sep or server is None or (self._specs is not None and name not in self._specs):
            raise ToolNotFound(f"Unknown MCP tool: {name!r}.")
        arguments = tool_execution.raw_tool_call.arguments
        auth = self._auth_for(label, server)

        async def run(*, cancellation_token: CancellationToken) -> ExecutionResult:
            # network work belongs in the callable (contract rule 3); the fresh
            # session is opened and closed in this task, so the stdio subprocess
            # is torn down even if the call is cancelled mid-flight
            result = await mcp_session.call_tool(server, tool_name, arguments, auth=auth)
            return to_execution_result(result)

        return run
