"""McpToolRegistry — MCP tools under the framework's tool contract.

A stateless view over `McpService`, holding two immutable references, so a
`/clear` may throw one away and build another (contract rule 13a).

How the four methods meet the contract:

- `get_tools` and `create_execution` read the durable catalog. No awaits that
  touch the network; the refresh runs in the service's own loop, which is rule
  3's "refreshed out of band".
- the approval context is written at birth into `extras`, which is what lets a
  cold resume decide identically with no listing.
- `decide` hands to the shared `PermissionStrategy`.
- `prepare` resolves from `tool_execution.tool_spec.metadata` first and the
  catalog only as a fallback, so a call approved in a previous process
  dispatches with nothing listed. All network work is inside the returned
  callable, which holds no lock (rule 2).

Arguments are NOT validated here. Rule 11 forbids it for exactly this case: the
remote server validates on its own side, and a bad argument comes back as its
own `-32602`.
"""

from __future__ import annotations

import logging

from luca.agent.contrib.simple_tool_registry.permissions import PermissionPolicy
from luca.agent.core import (
    AgentSession,
    ApprovalDecision,
    CancellationToken,
    ExecutionResult,
    ExecutionStatus,
    PreparedTool,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    ToolNotFound,
    ToolRegistry,
    ToolSpec,
)

from .mapping import approval_context, spec_identity
from .service import McpService

logger = logging.getLogger(__name__)


class McpToolRegistry(ToolRegistry):
    def __init__(self, service: McpService, permission_policy: PermissionPolicy) -> None:
        self._service = service
        self._policy = permission_policy

    async def get_tools(self, session: AgentSession, conversation_id: str) -> list[ToolSpec]:
        return self._service.specs()

    async def create_execution(
        self,
        session: AgentSession,
        conversation_id: str,
        call: ToolCall,
    ) -> ToolExecution:
        spec = self._service.spec(call.name)
        if spec is None:
            return ToolExecution(
                tool_call_id=call.id,
                raw_tool_call=call,
                tool_spec=None,
                status=ExecutionStatus.NOT_FOUND,
                error=ToolExecutionError(
                    error_type="ToolNotFound",
                    error_message=f"Unknown MCP tool: {call.name!r}.",
                ),
            )
        identity = spec_identity(spec)
        return ToolExecution(
            tool_call_id=call.id,
            raw_tool_call=call,
            tool_spec=spec,
            status=ExecutionStatus.PENDING,
            extras={"approval_context": approval_context(*identity)} if identity else {},
        )

    async def decide(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> ApprovalDecision:
        return await self._policy.decide(session, tool_execution)

    async def prepare(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> PreparedTool:
        spec = tool_execution.tool_spec or self._service.spec(tool_execution.raw_tool_call.name)
        identity = spec_identity(spec) if spec is not None else None
        if identity is None:
            raise ToolNotFound(f"Unknown MCP tool: {tool_execution.raw_tool_call.name!r}.")
        label, tool = identity
        # Captured now: the callable receives only the token (rule 5), and the
        # execution is a snapshot already stale by the time the body runs.
        arguments = dict(tool_execution.raw_tool_call.arguments)
        input_schema = dict(spec.input_schema) if spec.input_schema else None
        timeout_ms = spec.timeout_in_ms

        async def run(*, cancellation_token: CancellationToken) -> ExecutionResult:
            # Raises rather than returning `is_error=True`, and MCP draws the
            # same line: a tool that failed at its job answers `isError` and
            # COMPLETES, while a JSON-RPC error or a dead connection is the CALL
            # failing, which is the framework's FAILED.
            return await self._service.call(
                label,
                tool,
                arguments,
                input_schema=input_schema,
                timeout_ms=timeout_ms,
            )

        return run
