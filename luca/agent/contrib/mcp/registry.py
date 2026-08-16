"""McpToolRegistry — MCP tools under the framework's tool contract.

A stateless view over `McpService`. It holds two immutable references and
nothing else, which is what makes it safe for `_reset_session` to throw one
away and build another on every `/clear`, and what satisfies contract rule 13a:
there is no per-call state on `self` to be read back after an await and belong
to a different conversation by then.

How the four methods meet the contract:

- `get_tools` is a dict read of the durable catalog. No awaits that touch the
  network, ever. The refresh that keeps it current runs in the service's own
  loop, which is rule 3's "refreshed out of band" in the literal sense.
- `create_execution` is the same read plus an approval context, built from the
  existing `(permission, resource)` vocabulary so `permissions.rules` already
  expresses `{"permission": "mcp", "resource": "github/*"}` with no new schema.
  Writing it at birth into `extras` is what lets a cold resume decide
  identically with no listing, which the contract's composing-registries
  paragraph requires.
- `decide` hands straight to the shared `PermissionStrategy`, so an MCP tool
  passes the same gate as a shell tool and one "always allow" answer works the
  same way for both.
- `prepare` resolves locally and returns a closure. It resolves from
  `tool_execution.tool_spec.metadata` first and the catalog only as a fallback,
  so a call that was approved in a previous process dispatches even if nothing
  has been listed yet in this one. Every byte of network work is inside the
  returned callable, which holds no lock when it is handed back (rule 2).

WHAT IS NOT VALIDATED HERE: the arguments. Rule 11 says the core never
validates them and explains why in exactly our case — a registry delegating to
a remote server that validates on its own side would break under double
validation. A bad argument comes back as the server's own `-32602`.
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
        # Captured now, because the callable receives only the token (rule 5),
        # and because the execution it came from is a snapshot that is already
        # stale by the time the body runs.
        arguments = dict(tool_execution.raw_tool_call.arguments)
        input_schema = dict(spec.input_schema) if spec.input_schema else None
        timeout_ms = spec.timeout_in_ms

        async def run(*, cancellation_token: CancellationToken) -> ExecutionResult:
            # Raises deliberately rather than returning `is_error=True`. The two
            # mean different things here and MCP draws the same line: a tool
            # that failed at its job answers `isError` and COMPLETES, while a
            # JSON-RPC error or a dead connection is the CALL failing, which is
            # the framework's FAILED. Letting it propagate also puts the
            # traceback in the runner's own log with the conversation id, which
            # is where every other tool failure is already looked for.
            return await self._service.call(
                label,
                tool,
                arguments,
                input_schema=input_schema,
                timeout_ms=timeout_ms,
            )

        return run
