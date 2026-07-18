"""SimpleToolRegistry + ProxyToolRegistry — the batteries-included registries.

`SimpleToolRegistry` is the straightforward `ToolRegistry`: a static tool
list gated by one `PermissionPolicy` (`permissions.py`). Its
`create_execution` reproduces the classic preflight — resolve the tool,
validate the arguments, collect the duck-typed approval context — and its
`execute` re-resolves and re-validates before invoking the tool body.

The approval context is an implementation convention, not a core contract:
if a tool defines `get_approval_context(args, context) -> dict` (see
`resource_permissions.ResourcePermissionToolMixin`), the registry awaits it
with the VALIDATED arguments and stores the result under
`extras["approval_context"]` for the policy/application to read; a raise is
caught and turns the birth FAILED (per-call isolation).

`ProxyToolRegistry` composes registries: `get_tools` concatenates the
children's tools (duplicate names raise), rebuilding an internal
`{name → child}` routing cache; the other three methods route through that
cache and degrade on a miss — `create_execution` authors a NOT_FOUND draft,
`decide` allows (so `execute` produces the honest NOT_FOUND terminal rather
than a false REJECTED), and `execute` raises `ToolNotFound`. There is no
ordering contract beyond child order, and nesting needs nothing special — a
child proxy routes internally the same way.

Accepted degradation: on a cold cross-process resume of a gated session, the
cache has never been warmed (no LLM call happened yet), so pending calls
terminalize as NOT_FOUND instead of re-asking. Predictable, documented, for
now.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from luca.agent.core import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    CancellationToken,
    ExecutionResult,
    ExecutionStatus,
    InvalidToolArguments,
    Tool,
    ToolCall,
    ToolContext,
    ToolExecution,
    ToolExecutionError,
    ToolNotFound,
    ToolRegistry,
    ToolSpec,
)

from .permissions import PermissionPolicy


def _draft(
    call: ToolCall,
    *,
    tool_spec: ToolSpec | None,
    status: ExecutionStatus,
    error: ToolExecutionError | None = None,
    extras: dict | None = None,
) -> ToolExecution:
    """A birth draft with placeholder identity — the runner stamps `id`,
    `parent_id`, `created_at`, and `ended_at` (for a terminal birth)."""
    return ToolExecution(
        id="", created_at=0,
        tool_call_id=call.id,
        raw_tool_call=call,
        tool_spec=tool_spec,
        status=status,
        error=error,
        extras=extras or {},
    )


def _not_found_draft(call: ToolCall) -> ToolExecution:
    return _draft(
        call,
        tool_spec=None,
        status=ExecutionStatus.NOT_FOUND,
        error=ToolExecutionError(
            error_type="ToolNotFound",
            error_message=f"Unknown tool: {call.name!r}.",
        ),
    )


class SimpleToolRegistry(ToolRegistry):
    """A static tool list gated by one `PermissionPolicy`."""

    def __init__(
        self, tools: list[Tool], permission_policy: PermissionPolicy,
    ) -> None:
        self.tools = list(tools)
        self.permission_policy = permission_policy
        self.tools_by_name = {tool.name: tool for tool in self.tools}

    def get_tools(self, agent_session: AgentSession) -> list[Tool]:
        return list(self.tools)

    async def create_execution(
        self, call: ToolCall, context: ToolContext,
    ) -> ToolExecution:
        tool = self.tools_by_name.get(call.name)
        if tool is None:
            return _not_found_draft(call)
        tool_spec = tool.get_tool_spec()
        try:
            args = tool.Args.model_validate(call.arguments)
        except ValidationError as exc:
            return _draft(
                call,
                tool_spec=tool_spec,
                status=ExecutionStatus.INVALID,
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
                return _draft(
                    call,
                    tool_spec=tool_spec,
                    status=ExecutionStatus.FAILED,
                    error=ToolExecutionError(
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        details={"phase": "approval_context"},
                    ),
                )
        return _draft(
            call,
            tool_spec=tool_spec,
            status=ExecutionStatus.PENDING,
            extras=extras,
        )

    async def decide(
        self, tool_execution: ToolExecution, context: ToolContext,
    ) -> ApprovalDecision:
        return await self.permission_policy.decide(tool_execution)

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
            args = tool.Args.model_validate(tool_execution.raw_tool_call.arguments)
        except ValidationError as exc:
            raise InvalidToolArguments(
                f"Arguments for tool {name!r} are invalid.",
                errors=json.loads(exc.json(include_url=False)),
            ) from exc
        return await tool.execute(
            args.model_dump(), context, cancellation_token=cancellation_token,
        )


class ProxyToolRegistry(ToolRegistry):
    """Composition + routing over child registries. `get_tools` recomputes
    and caches the `{name → child}` route; the other methods route from the
    cache and degrade to NOT_FOUND on a miss."""

    def __init__(self, *registries: ToolRegistry) -> None:
        self.registries = list(registries)
        self._route: dict[str, ToolRegistry] = {}

    def add_registry(self, registry: ToolRegistry) -> None:
        self.registries.append(registry)

    def get_tools(self, agent_session: AgentSession) -> list[Tool]:
        tools: list[Tool] = []
        route: dict[str, ToolRegistry] = {}
        for registry in self.registries:
            for tool in registry.get_tools(agent_session):
                if tool.name in route:
                    raise ValueError(
                        f"Duplicate tool name across registries: {tool.name!r}."
                    )
                route[tool.name] = registry
                tools.append(tool)
        self._route = route
        return tools

    async def create_execution(
        self, call: ToolCall, context: ToolContext,
    ) -> ToolExecution:
        child = self._route.get(call.name)
        if child is None:
            return _not_found_draft(call)
        return await child.create_execution(call, context)

    async def decide(
        self, tool_execution: ToolExecution, context: ToolContext,
    ) -> ApprovalDecision:
        child = self._route.get(tool_execution.raw_tool_call.name)
        if child is None:
            # Allow: execute() then produces the honest terminal (NOT_FOUND)
            # rather than recording a false REJECTED.
            return ApprovalDecision(decision=ApprovalOption.ALLOW)
        return await child.decide(tool_execution, context)

    async def execute(
        self,
        tool_execution: ToolExecution,
        context: ToolContext,
        *,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        name = tool_execution.raw_tool_call.name
        child = self._route.get(name)
        if child is None:
            raise ToolNotFound(f"Unknown tool: {name!r}.")
        return await child.execute(
            tool_execution, context, cancellation_token=cancellation_token,
        )
