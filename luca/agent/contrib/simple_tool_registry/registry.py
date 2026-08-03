"""SimpleToolRegistry + ProxyToolRegistry — the batteries-included registries.

`SimpleToolRegistry` is the straightforward `ToolRegistry`: a static list of
contrib `Tool`s gated by one `PermissionPolicy` (`permissions.py`). Its
`create_execution` reproduces the classic preflight — resolve the tool,
validate the arguments, collect the duck-typed approval context — and its
`prepare` resolves and validates once more, then returns a closure binding the
validated arguments and the session. Each phase validates once, for a
different reason (birth status vs. dispatch), and neither result is thrown
away and re-derived inside the same phase.

The approval context is an implementation convention, not a core contract:
if a tool defines `get_approval_context(args, session, conversation_id) -> dict`
(see `resource_permissions.ResourcePermissionToolMixin`), the registry awaits
it with the VALIDATED arguments and stores the result under
`extras["approval_context"]` for the policy/application to read; a raise is
caught and turns the birth FAILED (per-call isolation), with the registry's
own `details={"phase": "approval_context"}`.

`ProxyToolRegistry` composes registries: `get_tools` concatenates the
children's specs (duplicate names raise), rebuilding an internal
`{name → child}` routing cache FOR THAT CONVERSATION. `decide` and `prepare`
resolve INDEPENDENTLY of that cache — on a miss they warm it once from the
children and try again — so a call left pending approval by a previous process
resolves on a fresh one, is gated by its owning child, and dispatches. That
matters for correctness, not just convenience: `prepare()` can now resolve a
name the cache never saw, so a blanket ALLOW on a cache miss would be a
permission bypass on a cold resume. What stays is ALLOW on a name that is
genuinely unresolvable after cache-independent resolution — `prepare()` raises
`ToolNotFound` for it and the call records NOT_FOUND, which is the honest
outcome; DENY there would record REJECTED for a tool that never existed.
Resolution must be the same question in `decide` and `prepare`; only then is
one answer safe.

`create_execution` still routes from the plain cache: a tool call only ever
arrives after an LLM call, which warmed that conversation's route on its way
out. There is no ordering contract beyond child order, and nesting needs
nothing special — a child proxy routes internally the same way.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from luca.agent.contrib.tools import Tool
from luca.agent.core import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    CancellationToken,
    ExecutionResult,
    ExecutionStatus,
    InvalidToolArguments,
    PreparedTool,
    ToolCall,
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
    """A birth draft with no identity — the runner stamps `id`,
    `parent_id`, `created_at`, and `ended_at` (for a terminal birth), and the
    ledger files the spec and stamps `tool_spec_id`."""
    return ToolExecution(
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
        self,
        tools: list[Tool],
        permission_policy: PermissionPolicy,
    ) -> None:
        self.tools = list(tools)
        self.permission_policy = permission_policy
        self.tools_by_name = {tool.name: tool for tool in self.tools}

    async def get_tools(
        self,
        session: AgentSession,
        conversation_id: str,
    ) -> list[ToolSpec]:
        return [tool.get_tool_spec() for tool in self.tools]

    async def create_execution(
        self,
        session: AgentSession,
        conversation_id: str,
        call: ToolCall,
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
                    error_message=(f"Arguments for tool {call.name!r} are invalid."),
                    details={
                        "errors": json.loads(exc.json(include_url=False)),
                    },
                ),
            )
        extras: dict = {}
        if hasattr(tool, "get_approval_context"):
            try:
                extras["approval_context"] = await tool.get_approval_context(
                    args.model_dump(),
                    session,
                    conversation_id,
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
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> ApprovalDecision:
        """Delegate to the policy. The policy is NOT handed `conversation_id`
        separately: `tool_execution.conversation_id` already carries it, which
        is the whole reason that field exists on the one entry type a consumer
        receives detached from a path."""
        return await self.permission_policy.decide(session, tool_execution)

    async def prepare(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> PreparedTool:
        """Resolve, validate, and hand back the closure that runs the body.

        Everything fallible happens here, before the runner writes RUNNING:
        `ToolNotFound` and `InvalidToolArguments` mean the body never ran and
        record with `started_at=None`. The closure binds the validated
        arguments, the session AND the conversation — nothing is re-derived at
        invocation time — and takes only the run's cancellation token. Binding
        the conversation here is what carries it into the tool body without
        adding a parameter to `PreparedTool`."""
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
        payload = args.model_dump()

        async def run(*, cancellation_token: CancellationToken) -> ExecutionResult:
            return await tool.execute(
                payload,
                session,
                conversation_id,
                cancellation_token=cancellation_token,
            )

        return run


class ProxyToolRegistry(ToolRegistry):
    """Composition + routing over child registries. `get_tools` recomputes
    and caches a `{name → child}` route PER CONVERSATION; `decide` and
    `prepare` resolve independently of that cache, warming it once on a miss.

    The route is keyed by conversation because `get_tools` may legitimately
    answer DIFFERENTLY per conversation — a child registry withholding a tool
    from a subagent is the whole point of the contract taking a conversation.
    One shared route would be wrong in both directions, and neither failure is
    a data race (the dict swap is atomic) — both are wrong routing. A route
    warmed by a conversation that was OFFERED a tool would resolve it for a
    conversation the tool was withheld from, dispatching a call the owning
    child deliberately did not advertise; and a route overwritten by a
    conversation that was NOT offered it makes the next call from a permitted
    conversation resolve to `None` and record a silent NOT_FOUND."""

    def __init__(self, *registries: ToolRegistry) -> None:
        self.registries = list(registries)
        self._routes: dict[str, dict[str, ToolRegistry]] = {}
        self._warmed: set[str] = set()

    def add_registry(self, registry: ToolRegistry) -> None:
        self.registries.append(registry)
        self._routes.clear()
        self._warmed.clear()

    async def get_tools(
        self,
        session: AgentSession,
        conversation_id: str,
    ) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        route: dict[str, ToolRegistry] = {}
        for registry in self.registries:
            for spec in await registry.get_tools(session, conversation_id):
                if spec.name in route:
                    raise ValueError(f"Duplicate tool name across registries: {spec.name!r}.")
                route[spec.name] = registry
                specs.append(spec)
        self._routes[conversation_id] = route
        self._warmed.add(conversation_id)
        return specs

    async def _resolve(
        self,
        session: AgentSession,
        conversation_id: str,
        name: str,
    ) -> ToolRegistry | None:
        """The child owning `name` for this conversation, or None. On a cold
        route it asks the children once and memoizes — bounded, cached after
        the first hit, and raced against the cancellation token by the runner.
        I/O inside `prepare()` is discouraged by the contract; this is the
        price of `ToolRegistry` having no resolution method, and the
        alternative (a blanket ALLOW on a route the proxy never warmed) is a
        permission bypass."""
        child = self._routes.get(conversation_id, {}).get(name)
        if child is not None:
            return child
        if conversation_id not in self._warmed:
            await self.get_tools(session, conversation_id)
            child = self._routes.get(conversation_id, {}).get(name)
        return child

    async def create_execution(
        self,
        session: AgentSession,
        conversation_id: str,
        call: ToolCall,
    ) -> ToolExecution:
        child = self._routes.get(conversation_id, {}).get(call.name)
        if child is None:
            return _not_found_draft(call)
        return await child.create_execution(session, conversation_id, call)

    async def decide(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> ApprovalDecision:
        child = await self._resolve(
            session,
            conversation_id,
            tool_execution.raw_tool_call.name,
        )
        if child is None:
            # Genuinely unresolvable: prepare() raises ToolNotFound and the
            # call records the honest NOT_FOUND rather than a false REJECTED.
            return ApprovalDecision(
                decision=ApprovalOption.ALLOW,
                metadata={"via": "unrouted"},
            )
        return await child.decide(session, conversation_id, tool_execution)

    async def prepare(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> PreparedTool:
        name = tool_execution.raw_tool_call.name
        child = await self._resolve(session, conversation_id, name)
        if child is None:
            raise ToolNotFound(f"Unknown tool: {name!r}.")
        return await child.prepare(session, conversation_id, tool_execution)
