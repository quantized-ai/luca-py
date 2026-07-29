"""McpToolRegistry — exposes external MCP tools, connecting per call.

Discovering what an MCP server offers is network I/O, and `get_tools` is async
precisely so a registry fronting a remote tool server can do it (rule 3's
"must not block" binds `prepare()`, not `get_tools`). So `get_tools` WAITS for
the listing rather than answering with whatever is cached at that instant: the
`ProxyToolRegistry` routing table, the permission gate and the model's tool
list all believe that answer, so it has to be true when it is given. The runner
races the whole `build_tool_list` step against the run's cancellation token, so
waiting here can never make `cancel()` a no-op.

`create_execution` waits on the same listing, because the `ToolSpec` it has to
produce is data only the server can supply. `prepare` and `decide` do not:
`prepare` routes on the `<label>__<tool>` prefix against static server config,
and `decide` hands straight to the permission policy. No method assumes another
ran first.

Listing state is tracked PER SERVER — a label is either listed or pending —
so a server that was unreachable is retried on the next ask while its healthy
siblings are left alone, and one dead server never costs the others their
tools. Concurrent callers share one pass through `_lock`; a caller cancelled
mid-pass leaves the servers that already answered listed and the rest pending,
with nothing latched.

Each listing runs under its own deadline (`resolve_listing_timeout_ms`), since
waiting is only safe if the wait is bounded — the cancellation race saves an
attended run, not an unattended one, and this is a library. Expiry is recorded
like any other listing failure and retried the same way; it never aborts the
run.

A tool CALL is bounded separately, by `resolve_call_timeout_ms` stamped onto
each spec as `ToolSpec.timeout_in_ms` and enforced by the runner at dispatch.
It defaults to `None` — inheriting the framework's own default — because the
value of the knob is bounding ONE server's calls without bounding every tool in
the agent, not overriding a core policy.

The network work of a *call* still happens inside the callable `prepare()`
returns. Approval delegates to the shared permission policy, so MCP tools pass
the same gate as every other tool; each is `ToolKind.OTHER` (the framework
cannot know an external tool's behavior) and declares no resources.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import httpx

    from luca.agent.contrib.simple_tool_registry.permissions import PermissionPolicy

# Separator between a server label and a tool name on the wire.
SEP = "__"

# How long one server's listing may take when nothing overrides it.
DEFAULT_LISTING_TIMEOUT_MS = 30_000
# An OAuth server's browser flow runs INSIDE the listing — the provider is an
# httpx auth flow bolted to the transport — and `oauth._AUTH_TIMEOUT` gives the
# human 300s to authorize. This ceiling is deliberately higher so that inner
# wait expires first, with its own message, instead of being cut off by this
# one. Pinned by a test rather than imported, to keep `oauth.py` lazy.
OAUTH_LISTING_TIMEOUT_MS = 330_000
TIMEOUT_ENV_VAR = "LUCA_DEFAULT_MCP_TIMEOUT_MS"
CALL_TIMEOUT_ENV_VAR = "LUCA_DEFAULT_MCP_CALL_TIMEOUT_MS"


def _env_default_ms(env: Mapping[str, str], name: str) -> int | None:
    """One env-var default, or None when it is unset. A malformed value raises
    rather than silently falling back: a timeout that quietly does nothing is
    worse than a startup failure."""
    raw = env.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        raise ValueError(
            f"{name} must be a positive whole number of milliseconds, got {raw!r}.",
        )
    return value


def resolve_listing_timeout_ms(server: McpServerDef, *, env: Mapping[str, str] | None = None) -> int:
    """How long this server's listing may take. Most specific wins:

        server.timeout_in_ms  >  LUCA_DEFAULT_MCP_TIMEOUT_MS  >  the constant

    with one exception: a server whose OAuth browser flow may run takes the
    long ceiling ahead of the env var, because a short global default set for
    some other reason would otherwise make that server impossible to connect.
    Its own `timeout_in_ms` still overrides even that.
    """
    if server.timeout_in_ms is not None:
        return server.timeout_in_ms
    if isinstance(server, HttpServer) and server.oauth:
        return OAUTH_LISTING_TIMEOUT_MS
    return _env_default_ms(os.environ if env is None else env, TIMEOUT_ENV_VAR) or DEFAULT_LISTING_TIMEOUT_MS


def resolve_call_timeout_ms(server: McpServerDef, *, env: Mapping[str, str] | None = None) -> int | None:
    """How long one of this server's TOOL CALLS may take, stamped onto every
    spec as `ToolSpec.timeout_in_ms`:

        server.call_timeout_in_ms  >  LUCA_DEFAULT_MCP_CALL_TIMEOUT_MS  >  None

    `None` deliberately, not a number. The framework already decided that a
    tool body's default deadline is `RuntimeConfig.tool_execution_timeout_in_ms`
    (PRD 6.6), and one contrib package should not quietly override a core
    policy — any figure picked for arbitrary third-party tools would silently
    kill someone's legitimately slow one. What this adds is the ability to bound
    ONE server's calls without bounding every tool in the agent.

    Deliberately has no OAuth exception and no listing default: a call is a
    different unit of work from a listing, and reusing one number would make it
    too tight for one or too generous for the other.

    Note this must stay a pure function of static server config. A `ToolSpec`
    that varies per call mints a fresh `session.tool_specs` row every time and
    silently defeats normalization (PRD section 8).
    """
    if server.call_timeout_in_ms is not None:
        return server.call_timeout_in_ms
    return _env_default_ms(os.environ if env is None else env, CALL_TIMEOUT_ENV_VAR)


def _to_tool_spec(namespaced: str, tool: types.Tool, *, timeout_in_ms: int | None) -> ToolSpec:
    """The server's tool as a `ToolSpec`. `inputSchema` is a JSON Schema dict,
    which is exactly what `ToolSpec.input_schema` wants (the empty object schema
    for a no-argument tool). `timeout_in_ms` is what the runner enforces on the
    body at dispatch; `None` inherits the framework's own default."""
    return ToolSpec(
        name=namespaced,
        description=tool.description or "",
        input_schema=tool.inputSchema or {"type": "object", "properties": {}},
        tool_kind=ToolKind.OTHER,
        timeout_in_ms=timeout_in_ms,
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
        permission_policy: PermissionPolicy,
        *,
        auth_factory: Callable[[str, HttpServer], httpx.Auth | None] | None = None,
    ) -> None:
        for label in servers:
            if SEP in label:
                raise ValueError(f"MCP server label {label!r} may not contain {SEP!r}")
        for var in (TIMEOUT_ENV_VAR, CALL_TIMEOUT_ENV_VAR):
            _env_default_ms(os.environ, var)  # fail fast on a malformed override
        self._servers = servers
        self._policy = permission_policy
        self._auth_factory = auth_factory
        # label -> {namespaced name -> spec}. A label is present iff that server
        # has listed successfully; absent means "still to try", which is what
        # makes a failed listing retryable without disturbing its siblings.
        self._specs: dict[str, dict[str, ToolSpec]] = {}
        self.failures: dict[str, str] = {}  # label -> last error, surfaced by the TUI notice
        self._lock = asyncio.Lock()  # one listing pass at a time; holds no state itself

    @property
    def connected_labels(self) -> list[str]:
        return sorted(self._specs)

    def _auth_for(self, label: str, server: McpServerDef):
        if isinstance(server, HttpServer) and server.oauth and self._auth_factory is not None:
            return self._auth_factory(label, server)
        return None

    def _unlisted(self) -> list[str]:
        """Servers that have not listed successfully yet — everything on a cold
        registry, and afterwards only the ones that failed."""
        return [label for label in self._servers if label not in self._specs]

    async def _list(self, label: str) -> None:
        """List one server, under its own deadline, and publish its slice.

        The deadline bounds the REQUEST for this server — connect, handshake
        and every pagination round trip — because that is the unit a caller
        cares about. It does NOT bound the teardown that follows an expiry: the
        SDK's `stdio_client` takes up to its own `PROCESS_TERMINATION_TIMEOUT`
        (2s) to SIGTERM, wait for, and finally SIGKILL the subprocess, and we
        wait that out rather than orphaning a child. So a 30s bound can cost
        ~32s in the worst case. That trade is deliberate; a leaked subprocess is
        worse than a late return.

        Servers list concurrently, so a slow one never eats another's budget.
        Any failure, expiry included, records the reason and leaves the label
        unlisted so the next ask retries it; the label's siblings are untouched
        either way. `asyncio.CancelledError` is a BaseException and deliberately
        not caught (rule 7).
        """
        server = self._servers[label]
        auth = self._auth_for(label, server)
        timeout_ms = resolve_listing_timeout_ms(server)
        try:
            async with asyncio.timeout(timeout_ms / 1000.0) as scope:
                tools = await mcp_session.list_tools(server, auth=auth)
        except TimeoutError as exc:
            # `asyncio.TimeoutError` IS the builtin, and `str()` of one is
            # empty — so this branch exists to record a readable reason, and
            # `scope.expired()` to tell our deadline from the server's own.
            self.failures[label] = (
                f"listing timed out after {timeout_ms}ms" if scope.expired() else (str(exc) or "TimeoutError")
            )
            return
        except Exception as exc:  # any connect/list failure → retried on the next ask
            self.failures[label] = str(exc) or type(exc).__name__
            return
        self.failures.pop(label, None)
        call_timeout_ms = resolve_call_timeout_ms(server)
        self._specs[label] = {
            f"{label}{SEP}{tool.name}": _to_tool_spec(
                f"{label}{SEP}{tool.name}",
                tool,
                timeout_in_ms=call_timeout_ms,
            )
            for tool in tools
        }

    async def wait_listed(self) -> None:
        """List every server that has not listed yet, once.

        Idempotent and safe to call from anywhere (rule 1's "warming a
        catalog"): once every server has answered it returns without touching
        the network. Concurrent callers share the one pass — the second to
        arrive finds nothing left unlisted and returns. Nothing is latched, so
        a caller cancelled mid-pass leaves the servers that already answered
        listed and the rest pending for the next ask.
        """
        if not self._unlisted():
            return
        async with self._lock:
            pending = self._unlisted()  # re-read: another caller may have just listed
            if not pending:
                return
            await asyncio.gather(*(self._list(label) for label in pending))

    async def get_tools(self, session: AgentSession) -> list[ToolSpec]:
        # WAITS for the listing — an empty list here is believed by the proxy's
        # routing table, the permission gate and the model alike (module docstring)
        await self.wait_listed()
        return [spec for by_name in self._specs.values() for spec in by_name.values()]

    async def create_execution(self, session: AgentSession, call: ToolCall) -> ToolExecution:
        # waits too: the ToolSpec this has to produce is data only the server
        # has, so it cannot assume get_tools already ran
        await self.wait_listed()
        label, _, _ = call.name.partition(SEP)
        spec = self._specs.get(label, {}).get(call.name)
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
        # does NOT wait for a listing, and does not need to: the route is the
        # label prefix against static server config, so a cold resume still
        # dispatches without a round trip. When that server HAS listed, use its
        # slice for a precise NOT_FOUND on a known server's unknown tool.
        name = tool_execution.raw_tool_call.name
        label, sep, tool_name = name.partition(SEP)
        server = self._servers.get(label)
        listed = self._specs.get(label)
        if not sep or server is None or (listed is not None and name not in listed):
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
