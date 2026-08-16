"""McpService — everything MCP owns that outlives a session.

This is the structural answer to the review finding that `/new` skipped the
startup notice and popped an OAuth browser mid-message. The mechanism behind
that bug was ownership: the connections lived on a plugin, `_reset_session`
rebuilt the runner for `/clear`, `/new`, `/resume` and fork, and each rebuild
produced a fresh plugin with empty state that re-discovered everything.

The suggested fix was to spawn the same startup worker from `_reset_session`
too. This does the other thing: the state moves somewhere a session reset
cannot reach. The service is built once in `AgentApp.__init__`, beside
`CheckpointService`, which has exactly the same reason to outlive a session.
The registry and the plugin become stateless views over it, so `_reset_session`
is left untouched and the bug has nowhere to come back from.

What lives here: one `ServerConnection` per configured server, the tool
catalog, the refresh loop, one shared `httpx.AsyncClient`, and the OAuth
providers. All of it process-scoped and none of it per-conversation, which
matters for registry contract rule 13a: the registry above holds only immutable
configuration, so nothing it reads after an await can belong to another
conversation.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from luca.agent.core import ExecutionResult, ToolSpec

from .catalog import DEFAULT_TTL_MS, MIN_TTL_MS, ToolCatalog
from .connection import ServerConnection
from .errors import McpServerGone
from .headers import param_headers
from .mapping import to_execution_result
from .servers import HttpServer, Server, ServerStatus

logger = logging.getLogger(__name__)


class McpService:
    """Owns every MCP connection for the life of the process."""

    def __init__(
        self,
        servers: dict[str, Server],
        *,
        catalog_path: Path | None = None,
        token_path: Path | None = None,
        client: httpx.AsyncClient | None = None,
        browser: Callable[[str], Any] | None = None,
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        self.servers = servers
        self.token_path = token_path
        self.catalog = ToolCatalog(catalog_path, **({"now_ms": now_ms} if now_ms else {}))
        self.catalog.track(servers)
        self.catalog.load(servers)
        self._owns_client = client is None
        needs_http = any(isinstance(server, HttpServer) for server in servers.values())
        # One client for every HTTP server: one pool, one set of limits, one
        # close path. Built only if something actually needs it.
        self._client = client if client is not None else (httpx.AsyncClient() if needs_http else None)
        self._auth = self._build_auth(browser)
        self._connections = {
            label: ServerConnection(
                server,
                client=self._client,
                auth=self._auth.get(label),
                on_tools_changed=self.catalog.invalidate,
            )
            for label, server in servers.items()
        }
        self._refresher: asyncio.Task | None = None
        self._starting: asyncio.Task | None = None
        self._first_listing: asyncio.Future | None = None

    def _build_auth(self, browser) -> dict[str, httpx.Auth]:
        """One OAuth provider per server, built once and kept.

        Built here rather than per operation, which is what the review asked
        for: a provider per call re-read the token file and re-ran discovery
        every time, and two of them could bind the same callback port or race
        each other's refresh into a rotated-token dead end.
        """
        wanted = {
            label: server for label, server in self.servers.items() if isinstance(server, HttpServer) and server.oauth
        }
        if not wanted:
            return {}
        from .oauth import OAuthProvider, TokenStore

        store = TokenStore(self.token_path)
        return {label: OAuthProvider(server, store=store, browser=browser) for label, server in wanted.items()}

    @property
    def cold(self) -> bool:
        """Nothing has ever been listed, so the model would see no MCP tools at
        all. The application uses this to decide whether the very first turn is
        worth a short wait."""
        return bool(self.servers) and self.catalog.cold

    def specs(self) -> list[ToolSpec]:
        """Local read. No awaits, no I/O — see the catalog module docstring."""
        return self.catalog.specs()

    def spec(self, name: str) -> ToolSpec | None:
        return self.catalog.spec(name)

    def status(self) -> list[ServerStatus]:
        return [
            connection.status().model_copy(
                update={
                    "tool_count": self.catalog.tool_count(label),
                    "rejected_tools": self.catalog.rejected(label),
                }
            )
            for label, connection in self._connections.items()
        ]

    async def start(self) -> None:
        """Connect to every server and list it, then keep the catalog fresh.

        Driven from the TUI's mount worker. Never raises: one unreachable
        server records its reason and leaves its siblings alone.

        Single-flight, and awaited by everyone. A second caller arriving while
        the first is still listing awaits the SAME work rather than returning
        early, so `await start()` always means "started" — the alternative
        makes a concurrent caller believe a listing finished when it has not,
        which is a race the caller cannot see and cannot fix.
        """
        if self._starting is None:
            self._starting = asyncio.create_task(self._start())
        await asyncio.shield(self._starting)

    async def _start(self) -> None:
        await self.refresh()
        self._refresher = asyncio.create_task(self._refresh_loop())

    async def refresh(self, labels: list[str] | None = None) -> None:
        """List the named servers, or every stale one, concurrently."""
        due = labels if labels is not None else self.catalog.stale(self.servers)
        if not due:
            self._resolve_first_listing()
            return
        await asyncio.gather(*(self._list(label) for label in due))
        self._resolve_first_listing()

    async def _list(self, label: str) -> None:
        """List one server. Records a failure rather than raising it: nothing
        MCP does may crash a run, and one dead server must not cost its
        siblings their tools."""
        connection = self._connections[label]
        try:
            tools, (ttl_ms, cache_scope) = await connection.list_tools()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            connection.error = str(exc) or type(exc).__name__
            logger.error("mcp server=%s could not be listed", label, exc_info=exc)
            return
        connection.error = None
        await self.catalog.put(
            label,
            self.servers[label],
            tools,
            ttl_ms=ttl_ms or DEFAULT_TTL_MS,
            cache_scope=cache_scope,
        )
        logger.info("mcp server=%s listed %d tools", label, self.catalog.tool_count(label))

    async def _refresh_loop(self) -> None:
        """Sleep until the earliest slice goes stale, refresh it, repeat.

        This is the "out of band" half of contract rule 3. Nothing in the tool
        path ever waits on it.
        """
        try:
            while True:
                await asyncio.sleep(self._sleep_for() / 1000)
                with contextlib.suppress(Exception):
                    await self.refresh()
        except asyncio.CancelledError:
            raise

    def _sleep_for(self) -> int:
        deadline = self.catalog.next_refresh_at(self.servers)
        if deadline is None:
            return DEFAULT_TTL_MS
        remaining = deadline - self.catalog._now_ms()
        return max(remaining, MIN_TTL_MS)

    async def first_listing(self, timeout_ms: int) -> None:
        """Wait for the startup listing, bounded, only while genuinely cold.

        The one place a wait is acceptable, and it is the APPLICATION's wait,
        not the registry's: the tool path must never block (rule 3), but a user
        who has just configured their first MCP server is better served by a
        visible two-second pause than by a first turn where the tools silently
        do not exist. Every later turn, and every turn of every later run,
        reads the durable catalog and waits for nothing.
        """
        if not self.cold:
            return
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            async with asyncio.timeout(timeout_ms / 1000):
                await asyncio.shield(self._first_listing_future())

    def _first_listing_future(self) -> asyncio.Future:
        if self._first_listing is None:
            self._first_listing = asyncio.get_running_loop().create_future()
        return self._first_listing

    def _resolve_first_listing(self) -> None:
        if self._first_listing is not None and not self._first_listing.done():
            self._first_listing.set_result(None)

    async def call(
        self,
        label: str,
        tool: str,
        arguments: dict,
        *,
        input_schema: dict | None = None,
        timeout_ms: int | None = None,
    ) -> ExecutionResult:
        """Invoke one tool. Every network operation of a call happens here,
        inside the callable `prepare()` returned (contract rule 3)."""
        connection = self._connections.get(label)
        if connection is None:
            raise McpServerGone(f"MCP server {label!r} is not configured.")
        # `x-mcp-header` mirroring is required of clients, and the values stay
        # in the body as well: a server MUST reject a mismatch between them.
        extra = param_headers(input_schema, arguments) if input_schema else None
        result = await connection.call_tool(tool, arguments, timeout_ms=timeout_ms, extra_headers=extra)
        return to_execution_result(result)

    async def aclose(self) -> None:
        """Shut every connection down. Called from the app's graceful paths.

        Not from `_reset_session`: `/clear` swaps the runner, not the servers,
        and closing here is precisely the bug this design removes.
        """
        refresher, self._refresher = self._refresher, None
        if refresher is not None:
            refresher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await refresher
        starting, self._starting = self._starting, None
        if starting is not None and not starting.done():
            starting.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await starting
        await asyncio.gather(*(connection.aclose() for connection in self._connections.values()))
        if self._client is not None and self._owns_client:
            await self._client.aclose()
