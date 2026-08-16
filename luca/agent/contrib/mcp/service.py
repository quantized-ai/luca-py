"""McpService — everything MCP owns that outlives a session.

One `ServerConnection` per configured server, the tool catalog, the refresh
loop, one shared `httpx.AsyncClient` and the OAuth providers.

BUILT ONCE, by the application, beside `CheckpointService`. `/clear`, `/new`,
`/resume` and fork all rebuild the runner through `_reset_session`; anything
owned by a plugin would reconnect and re-run OAuth on each of them. The registry
and plugin above are stateless views, which is also what satisfies contract rule
13a — they hold nothing that could belong to another conversation after an
await.
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
        """One OAuth provider per server, built once and kept, so token state
        and discovery are shared rather than redone per operation."""
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
        """Nothing has ever been listed, so the model would see no MCP tools."""
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

        Never raises: one unreachable server records its reason and leaves its
        siblings alone. Single-flight and awaited by everyone, so `await
        start()` always means started, even for a caller that arrived second.
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
        """List one server, recording a failure rather than raising it: one
        dead server must not cost its siblings their tools."""
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
        """Sleep until the earliest slice goes stale, refresh, repeat.

        The "out of band" half of contract rule 3; nothing in the tool path
        waits on it.
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

        The APPLICATION's wait, not the registry's: the tool path may never
        block (rule 3), but a first run is better served by a visible short
        pause than by a turn where the tools silently do not exist.
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
        # Mirrored into headers AND left in the body: a server must reject a
        # mismatch between the two.
        extra = param_headers(input_schema, arguments) if input_schema else None
        result = await connection.call_tool(tool, arguments, timeout_ms=timeout_ms, extra_headers=extra)
        return to_execution_result(result)

    async def aclose(self) -> None:
        """Shut every connection down, from the app's graceful paths only.

        Never from `_reset_session`: `/clear` swaps the runner, not the servers.
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
