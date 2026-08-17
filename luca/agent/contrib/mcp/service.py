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
from .errors import McpAuthRequired, McpServerGone
from .headers import param_headers
from .mapping import spec_identity, to_execution_result
from .servers import HttpServer, Server, ServerState, ServerStatus

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
        # Turned off for this process only. Nothing is written back to
        # `luca.json`, which luca reads and never edits.
        self._disabled: set[str] = set()
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
        if not self._disabled:
            return self.catalog.specs()
        return [spec for spec in self.catalog.specs() if not self._is_disabled(spec)]

    def _is_disabled(self, spec: ToolSpec) -> bool:
        identity = spec_identity(spec)
        return identity is not None and identity[0] in self._disabled

    def spec(self, name: str) -> ToolSpec | None:
        found = self.catalog.spec(name)
        return None if found is not None and self._is_disabled(found) else found

    def status(self) -> list[ServerStatus]:
        return [self._status(label, connection) for label, connection in self._connections.items()]

    def _status(self, label: str, connection: ServerConnection) -> ServerStatus:
        """One server's state, decided here because it depends on the catalog.

        CONNECTED means the model can call this server's tools, which is true
        as soon as there is a listing for it. A relaunch inside the TTL serves
        that listing off disk and never opens a connection at all, so asking
        the transport whether it is alive would report a working server as
        broken.
        """
        reported = connection.status()
        if label in self._disabled:
            state = ServerState.DISABLED
        elif connection.needs_auth:
            state = ServerState.NEEDS_AUTH
        elif connection.error is None and self.catalog.has(label):
            state = ServerState.CONNECTED
        else:
            state = ServerState.INACTIVE
        return reported.model_copy(
            update={
                "state": state,
                "tool_count": 0 if state is ServerState.DISABLED else self.catalog.tool_count(label),
                "rejected_tools": self.catalog.rejected(label),
            }
        )

    def status_for(self, label: str) -> ServerStatus | None:
        return next((status for status in self.status() if status.label == label), None)

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
        due = [label for label in due if label not in self._disabled]
        if not due:
            self._resolve_first_listing()
            return
        await asyncio.gather(*(self._list(label) for label in due))
        self._resolve_first_listing()

    async def _list(self, label: str) -> None:
        """List one server, recording a failure rather than raising it: one
        dead server must not cost its siblings their tools."""
        connection = self._connections[label]
        provider = self._auth.get(label)
        if provider is not None:
            # Ask BEFORE listing. A server that has never been authorized would
            # otherwise answer 401 and be reported as a failure, which it is
            # not — it is waiting for a login nobody has been offered yet.
            try:
                await provider.ensure_token(self._client, interactive=False)
            except McpAuthRequired:
                connection.needs_auth = True
                connection.error = None
                logger.info("mcp server=%s needs authorization", label)
                return
        try:
            tools, (ttl_ms, cache_scope) = await connection.list_tools()
        except asyncio.CancelledError:
            raise
        except McpAuthRequired as exc:
            # The 401 names the scopes it wants; the next login asks for them.
            if provider is not None:
                provider.observe_challenge(exc.challenge)
            connection.needs_auth = True
            connection.error = None
            logger.info("mcp server=%s needs authorization", label)
            return
        except Exception as exc:
            connection.error = str(exc) or type(exc).__name__
            logger.error("mcp server=%s could not be listed", label, exc_info=exc)
            return
        connection.error = None
        connection.needs_auth = False
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

    async def login(self, label: str) -> None:
        """Run the browser flow for one server, then list it.

        The only interactive entry point besides startup, and never reached
        from a turn: a browser opening mid-message is the behaviour this
        avoids.
        """
        provider = self._auth.get(label)
        if provider is None:
            raise McpAuthRequired(f"MCP server {label!r} is not configured for oauth.")
        await provider.authorize(self._client)
        self._connections[label].needs_auth = False
        await self.refresh([label])

    async def reconnect(self, label: str) -> None:
        """Drop what was negotiated and list the server again."""
        connection = self._connections.get(label)
        if connection is None:
            return
        self._disabled.discard(label)
        await connection.aclose()
        self.catalog.invalidate(label)
        await self.refresh([label])

    async def set_enabled(self, label: str, enabled: bool) -> None:
        """Withhold a server's tools from the model, for this process only.

        Its cached listing is kept, so re-enabling costs nothing.
        """
        if enabled:
            self._disabled.discard(label)
            if not self.catalog.has(label):
                await self.refresh([label])
            return
        self._disabled.add(label)
        await self._connections[label].aclose()

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
