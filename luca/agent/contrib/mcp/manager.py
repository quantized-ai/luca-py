"""McpManager — owns every server connection and routes tool calls.

Constructed (not connected) in `cli.py`, connected in the app's `on_mount`, and
closed in `on_unmount`. A server that fails to start contributes no tools and is
recorded in `failures`; it never crashes the app.
"""

from __future__ import annotations

import asyncio

from mcp import types

from .config import HttpServer, McpServerDef
from .connection import McpConnection

# Separator between a server label and a tool name on the wire.
SEP = "__"


class McpManager:
    def __init__(self, servers: dict[str, McpServerDef], *, auth_factory=None) -> None:
        for label in servers:
            if SEP in label:
                raise ValueError(f"MCP server label {label!r} may not contain {SEP!r}")
        self._servers = servers
        self._auth_factory = auth_factory  # (label, HttpServer) -> httpx.Auth | None
        self._connections: dict[str, McpConnection] = {}
        # every connection object created, so aclose can close one that is still
        # mid-start when the app exits (otherwise its subprocess leaks)
        self._all: list[McpConnection] = []
        self.failures: dict[str, str] = {}

    async def start_all(self) -> None:
        async def _start(label: str, server: McpServerDef) -> None:
            auth = None
            if isinstance(server, HttpServer) and server.oauth and self._auth_factory:
                auth = self._auth_factory(label, server)
            conn = McpConnection(label, server, auth=auth)
            self._all.append(conn)  # tracked before start, so aclose sees it
            try:
                await conn.start()
            except Exception as exc:
                self.failures[label] = str(exc)
                await conn.aclose()
                return
            self._connections[label] = conn

        await asyncio.gather(
            *(_start(label, server) for label, server in self._servers.items()),
        )

    def list_tools(self) -> list[tuple[str, types.Tool]]:
        """Every live tool, namespaced `label__toolname`."""
        return [(f"{label}{SEP}{tool.name}", tool) for label, conn in self._connections.items() for tool in conn.tools]

    async def call_tool(self, namespaced: str, arguments: dict) -> types.CallToolResult:
        label, _, name = namespaced.partition(SEP)
        conn = self._connections.get(label)
        if conn is None:
            raise KeyError(namespaced)
        return await conn.call_tool(name, arguments)

    async def aclose(self) -> None:
        await asyncio.gather(
            *(conn.aclose() for conn in self._all),
            return_exceptions=True,
        )
        self._connections.clear()
        self._all.clear()

    @property
    def connected_labels(self) -> list[str]:
        return list(self._connections)
