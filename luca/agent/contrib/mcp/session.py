"""Open a short-lived MCP session for one operation.

The `mcp` SDK's transports and `ClientSession` are async context managers that
must be entered and exited in the SAME task (anyio's cancel-scope rule).
Connecting per call keeps the whole lifecycle inside the caller's task, so no
long-lived actor is needed: `get_tools` lists inside its own task, and a
prepared tool call runs inside the runner's per-call task. Closing the context
tears the stdio subprocess down in that same task, even under cancellation.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.client.streamable_http import streamablehttp_client

from .config import McpServerDef, StdioServer


def _open_transport(server: McpServerDef, auth):
    if isinstance(server, StdioServer):
        env = None
        if server.env:
            # merge over the SDK's default env so PATH etc. survive
            env = {**get_default_environment(), **server.env}
        return stdio_client(
            StdioServerParameters(
                command=server.command,
                args=list(server.args),
                env=env,
            )
        )
    return streamablehttp_client(
        server.url,
        headers=dict(server.headers) or None,
        auth=auth,
    )


@contextlib.asynccontextmanager
async def open_session(server: McpServerDef, *, auth=None) -> AsyncIterator[ClientSession]:
    """A connected, initialized `ClientSession`, torn down on exit."""
    async with _open_transport(server, auth) as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def list_tools(server: McpServerDef, *, auth=None) -> list[types.Tool]:
    async with open_session(server, auth=auth) as session:
        return (await session.list_tools()).tools


async def call_tool(
    server: McpServerDef,
    name: str,
    arguments: dict,
    *,
    auth=None,
) -> types.CallToolResult:
    async with open_session(server, auth=auth) as session:
        return await session.call_tool(name, arguments)
