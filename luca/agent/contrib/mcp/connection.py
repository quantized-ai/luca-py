"""One MCP server as an actor.

The `mcp` SDK exposes its transports and `ClientSession` as async context
managers that must be entered and exited in the SAME task (anyio's cancel-scope
rule). Holding one open across the app's lifetime while calling it from other
tasks would trip "cancel scope in a different task". So each server runs its
whole session lifecycle inside a single `_serve` task; callers dispatch a
`call_tool` request over a queue and await a future. Enter and exit both happen
inside `_serve`, so the scope stays in one task.
"""

from __future__ import annotations

import asyncio
import contextlib

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.client.streamable_http import streamablehttp_client

from .config import McpServerDef, StdioServer

_SHUTDOWN = object()


class McpConnection:
    def __init__(self, label: str, server: McpServerDef, *, auth=None) -> None:
        self.label = label
        self._server = server
        self._auth = auth
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._ready = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._error: Exception | None = None
        self._done = False
        self.tools: list[types.Tool] = []

    @property
    def is_connected(self) -> bool:
        return self._task is not None and not self._done and self._error is None

    async def start(self) -> None:
        """Spawn the serve loop and wait until the session is initialized (tools
        listed) or the connection failed. Raises the failure to the caller."""
        self._task = asyncio.ensure_future(self._serve())
        await self._ready.wait()
        if self._error is not None:
            raise self._error

    def _transport(self):
        if isinstance(self._server, StdioServer):
            env = None
            if self._server.env:
                # merge over the SDK's default env so PATH etc. survive
                env = {**get_default_environment(), **self._server.env}
            return stdio_client(
                StdioServerParameters(
                    command=self._server.command,
                    args=list(self._server.args),
                    env=env,
                )
            )
        return streamablehttp_client(
            self._server.url,
            headers=dict(self._server.headers) or None,
            auth=self._auth,
        )

    async def _serve(self) -> None:
        try:
            async with self._transport() as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.tools = (await session.list_tools()).tools
                    self._ready.set()
                    while True:
                        item = await self._inbox.get()
                        if item is _SHUTDOWN:
                            return
                        name, arguments, fut = item
                        if fut.cancelled():
                            continue
                        try:
                            result = await session.call_tool(name, arguments)
                        except Exception as exc:  # a tool/protocol error → the caller
                            if not fut.cancelled():
                                fut.set_exception(exc)
                            continue
                        if not fut.cancelled():
                            fut.set_result(result)
        except Exception as exc:  # connection / initialize failure
            self._error = exc
        # A BaseException (the actor cancelled, e.g. transport death or aclose)
        # is left to propagate/close; `_done` below still marks it dead.
        finally:
            self._done = True
            self._ready.set()
            self._fail_pending()

    def _fail_pending(self) -> None:
        """Resolve every queued-but-unserviced caller so `call_tool()` never
        hangs once the actor has stopped."""
        while not self._inbox.empty():
            try:
                item = self._inbox.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is _SHUTDOWN:
                continue
            fut = item[2]
            if not fut.done():
                fut.set_exception(
                    RuntimeError(f"MCP server {self.label!r} disconnected"),
                )

    async def call_tool(self, name: str, arguments: dict) -> types.CallToolResult:
        if not self.is_connected:
            raise RuntimeError(f"MCP server {self.label!r} is not connected")
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._inbox.put((name, arguments, fut))
        return await fut

    async def aclose(self) -> None:
        if self._task is None:
            return
        if self._ready.is_set() and not self._done:
            # connected and serving — stop the loop gracefully in its own task
            await self._inbox.put(_SHUTDOWN)
        else:
            # still connecting, or already dead — cancel so the transport's
            # __aexit__ tears the subprocess/session down in its own task
            self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._task
        self._task = None
