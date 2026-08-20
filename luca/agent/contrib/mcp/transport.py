"""The two ways to reach an MCP server: a subprocess, or an HTTP endpoint.

Both expose `request`, `notify` and `aclose`, so the protocol layer above never
branches on transport.

stdio is one long-lived subprocess sharing one channel: many requests may be in
flight and are correlated by JSON-RPC id, so a reader task resolves a future per
id. The write lock is held for the write and the drain only, never across the
response wait (contract rule 13c). A crashed server is noticed by the reader
hitting EOF; the next request spawns a fresh process, which the stateless
protocol makes safe.

HTTP has no process, so a connection is just state: the negotiated era, the
legacy session id, and a borrowed reference to one shared `httpx.AsyncClient`.
Cancelling closes the response stream, which is itself the cancellation signal
in this revision.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import os
import signal
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any, Final

import httpx

from . import wire
from .errors import McpAuthRequired, McpProtocolError, McpServerGone
from .oauth.metadata import parse_challenge
from .servers import HttpServer, StdioServer
from .sse import SseDecoder

logger = logging.getLogger(__name__)

READ_SIZE: Final = 65_536
# Wait this long for a subprocess to exit after its stdin is closed before
# escalating. The spec's shutdown sequence: close stdin, wait, then terminate.
TERMINATE_GRACE_S: Final = 2.0
# A server that dies on startup must not be respawned in a hot loop.
RESPAWN_FLOOR_S: Final = 1.0

SESSION_ID_HEADER: Final = "Mcp-Session-Id"
ACCEPT: Final = "application/json, text/event-stream"

NotificationHandler = Callable[[Any], None]


class StdioTransport:
    """One MCP server as a long-lived subprocess."""

    def __init__(self, server: StdioServer, *, on_notification: NotificationHandler | None = None) -> None:
        self.server = server
        self._on_notification = on_notification
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task | None = None
        self._draining: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._ids = itertools.count(1)
        self._lock = asyncio.Lock()  # guards spawn and stdin writes, never a response wait
        self._last_spawn = 0.0
        self.restarts = 0
        # Set by `aclose`, cleared by a spawn: true only for a close that
        # landed WHILE one was in flight. `_teardown` runs unlocked so shutdown
        # never queues behind a long call, so it can swap `_process` to None
        # before `_ensure` assigns the process it just started — one nothing
        # will ever close, which `start_new_session=True` then keeps alive
        # through the terminal's own Ctrl-C.
        self._closed = False

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def request(self, frame: Mapping[str, Any], *, headers: Mapping[str, str] | None = None, timeout_s: float):
        # `headers` is ignored: stdio carries its metadata in the message body.
        # The signature is shared with HttpTransport so callers do not branch.
        del headers
        request_id = frame["id"]
        waiter = await self._write(frame, request_id=request_id)
        try:
            async with asyncio.timeout(timeout_s):
                return await waiter
        except (asyncio.CancelledError, TimeoutError):
            await self._cancel_remote(request_id)
            raise
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, frame: Mapping[str, Any], *, headers: Mapping[str, str] | None = None) -> None:
        del headers
        await self._write(frame)

    async def _write(self, frame: Mapping[str, Any], *, request_id: int | None = None) -> asyncio.Future | None:
        """Write one frame, registering its waiter if it expects an answer.

        Registration is inside the lock and after the spawn, so tearing down a
        dead predecessor cannot fail a waiter belonging to its successor.
        """
        async with self._lock:
            process = await self._ensure()
            waiter: asyncio.Future | None = None
            if request_id is not None:
                waiter = asyncio.get_running_loop().create_future()
                self._pending[request_id] = waiter
            try:
                process.stdin.write(wire.encode(frame))
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                if request_id is not None:
                    self._pending.pop(request_id, None)
                raise McpServerGone(f"MCP server {self.server.label!r} closed its input stream.") from exc
            return waiter

    async def _cancel_remote(self, request_id: int) -> None:
        frame = {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": request_id}}
        with contextlib.suppress(Exception):
            await asyncio.shield(self._write(frame))

    async def _ensure(self) -> asyncio.subprocess.Process:
        """The live process, spawning one if there is none."""
        if self.alive:
            return self._process
        loop = asyncio.get_running_loop()
        if self._process is not None:
            self.restarts += 1
            since = loop.time() - self._last_spawn
            if since < RESPAWN_FLOOR_S:
                await asyncio.sleep(RESPAWN_FLOOR_S - since)
            await self._teardown()
        self._last_spawn = loop.time()
        # A spawn was asked for, so any close BEFORE this point is spent; only
        # one arriving while the spawn is in flight has a process to catch.
        # `reconnect` closes and re-lists, and would otherwise never start.
        self._closed = False
        try:
            self._process = await asyncio.create_subprocess_exec(
                self.server.command,
                *self.server.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **self.server.environment},
                start_new_session=True,  # its own process group, so a kill takes children too
            )
        except OSError as exc:
            raise McpServerGone(f"Could not start MCP server {self.server.label!r}: {exc}") from exc
        if self._closed:
            # Shut down while this one was spawning. Nothing above will ever
            # ask for it again, so kill it here or it outlives the process.
            spawned, self._process = self._process, None
            _kill_process_group(spawned)
            await spawned.wait()
            raise McpServerGone(f"MCP server {self.server.label!r} was shut down.")
        self._reader = asyncio.create_task(self._read_loop(self._process))
        self._draining = asyncio.create_task(self._drain_stderr(self._process))
        return self._process

    async def _read_loop(self, process: asyncio.subprocess.Process) -> None:
        """Own stdout for the life of one process, resolving waiters by id."""
        try:
            async for line in _lines(process.stdout):
                self._dispatch(line)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("mcp server=%s reader stopped", self.server.label, exc_info=exc)
        finally:
            # EOF: nobody is coming, so fail every waiter rather than leaving
            # callers to their timeouts.
            self._fail_pending(McpServerGone(f"MCP server {self.server.label!r} exited."))

    def _dispatch(self, line: bytes) -> None:
        if not line.strip():
            return
        try:
            message = wire.decode(line)
        except McpProtocolError as exc:
            # One unreadable frame does not poison the channel.
            logger.warning("mcp server=%s sent an unreadable frame", self.server.label, exc_info=exc)
            return
        if wire.is_notification(message):
            if self._on_notification is not None:
                self._on_notification(message)
            return
        waiter = self._pending.pop(getattr(message, "id", None), None)
        if waiter is not None and not waiter.done():
            waiter.set_result(message)

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        """Read stderr forever and log it.

        Not optional: an undrained pipe fills and the server blocks on its next
        write, which looks exactly like a hang.
        """
        with contextlib.suppress(Exception):
            async for line in _lines(process.stderr):
                logger.debug("mcp server=%s stderr: %s", self.server.label, line.decode("utf-8", errors="replace"))

    def _fail_pending(self, error: Exception) -> None:
        pending, self._pending = self._pending, {}
        for waiter in pending.values():
            if not waiter.done():
                waiter.set_exception(error)

    async def aclose(self) -> None:
        self._closed = True
        await self._teardown()

    async def _teardown(self) -> None:
        """Close stdin, wait, then escalate — the spec's shutdown sequence.

        Swap-then-act so a second call finds nothing to do, and deliberately
        unlocked: shutting down must not queue behind a long-running call.
        """
        process, self._process = self._process, None
        reader, self._reader = self._reader, None
        draining, self._draining = self._draining, None
        for task in (reader, draining):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        if process is not None and process.returncode is None:
            with contextlib.suppress(Exception):
                process.stdin.close()
            try:
                async with asyncio.timeout(TERMINATE_GRACE_S):
                    await process.wait()
            except TimeoutError:
                _kill_process_group(process)
                await process.wait()
        if process is not None:
            for stream in (process.stdout, process.stderr):
                with contextlib.suppress(Exception):
                    stream.feed_eof()
            with contextlib.suppress(Exception):
                process.stdin.close()
        self._fail_pending(McpServerGone(f"MCP server {self.server.label!r} was shut down."))


class HttpTransport:
    """One MCP server behind a Streamable HTTP endpoint.

    The `httpx.AsyncClient` is owned by the service and shared by every HTTP
    server, so there is one pool and one close path.
    """

    def __init__(
        self,
        server: HttpServer,
        client: httpx.AsyncClient,
        *,
        on_notification: NotificationHandler | None = None,
        auth: httpx.Auth | None = None,
    ) -> None:
        self.server = server
        self._client = client
        self._on_notification = on_notification
        self._auth = auth
        # Handshake-era only; a modern server ignores it.
        self.session_id: str | None = None
        self.restarts = 0

    @property
    def alive(self) -> bool:
        return True

    async def request(self, frame: Mapping[str, Any], *, headers: Mapping[str, str] | None = None, timeout_s: float):
        request_headers = self._headers(headers)
        try:
            async with self._client.stream(
                "POST",
                self.server.url,
                json=dict(frame),
                headers=request_headers,
                timeout=timeout_s,
                auth=self._auth,
            ) as response:
                captured = response.headers.get(SESSION_ID_HEADER)
                if captured:
                    self.session_id = captured
                if response.status_code in (401, 403):
                    await response.aread()
                    raise McpAuthRequired(
                        f"MCP server {self.server.label!r} refused the request ({response.status_code}).",
                        challenge=parse_challenge(response.headers.get("www-authenticate")),
                    )
                if response.headers.get("content-type", "").startswith("text/event-stream"):
                    return await self._read_stream(response, frame.get("id"))
                await response.aread()
                return self._single(response)
        except httpx.HTTPError as exc:
            raise McpServerGone(f"MCP server {self.server.label!r} is unreachable: {exc}") from exc

    async def notify(self, frame: Mapping[str, Any], *, headers: Mapping[str, str] | None = None) -> None:
        """Fire and forget. A notification is answered with 202 and no body."""
        with contextlib.suppress(httpx.HTTPError):
            await self._client.post(
                self.server.url,
                json=dict(frame),
                headers=self._headers(headers),
                auth=self._auth,
            )

    def _headers(self, extra: Mapping[str, str] | None) -> dict[str, str]:
        headers = {"Accept": ACCEPT, "Content-Type": "application/json", **self.server.header_map}
        if extra:
            headers.update(extra)
        if self.session_id:
            headers[SESSION_ID_HEADER] = self.session_id
        return headers

    def _single(self, response: httpx.Response):
        if response.status_code >= 400 and not response.content:
            raise McpServerGone(f"MCP server {self.server.label!r} answered {response.status_code} with no body.")
        return wire.decode(response.content)

    async def _read_stream(self, response: httpx.Response, request_id):
        """Consume the SSE stream until the frame answering this request.

        A stream that ends without it is a dead request: this revision removed
        resumability, so there is nothing to reconnect to.
        """
        decoder = SseDecoder()
        async for chunk in response.aiter_bytes():
            for event in decoder.feed(chunk):
                answer = self._consume(event, request_id)
                if answer is not None:
                    return answer
        for event in decoder.flush():
            answer = self._consume(event, request_id)
            if answer is not None:
                return answer
        raise McpServerGone(f"MCP server {self.server.label!r} closed the stream before answering.")

    def _consume(self, event, request_id):
        try:
            message = wire.decode(event.data)
        except McpProtocolError as exc:
            logger.warning("mcp server=%s sent an unreadable SSE frame", self.server.label, exc_info=exc)
            return None
        if wire.is_notification(message):
            if self._on_notification is not None:
                self._on_notification(message)
            return None
        return message if getattr(message, "id", None) == request_id else None

    async def aclose(self) -> None:
        """Nothing to close: the client belongs to the service, and this
        revision has no session to terminate."""


Transport = StdioTransport | HttpTransport


async def _lines(stream: asyncio.StreamReader) -> AsyncIterator[bytes]:
    """Newline-delimited frames, with no length limit.

    `StreamReader.readline()` caps a line at 64 KiB and raises past it, which
    any tool returning a file's contents would trip.
    """
    buffer = bytearray()
    while chunk := await stream.read(READ_SIZE):
        buffer.extend(chunk)
        while (index := buffer.find(b"\n")) != -1:
            line = bytes(buffer[:index])
            del buffer[: index + 1]
            yield line
    if buffer:
        yield bytes(buffer)


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
