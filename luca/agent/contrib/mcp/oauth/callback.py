"""The loopback listener that catches the authorization redirect.

BINDS PORT 0 AND READS THE PORT BACK. `start()` returns only once bound and
exposes the port it got, so the redirect URI is built from a fact, two flows can
never collide on one port, and nothing has to sleep waiting for a bind.

RFC 8252 §7.3 requires authorization servers to accept any loopback port for a
native client, so this is the specified way rather than a workaround;
`redirect_port` is the escape hatch for servers that violate it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Final
from urllib.parse import parse_qs, urlsplit

logger = logging.getLogger(__name__)

HOST: Final = "127.0.0.1"

_PAGE: Final = (
    "<!doctype html><meta charset=utf-8><title>luca</title>"
    "<body style='font:16px system-ui;padding:3rem'>"
    "<h1>{heading}</h1><p>{detail}</p></body>"
)


class CallbackTimeout(Exception):
    """Nobody completed the flow in the browser."""


class LoopbackCallback:
    """A one-shot HTTP listener for a single authorization redirect."""

    def __init__(self, *, port: int = 0) -> None:
        self._requested = port
        self._server: asyncio.Server | None = None
        self._result: asyncio.Future | None = None
        self.port: int | None = None

    async def start(self) -> int:
        """Bind, and return the port actually bound. Everything downstream is
        built from the return value, so there is no window in which the port is
        unknown or the socket is not listening."""
        self._result = asyncio.get_running_loop().create_future()
        self._server = await asyncio.start_server(self._handle, HOST, self._requested)
        self.port = self._server.sockets[0].getsockname()[1]
        return self.port

    @property
    def redirect_uri(self) -> str:
        if self.port is None:
            raise RuntimeError("the callback listener has not been started")
        return f"http://{HOST}:{self.port}/callback"

    async def wait(self, *, timeout_s: float) -> dict[str, str]:
        """The query parameters the browser was redirected with."""
        try:
            async with asyncio.timeout(timeout_s):
                return await self._result
        except TimeoutError as exc:
            raise CallbackTimeout(
                f"No authorization response arrived at {self.redirect_uri} within {timeout_s:.0f}s."
            ) from exc

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await reader.readline()
            # Drain the headers so the browser's write completes before we close.
            while await reader.readline() not in (b"\r\n", b"\n", b""):
                pass
            query = _query(request)
            body = self._page(query)
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                + body
            )
            await writer.drain()
            if self._result is not None and not self._result.done() and query:
                self._result.set_result(query)
        except Exception as exc:  # a broken browser request must not kill the flow
            logger.debug("mcp oauth callback request failed", exc_info=exc)
        finally:
            writer.close()
            with __import__("contextlib").suppress(Exception):
                await writer.wait_closed()

    def _page(self, query: dict[str, str]) -> bytes:
        if "error" in query:
            heading, detail = "Authorization failed", query.get("error_description", query["error"])
        elif "code" in query:
            heading, detail = "Authorized", "You can close this tab and return to luca."
        else:
            heading, detail = "Nothing to do", "This page expects an authorization redirect."
        return _PAGE.format(heading=heading, detail=detail).encode("utf-8")

    async def aclose(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
            await server.wait_closed()
        if self._result is not None and not self._result.done():
            self._result.cancel()

    async def __aenter__(self) -> LoopbackCallback:
        await self.start()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()


def _query(request_line: bytes) -> dict[str, str]:
    parts = request_line.decode("utf-8", errors="replace").split(" ")
    if len(parts) < 2:
        return {}
    return {key: values[0] for key, values in parse_qs(urlsplit(parts[1]).query).items() if values}
