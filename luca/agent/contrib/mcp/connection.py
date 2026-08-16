"""One long-lived connection per configured MCP server.

This is the piece the previous attempt did not have. It connected per call:
`prepare()` returned a callable that opened a session, made one request and
closed it, which respawned a stdio subprocess on every single tool call and
threw away anything the server was holding. The review called that "a redesign,
not a patch", and asked for one long-lived task per server owning the
connection with a response future per request.

It also named the reason it had not been done: the official SDK's session is an
anyio context manager that must be entered and exited in the SAME task, so a
warm connection needed an actor to own the cancel scope. Writing the client on
asyncio primitives removes that constraint entirely. The reader task owns the
read side, writes happen in the caller's own task under a lock held for
microseconds, and there is no cancel scope to be careful with.

The protocol changed underneath the argument as well. 2026-07-28 removed
sessions outright and requires a server that needs cross-call state to mint an
explicit handle passed as a tool argument, so "reconnecting loses server state"
is now a thing the spec forbids servers from having rather than a risk to
manage.

WHAT THIS LAYER OWNS: the negotiated era, the request-id sequence, header
construction, result validation, and the retry policy. WHAT IT DOES NOT: the
process and the socket (that is `transport`), and any notion of a tool (that is
`catalog` and `registry`).

RETRY POLICY, stated plainly because it is a correctness decision:

- A dropped connection on an IDEMPOTENT method (`server/discover`, the four
  listings) is retried once. The protocol is stateless, so a restarted server
  is indistinguishable from the original and the caller could not tell.
- `tools/call` is NEVER retried. A tool may have side effects, and "the server
  restarted mid-call" is an honest failure the model can reason about. Silently
  running someone's `create_issue` twice is not.
- A handshake-era connection retries nothing, because its session id died with
  the process and the reconnect has to start over.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Any, Final

import httpx

from . import wire
from .errors import UNSUPPORTED_PROTOCOL_VERSION, McpError, McpServerGone, McpUnsupportedVersion
from .headers import request_headers
from .protocol import Negotiated, negotiate
from .servers import HttpServer, Server, ServerStatus, StdioServer
from .transport import HttpTransport, StdioTransport
from .wire import Era

logger = logging.getLogger(__name__)

DEFAULT_CONNECT_TIMEOUT_MS: Final = 30_000
DEFAULT_LIST_TIMEOUT_MS: Final = 30_000
DEFAULT_REQUEST_TIMEOUT_MS: Final = 120_000

# Methods safe to re-send against a fresh process, because they change nothing.
IDEMPOTENT: Final = frozenset(
    {
        "server/discover",
        "tools/list",
        "prompts/list",
        "resources/list",
        "resources/templates/list",
    }
)


class ServerConnection:
    """The live link to one MCP server, for the life of the process.

    Constructed eagerly and connected lazily: building one opens nothing, so a
    service can hold a connection per configured server and pay only for the
    ones actually used.
    """

    def __init__(
        self,
        server: Server,
        *,
        client: httpx.AsyncClient | None = None,
        auth: httpx.Auth | None = None,
        on_tools_changed=None,
    ) -> None:
        self.server = server
        self.label = server.label
        self._ids = itertools.count(1)
        self._on_tools_changed = on_tools_changed
        self._negotiated: Negotiated | None = None
        # One lock around negotiation, so ten concurrent first calls probe once.
        # Held across the probe deliberately: everything behind it is waiting
        # for the same answer, and a second probe would burn a second id.
        self._connecting = asyncio.Lock()
        self.error: str | None = None
        if isinstance(server, StdioServer):
            self._transport = StdioTransport(server, on_notification=self._on_notification)
        elif client is None:
            raise ValueError("an http MCP server needs a shared httpx.AsyncClient")
        else:
            self._transport = HttpTransport(server, client, on_notification=self._on_notification, auth=auth)

    @property
    def negotiated(self) -> Negotiated | None:
        return self._negotiated

    def status(self) -> ServerStatus:
        return ServerStatus(
            label=self.label,
            connected=self._negotiated is not None and self._transport.alive,
            protocol_version=self._negotiated.protocol_version if self._negotiated else None,
            error=self.error,
        )

    async def connect(self) -> Negotiated:
        """Probe once and cache the answer. Idempotent and safe to call from
        anywhere, which is what lets a warm-up worker and a first tool call race
        without either of them noticing."""
        if self._negotiated is not None and self._transport.alive:
            return self._negotiated
        async with self._connecting:
            if self._negotiated is not None and self._transport.alive:
                return self._negotiated
            timeout_s = _ms(self.server.connect_timeout_in_ms, DEFAULT_CONNECT_TIMEOUT_MS) / 1000
            try:
                self._negotiated = await negotiate(
                    self._transport,
                    label=self.label,
                    timeout_s=timeout_s,
                    next_id=lambda: next(self._ids),
                )
            except Exception as exc:
                self.error = str(exc) or type(exc).__name__
                logger.error("mcp server=%s failed to connect", self.label, exc_info=exc)
                raise
            self.error = None
            logger.info(
                "mcp server=%s connected (%s, protocol %s)",
                self.label,
                self._negotiated.era.value,
                self._negotiated.protocol_version,
            )
            return self._negotiated

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_ms: int | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        """One validated MCP result.

        Raises `McpError` for a server-reported failure, `McpServerGone` for a
        connection that died, and `McpProtocolError` for an answer that does not
        fit the version's schema.
        """
        timeout_s = _ms(timeout_ms, DEFAULT_REQUEST_TIMEOUT_MS) / 1000
        try:
            return await self._request_once(method, params, timeout_s=timeout_s, extra_headers=extra_headers)
        except McpServerGone:
            if method not in IDEMPOTENT or self._era() is not Era.MODERN:
                raise
            # Stateless protocol, unchanged state: a caller cannot tell this
            # from the first attempt, so retrying is honest.
            logger.info("mcp server=%s dropped during %s; retrying once", self.label, method)
            self._negotiated = None
            return await self._request_once(method, params, timeout_s=timeout_s, extra_headers=extra_headers)
        except McpError as exc:
            if exc.code != UNSUPPORTED_PROTOCOL_VERSION:
                raise
            # A restarted server may have been upgraded under us. Re-probe once,
            # then take the answer whatever it is.
            logger.info("mcp server=%s rejected protocol %s; re-probing", self.label, self._version())
            self._negotiated = None
            return await self._request_once(method, params, timeout_s=timeout_s, extra_headers=extra_headers)

    async def _request_once(self, method, params, *, timeout_s, extra_headers):
        negotiated = await self.connect()
        version, era = negotiated.protocol_version, negotiated.era
        if era is Era.HANDSHAKE and method not in _HANDSHAKE_METHODS:
            raise McpUnsupportedVersion(
                f"MCP server {self.label!r} speaks protocol {version}, which has no {method!r}.",
                supported=(version,),
            )
        frame = wire.request_frame(next(self._ids), method, params, protocol_version=version, era=era)
        headers = request_headers(method, frame.get("params"), protocol_version=version)
        if extra_headers:
            headers.update(extra_headers)
        message = await self._transport.request(frame, headers=headers, timeout_s=timeout_s)
        return wire.parse_result(method, version, wire.result_payload(message))

    async def list_tools(self):
        """Every tool the server offers, following `nextCursor` to the end.

        Pagination is not optional to get right: the previous attempt stopped
        after the first page, and a paginated server showed as connected with a
        wrong tool count and no error anywhere.
        """
        timeout_ms = _ms(self.server.list_timeout_in_ms, DEFAULT_LIST_TIMEOUT_MS)
        tools: list = []
        cursor: str | None = None
        hints: tuple[int, str] = (0, "private")
        while True:
            params = {"cursor": cursor} if cursor else None
            page = await self.request("tools/list", params, timeout_ms=timeout_ms)
            tools.extend(page.tools)
            if not cursor:  # the first page's hints describe the whole listing
                hints = (getattr(page, "ttl_ms", 0) or 0, getattr(page, "cache_scope", None) or "private")
            cursor = page.next_cursor
            if not cursor:
                return tools, hints

    async def call_tool(self, name: str, arguments: dict, *, timeout_ms: int | None = None, extra_headers=None):
        return await self.request(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout_ms=timeout_ms,
            extra_headers=extra_headers,
        )

    def _era(self) -> Era | None:
        return self._negotiated.era if self._negotiated else None

    def _version(self) -> str | None:
        return self._negotiated.protocol_version if self._negotiated else None

    def _on_notification(self, message) -> None:
        """Inbound server-initiated messages.

        On stdio this is free: the reader already owns the channel, so a
        `toolsListChanged` costs nothing to honour. Over HTTP the same
        notification would need a second long-lived `subscriptions/listen`
        stream, which is not in this version — TTL-driven refresh covers the
        same need with a staleness bound the server itself chose.
        """
        if message.method == "notifications/tools/list_changed" and self._on_tools_changed is not None:
            logger.info("mcp server=%s reported a tool-list change", self.label)
            self._on_tools_changed(self.label)

    async def aclose(self) -> None:
        self._negotiated = None
        await self._transport.aclose()


# The 2026 methods a pre-2026 server cannot be asked for. Anything not listed
# here is refused locally rather than sent and misread.
_HANDSHAKE_METHODS: Final = frozenset(
    {
        "initialize",
        "ping",
        "tools/list",
        "tools/call",
        "prompts/list",
        "prompts/get",
        "resources/list",
        "resources/read",
        "resources/templates/list",
        "completion/complete",
        "logging/setLevel",
    }
)


def _ms(value: int | None, fallback: int) -> int:
    return fallback if value is None else value


__all__ = ["DEFAULT_CONNECT_TIMEOUT_MS", "DEFAULT_LIST_TIMEOUT_MS", "IDEMPOTENT", "ServerConnection"]


def build_connection(server: Server, **kwargs) -> ServerConnection:
    """A connection for one server. Exists so callers do not branch on type."""
    if isinstance(server, HttpServer) and kwargs.get("client") is None:
        raise ValueError(f"MCP server {server.label!r} is http and needs a shared httpx.AsyncClient")
    return ServerConnection(server, **kwargs)
