"""One long-lived connection per configured MCP server.

Owns the negotiated era, the request-id sequence, header construction, result
validation and the retry policy. Not the process or the socket (`transport`),
and not any notion of a tool (`catalog`, `registry`).

RETRY POLICY, because it is a correctness decision:

- A dropped connection on an IDEMPOTENT method (`server/discover`, the
  listings) is retried once. The protocol is stateless, so a restarted server
  is indistinguishable from the original and the caller could not tell.
- `tools/call` is NEVER retried. A tool may have side effects, and "the server
  restarted mid-call" is an honest failure the model can reason about.
- A handshake-era connection retries nothing: its session id died with the
  process.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Any, Final

import httpx

from . import wire
from .errors import UNSUPPORTED_PROTOCOL_VERSION, McpAuthRequired, McpError, McpServerGone, McpUnsupportedVersion
from .headers import request_headers
from .protocol import Negotiated, negotiate
from .servers import HttpServer, Server, ServerState, ServerStatus, StdioServer
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
        # Held across the probe, so concurrent first calls share one answer.
        self._connecting = asyncio.Lock()
        self.error: str | None = None
        # Distinct from `error`: nothing is wrong, it has just never been
        # authorized, and the UI offers a different action for it.
        self.needs_auth = False
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
        """What this connection knows. The STATE is the service's to decide —
        only it knows whether a cached listing makes this server usable without
        a live connection."""
        return ServerStatus(
            label=self.label,
            state=ServerState.NEEDS_AUTH if self.needs_auth else ServerState.INACTIVE,
            oauth=getattr(self.server, "oauth", False),
            protocol_version=self._negotiated.protocol_version if self._negotiated else None,
            error=self.error,
        )

    async def connect(self) -> Negotiated:
        """Probe once and cache the answer. Idempotent."""
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
            except McpAuthRequired:
                # Not a failure and not worth a traceback: the user has simply
                # not logged in yet.
                self.needs_auth = True
                raise
            except Exception as exc:
                self.error = str(exc) or type(exc).__name__
                logger.error("mcp server=%s failed to connect", self.label, exc_info=exc)
                raise
            self.error = None
            self.needs_auth = False
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
            logger.info("mcp server=%s dropped during %s; retrying once", self.label, method)
            self._negotiated = None
            return await self._request_once(method, params, timeout_s=timeout_s, extra_headers=extra_headers)
        except McpError as exc:
            if exc.code != UNSUPPORTED_PROTOCOL_VERSION:
                raise
            # A restarted server may have been upgraded under us.
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
        """Every tool the server offers, following `nextCursor` to the end."""
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

        Free on stdio, where the reader already owns the channel. The HTTP
        equivalent needs a `subscriptions/listen` stream and is not implemented;
        TTL-driven refresh covers the same need.
        """
        if message.method == "notifications/tools/list_changed" and self._on_tools_changed is not None:
            logger.info("mcp server=%s reported a tool-list change", self.label)
            self._on_tools_changed(self.label)

    async def aclose(self) -> None:
        self._negotiated = None
        await self._transport.aclose()


# What a pre-2026 server may be asked. Anything else is refused locally rather
# than sent and misread by a server that does not check for `initialize`.
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
