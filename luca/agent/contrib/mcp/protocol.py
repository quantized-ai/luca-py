"""Deciding which protocol a server speaks, once per connection.

`server/discover` is sent first and the answer read three ways: a
`DiscoverResult` means modern; a spec-reserved error code means modern but
version-mismatched, so retry rather than fall back; anything else, `-32601` and
a timeout included, means legacy and the `initialize` handshake.

The spec is explicit that the fallback "MUST NOT be keyed to one specific error
code" — a legacy server answers an unknown pre-`initialize` request with
whatever it happens to return — so the discriminator is membership in the
reserved range and everything else is evidence of age.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from . import wire
from .errors import McpError, McpProtocolError, McpUnsupportedVersion, is_modern_protocol_error
from .headers import request_headers
from .wire import Era

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Negotiated:
    """What one probe settled. Thrown away on reconnect, because a restarted
    server may have been upgraded."""

    era: Era
    protocol_version: str
    capabilities: dict[str, Any] = field(default_factory=dict)
    server_info: dict[str, Any] | None = None
    instructions: str | None = None


async def negotiate(transport, *, label: str, timeout_s: float, next_id) -> Negotiated:
    """Probe once and settle on an era and a version.

    `next_id` is the connection's id source, so a probe and the requests after
    it never reuse an id on a shared stdio channel.
    """
    try:
        return await _discover(transport, wire.PREFERRED_MODERN_VERSION, timeout_s=timeout_s, next_id=next_id)
    except _Retry as retry:
        # The server is modern and named a version we speak. Ask again in that
        # version rather than falling back, which the spec forbids here.
        logger.debug("mcp server=%s asked for protocol %s", label, retry.version)
        try:
            return await _discover(transport, retry.version, timeout_s=timeout_s, next_id=next_id)
        except _Retry as again:
            # Twice is enough, and an internal marker must not escape.
            raise McpUnsupportedVersion(
                f"MCP server {label!r} keeps asking for a different protocol version ({again.version}).",
                supported=(again.version,),
            ) from again
    except _NotModern as reason:
        logger.debug("mcp server=%s is not modern (%s); falling back to initialize", label, reason)
    return await _handshake(transport, timeout_s=timeout_s, next_id=next_id)


class _NotModern(Exception):
    """Internal: the probe found no evidence of a modern server."""


async def _discover(transport, version: str, *, timeout_s: float, next_id) -> Negotiated:
    frame = wire.request_frame(next_id(), "server/discover", None, protocol_version=version, era=Era.MODERN)
    headers = _headers("server/discover", None, version)
    try:
        message = await transport.request(frame, headers=headers, timeout_s=timeout_s)
        payload = wire.result_payload(message)
    except McpError as exc:
        if not is_modern_protocol_error(exc.code):
            raise _NotModern(f"error {exc.code}") from exc
        raise _mismatch(exc) from exc
    except McpProtocolError as exc:
        raise _NotModern("unparseable answer") from exc
    except TimeoutError as exc:
        raise _NotModern("no answer") from exc

    result = wire.parse_result("server/discover", version, payload)
    chosen = _mutual(list(result.supported_versions))
    return Negotiated(
        era=Era.MODERN,
        protocol_version=chosen,
        capabilities=result.capabilities.model_dump(mode="json", by_alias=True, exclude_none=True),
        server_info=None,
        instructions=result.instructions,
    )


def _mismatch(exc: McpError) -> Exception:
    """A modern server that will not speak the version we opened with. Returns
    the exception to raise rather than raising, so the caller keeps `from exc`."""
    advertised = exc.data.get("supported") if isinstance(exc.data, dict) else None
    if isinstance(advertised, list):
        try:
            return _Retry(_mutual([str(version) for version in advertised]))
        except McpUnsupportedVersion as unsupported:
            return unsupported
    return McpUnsupportedVersion(str(exc), supported=())


class _Retry(Exception):
    """Internal: the server named a version we can speak."""

    def __init__(self, version: str) -> None:
        super().__init__(version)
        self.version = version


def _mutual(advertised: list[str]) -> str:
    """The best version both sides speak. Modern beats handshake, because a
    server offering both is better served by the era that needs no session."""
    for candidate in (*wire.MODERN_VERSIONS, *reversed(wire.HANDSHAKE_VERSIONS)):
        if candidate in advertised:
            return candidate
    raise McpUnsupportedVersion(
        f"No common MCP protocol version. The server offers {', '.join(advertised) or 'nothing'}.",
        supported=tuple(advertised),
    )


async def _handshake(transport, *, timeout_s: float, next_id) -> Negotiated:
    """The pre-2026 `initialize` exchange. Our version is an opening bid; the
    server's answer is authoritative and may legitimately be older."""
    proposed = wire.PREFERRED_HANDSHAKE_VERSION
    params = {
        "protocolVersion": proposed,
        "capabilities": wire.CLIENT_CAPABILITIES,
        "clientInfo": wire.CLIENT_INFO,
    }
    frame = wire.request_frame(next_id(), "initialize", params, protocol_version=proposed, era=Era.HANDSHAKE)
    message = await transport.request(frame, headers=_headers("initialize", params, proposed), timeout_s=timeout_s)
    result = wire.parse_result("initialize", proposed, wire.result_payload(message))

    agreed = result.protocol_version
    if agreed not in wire.HANDSHAKE_VERSIONS and agreed not in wire.MODERN_VERSIONS:
        raise McpUnsupportedVersion(
            f"The server answered `initialize` with protocol {agreed!r}, which this client does not speak.",
            supported=(agreed,),
        )
    # Not usable until the server is told the client is ready.
    await transport.notify(
        wire.notification_frame("notifications/initialized", None, protocol_version=agreed, era=Era.HANDSHAKE),
        headers=_headers("notifications/initialized", None, agreed),
    )
    return Negotiated(
        era=Era.HANDSHAKE,
        protocol_version=agreed,
        capabilities=result.capabilities.model_dump(mode="json", by_alias=True, exclude_none=True),
        server_info=result.server_info.model_dump(mode="json", by_alias=True, exclude_none=True),
        instructions=result.instructions,
    )


def _headers(method: str, params: dict | None, version: str) -> dict[str, str]:
    """Standard headers for the HTTP transport, ignored by stdio. Computed from
    the same method and params that built the frame, because a server must
    reject a request whose headers disagree with its body."""
    return request_headers(method, params, protocol_version=version)
