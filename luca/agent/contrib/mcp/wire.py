"""JSON-RPC framing for both MCP protocol eras, and result validation.

The only module that imports `mcp-types`, so everything above it can be read and
tested without the optional group installed.

THE TWO ERAS. Up to 2025-11-25 a connection began with an `initialize`
handshake and later requests carried nothing about the protocol. From 2026-07-28
there is no handshake, so every request carries its own version, capabilities
and identity in `params._meta`. That is the whole difference here, and why
`request_frame` takes an era.

Result validation is version-gated by `mcp_types.methods`, keyed on `(method,
version)`, so asking a 2025-era server for `server/discover` fails locally.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import Enum
from typing import Any, Final

try:
    import mcp_types
    from mcp_types.jsonrpc import (
        JSONRPCError,
        JSONRPCMessage,
        JSONRPCNotification,
        JSONRPCResponse,
        jsonrpc_message_adapter,
    )
    from mcp_types.methods import CACHEABLE_METHODS, parse_server_result
    from mcp_types.version import (
        HANDSHAKE_PROTOCOL_VERSIONS,
        LATEST_HANDSHAKE_VERSION,
        LATEST_MODERN_VERSION,
        MODERN_PROTOCOL_VERSIONS,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("luca.agent.contrib.mcp needs mcp-types: install the `mcp` dependency group (uv sync).") from exc

from pydantic import ValidationError

from luca import __version__

from .errors import McpError, McpProtocolError

# The `_meta` keys the 2026-07-28 revision defines for per-request metadata.
META_PROTOCOL_VERSION: Final = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_CAPABILITIES: Final = "io.modelcontextprotocol/clientCapabilities"
META_CLIENT_INFO: Final = "io.modelcontextprotocol/clientInfo"

CLIENT_INFO: Final = {"name": "luca", "version": __version__}

# Empty, and honestly so. Roots, sampling and elicitation are all deprecated or
# unsupported here, and claiming a capability we do not implement invites a
# server to hand back an `InputRequiredResult` we cannot answer.
CLIENT_CAPABILITIES: Final[dict[str, Any]] = {}

PREFERRED_MODERN_VERSION: Final = LATEST_MODERN_VERSION
PREFERRED_HANDSHAKE_VERSION: Final = LATEST_HANDSHAKE_VERSION

# Re-exported so the layers above negotiate against one list and never import
# `mcp_types` themselves.
MODERN_VERSIONS: Final = MODERN_PROTOCOL_VERSIONS
HANDSHAKE_VERSIONS: Final = HANDSHAKE_PROTOCOL_VERSIONS


class Era(str, Enum):
    """Which shape of the protocol a server speaks. Two values rather than a
    version string: every pre-2026 version is framed identically."""

    MODERN = "modern"
    HANDSHAKE = "handshake"

    @classmethod
    def of(cls, protocol_version: str) -> Era:
        if protocol_version in MODERN_PROTOCOL_VERSIONS:
            return cls.MODERN
        if protocol_version in HANDSHAKE_PROTOCOL_VERSIONS:
            return cls.HANDSHAKE
        raise McpProtocolError(f"Unknown MCP protocol version {protocol_version!r}.")


def request_frame(
    request_id: int,
    method: str,
    params: Mapping[str, Any] | None = None,
    *,
    protocol_version: str,
    era: Era,
) -> dict[str, Any]:
    """One JSON-RPC request, framed for the era the server speaks."""
    frame: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    body = _with_meta(params, protocol_version=protocol_version, era=era)
    if body is not None:
        frame["params"] = body
    return frame


def notification_frame(
    method: str,
    params: Mapping[str, Any] | None = None,
    *,
    protocol_version: str,
    era: Era,
) -> dict[str, Any]:
    """One JSON-RPC notification. Used for `notifications/cancelled` on stdio
    and for `notifications/initialized` on a handshake connection."""
    frame: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    body = _with_meta(params, protocol_version=protocol_version, era=era)
    if body is not None:
        frame["params"] = body
    return frame


def _with_meta(
    params: Mapping[str, Any] | None,
    *,
    protocol_version: str,
    era: Era,
) -> dict[str, Any] | None:
    """Params with the modern `_meta` block merged in, or untouched on a
    handshake connection. An existing `_meta` is extended, never replaced."""
    if era is not Era.MODERN:
        return dict(params) if params is not None else None
    body = dict(params) if params is not None else {}
    meta = dict(body.get("_meta") or {})
    meta[META_PROTOCOL_VERSION] = protocol_version
    meta[META_CLIENT_CAPABILITIES] = CLIENT_CAPABILITIES
    meta[META_CLIENT_INFO] = CLIENT_INFO
    body["_meta"] = meta
    return body


def encode(frame: Mapping[str, Any]) -> bytes:
    """One frame as a stdio line. The spec forbids an embedded newline in a
    stdio message, which compact JSON with `ensure_ascii` guarantees."""
    return json.dumps(frame, separators=(",", ":")).encode("utf-8") + b"\n"


def decode(raw: str | bytes) -> JSONRPCMessage:
    """One inbound frame as a typed JSON-RPC message.

    Raises `McpProtocolError`, not `McpError`: the peer has not reported a
    failure, it has broken the wire contract.
    """
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        excerpt = raw[:120] if isinstance(raw, str) else raw[:120].decode("utf-8", errors="replace")
        raise McpProtocolError(f"MCP server produced non-JSON output: {excerpt!r}") from exc
    try:
        return jsonrpc_message_adapter.validate_python(payload)
    except ValidationError as exc:
        raise McpProtocolError(f"MCP server produced a frame that is not JSON-RPC: {payload!r}") from exc


def result_payload(message: JSONRPCResponse | JSONRPCError) -> dict[str, Any]:
    """The `result` of a response, or the error raised."""
    if isinstance(message, JSONRPCError):
        raise McpError(message.error.code, message.error.message, message.error.data)
    return dict(message.result)


def parse_result(method: str, protocol_version: str, payload: Mapping[str, Any]) -> mcp_types.Result:
    """A raw result validated against the surface for this method and version.

    A method the version does not define and a payload that does not fit both
    mean the same thing: this server did not answer the question asked. Modern
    results are filled in first; see `_tolerated`.
    """
    if Era.of(protocol_version) is Era.MODERN:
        payload = _tolerated(method, payload)
    try:
        return parse_server_result(method, protocol_version, payload)
    except (KeyError, ValueError, ValidationError) as exc:
        raise McpProtocolError(
            f"MCP server returned an invalid {method!r} result for protocol {protocol_version}: {exc}"
        ) from exc


def _tolerated(method: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """A modern result with the required-but-unread fields filled in.

    The 2026 schema requires `resultType` everywhere and `ttlMs` / `cacheScope`
    on cacheable results. Losing a whole tool listing over a freshness hint this
    client does not depend on is the wrong trade, so ABSENCE is interpreted and
    only absence — a stated value is always believed.

    Absent `resultType` means `"complete"`, which is the spec's own rule for
    clients. Absent `cacheScope` means `"private"`, the conservative reading.
    Absent `ttlMs` means `0`, read the same way as a stated `0`: refresh on our
    own schedule. The cache exists either way, because rule 3 requires it.
    """
    filled = dict(payload)
    filled.setdefault("resultType", "complete")
    if method in CACHEABLE_METHODS:
        filled.setdefault("ttlMs", 0)
        filled.setdefault("cacheScope", "private")
    return filled


def is_notification(message: JSONRPCMessage) -> bool:
    return isinstance(message, JSONRPCNotification)
