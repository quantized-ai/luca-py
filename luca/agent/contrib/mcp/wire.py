"""JSON-RPC framing for both MCP protocol eras, and result validation.

This is the only module that imports `mcp-types`, so everything above it can be
read and tested without the optional group installed. What we take from that
package is narrow on purpose: the generated wire models and the per-version
surface map, which together answer "is this a valid `tools/list` result for the
version this server speaks". Everything about how a frame is BUILT lives here,
because that is where the two eras differ and where our own decisions are.

THE TWO ERAS, IN ONE PARAGRAPH. Up to 2025-11-25 a connection began with an
`initialize` handshake and every later request inherited the version agreed
there; the request itself carried nothing about the protocol. From 2026-07-28
there is no handshake and no session, so every request carries its own protocol
version, client capabilities and client identity in `params._meta`. That is the
whole difference at this layer, and it is why `request_frame` takes an era: the
same method and params produce a different frame depending on which one the
server answered the probe with.

Result VALIDATION is version-gated by `mcp_types.methods`, which keys its
surface map on `(method, version)`. A method a version does not define is
absent from the map, so asking a 2025-era server for `server/discover` fails
here rather than at the far end, and a 2026-only field arriving from a legacy
server is rejected rather than quietly believed.
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


class Era(str, Enum):
    """Which shape of the protocol a server speaks.

    Two values rather than a version string because the version alone does not
    tell the framing layer what to do, and every pre-2026 version is framed
    identically from a client's point of view.
    """

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
    handshake connection where the block does not exist.

    An existing `_meta` is preserved and extended rather than replaced: a
    caller may already have put a progress token or a trace context there.
    """
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

    A frame that is not JSON, or is JSON that is not JSON-RPC, raises
    `McpProtocolError`. That is deliberately not `McpError`: the peer has not
    reported a failure, it has broken the wire contract.
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
    """The `result` of a response, or the error raised.

    Every JSON-RPC error the client sees funnels through here, which is what
    lets the call path react to `code` without every caller unpacking a frame.
    """
    if isinstance(message, JSONRPCError):
        raise McpError(message.error.code, message.error.message, message.error.data)
    return dict(message.result)


def parse_result(method: str, protocol_version: str, payload: Mapping[str, Any]) -> mcp_types.Result:
    """A raw result validated against the surface for this method and version.

    Wraps two failures into one: a method the version does not define (absent
    from the surface map) and a payload that does not fit the schema. Both mean
    the same thing to a caller, which is that this server did not answer the
    question that was asked.

    Modern results are filled in before validating; see `_tolerated`.
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

    The 2026 schema makes `resultType` required on every result, and `ttlMs`
    and `cacheScope` required on the cacheable ones. Validating strictly would
    mean losing a server's whole tool listing over a freshness hint, which is a
    bad trade for fields this client does not depend on. So absence is
    interpreted, and only absence: a server that states a value is believed.

    - `resultType` absent means `"complete"`. That is the spec's own rule for
      clients, written for earlier-protocol servers and applied here to a
      modern server that simply left it out.
    - `cacheScope` absent means `"private"`, the conservative reading: a
      listing is never shared across credentials unless the server says it may
      be.
    - `ttlMs` absent means `0`, which the catalog reads the same way it reads a
      stated `0`: no useful freshness hint, so refresh on the client's own
      schedule. The two cannot be told apart afterwards, and nothing is lost by
      that, because a registry fronting a remote server has to keep a cached
      listing whatever the server would prefer (contract rule 3). The hint sets
      how eagerly that cache is refreshed, never whether it exists.
    """
    filled = dict(payload)
    filled.setdefault("resultType", "complete")
    if method in CACHEABLE_METHODS:
        filled.setdefault("ttlMs", 0)
        filled.setdefault("cacheScope", "private")
    return filled


def is_notification(message: JSONRPCMessage) -> bool:
    return isinstance(message, JSONRPCNotification)
