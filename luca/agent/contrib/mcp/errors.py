"""Errors raised by the MCP client, and the JSON-RPC codes worth naming."""

from __future__ import annotations

from typing import Final

# Allocated to the MCP spec by the 2026-07-28 error-code policy (-32020 to
# -32099). Their presence in a response is what proves the peer is modern.
HEADER_MISMATCH: Final = -32020
MISSING_REQUIRED_CLIENT_CAPABILITY: Final = -32021
UNSUPPORTED_PROTOCOL_VERSION: Final = -32022

MODERN_PROTOCOL_ERRORS: Final = frozenset(
    {HEADER_MISMATCH, MISSING_REQUIRED_CLIENT_CAPABILITY, UNSUPPORTED_PROTOCOL_VERSION}
)

# Plain JSON-RPC, named because the call path reacts to them: METHOD_NOT_FOUND
# retires a tool from the cached listing, INVALID_PARAMS is the server's own
# argument validation and is passed through verbatim (contract rule 11).
METHOD_NOT_FOUND: Final = -32601
INVALID_PARAMS: Final = -32602


def is_modern_protocol_error(code: int) -> bool:
    """Whether this code proves the peer speaks a modern protocol.

    Only the spec-reserved range counts. `-32601` does NOT: it is the most
    common answer a legacy server gives to `server/discover`.
    """
    return code in MODERN_PROTOCOL_ERRORS


class McpError(Exception):
    """A JSON-RPC error response from a server."""

    def __init__(self, code: int, message: str, data: object = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def __str__(self) -> str:
        return f"{self.message} (code {self.code})"


class McpProtocolError(Exception):
    """The peer broke the wire contract: unparseable frame, a response whose id
    was never sent, a result that fails its schema. Distinct from `McpError`,
    which is the server correctly reporting a failure."""


class McpServerGone(Exception):
    """The connection died with the request in flight.

    Its own type because it is RECOVERABLE for idempotent methods: the protocol
    is stateless, so a restarted server is indistinguishable from the original.
    """


class McpUnsupportedVersion(Exception):
    """No protocol version is common to this client and this server."""

    def __init__(self, message: str, *, supported: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.supported = supported


class McpAuthRequired(Exception):
    """The server answered 401, or refused a token that cannot be refreshed
    without a human."""
