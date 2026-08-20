"""MCP support: connect to external MCP servers and expose their tools.

Speaks the 2026-07-28 revision, which removed the `initialize` handshake and
protocol-level sessions, and falls back to the handshake era for the servers
that have not migrated. The client is written here on `httpx` and `asyncio`
rather than taken from the official SDK, for the same reason `luca.client`
writes its own provider transports: the only new dependency is `mcp-types`
(pydantic and typing-extensions), and owning the connection model is what makes
one long-lived connection per server possible.

The config models are import-safe without the optional group, because
`LucaConfig` names them at module scope and the TUI has to run whether or not
MCP is installed. Everything that touches the wire imports `mcp-types` and is
loaded lazily on first access.
"""

from __future__ import annotations

from .config import McpConfigError, McpServerSettings, McpSettings
from .errors import (
    McpAuthRequired,
    McpError,
    McpProtocolError,
    McpServerGone,
    McpUnsupportedVersion,
)
from .servers import HttpServer, Server, ServerStatus, StdioServer

__all__ = [
    "Era",
    "HttpServer",
    "McpAuthRequired",
    "McpConfigError",
    "McpError",
    "McpProtocolError",
    "McpServerGone",
    "McpServerSettings",
    "McpSettings",
    "McpUnsupportedVersion",
    "Server",
    "ServerStatus",
    "StdioServer",
    "to_execution_result",
    "to_tool_spec",
    "wire_name",
]

_LAZY = {
    "Era": ("wire", "Era"),
    "to_execution_result": ("mapping", "to_execution_result"),
    "to_tool_spec": ("mapping", "to_tool_spec"),
    "wire_name": ("mapping", "wire_name"),
}


def __getattr__(name: str) -> object:
    """Defer the `mcp-types` import to first use of anything that needs it."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    module = __import__(f"{__name__}.{module_name}", fromlist=[attribute])
    return getattr(module, attribute)
