"""MCP support: connect to external MCP servers and expose their tools.

The config types (`McpServerDef`) and `McpTool` are import-safe without the
`mcp` SDK, so `LucaConfig` and the TUI parse and run without the optional
extra. Everything that actually talks to a server (`McpManager`, `McpPlugin`,
`build_manager`) imports the SDK and is loaded lazily on first access.
"""

from __future__ import annotations

from .config import HttpServer, McpServerDef, StdioServer
from .tool import McpTool

__all__ = [
    "McpServerDef",
    "StdioServer",
    "HttpServer",
    "McpTool",
    "McpManager",
    "McpPlugin",
    "build_manager",
]


def __getattr__(name: str):  # lazy: import the SDK only when these are used
    if name == "McpManager":
        from .manager import McpManager

        return McpManager
    if name == "McpPlugin":
        from .plugin import McpPlugin

        return McpPlugin
    if name == "build_manager":
        from .factory import build_manager

        return build_manager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
