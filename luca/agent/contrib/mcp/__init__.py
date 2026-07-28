"""MCP support: connect to external MCP servers and expose their tools.

The config types (`McpServerDef`, `StdioServer`, `HttpServer`) are import-safe
without the `mcp` SDK, so `LucaConfig` and the TUI parse and run without the
optional extra. Everything that actually talks to a server (`McpToolRegistry`,
`McpPlugin`, `build_mcp_plugin`) imports the SDK and is loaded lazily on first
access. Connections are opened per call, so there is no manager to own.
"""

from __future__ import annotations

from .config import HttpServer, McpServerDef, StdioServer

__all__ = [
    "McpServerDef",
    "StdioServer",
    "HttpServer",
    "McpToolRegistry",
    "McpPlugin",
    "build_mcp_plugin",
]


def __getattr__(name: str):  # lazy: import the SDK only when these are used
    if name == "McpToolRegistry":
        from .registry import McpToolRegistry

        return McpToolRegistry
    if name == "McpPlugin":
        from .plugin import McpPlugin

        return McpPlugin
    if name == "build_mcp_plugin":
        from .factory import build_mcp_plugin

        return build_mcp_plugin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
