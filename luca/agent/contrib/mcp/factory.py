"""build_mcp_plugin — turn config server definitions into an McpPlugin.

Wires the OAuth auth factory only when an HTTP server asks for it, so the OAuth
code path (and its browser/callback machinery) is untouched otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import HttpServer, McpServerDef
from .plugin import McpPlugin


def _oauth_store_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "luca"


def build_mcp_plugin(
    servers: dict[str, McpServerDef],
    permission_policy,
    *,
    oauth_store_dir: Path | None = None,
) -> McpPlugin | None:
    """An McpPlugin over the enabled servers, or None if there are none."""
    enabled = {label: server for label, server in servers.items() if server.enabled}
    if not enabled:
        return None
    auth_factory = None
    if any(isinstance(server, HttpServer) and server.oauth for server in enabled.values()):
        from .oauth import make_auth_factory

        auth_factory = make_auth_factory(oauth_store_dir or _oauth_store_dir())
    return McpPlugin(enabled, permission_policy, auth_factory=auth_factory)
