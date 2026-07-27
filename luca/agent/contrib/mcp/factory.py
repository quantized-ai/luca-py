"""build_manager — turn config server definitions into an McpManager.

Wires the OAuth auth factory only when an HTTP server asks for it, so the OAuth
code path (and its browser/callback machinery) is untouched otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import HttpServer, McpServerDef
from .manager import McpManager


def _oauth_store_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "luca"


def build_manager(
    servers: dict[str, McpServerDef],
    *,
    oauth_store_dir: Path | None = None,
) -> McpManager | None:
    """An McpManager over the enabled servers, or None if there are none."""
    enabled = {name: server for name, server in servers.items() if server.enabled}
    if not enabled:
        return None
    auth_factory = None
    if any(isinstance(server, HttpServer) and server.oauth for server in enabled.values()):
        from .oauth import make_auth_factory

        auth_factory = make_auth_factory(oauth_store_dir or _oauth_store_dir())
    return McpManager(enabled, auth_factory=auth_factory)
