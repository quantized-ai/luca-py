"""Turning the `mcp` config block into a live service.

Lives on the TUI side of the line rather than in `luca.agent.contrib.mcp`,
because deciding WHERE state goes is an application question: the package takes
paths, it does not go looking for them. That is also what keeps the MCP package
free of any import of this one, whose package root pulls in Textual.

Both files sit beside `auth.json` under `~/.local/share/luca/mcp/`, which is
data luca writes rather than configuration a person edits. The tool catalog is
a cache; the token file is rewritten on every refresh. Neither belongs in
`~/.config`.
"""

from __future__ import annotations

from pathlib import Path

from luca.agent.contrib.mcp.config import McpConfigError, McpSettings

from .auth import auth_home


def mcp_home() -> Path:
    return auth_home() / "mcp"


def build_mcp_service(settings: McpSettings | None, *, home: Path | None = None):
    """An `McpService` over the enabled servers, or None when there are none.

    None rather than an empty service, so the plugin is never installed and a
    user who configured nothing pays nothing — no prompt part, no registry, no
    `httpx.AsyncClient`.

    Raises `McpConfigError` for a server that cannot be resolved (an undefined
    `${VAR}`, an illegal label). The CLI catches it beside `LucaConfigError`,
    so a typo is a readable startup failure rather than a stack trace.
    """
    if settings is None or settings.enabled is False:
        return None
    servers = settings.build()
    if not servers:
        return None
    from luca.agent.contrib.mcp.service import McpService

    root = home or mcp_home()
    return McpService(servers, catalog_path=root / "catalog.json", token_path=root / "mcp-auth.json")


__all__ = ["McpConfigError", "build_mcp_service", "mcp_home"]
