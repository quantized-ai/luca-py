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
from luca.agent.contrib.mcp.servers import ServerState, ServerStatus

from . import state as vm
from .auth import auth_home

# What enter does to a row, per state.
ACTIONS: dict[ServerState, str] = {
    ServerState.CONNECTED: "reconnect",
    ServerState.NEEDS_AUTH: "authenticate",
    ServerState.FAILED: "retry",
    ServerState.DISABLED: "enable",
}


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


def build_mcp_state(statuses: list[ServerStatus], *, selected: int = 0) -> vm.McpState:
    """The `/mcp` screen's state from what the service knows.

    Pure, so the design catalog can render every combination of states without
    a live service or a network.
    """
    rows = [
        vm.McpRow(
            label=status.label,
            state=status.state.value,
            detail=_detail(status),
            action=ACTIONS[status.state],
        )
        for status in statuses
    ]
    notes = [
        f"{status.label}/{tool} excluded: {why}"
        for status in statuses
        for tool, why in sorted(status.rejected_tools.items())
    ]
    live = sum(1 for status in statuses if status.state is ServerState.CONNECTED)
    tools = sum(status.tool_count for status in statuses)
    return vm.McpState(
        count_line=f"{live} of {len(statuses)} connected · {tools} tools",
        rows=rows,
        selected=max(0, min(selected, len(rows) - 1)) if rows else 0,
        notes=notes,
    )


def _detail(status: ServerStatus) -> str:
    match status.state:
        case ServerState.CONNECTED:
            protocol = f" · {status.protocol_version}" if status.protocol_version else ""
            return f"{status.tool_count} tools{protocol}"
        case ServerState.NEEDS_AUTH:
            return "not authenticated"
        case ServerState.DISABLED:
            return "disabled for this session"
    return status.error or "not connected"


__all__ = ["McpConfigError", "build_mcp_service", "build_mcp_state", "mcp_home"]
