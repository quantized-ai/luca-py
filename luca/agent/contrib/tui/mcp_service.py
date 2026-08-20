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
    ServerState.CONNECTED: "tools",
    ServerState.STALE: "tools",
    ServerState.CONNECTING: "wait",
    ServerState.NEEDS_AUTH: "authenticate",
    ServerState.INACTIVE: "retry",
    ServerState.DISABLED: "enable",
}

# States whose tools are on offer to the model.
_WORKING = (ServerState.CONNECTED, ServerState.STALE)


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


def build_mcp_state(
    statuses: list[ServerStatus],
    *,
    selected: int = 0,
    message: str | None = None,
    message_is_error: bool = False,
    busy: tuple[str, str] | None = None,
    detail: vm.McpDetail | None = None,
) -> vm.McpState:
    """The `/mcp` screen's state from what the service knows.

    Pure, so the design catalog can render every combination of states without
    a live service or a network. `busy` is `(label, what is happening to it)`
    for the one action that can be in flight.
    """
    rows = [
        vm.McpRow(
            label=status.label,
            state=status.state.value,
            detail=busy[1] if busy and busy[0] == status.label else _detail(status),
            action=ACTIONS[status.state],
            busy=bool(busy and busy[0] == status.label),
        )
        for status in statuses
    ]
    live = sum(1 for status in statuses if status.state is ServerState.CONNECTED)
    tools = sum(status.tool_count for status in statuses)
    return vm.McpState(
        count_line=f"{live} of {len(statuses)} connected · {_tools(tools)}",
        rows=rows,
        selected=max(0, min(selected, len(rows) - 1)) if rows else 0,
        message=message,
        message_is_error=message_is_error,
        detail=detail,
    )


def build_mcp_detail(status: ServerStatus, specs) -> vm.McpDetail:
    """One server's tools, offered first and refused after. The refused ones
    belong here rather than under the server list: a reason a tool is missing
    only means something beside the tools that are not."""
    rows = [
        vm.McpToolRow(name=_remote_name(spec), summary=_summary(spec.description))
        for spec in sorted(specs, key=_remote_name)
    ]
    rows += [vm.McpToolRow(name=name, summary="", excluded=why) for name, why in sorted(status.rejected_tools.items())]
    excluded = f" · {len(status.rejected_tools)} excluded" if status.rejected_tools else ""
    return vm.McpDetail(label=status.label, count_line=f"{_tools(len(specs))}{excluded}", rows=rows)


def mcp_hints(state: vm.McpState) -> list[str]:
    """What the keys do here, read off the selected row's own action: a legend
    saying `d disable` over a disabled server is worse than no legend."""
    if state.detail is not None:
        return ["↑↓ move", "esc back"]
    row = state.rows[state.selected] if state.rows else None
    if row is None:
        return ["esc back"]
    if row.busy:
        return ["esc cancel"]
    if row.state == "disabled":
        return ["↑↓ move", f"enter {row.action}", "esc back"]
    return ["↑↓ move", f"enter {row.action}", "a authenticate", "r reconnect", "d disable", "esc back"]


def _remote_name(spec) -> str:
    """The name the SERVER calls this tool. `spec.name` is the wire name, which
    is prefixed and may have been truncated to fit a provider's cap."""
    entry = (spec.metadata or {}).get("mcp")
    remote = entry.get("tool") if isinstance(entry, dict) else None
    return remote if isinstance(remote, str) else spec.name


def _summary(description: str) -> str:
    return description.strip().splitlines()[0] if description.strip() else ""


def _detail(status: ServerStatus) -> str:
    match status.state:
        case ServerState.CONNECTED:
            protocol = f" · {status.protocol_version}" if status.protocol_version else ""
            excluded = f" · {len(status.rejected_tools)} excluded" if status.rejected_tools else ""
            return f"{_tools(status.tool_count)}{excluded}{protocol}"
        case ServerState.STALE:
            # The count leads: those tools are still on offer.
            return f"{_tools(status.tool_count)} · last refresh failed: {status.error}"
        case ServerState.CONNECTING:
            return "connecting…"
        case ServerState.NEEDS_AUTH:
            return "not authenticated"
        case ServerState.DISABLED:
            return "disabled for this session"
    return status.error or "not connected yet"


def _tools(count: int) -> str:
    return "1 tool" if count == 1 else f"{count} tools"


def mcp_boot_notice(statuses: list[ServerStatus]) -> str | None:
    """The one line the transcript gets at startup, or None when there is
    nothing to say.

    Leads with what worked and trails what still needs doing, in one colour: a
    server nobody mentions is one nobody knows to go and fix, and a red line at
    boot is how every other line stops being read.
    """
    connected = [status for status in statuses if status.state in _WORKING]
    waiting = [status.label for status in statuses if status.state is ServerState.NEEDS_AUTH]
    failed = [status.label for status in statuses if status.state is ServerState.INACTIVE]
    if not connected and not waiting and not failed:
        return None
    head = (
        "MCP connected: " + ", ".join(f"{status.label} ({_tools(status.tool_count)})" for status in connected)
        if connected
        else "MCP: nothing connected"
    )
    trailing = []
    if waiting:
        trailing.append(f"{', '.join(waiting)} needs a login")
    if failed:
        trailing.append(f"{', '.join(failed)} failed")
    return f"{head} · {' · '.join(trailing)} — /mcp" if trailing else head


__all__ = [
    "McpConfigError",
    "build_mcp_detail",
    "build_mcp_service",
    "build_mcp_state",
    "mcp_boot_notice",
    "mcp_hints",
    "mcp_home",
]
