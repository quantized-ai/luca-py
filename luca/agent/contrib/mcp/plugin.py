"""McpPlugin — bundles the MCP tool registry for `PluginAgentSessionRunner`.

Built in `wiring.build_runner` after the shared `PermissionStrategy` exists, so
MCP tools share the single approval gate with the shell and memory tools. It
holds only config: the registry it hands back connects to servers per call, so
there is nothing to start or stop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from luca.agent.core import AgentSession

from .config import McpServerDef
from .registry import McpToolRegistry

if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx

    from luca.agent.contrib.simple_tool_registry.permissions import PermissionPolicy

    from .config import HttpServer


class McpPlugin:
    def __init__(
        self,
        servers: dict[str, McpServerDef],
        permission_policy: PermissionPolicy,
        *,
        auth_factory: Callable[[str, HttpServer], httpx.Auth | None] | None = None,
    ) -> None:
        self._servers = servers
        self._policy = permission_policy
        self._auth_factory = auth_factory

    def get_tool_registry(self, agent_session: AgentSession) -> McpToolRegistry:
        return McpToolRegistry(self._servers, self._policy, auth_factory=self._auth_factory)
