"""McpPlugin — bundles the MCP tool registry for `PluginAgentSessionRunner`.

Built in `wiring.build_runner` after the shared `PermissionStrategy` exists, so
MCP tools share the single approval gate with the shell and memory tools.
"""

from __future__ import annotations

from luca.agent.core import AgentSession

from .manager import McpManager
from .registry import McpToolRegistry


class McpPlugin:
    def __init__(self, manager: McpManager, permission_policy) -> None:
        self._manager = manager
        self._policy = permission_policy

    def get_tool_registry(self, agent_session: AgentSession) -> McpToolRegistry:
        return McpToolRegistry(self._manager, self._policy)
