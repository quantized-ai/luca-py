"""`SubAgentPlugin` — installs the `task` tool, gated by the shared strategy.

A plain class implementing the plugin hooks (`get_tool_registry`,
`get_system_prompt_parts`); pass it as `plugins=[...]` to
`PluginAgentSessionRunner`. The `task` tool is gated by the same
`PermissionPolicy` every other registry shares, so one approval gate serves it
too.
"""

from __future__ import annotations

from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry
from luca.agent.contrib.simple_tool_registry.permissions import PermissionPolicy
from luca.agent.core import AgentSession

from .manager import SubAgentManager
from .tool import SpawnSubAgentTool

_SYSTEM_PROMPT = (
    "### Sub-agents\n"
    "You can delegate read-only research to background sub-agents with the "
    "`task` tool. Use it for self-contained exploration — mapping a subsystem, "
    "locating code, answering a focused question — and spawn several at once "
    "for independent questions. Each sub-agent reads with read/glob/grep in its "
    "own context and cannot modify anything; its findings are delivered back to "
    "you when it finishes."
)


class SubAgentPlugin:
    def __init__(self, manager: SubAgentManager, permission_policy: PermissionPolicy) -> None:
        self._manager = manager
        self._permission_policy = permission_policy

    def get_tool_registry(self, agent_session: AgentSession) -> SimpleToolRegistry:
        return SimpleToolRegistry(
            tools=[SpawnSubAgentTool(self._manager)],
            permission_policy=self._permission_policy,
        )

    def get_system_prompt_parts(self, agent_session: AgentSession) -> list[str]:
        return [_SYSTEM_PROMPT]
