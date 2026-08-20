"""McpPlugin — the registry and a prompt part, bundled for the runner.

Holds two references and no state, because it is rebuilt on every `/clear`,
`/resume` and fork. Everything durable lives on the `McpService`.

IT HAS NO `aclose`, deliberately: `_close_plugins()` runs from `_reset_session`
as well as from quit, so closing connections here would tear every server down
on `/clear`. The application closes the service, from `_quit` only.
"""

from __future__ import annotations

from luca.agent.contrib.simple_tool_registry.permissions import PermissionPolicy
from luca.agent.core import AgentSession, SystemPromptPart

from .registry import McpToolRegistry
from .service import McpService


class McpPlugin:
    def __init__(self, service: McpService, permission_policy: PermissionPolicy) -> None:
        self._service = service
        self._policy = permission_policy

    def get_tool_registry(self, agent_session: AgentSession) -> McpToolRegistry:
        return McpToolRegistry(self._service, self._policy)

    def get_system_prompt_parts(self, agent_session: AgentSession) -> list:
        return [self._prompt_part]

    def _prompt_part(self, session: AgentSession, conversation_id: str) -> SystemPromptPart | None:
        """Name the connected servers, so the model knows where its tools came
        from.

        A callable, so it re-resolves per request: servers connect after the
        first prompt is assembled, and a part fixed at construction would say
        "no MCP servers" for the rest of the session.
        """
        connected = [status for status in self._service.status() if status.tool_count]
        if not connected:
            return None
        lines = [f"- {status.label}: {status.tool_count} tools" for status in connected]
        return SystemPromptPart(
            text="MCP servers connected, exposing their tools as `mcp__<server>__<tool>`:\n" + "\n".join(lines),
            source="mcp",
        )
