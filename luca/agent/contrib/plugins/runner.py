"""PluginAgentSessionRunner — plugin composition over the core runner.

Construction-time sugar only: each plugin hook runs ONCE here, and its result
extends the matching collaborator — in plugin order, after the
directly-passed items. Tool registries compose through one
`ProxyToolRegistry` (the directly-passed registry first, then one child per
plugin); prompt parts and middleware extend the constructor lists (plugin
parts coerce exactly like constructor parts, in the base `__init__`). A
runner built with plugins is equivalent to — and compares equal to — one
built with the same objects composed directly.
"""

from __future__ import annotations

from luca.agent.core import AgentSession, AgentSessionRunner, ToolRegistry
from luca.agent.contrib.simple_tool_registry import ProxyToolRegistry


class PluginAgentSessionRunner(AgentSessionRunner):
    """`AgentSessionRunner` that accepts `plugins=[...]` and composes their
    contributions at construction (see the module docstring)."""

    def __init__(
        self,
        session: AgentSession,
        tool_registry: ToolRegistry | None = None,
        system_prompt_parts: list | None = None,
        system_prompt_assembler=None,
        *,
        provider=None,
        conversation_projector=None,
        context_manager=None,
        middleware: list | None = None,
        plugins: list | None = None,
    ) -> None:
        plugins = list(plugins or [])
        proxy = ProxyToolRegistry()
        if tool_registry is not None:
            proxy.add_registry(tool_registry)
        system_prompt_parts = list(system_prompt_parts or [])
        middleware = list(middleware or [])
        for plugin in plugins:
            if hasattr(plugin, "get_tool_registry"):
                registry = plugin.get_tool_registry(session)
                if registry is not None:
                    proxy.add_registry(registry)
            if hasattr(plugin, "get_system_prompt_parts"):
                system_prompt_parts += (
                    plugin.get_system_prompt_parts(session) or []
                )
            if hasattr(plugin, "get_middleware"):
                middleware += plugin.get_middleware(session) or []
        super().__init__(
            session,
            tool_registry=proxy,
            system_prompt_parts=system_prompt_parts,
            system_prompt_assembler=system_prompt_assembler,
            provider=provider,
            conversation_projector=conversation_projector,
            context_manager=context_manager,
            middleware=middleware,
        )
        self.plugins = plugins
