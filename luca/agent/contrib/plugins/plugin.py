"""Agent plugins — capability bundles for `PluginAgentSessionRunner`.

A plugin bundles the pieces an agent capability usually ships together — a
tool registry, system-prompt parts, and middleware — behind one object, so an
application installs the capability with a single `plugins=[...]` item
instead of composing three collaborators by hand. The core runner knows
nothing about plugins: `PluginAgentSessionRunner` (runner.py in this package)
is the composition layer, and it is pure construction-time sugar — a runner
built with a plugin is equivalent to one built with the same objects passed
directly.

`BasePlugin` is a reference base class — subclassing it is optional and
discouraged. Hooks are duck-typed: if a plugin object implements a hook, it
is called; if not, `hasattr` skips it. Any plain Python class that defines
the hooks it needs is a valid plugin, and a plain class is the recommended
way to write one (matching the middleware convention).

Every hook receives the `AgentSession` the runner is being constructed over;
`None` is treated as an empty contribution.
"""

from __future__ import annotations

from luca.agent.core import AgentSession, ToolRegistry
from luca.agent.core.system_prompt import SystemPromptPartInput


class BasePlugin:
    def get_tool_registry(
        self, agent_session: AgentSession,
    ) -> ToolRegistry | None:
        """The registry bundling this plugin's tools — added as a child of
        the runner's composed `ProxyToolRegistry`. A multi-registry plugin
        returns its own proxy; `None` contributes no tools."""
        return None

    def get_system_prompt_parts(
        self, agent_session: AgentSession,
    ) -> list[SystemPromptPartInput]:
        """Parts to append to `system_prompt_parts` — any form the runner
        constructor accepts (str / dict / `SystemPromptPart` / callable),
        coerced the same way."""
        return []

    def get_middleware(self, agent_session: AgentSession) -> list:
        """Middleware instances to append to the runner's `middleware` list."""
        return []
