"""luca.agent.contrib.plugins — capability bundles over the core runner.

A plugin bundles a tool registry, system-prompt parts, and middleware behind
one object; `PluginAgentSessionRunner` composes them at construction. The
core runner knows nothing about plugins — this package is the composition
layer, built (like any application) on core's public surface plus
`contrib.simple_tool_registry`'s `ProxyToolRegistry`.
"""

from .plugin import BasePlugin
from .runner import PluginAgentSessionRunner

__all__ = [
    "BasePlugin",
    "PluginAgentSessionRunner",
]
