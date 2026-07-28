"""Sub-agents — delegate read-only research to background sub-agents.

The `task` tool (installed by `SubAgentPlugin`) lets the parent agent spawn one
of the built-in read-only sub-agent types. Each runs as its own
`AgentSessionRunner` over its own `AgentSession`, managed by `SubAgentManager`,
which drains it in the background and streams `SubAgentTask` status/result
snapshots. See `docs/agent/contrib/subagents/README.md`.
"""

from __future__ import annotations

from .manager import (
    RunnerFactory,
    SubAgentManager,
    SubAgentTask,
    TaskStatus,
)
from .plugin import SubAgentPlugin
from .tool import SpawnSubAgentTool
from .tools import build_readonly_registry, readonly_tools
from .types import BUILTIN_AGENT_TYPES, SubAgentType

__all__ = [
    "BUILTIN_AGENT_TYPES",
    "RunnerFactory",
    "SpawnSubAgentTool",
    "SubAgentManager",
    "SubAgentPlugin",
    "SubAgentTask",
    "SubAgentType",
    "TaskStatus",
    "build_readonly_registry",
    "readonly_tools",
]
