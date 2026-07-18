"""luca.agent.contrib.memory — agent memory, packaged as a plugin.

Two in-memory capabilities: a scratchpad — the agent's private working
memory, read and written through two tools, each write fully replacing the
content — and a todo list — its task tracker, updated whole-list-at-a-time
through `update_todos`. `MemoryPlugin` bundles the tools (in their own
auto-allowing registry) with the system-prompt parts that teach the model to
use them; install it with `plugins=[MemoryPlugin()]` on a
`PluginAgentSessionRunner` (`luca.agent.contrib.plugins`).
"""

from .plugin import (
    MemoryPlugin,
    ReadScratchPadTool,
    ReadTodoTool,
    TodoItem,
    TodoStatus,
    UpdateTodosTool,
    WriteScratchPadTool,
)

__all__ = [
    "MemoryPlugin",
    "ReadScratchPadTool",
    "ReadTodoTool",
    "TodoItem",
    "TodoStatus",
    "UpdateTodosTool",
    "WriteScratchPadTool",
]
