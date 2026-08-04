"""luca.agent.contrib.memory — agent memory, packaged as a plugin.

Two in-memory capabilities: a scratchpad — the agent's private working
memory, read and written through two tools, each write fully replacing the
content — and a todo list — its task tracker, updated whole-list-at-a-time
through `update_todos`, which numbers the items itself and reports the list
back on `structured_content`. `MemoryPlugin` bundles the tools (in their own
auto-allowing registry) with the system-prompt parts that teach the model to
use them; install it with `plugins=[MemoryPlugin()]` on a
`PluginAgentSessionRunner` (`luca.agent.contrib.plugins`).

The reading helpers — `is_todo_tool`, `todos_of`, `changed_of`,
`todos_from_session` — are how an application renders the list without
hard-coding tool names or re-deriving the id scheme.
"""

from .plugin import (
    MEMORY_NAMESPACE,
    OPEN_STATUSES,
    TODO_TOOL_NAMES,
    MemoryPlugin,
    ReadScratchPadTool,
    ReadTodoTool,
    StoredTodo,
    TodoItem,
    TodoLifecycleMiddleware,
    TodoListResult,
    TodoStatus,
    UpdateTodosTool,
    WriteScratchPadTool,
    changed_of,
    is_open,
    is_todo_tool,
    is_todo_update,
    todos_from_session,
    todos_of,
)

__all__ = [
    "MEMORY_NAMESPACE",
    "OPEN_STATUSES",
    "TODO_TOOL_NAMES",
    "MemoryPlugin",
    "ReadScratchPadTool",
    "ReadTodoTool",
    "StoredTodo",
    "TodoItem",
    "TodoLifecycleMiddleware",
    "TodoListResult",
    "TodoStatus",
    "UpdateTodosTool",
    "WriteScratchPadTool",
    "changed_of",
    "is_open",
    "is_todo_tool",
    "is_todo_update",
    "todos_from_session",
    "todos_of",
]
