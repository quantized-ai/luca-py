"""The memory tools and the `MemoryPlugin` that bundles them.

Two in-memory capabilities, each backed by a store the plugin owns and hands
to its tool pair (so a write is immediately visible to the next read):

- **Scratchpad** — one shared string; each write fully replaces the content.
- **Todo list** — a list of `{"content", "status"}` items; `update_todos`
  replaces the whole list in one call (the model re-sends every item,
  including the unchanged ones).

One plugin instance = one scratchpad + one todo list; the stores live on the
plugin, not the session — nothing here persists or serializes.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from luca.agent.core import (
    AgentSession,
    CancellationToken,
    Tool,
    ToolContext,
)
from luca.agent.contrib.simple_tool_registry import (
    SimpleToolRegistry,
    YoloPermissionPolicy,
)

SCRATCHPAD_SYSTEM_PROMPT = """
The following tools are available:
### Scratchpad (read_scratchpad / write_scratchpad)
Your private working memory.
Use it to think through an approach, store intermediate findings, or draft content before committing.
Each write fully replaces the previous content.
""".strip()

TODO_SYSTEM_PROMPT = """
### Todo list (read_todo / update_todos)
Your task tracker for multi-step work.
Each item is {"content": <what to do>, "status": pending | in_progress | completed | cancelled}.
update_todos replaces the whole list at once: always send every item, not just the ones that changed.
""".strip()


class ReadScratchPadTool(Tool):
    name = "read_scratchpad"
    description = "Read from a in-memory scratchpad"

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

    def __init__(self, store: dict) -> None:
        self.store = store

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        return self.store.get("content", "")


class WriteScratchPadTool(Tool):
    name = "write_scratchpad"
    description = "Write some content a in-memory scratchpad"

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        content: str = Field(description="The content to write to the scratchpad")

    def __init__(self, store: dict) -> None:
        self.store = store

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        self.store["content"] = args["content"]
        return "Scratchpad updated successfully"


class TodoStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TodoItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(description="What this todo item is about")
    status: TodoStatus = Field(description="The item's current status")


class ReadTodoTool(Tool):
    name = "read_todo"
    description = "Read the current todo list"

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

    def __init__(self, store: dict) -> None:
        self.store = store

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        return repr(self.store.get("todos", []))


class UpdateTodosTool(Tool):
    name = "update_todos"
    description = (
        "Replace the todo list in one operation — send the complete list, "
        "including the items that did not change"
    )

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        todos: list[TodoItem] = Field(
            description="The complete todo list; replaces the current list entirely"
        )

    def __init__(self, store: dict) -> None:
        self.store = store

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        # Store JSON-clean dicts: the validated args carry TodoStatus members,
        # which would repr() as enums on the next read_todo.
        self.store["todos"] = [
            {"content": item["content"], "status": TodoStatus(item["status"]).value}
            for item in args["todos"]
        ]
        return "Todo list updated successfully"


class MemoryPlugin:
    """Bundles the memory tools (scratchpad + todo list) with the
    system-prompt parts that teach the model to use them. A plain class
    implementing only the plugin hooks it needs (no middleware). The tools
    ship in their own auto-allowing registry — an application that wants
    them gated composes its own registry over `get_tools()`'s output."""

    def __init__(self) -> None:
        self.scratchpad_store: dict = {}
        self.todo_store: dict = {}

    def get_tools(self) -> list[Tool]:
        return [
            ReadScratchPadTool(self.scratchpad_store),
            WriteScratchPadTool(self.scratchpad_store),
            ReadTodoTool(self.todo_store),
            UpdateTodosTool(self.todo_store),
        ]

    def get_tool_registry(self, agent_session: AgentSession) -> SimpleToolRegistry:
        return SimpleToolRegistry(
            tools=self.get_tools(), permission_policy=YoloPermissionPolicy(),
        )

    def get_system_prompt_parts(self, agent_session: AgentSession) -> list[str]:
        return [SCRATCHPAD_SYSTEM_PROMPT, TODO_SYSTEM_PROMPT]
