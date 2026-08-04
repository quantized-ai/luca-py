"""The memory tools and the `MemoryPlugin` that bundles them.

Two in-memory capabilities, each backed by a store the plugin owns and hands
to its tool pair (so a write is immediately visible to the next read):

- **Scratchpad** — one string per conversation; each write fully replaces the
  content.
- **Todo list** — a list of `{"id", "content", "status"}` items per
  conversation; `update_todos` replaces the whole list in one call (the model
  re-sends every item, including the unchanged ones).

Each store is keyed BY CONVERSATION. A plugin instance is shared by the main
agent and every subagent, so one flat dict would mean the parent and its
children writing over each other's todo list — private working memory that
silently stops being private the moment a second conversation exists. Tool
dispatch within one conversation is sequential, so a per-conversation slot
needs no lock.

IDS ARE ASSIGNED HERE, NOT BY THE MODEL. `update_todos` takes the same
`{content, status}` items it always did and matches them against the list it
is replacing: same content keeps its id, new content mints one from a
per-conversation counter. Ids are what a user points at ("complete #2"), so
they have to survive a reordering, and a model asked to preserve its own
numbering does not reliably do so. The counter is never reset — an item
added after two that have been dropped is `#3`, not a second `#1`.

A finished list is not swept up on any schedule. `update_todos` replaces it
whole, so the next plan simply overwrites it, and the numbering rule already
hands unseen content a fresh id. Nothing here observes messages or turns.

WHOEVER BUILDS THE PLUGIN OWNS THE STORES. `MemoryPlugin()` makes its own
dicts and everything works, for exactly as long as the process lives. Pass
dicts in and they are the store: the tools mutate what they are given, so an
application that hands in state it persists gets persistence, and one that
hands in nothing gets a clean list every run. Nothing in this package reads or
writes a file, and nothing here knows what a session save is.
"""

from __future__ import annotations

import json
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from luca.agent.contrib.simple_tool_registry import (
    SimpleToolRegistry,
    YoloPermissionPolicy,
)
from luca.agent.contrib.tools import Tool
from luca.agent.core import (
    AgentSession,
    CancellationToken,
    ExecutionResult,
    ExecutionStatus,
    TextContent,
    ToolExecution,
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
Items are numbered for you and both tools return the numbered list, so you never send an id yourself —
and you rarely need read_todo, because update_todos already reports what the list now holds.
An item keeps its number for as long as its content is unchanged; rewording an item gives it a new one.
Mark the item you are working on in_progress, and complete it before starting the next.
Once the work is done, send the next plan whenever you start one — it replaces the old list, and numbering carries on.
""".strip()

# The namespace every tool in this package declares, and the two names that
# make up the todo pair. Exported so a renderer can recognise a todo call
# without hard-coding the strings — matching on the bare name would swallow an
# application's unrelated `update_todos`.
MEMORY_NAMESPACE = "contrib.memory"
TODO_TOOL_NAMES = frozenset({"read_todo", "update_todos"})


class ReadScratchPadTool(Tool):
    namespace = MEMORY_NAMESPACE
    name = "read_scratchpad"
    description = "Read from a in-memory scratchpad"

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

    def __init__(self, store: dict) -> None:
        self.store = store

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        return self.store.get(conversation_id, {}).get("content", "")


class WriteScratchPadTool(Tool):
    namespace = MEMORY_NAMESPACE
    name = "write_scratchpad"
    description = "Write some content a in-memory scratchpad"

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        content: str = Field(description="The content to write to the scratchpad")

    def __init__(self, store: dict) -> None:
        self.store = store

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        self.store.setdefault(conversation_id, {})["content"] = args["content"]
        return "Scratchpad updated successfully"


# ── the todo list ─────────────────────────────────────────────────────────────


class TodoStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# An item still asking for work. The other two statuses are settled: an item
# the agent finished and one it decided against both leave nothing to do, and
# a list with nothing open is a list that has run its course.
OPEN_STATUSES = frozenset({TodoStatus.PENDING.value, TodoStatus.IN_PROGRESS.value})


def is_open(status: str) -> bool:
    return status in OPEN_STATUSES


class TodoItem(BaseModel):
    """One item as the MODEL sends it. No id: the store assigns those."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(description="What this todo item is about")
    status: TodoStatus = Field(description="The item's current status")


class StoredTodo(BaseModel):
    """One item as the store holds it — the model's item plus its id."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(description="The item's stable number, assigned by the store")
    content: str = Field(description="What this todo item is about")
    status: TodoStatus = Field(description="The item's current status")


class TodoListResult(BaseModel):
    """`update_todos`' machine-readable payload: the list as stored, plus the
    ids whose status this call moved. An application renders the list; the
    `changed` ids are what lets it show the work as it happens rather than
    only the settled result."""

    model_config = ConfigDict(extra="forbid")

    todos: list[StoredTodo]
    changed: list[int]


def new_slot() -> dict:
    return {"todos": [], "next_id": 1}


def assign_ids(previous: list[dict], incoming: list[dict], next_id: int) -> tuple[list[dict], int]:
    """Carry ids from `previous` onto `incoming`, matching on content.

    Duplicate contents are matched in order, so two items reading "retry" keep
    their own ids instead of collapsing onto the first. Anything unmatched is
    new and takes the next number."""
    available: dict[str, list[int]] = {}
    for item in previous:
        available.setdefault(item["content"], []).append(item["id"])
    todos: list[dict] = []
    for item in incoming:
        content = item["content"]
        pool = available.get(content)
        if pool:
            todo_id = pool.pop(0)
        else:
            todo_id, next_id = next_id, next_id + 1
        todos.append({"id": todo_id, "content": content, "status": TodoStatus(item["status"]).value})
    return todos, next_id


def changed_ids(previous: list[dict], todos: list[dict]) -> list[int]:
    """The ids this write moved: a new item, or one whose status changed."""
    before = {item["id"]: item["status"] for item in previous}
    return [item["id"] for item in todos if before.get(item["id"]) != item["status"]]


class ReadTodoTool(Tool):
    namespace = MEMORY_NAMESPACE
    name = "read_todo"
    description = "Read the current todo list"
    output_schema = TodoListResult

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

    def __init__(self, store: dict) -> None:
        self.store = store

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        todos = self.store.get(conversation_id, {}).get("todos", [])
        return ExecutionResult(
            content=[TextContent(text=json.dumps(todos, ensure_ascii=False))],
            structured_content=TodoListResult(todos=todos, changed=[]).model_dump(mode="json"),
        )


class UpdateTodosTool(Tool):
    namespace = MEMORY_NAMESPACE
    name = "update_todos"
    description = (
        "Replace the todo list in one operation — send the complete list, including the items that did not change"
    )
    output_schema = TodoListResult

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        todos: list[TodoItem] = Field(description="The complete todo list; replaces the current list entirely")

    def __init__(self, store: dict) -> None:
        self.store = store

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        slot = self.store.setdefault(conversation_id, new_slot())
        todos, next_id = assign_ids(slot["todos"], args["todos"], slot["next_id"])
        changed = changed_ids(slot["todos"], todos)
        slot["todos"], slot["next_id"] = todos, next_id
        # The numbered list goes back to the model in the same breath: it is
        # how the ids it never sent become usable, and it is why a model has
        # no reason to follow every write with a read.
        return ExecutionResult(
            content=[
                TextContent(text="Todo list updated successfully"),
                TextContent(text=json.dumps(todos, ensure_ascii=False)),
            ],
            structured_content=TodoListResult(todos=todos, changed=changed).model_dump(mode="json"),
        )


# ── reading todos back off a session ──────────────────────────────────────────


def is_todo_tool(execution: ToolExecution) -> bool:
    """Either half of the todo pair, matched by DECLARATION. An application's
    own tool of the same name is not this package's."""
    spec = execution.tool_spec
    return spec is not None and spec.namespace == MEMORY_NAMESPACE and spec.name in TODO_TOOL_NAMES


def is_todo_update(execution: ToolExecution) -> bool:
    spec = execution.tool_spec
    return spec is not None and spec.namespace == MEMORY_NAMESPACE and spec.name == "update_todos"


def todos_of(execution: ToolExecution) -> list[dict] | None:
    """The list an `update_todos` execution wrote, or None if it wrote none.

    Read off `structured_content` — the store's own record of what it holds —
    never off `raw_tool_call.arguments`, which is what the model ASKED for and
    carries no ids."""
    if execution.status is not ExecutionStatus.COMPLETED or execution.result is None:
        return None
    payload = execution.result.structured_content
    if not isinstance(payload, dict):
        return None
    todos = payload.get("todos")
    return todos if isinstance(todos, list) else None


def changed_of(execution: ToolExecution) -> list[int]:
    payload = (execution.result.structured_content if execution.result else None) or {}
    changed = payload.get("changed")
    return changed if isinstance(changed, list) else []


class MemoryPlugin:
    """Bundles the memory tools (scratchpad + todo list) with the
    system-prompt parts that teach the model to use them. A plain class
    implementing only the plugin hooks it needs. The tools ship in their own
    auto-allowing registry — an application that wants them gated composes its
    own registry over `get_tools()`'s output.

    THE STORES ARE ARGUMENTS, and this is the whole persistence story:

        MemoryPlugin()                      # works, forgets everything on exit
        MemoryPlugin(todo_store=mine)       # works, and `mine` is the memory

    Each store is a plain `{conversation_id: slot}` dict the tools mutate in
    place. Hand in a dict you keep somewhere durable — `AgentSession.extras`
    is the obvious somewhere, since it is saved with the session — and a
    resumed run picks the list up mid-plan with its numbering intact. Hand in
    nothing and the plugin makes its own, which is the right default for a
    script and wrong for anything a user comes back to. Both dicts are
    JSON-shaped all the way down, so whoever owns them can store them however
    they like.

    Compaction is the application's to handle too: it installs a NEW
    conversation id, and the slot is still filed under the old one. An
    application that persists the stores re-keys them when it sees
    `CompactionFinished.new_conversation_id`; one that does not, does not
    care."""

    def __init__(self, scratchpad_store: dict | None = None, todo_store: dict | None = None) -> None:
        self.scratchpad_store: dict = {} if scratchpad_store is None else scratchpad_store
        self.todo_store: dict = {} if todo_store is None else todo_store

    def get_tools(self) -> list[Tool]:
        return [
            ReadScratchPadTool(self.scratchpad_store),
            WriteScratchPadTool(self.scratchpad_store),
            ReadTodoTool(self.todo_store),
            UpdateTodosTool(self.todo_store),
        ]

    def get_tool_registry(self, agent_session: AgentSession) -> SimpleToolRegistry:
        return SimpleToolRegistry(
            tools=self.get_tools(),
            permission_policy=YoloPermissionPolicy(),
        )

    def get_system_prompt_parts(self, agent_session: AgentSession) -> list[str]:
        return [SCRATCHPAD_SYSTEM_PROMPT, TODO_SYSTEM_PROMPT]
