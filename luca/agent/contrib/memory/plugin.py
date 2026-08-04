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
numbering does not reliably do so.

TWO LIFETIMES, DELIBERATELY DIFFERENT. The list is cleared once every item
has settled and the user speaks again — a finished plan should not follow the
conversation around. The counter is never reset, so the todo after a cleared
pair of items is `#3` rather than a second `#1`.

One plugin instance = one scratchpad + one todo list PER CONVERSATION; the
stores live on the plugin, not the session. Nothing here serializes — but
`hydrate()` replays a stored session's todo history back into the store, so
resuming a session mid-plan does not lose the plan.
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
    ContentPart,
    ExecutionResult,
    ExecutionStatus,
    TextContent,
    ToolExecution,
    UserMessage,
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
Once every item is completed the list is cleared the next time the user writes, and numbering carries on.
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


def clear_settled(store: dict, conversation_id: str) -> bool:
    """Empty a list with nothing open, keeping the id counter. True when it
    cleared something."""
    slot = store.get(conversation_id)
    if not slot or not slot["todos"]:
        return False
    if any(is_open(item["status"]) for item in slot["todos"]):
        return False
    slot["todos"] = []
    return True


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


def todos_from_session(session: AgentSession, conversation_id: str | None = None) -> dict:
    """Replay a stored conversation's todo history into a fresh store slot.

    The whole lifecycle, not just the last write: each `update_todos` sets the
    list and each user message clears a settled one, which is exactly what ran
    live. So a session whose plan finished two turns ago rebuilds as an empty
    list with the counter carried forward, and one abandoned mid-plan rebuilds
    with the open items intact."""
    conversation_id = conversation_id or session.main_conversation_id
    conversation = session.conversations.get(conversation_id)
    slot = new_slot()
    if conversation is None:
        return slot
    store = {conversation_id: slot}
    for node_id in conversation.nodes:
        entry = session.entries.get(node_id)
        if isinstance(entry, UserMessage):
            clear_settled(store, conversation_id)
        elif isinstance(entry, ToolExecution) and is_todo_update(entry):
            todos = todos_of(entry)
            if todos is None:
                continue
            slot["todos"] = [dict(item) for item in todos]
            # The counter tracks the highest id ever handed out, not the
            # highest one still on the list: a cleared list must not reissue
            # the numbers the user just saw.
            for item in todos:
                slot["next_id"] = max(slot["next_id"], item["id"] + 1)
    return slot


class TodoLifecycleMiddleware:
    """Clears a settled todo list when the user speaks again.

    `before_post_message` is the one hook that fires exactly at the boundary
    the rule is about, and user messages only ever land on the main
    conversation — a subagent's list lives and dies with the subagent."""

    def __init__(self, store: dict, conversation_id: str) -> None:
        self.store = store
        self.conversation_id = conversation_id

    def before_post_message(self, parts: list[ContentPart]) -> list[ContentPart]:
        clear_settled(self.store, self.conversation_id)
        return parts


class MemoryPlugin:
    """Bundles the memory tools (scratchpad + todo list) with the
    system-prompt parts that teach the model to use them. A plain class
    implementing only the plugin hooks it needs. The tools ship in their own
    auto-allowing registry — an application that wants them gated composes its
    own registry over `get_tools()`'s output."""

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
            tools=self.get_tools(),
            permission_policy=YoloPermissionPolicy(),
        )

    def get_system_prompt_parts(self, agent_session: AgentSession) -> list[str]:
        return [SCRATCHPAD_SYSTEM_PROMPT, TODO_SYSTEM_PROMPT]

    def get_middleware(self, agent_session: AgentSession) -> list:
        """The lifecycle middleware — AND the moment the plugin adopts the
        session it is being installed on. Hydrating here rather than in the
        constructor is what makes the plugin resumable: it is called once,
        with the session, at runner construction, which is the only point the
        plugin ever sees one."""
        self.hydrate(agent_session)
        return [TodoLifecycleMiddleware(self.todo_store, agent_session.main_conversation_id)]

    def hydrate(self, agent_session: AgentSession) -> None:
        """Rebuild the main conversation's todo list from what the session
        already records. A fresh session replays to an empty list."""
        self.todo_store[agent_session.main_conversation_id] = todos_from_session(agent_session)
