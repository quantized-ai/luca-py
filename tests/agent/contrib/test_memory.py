"""Self-scoped tests for `luca.agent.contrib.memory`: the plugin hands out
its tool registry + prompt parts, and each tool pair reads/writes its shared
store — the scratchpad (each write fully replaces the content) and the todo
list (`update_todos` replaces the whole list in one call).

The memory tools are contrib `Tool`s (`luca.agent.contrib.tools`), so their
bodies take the live `AgentSession` where they used to take a `ToolContext`.
The package reads NOTHING from it — the stores are handed in — so one inert
session literal is passed to every entry point and is an invariant everywhere.

No runner here — the plugin-to-runner wiring is covered by
`tests/agent/contrib/test_plugins.py`, and the TUI's choice to keep the stores
on the session by `tests/agent/contrib/tui/`.
"""

import pytest
from pydantic import ValidationError

from luca.agent.contrib.memory import (
    MemoryPlugin,
    ReadScratchPadTool,
    ReadTodoTool,
    TodoListResult,
    UpdateTodosTool,
    WriteScratchPadTool,
)
from luca.agent.contrib.memory.plugin import (
    SCRATCHPAD_SYSTEM_PROMPT,
    TODO_SYSTEM_PROMPT,
)
from luca.agent.contrib.simple_tool_registry import (
    SimpleToolRegistry,
    YoloPermissionPolicy,
)
from luca.agent.core import (
    AgentSession,
    CancellationToken,
    ExecutionResult,
    SessionConfig,
    TextContent,
    ToolSpec,
)
from tests.agent.scenarios import (
    MODEL,
    conversation,
    main_conversation,
)

SESSION = AgentSession(
    id="s_memory",
    conversations={"c1": conversation("c1", [], created_at=500, updated_at=500)},
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)


def run_kwargs() -> dict:
    return {"cancellation_token": CancellationToken()}


CONVERSATION = main_conversation(SESSION).id


# ── the specs the model is shown ──────────────────────────────────────────────
# Written out whole rather than derived from `Args`: this is the package's
# public product — the four advertisements a registry hands the provider — and
# `input_schema` is now a required part of it. All four declare the package's
# `namespace`, which travels onto every execution's durable spec snapshot.

READ_SCRATCHPAD_SPEC = ToolSpec(
    name="read_scratchpad",
    description="Read from a in-memory scratchpad",
    namespace="contrib.memory",
    input_schema={
        "additionalProperties": False,
        "properties": {},
        "title": "Args",
        "type": "object",
    },
)

WRITE_SCRATCHPAD_SPEC = ToolSpec(
    name="write_scratchpad",
    description="Write some content a in-memory scratchpad",
    namespace="contrib.memory",
    input_schema={
        "additionalProperties": False,
        "properties": {
            "content": {
                "description": "The content to write to the scratchpad",
                "title": "Content",
                "type": "string",
            },
        },
        "required": ["content"],
        "title": "Args",
        "type": "object",
    },
)

# Both todo tools report the same shape, so the payload schema is written
# once. Ids are ON the way out and absent from `update_todos`' arguments: the
# store numbers the list, the model never does.
TODO_STATUS_SCHEMA = {
    "enum": ["pending", "in_progress", "completed", "cancelled"],
    "title": "TodoStatus",
    "type": "string",
}

TODO_LIST_RESULT_SCHEMA = {
    "$defs": {
        "StoredTodo": {
            "additionalProperties": False,
            "description": "One item as the store holds it — the model's item plus its id.",
            "properties": {
                "id": {
                    "description": "The item's stable number, assigned by the store",
                    "title": "Id",
                    "type": "integer",
                },
                "content": {
                    "description": "What this todo item is about",
                    "title": "Content",
                    "type": "string",
                },
                "status": {"$ref": "#/$defs/TodoStatus", "description": "The item's current status"},
            },
            "required": ["id", "content", "status"],
            "title": "StoredTodo",
            "type": "object",
        },
        "TodoStatus": TODO_STATUS_SCHEMA,
    },
    "additionalProperties": False,
    "description": TodoListResult.__doc__,
    "properties": {
        "todos": {"items": {"$ref": "#/$defs/StoredTodo"}, "title": "Todos", "type": "array"},
        "changed": {"items": {"type": "integer"}, "title": "Changed", "type": "array"},
    },
    "required": ["todos", "changed"],
    "title": "TodoListResult",
    "type": "object",
}

READ_TODO_SPEC = ToolSpec(
    name="read_todo",
    description="Read the current todo list",
    namespace="contrib.memory",
    input_schema={
        "additionalProperties": False,
        "properties": {},
        "title": "Args",
        "type": "object",
    },
    output_schema=TODO_LIST_RESULT_SCHEMA,
)

UPDATE_TODOS_SPEC = ToolSpec(
    name="update_todos",
    namespace="contrib.memory",
    description=(
        "Replace the todo list in one operation — send the complete list, including the items that did not change"
    ),
    input_schema={
        "$defs": {
            "TodoItem": {
                "additionalProperties": False,
                "description": "One item as the MODEL sends it. No id: the store assigns those.",
                "properties": {
                    "content": {
                        "description": "What this todo item is about",
                        "title": "Content",
                        "type": "string",
                    },
                    "status": {
                        "$ref": "#/$defs/TodoStatus",
                        "description": "The item's current status",
                    },
                },
                "required": ["content", "status"],
                "title": "TodoItem",
                "type": "object",
            },
            "TodoStatus": TODO_STATUS_SCHEMA,
        },
        "additionalProperties": False,
        "properties": {
            "todos": {
                "description": "The complete todo list; replaces the current list entirely",
                "items": {"$ref": "#/$defs/TodoItem"},
                "title": "Todos",
                "type": "array",
            },
        },
        "required": ["todos"],
        "title": "Args",
        "type": "object",
    },
    output_schema=TODO_LIST_RESULT_SCHEMA,
)


# ── the plugin surface ────────────────────────────────────────────────────────


def test_get_tools_returns_the_memory_tools_sharing_the_plugin_stores():
    plugin = MemoryPlugin()

    tools = plugin.get_tools()

    assert [type(tool) for tool in tools] == [
        ReadScratchPadTool,
        WriteScratchPadTool,
        ReadTodoTool,
        UpdateTodosTool,
    ]
    assert tools[0].store is plugin.scratchpad_store
    assert tools[1].store is plugin.scratchpad_store
    assert tools[2].store is plugin.todo_store
    assert tools[3].store is plugin.todo_store


def test_a_plugin_given_no_stores_makes_its_own():
    plugin = MemoryPlugin()

    assert (plugin.scratchpad_store, plugin.todo_store) == ({}, {})


def test_the_stores_a_plugin_is_given_are_the_stores_its_tools_write():
    # The whole persistence contract: hand in a dict and it IS the memory.
    # Whoever owns the dict decides whether it outlives the process.
    scratchpad: dict = {}
    todos: dict = {"c1": {"todos": [{"id": 1, "content": "T1", "status": "pending"}], "next_id": 2}}

    plugin = MemoryPlugin(scratchpad_store=scratchpad, todo_store=todos)

    assert [tool.store for tool in plugin.get_tools()] == [scratchpad, scratchpad, todos, todos]


async def test_a_handed_in_todo_store_is_mutated_in_place():
    todos: dict = {CONVERSATION: {"todos": [{"id": 1, "content": "T1", "status": "pending"}], "next_id": 2}}
    _, _, _, update_todos = MemoryPlugin(todo_store=todos).get_tools()

    await write(update_todos, items(("T1", "completed"), ("T2", "pending")))

    assert todos == {
        CONVERSATION: {
            "todos": [
                {"id": 1, "content": "T1", "status": "completed"},
                {"id": 2, "content": "T2", "status": "pending"},
            ],
            "next_id": 3,
        }
    }


def test_get_tool_registry_wraps_the_tools_in_an_auto_allowing_registry():
    plugin = MemoryPlugin()

    registry = plugin.get_tool_registry(SESSION)

    assert type(registry) is SimpleToolRegistry
    assert type(registry.permission_policy) is YoloPermissionPolicy
    assert [type(tool) for tool in registry.tools] == [
        ReadScratchPadTool,
        WriteScratchPadTool,
        ReadTodoTool,
        UpdateTodosTool,
    ]
    assert registry.tools[0].store is plugin.scratchpad_store


async def test_registry_get_tools_advertises_the_four_memory_specs():
    registry = MemoryPlugin().get_tool_registry(SESSION)

    specs = await registry.get_tools(SESSION, CONVERSATION)

    assert specs == [
        READ_SCRATCHPAD_SPEC,
        WRITE_SCRATCHPAD_SPEC,
        READ_TODO_SPEC,
        UPDATE_TODOS_SPEC,
    ]


def test_get_system_prompt_parts_returns_the_scratchpad_and_todo_parts():
    plugin = MemoryPlugin()

    parts = plugin.get_system_prompt_parts(SESSION)

    assert parts == [SCRATCHPAD_SYSTEM_PROMPT, TODO_SYSTEM_PROMPT]


# ── scratchpad behavior ───────────────────────────────────────────────────────


async def test_read_empty_scratchpad_returns_empty_string():
    read, _, _, _ = MemoryPlugin().get_tools()

    assert await read._execute({}, SESSION, CONVERSATION, **run_kwargs()) == ""


async def test_write_then_read_round_trips():
    read, write, _, _ = MemoryPlugin().get_tools()

    output = await write._execute(
        {"content": "plan: step 1"},
        SESSION,
        CONVERSATION,
        **run_kwargs(),
    )

    assert output == "Scratchpad updated successfully"
    assert await read._execute({}, SESSION, CONVERSATION, **run_kwargs()) == "plan: step 1"


async def test_write_fully_replaces_previous_content():
    read, write, _, _ = MemoryPlugin().get_tools()
    await write._execute({"content": "first draft"}, SESSION, CONVERSATION, **run_kwargs())

    await write._execute({"content": "second draft"}, SESSION, CONVERSATION, **run_kwargs())

    assert await read._execute({}, SESSION, CONVERSATION, **run_kwargs()) == "second draft"


async def test_each_plugin_instance_owns_its_own_scratchpad():
    plugin_a = MemoryPlugin()
    plugin_b = MemoryPlugin()
    _, write_a, _, _ = plugin_a.get_tools()
    read_b, _, _, _ = plugin_b.get_tools()

    await write_a._execute({"content": "private to a"}, SESSION, CONVERSATION, **run_kwargs())

    assert await read_b._execute({}, SESSION, CONVERSATION, **run_kwargs()) == ""


# ── todo-list behavior ────────────────────────────────────────────────────────


def items(*pairs: tuple[str, str]) -> dict:
    return {"todos": [{"content": content, "status": status} for content, status in pairs]}


async def write(tool, args: dict) -> ExecutionResult:
    return await tool.execute(args, SESSION, CONVERSATION, **run_kwargs())


async def test_read_empty_todo_list_returns_an_empty_list():
    _, _, read_todo, _ = MemoryPlugin().get_tools()

    assert await write(read_todo, {}) == ExecutionResult(
        content=[TextContent(text="[]")],
        structured_content={"todos": [], "changed": []},
    )


async def test_update_todos_numbers_the_list_and_reports_it_back():
    _, _, _, update_todos = MemoryPlugin().get_tools()

    assert await write(update_todos, items(("T1", "pending"), ("T2", "in_progress"))) == ExecutionResult(
        content=[
            TextContent(text="Todo list updated successfully"),
            TextContent(
                text='[{"id": 1, "content": "T1", "status": "pending"}, '
                '{"id": 2, "content": "T2", "status": "in_progress"}]'
            ),
        ],
        structured_content={
            "todos": [
                {"id": 1, "content": "T1", "status": "pending"},
                {"id": 2, "content": "T2", "status": "in_progress"},
            ],
            "changed": [1, 2],
        },
    )


async def test_update_todos_then_read_round_trips():
    _, _, read_todo, update_todos = MemoryPlugin().get_tools()
    await write(update_todos, items(("T1", "pending")))

    assert await write(read_todo, {}) == ExecutionResult(
        content=[TextContent(text='[{"id": 1, "content": "T1", "status": "pending"}]')],
        structured_content={"todos": [{"id": 1, "content": "T1", "status": "pending"}], "changed": []},
    )


async def test_update_todos_replaces_the_whole_list():
    _, _, _, update_todos = MemoryPlugin().get_tools()
    await write(update_todos, items(("T1", "pending"), ("T2", "pending"), ("T3", "pending")))

    result = await write(update_todos, items(("T1", "pending"), ("T2", "completed")))

    assert result.structured_content == {
        "todos": [
            {"id": 1, "content": "T1", "status": "pending"},
            {"id": 2, "content": "T2", "status": "completed"},
        ],
        "changed": [2],
    }


async def test_an_item_keeps_its_id_through_a_reordering():
    _, _, _, update_todos = MemoryPlugin().get_tools()
    await write(update_todos, items(("T1", "pending"), ("T2", "pending")))

    result = await write(update_todos, items(("T2", "completed"), ("T1", "pending")))

    assert result.structured_content == {
        "todos": [
            {"id": 2, "content": "T2", "status": "completed"},
            {"id": 1, "content": "T1", "status": "pending"},
        ],
        "changed": [2],
    }


async def test_duplicate_contents_keep_their_own_ids():
    _, _, _, update_todos = MemoryPlugin().get_tools()
    await write(update_todos, items(("retry", "pending"), ("retry", "pending")))

    result = await write(update_todos, items(("retry", "completed"), ("retry", "pending")))

    assert result.structured_content["todos"] == [
        {"id": 1, "content": "retry", "status": "completed"},
        {"id": 2, "content": "retry", "status": "pending"},
    ]


async def test_rewording_an_item_mints_a_new_id():
    _, _, _, update_todos = MemoryPlugin().get_tools()
    await write(update_todos, items(("T1", "pending")))

    result = await write(update_todos, items(("T1, but rephrased", "pending")))

    assert result.structured_content == {
        "todos": [{"id": 2, "content": "T1, but rephrased", "status": "pending"}],
        "changed": [2],
    }


async def test_update_todos_stores_registry_validated_args_as_plain_text():
    # The registry's prepare() hands execute() the
    # Args.model_validate(...).model_dump() dict, whose statuses are
    # TodoStatus members — the store (and the next read_todo) must still see
    # plain strings.
    _, _, read_todo, update_todos = MemoryPlugin().get_tools()
    args = UpdateTodosTool.Args.model_validate({"todos": [{"content": "T1", "status": "completed"}]}).model_dump()

    await write(update_todos, args)

    assert (await write(read_todo, {})).content == [
        TextContent(text='[{"id": 1, "content": "T1", "status": "completed"}]')
    ]


def test_update_todos_args_reject_an_unknown_status():
    with pytest.raises(ValidationError):
        UpdateTodosTool.Args.model_validate({"todos": [{"content": "T1", "status": "done"}]})


# ── the list's lifetime ───────────────────────────────────────────────────────


async def test_a_settled_list_stands_until_the_next_write():
    # Nothing sweeps a finished plan up on a schedule: these two tools write
    # the store and nothing else does.
    plugin = MemoryPlugin()
    _, _, read_todo, update_todos = plugin.get_tools()
    await write(update_todos, items(("T1", "completed"), ("T2", "cancelled")))

    assert (await write(read_todo, {})).structured_content == {
        "todos": [
            {"id": 1, "content": "T1", "status": "completed"},
            {"id": 2, "content": "T2", "status": "cancelled"},
        ],
        "changed": [],
    }


async def test_numbering_carries_on_into_the_plan_after_a_settled_one():
    # Replacement IS the lifecycle: unseen content takes the next number, so
    # the plan after two finished items starts at #3, never a second #1.
    _, _, _, update_todos = MemoryPlugin().get_tools()
    await write(update_todos, items(("T1", "completed"), ("T2", "completed")))

    result = await write(update_todos, items(("T3", "pending")))

    assert result.structured_content == {
        "todos": [{"id": 3, "content": "T3", "status": "pending"}],
        "changed": [3],
    }


async def test_a_finished_item_that_comes_back_keeps_its_number():
    # Content is the identity, and it does not expire: a task reopened after
    # it settled is the same task, under the number the user already saw.
    _, _, _, update_todos = MemoryPlugin().get_tools()
    await write(update_todos, items(("T1", "completed")))

    result = await write(update_todos, items(("T1", "in_progress")))

    assert result.structured_content == {
        "todos": [{"id": 1, "content": "T1", "status": "in_progress"}],
        "changed": [1],
    }
