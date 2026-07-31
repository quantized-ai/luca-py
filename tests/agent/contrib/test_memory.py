"""Self-scoped tests for `luca.agent.contrib.memory`: the plugin hands out
its tool registry + prompt parts, and each tool pair reads/writes its shared
store — the scratchpad (each write fully replaces the content) and the todo
list (`update_todos` replaces the whole list in one call).

The memory tools are contrib `Tool`s (`luca.agent.contrib.tools`), so their
bodies take the live `AgentSession` where they used to take a `ToolContext`.
The package reads NOTHING from it — the stores live on the plugin — so one
inert session literal is handed to every entry point and is an invariant
everywhere.

No runner here — the plugin-to-runner wiring is covered by
`tests/agent/contrib/test_plugins.py`.
"""

import pytest
from pydantic import ValidationError

from luca.agent.contrib.memory import (
    MemoryPlugin,
    ReadScratchPadTool,
    ReadTodoTool,
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
    SessionConfig,
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
            "TodoStatus": {
                "enum": ["pending", "in_progress", "completed", "cancelled"],
                "title": "TodoStatus",
                "type": "string",
            },
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


async def test_read_empty_todo_list_returns_empty_list_repr():
    _, _, read_todo, _ = MemoryPlugin().get_tools()

    assert await read_todo._execute({}, SESSION, CONVERSATION, **run_kwargs()) == "[]"


async def test_update_todos_then_read_round_trips():
    _, _, read_todo, update_todos = MemoryPlugin().get_tools()

    output = await update_todos._execute(
        {
            "todos": [
                {"content": "T1", "status": "pending"},
                {"content": "T2", "status": "in_progress"},
            ]
        },
        SESSION,
        CONVERSATION,
        **run_kwargs(),
    )

    assert output == "Todo list updated successfully"
    assert await read_todo._execute({}, SESSION, CONVERSATION, **run_kwargs()) == (
        "[{'content': 'T1', 'status': 'pending'}, {'content': 'T2', 'status': 'in_progress'}]"
    )


async def test_update_todos_replaces_the_whole_list():
    _, _, read_todo, update_todos = MemoryPlugin().get_tools()
    await update_todos._execute(
        {
            "todos": [
                {"content": "T1", "status": "pending"},
                {"content": "T2", "status": "pending"},
                {"content": "T3", "status": "pending"},
            ]
        },
        SESSION,
        CONVERSATION,
        **run_kwargs(),
    )

    await update_todos._execute(
        {
            "todos": [
                {"content": "T1", "status": "pending"},
                {"content": "T2", "status": "completed"},
            ]
        },
        SESSION,
        CONVERSATION,
        **run_kwargs(),
    )

    assert await read_todo._execute({}, SESSION, CONVERSATION, **run_kwargs()) == (
        "[{'content': 'T1', 'status': 'pending'}, {'content': 'T2', 'status': 'completed'}]"
    )


async def test_update_todos_stores_registry_validated_args_as_plain_text():
    # The registry's prepare() hands _execute the
    # Args.model_validate(...).model_dump() dict, whose statuses are
    # TodoStatus members — the store (and the next read_todo) must still see
    # plain strings.
    _, _, read_todo, update_todos = MemoryPlugin().get_tools()
    args = UpdateTodosTool.Args.model_validate({"todos": [{"content": "T1", "status": "completed"}]}).model_dump()

    await update_todos._execute(args, SESSION, CONVERSATION, **run_kwargs())

    assert await read_todo._execute({}, SESSION, CONVERSATION, **run_kwargs()) == (
        "[{'content': 'T1', 'status': 'completed'}]"
    )


def test_update_todos_args_reject_an_unknown_status():
    with pytest.raises(ValidationError):
        UpdateTodosTool.Args.model_validate({"todos": [{"content": "T1", "status": "done"}]})
