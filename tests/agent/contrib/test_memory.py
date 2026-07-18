"""Self-scoped tests for `luca.agent.contrib.memory`: the plugin hands out
its tool registry + prompt parts, and each tool pair reads/writes its shared
store — the scratchpad (each write fully replaces the content) and the todo
list (`update_todos` replaces the whole list in one call).

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
    Conversation,
    LLMConfig,
    SessionConfig,
    ToolContext,
)

MODEL = LLMConfig(model="test-model", provider="faux")

SESSION = AgentSession(
    id="s_memory",
    active_conversation=Conversation(id="c1", nodes=[], created_at=500, updated_at=500),
    session_config=SessionConfig(llm_config=MODEL),
)

CONTEXT = ToolContext(session_id="s_memory", model=MODEL)


def run_kwargs() -> dict:
    return {"cancellation_token": CancellationToken()}


# ── the plugin surface ────────────────────────────────────────────────────────


def test_get_tools_returns_the_memory_tools_sharing_the_plugin_stores():
    plugin = MemoryPlugin()

    tools = plugin.get_tools()

    assert [type(tool) for tool in tools] == [
        ReadScratchPadTool, WriteScratchPadTool, ReadTodoTool, UpdateTodosTool,
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
    assert [type(tool) for tool in registry.get_tools(SESSION)] == [
        ReadScratchPadTool, WriteScratchPadTool, ReadTodoTool, UpdateTodosTool,
    ]
    assert registry.get_tools(SESSION)[0].store is plugin.scratchpad_store


def test_get_system_prompt_parts_returns_the_scratchpad_and_todo_parts():
    plugin = MemoryPlugin()

    parts = plugin.get_system_prompt_parts(SESSION)

    assert parts == [SCRATCHPAD_SYSTEM_PROMPT, TODO_SYSTEM_PROMPT]


# ── scratchpad behavior ───────────────────────────────────────────────────────


async def test_read_empty_scratchpad_returns_empty_string():
    read, _, _, _ = MemoryPlugin().get_tools()

    assert await read._execute({}, CONTEXT, **run_kwargs()) == ""


async def test_write_then_read_round_trips():
    read, write, _, _ = MemoryPlugin().get_tools()

    output = await write._execute(
        {"content": "plan: step 1"}, CONTEXT, **run_kwargs(),
    )

    assert output == "Scratchpad updated successfully"
    assert await read._execute({}, CONTEXT, **run_kwargs()) == "plan: step 1"


async def test_write_fully_replaces_previous_content():
    read, write, _, _ = MemoryPlugin().get_tools()
    await write._execute({"content": "first draft"}, CONTEXT, **run_kwargs())

    await write._execute({"content": "second draft"}, CONTEXT, **run_kwargs())

    assert await read._execute({}, CONTEXT, **run_kwargs()) == "second draft"


async def test_each_plugin_instance_owns_its_own_scratchpad():
    plugin_a = MemoryPlugin()
    plugin_b = MemoryPlugin()
    _, write_a, _, _ = plugin_a.get_tools()
    read_b, _, _, _ = plugin_b.get_tools()

    await write_a._execute({"content": "private to a"}, CONTEXT, **run_kwargs())

    assert await read_b._execute({}, CONTEXT, **run_kwargs()) == ""


# ── todo-list behavior ────────────────────────────────────────────────────────


async def test_read_empty_todo_list_returns_empty_list_repr():
    _, _, read_todo, _ = MemoryPlugin().get_tools()

    assert await read_todo._execute({}, CONTEXT, **run_kwargs()) == "[]"


async def test_update_todos_then_read_round_trips():
    _, _, read_todo, update_todos = MemoryPlugin().get_tools()

    output = await update_todos._execute(
        {"todos": [
            {"content": "T1", "status": "pending"},
            {"content": "T2", "status": "in_progress"},
        ]},
        CONTEXT,
        **run_kwargs(),
    )

    assert output == "Todo list updated successfully"
    assert await read_todo._execute({}, CONTEXT, **run_kwargs()) == (
        "[{'content': 'T1', 'status': 'pending'}, "
        "{'content': 'T2', 'status': 'in_progress'}]"
    )


async def test_update_todos_replaces_the_whole_list():
    _, _, read_todo, update_todos = MemoryPlugin().get_tools()
    await update_todos._execute(
        {"todos": [
            {"content": "T1", "status": "pending"},
            {"content": "T2", "status": "pending"},
            {"content": "T3", "status": "pending"},
        ]},
        CONTEXT,
        **run_kwargs(),
    )

    await update_todos._execute(
        {"todos": [
            {"content": "T1", "status": "pending"},
            {"content": "T2", "status": "completed"},
        ]},
        CONTEXT,
        **run_kwargs(),
    )

    assert await read_todo._execute({}, CONTEXT, **run_kwargs()) == (
        "[{'content': 'T1', 'status': 'pending'}, "
        "{'content': 'T2', 'status': 'completed'}]"
    )


async def test_update_todos_stores_registry_validated_args_as_plain_text():
    # The registry hands _execute the Args.model_validate(...).model_dump()
    # dict, whose statuses are TodoStatus members — the store (and the next
    # read_todo) must still see plain strings.
    _, _, read_todo, update_todos = MemoryPlugin().get_tools()
    args = UpdateTodosTool.Args.model_validate(
        {"todos": [{"content": "T1", "status": "completed"}]}
    ).model_dump()

    await update_todos._execute(args, CONTEXT, **run_kwargs())

    assert await read_todo._execute({}, CONTEXT, **run_kwargs()) == (
        "[{'content': 'T1', 'status': 'completed'}]"
    )


def test_update_todos_args_reject_an_unknown_status():
    with pytest.raises(ValidationError):
        UpdateTodosTool.Args.model_validate(
            {"todos": [{"content": "T1", "status": "done"}]}
        )
