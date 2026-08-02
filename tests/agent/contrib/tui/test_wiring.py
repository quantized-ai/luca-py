"""`build_runner` — the full demo composition, and the demo math tools."""

import pytest

from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.resource_permissions import PermissionMode
from luca.agent.contrib.tui.wiring import (
    AddTool,
    MultiplyTool,
    SubtractTool,
    build_faux_provider,
    build_runner,
)
from luca.agent.core import AgentSessionRunner as _Runner  # noqa: F401
from luca.agent.core.context import CancellationToken
from luca.agent.core.models import ExecutionResult, TextContent
from luca.client.types import Tool as LucaTool
from tests.agent.scenarios import (
    main_conversation,
)

from .helpers import fresh_session

MATH_TOOLS = {"add", "subtract", "multiply"}
SHELL_TOOLS = {"read", "glob", "grep", "edit", "write", "apply_patch", "bash"}
MEMORY_TOOLS = {"read_scratchpad", "write_scratchpad", "read_todo", "update_todos"}
SKILL_TOOLS = {"skill"}

# What one demo tool looks like on the wire: the `ToolSpec.input_schema`
# snapshotted from `BinaryOp` at `get_tool_spec()` time, mapped straight onto
# `luca.client.Tool.parameters` by the adapter.
MULTIPLY_WIRE_TOOL = LucaTool(
    name="multiply",
    description="Multiply two numbers and return the product.",
    parameters={
        "properties": {
            "a": {"description": "The first operand.", "title": "A", "type": "number"},
            "b": {"description": "The second operand.", "title": "B", "type": "number"},
        },
        "required": ["a", "b"],
        "title": "BinaryOp",
        "type": "object",
    },
)


async def _wire_tools(runner):
    """What the model is actually shown: resolve the specs, then project them
    onto the wire (which drops any private tool)."""
    specs = await runner.resolve_tool_specs(runner.main_conversation_id)
    return runner.build_tool_list(runner.main_conversation_id, specs)


async def test_build_runner_composes_all_tool_families(tmp_path):
    session = fresh_session()

    runner, strategy = build_runner(session, workspace=tmp_path)

    assert isinstance(runner, PluginAgentSessionRunner)
    assert runner.session is session
    names = {tool.name for tool in await _wire_tools(runner)}
    assert names == MATH_TOOLS | SHELL_TOOLS | MEMORY_TOOLS | SKILL_TOOLS
    assert strategy.mode is PermissionMode.ASK


async def test_no_skills_withholds_the_skill_tool(tmp_path):
    runner, _ = build_runner(fresh_session(), workspace=tmp_path, skills=False)

    names = {tool.name for tool in await _wire_tools(runner)}

    assert names == MATH_TOOLS | SHELL_TOOLS | MEMORY_TOOLS


async def test_build_runner_puts_the_math_argument_schema_on_the_wire(tmp_path):
    runner, _ = build_runner(fresh_session(), workspace=tmp_path)

    tools = await _wire_tools(runner)

    # Membership, not equality: the same list carries the shell and memory
    # families, whose schemas this test does not own.
    assert MULTIPLY_WIRE_TOOL in tools


def test_build_runner_mode_passthrough(tmp_path):
    session = fresh_session()

    _, strategy = build_runner(session, workspace=tmp_path, mode="yolo")

    assert strategy.mode is PermissionMode.YOLO


@pytest.mark.parametrize(
    ("math_tool", "expected"),
    [(AddTool(), "9.0"), (SubtractTool(), "5.0"), (MultiplyTool(), "14.0")],
)
async def test_math_tools_execute_against_the_live_session(math_tool, expected):
    session = fresh_session()

    result = await math_tool.execute(
        {"a": 7.0, "b": 2.0},
        session,
        main_conversation(session).id,
        cancellation_token=CancellationToken(),
    )

    assert result == ExecutionResult(content=[TextContent(text=expected)])


def test_faux_provider_starts_unconsumed():
    provider = build_faux_provider()

    assert provider.requests == []
