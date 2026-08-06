"""`build_runner` — the full demo composition, and the demo math tools."""

import pytest

from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.prompts import BASE, load_prompt
from luca.agent.contrib.resource_permissions import PermissionMode
from luca.agent.contrib.tui.wiring import (
    AddTool,
    MultiplyTool,
    SubtractTool,
    build_faux_provider,
    build_runner,
)
from luca.agent.core import AgentSessionRunner as _Runner  # noqa: F401
from luca.agent.core.adapter import tool_spec_to_luca_tool
from luca.agent.core.context import CancellationToken
from luca.agent.core.models import ExecutionResult, ExecutionStatus, LLMConfig, TextContent, ToolExecution
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
    """What the model is actually shown: resolve the specs, narrow them to the
    model-visible ones (which drops any private tool), then adapt onto the
    wire — the same three steps `_collect_tools` takes."""
    specs = await runner.resolve_tool_specs(runner.main_conversation_id)
    visible = runner.build_tool_list(runner.main_conversation_id, specs)
    return [tool_spec_to_luca_tool(spec) for spec in visible]


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


# These assert on fragments rather than the whole prompt: the assembled string
# also carries every other plugin's tool blurb, which these tests do not own.
def _prompt(runner, model="anthropic/claude-sonnet-5"):
    runner.session.session_config.llm_config = LLMConfig(model=model, provider="openrouter")
    return runner.build_system_message(runner.main_conversation_id)


def test_the_base_prompt_opens_and_the_family_addendum_follows_the_model(tmp_path):
    runner, _ = build_runner(fresh_session(), workspace=tmp_path)

    claude = _prompt(runner, "anthropic/claude-sonnet-5")
    gpt = _prompt(runner, "openai/gpt-5.4-mini")

    assert claude.startswith(load_prompt(BASE))
    assert load_prompt("anthropic") in claude
    assert load_prompt("gpt") not in claude
    assert load_prompt("gpt") in gpt
    assert load_prompt("anthropic") not in gpt


def test_the_project_instructions_are_the_last_thing_in_the_prompt(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("House rule: never use tabs.")

    prompt = _prompt(build_runner(fresh_session(), workspace=tmp_path)[0])

    assert prompt.endswith(f"--- {tmp_path / 'AGENTS.md'} ---\nHouse rule: never use tabs.")


def test_no_instructions_withholds_the_project_rules(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("House rule: never use tabs.")

    prompt = _prompt(build_runner(fresh_session(), workspace=tmp_path, instructions=False)[0])

    assert "never use tabs" not in prompt


def test_extra_instructions_are_read_too(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "conventions.md").write_text("Two spaces, always.")

    runner, _ = build_runner(
        fresh_session(),
        workspace=tmp_path,
        extra_instructions=["conventions.md"],
    )

    assert "Two spaces, always." in _prompt(runner)


def test_the_environment_block_sits_between_the_tools_and_the_instructions(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("House rule.")

    prompt = _prompt(build_runner(fresh_session(), workspace=tmp_path)[0])

    assert prompt.index("### Shell access") < prompt.index("### Environment") < prompt.index("### Project instructions")


def test_faux_provider_starts_unconsumed():
    provider = build_faux_provider()

    assert provider.requests == []


# ── provider-native tools ────────────────────────────────────────────────────


async def test_an_anthropic_session_gets_the_providers_editor(tmp_path):
    session = fresh_session()
    session.session_config.llm_config = LLMConfig(model="claude-opus-5", provider="anthropic")

    runner, _ = build_runner(session, workspace=tmp_path)

    names = {tool.name for tool in await _wire_tools(runner)}
    assert "str_replace_based_edit_tool" in names
    assert {"read", "edit", "write"}.isdisjoint(names)


async def test_a_non_anthropic_session_keeps_lucas_own_tools(tmp_path):
    session = fresh_session()
    session.session_config.llm_config = LLMConfig(model="gpt-5.4", provider="openai")

    runner, _ = build_runner(session, workspace=tmp_path)

    names = {tool.name for tool in await _wire_tools(runner)}
    assert {"read", "edit", "write"} <= names
    assert "str_replace_based_edit_tool" not in names


async def test_bedrock_keeps_lucas_own_tools_even_for_a_claude(tmp_path):
    # Converse does not accept a tool declared by `type`, so a model-family
    # check would confidently send something the API rejects
    session = fresh_session()
    session.session_config.llm_config = LLMConfig(model="anthropic.claude-opus-5", provider="bedrock")

    runner, _ = build_runner(session, workspace=tmp_path)

    names = {tool.name for tool in await _wire_tools(runner)}
    assert {"read", "edit", "write"} <= names
    assert "str_replace_based_edit_tool" not in names


async def test_native_tools_off_restores_lucas_own(tmp_path):
    session = fresh_session()
    session.session_config.llm_config = LLMConfig(model="claude-opus-5", provider="anthropic")

    runner, _ = build_runner(session, workspace=tmp_path, native_tools=False)

    names = {tool.name for tool in await _wire_tools(runner)}
    assert {"read", "edit", "write"} <= names
    assert "str_replace_based_edit_tool" not in names


async def test_the_native_editor_reaches_the_wire_without_a_schema(tmp_path):
    session = fresh_session()
    session.session_config.llm_config = LLMConfig(model="claude-opus-5", provider="anthropic")

    runner, _ = build_runner(session, workspace=tmp_path)

    editor = next(t for t in await _wire_tools(runner) if t.name == "str_replace_based_edit_tool")
    assert editor.provider_type == "text_editor_20250728"


async def test_switching_model_mid_session_rebuilds_the_wire_tools(tmp_path):
    """`/model` reassigns the session's llm_config and nothing rebuilds the
    runner. With the tool set frozen at composition, the next request carries
    Anthropic's editor to the Responses transport, which refuses it before the
    HTTP call — on that turn and on every retry after it."""
    session = fresh_session()
    session.session_config.llm_config = LLMConfig(model="claude-opus-5", provider="anthropic")
    runner, _ = build_runner(session, workspace=tmp_path)
    assert "str_replace_based_edit_tool" in {tool.name for tool in await _wire_tools(runner)}

    session.session_config.llm_config = LLMConfig(model="gpt-5.4", provider="openai")

    tools = await _wire_tools(runner)
    assert {tool.name for tool in tools if tool.provider_type} == {"apply_patch", "shell"}
    assert "str_replace_based_edit_tool" not in {tool.name for tool in tools}


# ── provider-native tools through the real runner ────────────────────────────
#
# They live HERE and not in tests/agent/contrib/shell/ because that package's
# tests are self-scoped units: no runner, no registry, no session.


def _anthropic_session():
    session = fresh_session()
    session.session_config.llm_config = LLMConfig(model="claude-opus-5", provider="anthropic")
    return session


async def test_a_native_tool_call_drives_end_to_end(tmp_path):
    """A scripted `tool_use` naming the provider's tool: resolved, approved,
    executed, and recorded like any other call."""
    from luca.client.testing import FauxProvider, faux_assistant_message, faux_text, faux_tool_call

    target = tmp_path / "hello.py"
    target.write_text("print('old')\n")
    session = _anthropic_session()
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("str_replace_based_edit_tool", {"command": "view", "path": str(target)}, id="t1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [
                    faux_tool_call(
                        "str_replace_based_edit_tool",
                        {"command": "str_replace", "path": str(target), "old_str": "old", "new_str": "new"},
                        id="t2",
                    )
                ],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    runner, _ = build_runner(session, workspace=tmp_path, provider=faux, mode="yolo")

    runner.post_message("edit the file")
    await runner.run()

    executions = [e for e in session.entries.values() if isinstance(e, ToolExecution)]
    assert [e.raw_tool_call.name for e in executions] == ["str_replace_based_edit_tool", "str_replace_based_edit_tool"]
    assert [e.status for e in executions] == [ExecutionStatus.COMPLETED, ExecutionStatus.COMPLETED]
    assert target.read_text() == "print('new')\n"


async def test_the_wire_list_swaps_the_editor_in_and_lucas_own_out(tmp_path):

    session = _anthropic_session()
    runner, _ = build_runner(session, workspace=tmp_path)

    specs = await runner.resolve_tool_specs(runner.main_conversation_id)
    tools = {tool.name: tool.provider_type for tool in runner.build_tool_list(runner.main_conversation_id, specs)}

    assert tools["str_replace_based_edit_tool"] == "text_editor_20250728"
    assert tools["bash"] == "bash_20250124"
    assert {"read", "edit", "write"}.isdisjoint(tools)
    # the tools with no native counterpart still travel as ordinary ones
    assert tools["glob"] is None
    assert tools["apply_patch"] is None
