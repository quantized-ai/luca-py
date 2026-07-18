"""`build_runner` — the full demo composition."""

from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.resource_permissions import PermissionMode
from luca.agent.contrib.tui.wiring import build_faux_provider, build_runner

from .helpers import fresh_session

MATH_TOOLS = {"add", "subtract", "multiply"}
SHELL_TOOLS = {"read", "glob", "grep", "edit", "write", "apply_patch", "bash"}
MEMORY_TOOLS = {"read_scratchpad", "write_scratchpad", "read_todo", "update_todos"}


def test_build_runner_composes_all_tool_families(tmp_path):
    session = fresh_session()

    runner, strategy = build_runner(session, workspace=tmp_path)

    assert isinstance(runner, PluginAgentSessionRunner)
    assert runner.session is session
    names = {tool.name for tool in runner.build_tool_list()}
    assert names == MATH_TOOLS | SHELL_TOOLS | MEMORY_TOOLS
    assert strategy.mode is PermissionMode.ASK


def test_build_runner_mode_passthrough(tmp_path):
    session = fresh_session()

    _, strategy = build_runner(session, workspace=tmp_path, mode="yolo")

    assert strategy.mode is PermissionMode.YOLO


def test_faux_provider_starts_unconsumed():
    provider = build_faux_provider()

    assert provider.requests == []
