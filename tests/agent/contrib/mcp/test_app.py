"""The app lists MCP servers once at startup (a worker), exposes their tools,
and posts a notice. Per-call, so there is no connection to close on exit; a
leaked subprocess from the listing would trip the ResourceWarning-as-error rule."""

import pathlib
import sys

import pytest

pytest.importorskip("textual")

from luca.agent.contrib.mcp.config import StdioServer
from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.wiring import faux_model
from luca.agent.core.runner import AgentSessionRunner
from luca.client.testing import FauxProvider

_FIXTURE = pathlib.Path(__file__).parent / "mcp_server_fixture.py"


async def test_the_app_connects_mcp_at_startup_and_exposes_tools(tmp_path):
    app = AgentApp(
        AgentSessionRunner.new_session(faux_model()),
        provider=FauxProvider(),
        workspace=tmp_path,
        session_dir=tmp_path,
        mode="yolo",
        mcp_servers={"t": StdioServer(command=sys.executable, args=[str(_FIXTURE)])},
    )

    async with app.run_test():
        await app._mcp_worker.wait()  # the startup connect worker listed, no timing bet
        registry = app._find_mcp_registry()
        wire_tools = [tool.name for tool in await app.runner.build_tool_list()]

    assert registry.connected_labels == ["t"]
    assert "t__echo" in wire_tools
    assert "t__add" in wire_tools
