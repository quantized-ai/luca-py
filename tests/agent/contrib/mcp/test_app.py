"""The app connects MCP on mount, exposes the tools, and cleans up on exit
(a leaked subprocess would trip the repo's ResourceWarning-as-error rule)."""

import pathlib
import sys

import pytest

pytest.importorskip("textual")

from luca.agent.contrib.mcp.config import StdioServer
from luca.agent.contrib.mcp.factory import build_manager
from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.wiring import faux_model
from luca.agent.core.runner import AgentSessionRunner
from luca.client.testing import FauxProvider

_FIXTURE = pathlib.Path(__file__).parent / "mcp_server_fixture.py"


async def test_the_app_connects_mcp_exposes_tools_and_closes_on_exit(tmp_path):
    manager = build_manager({"t": StdioServer(command=sys.executable, args=[str(_FIXTURE)])})
    app = AgentApp(
        AgentSessionRunner.new_session(faux_model()),
        provider=FauxProvider(),
        workspace=tmp_path,
        session_dir=tmp_path,
        mode="yolo",
        mcp_manager=manager,
    )

    async with app.run_test():
        await app._mcp_worker.wait()  # the connect worker has finished, no timing bet
        wire_tools = [tool.name for tool in app.runner.build_tool_list()]
        assert "t__echo" in wire_tools
        assert "t__add" in wire_tools

    assert manager.connected_labels == []  # on_unmount closed the connection
