"""MCP server config validation and build_mcp_plugin filtering."""

import pytest
from pydantic import ValidationError

from luca.agent.contrib.mcp import build_mcp_plugin
from luca.agent.contrib.mcp.config import HttpServer, StdioServer
from luca.agent.contrib.simple_tool_registry import YoloPermissionPolicy
from luca.agent.contrib.tui.config import LucaConfig


def test_a_stdio_and_an_http_server_parse_on_the_discriminator():
    config = LucaConfig.model_validate(
        {
            "mcp": {
                "files": {"type": "stdio", "command": "npx", "args": ["-y", "srv"]},
                "remote": {"type": "http", "url": "https://x/mcp", "headers": {"A": "b"}},
            }
        }
    )
    assert config.mcp["files"] == StdioServer(command="npx", args=["-y", "srv"])
    assert config.mcp["remote"] == HttpServer(url="https://x/mcp", headers={"A": "b"})


def test_an_unknown_field_on_a_server_is_rejected():
    with pytest.raises(ValidationError):
        LucaConfig.model_validate({"mcp": {"s": {"type": "stdio", "command": "x", "nope": 1}}})


def test_build_mcp_plugin_drops_disabled_servers_and_returns_none_when_empty():
    servers = {
        "on": StdioServer(command="a"),
        "off": StdioServer(command="b", enabled=False),
    }
    plugin = build_mcp_plugin(servers, YoloPermissionPolicy())
    assert list(plugin._servers) == ["on"]
    assert build_mcp_plugin({"off": StdioServer(command="b", enabled=False)}, YoloPermissionPolicy()) is None
