"""MCP in the TUI, and the lifecycle finding that shaped the design.

The review of the earlier attempt found that `/new` skipped the startup notice
and popped an OAuth browser mid-message, because `_reset_session` rebuilt the
runner and every rebuild produced a plugin with empty state. The suggested fix
was to spawn the same worker from `_reset_session` too.

This does the other thing, so the headline tests here are about what does NOT
happen: `/clear` keeps the same live connections, the same catalog and the same
tokens, because none of that is owned by anything a session reset touches.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from luca.agent.contrib.mcp.config import McpConfigError, McpSettings
from luca.agent.contrib.tui.app import AgentApp
from luca.agent.contrib.tui.mcp_service import build_mcp_service
from luca.agent.contrib.tui.wiring import build_runner

from .helpers import fresh_session

FIXTURE = str(Path(__file__).parents[1] / "mcp" / "server_fixture.py")

pytestmark = pytest.mark.usefixtures("_isolated_config_environment")


def settings(**over) -> McpSettings:
    return McpSettings.model_validate({"servers": {"fx": {"command": sys.executable, "args": [FIXTURE], **over}}})


def agent_app(tmp_path, *, mcp_settings=None, mcp=True) -> AgentApp:
    return AgentApp(
        fresh_session(),
        workspace=tmp_path,
        session_dir=tmp_path,
        skills=False,
        instructions=False,
        mcp=mcp,
        mcp_settings=mcp_settings,
    )


def test_no_configured_servers_means_no_service(tmp_path):
    # A user who configured nothing pays nothing: no plugin, no prompt part,
    # no httpx client.
    assert agent_app(tmp_path).mcp is None


def test_no_mcp_switches_it_off_even_when_servers_are_configured(tmp_path):
    assert agent_app(tmp_path, mcp_settings=settings(), mcp=False).mcp is None


def test_disabling_the_block_switches_it_off(tmp_path):
    disabled = McpSettings.model_validate({"enabled": False, "servers": {"fx": {"command": "x"}}})

    assert build_mcp_service(disabled, home=tmp_path) is None


def test_an_undefined_environment_reference_fails_at_boot(tmp_path):
    # Caught beside LucaConfigError in `main`, so a typo is a readable startup
    # failure rather than a stack trace on the first turn.
    broken = McpSettings.model_validate({"servers": {"fx": {"command": "x", "env": {"T": "${NOPE}"}}}})

    with pytest.raises(McpConfigError):
        build_mcp_service(broken, home=tmp_path)


def test_the_service_state_lives_beside_the_other_data_luca_writes(tmp_path):
    service = build_mcp_service(settings(), home=tmp_path / "mcp")

    assert (service.catalog.path, service.token_path) == (
        tmp_path / "mcp" / "catalog.json",
        tmp_path / "mcp" / "mcp-auth.json",
    )


async def test_a_connected_server_offers_its_tools_to_the_model(tmp_path):
    app = agent_app(tmp_path, mcp_settings=settings())
    async with app.run_test(size=(105, 35)) as pilot:
        await app.mcp.start()
        await pilot.pause()

        specs = await app.runner.tool_registry.get_tools(app.runner.session, app.runner.session.main_conversation_id)

        assert "mcp__fx__tool_0" in [spec.name for spec in specs]
        await app.mcp.aclose()


async def test_clear_does_not_reconnect(tmp_path):
    # THE REGRESSION. `/clear` goes through `_reset_session`, which rebuilds
    # the runner. If MCP state lived on the plugin this would re-discover every
    # server and re-run OAuth in the middle of the session.
    app = agent_app(tmp_path, mcp_settings=settings())
    async with app.run_test(size=(105, 35)) as pilot:
        await app.mcp.start()
        service = app.mcp
        connection = service._connections["fx"]
        negotiated = connection.negotiated

        from luca.agent.contrib.tui.commands import dispatch

        await dispatch(app, "/clear")
        await pilot.pause()

        assert app.mcp is service
        assert app.mcp._connections["fx"] is connection
        assert app.mcp._connections["fx"].negotiated is negotiated
        await app.mcp.aclose()


async def test_the_rebuilt_runner_still_has_the_mcp_tools(tmp_path):
    app = agent_app(tmp_path, mcp_settings=settings())
    async with app.run_test(size=(105, 35)) as pilot:
        await app.mcp.start()
        from luca.agent.contrib.tui.commands import dispatch

        await dispatch(app, "/clear")
        await pilot.pause()

        specs = await app.runner.tool_registry.get_tools(app.runner.session, app.runner.session.main_conversation_id)

        assert "mcp__fx__tool_0" in [spec.name for spec in specs]
        await app.mcp.aclose()


async def test_closing_the_plugins_does_not_close_the_connections(tmp_path):
    # `_close_plugins` runs on `/clear` as well as on quit, so the MCP plugin
    # deliberately has no `aclose`.
    app = agent_app(tmp_path, mcp_settings=settings())
    async with app.run_test(size=(105, 35)):
        await app.mcp.start()

        await app._close_plugins()

        assert app.mcp._connections["fx"].negotiated is not None
        await app.mcp.aclose()


async def test_mcp_reports_its_servers(tmp_path):
    app = agent_app(tmp_path, mcp_settings=settings())
    async with app.run_test(size=(105, 35)) as pilot:
        await app.mcp.start()
        from luca.agent.contrib.tui.commands import dispatch

        await dispatch(app, "/mcp")
        await pilot.pause()

        from luca.agent.contrib.tui.blocks import NoticeLine

        assert "fx: 1 tools (2026-07-28)" in "\n".join(line.text for line in app.query(NoticeLine))
        await app.mcp.aclose()


async def test_mcp_says_so_when_it_is_off(tmp_path):
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        from luca.agent.contrib.tui.commands import dispatch

        await dispatch(app, "/mcp")
        await pilot.pause()

        from luca.agent.contrib.tui.blocks import NoticeLine

        assert "MCP is off" in "\n".join(line.text for line in app.query(NoticeLine))


def test_build_runner_installs_the_plugin_only_when_a_service_is_given(tmp_path):
    session = fresh_session()
    without, _, _ = build_runner(session, workspace=tmp_path, skills=False, instructions=False)

    assert not any(type(plugin).__name__ == "McpPlugin" for plugin in without.plugins)


def test_build_runner_installs_the_plugin_with_a_service(tmp_path):
    service = build_mcp_service(settings(), home=tmp_path / "mcp")
    runner, strategy, _ = build_runner(
        fresh_session(), workspace=tmp_path, skills=False, instructions=False, mcp=service
    )

    [plugin] = [plugin for plugin in runner.plugins if type(plugin).__name__ == "McpPlugin"]

    # The one shared gate, so an MCP tool passes the same approval path as a
    # shell tool rather than a second one of its own.
    assert plugin._policy is strategy
