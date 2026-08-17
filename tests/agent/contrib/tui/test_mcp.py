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


async def test_mcp_opens_a_screen_listing_the_servers(tmp_path):
    app = agent_app(tmp_path, mcp_settings=settings())
    async with app.run_test(size=(105, 35)) as pilot:
        await app.mcp.start()
        from luca.agent.contrib.tui.commands import dispatch
        from luca.agent.contrib.tui.modals import McpScreen

        await dispatch(app, "/mcp")
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, McpScreen)
        assert [(row.label, row.state, row.detail) for row in screen.state.rows] == [
            ("fx", "connected", "1 tool · 2026-07-28")
        ]
        await app.mcp.aclose()


async def test_the_screen_offers_a_login_for_an_unauthenticated_server(tmp_path):
    # The row a 401 used to render in red. It is an action, not a failure.
    from luca.agent.contrib.mcp.servers import ServerState, ServerStatus
    from luca.agent.contrib.tui.mcp_service import build_mcp_state

    state = build_mcp_state(
        [
            ServerStatus(label="files", state=ServerState.CONNECTED, tool_count=14, protocol_version="2025-11-25"),
            ServerStatus(label="linear", state=ServerState.NEEDS_AUTH, oauth=True),
        ]
    )

    assert [(row.label, row.state, row.detail, row.action) for row in state.rows] == [
        ("files", "connected", "14 tools · 2025-11-25", "tools"),
        ("linear", "needs_auth", "not authenticated", "authenticate"),
    ]
    assert state.count_line == "1 of 2 connected · 14 tools"


async def test_an_excluded_tool_is_listed_with_the_others_and_says_why(tmp_path):
    # Under the server it belongs to, next to the tools it is missing from.
    # As a footnote below the server list it was a reason with no subject.
    from luca.agent.contrib.mcp.servers import ServerState, ServerStatus
    from luca.agent.contrib.tui.mcp_service import build_mcp_detail
    from luca.agent.core import ToolSpec

    status = ServerStatus(
        label="airtable",
        state=ServerState.CONNECTED,
        tool_count=1,
        rejected_tools={"bulk": "'x-mcp-header' at rows is on a 'array' parameter"},
    )
    spec = ToolSpec(
        name="mcp__airtable__list_records",
        description="List records in a table.",
        input_schema={"type": "object", "properties": {}},
        metadata={"mcp": {"server": "airtable", "tool": "list_records"}},
    )

    detail = build_mcp_detail(status, [spec])

    assert [(row.name, row.summary, row.excluded) for row in detail.rows] == [
        ("list_records", "List records in a table.", None),
        ("bulk", "", "'x-mcp-header' at rows is on a 'array' parameter"),
    ]
    assert detail.count_line == "1 tool · 1 excluded"


async def test_a_stale_server_still_says_it_has_tools(tmp_path):
    from luca.agent.contrib.mcp.servers import ServerState, ServerStatus
    from luca.agent.contrib.tui.mcp_service import build_mcp_state

    state = build_mcp_state([ServerStatus(label="github", state=ServerState.STALE, tool_count=9, error="timed out")])

    assert [(row.state, row.detail, row.action) for row in state.rows] == [
        ("stale", "9 tools · last refresh failed: timed out", "tools")
    ]


async def test_a_server_being_listed_says_so_rather_than_looking_broken(tmp_path):
    from luca.agent.contrib.mcp.servers import ServerState, ServerStatus
    from luca.agent.contrib.tui.mcp_service import build_mcp_state

    state = build_mcp_state([ServerStatus(label="files", state=ServerState.CONNECTING)])

    assert [(row.state, row.detail) for row in state.rows] == [("connecting", "connecting…")]


async def test_the_boot_notice_names_what_still_needs_doing(tmp_path):
    # Reporting only successes left a server waiting on a login invisible
    # until somebody thought to type `/mcp`.
    from luca.agent.contrib.mcp.servers import ServerState, ServerStatus
    from luca.agent.contrib.tui.mcp_service import mcp_boot_notice

    notice = mcp_boot_notice(
        [
            ServerStatus(label="files", state=ServerState.CONNECTED, tool_count=14),
            ServerStatus(label="linear", state=ServerState.NEEDS_AUTH, oauth=True),
            ServerStatus(label="staging", state=ServerState.INACTIVE, error="no such file"),
        ]
    )

    assert notice == "MCP connected: files (14 tools) · linear needs a login · staging failed — /mcp"


async def test_the_boot_notice_is_silent_when_nothing_is_configured(tmp_path):
    from luca.agent.contrib.tui.mcp_service import mcp_boot_notice

    assert mcp_boot_notice([]) is None


async def test_the_hints_say_what_the_keys_do_to_this_row(tmp_path):
    # `d disable` over a server that is already disabled is worse than no
    # legend, so the legend follows the selection.
    from luca.agent.contrib.mcp.servers import ServerState, ServerStatus
    from luca.agent.contrib.tui.mcp_service import build_mcp_state, mcp_hints

    state = build_mcp_state([ServerStatus(label="notion", state=ServerState.DISABLED)])

    assert mcp_hints(state) == ["↑↓ move", "enter enable", "esc back"]


async def test_mcp_says_so_when_it_is_off(tmp_path):
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        from luca.agent.contrib.tui.commands import dispatch

        await dispatch(app, "/mcp")
        await pilot.pause()

        from luca.agent.contrib.tui.blocks import NoticeLine

        assert "MCP is off" in "\n".join(line.text for line in app.query(NoticeLine))


async def test_disabling_a_server_from_the_screen_withholds_its_tools(tmp_path):
    app = agent_app(tmp_path, mcp_settings=settings())
    async with app.run_test(size=(105, 35)) as pilot:
        await app.mcp.start()
        from luca.agent.contrib.tui.commands import dispatch
        from luca.agent.contrib.tui.modals import McpScreen

        await dispatch(app, "/mcp")
        await pilot.pause()
        screen = app.screen
        screen.post_message(McpScreen.Toggle(screen, screen.state.rows[0]))
        await pilot.pause()
        await pilot.pause()

        specs = await app.runner.tool_registry.get_tools(app.runner.session, app.runner.session.main_conversation_id)

        assert [spec.name for spec in specs if spec.name.startswith("mcp__")] == []
        await app.mcp.aclose()


async def test_enter_on_a_connected_server_shows_the_tools_it_offers(tmp_path):
    app = agent_app(tmp_path, mcp_settings=settings())
    async with app.run_test(size=(105, 35)) as pilot:
        await app.mcp.start()
        from luca.agent.contrib.tui.commands import dispatch

        await dispatch(app, "/mcp")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert app.screen.state.detail.label == "fx"
        assert [(row.name, row.excluded) for row in app.screen.state.detail.rows] == [("tool_0", None)]
        await app.mcp.aclose()


async def test_escape_leaves_the_tool_list_before_it_leaves_the_screen(tmp_path):
    # Each `esc` undoes the last thing that happened, so a drill-in does not
    # cost you the screen you drilled in from.
    app = agent_app(tmp_path, mcp_settings=settings())
    async with app.run_test(size=(105, 35)) as pilot:
        await app.mcp.start()
        from luca.agent.contrib.tui.commands import dispatch
        from luca.agent.contrib.tui.modals import McpScreen

        await dispatch(app, "/mcp")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()

        assert isinstance(app.screen, McpScreen)
        assert app.screen.state.detail is None
        await app.mcp.aclose()


async def test_the_screen_stays_up_while_an_action_runs(tmp_path):
    # It used to dismiss first, so authorizing meant five minutes of a
    # terminal with nothing on it saying what was being waited for.
    import asyncio

    app = agent_app(tmp_path, mcp_settings=settings())
    async with app.run_test(size=(105, 35)) as pilot:
        await app.mcp.start()
        from luca.agent.contrib.tui.commands import dispatch
        from luca.agent.contrib.tui.modals import McpScreen

        await dispatch(app, "/mcp")
        await pilot.pause()
        screen = app.screen
        held = asyncio.Event()

        async def slow(label, enabled):
            await held.wait()

        app.mcp.set_enabled = slow
        screen.post_message(McpScreen.Toggle(screen, screen.state.rows[0]))
        await pilot.pause()
        await pilot.pause()

        assert isinstance(app.screen, McpScreen)
        assert [(row.busy, row.detail) for row in app.screen.state.rows] == [(True, "disabling…")]

        held.set()
        await pilot.pause()
        await app.mcp.aclose()


async def test_escape_cancels_an_action_instead_of_closing_the_screen(tmp_path):
    import asyncio

    app = agent_app(tmp_path, mcp_settings=settings())
    async with app.run_test(size=(105, 35)) as pilot:
        await app.mcp.start()
        from luca.agent.contrib.tui.commands import dispatch
        from luca.agent.contrib.tui.modals import McpScreen

        await dispatch(app, "/mcp")
        await pilot.pause()
        screen = app.screen

        async def never(label, enabled):
            await asyncio.Event().wait()

        app.mcp.set_enabled = never
        screen.post_message(McpScreen.Toggle(screen, screen.state.rows[0]))
        await pilot.pause()
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        assert isinstance(app.screen, McpScreen)
        assert app.screen.state.message == "fx: cancelled"
        await app.mcp.aclose()


async def test_mcp_with_a_server_name_selects_that_row(tmp_path):
    # What a failed tool call's own error message tells the user to type.
    app = agent_app(tmp_path, mcp_settings=settings())
    async with app.run_test(size=(105, 35)) as pilot:
        await app.mcp.start()
        from luca.agent.contrib.tui.commands import dispatch

        await dispatch(app, "/mcp fx")
        await pilot.pause()

        assert app.screen.state.rows[app.screen.state.selected].label == "fx"
        await app.mcp.aclose()


async def test_mcp_with_an_unknown_server_says_so(tmp_path):
    app = agent_app(tmp_path, mcp_settings=settings())
    async with app.run_test(size=(105, 35)) as pilot:
        await app.mcp.start()
        from luca.agent.contrib.tui.blocks import NoticeLine
        from luca.agent.contrib.tui.commands import dispatch

        await dispatch(app, "/mcp nope")
        await pilot.pause()

        assert "No MCP server called 'nope'" in "\n".join(line.text for line in app.query(NoticeLine))
        await app.mcp.aclose()


async def test_the_first_turn_of_a_cold_run_waits_for_the_tools(tmp_path):
    # `first_listing` existed, was tested, and was called by nobody, so the
    # first message of the first run went out with no MCP tools attached.
    import asyncio

    app = agent_app(tmp_path, mcp_settings=settings())
    async with app.run_test(size=(105, 35)):
        assert app.mcp.cold is True
        connecting = asyncio.create_task(app.mcp.start())

        await app._mcp_first_listing()

        assert app.mcp.cold is False
        await connecting
        await app.mcp.aclose()


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


async def test_a_failed_action_reports_on_the_screen_not_behind_it(tmp_path):
    # The screen reopens over the transcript, so a notice posted by an action
    # is a report nobody can read. This is why an authorize that failed looked
    # like nothing had happened.
    app = agent_app(tmp_path, mcp_settings=settings())
    async with app.run_test(size=(105, 35)) as pilot:
        await app.mcp.start()
        from luca.agent.contrib.tui.commands import dispatch
        from luca.agent.contrib.tui.modals import McpScreen

        await dispatch(app, "/mcp")
        await pilot.pause()
        screen = app.screen
        screen.post_message(McpScreen.Authenticate(screen, screen.state.rows[0]))
        await pilot.pause()
        await pilot.pause()

        assert app.screen.state.message == "fx does not use oauth — it needs no sign-in"
        await app.mcp.aclose()


async def test_a_successful_action_says_what_it_did(tmp_path):
    app = agent_app(tmp_path, mcp_settings=settings())
    async with app.run_test(size=(105, 35)) as pilot:
        await app.mcp.start()
        from luca.agent.contrib.tui.commands import dispatch
        from luca.agent.contrib.tui.modals import McpScreen

        await dispatch(app, "/mcp")
        await pilot.pause()
        screen = app.screen
        screen.post_message(McpScreen.Toggle(screen, screen.state.rows[0]))
        await pilot.pause()
        await pilot.pause()

        assert app.screen.state.message == "fx disabled"
        assert app.screen.state.message_is_error is False
        await app.mcp.aclose()


async def test_a_disabled_row_offers_to_bring_it_back(tmp_path):
    from luca.agent.contrib.mcp.servers import ServerState, ServerStatus
    from luca.agent.contrib.tui.mcp_service import build_mcp_state

    state = build_mcp_state([ServerStatus(label="notion", state=ServerState.DISABLED)])

    assert [(row.state, row.detail, row.action) for row in state.rows] == [
        ("disabled", "disabled for this session", "enable")
    ]


async def test_an_empty_server_list_renders_without_a_selection(tmp_path):
    from luca.agent.contrib.tui.mcp_service import build_mcp_state

    state = build_mcp_state([])

    assert (state.count_line, state.rows, state.selected) == ("0 of 0 connected · 0 tools", [], 0)


async def test_the_selection_is_clamped_when_servers_disappear(tmp_path):
    from luca.agent.contrib.mcp.servers import ServerState, ServerStatus
    from luca.agent.contrib.tui.mcp_service import build_mcp_state

    state = build_mcp_state([ServerStatus(label="only", state=ServerState.CONNECTED)], selected=7)

    assert state.selected == 0
