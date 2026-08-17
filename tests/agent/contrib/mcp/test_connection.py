"""One long-lived connection per server, against a real stdio subprocess.

Every test that starts a server must close it: the suite runs with
`-W error::ResourceWarning`, so a leaked child or pipe fails the build rather
than lingering. The `connect` fixture is the only door.

Nothing here races two timed things. Cancellation is asserted by asking the
server what it received, restart by observing that the pid changed, and
concurrency by the order the answers come back — all facts, not durations.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from luca.agent.contrib.mcp.connection import ServerConnection
from luca.agent.contrib.mcp.errors import McpError, McpServerGone, McpUnsupportedVersion
from luca.agent.contrib.mcp.servers import StdioServer
from luca.agent.contrib.mcp.wire import Era

FIXTURE = str(Path(__file__).parent / "server_fixture.py")
RECEIVED = "__received__"  # the fixture's introspection tool


def server(label: str = "fixture", *args: str, **over) -> StdioServer:
    return StdioServer(label=label, command=sys.executable, args=(FIXTURE, *args), **over)


@pytest.fixture
async def connect():
    """Hands out connections and closes every one of them."""
    opened: list[ServerConnection] = []

    def make(*args: str, **over) -> ServerConnection:
        connection = ServerConnection(server("fixture", *args, **over))
        opened.append(connection)
        return connection

    yield make
    for connection in opened:
        await connection.aclose()


async def test_a_modern_server_is_discovered(connect):
    negotiated = await connect().connect()

    assert (negotiated.era, negotiated.protocol_version) == (Era.MODERN, "2026-07-28")


async def test_a_legacy_server_falls_back_to_the_handshake(connect):
    # The fixture answers `server/discover` with -32601, which is what a
    # pre-2026 server does and is deliberately not a spec-reserved code.
    negotiated = await connect("--era", "legacy").connect()

    assert (negotiated.era, negotiated.protocol_version) == (Era.HANDSHAKE, "2025-06-18")


async def test_a_server_that_never_answers_the_probe_falls_back(connect):
    negotiated = await connect("--era", "silent", connect_timeout_in_ms=300).connect()

    assert negotiated.era is Era.HANDSHAKE


async def test_a_version_mismatch_is_renegotiated_rather_than_downgraded(connect):
    # A spec-reserved code proves the peer is modern, so the client retries in
    # the version it named instead of falling back to `initialize`.
    negotiated = await connect("--demand-version", "2026-07-28").connect()

    assert negotiated.era is Era.MODERN


async def test_the_handshake_is_completed_before_the_connection_is_used(connect):
    connection = connect("--era", "legacy")
    await connection.connect()

    received = await connection.call_tool(RECEIVED, {})

    assert [message["method"] for message in received.structured_content["received"]] == ["notifications/initialized"]


async def test_connecting_twice_probes_once(connect):
    connection = connect()
    first = await connection.connect()

    assert await connection.connect() is first


async def test_concurrent_first_calls_share_one_probe(connect):
    connection = connect()

    results = await asyncio.gather(connection.connect(), connection.connect(), connection.connect())

    assert results[0] is results[1] is results[2]


async def test_a_listing_follows_every_page(connect):
    # The previous attempt stopped after page one, and a paginated server
    # showed as connected with a wrong tool count and no error anywhere.
    tools, _ = await connect("--pages", "3").list_tools()

    assert [tool.name for tool in tools] == ["tool_0", "tool_1", "tool_2"]


async def test_a_listing_carries_the_servers_cache_hints(connect):
    _, hints = await connect("--ttl-ms", "45000").list_tools()

    assert hints == (45000, "public")


async def test_a_tool_call_returns_its_content(connect):
    result = await connect().call_tool("greet", {})

    assert [block.text for block in result.content] == ["greet #1"]


async def test_requests_are_multiplexed_rather_than_serialized(connect):
    # The slow call is issued first and answers last, which can only happen if
    # both were in flight on the one channel at the same time.
    connection = connect("--delay-tool", "slow", "--delay-seconds", "0.3")

    slow = asyncio.create_task(connection.call_tool("slow", {}))
    await asyncio.sleep(0)
    fast = await connection.call_tool("fast", {})

    assert [block.text for block in fast.content] == ["fast #2"]
    assert [block.text for block in (await slow).content] == ["slow #1"]


async def test_cancelling_a_call_tells_the_server_to_stop(connect):
    connection = connect("--delay-tool", "slow", "--delay-seconds", "5")
    call = asyncio.create_task(connection.call_tool("slow", {}))
    await asyncio.sleep(0.1)  # let the request reach the server before cancelling

    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    received = await connection.call_tool(RECEIVED, {})

    assert [message["method"] for message in received.structured_content["received"]] == ["notifications/cancelled"]


async def test_a_server_reported_error_keeps_its_code(connect):
    # `prompts/list` is a real method the fixture does not implement, so the
    # answer is the server's own -32601 rather than a local refusal.
    connection = connect()

    with pytest.raises(McpError) as caught:
        await connection.request("prompts/list")

    assert caught.value.code == -32601


async def test_a_2026_only_method_is_refused_locally_on_a_legacy_server(connect):
    # Sending it would be worse than refusing: a legacy server that does not
    # check for `initialize` would answer it under legacy semantics.
    connection = connect("--era", "legacy")

    with pytest.raises(McpUnsupportedVersion):
        await connection.request("subscriptions/listen")


async def test_a_crashed_server_fails_the_call_rather_than_repeating_it(connect, tmp_path):
    # `tools/call` is never auto-retried: a tool may have side effects, and
    # running someone's create_issue twice is worse than an honest failure.
    connection = connect("--crash-marker", str(tmp_path / "crashed"))

    with pytest.raises(McpServerGone):
        await connection.call_tool("boom", {})


async def test_the_next_call_after_a_crash_spawns_a_fresh_server(connect, tmp_path):
    connection = connect("--crash-marker", str(tmp_path / "crashed"))
    with pytest.raises(McpServerGone):
        await connection.call_tool("boom", {})

    result = await connection.call_tool("fine", {})

    assert [block.text for block in result.content] == ["fine #1"]  # a fresh process, counting from one


async def test_closing_a_connection_that_never_connected_is_a_no_op(connect):
    await connect().aclose()


async def test_closing_twice_is_a_no_op(connect):
    connection = connect()
    await connection.connect()

    await connection.aclose()
    await connection.aclose()


async def test_a_call_in_flight_when_the_connection_closes_is_failed(connect):
    connection = connect("--delay-tool", "slow", "--delay-seconds", "5")
    call = asyncio.create_task(connection.call_tool("slow", {}))
    await asyncio.sleep(0.1)

    await connection.aclose()

    with pytest.raises(McpServerGone):
        await call


async def test_a_status_reports_the_negotiated_version(connect):
    connection = connect()
    await connection.connect()

    # The CONNECTION reports its own facts only; the state a user sees is the
    # service's, because it depends on the catalog.
    assert connection.status().model_dump() == {
        "label": "fixture",
        "state": "inactive",
        "oauth": False,
        "protocol_version": "2026-07-28",
        "tool_count": 0,
        "error": None,
        "rejected_tools": {},
    }


async def test_a_server_that_cannot_start_records_why(connect):
    connection = ServerConnection(StdioServer(label="broken", command="/nonexistent/mcp-server"))

    with pytest.raises(McpServerGone):
        await connection.connect()

    assert "broken" in connection.error
