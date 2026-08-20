"""The service, the durable catalog, and the registry over them.

Everything runs against the real stdio fixture, because a double would prove
nothing about the part that matters: that a listing survives the process and a
second run serves tools with no network at all.

The clock is injected wherever a TTL is involved, so no test waits for one.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from luca.agent.contrib.mcp.registry import McpToolRegistry
from luca.agent.contrib.mcp.servers import StdioServer
from luca.agent.contrib.mcp.service import McpService
from luca.agent.contrib.simple_tool_registry.permissions import YoloPermissionPolicy
from luca.agent.core import ApprovalOption, ExecutionStatus, TextContent, ToolCall, ToolNotFound

FIXTURE = str(Path(__file__).parent / "server_fixture.py")


def stdio(*args: str, label: str = "fx", **over) -> StdioServer:
    """The fixture server. `label` is keyword-only so a positional argument is
    always a fixture flag and never silently renames the server."""
    return StdioServer(label=label, command=sys.executable, args=(FIXTURE, *args), **over)


class Clock:
    """A hand-wound clock, so TTL behaviour costs no wall time."""

    def __init__(self, now: int = 1_000_000) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def watch_refresh(service: McpService) -> asyncio.Event:
    """An event that fires the next time the refresh LOOP runs a refresh.

    A test waits on the thing happening rather than on time passing, so
    nothing here races a clock.
    """
    fired = asyncio.Event()
    original = service.refresh

    async def watched(labels=None):
        await original(labels)
        fired.set()

    service.refresh = watched
    return fired


@pytest.fixture
async def service():
    """Hands out services and closes every one of them."""
    opened: list[McpService] = []

    def make(servers, **over) -> McpService:
        built = McpService(servers, **over)
        opened.append(built)
        return built

    yield make
    for built in opened:
        await built.aclose()


async def test_a_started_service_publishes_its_tools(service):
    started = service({"fx": stdio()})

    await started.start()

    assert [spec.name for spec in started.specs()] == ["mcp__fx__tool_0"]


async def test_a_paginated_server_publishes_every_page(service):
    started = service({"fx": stdio("--pages", "3")})

    await started.start()

    assert [spec.name for spec in started.specs()] == ["mcp__fx__tool_0", "mcp__fx__tool_1", "mcp__fx__tool_2"]


async def test_two_servers_are_listed_independently(service):
    started = service({"a": stdio(label="a"), "b": stdio(label="b")})

    await started.start()

    assert [spec.name for spec in started.specs()] == ["mcp__a__tool_0", "mcp__b__tool_0"]


async def test_one_unreachable_server_does_not_cost_the_others_their_tools(service):
    started = service({"ok": stdio(label="ok"), "broken": StdioServer(label="broken", command="/nonexistent/server")})

    await started.start()

    assert [spec.name for spec in started.specs()] == ["mcp__ok__tool_0"]
    assert [status.label for status in started.status() if status.error] == ["broken"]


async def test_a_listing_survives_the_process(service, tmp_path):
    catalog = tmp_path / "catalog.json"
    first = service({"fx": stdio()}, catalog_path=catalog)
    await first.start()
    await first.aclose()

    # A second run, with nothing started and no network touched at all.
    second = service({"fx": stdio()}, catalog_path=catalog)

    assert [spec.name for spec in second.specs()] == ["mcp__fx__tool_0"]
    assert second.cold is False


async def test_a_changed_command_line_discards_the_cached_listing(service, tmp_path):
    catalog = tmp_path / "catalog.json"
    first = service({"fx": stdio()}, catalog_path=catalog)
    await first.start()
    await first.aclose()

    # Same label, different server. The old tools must not be served for it.
    second = service({"fx": stdio("--pages", "2")}, catalog_path=catalog)

    assert second.specs() == []


async def test_a_service_with_no_cache_is_cold(service, tmp_path):
    assert service({"fx": stdio()}, catalog_path=tmp_path / "catalog.json").cold is True


async def test_a_service_with_no_servers_is_not_cold(service):
    # Nothing to wait for, so the first turn must not pause.
    assert service({}).cold is False


async def test_a_fresh_slice_is_not_refreshed(service, tmp_path):
    clock = Clock()
    started = service({"fx": stdio()}, catalog_path=tmp_path / "c.json", now_ms=clock)
    await started.start()

    clock.now += 30_000  # inside the fixture's 60s ttlMs

    assert started.catalog.stale(["fx"]) == []


async def test_an_expired_slice_is_due_but_still_served(service, tmp_path):
    # Stale-while-revalidate: losing tools mid-conversation because a freshness
    # hint lapsed is worse than a slightly out-of-date description.
    clock = Clock()
    started = service({"fx": stdio()}, catalog_path=tmp_path / "c.json", now_ms=clock)
    await started.start()

    clock.now += 90_000  # past the fixture's 60s ttlMs

    assert started.catalog.stale(["fx"]) == ["fx"]
    assert [spec.name for spec in started.specs()] == ["mcp__fx__tool_0"]


async def test_a_tool_call_returns_its_content(service):
    started = service({"fx": stdio()})
    await started.start()

    result = await started.call("fx", "tool_0", {})

    assert result.content == [TextContent(text="tool_0 #1")]


async def test_the_registry_serves_the_catalog_without_touching_the_network(service, tmp_path):
    catalog = tmp_path / "catalog.json"
    warm = service({"fx": stdio()}, catalog_path=catalog)
    await warm.start()
    await warm.aclose()
    # Never started, so any listing would have to happen inside get_tools.
    cold = service({"fx": stdio()}, catalog_path=catalog)
    registry = McpToolRegistry(cold, YoloPermissionPolicy())

    specs = await registry.get_tools(None, "conv-1")

    assert [spec.name for spec in specs] == ["mcp__fx__tool_0"]


async def test_an_execution_is_born_pending_with_an_approval_context(service):
    started = service({"fx": stdio()})
    await started.start()
    registry = McpToolRegistry(started, YoloPermissionPolicy())

    execution = await registry.create_execution(
        None, "conv-1", ToolCall(id="call-1", name="mcp__fx__tool_0", arguments={})
    )

    assert execution.status is ExecutionStatus.PENDING
    assert execution.extras["approval_context"]["requests"][0]["resources"] == [
        {"permission": "mcp", "resource": "fx/tool_0"}
    ]


async def test_an_unknown_tool_is_born_not_found(service):
    started = service({"fx": stdio()})
    await started.start()
    registry = McpToolRegistry(started, YoloPermissionPolicy())

    execution = await registry.create_execution(None, "conv-1", ToolCall(id="c", name="mcp__fx__nope", arguments={}))

    assert (execution.status, execution.tool_spec) == (ExecutionStatus.NOT_FOUND, None)


async def test_decide_delegates_to_the_shared_policy(service):
    started = service({"fx": stdio()})
    await started.start()
    registry = McpToolRegistry(started, YoloPermissionPolicy())
    execution = await registry.create_execution(None, "conv-1", ToolCall(id="c", name="mcp__fx__tool_0", arguments={}))

    decision = await registry.decide(None, "conv-1", execution)

    assert decision.decision is ApprovalOption.ALLOW


async def test_prepare_returns_a_callable_without_running_the_body(service):
    started = service({"fx": stdio()})
    await started.start()
    registry = McpToolRegistry(started, YoloPermissionPolicy())
    execution = await registry.create_execution(None, "conv-1", ToolCall(id="c", name="mcp__fx__tool_0", arguments={}))

    prepared = await registry.prepare(None, "conv-1", execution)

    assert callable(prepared)
    assert (await prepared(cancellation_token=None)).content == [TextContent(text="tool_0 #1")]


async def test_prepare_resolves_from_the_spec_on_a_cold_resume(service, tmp_path):
    # A call approved in a previous process, dispatched by a service that has
    # listed nothing. The identity rides on the spec, so no catalog is needed.
    catalog = tmp_path / "catalog.json"
    warm = service({"fx": stdio()}, catalog_path=catalog)
    await warm.start()
    registry = McpToolRegistry(warm, YoloPermissionPolicy())
    execution = await registry.create_execution(None, "conv-1", ToolCall(id="c", name="mcp__fx__tool_0", arguments={}))
    await warm.aclose()

    fresh = service({"fx": stdio()})  # no catalog file at all
    prepared = await McpToolRegistry(fresh, YoloPermissionPolicy()).prepare(None, "conv-1", execution)

    assert (await prepared(cancellation_token=None)).content == [TextContent(text="tool_0 #1")]


async def test_prepare_raises_for_a_tool_it_cannot_resolve(service):
    registry = McpToolRegistry(service({}), YoloPermissionPolicy())
    execution = await registry.create_execution(None, "conv-1", ToolCall(id="c", name="whatever", arguments={}))

    with pytest.raises(ToolNotFound):
        await registry.prepare(None, "conv-1", execution)


async def test_starting_twice_lists_once(service):
    started = service({"fx": stdio()})
    await started.start()

    await started.start()

    assert [spec.name for spec in started.specs()] == ["mcp__fx__tool_0"]


async def test_closing_a_service_that_never_started_is_a_no_op(service):
    await service({"fx": stdio()}).aclose()


async def test_first_listing_returns_immediately_when_warm(service, tmp_path):
    catalog = tmp_path / "catalog.json"
    warm = service({"fx": stdio()}, catalog_path=catalog)
    await warm.start()
    await warm.aclose()

    # No wait at all: there is nothing cold about this one.
    await asyncio.wait_for(service({"fx": stdio()}, catalog_path=catalog).first_listing(50), timeout=1)


async def test_first_listing_is_released_by_the_startup_listing(service):
    started = service({"fx": stdio()})
    waiting = asyncio.create_task(started.first_listing(10_000))
    await asyncio.sleep(0)

    await started.start()

    await asyncio.wait_for(waiting, timeout=5)


async def test_an_unauthorized_oauth_server_is_not_a_failure(service):
    # It has never been logged into. Reporting that in red beside a real
    # connection failure is what this state exists to avoid.
    import httpx

    from luca.agent.contrib.mcp.servers import HttpServer, ServerState

    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as client:
        started = service(
            {"remote": HttpServer(label="remote", url="https://example.test/mcp", oauth=True)},
            client=client,
        )
        await started.start()

        [status] = started.status()

        assert (status.state, status.error, status.oauth) == (ServerState.NEEDS_AUTH, None, True)


async def test_an_unreachable_server_is_inactive_and_says_why(service):
    from luca.agent.contrib.mcp.servers import ServerState

    started = service({"broken": StdioServer(label="broken", command="/nonexistent/server")})
    await started.start()

    [status] = started.status()

    assert status.state is ServerState.INACTIVE
    assert status.error


async def test_a_server_served_from_the_cache_is_connected(service, tmp_path):
    # The bug this replaced: a relaunch inside the TTL lists nothing, so no
    # connection is ever opened, and asking the transport whether it was alive
    # reported a server whose tools work perfectly as failed.
    from luca.agent.contrib.mcp.servers import ServerState

    catalog = tmp_path / "catalog.json"
    first = service({"fx": stdio()}, catalog_path=catalog)
    await first.start()
    await first.aclose()

    second = service({"fx": stdio()}, catalog_path=catalog)
    await second.start()

    [status] = second.status()

    assert status.state is ServerState.CONNECTED
    assert (status.error, status.tool_count) == (None, 1)


async def test_a_disabled_server_withholds_its_tools(service):
    from luca.agent.contrib.mcp.servers import ServerState

    started = service({"fx": stdio()})
    await started.start()

    await started.set_enabled("fx", False)

    assert started.specs() == []
    assert started.status()[0].state is ServerState.DISABLED


async def test_re_enabling_serves_the_cached_listing_again(service, tmp_path):
    started = service({"fx": stdio()}, catalog_path=tmp_path / "c.json")
    await started.start()
    await started.set_enabled("fx", False)

    await started.set_enabled("fx", True)

    assert [spec.name for spec in started.specs()] == ["mcp__fx__tool_0"]


async def test_a_disabled_tool_cannot_be_resolved_by_name(service):
    started = service({"fx": stdio()})
    await started.start()

    await started.set_enabled("fx", False)

    assert started.spec("mcp__fx__tool_0") is None


async def test_reconnecting_lists_the_server_again(service):
    started = service({"fx": stdio()})
    await started.start()

    await started.reconnect("fx")

    assert [spec.name for spec in started.specs()] == ["mcp__fx__tool_0"]


async def test_login_on_a_server_without_oauth_says_so(service):
    from luca.agent.contrib.mcp.errors import McpAuthRequired

    started = service({"fx": stdio()})

    with pytest.raises(McpAuthRequired, match="not configured for oauth"):
        await started.login("fx")


async def test_reconnecting_a_disabled_server_leaves_it_disabled(service):
    # `r` used to discard the disable as a side effect, so the row you asked to
    # reconnect came back switched on with nothing saying it had been.
    from luca.agent.contrib.mcp.servers import ServerState

    started = service({"fx": stdio()})
    await started.start()
    await started.set_enabled("fx", False)

    await started.reconnect("fx")

    assert started.status()[0].state is ServerState.DISABLED
    assert started.specs() == []


async def test_a_tool_list_change_relists_without_waiting_out_the_ttl(service, tmp_path):
    # `invalidate` marked the slice due, but the refresh loop was asleep on the
    # deadline it computed before that, so "immediately" meant "within a TTL".
    clock = Clock()
    started = service({"fx": stdio()}, catalog_path=tmp_path / "c.json", now_ms=clock)
    await started.start()
    relisted = watch_refresh(started)
    clock.now += 1

    started._tools_changed("fx")

    await asyncio.wait_for(relisted.wait(), timeout=5)
    assert started.catalog._slices["fx"]["fetched_at"] == clock.now


async def test_a_failed_refresh_over_a_cached_listing_is_stale_not_inactive(service):
    # The tools are genuinely still being offered to the model, so reporting
    # this server as merely unreachable leaves every call it fails unexplained.
    from luca.agent.contrib.mcp.servers import ServerState

    started = service({"fx": stdio()})
    await started.start()

    started._connections["fx"].error = "connection reset"

    [status] = started.status()
    assert (status.state, status.tool_count, status.error) == (ServerState.STALE, 1, "connection reset")
    assert [spec.name for spec in started.specs()] == ["mcp__fx__tool_0"]


async def test_a_server_being_listed_for_the_first_time_is_connecting(service):
    # Opening `/mcp` during boot used to paint every server red, which is a
    # verdict on a question that has not finished being asked.
    from luca.agent.contrib.mcp.servers import ServerState

    started = service({"fx": stdio()})
    started._listing.add("fx")

    [status] = started.status()

    assert (status.state, status.error) == (ServerState.CONNECTING, None)


# ── the oauth call path ───────────────────────────────────────────────────────
#
# A lapsed token used to be nobody's to renew: `_list` refreshed, `call` did
# not, so the call failed and the server went on reporting itself connected.

REMOTE_URL = "https://example.test/mcp"
_DISCOVER = {
    "supportedVersions": ["2026-07-28"],
    "capabilities": {"tools": {}},
    "resultType": "complete",
    "ttlMs": 60000,
    "cacheScope": "public",
}
_TOOLS = {
    "tools": [{"name": "search", "inputSchema": {"type": "object"}}],
    "resultType": "complete",
    "ttlMs": 60000,
    "cacheScope": "public",
}
_CALL = {"content": [{"type": "text", "text": "found"}], "resultType": "complete"}


class RemoteWithAuth:
    """One handler for both halves of an authorized server: the OAuth metadata
    and token endpoints, and the MCP endpoint itself.

    `reject` makes the MCP endpoint answer 401 however good the token is, which
    is how a revoked grant looks from here.
    """

    def __init__(self, *, reject: bool = False) -> None:
        self.reject = reject
        self.tokens: list[dict] = []

    def __call__(self, request):
        import json

        import httpx

        path = request.url.path
        if path.endswith("/.well-known/oauth-protected-resource/mcp"):
            return httpx.Response(200, json={"authorization_servers": ["https://auth.example.test"]})
        if "oauth-authorization-server" in path:
            return httpx.Response(
                200,
                json={
                    "issuer": "https://auth.example.test",
                    "authorization_endpoint": "https://auth.example.test/authorize",
                    "token_endpoint": "https://auth.example.test/token",
                },
            )
        if path.endswith("/token"):
            self.tokens.append(dict(pair.split("=", 1) for pair in request.content.decode().split("&")))
            return httpx.Response(200, json={"access_token": "fresh", "expires_in": 3600})
        if self.reject:
            return httpx.Response(401, json={"error": "unauthorized"})
        body = json.loads(request.content)
        results = {"server/discover": _DISCOVER, "tools/list": _TOOLS, "tools/call": _CALL}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": results[body["method"]]})


async def redirect_back(url: str) -> None:
    """Stand in for the human at the browser: read the state off the
    authorization URL and complete the redirect, as the server would."""
    import httpx

    query = httpx.URL(url).params
    async with httpx.AsyncClient() as agent:
        await agent.get(
            query["redirect_uri"],
            params={"code": "the-code", "state": query["state"], "iss": "https://auth.example.test"},
        )


async def test_a_call_renews_a_token_that_lapsed_since_the_last_listing(service):
    import httpx

    from luca.agent.contrib.mcp.oauth.store import OAuthToken
    from luca.agent.contrib.mcp.servers import HttpServer

    handler = RemoteWithAuth()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        started = service(
            {"remote": HttpServer(label="remote", url=REMOTE_URL, oauth=True, client_id="fixed-client")}, client=client
        )
        await started.start()
        provider = started._auth["remote"]
        provider._token = OAuthToken(access_token="stale", refresh_token="r0", expires_at=0)

        result = await started.call("remote", "search", {})

        assert result.content == [TextContent(text="found")]
        assert [form["grant_type"] for form in handler.tokens] == ["refresh_token"]


async def test_a_slice_listed_under_another_login_is_dropped(service, tmp_path):
    # `HttpServer.credential_fingerprint` hashes configured headers, and an
    # OAuth token is not configuration, so two accounts on one URL look
    # identical to `_applies` and the previous one's tools were served.
    import httpx

    from luca.agent.contrib.mcp.oauth.store import OAuthToken
    from luca.agent.contrib.mcp.servers import HttpServer

    catalog = tmp_path / "catalog.json"
    remote = {"remote": HttpServer(label="remote", url=REMOTE_URL, oauth=True, client_id="fixed-client")}
    async with httpx.AsyncClient(transport=httpx.MockTransport(RemoteWithAuth())) as client:
        first = service(remote, client=client, catalog_path=catalog)
        first._auth["remote"]._token = OAuthToken(access_token="account-a")
        first._auth["remote"]._loaded = True
        await first.start()
        assert [spec.name for spec in first.specs()] == ["mcp__remote__search"]
        await first.aclose()

        # A second run, same URL and same config, signed in as somebody else.
        second = service(remote, client=client, catalog_path=catalog)
        second._auth["remote"]._token = OAuthToken(access_token="account-b")
        second._auth["remote"]._loaded = True
        assert [spec.name for spec in second.specs()] == ["mcp__remote__search"]  # adopted at boot

        second.catalog.verify_credential("remote", second._auth["remote"].credential_fingerprint())

        assert second.specs() == []


async def test_a_slice_listed_under_the_same_login_survives(service, tmp_path):
    import httpx

    from luca.agent.contrib.mcp.oauth.store import OAuthToken
    from luca.agent.contrib.mcp.servers import HttpServer

    catalog = tmp_path / "catalog.json"
    remote = {"remote": HttpServer(label="remote", url=REMOTE_URL, oauth=True, client_id="fixed-client")}
    async with httpx.AsyncClient(transport=httpx.MockTransport(RemoteWithAuth())) as client:
        first = service(remote, client=client, catalog_path=catalog)
        first._auth["remote"]._token = OAuthToken(access_token="account-a")
        first._auth["remote"]._loaded = True
        await first.start()
        await first.aclose()

        second = service(remote, client=client, catalog_path=catalog)
        second._auth["remote"]._token = OAuthToken(access_token="account-a")
        second._auth["remote"]._loaded = True

        second.catalog.verify_credential("remote", second._auth["remote"].credential_fingerprint())

        assert [spec.name for spec in second.specs()] == ["mcp__remote__search"]


async def test_a_login_whose_token_is_then_refused_does_not_report_success(service):
    # What `/mcp` used to say when Linear minted a token its own resource
    # server rejected: "linear authorized", with no tools and nothing to
    # explain why. `refresh` records that failure rather than raising it.
    import httpx

    from luca.agent.contrib.mcp.errors import McpAuthRequired
    from luca.agent.contrib.mcp.servers import HttpServer

    handler = RemoteWithAuth(reject=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        started = service(
            {"remote": HttpServer(label="remote", url=REMOTE_URL, oauth=True, client_id="fixed-client")},
            client=client,
            browser=redirect_back,
        )

        with pytest.raises(McpAuthRequired, match="issued a token and then refused it"):
            await started.login("remote")


async def test_a_call_refused_with_a_401_stops_the_server_claiming_to_be_connected(service):
    import httpx

    from luca.agent.contrib.mcp.errors import McpAuthRequired
    from luca.agent.contrib.mcp.oauth.store import OAuthToken
    from luca.agent.contrib.mcp.servers import HttpServer, ServerState

    handler = RemoteWithAuth()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        started = service(
            {"remote": HttpServer(label="remote", url=REMOTE_URL, oauth=True, client_id="fixed-client")}, client=client
        )
        await started.start()
        started._auth["remote"]._token = OAuthToken(access_token="revoked")
        handler.reject = True

        with pytest.raises(McpAuthRequired):
            await started.call("remote", "search", {})

        assert started.status()[0].state is ServerState.NEEDS_AUTH
