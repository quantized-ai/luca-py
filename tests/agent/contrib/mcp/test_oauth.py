"""Browser OAuth: the pure rules, the loopback listener, and the provider.

The three findings the previous attempt drew are each pinned by a test here:
the callback binds a real port instead of deriving one from the label, the
browser opens off the event loop, and concurrent refreshes collapse into one.

Nothing sleeps waiting for the listener: `start()` returns the bound port, so
readiness is a return value rather than a duration.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from luca.agent.contrib.mcp.errors import McpAuthRequired
from luca.agent.contrib.mcp.oauth import (
    IssuerMismatch,
    LoopbackCallback,
    OAuthProvider,
    OAuthToken,
    Pkce,
    TokenStore,
    check_issuer,
    metadata_urls,
    new_state,
    protected_resource_url,
    state_matches,
)
from luca.agent.contrib.mcp.oauth.metadata import canonical_resource, parse_challenge
from luca.agent.contrib.mcp.oauth.store import ClientRegistration, IssuerRecord
from luca.agent.contrib.mcp.servers import HttpServer

ISSUER = "https://auth.example.test"
SERVER = HttpServer(label="remote", url="https://api.example.test/mcp", oauth=True, client_id="fixed-client")


def test_a_verifier_and_its_challenge_are_the_s256_pair():
    import base64
    import hashlib

    pkce = Pkce.create()
    expected = base64.urlsafe_b64encode(hashlib.sha256(pkce.verifier.encode()).digest()).decode().rstrip("=")

    assert (pkce.challenge, pkce.method) == (expected, "S256")


def test_two_verifiers_are_never_the_same():
    assert Pkce.create().verifier != Pkce.create().verifier


def test_a_state_matches_itself():
    state = new_state()

    assert state_matches(state, state) is True


def test_a_different_state_does_not_match():
    assert state_matches(new_state(), new_state()) is False


def test_an_absent_issuer_is_allowed():
    # SHOULD-level for servers, so refusing every one that predates RFC 9207
    # would lock people out for no security gain.
    check_issuer(ISSUER, None)


def test_a_matching_issuer_passes():
    check_issuer(ISSUER, ISSUER)


def test_a_trailing_slash_is_a_mismatch():
    # RFC 9207 forbids trailing-slash, case and percent-encoding normalization
    # before this comparison: each one is a way for two different issuers to
    # look equal.
    with pytest.raises(IssuerMismatch):
        check_issuer(ISSUER, ISSUER + "/")


def test_an_absent_iss_is_refused_when_the_server_advertises_it():
    with pytest.raises(IssuerMismatch, match="did not send one"):
        check_issuer(ISSUER, None, advertised=True)


def test_a_different_issuer_stops_the_code_being_redeemed():
    # The mix-up attack: a code minted by somebody else, handed to the wrong
    # token endpoint.
    with pytest.raises(IssuerMismatch):
        check_issuer(ISSUER, "https://evil.example.test")


def test_the_protected_resource_url_keeps_the_resource_path():
    assert protected_resource_url("https://api.example.test/mcp") == (
        "https://api.example.test/.well-known/oauth-protected-resource/mcp"
    )


def test_metadata_is_looked_for_in_every_conventional_place():
    assert metadata_urls("https://auth.example.test/tenant") == [
        "https://auth.example.test/.well-known/oauth-authorization-server/tenant",
        "https://auth.example.test/.well-known/openid-configuration/tenant",
        "https://auth.example.test/tenant/.well-known/openid-configuration",
    ]


async def test_the_callback_binds_a_real_port_and_says_which():
    # The previous attempt derived the port from a CRC of the server label, so
    # two flows collided, and it slept because nothing told it when the socket
    # was ready. Both problems are this one line.
    async with LoopbackCallback() as callback:
        assert callback.port > 0
        assert callback.redirect_uri == f"http://127.0.0.1:{callback.port}/callback"


async def test_two_callbacks_never_take_the_same_port():
    async with LoopbackCallback() as first, LoopbackCallback() as second:
        assert first.port != second.port


async def test_the_callback_hands_back_the_query_it_was_redirected_with():
    async with LoopbackCallback() as callback, httpx.AsyncClient() as client:
        waiting = asyncio.create_task(callback.wait(timeout_s=5))
        await client.get(callback.redirect_uri, params={"code": "abc", "state": "xyz"})

        assert await waiting == {"code": "abc", "state": "xyz"}


async def test_the_callback_reports_an_error_response_too():
    async with LoopbackCallback() as callback, httpx.AsyncClient() as client:
        waiting = asyncio.create_task(callback.wait(timeout_s=5))
        await client.get(callback.redirect_uri, params={"error": "access_denied"})

        assert await waiting == {"error": "access_denied"}


async def test_a_token_store_round_trips_through_a_file(tmp_path):
    store = TokenStore(tmp_path / "mcp-auth.json")
    await store.put(ISSUER, IssuerRecord(token=OAuthToken(access_token="t1", refresh_token="r1")))

    reread = TokenStore(tmp_path / "mcp-auth.json")

    assert (await reread.get(ISSUER)).token == OAuthToken(access_token="t1", refresh_token="r1")


async def test_a_token_store_is_keyed_by_issuer_not_by_label(tmp_path):
    # Two labels on one authorization server share a login, and renaming a
    # label does not orphan the token.
    store = TokenStore(tmp_path / "mcp-auth.json")
    await store.put(ISSUER, IssuerRecord(token=OAuthToken(access_token="t1")))

    assert (await store.get(ISSUER + "/")).token.access_token == "t1"


async def test_the_token_file_is_not_world_readable(tmp_path):
    path = tmp_path / "mcp-auth.json"
    await TokenStore(path).put(ISSUER, IssuerRecord(token=OAuthToken(access_token="t1")))

    assert path.stat().st_mode & 0o077 == 0


async def test_an_unreadable_store_is_treated_as_empty(tmp_path):
    path = tmp_path / "mcp-auth.json"
    path.write_text("{not json")

    assert (await TokenStore(path).get(ISSUER)) == IssuerRecord()


def test_a_token_with_no_expiry_never_expires():
    assert OAuthToken(access_token="t").expired(now=10**12) is False


def test_a_token_is_refreshed_before_it_actually_expires():
    # A call must not race its own credential over the wire.
    assert OAuthToken(access_token="t", expires_at=1000).expired(now=960) is True


def test_a_token_response_becomes_an_absolute_expiry():
    token = OAuthToken.from_response({"access_token": "t", "expires_in": 3600}, now=1000)

    assert token.expires_at == 4600


class AuthServer:
    """A minimal authorization server, and a count of what it was asked."""

    def __init__(self) -> None:
        self.token_requests: list[dict] = []
        self.rotation = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/.well-known/oauth-protected-resource/mcp"):
            return httpx.Response(200, json={"authorization_servers": [ISSUER]})
        if "oauth-authorization-server" in path:
            return httpx.Response(
                200,
                json={
                    "issuer": ISSUER,
                    "authorization_endpoint": f"{ISSUER}/authorize",
                    "token_endpoint": f"{ISSUER}/token",
                },
            )
        if path.endswith("/token"):
            form = dict(pair.split("=", 1) for pair in request.content.decode().split("&"))
            self.token_requests.append(form)
            self.rotation += 1
            return httpx.Response(
                200,
                json={
                    "access_token": f"access-{self.rotation}",
                    "refresh_token": f"refresh-{self.rotation}",
                    "expires_in": 3600,
                },
            )
        return httpx.Response(404, json={})


@pytest.fixture
async def auth_client():
    server = AuthServer()
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        yield server, client


async def test_concurrent_refreshes_hit_the_token_endpoint_once(auth_client):
    # Two refreshes racing each redeem the same rotating refresh token, and
    # whichever lands second is permanently invalid. The single-flight future
    # is what stops it.
    server, client = auth_client
    provider = OAuthProvider(SERVER, store=TokenStore(None))
    provider._token = OAuthToken(access_token="old", refresh_token="r0", expires_at=0)
    provider._loaded = True

    await asyncio.gather(*(provider.ensure_token(client, interactive=False) for _ in range(5)))

    assert len(server.token_requests) == 1


async def test_a_refresh_keeps_a_refresh_token_the_server_did_not_rotate(auth_client):
    server, _ = auth_client
    provider = OAuthProvider(SERVER, store=TokenStore(None))
    provider._token = OAuthToken(access_token="old", refresh_token="keep-me", expires_at=0)
    provider._loaded = True

    def no_rotation(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/token"):
            return httpx.Response(200, json={"access_token": "new", "expires_in": 3600})
        return server(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(no_rotation)) as plain:
        await provider.ensure_token(plain, interactive=False)

    assert provider._token.refresh_token == "keep-me"


async def test_a_call_never_opens_a_browser(auth_client):
    # The previous attempt authorized inside `get_tools`, so a browser window
    # opened in the middle of somebody's first message.
    _, client = auth_client
    opened: list[str] = []
    provider = OAuthProvider(SERVER, store=TokenStore(None), browser=lambda url: opened.append(url))

    with pytest.raises(McpAuthRequired, match="/mcp login remote"):
        await provider.ensure_token(client, interactive=False)

    assert opened == []


async def test_a_valid_token_needs_no_network_at_all(auth_client):
    server, client = auth_client
    provider = OAuthProvider(SERVER, store=TokenStore(None))
    provider._token = OAuthToken(access_token="good", expires_at=10**12)
    provider._loaded = True

    await provider.ensure_token(client, interactive=False)

    assert server.token_requests == []


async def test_the_token_is_attached_to_outgoing_requests():
    provider = OAuthProvider(SERVER, store=TokenStore(None))
    provider._token = OAuthToken(access_token="good")

    request = next(provider.auth_flow(httpx.Request("POST", SERVER.url)))

    assert request.headers["authorization"] == "Bearer good"


async def test_an_interactive_login_completes_the_whole_flow(auth_client, tmp_path):
    server, client = auth_client
    store = TokenStore(tmp_path / "mcp-auth.json")

    async def browser(url: str) -> None:
        # Stand in for the human: read the state off the authorization URL and
        # redirect back, exactly as the authorization server would.
        query = httpx.URL(url).params
        async with httpx.AsyncClient() as agent:
            await agent.get(query["redirect_uri"], params={"code": "the-code", "state": query["state"], "iss": ISSUER})

    provider = OAuthProvider(SERVER, store=store, browser=browser)
    await provider.authorize(client)

    assert provider._token.access_token == "access-1"
    assert server.token_requests[0]["grant_type"] == "authorization_code"
    assert (await store.get(ISSUER)).token.access_token == "access-1"


async def test_a_login_verifies_pkce_and_the_issuer_before_redeeming(auth_client):
    server, client = auth_client
    seen: dict = {}

    async def browser(url: str) -> None:
        query = httpx.URL(url).params
        seen.update(query)
        async with httpx.AsyncClient() as agent:
            await agent.get(query["redirect_uri"], params={"code": "c", "state": query["state"], "iss": ISSUER})

    await OAuthProvider(SERVER, store=TokenStore(None), browser=browser).authorize(client)

    assert seen["code_challenge_method"] == "S256"
    assert "code_verifier" in server.token_requests[0]


async def test_a_login_from_the_wrong_issuer_is_refused(auth_client):
    _, client = auth_client

    async def browser(url: str) -> None:
        query = httpx.URL(url).params
        async with httpx.AsyncClient() as agent:
            await agent.get(
                query["redirect_uri"],
                params={"code": "c", "state": query["state"], "iss": "https://evil.example.test"},
            )

    with pytest.raises(IssuerMismatch):
        await OAuthProvider(SERVER, store=TokenStore(None), browser=browser).authorize(client)


async def test_a_login_with_a_forged_state_is_refused(auth_client):
    _, client = auth_client

    async def browser(url: str) -> None:
        query = httpx.URL(url).params
        async with httpx.AsyncClient() as agent:
            await agent.get(query["redirect_uri"], params={"code": "c", "state": "forged"})

    with pytest.raises(McpAuthRequired, match="wrong state"):
        await OAuthProvider(SERVER, store=TokenStore(None), browser=browser).authorize(client)


async def test_a_denied_login_reports_what_the_server_said(auth_client):
    _, client = auth_client

    async def browser(url: str) -> None:
        query = httpx.URL(url).params
        async with httpx.AsyncClient() as agent:
            await agent.get(
                query["redirect_uri"],
                params={"error": "access_denied", "error_description": "the user said no"},
            )

    with pytest.raises(McpAuthRequired, match="the user said no"):
        await OAuthProvider(SERVER, store=TokenStore(None), browser=browser).authorize(client)


async def test_a_dynamic_registration_declares_itself_native(auth_client):
    # Without `application_type: native` an OpenID provider applies web
    # redirect rules and rejects the loopback URI this client has to use.
    registered: list[dict] = []
    server, _ = auth_client

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/register"):
            registered.append(json.loads(request.content))
            return httpx.Response(200, json={"client_id": "dynamic-id"})
        if "oauth-authorization-server" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "issuer": ISSUER,
                    "authorization_endpoint": f"{ISSUER}/authorize",
                    "token_endpoint": f"{ISSUER}/token",
                    "registration_endpoint": f"{ISSUER}/register",
                },
            )
        return server(request)

    async def browser(url: str) -> None:
        query = httpx.URL(url).params
        async with httpx.AsyncClient() as agent:
            await agent.get(query["redirect_uri"], params={"code": "c", "state": query["state"], "iss": ISSUER})

    unregistered = HttpServer(label="remote", url="https://api.example.test/mcp", oauth=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await OAuthProvider(unregistered, store=TokenStore(None), browser=browser).authorize(client)

    assert registered[0]["application_type"] == "native"
    assert registered[0]["redirect_uris"][0].startswith("http://127.0.0.1:")


async def test_a_stored_registration_is_reused(tmp_path):
    store = TokenStore(tmp_path / "mcp-auth.json")
    await store.put(ISSUER, IssuerRecord(registration=ClientRegistration(client_id="remembered")))
    provider = OAuthProvider(HttpServer(label="r", url="https://api.example.test/mcp", oauth=True), store=store)
    provider._issuer = ISSUER

    await provider._load()

    assert provider._client_id() == "remembered"


# ── RFC 8707 resource indicators ──────────────────────────────────────────────


def test_the_canonical_resource_lowercases_and_drops_a_trailing_slash():
    assert canonical_resource("https://MCP.Linear.app/mcp/") == "https://mcp.linear.app/mcp"


def test_the_canonical_resource_drops_a_fragment():
    # The spec lists a fragment as making a resource URI invalid.
    assert canonical_resource("https://mcp.example.com/mcp#x") == "https://mcp.example.com/mcp"


def test_a_challenge_is_parsed_into_its_parameters():
    header = 'Bearer resource_metadata="https://x/.well-known/oauth-protected-resource", scope="read write"'

    assert parse_challenge(header) == {
        "resource_metadata": "https://x/.well-known/oauth-protected-resource",
        "scope": "read write",
    }


def test_no_challenge_header_parses_to_nothing():
    assert parse_challenge(None) == {}


async def test_the_resource_is_sent_on_both_the_authorization_and_token_requests(auth_client):
    # The MUST that made a completed login still answer 401: without it the
    # authorization server mints a token with no audience, and the MCP server
    # is required to refuse it.
    server, client = auth_client
    seen: dict = {}

    async def browser(url: str) -> None:
        query = httpx.URL(url).params
        seen.update(query)
        async with httpx.AsyncClient() as agent:
            await agent.get(query["redirect_uri"], params={"code": "c", "state": query["state"], "iss": ISSUER})

    await OAuthProvider(SERVER, store=TokenStore(None), browser=browser).authorize(client)

    assert seen["resource"] == "https://api.example.test/mcp"
    assert server.token_requests[0]["resource"] == "https%3A%2F%2Fapi.example.test%2Fmcp"


async def test_a_refresh_also_carries_the_resource(auth_client):
    server, client = auth_client
    provider = OAuthProvider(SERVER, store=TokenStore(None))
    provider._token = OAuthToken(access_token="old", refresh_token="r0", expires_at=0)
    provider._loaded = True

    await provider.ensure_token(client, interactive=False)

    assert server.token_requests[0]["resource"] == "https%3A%2F%2Fapi.example.test%2Fmcp"


async def test_scopes_come_from_the_protected_resource_not_the_authorization_server():
    # Two documents share the field name and mean different things. The
    # resource's list is what THIS server needs.
    def handler(request: httpx.Request) -> httpx.Response:
        if "oauth-protected-resource" in request.url.path:
            return httpx.Response(
                200,
                json={"authorization_servers": [ISSUER], "scopes_supported": ["issues:read"]},
            )
        if "oauth-authorization-server" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "issuer": ISSUER,
                    "authorization_endpoint": f"{ISSUER}/authorize",
                    "token_endpoint": f"{ISSUER}/token",
                    "scopes_supported": ["something:else"],
                },
            )
        return httpx.Response(200, json={"access_token": "a", "expires_in": 60})

    seen: dict = {}

    async def browser(url: str) -> None:
        query = httpx.URL(url).params
        seen.update(query)
        async with httpx.AsyncClient() as agent:
            await agent.get(query["redirect_uri"], params={"code": "c", "state": query["state"], "iss": ISSUER})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await OAuthProvider(SERVER, store=TokenStore(None), browser=browser).authorize(client)

    assert seen["scope"] == "issues:read"


async def test_a_challenge_scope_wins_over_the_advertised_one(auth_client):
    _, client = auth_client
    seen: dict = {}

    async def browser(url: str) -> None:
        query = httpx.URL(url).params
        seen.update(query)
        async with httpx.AsyncClient() as agent:
            await agent.get(query["redirect_uri"], params={"code": "c", "state": query["state"], "iss": ISSUER})

    provider = OAuthProvider(SERVER, store=TokenStore(None), browser=browser)
    provider.observe_challenge({"scope": "issues:write"})
    await provider.authorize(client)

    assert seen["scope"] == "issues:write"


async def test_a_login_produces_a_token_the_mcp_server_accepts(tmp_path):
    """The end-to-end regression: authorize, then list, against a server that
    enforces audience binding the way the spec requires of it.

    Without the `resource` parameter the authorization server mints a token
    with no audience, the MCP server refuses it, and the login looks like it
    silently did nothing — the browser says Authorized and the client still
    says not authenticated.
    """
    import json

    from luca.agent.contrib.mcp.servers import HttpServer, ServerState
    from luca.agent.contrib.mcp.service import McpService

    MCP_URL = "https://mcp.example.test/mcp"
    RESOURCE = "https://mcp.example.test/mcp"
    issued: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "oauth-protected-resource" in path:
            return httpx.Response(
                200,
                json={"resource": RESOURCE, "authorization_servers": [ISSUER], "scopes_supported": ["read"]},
            )
        if "oauth-authorization-server" in path:
            return httpx.Response(
                200,
                json={
                    "issuer": ISSUER,
                    "authorization_endpoint": f"{ISSUER}/authorize",
                    "token_endpoint": f"{ISSUER}/token",
                },
            )
        if path.endswith("/token"):
            form = dict(pair.split("=", 1) for pair in request.content.decode().split("&"))
            # The audience the token is minted for, exactly as RFC 8707 says.
            issued["audience"] = form.get("resource", "")
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        # The MCP endpoint: refuse any token not bound to this resource.
        from urllib.parse import quote

        if request.headers.get("authorization") != "Bearer tok" or issued.get("audience") != quote(RESOURCE, safe=""):
            return httpx.Response(
                401,
                json={},
                headers={"WWW-Authenticate": f'Bearer resource_metadata="{MCP_URL}", scope="read"'},
            )
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "supportedVersions": ["2026-07-28"],
                    "capabilities": {"tools": {}},
                    "resultType": "complete",
                    "ttlMs": 60000,
                    "cacheScope": "private",
                }
                if body["method"] == "server/discover"
                else {
                    "tools": [{"name": "search", "inputSchema": {"type": "object"}}],
                    "resultType": "complete",
                    "ttlMs": 60000,
                    "cacheScope": "private",
                },
            },
        )

    async def browser(url: str) -> None:
        query = httpx.URL(url).params
        async with httpx.AsyncClient() as agent:
            await agent.get(query["redirect_uri"], params={"code": "c", "state": query["state"], "iss": ISSUER})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = McpService(
            {"remote": HttpServer(label="remote", url=MCP_URL, oauth=True, client_id="fixed")},
            client=client,
            token_path=tmp_path / "tokens.json",
            browser=browser,
        )
        await service.start()
        assert service.status()[0].state is ServerState.NEEDS_AUTH

        await service.login("remote")

        assert service.status()[0].state is ServerState.CONNECTED
        assert [spec.name for spec in service.specs()] == ["mcp__remote__search"]
        await service.aclose()
