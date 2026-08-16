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


def test_a_trailing_slash_is_not_a_mismatch():
    check_issuer(ISSUER, ISSUER + "/")


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
