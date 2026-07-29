"""OAuth token storage and the localhost redirect capture."""

import asyncio
import stat
import threading

import httpx
from mcp.shared.auth import OAuthToken

from luca.agent.contrib.mcp import oauth
from luca.agent.contrib.mcp.config import HttpServer
from luca.agent.contrib.mcp.oauth import (
    FileTokenStorage,
    capture_authorization_code,
    make_auth_factory,
)
from luca.agent.contrib.mcp.registry import McpToolRegistry
from luca.agent.contrib.simple_tool_registry import YoloPermissionPolicy


async def test_the_token_store_round_trips_and_isolates_servers(tmp_path):
    store = FileTokenStorage(tmp_path, "srv")
    assert await store.get_tokens() is None
    await store.set_tokens(OAuthToken(access_token="a", token_type="Bearer", refresh_token="r"))
    assert await store.get_tokens() == OAuthToken(access_token="a", token_type="Bearer", refresh_token="r")
    # a different server shares the file but not the section
    assert await FileTokenStorage(tmp_path, "other").get_tokens() is None


async def test_the_token_file_is_owner_only(tmp_path):
    # it holds a long-lived refresh token in plaintext: anyone who can read it
    # keeps access to that server until the token is revoked
    store = FileTokenStorage(tmp_path, "srv")

    await store.set_tokens(OAuthToken(access_token="a", token_type="Bearer", refresh_token="r"))

    assert stat.S_IMODE((tmp_path / "mcp-auth.json").stat().st_mode) == 0o600


async def test_an_existing_token_file_is_tightened_on_the_next_write(tmp_path):
    # a file written before this rule (or by something else) keeps its old mode
    # through `write_text`, so the chmod has to run on every write, not just
    # when the file is created
    path = tmp_path / "mcp-auth.json"
    path.write_text("{}")
    path.chmod(0o644)

    await FileTokenStorage(tmp_path, "srv").set_tokens(OAuthToken(access_token="a", token_type="Bearer"))

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


async def test_the_store_directory_is_created_owner_only(tmp_path):
    store_dir = tmp_path / "not-yet-there"

    await FileTokenStorage(store_dir, "srv").set_tokens(OAuthToken(access_token="a", token_type="Bearer"))

    assert stat.S_IMODE(store_dir.stat().st_mode) == 0o700


async def test_make_auth_factory_builds_a_provider_per_server(tmp_path):
    provider = make_auth_factory(tmp_path)("srv", HttpServer(url="https://x", oauth=True))
    assert isinstance(provider, httpx.Auth)


async def test_capture_authorization_code_reads_the_redirect():
    port = 42137
    capture = asyncio.ensure_future(capture_authorization_code(port))
    # retry the connect so the test waits for the bind instead of betting on a delay
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(retries=5)) as client:
        await client.get(f"http://localhost:{port}/callback?code=THECODE&state=xyz")
    assert await capture == ("THECODE", "xyz")


async def test_the_browser_opens_off_the_event_loop(monkeypatch):
    # webbrowser.open is synchronous and can block (macOS shells out to `open`);
    # inline it would stall the loop, so the redirect handler must to_thread it
    seen = {}

    def fake_open(url):
        seen["url"] = url
        seen["thread"] = threading.get_ident()
        return True

    monkeypatch.setattr(oauth.webbrowser, "open", fake_open)

    await oauth._open_in_browser("https://auth.example/authorize")

    assert seen["url"] == "https://auth.example/authorize"
    assert seen["thread"] != threading.get_ident()  # ran in a worker thread, not on the loop


async def test_the_oauth_provider_is_cached_per_label():
    # one provider per label, so token and discovery state is shared across every
    # listing and call instead of re-created (and re-reading mcp-auth.json) each time
    calls = []

    def factory(label, server):
        calls.append(label)
        return httpx.BasicAuth(label, "secret")

    servers = {
        "a": HttpServer(url="https://a", oauth=True),
        "b": HttpServer(url="https://b", oauth=True),
    }
    registry = McpToolRegistry(servers, YoloPermissionPolicy(), auth_factory=factory)

    a1 = registry._auth_for("a", servers["a"])
    a2 = registry._auth_for("a", servers["a"])
    b1 = registry._auth_for("b", servers["b"])

    assert a1 is a2  # same label reuses the cached provider
    assert a1 is not b1  # a different label gets its own
    assert calls == ["a", "b"]  # the factory ran once per label, not per operation
