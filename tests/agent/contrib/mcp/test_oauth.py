"""OAuth token storage and the localhost redirect capture."""

import asyncio

import httpx
from mcp.shared.auth import OAuthToken

from luca.agent.contrib.mcp.config import HttpServer
from luca.agent.contrib.mcp.oauth import (
    FileTokenStorage,
    capture_authorization_code,
    make_auth_factory,
)


async def test_the_token_store_round_trips_and_isolates_servers(tmp_path):
    store = FileTokenStorage(tmp_path, "srv")
    assert await store.get_tokens() is None
    await store.set_tokens(OAuthToken(access_token="a", token_type="Bearer", refresh_token="r"))
    assert await store.get_tokens() == OAuthToken(access_token="a", token_type="Bearer", refresh_token="r")
    # a different server shares the file but not the section
    assert await FileTokenStorage(tmp_path, "other").get_tokens() is None


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
