"""OAuth for remote MCP servers, on top of the SDK's `OAuthClientProvider`.

Tokens (and the dynamically-registered client) persist to
`<store_dir>/mcp-auth.json`, keyed by server label, so the browser flow runs
only once. A one-shot localhost server captures the authorization redirect.
"""

from __future__ import annotations

import asyncio
import json
import webbrowser
import zlib
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)

from .config import HttpServer

# The localhost redirect port is derived per server so two OAuth servers do not
# collide, and stays stable across runs so the registered redirect URI matches.
_CALLBACK_PORT_BASE = 41293
_AUTH_TIMEOUT = 300.0  # seconds to wait for the browser redirect


def _port_for(label: str) -> int:
    return _CALLBACK_PORT_BASE + (zlib.crc32(label.encode()) % 1000)


class FileTokenStorage(TokenStorage):
    """One server's slice of `<store_dir>/mcp-auth.json`."""

    def __init__(self, store_dir: Path, label: str) -> None:
        self._path = Path(store_dir) / "mcp-auth.json"
        self._label = label

    def _read(self) -> dict:
        try:
            return json.loads(self._path.read_text()).get(self._label, {})
        except (OSError, json.JSONDecodeError):
            return {}

    def _write(self, section: dict) -> None:
        try:
            whole = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            whole = {}
        whole[self._label] = {**whole.get(self._label, {}), **section}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(whole, indent=2))

    async def get_tokens(self) -> OAuthToken | None:
        raw = self._read().get("tokens")
        return OAuthToken.model_validate(raw) if raw else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._write({"tokens": tokens.model_dump(mode="json")})

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = self._read().get("client_info")
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._write({"client_info": client_info.model_dump(mode="json")})


async def capture_authorization_code(
    port: int,
    *,
    timeout: float = _AUTH_TIMEOUT,
) -> tuple[str, str | None]:
    """Run a one-shot localhost server and return the `(code, state)` from the
    OAuth redirect. Times out so an abandoned authorization does not hang."""
    loop = asyncio.get_running_loop()
    result: asyncio.Future = loop.create_future()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request_line = (await reader.readline()).decode("latin-1")
        target = request_line.split(" ")[1] if " " in request_line else ""
        query = parse_qs(urlparse(target).query)
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"
            b"<html><body>Authorized. You can close this tab and return to luca."
            b"</body></html>",
        )
        await writer.drain()
        writer.close()
        if not result.done():
            result.set_result(
                (query.get("code", [""])[0], query.get("state", [None])[0]),
            )

    server = await asyncio.start_server(handle, "localhost", port)
    async with server:
        return await asyncio.wait_for(result, timeout)


def make_auth_factory(store_dir: Path):
    """A `(label, HttpServer) -> OAuthClientProvider` factory the manager uses to
    authenticate an HTTP connection."""

    def factory(label: str, server: HttpServer) -> OAuthClientProvider:
        port = _port_for(label)
        redirect_uri = f"http://localhost:{port}/callback"

        async def redirect_handler(auth_url: str) -> None:
            webbrowser.open(auth_url)

        async def callback_handler() -> tuple[str, str | None]:
            return await capture_authorization_code(port)

        return OAuthClientProvider(
            server_url=server.url,
            client_metadata=OAuthClientMetadata(
                client_name="luca",
                redirect_uris=[redirect_uri],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                token_endpoint_auth_method="none",
            ),
            storage=FileTokenStorage(store_dir, label),
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
        )

    return factory
