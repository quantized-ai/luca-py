"""OAuthProvider — one per server, for the life of the process.

The review of the previous attempt found a fresh provider built for every
listing and every tool call, and named three consequences: no in-memory token
state, so every operation re-read the file and re-ran discovery; a callback
port derived from the label, so two flows collided; and two refreshes racing,
where a server that rotates refresh tokens leaves the loser permanently
invalid.

All three are ownership problems, and all three are fixed by owning it once.
`McpService` builds one of these per server at construction and keeps it, so
discovery happens once, the live token is in memory, and the refresh is
single-flight through one future that every concurrent caller awaits.

It is an `httpx.Auth`, so the token is attached by the HTTP layer and nothing
above has to remember to. The flow itself only ever runs from an explicit
entry point — the startup worker or `/mcp login` — never from inside a turn:
the previous attempt authorized inside `get_tools`, which is why a browser
window opened in the middle of somebody's first message.
"""

from __future__ import annotations

import asyncio
import logging
import webbrowser
from collections.abc import Callable
from typing import Any, Final
from urllib.parse import urlencode

import httpx

from ..errors import McpAuthRequired
from ..servers import HttpServer
from .callback import LoopbackCallback
from .metadata import (
    AuthorizationServerMetadata,
    ProtectedResourceMetadata,
    check_issuer,
    fallback_metadata,
    metadata_urls,
    protected_resource_url,
)
from .pkce import Pkce, new_state, state_matches
from .store import ClientRegistration, IssuerRecord, OAuthToken, Registration, TokenStore

logger = logging.getLogger(__name__)

# How long the human has to finish in the browser.
AUTHORIZE_TIMEOUT_S: Final = 300.0
DISCOVERY_TIMEOUT_S: Final = 30.0


class OAuthProvider(httpx.Auth):
    def __init__(
        self,
        server: HttpServer,
        *,
        store: TokenStore,
        browser: Callable[[str], Any] | None = None,
    ) -> None:
        self.server = server
        self._store = store
        self._open = browser or _open_browser
        self._token: OAuthToken | None = None
        self._issuer: str | None = None
        self._metadata: AuthorizationServerMetadata | None = None
        self._registration: ClientRegistration | None = None
        self._loaded = False
        # The single-flight slot. Everything that notices an expired token
        # awaits the SAME future, so one refresh happens and a rotated refresh
        # token cannot be spent twice.
        self._refreshing: asyncio.Future | None = None
        self._lock = asyncio.Lock()

    def auth_flow(self, request: httpx.Request):
        """Attach whatever token is in memory.

        Synchronous by `httpx.Auth`'s contract, so it cannot refresh: it
        attaches what it has, and a 401 comes back to the caller as
        `McpAuthRequired` for the async path to deal with.
        """
        if self._token is not None:
            request.headers["Authorization"] = f"{self._token.token_type} {self._token.access_token}"
        yield request

    async def ensure_token(self, client: httpx.AsyncClient, *, interactive: bool) -> None:
        """Make sure a usable token is in memory, refreshing or logging in.

        `interactive=False` is the path a tool call takes: it will refresh
        silently but never open a browser, because a login prompt appearing
        mid-turn is the behaviour this design is fixing.
        """
        await self._load()
        if self._token is not None and not self._token.expired():
            return
        if self._token is not None and self._token.refresh_token:
            await self._refresh(client)
            if self._token is not None and not self._token.expired():
                return
        if not interactive:
            raise McpAuthRequired(
                f"MCP server {self.server.label!r} needs authorization. Run `/mcp login {self.server.label}`."
            )
        await self.authorize(client)

    async def _refresh(self, client: httpx.AsyncClient) -> None:
        """Exchange the refresh token, exactly once however many callers ask.

        The first caller installs the future and does the work; everyone else
        awaits it. Without this, two concurrent 401s each redeem the same
        rotating refresh token and whichever lands second is dead.
        """
        if self._refreshing is not None:
            await asyncio.shield(self._refreshing)
            return
        loop = asyncio.get_running_loop()
        self._refreshing = loop.create_future()
        try:
            await self._discover(client)
            payload = await self._token_request(
                client,
                {
                    "grant_type": "refresh_token",
                    "refresh_token": self._token.refresh_token,
                    "client_id": self._client_id(),
                },
            )
            token = OAuthToken.from_response(payload)
            # A server that rotates keeps the old refresh token usable only
            # until now; one that does not rotate omits it, and dropping ours
            # would force a browser login on the next expiry.
            if token.refresh_token is None and self._token is not None:
                token = token.model_copy(update={"refresh_token": self._token.refresh_token})
            self._token = token
            await self._persist()
        except Exception as exc:
            logger.warning("mcp server=%s token refresh failed", self.server.label, exc_info=exc)
            self._token = None  # fall through to an interactive login
        finally:
            future, self._refreshing = self._refreshing, None
            if not future.done():
                future.set_result(None)

    async def authorize(self, client: httpx.AsyncClient) -> None:
        """The authorization-code flow, with PKCE and a loopback redirect."""
        async with self._lock:
            await self._discover(client)
            metadata = self._metadata
            if metadata is None or not metadata.authorization_endpoint:
                raise McpAuthRequired(f"MCP server {self.server.label!r} publishes no authorization endpoint.")

            async with LoopbackCallback(port=self.server.redirect_port or 0) as callback:
                # Bound before the URI exists, so nothing sleeps waiting for it
                # and two servers can never pick the same port.
                redirect_uri = callback.redirect_uri
                await self._register(client, redirect_uri)
                pkce, state = Pkce.create(), new_state()
                url = self._authorization_url(metadata, redirect_uri, pkce, state)
                logger.info("mcp server=%s opening a browser to authorize", self.server.label)
                await self._open(url)
                query = await callback.wait(timeout_s=AUTHORIZE_TIMEOUT_S)

            if "error" in query:
                raise McpAuthRequired(
                    f"Authorization for {self.server.label!r} failed: "
                    f"{query.get('error_description') or query['error']}"
                )
            if not state_matches(state, query.get("state", "")):
                raise McpAuthRequired(f"Authorization for {self.server.label!r} came back with the wrong state.")
            # RFC 9207, before the code goes anywhere.
            check_issuer(self._issuer or "", query.get("iss"))

            payload = await self._token_request(
                client,
                {
                    "grant_type": "authorization_code",
                    "code": query["code"],
                    "redirect_uri": redirect_uri,
                    "client_id": self._client_id(),
                    "code_verifier": pkce.verifier,
                },
            )
            self._token = OAuthToken.from_response(payload)
            await self._persist()
            logger.info("mcp server=%s authorized", self.server.label)

    def _authorization_url(
        self,
        metadata: AuthorizationServerMetadata,
        redirect_uri: str,
        pkce: Pkce,
        state: str,
    ) -> str:
        query = {
            "response_type": "code",
            "client_id": self._client_id(),
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": pkce.challenge,
            "code_challenge_method": pkce.method,
        }
        if metadata.scopes_supported:
            query["scope"] = " ".join(metadata.scopes_supported)
        separator = "&" if "?" in (metadata.authorization_endpoint or "") else "?"
        return f"{metadata.authorization_endpoint}{separator}{urlencode(query)}"

    async def _token_request(self, client: httpx.AsyncClient, form: dict) -> dict:
        endpoint = self._metadata.token_endpoint if self._metadata else None
        if not endpoint:
            raise McpAuthRequired(f"MCP server {self.server.label!r} publishes no token endpoint.")
        # `auth=None`: this request must NOT carry the token it is trying to
        # obtain, and the client's default auth is this very provider.
        response = await client.post(endpoint, data=form, auth=None, timeout=DISCOVERY_TIMEOUT_S)
        if response.status_code >= 400:
            raise McpAuthRequired(
                f"The token endpoint for {self.server.label!r} answered {response.status_code}: {response.text[:200]}"
            )
        return response.json()

    async def _discover(self, client: httpx.AsyncClient) -> None:
        """Find the authorization server and its endpoints. Once per process."""
        if self._metadata is not None:
            return
        issuer = await self._find_issuer(client)
        self._issuer = issuer
        for url in metadata_urls(issuer):
            try:
                response = await client.get(url, auth=None, timeout=DISCOVERY_TIMEOUT_S)
            except httpx.HTTPError:
                continue
            if response.status_code == 200:
                try:
                    self._metadata = AuthorizationServerMetadata.model_validate(response.json())
                    return
                except ValueError:
                    continue
        # Plenty of small servers publish nothing and still implement the
        # conventional endpoints under the issuer.
        logger.debug("mcp server=%s publishes no AS metadata; assuming the conventional endpoints", self.server.label)
        self._metadata = fallback_metadata(issuer)

    async def _find_issuer(self, client: httpx.AsyncClient) -> str:
        """Ask the resource who guards it, falling back to its own origin."""
        try:
            response = await client.get(protected_resource_url(self.server.url), auth=None, timeout=DISCOVERY_TIMEOUT_S)
            if response.status_code == 200:
                resource = ProtectedResourceMetadata.model_validate(response.json())
                if resource.authorization_servers:
                    return resource.authorization_servers[0]
        except (httpx.HTTPError, ValueError):
            pass
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(self.server.url)
        return urlunsplit((parts.scheme, parts.netloc, "", "", ""))

    async def _register(self, client: httpx.AsyncClient, redirect_uri: str) -> None:
        """Get a client id: configured, remembered, or dynamically registered.

        A configured `client_id` wins and is also the field that will take a
        Client ID Metadata Document URL, which the 2026-07-28 revision prefers
        over dynamic registration. CIMD needs a publicly hosted document, which
        is a project artifact rather than something a client can synthesize, so
        DCR is what ships here.
        """
        if self.server.client_id or self._registration is not None:
            return
        endpoint = self._metadata.registration_endpoint if self._metadata else None
        if not endpoint:
            raise McpAuthRequired(
                f"MCP server {self.server.label!r} supports neither a configured client_id nor registration."
            )
        body = Registration(redirect_uris=[redirect_uri]).model_dump(mode="json")
        response = await client.post(endpoint, json=body, auth=None, timeout=DISCOVERY_TIMEOUT_S)
        if response.status_code >= 400:
            raise McpAuthRequired(
                f"Registering with {self.server.label!r} failed ({response.status_code}): {response.text[:200]}"
            )
        payload = response.json()
        self._registration = ClientRegistration(
            client_id=payload["client_id"],
            client_secret=payload.get("client_secret"),
        )
        await self._persist()

    def _client_id(self) -> str:
        if self.server.client_id:
            return self.server.client_id
        if self._registration is not None:
            return self._registration.client_id
        raise McpAuthRequired(f"MCP server {self.server.label!r} has no client id.")

    async def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        issuer = self._issuer or _origin(self.server.url)
        record = await self._store.get(issuer)
        self._token, self._registration = record.token, record.registration

    async def _persist(self) -> None:
        await self._store.put(
            self._issuer or _origin(self.server.url),
            IssuerRecord(token=self._token, registration=self._registration),
        )


async def _open_browser(url: str) -> None:
    """Off the loop.

    `webbrowser.open` is synchronous and slow — on macOS it shells out to
    `open`, on Linux it may spawn and wait on a browser — and calling it from a
    coroutine stalls everything: the TUI stops repainting and the cancellation
    token cannot be observed. This was a review finding on the previous
    attempt.
    """
    await asyncio.to_thread(webbrowser.open, url)


def _origin(url: str) -> str:
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))
