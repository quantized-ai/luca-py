"""OAuthProvider — one per server, for the life of the process.

Built once by `McpService`, so discovery happens once, the live token is in
memory, and refresh is single-flight through one future every concurrent caller
awaits. A provider per operation would re-read the token file, redo discovery,
and let two refreshes race a rotating token into a dead end.

An `httpx.Auth`, so the token is attached by the HTTP layer. The interactive
flow runs only from an explicit entry point — startup or `/mcp login` — never
from inside a turn, so a browser cannot open mid-message.
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
    canonical_resource,
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
        # The token's audience (RFC 8707). A token minted without it is refused
        # by the MCP server, which looks like a login that silently did nothing.
        self._resource = canonical_resource(server.url)
        self._scopes: list[str] = []
        # The last `WWW-Authenticate` the server sent, which names the scopes it
        # actually wants and where its metadata lives.
        self._challenge: dict[str, str] = {}
        self._loaded = False
        # Single-flight: everyone who notices an expired token awaits the SAME
        # future, so a rotated refresh token cannot be spent twice.
        self._refreshing: asyncio.Future | None = None
        self._lock = asyncio.Lock()

    def auth_flow(self, request: httpx.Request):
        """Attach whatever token is in memory. Synchronous by `httpx.Auth`'s
        contract, so it cannot refresh; a 401 comes back as `McpAuthRequired`
        for the async path to deal with."""
        if self._token is not None:
            request.headers["Authorization"] = f"{self._token.token_type} {self._token.access_token}"
        yield request

    def observe_challenge(self, challenge: dict[str, str]) -> None:
        """Remember what a 401 or 403 asked for, so the next login requests it."""
        if challenge:
            self._challenge = challenge

    async def ensure_token(self, client: httpx.AsyncClient, *, interactive: bool) -> None:
        """Make sure a usable token is in memory, refreshing or logging in.

        `interactive=False` is the path a tool call takes: refresh silently,
        never open a browser.
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
        """Exchange the refresh token, exactly once however many callers ask."""
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
                    "resource": self._resource,
                },
            )
            token = OAuthToken.from_response(payload)
            # A server that does not rotate omits it, and dropping ours would
            # force a browser login on the next expiry.
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
            check_issuer(
                self._issuer or "",
                query.get("iss"),
                advertised=bool(self._metadata and self._metadata.authorization_response_iss_parameter_supported),
            )

            payload = await self._token_request(
                client,
                {
                    "grant_type": "authorization_code",
                    "code": query["code"],
                    "redirect_uri": redirect_uri,
                    "client_id": self._client_id(),
                    "code_verifier": pkce.verifier,
                    "resource": self._resource,
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
            # MUST be sent whether or not the server is known to support it.
            "resource": self._resource,
        }
        if self._scopes:
            query["scope"] = " ".join(self._scopes)
        separator = "&" if "?" in (metadata.authorization_endpoint or "") else "?"
        return f"{metadata.authorization_endpoint}{separator}{urlencode(query)}"

    async def _token_request(self, client: httpx.AsyncClient, form: dict) -> dict:
        endpoint = self._metadata.token_endpoint if self._metadata else None
        if not endpoint:
            raise McpAuthRequired(f"MCP server {self.server.label!r} publishes no token endpoint.")
        # `auth=None`: this request must not carry the token it is obtaining.
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
        # Many small servers publish nothing and still implement the
        # conventional endpoints under the issuer.
        logger.debug("mcp server=%s publishes no AS metadata; assuming the conventional endpoints", self.server.label)
        self._metadata = fallback_metadata(issuer)

    async def _find_issuer(self, client: httpx.AsyncClient) -> str:
        """Ask the RESOURCE who guards it, and what scopes it wants.

        Both come from the protected-resource document, not the authorization
        server's: `scopes_supported` there is what this MCP server needs, and
        the authorization server's own list is a different thing that happens
        to share the field name.
        """
        url = self._challenge.get("resource_metadata") or protected_resource_url(self.server.url)
        try:
            response = await client.get(url, auth=None, timeout=DISCOVERY_TIMEOUT_S)
            if response.status_code == 200:
                resource = ProtectedResourceMetadata.model_validate(response.json())
                if resource.resource:
                    self._resource = canonical_resource(resource.resource)
                self._scopes = self._select_scopes(resource.scopes_supported)
                if resource.authorization_servers:
                    return resource.authorization_servers[0]
        except (httpx.HTTPError, ValueError):
            pass
        self._scopes = self._select_scopes([])
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(self.server.url)
        return urlunsplit((parts.scheme, parts.netloc, "", "", ""))

    def _select_scopes(self, advertised: list[str]) -> list[str]:
        """The spec's priority: the challenge wins, then what the resource says
        it needs, then nothing. A scope we invent is a scope the user is asked
        to grant for no reason."""
        challenged = self._challenge.get("scope", "").split()
        if challenged:
            # Authoritative for the operation that failed, and unioned with
            # what we already asked for so a step-up does not drop a grant.
            return sorted(set(challenged) | set(self._scopes))
        return list(advertised)

    async def _register(self, client: httpx.AsyncClient, redirect_uri: str) -> None:
        """Get a client id: configured, remembered, or dynamically registered.

        A configured `client_id` wins, and is the field that will later take a
        Client ID Metadata Document URL — CIMD needs a publicly hosted document,
        which is a project artifact rather than something a client synthesizes.
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
    """Off the loop: `webbrowser.open` is synchronous and slow (macOS shells out
    to `open`), and calling it from a coroutine stalls the TUI's repaint and the
    cancellation token with it."""
    await asyncio.to_thread(webbrowser.open, url)


def _origin(url: str) -> str:
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))
