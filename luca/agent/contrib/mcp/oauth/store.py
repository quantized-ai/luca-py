"""Where MCP OAuth tokens live.

KEYED BY ISSUER, NOT BY SERVER LABEL, so two labels on one authorization server
share a login, renaming a label does not orphan a token, and the spec's rule
that credentials are bound to their issuer falls out of the key.

AND AN INDEX BESIDE IT, because the issuer is discovered and the lookup is not.
A cold process knows only the MCP server's own URL, and the issuer that guards
it takes a network round trip to learn — so a token written under
`auth.example.test` was looked for under `api.example.test` and never found
again. The index maps the server's canonical URI to its issuer, turning that
lookup back into a local read.

A sibling of `auth.json` rather than a section in it: different shape, and a
very different cadence — `auth.json` is hand-edited and read-only to luca, this
is rewritten on every refresh.

Every read and write goes through `asyncio.to_thread` (rule 8).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# 2 adds the `resources` index. A version 1 file is read as empty, which costs
# one more login and no migration code.
STORE_VERSION: Final = 2
# Refresh this long before the token actually expires, so a call does not race
# its own credential.
EXPIRY_SKEW_S: Final = 60
BEARER: Final = "Bearer"


class OAuthToken(BaseModel):
    access_token: str
    token_type: str = BEARER  # as the server spelled it; `authorization()` canonicalizes
    refresh_token: str | None = None
    expires_at: float | None = None  # epoch seconds, computed from expires_in on receipt
    scope: str | None = None
    model_config = ConfigDict(extra="forbid")

    def expired(self, *, now: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now if now is not None else time.time()) >= self.expires_at - EXPIRY_SKEW_S

    def authorization(self) -> str:
        """The `Authorization` header value.

        RFC 6749 §7.1 makes `token_type` case-insensitive and Linear answers
        `"bearer"`, but its resource server accepts only `Bearer` — so echoing
        the server's own spelling back at it turns a login that worked into a
        401 on every request after it. Send the canonical scheme.
        """
        scheme = BEARER if self.token_type.lower() == BEARER.lower() else self.token_type
        return f"{scheme} {self.access_token}"

    @classmethod
    def from_response(cls, payload: dict, *, now: float | None = None) -> OAuthToken:
        expires_in = payload.get("expires_in")
        moment = now if now is not None else time.time()
        return cls(
            access_token=payload["access_token"],
            token_type=payload.get("token_type") or BEARER,
            refresh_token=payload.get("refresh_token"),
            expires_at=moment + float(expires_in) if expires_in is not None else None,
            scope=payload.get("scope"),
        )


class ClientRegistration(BaseModel):
    client_id: str
    client_secret: str | None = None
    model_config = ConfigDict(extra="forbid")


class IssuerRecord(BaseModel):
    token: OAuthToken | None = None
    registration: ClientRegistration | None = None
    model_config = ConfigDict(extra="forbid")


class TokenStore:
    """The token file, read and written off the event loop. `path=None` makes it
    in-memory only."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._records: dict[str, IssuerRecord] | None = None
        self._resources: dict[str, str] = {}  # server canonical URI -> issuer
        self._lock = asyncio.Lock()

    async def get(self, issuer: str) -> IssuerRecord:
        records = await self._load()
        return records.get(_key(issuer)) or IssuerRecord()

    async def issuer_for(self, resource: str) -> str | None:
        """Which authorization server guards this MCP server, as last recorded.
        A cold process asks this before it has discovered anything."""
        await self._load()
        return self._resources.get(_key(resource))

    async def put(self, issuer: str, record: IssuerRecord, *, resource: str | None = None) -> None:
        async with self._lock:
            records = await self._load()
            records[_key(issuer)] = record
            if resource:
                self._resources[_key(resource)] = _key(issuer)
            snapshot = {key: value.model_dump(mode="json", exclude_none=True) for key, value in records.items()}
            resources = dict(self._resources)
        # Outside the lock, or every server's refresh serializes behind one disk.
        await self._write(snapshot, resources)

    async def _load(self) -> dict[str, IssuerRecord]:
        if self._records is not None:
            return self._records
        self._records, self._resources = await asyncio.to_thread(self._load_sync)
        return self._records

    def _load_sync(self) -> tuple[dict[str, IssuerRecord], dict[str, str]]:
        if self.path is None or not self.path.exists():
            return {}, {}
        try:
            document = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("the mcp token store at %s is unreadable; treating it as empty", self.path, exc_info=exc)
            return {}, {}
        if not isinstance(document, dict) or document.get("version") != STORE_VERSION:
            return {}, {}
        records: dict[str, IssuerRecord] = {}
        for issuer, raw in (document.get("issuers") or {}).items():
            try:
                records[issuer] = IssuerRecord.model_validate(raw)
            except ValueError:
                continue  # one unreadable entry must not cost the others their login
        resources = document.get("resources")
        index = (
            {str(key): str(value) for key, value in resources.items() if isinstance(value, str)}
            if isinstance(resources, dict)
            else {}
        )
        return records, index

    async def _write(self, snapshot: dict, resources: dict[str, str]) -> None:
        if self.path is None:
            return
        await asyncio.to_thread(self._write_sync, snapshot, resources)

    def _write_sync(self, snapshot: dict, resources: dict[str, str]) -> None:
        document = {"version": STORE_VERSION, "issuers": snapshot, "resources": resources}
        # Named for THIS process: the path is shared by every luca on the
        # machine, and one fixed name lets two of them interleave into it.
        temporary = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            temporary.write_text(json.dumps(document, indent=2))
            temporary.chmod(0o600)  # before it has the real name, so it is never briefly world-readable
            temporary.replace(self.path)
        except OSError as exc:
            # Logging in again next run, not a broken session.
            logger.warning("could not write the mcp token store to %s", self.path, exc_info=exc)
            temporary.unlink(missing_ok=True)


def _key(issuer: str) -> str:
    return issuer.rstrip("/")


class Registration(BaseModel):
    """A client registration request body, per RFC 7591.

    `application_type: "native"` is required of MCP clients and is not
    cosmetic: without it an OpenID provider applies web redirect rules and
    rejects the loopback URI this client has to use.
    """

    client_name: str = "luca"
    redirect_uris: list[str] = Field(default_factory=list)
    grant_types: list[str] = Field(default_factory=lambda: ["authorization_code", "refresh_token"])
    response_types: list[str] = Field(default_factory=lambda: ["code"])
    token_endpoint_auth_method: str = "none"
    application_type: str = "native"
    model_config = ConfigDict(extra="forbid")
