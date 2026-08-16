"""Where MCP OAuth tokens live.

KEYED BY ISSUER, NOT BY SERVER LABEL. Two labels pointing at the same
authorization server share one login, renaming a label in `luca.json` does not
orphan a token, and the 2026-07-28 spec's rule that client credentials are
bound to the issuer that minted them falls out of the key rather than having to
be remembered.

A sibling of `auth.json` rather than a section in it. Different shape, and a
very different rewrite cadence: `auth.json` is hand-edited and read-only to
luca, while this file is rewritten on every token refresh.

Every read and write goes through `asyncio.to_thread`. Contract rule 8 is
explicit that blocking synchronous work must, and the OAuth path runs inside
the tool call: a synchronous `write_text` there stalls the whole loop,
including the TUI's repaint, and cancellation cannot interrupt it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

STORE_VERSION: Final = 1
# Refresh this long before the token actually expires, so a call does not race
# its own credential.
EXPIRY_SKEW_S: Final = 60


class OAuthToken(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    refresh_token: str | None = None
    expires_at: float | None = None  # epoch seconds, computed from expires_in on receipt
    scope: str | None = None
    model_config = ConfigDict(extra="forbid")

    def expired(self, *, now: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now if now is not None else time.time()) >= self.expires_at - EXPIRY_SKEW_S

    @classmethod
    def from_response(cls, payload: dict, *, now: float | None = None) -> OAuthToken:
        expires_in = payload.get("expires_in")
        moment = now if now is not None else time.time()
        return cls(
            access_token=payload["access_token"],
            token_type=payload.get("token_type") or "Bearer",
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
    """The token file, read and written off the event loop.

    `path=None` makes it in-memory only, which is what tests and a read-only
    home directory both get.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._records: dict[str, IssuerRecord] | None = None
        self._lock = asyncio.Lock()

    async def get(self, issuer: str) -> IssuerRecord:
        records = await self._load()
        return records.get(_key(issuer)) or IssuerRecord()

    async def put(self, issuer: str, record: IssuerRecord) -> None:
        async with self._lock:
            records = await self._load()
            records[_key(issuer)] = record
            snapshot = {key: value.model_dump(mode="json", exclude_none=True) for key, value in records.items()}
        # Outside the lock: the write is the slow part and holding a lock
        # across it would serialize every server's refresh behind one disk.
        await self._write(snapshot)

    async def _load(self) -> dict[str, IssuerRecord]:
        if self._records is not None:
            return self._records
        self._records = await asyncio.to_thread(self._load_sync)
        return self._records

    def _load_sync(self) -> dict[str, IssuerRecord]:
        if self.path is None or not self.path.exists():
            return {}
        try:
            document = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("the mcp token store at %s is unreadable; treating it as empty", self.path, exc_info=exc)
            return {}
        if not isinstance(document, dict) or document.get("version") != STORE_VERSION:
            return {}
        records: dict[str, IssuerRecord] = {}
        for issuer, raw in (document.get("issuers") or {}).items():
            try:
                records[issuer] = IssuerRecord.model_validate(raw)
            except ValueError:
                continue  # one unreadable entry must not cost the others their login
        return records

    async def _write(self, snapshot: dict) -> None:
        if self.path is None:
            return
        await asyncio.to_thread(self._write_sync, snapshot)

    def _write_sync(self, snapshot: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"version": STORE_VERSION, "issuers": snapshot}, indent=2))
            temporary.chmod(0o600)  # before it has the real name, so it is never briefly world-readable
            temporary.replace(self.path)
        except OSError as exc:
            # An unwritable store means logging in again next run, not a broken
            # session: the token in memory still works.
            logger.warning("could not write the mcp token store to %s", self.path, exc_info=exc)


def _key(issuer: str) -> str:
    return issuer.rstrip("/")


class Registration(BaseModel):
    """A client registration request body, per RFC 7591.

    `application_type: "native"` is required of MCP clients by the 2026-07-28
    revision, and it is not cosmetic: without it an OpenID provider applies web
    redirect rules and rejects the loopback URI this client has to use.
    """

    client_name: str = "luca"
    redirect_uris: list[str] = Field(default_factory=list)
    grant_types: list[str] = Field(default_factory=lambda: ["authorization_code", "refresh_token"])
    response_types: list[str] = Field(default_factory=lambda: ["code"])
    token_endpoint_auth_method: str = "none"
    application_type: str = "native"
    model_config = ConfigDict(extra="forbid")
