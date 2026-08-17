"""Runtime value objects for one configured MCP server.

Separate from `config.py`: that is the shape a human writes, this is the
resolved thing the connection layer talks to, frozen so it can be shared across
tasks without a copy.

THREE HASHES, because the durable catalog has to decide at boot, with no
network, whether a cached slice still applies:

- `identity()` names the SERVER (command line, or URL) and is the catalog key.
  Excludes the label, so a rename keeps the listing and the OAuth token.
- `definition_hash()` names the CONFIGURATION minus credentials. A mismatch
  drops the slice: a changed `command` must never serve the old tool list.
- `credential_fingerprint()` names WHO IS ASKING — the env and header VALUES.
  A `cacheScope: "private"` listing is keyed by this too, so one user's cached
  listing is never served to another. It covers what is CONFIGURED, which is
  every credential except an OAuth token: that one arrives from a browser, is
  not on this model, and is not known until it has been read off disk, so
  `ToolCatalog.verify_credential` checks it at the first listing instead.

Values are hashed, never stored, so a fingerprint can go in the catalog file
without putting a token in it.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_FROZEN = ConfigDict(extra="forbid", frozen=True)


def _digest(payload: object) -> str:
    """A stable hash of any JSON-shaped value, so the same configuration
    produces the same digest across runs and machines."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class StdioServer(BaseModel):
    """A local MCP server run as a subprocess over stdio."""

    type: Literal["stdio"] = "stdio"
    label: str
    command: str
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()  # pairs, not a dict, so the model can be frozen
    connect_timeout_in_ms: int | None = None
    list_timeout_in_ms: int | None = None
    call_timeout_in_ms: int | None = None
    model_config = _FROZEN

    def identity(self) -> str:
        return _digest(["stdio", self.command, list(self.args)])

    def definition_hash(self) -> str:
        return _digest(["stdio", self.command, list(self.args), sorted(name for name, _ in self.env)])

    def credential_fingerprint(self) -> str:
        return _digest(sorted(self.env))

    @property
    def environment(self) -> dict[str, str]:
        return dict(self.env)


class HttpServer(BaseModel):
    """A remote MCP server over Streamable HTTP."""

    type: Literal["http"] = "http"
    label: str
    url: str
    headers: tuple[tuple[str, str], ...] = ()
    oauth: bool = False
    client_id: str | None = None
    redirect_port: int | None = None
    connect_timeout_in_ms: int | None = None
    list_timeout_in_ms: int | None = None
    call_timeout_in_ms: int | None = None
    model_config = _FROZEN

    def identity(self) -> str:
        return _digest(["http", self.url])

    def definition_hash(self) -> str:
        return _digest(["http", self.url, sorted(name.lower() for name, _ in self.headers), self.oauth])

    def credential_fingerprint(self) -> str:
        # Header NAMES are configuration and live in `definition_hash`; the
        # values are the secret.
        return _digest(sorted((name.lower(), value) for name, value in self.headers))

    @property
    def header_map(self) -> dict[str, str]:
        return dict(self.headers)


Server = StdioServer | HttpServer


class ServerState(str, Enum):
    """Whether a server's tools are available to the model, and if not, why.

    CONNECTED asks exactly that question and not "is a process running": the
    catalog is durable and transports are spawned lazily, so a server whose
    tools came off disk and whose subprocess has never started this run is
    working, not broken.

    STALE is that answer with a caveat: the tools are still there, from a
    listing that outlived the refresh which failed. Keeping them is what the
    durable catalog is for, but calling that CONNECTED would hide a server that
    has gone away until every call fails.

    CONNECTING is a listing in flight, which without it the boot window reports
    as unreachable — a verdict on a question that has not finished being asked.

    The rest are each a different thing for the user to do: sign in, look at
    the error, or switch it back on.
    """

    CONNECTED = "connected"
    STALE = "stale"
    CONNECTING = "connecting"
    NEEDS_AUTH = "needs_auth"
    INACTIVE = "inactive"
    DISABLED = "disabled"


class ServerStatus(BaseModel):
    """What the service knows about one server, for `/mcp` and the startup
    notice. Not persisted; rebuilt from live state on every ask."""

    label: str
    state: ServerState = ServerState.INACTIVE
    oauth: bool = False  # whether "authenticate" is an action this server has
    protocol_version: str | None = None
    tool_count: int = 0
    error: str | None = None
    # Tools we refused to expose, with the reason. The spec requires excluding a
    # tool with invalid `x-mcp-header` annotations; recording why lets `/mcp`
    # explain the absence.
    rejected_tools: dict[str, str] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")
