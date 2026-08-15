"""Runtime value objects for one configured MCP server.

Separate from `config.py` because the two answer different questions. Config is
the shape a human writes in `luca.json`, permissive about what is absent. A
`Server` is the resolved, validated thing the connection layer talks to, frozen
so it can be shared across tasks without a copy.

THREE HASHES, THREE JOBS. They exist because the durable tool catalog has to
decide, at boot and with no network, whether a cached slice still applies.

- `identity()` names the SERVER: the command line, or the URL. It is the
  catalog key. Deliberately excludes the label, so renaming `github` to `gh` in
  `luca.json` keeps the cached listing and the stored OAuth token.
- `definition_hash()` names the CONFIGURATION: everything that could change
  what the server answers, minus credentials. A mismatch drops the cached slice
  outright, because a changed `command` or `url` must never serve the old tool
  list.
- `credential_fingerprint()` names WHO IS ASKING: the environment values and
  header values, which is where secrets live. A `tools/list` result marked
  `cacheScope: "private"` is keyed by this as well, so one user's cached
  listing is never served to another.

Values are hashed, never stored, so a fingerprint can be written to the catalog
file without putting a token in it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_FROZEN = ConfigDict(extra="forbid", frozen=True)


def _digest(payload: object) -> str:
    """A stable hash of any JSON-shaped value. Sorted keys, compact separators,
    so the same configuration always produces the same digest across runs and
    machines."""
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
        # Header NAMES are configuration and live in `definition_hash`; their
        # values are the secret, and are all that varies between two users
        # pointed at the same server.
        return _digest(sorted((name.lower(), value) for name, value in self.headers))

    @property
    def header_map(self) -> dict[str, str]:
        return dict(self.headers)


Server = StdioServer | HttpServer


class ServerStatus(BaseModel):
    """What the service knows about one server, for `/mcp` and the startup
    notice. Not persisted; rebuilt from live state on every ask."""

    label: str
    connected: bool = False
    protocol_version: str | None = None
    tool_count: int = 0
    error: str | None = None
    # Tools the server offered that we refused to expose, with the reason. The
    # 2026-07-28 spec REQUIRES a client to exclude a tool whose `x-mcp-header`
    # annotations are invalid; recording why means `/mcp` can explain an
    # absence instead of leaving the user to notice a missing tool.
    rejected_tools: dict[str, str] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")
