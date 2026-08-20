"""The `mcp` block of `luca.json`. Pure pydantic, no `mcp-types`, no SDK.

Import-safe on purpose: `LucaConfig` names these models at module scope, so the
TUI parses and runs whether or not the optional `mcp` group is installed.

ONE FLAT MODEL, NOT A DISCRIMINATED UNION. `luca.json` is read twice and merged
field by field, so a home file defining `github` as http and a project file
redefining it as stdio would merge into one dict carrying `type`, `url` AND
`command` — which either fails unreadably or validates into the wrong member.
The fix is the shape `PermissionRule` already uses: one strict model, a
`model_validator`, and a `to_server()` builder, so the transport is inferred
from the fields present and a cross-file redefinition fails by name.

ENVIRONMENT REFERENCES. `env` and `headers` values may contain `${VAR}`,
expanded from the process environment when the server is built. Literal
substitution, not shell execution, and no `.env` is read: the variable has to be
exported already, exactly like every provider API key.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .servers import HttpServer, Server, StdioServer

_STRICT = ConfigDict(extra="forbid")

# `${VAR}` only. Bare `$VAR` is deliberately not supported: it makes a literal
# dollar in a password a silent failure, and there is no shell here to match.
_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

_STDIO_ONLY = ("command", "args", "env")
_HTTP_ONLY = ("url", "headers", "oauth", "client_id", "redirect_port")


class McpConfigError(ValueError):
    """A configured MCP server cannot be resolved into a connectable one."""


def expand(value: str, environ: Mapping[str, str], *, where: str) -> str:
    """Substitute `${VAR}` from `environ`. An unset variable raises rather than
    expanding to empty, which would surface as an unexplained 401."""
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        found = environ.get(name)
        if found is None:
            missing.append(name)
            return ""
        return found

    expanded = _REFERENCE.sub(replace, value)
    if missing:
        names = ", ".join(sorted(set(missing)))
        raise McpConfigError(
            f"{where} references undefined environment {'variables' if len(missing) > 1 else 'variable'}: {names}."
        )
    return expanded


class McpServerSettings(BaseModel):
    """One entry under `mcp.servers`. Either a command or a URL, never both."""

    # stdio
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    # http
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    oauth: bool = False
    client_id: str | None = None
    redirect_port: int | None = Field(default=None, gt=0, le=65535)
    # both
    enabled: bool = True
    connect_timeout_in_ms: int | None = Field(default=None, gt=0)
    list_timeout_in_ms: int | None = Field(default=None, gt=0)
    call_timeout_in_ms: int | None = Field(default=None, gt=0)
    model_config = _STRICT

    @model_validator(mode="after")
    def _one_transport(self) -> McpServerSettings:
        if self.command is not None and self.url is not None:
            raise ValueError(
                "sets both 'command' and 'url', so it is neither a stdio nor an http server. "
                "A luca.json that redefines a server already defined in another luca.json must "
                "give the whole definition, because the two files are merged field by field."
            )
        if self.command is None and self.url is None:
            raise ValueError("sets neither 'command' (stdio) nor 'url' (http).")
        wrong = [name for name in (_HTTP_ONLY if self.command is not None else _STDIO_ONLY) if _is_set(self, name)]
        if wrong:
            kind = "stdio" if self.command is not None else "http"
            raise ValueError(f"is a {kind} server but sets {', '.join(repr(name) for name in wrong)}.")
        return self

    def to_server(self, label: str, *, environ: Mapping[str, str] | None = None) -> Server:
        """The connectable server. Raises `McpConfigError` when an environment
        reference cannot be resolved."""
        env = os.environ if environ is None else environ
        shared = {
            "label": label,
            "connect_timeout_in_ms": self.connect_timeout_in_ms,
            "list_timeout_in_ms": self.list_timeout_in_ms,
            "call_timeout_in_ms": self.call_timeout_in_ms,
        }
        if self.command is not None:
            return StdioServer(
                command=self.command,
                args=tuple(self.args),
                env=tuple(
                    (name, expand(value, env, where=f"mcp server {label!r} env {name!r}"))
                    for name, value in sorted(self.env.items())
                ),
                **shared,
            )
        return HttpServer(
            url=self.url or "",
            headers=tuple(
                (name, expand(value, env, where=f"mcp server {label!r} header {name!r}"))
                for name, value in sorted(self.headers.items())
            ),
            oauth=self.oauth,
            client_id=self.client_id,
            redirect_port=self.redirect_port,
            **shared,
        )


class McpSettings(BaseModel):
    """The `mcp` block. A block rather than a bare `dict[label, server]` so
    `enabled` has somewhere to live that a server label cannot collide with."""

    enabled: bool | None = None
    servers: dict[str, McpServerSettings] = Field(default_factory=dict)
    model_config = _STRICT

    def build(self, *, environ: Mapping[str, str] | None = None) -> dict[str, Server]:
        """Every enabled server, resolved. Labels are validated here because the
        label is the dict KEY, which pydantic never sees as a field."""
        built: dict[str, Server] = {}
        for label, settings in self.servers.items():
            if not settings.enabled:
                continue
            if not label or not _LABEL.fullmatch(label):
                raise McpConfigError(
                    f"mcp server label {label!r} may only contain letters, digits, underscores and hyphens."
                )
            built[label] = settings.to_server(label, environ=environ)
        return built


# The label becomes part of every tool name the model sees, so it is held to
# the character set every provider accepts in a tool name.
_LABEL = re.compile(r"[A-Za-z0-9_-]+")


def _is_set(settings: McpServerSettings, name: str) -> bool:
    """Whether a field carries a real value. An empty dict or list counts as
    unset: pydantic fills those in whether or not the user wrote them."""
    value = getattr(settings, name)
    if value is None or value is False:
        return False
    return bool(value) if isinstance(value, dict | list) else True
