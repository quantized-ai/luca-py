"""MCP server definitions for `luca.json` — pure pydantic, no SDK import.

Kept free of the `mcp` SDK so `LucaConfig` (and the whole TUI) parses and runs
without the optional `mcp` extra installed; the SDK is imported lazily, only
when servers are actually configured and a connection is built.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

_STRICT = ConfigDict(extra="forbid")


class StdioServer(BaseModel):
    """A local MCP server run as a subprocess over stdio."""

    type: Literal["stdio"] = "stdio"
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    # How long this server's tool listing may take. None defers to the resolved
    # default (`registry.resolve_listing_timeout_ms`). Bounds the LISTING only.
    timeout_in_ms: int | None = Field(default=None, gt=0)
    # How long one of this server's TOOL CALLS may take, stamped onto every
    # spec as `ToolSpec.timeout_in_ms` and enforced by the runner at dispatch.
    # None inherits `RuntimeConfig.tool_execution_timeout_in_ms`, so leaving it
    # unset changes nothing. Separate from `timeout_in_ms` because a listing is
    # a quick metadata fetch and a call may legitimately run for minutes.
    call_timeout_in_ms: int | None = Field(default=None, gt=0)
    model_config = _STRICT


class HttpServer(BaseModel):
    """A remote MCP server over Streamable HTTP. `oauth` turns on the browser
    OAuth flow; otherwise `headers` carries any static auth."""

    type: Literal["http"] = "http"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    oauth: bool = False
    enabled: bool = True
    # See `StdioServer.timeout_in_ms`. An `oauth` server that leaves this unset
    # gets a much longer default, because its browser flow runs inside the
    # listing and waits on a human.
    timeout_in_ms: int | None = Field(default=None, gt=0)
    call_timeout_in_ms: int | None = Field(default=None, gt=0)  # see `StdioServer`
    model_config = _STRICT


McpServerDef = Annotated[StdioServer | HttpServer, Field(discriminator="type")]
