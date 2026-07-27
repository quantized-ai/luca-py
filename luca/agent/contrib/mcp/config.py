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
    model_config = _STRICT


class HttpServer(BaseModel):
    """A remote MCP server over Streamable HTTP. `oauth` turns on the browser
    OAuth flow; otherwise `headers` carries any static auth."""

    type: Literal["http"] = "http"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    oauth: bool = False
    enabled: bool = True
    model_config = _STRICT


McpServerDef = Annotated[StdioServer | HttpServer, Field(discriminator="type")]
