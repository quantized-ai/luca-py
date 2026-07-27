"""McpTool — a tool advertised by an MCP server.

Its schema is the server's runtime `inputSchema` (a JSON-schema dict), not a
compile-time pydantic `Args`. `adapter.tool_to_luca_tool` passes `input_schema`
straight to the wire tool when present, so no pydantic model is built.
"""

from __future__ import annotations

from luca.agent.core import Tool


class McpTool(Tool):
    def __init__(self, *, name: str, description: str | None, input_schema: dict | None) -> None:
        self.name = name
        self.description = description or ""
        # the server's JSON schema, carried verbatim to the model
        self.input_schema = input_schema or {"type": "object", "properties": {}}
        # tool_kind stays the base ClassVar OTHER — the framework can't know an
        # external tool's behavior, so it never claims READ/EDIT/EXECUTE.
