"""Anthropic native tools: Text Editor, Bash, Web Search, and Web Fetch.

Only the DECLARATION differs from a standard tool on this wire: native calls
arrive as ordinary tool_use blocks (base ToolCall, wire names kept) and
their results ride ordinary tool_result blocks (plain ToolMessage) —
validated live 2026-08-06, streaming included. So each projector overrides
`project_tool_to_llm` and nothing else, and no call/message subclasses
exist. Versions are pinned to the current provider versions; every older
tool version targets a retired model family.

The caller executes: owns the file edits, the persistent bash session and
its `{"restart": true}` handling, timeouts, and honest failure results.

Web search and web fetch are the exception: Anthropic hosts the execution,
so their operations arrive as server_tool_use / *_tool_result blocks the
transport keeps verbatim (PrivateProviderBlock) beside their portable
projections (WebSearchBlock / WebFetchBlock) — never as ToolCalls, and
nothing comes back from the caller. Wire shapes validated live 2026-08-15/19
(specs/0011-websearch-native-tools/notebooks/).
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import PositiveInt, model_validator

from ...types.tools import ApproximateLocation, BaseTool, ToolProjector
from .transport import AnthropicToolProjector


class TextEditorProjector(AnthropicToolProjector):
    def project_tool_to_llm(self, tool: TextEditorTool) -> dict:
        decl: dict = {"type": "text_editor_20250728", "name": tool.name}
        if tool.max_characters is not None:
            decl["max_characters"] = tool.max_characters
        return decl


class TextEditorTool(BaseTool):
    """Declares Anthropic's text editor. Calls arrive as base ToolCalls named
    "str_replace_based_edit_tool" with the command in `arguments`
    (`view` / `create` / `str_replace` / `insert`)."""

    name: ClassVar[str] = "str_replace_based_edit_tool"
    max_characters: int | None = None

    def get_projector(self) -> ToolProjector:
        return TextEditorProjector()


class BashProjector(AnthropicToolProjector):
    def project_tool_to_llm(self, tool: BaseTool) -> dict:
        return {"type": "bash_20250124", "name": tool.name}


class BashTool(BaseTool):
    """Declares Anthropic's bash tool. Calls arrive as base ToolCalls named
    "bash" with `{"command": ...}` (or `{"restart": true}`) in `arguments`."""

    name: ClassVar[str] = "bash"

    def get_projector(self) -> ToolProjector:
        return BashProjector()


# --- Web Search / Web Fetch ------------------------------------------------

AnthropicWebCaller = Literal[
    "direct",
    "code_execution_20260120",
]

ResponseInclusion = Literal["full", "excluded"]


class BaseWebTool(BaseTool):
    """The controls Anthropic's two web server tools share. Every optional
    parameter is omitted from the wire when unset, preserving provider
    defaults."""

    max_uses: PositiveInt | None = None

    # Mutually exclusive.
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None

    allowed_callers: list[AnthropicWebCaller] | None = None
    response_inclusion: ResponseInclusion | None = None

    @model_validator(mode="after")
    def _domains_are_exclusive(self) -> BaseWebTool:
        if self.allowed_domains is not None and self.blocked_domains is not None:
            raise ValueError("allowed_domains and blocked_domains are mutually exclusive")
        return self


class BaseWebToolProjector(AnthropicToolProjector):
    """Declaration shared by the two web tools; subclasses supply the
    versioned wire type and any tool-specific options."""

    WIRE_TYPE: ClassVar[str] = ""

    def project_tool_to_llm(self, tool: BaseWebTool) -> dict:
        decl: dict = {"type": self.WIRE_TYPE, "name": tool.name}
        if tool.max_uses is not None:
            decl["max_uses"] = tool.max_uses
        if tool.allowed_domains is not None:
            decl["allowed_domains"] = list(tool.allowed_domains)
        if tool.blocked_domains is not None:
            decl["blocked_domains"] = list(tool.blocked_domains)
        if tool.allowed_callers is not None:
            decl["allowed_callers"] = list(tool.allowed_callers)
        if tool.response_inclusion is not None:
            decl["response_inclusion"] = tool.response_inclusion
        return decl


class WebSearchProjector(BaseWebToolProjector):
    WIRE_TYPE: ClassVar[str] = "web_search_20260318"

    def project_tool_to_llm(self, tool: WebSearchTool) -> dict:
        decl = super().project_tool_to_llm(tool)
        if tool.user_location is not None:
            decl["user_location"] = {"type": "approximate", **tool.user_location.model_dump(exclude_none=True)}
        return decl


class WebSearchTool(BaseWebTool):
    """Declares Anthropic's web_search server tool. Anthropic executes it:
    operations arrive as content blocks (PrivateProviderBlock plus
    WebSearchBlock), never as ToolCalls."""

    name: ClassVar[str] = "web_search"

    user_location: ApproximateLocation | None = None

    def get_projector(self) -> ToolProjector:
        return WebSearchProjector()


class WebFetchProjector(BaseWebToolProjector):
    WIRE_TYPE: ClassVar[str] = "web_fetch_20260318"

    def project_tool_to_llm(self, tool: WebFetchTool) -> dict:
        decl = super().project_tool_to_llm(tool)
        if tool.max_content_tokens is not None:
            decl["max_content_tokens"] = tool.max_content_tokens
        if tool.citations is not None:
            decl["citations"] = {"enabled": tool.citations}
        if tool.use_cache is not None:
            decl["use_cache"] = tool.use_cache
        return decl


class WebFetchTool(BaseWebTool):
    """Declares Anthropic's web_fetch server tool. Anthropic executes it:
    operations arrive as content blocks (PrivateProviderBlock plus
    WebFetchBlock), never as ToolCalls."""

    name: ClassVar[str] = "web_fetch"

    max_content_tokens: PositiveInt | None = None
    citations: bool | None = None
    use_cache: bool | None = None

    def get_projector(self) -> ToolProjector:
        return WebFetchProjector()
