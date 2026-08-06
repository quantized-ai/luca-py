"""Anthropic native tools: Text Editor and Bash.

Only the DECLARATION differs from a standard tool on this wire: native calls
arrive as ordinary tool_use blocks (base ToolCall, wire names kept) and
their results ride ordinary tool_result blocks (plain ToolMessage) —
validated live 2026-08-06, streaming included. So each projector overrides
`project_tool_to_llm` and nothing else, and no call/message subclasses
exist. Versions are pinned to the current provider versions; every older
tool version targets a retired model family.

The caller executes: owns the file edits, the persistent bash session and
its `{"restart": true}` handling, timeouts, and honest failure results.
"""

from __future__ import annotations

from typing import ClassVar

from ...types.tools import BaseTool, ToolProjector
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
