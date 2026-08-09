"""The test tool subset for spec 0010's battery — fixtures, NOT production
tools.

Four generics and four natives, under the SHIPPED tools' exact internal names,
so the real `ShellNativeMiddleware`'s tables address them unchanged. Every
`execute()` returns a hardcoded dummy `ExecutionResult` — except
`OpenAIShellTool`, which also writes the structured result extras: that
asymmetry (a native result captured at birth vs. plain text) is what the
projection battery turns on. The real tools' own behaviour — patching files,
running commands, the read-first guard — is covered by
`tests/agent/contrib/shell/native/`; this battery is about what reaches
`acompletion()`.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from luca.agent.contrib.tools import Tool
from luca.agent.core import AgentSession, CancellationToken, ExecutionResult, TextContent, ToolKind


class _DummyTool(Tool):
    """Every fixture tool returns `"<name> executed"` unless it overrides
    `execute` itself."""

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        return ExecutionResult(content=[TextContent(text=f"{self.name} executed")])


# ── the four generics ────────────────────────────────────────────────────────


class ReadTool(_DummyTool):
    name: ClassVar[str] = "read"
    description: ClassVar[str] = "Read a file."
    tool_kind: ClassVar[ToolKind] = ToolKind.READ

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        file_path: str


class DeleteFileTool(_DummyTool):
    name: ClassVar[str] = "delete_file"
    description: ClassVar[str] = "Delete a file."
    tool_kind: ClassVar[ToolKind] = ToolKind.DELETE

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        path: str


class ApplyPatchGenericTool(_DummyTool):
    name: ClassVar[str] = "apply_patch"
    description: ClassVar[str] = "Apply a patch."
    tool_kind: ClassVar[ToolKind] = ToolKind.EDIT

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        patch_text: str


class BashGenericTool(_DummyTool):
    name: ClassVar[str] = "bash"
    description: ClassVar[str] = "Run a shell command."
    tool_kind: ClassVar[ToolKind] = ToolKind.EXECUTE

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        command: str


# ── the four natives ─────────────────────────────────────────────────────────


class OpenAIApplyPatchTool(_DummyTool):
    name: ClassVar[str] = "openai_apply_patch"
    description: ClassVar[str] = "Apply a V4A patch (OpenAI-native)."
    tool_kind: ClassVar[ToolKind] = ToolKind.EDIT

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        type: Literal["create_file", "update_file", "delete_file"]
        path: str
        diff: str | None = None  # absent for delete_file


class OpenAIShellTool(Tool):
    """The one fixture tool with a structured native result: the per-command
    split exists only at execution time, so the native shape is captured AT
    BIRTH in `result.extras` (design principle 5)."""

    name: ClassVar[str] = "openai_shell"
    description: ClassVar[str] = "Run shell commands (OpenAI-native)."
    tool_kind: ClassVar[ToolKind] = ToolKind.EXECUTE

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        commands: list[str] = Field(min_length=1)
        timeout_ms: int | None = None

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        return ExecutionResult(
            content=[TextContent(text="shell executed")],
            extras={
                "custom_type": "shell_call_output",
                "results": [
                    {
                        "stdout": "shell executed",
                        "stderr": "",
                        "outcome": {"type": "exit", "exit_code": 0},
                    },
                ],
            },
        )


class AnthropicTextEditorTool(_DummyTool):
    name: ClassVar[str] = "anthropic_text_editor_20250728"
    description: ClassVar[str] = "Edit a file (Anthropic-native)."
    tool_kind: ClassVar[ToolKind] = ToolKind.EDIT

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        command: Literal["view", "create", "str_replace", "insert"]
        path: str
        file_text: str | None = None
        old_str: str | None = None
        new_str: str | None = None
        insert_line: int | None = None
        insert_text: str | None = None
        view_range: list[int] | None = None


class AnthropicBashTool(_DummyTool):
    name: ClassVar[str] = "anthropic_bash_20250124"
    description: ClassVar[str] = "Run a shell command (Anthropic-native)."
    tool_kind: ClassVar[ToolKind] = ToolKind.EXECUTE

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        command: str | None = None
        restart: bool | None = None


ALL_TOOL_CLASSES: tuple[type[Tool], ...] = (
    OpenAIApplyPatchTool,
    OpenAIShellTool,
    AnthropicTextEditorTool,
    AnthropicBashTool,
    ReadTool,
    DeleteFileTool,
    ApplyPatchGenericTool,
    BashGenericTool,
)
