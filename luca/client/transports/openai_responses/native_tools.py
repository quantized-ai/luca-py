"""OpenAI Responses native tools: Apply Patch and Local Shell.

Tool, call class, (for shell) message class, and projector live together —
importing this module registers the call/message types, and the package
__init__ imports it, so any `luca.client` import registers them.

The client declares the tools, surfaces their calls, and projects their
results; the CALLER executes — applies the diffs, owns the persistent shell
session, timeouts, truncation, and honest failure messages.

Wire shapes validated live on 2026-08-06 (captures in
specs/0009-provider-native-tools/captures/).
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...exceptions import BadRequestError
from ...types.content import ImageBlock, TextBlock, ToolCall
from ...types.messages import ToolMessage
from ...types.tools import BaseTool, ToolProjector
from .transport import OpenAIResponsesToolProjector

# --- Apply Patch -----------------------------------------------------------


class ApplyPatchToolCall(ToolCall):
    """An `apply_patch_call` output item. `arguments` is the storage the
    whole SDK reads; `operation` is the typed accessor over it."""

    type: Literal["apply_patch_call"] = "apply_patch_call"
    item_id: str | None = None  # the wire item "id" (apc_…); replayed
    status: str = "completed"  # wire item status ("in_progress" while streaming)

    @property
    def operation(self) -> dict:
        """`{"type": "create_file"|"update_file"|"delete_file", "path", "diff"?}`"""
        return self.arguments


class ApplyPatchProjector(OpenAIResponsesToolProjector):
    def project_tool_to_llm(self, tool: BaseTool) -> dict:
        return {"type": "apply_patch"}

    def build_tool_call(self, item: dict) -> ApplyPatchToolCall:
        # wire: {"type": "apply_patch_call", "id": "apc_…", "call_id": "call_…",
        #        "status": "completed",
        #        "operation": {"type": "create_file", "path": "…", "diff": "+…"}}
        return ApplyPatchToolCall(
            id=item["call_id"],
            name=ApplyPatchTool.name,  # synthesized: the wire item has no name
            arguments=dict(item.get("operation") or {}),
            item_id=item.get("id"),
            status=item.get("status", "completed"),
            complete=True,
        )

    def project_tool_call_to_llm(self, tool_call: ApplyPatchToolCall) -> dict:
        # Validated live: the full item — id and status included — is
        # accepted on every later turn under store:false.
        return {
            "type": "apply_patch_call",
            "id": tool_call.item_id,
            "call_id": tool_call.id,
            "status": tool_call.status,
            "operation": tool_call.arguments,
        }

    def project_tool_message_to_llm(
        self,
        message: ToolMessage,
        tool_call: ToolCall | None,
    ) -> dict:
        # A plain ToolMessage suffices: is_error IS the status; a dedicated
        # message class carrying its own status field would drift from it.
        return {
            "type": "apply_patch_call_output",
            "call_id": message.tool_call_id,
            "status": "failed" if message.is_error else "completed",
            "output": self._output_text(message),
        }

    def project_tool_choice_to_llm(self, tool: BaseTool) -> dict:
        return {"type": "apply_patch"}  # native tools force by TYPE, not name


ApplyPatchToolCall.projector_class = ApplyPatchProjector


class ApplyPatchTool(BaseTool):
    """Declares OpenAI's apply_patch tool. Calls surface as
    ApplyPatchToolCall named "apply_patch"; results are plain ToolMessages
    (`is_error` becomes the wire status)."""

    name: ClassVar[str] = "apply_patch"

    def get_projector(self) -> ToolProjector:
        return ApplyPatchProjector()


# --- Local Shell -----------------------------------------------------------


class ShellExitOutcome(BaseModel):
    type: Literal["exit"] = "exit"
    exit_code: int

    model_config = ConfigDict(extra="forbid")


class ShellTimeoutOutcome(BaseModel):
    type: Literal["timeout"] = "timeout"

    model_config = ConfigDict(extra="forbid")


class ShellCommandResult(BaseModel):
    stdout: str
    stderr: str
    outcome: Annotated[ShellExitOutcome | ShellTimeoutOutcome, Field(discriminator="type")]

    model_config = ConfigDict(extra="forbid")


class ShellToolCall(ToolCall):
    """A `shell_call` output item. `arguments` is the storage; `action` is
    the typed accessor."""

    type: Literal["shell_call"] = "shell_call"
    item_id: str | None = None
    status: str = "completed"

    @property
    def action(self) -> dict:
        """`{"commands": [...], "timeout_ms"?, "max_output_length"?}`"""
        return self.arguments


class ShellToolMessage(ToolMessage):
    """Structured shell results — per-command stdout/stderr/outcome cannot
    ride prose. `content` stays available as an optional human summary."""

    type: Literal["shell_call_output"] = "shell_call_output"
    results: list[ShellCommandResult]
    content: str | list[TextBlock | ImageBlock] = ""


class ShellProjector(OpenAIResponsesToolProjector):
    def project_tool_to_llm(self, tool: BaseTool) -> dict:
        return {"type": "shell", "environment": {"type": "local"}}

    def build_tool_call(self, item: dict) -> ShellToolCall:
        # wire: {"type": "shell_call", "id": "sh_…", "call_id": "call_…",
        #        "status": "…", "action": {"commands": [...],
        #        "timeout_ms": …, "max_output_length": …}}
        return ShellToolCall(
            id=item["call_id"],
            name=LocalShellTool.name,  # synthesized: the wire item has no name
            arguments=dict(item.get("action") or {}),
            item_id=item.get("id"),
            status=item.get("status", "completed"),
            complete=True,
        )

    def project_tool_call_to_llm(self, tool_call: ShellToolCall) -> dict:
        return {
            "type": "shell_call",
            "id": tool_call.item_id,
            "call_id": tool_call.id,
            "status": tool_call.status,
            "action": tool_call.arguments,
        }

    def project_tool_message_to_llm(
        self,
        message: ToolMessage,
        tool_call: ToolCall | None,
    ) -> dict:
        if not isinstance(message, ShellToolMessage):
            raise BadRequestError(
                f"shell call {message.tool_call_id!r} needs a ShellToolMessage "
                "(per-command stdout/stderr/exit codes); got a plain ToolMessage.",
            )
        out: dict = {
            "type": "shell_call_output",
            "call_id": message.tool_call_id,
            "output": [r.model_dump() for r in message.results],
        }
        # Validated live: the result echoes max_output_length when the CALL
        # carried it — the reason this method receives the originating call.
        if tool_call is not None and tool_call.arguments.get("max_output_length") is not None:
            out["max_output_length"] = tool_call.arguments["max_output_length"]
        return out

    def project_tool_choice_to_llm(self, tool: BaseTool) -> dict:
        return {"type": "shell"}


ShellToolCall.projector_class = ShellProjector


class LocalShellTool(BaseTool):
    """Declares OpenAI's local shell tool. Calls surface as ShellToolCall
    named "shell"; results must be ShellToolMessages."""

    name: ClassVar[str] = "shell"

    def get_projector(self) -> ToolProjector:
        return ShellProjector()
