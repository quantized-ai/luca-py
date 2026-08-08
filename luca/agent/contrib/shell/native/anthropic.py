"""Anthropic's provider-native tools: `text_editor_20250728` and
`bash_20250124`.

Same shape as the OpenAI pair: ordinary `Tool`s under stable internal names,
turned into `{"type": "text_editor_20250728", …}` /
`{"type": "bash_20250124", …}` by `ShellNativeMiddleware`. Version numbers
are part of the identity — when Anthropic ships `text_editor_20260401` that
is a NEW tool with a new name, and the old one's stored calls keep projecting
verbatim.

The editor's permission verb varies by command, which is the whole point of
building the requests here rather than once per tool: `view` asks for `read`
(auto-allowed inside the workspace by the plugin's seeded rules, exactly like
the generic `read`), `create` asks for `write`, and `str_replace` / `insert`
ask for `edit`.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from luca.agent.contrib.resource_permissions import (
    AnswerOption,
    PermissionRequest,
    ResourcePermission,
)
from luca.agent.core import (
    AgentSession,
    CancellationToken,
    ExecutionResult,
    TextContent,
    ToolKind,
)
from luca.client.native import NativeToolError
from luca.client.native.text_editor import insert as insert_text_at, str_replace, view as render_view

from ..tools import BashTool, FileReadTracker, ShellTool, ShellToolError, _file_lock
from ._files import SourceFile, normalize, read_source, unified_diff, write_source

TEXT_EDITOR_DESCRIPTION = """View and edit files.

- `view` returns a file with numbered lines (optionally a `view_range`), or a directory listing.
- `create` writes `file_text` to `path`, replacing an existing file.
- `str_replace` swaps `old_str` for `new_str`; `old_str` must match exactly once.
- `insert` puts `insert_text` after line `insert_line` (`0` = the top of the file).

You must view a file before replacing it with `create`."""

BASH_DESCRIPTION = """Run one shell command and return its output.

Each call runs in a fresh shell, so there is no state to carry between calls and `restart` has nothing to restart."""

RESTART_OUTPUT = "Nothing to restart: every command runs in its own fresh shell."

# The verb each editor command asks approval for. `view` deliberately lands in
# the plugin's auto-allowed read tier.
EDITOR_PERMISSIONS = {
    "view": "read",
    "create": "write",
    "str_replace": "edit",
    "insert": "edit",
}


class AnthropicTextEditorTool(ShellTool):
    """`str_replace_based_edit_tool`, version `text_editor_20250728`.

    The four commands' text transforms are `luca.client.native.text_editor`'s
    pure functions; this tool owns the filesystem, the per-command approval
    step, and the read-before-overwrite guard (`view` feeds the same
    `FileReadTracker` the generic `read` does, so either satisfies it)."""

    name = "anthropic_text_editor_20250728"
    title = "Edit file"
    description = TEXT_EDITOR_DESCRIPTION
    tool_kind = ToolKind.EDIT

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        command: Literal["view", "create", "str_replace", "insert"] = Field(
            description="The operation to perform",
        )
        path: str = Field(min_length=1, description="The file or directory to operate on")
        file_text: str | None = Field(default=None, description="The full contents, for create")
        old_str: str | None = Field(default=None, description="The text to replace, for str_replace")
        new_str: str | None = Field(default=None, description="The replacement, for str_replace")
        insert_line: int | None = Field(
            default=None,
            ge=0,
            description="Insert after this line number; 0 is the top of the file",
        )
        insert_text: str | None = Field(default=None, description="The text to insert, for insert")
        view_range: list[int] | None = Field(
            default=None,
            description="[start, end] line numbers for view; -1 as end means the last line",
        )

    def __init__(
        self,
        workdir: str | os.PathLike[str] | None = None,
        tracker: FileReadTracker | None = None,
    ) -> None:
        super().__init__(workdir)
        self.tracker = tracker or FileReadTracker()

    def build_permission_requests(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
    ) -> list[PermissionRequest]:
        target = self._resolve(args["path"])
        permission = EDITOR_PERMISSIONS[args["command"]]
        scope = self._access_scope(target)
        return [
            self._access_request(scope),
            PermissionRequest(
                resources=[ResourcePermission(permission=permission, resource=str(target))],
                answer_options=[
                    AnswerOption(
                        resource_permissions=[
                            ResourcePermission(permission=permission, resource=f"{scope}/*"),
                        ],
                        metadata={"preview": f"{permission.capitalize()} files under {scope}"},
                    ),
                ],
                metadata={"preview": f"{permission.capitalize()} {target}"},
            ),
        ]

    async def _run(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        target = self._resolve(args["path"])
        if args["command"] == "view":
            return await asyncio.to_thread(self._view, args, conversation_id, target)
        async with _file_lock(target):
            return await asyncio.to_thread(self._edit, args, conversation_id, target)

    # ── view ────────────────────────────────────────────────────────────────

    def _view(self, args: dict, conversation_id: str, target: Path) -> ExecutionResult:
        if not target.exists():
            raise ShellToolError(f"Path not found: {target}")
        if target.is_dir():
            return ExecutionResult(content=[TextContent(text=self._listing(target))])
        source = self._readable(target)
        try:
            rendered = render_view(source.text, args.get("view_range"))
        except NativeToolError as error:
            raise ShellToolError(str(error)) from error
        # `view` is this mode's `read`, so it satisfies the read-first guard.
        self.tracker.record(conversation_id, target)
        return ExecutionResult(content=[TextContent(text=rendered)])

    def _listing(self, target: Path) -> str:
        try:
            entries = sorted(target.iterdir(), key=lambda path: path.name)
        except OSError as error:
            raise ShellToolError(f"Failed to list {target}: {error}") from error
        return "\n".join(
            f"{entry.name}/" if entry.is_dir() and not entry.is_symlink() else entry.name
            for entry in entries
            if not entry.name.startswith(".")
        )

    # ── the three writes ────────────────────────────────────────────────────

    def _edit(self, args: dict, conversation_id: str, target: Path) -> ExecutionResult:
        if target.is_dir():
            raise ShellToolError(f"Path is a directory, not a file: {target}")
        if args["command"] == "create":
            return self._create(args, conversation_id, target)
        if not target.exists():
            raise ShellToolError(f"File not found: {target}")
        source = self._readable(target)
        content = self._transform(args, source.text)
        self._commit(target, content, source)
        self.tracker.record(conversation_id, target)
        return ExecutionResult(
            content=[TextContent(text=f"Updated {args['path']}")],
            metadata={"diff": unified_diff(source.text, content, args["path"]), "created": False},
        )

    def _create(self, args: dict, conversation_id: str, target: Path) -> ExecutionResult:
        if args["file_text"] is None:
            raise ShellToolError("file_text is required for create.")
        existed = target.exists()
        if existed and not self.tracker.was_read(conversation_id, target):
            raise ShellToolError(
                f"File has not been read yet: view {target} before replacing it with create.",
            )
        source = self._readable(target) if existed else None
        self._commit(target, args["file_text"], source)
        self.tracker.record(conversation_id, target)
        return ExecutionResult(
            content=[TextContent(text=f"Created {args['path']}")],
            metadata={
                "diff": unified_diff(source.text if source else "", args["file_text"], args["path"]),
                "created": not existed,
            },
        )

    def _transform(self, args: dict, text: str) -> str:
        # Model-supplied text is normalized to LF like the file already is —
        # otherwise a CRLF `old_str` never matches a line it is looking at.
        # `write_source` puts the file's own convention back.
        try:
            if args["command"] == "str_replace":
                if args["old_str"] is None or args["new_str"] is None:
                    raise ShellToolError("old_str and new_str are required for str_replace.")
                return str_replace(text, normalize(args["old_str"]), normalize(args["new_str"]))
            if args["insert_line"] is None or args["insert_text"] is None:
                raise ShellToolError("insert_line and insert_text are required for insert.")
            return insert_text_at(text, args["insert_line"], normalize(args["insert_text"]))
        except NativeToolError as error:
            raise ShellToolError(str(error)) from error

    def _readable(self, target: Path) -> SourceFile:
        try:
            return read_source(target)
        except UnicodeDecodeError as error:
            raise ShellToolError(f"File is not valid UTF-8: {target}") from error
        except OSError as error:
            raise ShellToolError(f"Failed to read {target}: {error}") from error

    def _commit(self, target: Path, content: str, source: SourceFile | None) -> None:
        try:
            write_source(target, content, source)
        except OSError as error:
            raise ShellToolError(f"Failed to write {target}: {error}") from error


class AnthropicBashTool(BashTool):
    """`bash`, version `bash_20250124`.

    Identical to the generic `bash` except for the argument shape: one
    `command`, or `{"restart": true}`. The wire tool assumes a PERSISTENT
    session, which this suite deliberately does not have — every call is its
    own process — so a restart is answered with a successful result saying
    exactly that. An error result would teach the model the tool is broken,
    when what it asked for is already true."""

    name = "anthropic_bash_20250124"
    title = "Bash"
    description = BASH_DESCRIPTION
    tool_kind = ToolKind.EXECUTE

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        command: str | None = Field(default=None, description="The command to execute")
        restart: bool | None = Field(default=None, description="Restart the shell session")

        @field_validator("command")
        @classmethod
        def _command_not_blank(cls, value: str | None) -> str | None:
            # The generic `bash` rejects a blank command through the same
            # validator. Without it here, a whitespace-only command reaches
            # `build_permission_requests`, whose `command.split()[0]` raises
            # IndexError — the call would record FAILED with "list index out of
            # range" instead of the INVALID a bad argument deserves.
            if value is not None and not value.strip():
                raise ValueError("command must not be blank")
            return value

    def __init__(
        self,
        workdir: str | os.PathLike[str] | None = None,
        shell: str | None = None,
        output_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__(workdir, shell, output_dir)
        self.description = BASH_DESCRIPTION

    def build_permission_requests(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
    ) -> list[PermissionRequest]:
        if not args.get("command"):
            # A restart runs nothing; the directory step alone keeps the shape
            # every call in this suite declares.
            return [self._access_request(self.workdir)]
        return super().build_permission_requests(args, session, conversation_id)

    async def _run(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        if not args.get("command"):
            if not args.get("restart"):
                raise ShellToolError("Provide a command, or restart: true.")
            return ExecutionResult(content=[TextContent(text=RESTART_OUTPUT)])
        return await super()._run(
            {"command": args["command"], "timeout": None, "workdir": None},
            session,
            conversation_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            cancellation_token=cancellation_token,
        )
