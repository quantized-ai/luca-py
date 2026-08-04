"""Provider-defined tools, executed by luca.

Anthropic defines a text editor its models are trained on. The request carries
a type string instead of a schema, the model emits an ordinary `tool_use`
block, and the CLIENT executes it — so everything luca already does around a
tool call (approval, the read-first guard, the recorded execution, replay)
applies unchanged.

Nothing here reimplements file handling. Each command translates the provider's
arguments into luca's and delegates to the tool that already does the work, so
the BOM handling, the locking, the diff rendering, the error wording and the
`FileReadTracker` contract stay in exactly one place. A second implementation
of "replace this string in that file" is how the two would drift.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from luca.agent.contrib.resource_permissions import PermissionRequest
from luca.agent.core import AgentSession, CancellationToken, ExecutionResult, ToolKind

from .tools import (
    EditTool,
    FileReadTracker,
    ReadTool,
    ShellTool,
    ShellToolError,
    WriteTool,
)

# Anthropic keys the editor's version to the model generation: Claude 4 and
# later take the 2025-07-28 tool, earlier models the 2025-01-24 one. Same name
# and same commands; the type string is what differs.
TEXT_EDITOR_TYPE_CURRENT = "text_editor_20250728"
TEXT_EDITOR_TYPE_LEGACY = "text_editor_20250124"

TEXT_EDITOR_NAME = "str_replace_based_edit_tool"


class NativeTextEditorTool(ShellTool):
    """Anthropic's `str_replace_based_edit_tool`.

    `description` and `Args` never reach the wire — the provider owns the
    schema. They are still required and still honest: `Args` validates the
    incoming call, and the permission layer reads the resolved path out of it.
    """

    name = TEXT_EDITOR_NAME
    description = (
        "Anthropic's text editor tool: view, create and edit files. The schema "
        "is defined by the provider; this description is never sent."
    )
    tool_kind = ToolKind.EDIT

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        command: Literal["view", "str_replace", "create", "insert"] = Field(
            description="Which editor operation to run",
        )
        path: str = Field(min_length=1, description="Absolute path to the file or directory")
        file_text: str | None = Field(default=None, description="Full contents, for `create`")
        insert_line: int | None = Field(default=None, description="Line to insert after, for `insert`")
        new_str: str | None = Field(default=None, description="Replacement or inserted text")
        old_str: str | None = Field(default=None, description="Text to replace, for `str_replace`")
        view_range: list[int] | None = Field(default=None, description="[start, end] lines, for `view`")

    def __init__(
        self,
        workdir: str | os.PathLike[str] | None = None,
        tracker: FileReadTracker | None = None,
        *,
        provider_type: str = TEXT_EDITOR_TYPE_CURRENT,
    ) -> None:
        super().__init__(workdir)
        # An INSTANCE attribute shadowing the ClassVar: the version is keyed to
        # the model, so two sessions in one process can want different ones.
        self.provider_type = provider_type
        self.tracker = tracker or FileReadTracker()
        self.read = ReadTool(workdir, self.tracker)
        self.edit = EditTool(workdir, self.tracker)
        self.write = WriteTool(workdir, self.tracker)

    # ── permissions ──────────────────────────────────────────────────────────

    def build_permission_requests(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
    ) -> list[PermissionRequest]:
        """The delegate's own requests, against the translated arguments.

        Written this way rather than rebuilt so a native call is gated by
        exactly the rules a `read` or an `edit` is. Approval is the only
        containment this product has; a native tool that asked for something
        subtly different would be a hole that no test names."""
        delegate = self._delegate(args["command"])
        return delegate.build_permission_requests(
            self._translate(args),
            session,
            conversation_id,
        )

    def _delegate(self, command: str) -> ShellTool:
        if command == "view":
            return self.read
        if command == "create":
            return self.write
        return self.edit  # str_replace and insert both mutate in place

    def _translate(self, args: dict) -> dict:
        """The provider's argument names in luca's. Only the fields the
        delegate's permission builder reads are needed here."""
        return {"file_path": args["path"]}

    # ── execution ────────────────────────────────────────────────────────────

    async def _run(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        command = args["command"]
        if command == "view":
            return await self._view(args, session, conversation_id, cancellation_token)
        if command == "create":
            return await self._create(args, session, conversation_id, cancellation_token)
        if command == "str_replace":
            return await self._str_replace(args, session, conversation_id, cancellation_token)
        return await self._insert(args, session, conversation_id, cancellation_token)

    async def _view(self, args, session, conversation_id, token) -> ExecutionResult:
        """`view_range` is [start, end] with -1 meaning "to the end"; luca's
        read takes an offset and a count."""
        translated: dict = {"file_path": args["path"]}
        window = args.get("view_range")
        if window:
            start, end = window[0], window[1]
            translated["offset"] = start
            if end != -1:
                translated["limit"] = max(end - start + 1, 1)
        return await self.read.execute(
            self.read.Args.model_validate(translated).model_dump(),
            session,
            conversation_id,
            cancellation_token=token,
        )

    async def _create(self, args, session, conversation_id, token) -> ExecutionResult:
        if args.get("file_text") is None:
            raise ShellToolError("`create` requires file_text.")
        return await self.write.execute(
            {"file_path": args["path"], "content": args["file_text"]},
            session,
            conversation_id,
            cancellation_token=token,
        )

    async def _str_replace(self, args, session, conversation_id, token) -> ExecutionResult:
        if args.get("old_str") is None:
            raise ShellToolError("`str_replace` requires old_str.")
        return await self.edit.execute(
            {
                "file_path": args["path"],
                "old_string": args["old_str"],
                # The provider allows omitting new_str to delete the match.
                "new_string": args.get("new_str") or "",
                "replace_all": False,
            },
            session,
            conversation_id,
            cancellation_token=token,
        )

    async def _insert(self, args, session, conversation_id, token) -> ExecutionResult:
        """Insert after a 1-based line number; 0 means the top of the file.

        There is no luca tool for this, so the splice happens here and the
        WRITE goes through `write` — which is what re-checks the read-first
        guard, takes the file lock, preserves the BOM and records the read."""
        line = args.get("insert_line")
        if line is None:
            raise ShellToolError("`insert` requires insert_line.")
        if args.get("new_str") is None:
            raise ShellToolError("`insert` requires new_str.")
        path = self._resolve(args["path"])
        if not path.is_file():
            raise ShellToolError(f"File not found: {path}")
        if not self.tracker.was_read(conversation_id, path):
            raise ShellToolError(f"File has not been read yet: view {path} before inserting into it.")
        lines = self._read_lines(path)
        if not 0 <= line <= len(lines):
            raise ShellToolError(f"insert_line {line} is out of range for {path} ({len(lines)} lines).")
        inserted = args["new_str"]
        if not inserted.endswith("\n"):
            inserted += "\n"
        lines[line:line] = [inserted]
        return await self.write.execute(
            {"file_path": args["path"], "content": "".join(lines)},
            session,
            conversation_id,
            cancellation_token=token,
        )

    def _read_lines(self, path: Path) -> list[str]:
        try:
            return path.read_text(encoding="utf-8").splitlines(keepends=True)
        except UnicodeDecodeError as error:
            raise ShellToolError(f"File is not valid UTF-8: {path}") from error
        except OSError as error:
            raise ShellToolError(f"Failed to read file: {error}") from error


# ── which native tools a session may use ─────────────────────────────────────


def _resolve_transport(provider: str) -> type | None:
    """The transport class a provider name routes through, or None when the
    provider is not registered."""
    from luca.client.providers import PROVIDERS

    entry = PROVIDERS.get(provider)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("default_transport_class")
    return getattr(entry, "default_transport_class", None)


def _editor_type(model: str) -> str:
    """Anthropic keys the editor to the model generation: Claude 4 and later
    take the 2025-07-28 tool, Claude 3 the 2025-01-24 one.

    The generation is the first number in the id, which holds across every
    shape in the catalog: `claude-3-haiku` (3), `claude-opus-4-8` (4),
    `claude-opus-5` (5), `claude-fable-5` (5). An id with no number at all is
    something released after this code was written, so it gets the current
    tool rather than the legacy one."""
    found = re.search(r"claude[^0-9]*([0-9]+)", model.lower())
    if found and int(found.group(1)) < 4:
        return TEXT_EDITOR_TYPE_LEGACY
    return TEXT_EDITOR_TYPE_CURRENT


def native_editor_type(provider: str, model: str) -> str | None:
    """The text-editor type string for this route, or None when there is none.

    Keyed on the TRANSPORT, not on the model. Bedrock and OpenRouter both serve
    Claude models, but neither speaks the Messages API, and neither accepts a
    tool declared by `type` — so a model-family check would confidently send a
    tool that gets rejected. Only the provider whose transport is Anthropic's
    own qualifies."""
    from luca.client.transports.anthropic.transport import AnthropicTransport

    transport = _resolve_transport(provider)
    if transport is None or not issubclass(transport, AnthropicTransport):
        return None
    if "claude" not in model.lower():
        return None
    return _editor_type(model)
