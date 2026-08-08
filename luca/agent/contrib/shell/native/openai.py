"""OpenAI's provider-native tools: `apply_patch` and `shell`.

Two ordinary `Tool`s. Their `name` is the stable internal identity every
framework lookup keys on (`ToolSpec.name`, approvals, doom-loop, resolution);
`ShellNativeMiddleware` is what turns them into `{"type": "apply_patch"}` /
`{"type": "shell", "environment": {"type": "local"}}` on the wire and adopts
the calls that come back. Nothing here knows about the wire.

The permission VERBS are deliberately the generic tools' verbs — `apply_patch`
for the patcher, `bash` per command for the shell — so a rule or an "always
allow" answer keeps its meaning when the session switches providers and the
generic tools come back.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
from luca.client.native import NativeToolError, apply_diff

from ..tools import (
    BASH_DEFAULT_TIMEOUT_MS,
    BASH_MAX_OUTPUT_BYTES,
    BASH_MAX_OUTPUT_LINES,
    BashTool,
    FileReadTracker,
    ShellTool,
    ShellToolError,
    _file_lock,
    _kill_process_group,
)
from ._files import SourceFile, read_lossy, read_source, unified_diff, write_source

APPLY_PATCH_DESCRIPTION = """Create, update or delete ONE file.

- `create_file` writes a new file (overwriting an existing one) from a diff of `+` lines.
- `update_file` applies a V4A diff — `@@` context lines, then ` ` keep / `-` remove / `+` add lines — and can rename the file at the same time with `move_to`.
- `delete_file` removes the file; no diff.

You must read a file before overwriting it with `create_file`."""

SHELL_DESCRIPTION = """Run shell commands.

Each entry in `commands` runs in its own fresh shell, in order; a failing command does NOT stop the ones after it, so use `&&` inside one entry when a step depends on the one before. `timeout_ms` bounds each command and `max_output_length` caps each command's captured output."""


class OpenAIApplyPatchTool(ShellTool):
    """`apply_patch`: one create / update / delete operation per call.

    The V4A grammar itself is `luca.client.native.apply_diff`; this tool owns
    the filesystem, the approval steps and the read-before-overwrite guard."""

    name = "openai_apply_patch"
    title = "Apply patch"
    description = APPLY_PATCH_DESCRIPTION
    tool_kind = ToolKind.EDIT

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        type: Literal["create_file", "update_file", "delete_file"] = Field(
            description="The operation to perform",
        )
        path: str = Field(min_length=1, description="The file to operate on")
        diff: str | None = Field(
            default=None,
            description="The V4A diff; required for create_file and update_file",
        )
        move_to: str | None = Field(
            default=None,
            description="Rename the updated file to this path",
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
        resources = [ResourcePermission(permission="apply_patch", resource=str(target))]
        labels = [args["path"]]
        directories = [target.parent]
        if args.get("move_to"):
            destination = self._resolve(args["move_to"])
            resources.append(
                ResourcePermission(permission="apply_patch", resource=str(destination)),
            )
            labels.append(args["move_to"])
            directories.append(destination.parent)
        return [
            self._access_request(*directories),
            PermissionRequest(
                resources=resources,
                answer_options=[
                    AnswerOption(
                        resource_permissions=[
                            ResourcePermission(
                                permission="apply_patch",
                                resource=f"{target.parent}/*",
                            )
                        ],
                        metadata={"preview": f"Patch files under {target.parent}"},
                    ),
                ],
                metadata={"preview": f"Apply patch to {', '.join(labels)}"},
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
        async with _file_lock(target):
            return await asyncio.to_thread(self._apply, args, conversation_id, target)

    def _apply(self, args: dict, conversation_id: str, target: Path) -> ExecutionResult:
        if args["type"] == "delete_file":
            return self._delete(args["path"], target)
        if args["diff"] is None:
            raise ShellToolError(f"diff is required for {args['type']}.")
        if args["type"] == "create_file":
            return self._create(args["path"], args["diff"], conversation_id, target)
        return self._update(args, conversation_id, target)

    def _delete(self, display: str, target: Path) -> ExecutionResult:
        if not target.exists():
            raise ShellToolError(f"File not found: {target}")
        if target.is_dir():
            raise ShellToolError(f"Path is a directory, not a file: {target}")
        # Lossy on purpose: the text is only for the metadata diff, and a
        # delete must work on a `.pyc` or a screenshot — under OpenAI natives
        # this tool IS the only way to remove a file.
        before = read_lossy(target)
        try:
            target.unlink()
        except OSError as error:
            raise ShellToolError(f"Failed to delete {target}: {error}") from error
        return ExecutionResult(
            content=[TextContent(text=f"Deleted {display}")],
            metadata={"diff": unified_diff(before, "", display), "created": False},
        )

    def _create(
        self,
        display: str,
        diff: str,
        conversation_id: str,
        target: Path,
    ) -> ExecutionResult:
        existed = target.exists()
        if existed:
            if target.is_dir():
                raise ShellToolError(f"Path is a directory, not a file: {target}")
            if not self.tracker.was_read(conversation_id, target):
                raise ShellToolError(
                    f"File has not been read yet: read {target} before overwriting it"
                    " with create_file, or use update_file to patch it in place.",
                )
        source = self._readable(target) if existed else None
        content = self._transform("", diff, mode="create")
        self._commit(target, content, source)
        self.tracker.record(conversation_id, target)
        return ExecutionResult(
            content=[TextContent(text=f"Created {display}")],
            metadata={
                "diff": unified_diff(source.text if source else "", content, display),
                "created": not existed,
            },
        )

    def _update(self, args: dict, conversation_id: str, target: Path) -> ExecutionResult:
        if not target.exists():
            raise ShellToolError(f"File not found: {target}")
        if target.is_dir():
            raise ShellToolError(f"Path is a directory, not a file: {target}")
        source = self._readable(target)
        content = self._transform(source.text, args["diff"])
        display = args.get("move_to") or args["path"]
        destination = self._resolve(args["move_to"]) if args.get("move_to") else target
        self._commit(destination, content, source)
        if destination != target:
            try:
                target.unlink()
            except OSError as error:
                raise ShellToolError(f"Failed to remove {target} after the move: {error}") from error
        self.tracker.record(conversation_id, destination)
        text = f"Updated {args['path']}"
        if destination != target:
            text += f"\nMoved {args['path']} to {args['move_to']}"
        return ExecutionResult(
            content=[TextContent(text=text)],
            metadata={
                "diff": unified_diff(source.text, content, display),
                "created": False,
                "move_to": args.get("move_to"),
            },
        )

    def _readable(self, target: Path) -> SourceFile:
        try:
            return read_source(target)
        except UnicodeDecodeError as error:
            raise ShellToolError(f"File is not valid UTF-8: {target}") from error
        except OSError as error:
            raise ShellToolError(f"Failed to read {target}: {error}") from error

    def _transform(self, text: str, diff: str, mode: str = "default") -> str:
        try:
            return apply_diff(text, diff, mode=mode)
        except NativeToolError as error:
            raise ShellToolError(str(error)) from error

    def _commit(self, target: Path, content: str, source: SourceFile | None) -> None:
        try:
            write_source(target, content, source)
        except OSError as error:
            raise ShellToolError(f"Failed to write {target}: {error}") from error


class OpenAIShellTool(BashTool):
    """`shell`: a LIST of commands, each in its own process.

    The one tool in the suite whose RESULT has a native shape — per-command
    stdout, stderr and outcome cannot ride prose — so `execute()` captures it
    into `ExecutionResult.extras` at birth, the only moment the per-command
    split exists. Everything downstream stores that dict verbatim and the
    middleware puts it back on the wire."""

    name = "openai_shell"
    title = "Shell"
    description = SHELL_DESCRIPTION
    tool_kind = ToolKind.EXECUTE

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        commands: list[str] = Field(
            min_length=1,
            description="Shell commands, run in order, each in a fresh shell",
        )
        timeout_ms: int | None = Field(
            default=None,
            gt=0,
            description="Per-command timeout in milliseconds",
        )
        max_output_length: int | None = Field(
            default=None,
            gt=0,
            description="Per-command cap on captured output, in characters",
        )

    def __init__(
        self,
        workdir: str | os.PathLike[str] | None = None,
        shell: str | None = None,
        output_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__(workdir, shell, output_dir)
        # BashTool's __init__ formats its own template into `self.description`;
        # this tool's description is its own.
        self.description = SHELL_DESCRIPTION

    def build_permission_requests(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
    ) -> list[PermissionRequest]:
        commands = [command.strip() for command in args["commands"]]
        heads = []
        for command in commands:
            head = command.split()[0] if command.split() else command
            if head not in heads:
                heads.append(head)
        return [
            self._access_request(self.workdir),
            PermissionRequest(
                resources=[ResourcePermission(permission="bash", resource=command) for command in commands],
                answer_options=[
                    AnswerOption(
                        resource_permissions=[
                            ResourcePermission(permission="bash", resource=f"{head} *"),
                        ],
                        metadata={"preview": f"Run any '{head}' command"},
                    )
                    for head in heads
                ],
                metadata={"preview": f"Run: {'; '.join(commands)}"},
            ),
        ]

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
        # Not `_run`: this tool builds its own result (the native extras), and
        # a domain failure here would still have partial per-command output to
        # report, so the base class's error-to-result wrapper is bypassed.
        if not self.workdir.is_dir():
            return ExecutionResult(
                content=[TextContent(text=f"workdir does not exist: {self.workdir}")],
                is_error=True,
            )
        timeout_ms = args.get("timeout_ms") or BASH_DEFAULT_TIMEOUT_MS
        results = []
        for command in args["commands"]:
            results.append(await self._run_command(command, timeout_ms, cancellation_token))
            if cancellation_token.cancelled:
                break
        return self._assemble(*self._cap(results, args.get("max_output_length")))

    async def _run_command(
        self,
        command: str,
        timeout_ms: int,
        cancellation_token: CancellationToken,
    ) -> dict:
        try:
            process = await asyncio.create_subprocess_exec(
                self.shell,
                "-c",
                command,
                cwd=self.workdir,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as error:
            return {
                "stdout": "",
                "stderr": f"Failed to start shell: {error}",
                "outcome": {"type": "exit", "exit_code": 1},
            }
        out: list[bytes] = []
        err: list[bytes] = []
        streams = asyncio.create_task(self._pump(process, out, err))
        cancelled = asyncio.create_task(cancellation_token.wait_cancelled())
        timed_out = False
        try:
            done, _ = await asyncio.wait(
                {streams, cancelled},
                timeout=timeout_ms / 1_000,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if streams in done:
                await process.wait()
            else:
                timed_out = cancelled not in done
                _kill_process_group(process)
                await process.wait()
                await streams
        except asyncio.CancelledError:
            _kill_process_group(process)
            await process.wait()
            raise
        finally:
            for task in (cancelled, streams):
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        outcome = {"type": "timeout"} if timed_out else {"type": "exit", "exit_code": process.returncode}
        return {"stdout": _decode(out), "stderr": _decode(err), "outcome": outcome}

    async def _pump(
        self,
        process: asyncio.subprocess.Process,
        out: list[bytes],
        err: list[bytes],
    ) -> None:
        async def drain(stream, chunks: list[bytes]) -> None:
            while chunk := await stream.read(65_536):
                chunks.append(chunk)

        await asyncio.gather(drain(process.stdout, out), drain(process.stderr, err))

    def _cap(self, results: list[dict], max_output_length: int | None) -> tuple[list[dict], str | None]:
        """Clip every CAPTURED stream, not just the prose.

        This is the whole reason the method exists. `bash` caps its result and
        spills the rest to a file, but on this tool the prose never reaches the
        model — the projector builds the wire item from `extras["results"]`
        alone — so a cap applied to `content` would be inert and one shell call
        could put a megabyte of build log into every later request. The budget
        is `bash`'s own (2000 lines / 50 KiB per stream), tightened further by
        the model's `max_output_length` when it asked for one; anything dropped
        goes to a single spill file the clipped text names."""
        limit = min(max_output_length, BASH_MAX_OUTPUT_BYTES) if max_output_length else BASH_MAX_OUTPUT_BYTES
        if all(_fits(stream, limit) for result in results for stream in (result["stdout"], result["stderr"])):
            return results, None
        output_path = self._spill(results)
        note = f"...output truncated...\nFull output saved to: {output_path}\n\n"
        return [
            {
                **result,
                "stdout": _clip(result["stdout"], limit, note),
                "stderr": _clip(result["stderr"], limit, note),
            }
            for result in results
        ], output_path

    def _spill(self, results: list[dict]) -> str:
        """Every command's raw output, labelled, in one temp file — the same
        escape hatch `bash` gives, so nothing is actually lost."""
        handle, output_path = tempfile.mkstemp(prefix="shell_output_", suffix=".txt", dir=self.output_dir)
        with os.fdopen(handle, "w", encoding="utf-8", errors="replace") as stream:
            for index, result in enumerate(results, 1):
                stream.write(f"=== command {index} stdout ===\n{result['stdout']}\n")
                stream.write(f"=== command {index} stderr ===\n{result['stderr']}\n")
        return output_path

    def _assemble(self, results: list[dict], output_path: str | None) -> ExecutionResult:
        """One `ExecutionResult`: the per-command shapes in `extras` (what the
        model actually receives once the middleware puts them on the wire), and
        prose rendered from THOSE SAME capped streams for every other reader —
        the transcript, the event stream, a provider with no native shell."""
        blocks = []
        failed = False
        for command_result in results:
            outcome = command_result["outcome"]
            body = "".join(part for part in (command_result["stdout"], command_result["stderr"]) if part)
            if outcome["type"] == "timeout":
                failed = True
                body += "\n(timed out)" if body else "(timed out)"
            elif outcome["exit_code"] != 0:
                failed = True
                body += f"\n(exit {outcome['exit_code']})" if body else f"(exit {outcome['exit_code']})"
            blocks.append(body or "(no output)")
        last = results[-1]["outcome"] if results else {"type": "exit", "exit_code": 0}
        return ExecutionResult(
            content=[TextContent(text="\n".join(blocks))],
            metadata={
                "exit": last.get("exit_code"),
                "truncated": output_path is not None,
                "output_path": output_path,
            },
            is_error=failed,
            extras={"custom_type": "shell_call_output", "results": results},
        )


def _decode(chunks: list[bytes]) -> str:
    return b"".join(chunks).decode("utf-8", errors="replace")


def _fits(text: str, limit: int) -> bool:
    return len(text) <= limit and text.count("\n") < BASH_MAX_OUTPUT_LINES


def _clip(text: str, limit: int, note: str) -> str:
    """Keep the TAIL — a failing command's message is at the end — behind a
    note naming the file that holds the whole thing. The note is overhead on
    top of the budget rather than part of it, so a small `max_output_length`
    still buys that many characters of real output."""
    if _fits(text, limit):
        return text
    return note + "\n".join(text.split("\n")[-BASH_MAX_OUTPUT_LINES:])[-limit:]
