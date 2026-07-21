"""The shell tools: read, glob, grep, edit, write, apply_patch, bash.

Filesystem and process tools bound to an "active working directory" fixed at
construction (relative arguments resolve against it; absolute arguments are
used directly). Every tool mixes in `ResourcePermissionToolMixin` and
declares the (permission, resource) pairs it touches — plus broader
suggested grants — through `build_permission_requests`, so the
`resource_permissions.PermissionStrategy` can gate calls per path / command.

Every call declares two approval steps: an `access_directory` request
naming the directories the call touches (`ShellAccessPlugin` seeds rules
that auto-cover it inside the permitted roots), then the tool's own verb
request (`read <path>`, `bash <command>`, …).

Domain failures (missing file, ambiguous edit, bad regex) are results, not
exceptions: each tool raises `ShellToolError` internally and the shared
`execute` wrapper returns it as an `ExecutionResult` with `is_error=True`.

`read`, `edit` and `write` share a `FileReadTracker` — the read tool records
every file it returns, and the mutating tools refuse to touch an existing
file that was never read (the LLM-facing read-first contract). One tracker
instance must be shared across the three tools for the contract to hold; the
shell plugin owns that wiring.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import difflib
import json
import mimetypes
import os
import platform
import shutil
import signal
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from luca.agent.contrib.resource_permissions import (
    AnswerOption,
    PermissionRequest,
    ResourcePermission,
    ResourcePermissionToolMixin,
)
from luca.agent.core import (
    CancellationToken,
    ExecutionResult,
    ImageBase64,
    ImageContent,
    TextContent,
    Tool,
    ToolContext,
    ToolKind,
)

from .patch import AddOp, DeleteOp, PatchError, UpdateOp, apply_update, parse_patch
from .replace import OldStringAmbiguous, OldStringNotFound
from .replace import replace as replace_text

DEFAULT_READ_LIMIT = 2_000
MAX_LINE_LENGTH = 2_000
MAX_BYTES = 50 * 1024
SAMPLE_BYTES = 4_096
SUPPORTED_IMAGE_MIMES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}

KNOWN_BINARY_EXTENSIONS = {
    # archives
    ".7z", ".bz2", ".gz", ".rar", ".tar", ".tgz", ".xz", ".zip",
    # executables, libraries, object files
    ".a", ".bin", ".dll", ".dylib", ".exe", ".lib", ".o", ".so",
    # bytecode
    ".class", ".jar", ".pyc", ".pyo", ".war",
    # office documents
    ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    # misc
    ".db", ".sqlite", ".wasm",
}

SEARCH_MAX_RESULTS = 100

BASH_DEFAULT_TIMEOUT_MS = 120_000
BASH_MAX_OUTPUT_LINES = 2_000
BASH_MAX_OUTPUT_BYTES = 50 * 1024

_BOM_BYTES = b"\xef\xbb\xbf"
_BOM_CHAR = "\ufeff"


class ShellToolError(Exception):
    """A domain failure the LLM should see as an is_error result."""


class FileReadTracker:
    """Which resolved paths have been read (or written) this session — the
    state behind the mutating tools' read-first contract."""

    def __init__(self) -> None:
        self._paths: set[str] = set()

    def record(self, path: str | os.PathLike[str]) -> None:
        self._paths.add(str(path))

    def was_read(self, path: str | os.PathLike[str]) -> bool:
        return str(path) in self._paths


_FILE_LOCKS: dict[str, asyncio.Lock] = {}


def _file_lock(path: Path) -> asyncio.Lock:
    return _FILE_LOCKS.setdefault(str(path), asyncio.Lock())


def _unified_diff(before: str, after: str, from_label: str, to_label: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=from_label,
            tofile=to_label,
        ),
    )


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


class ShellTool(ResourcePermissionToolMixin, Tool):
    """Base shell tool: working-directory resolution plus the error-to-result
    `execute` wrapper. Subclasses implement `_run`."""

    namespace = "contrib.shell"

    def __init__(self, workdir: str | os.PathLike[str] | None = None) -> None:
        self.workdir = (
            Path(os.path.normpath(workdir)) if workdir is not None else Path.cwd()
        )

    def _resolve(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.workdir / candidate
        return Path(os.path.normpath(candidate))

    def _access_scope(self, path: Path) -> Path:
        """The directory a target path needs access to: an existing
        directory is its own scope; anything else scopes to its parent."""
        return path if path.is_dir() else path.parent

    def _access_request(self, *directories: Path) -> PermissionRequest:
        """The `access_directory` approval step every call opens with. One
        request covering each distinct directory the call touches; each
        answer option grants a directory and everything under it."""
        unique: list[Path] = []
        for directory in directories:
            if directory not in unique:
                unique.append(directory)
        noun = "directory" if len(unique) == 1 else "directories"
        listing = ", ".join(str(directory) for directory in unique)
        return PermissionRequest(
            resources=[
                ResourcePermission(
                    permission="access_directory", resource=str(directory),
                )
                for directory in unique
            ],
            answer_options=[
                AnswerOption(
                    resource_permissions=[
                        ResourcePermission(
                            permission="access_directory", resource=str(directory),
                        ),
                        ResourcePermission(
                            permission="access_directory", resource=f"{directory}/*",
                        ),
                    ],
                    metadata={"preview": f"Always allow access to {directory}"},
                )
                for directory in unique
            ],
            metadata={"preview": f"Access {noun} {listing}"},
        )

    async def _run(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        raise NotImplementedError

    async def execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        try:
            return await self._run(args, context, cancellation_token=cancellation_token)
        except ShellToolError as error:
            return ExecutionResult(
                content=[TextContent(text=str(error))], is_error=True,
            )


class RipgrepTool(ShellTool):
    """Shared base for the ripgrep-backed search tools (`glob`, `grep`)."""

    def __init__(
        self,
        workdir: str | os.PathLike[str] | None = None,
        rg_path: str | None = None,
    ) -> None:
        super().__init__(workdir)
        self.rg_path = rg_path

    def _rg_binary(self) -> str:
        binary = self.rg_path or shutil.which("rg")
        if binary is None:
            raise ShellToolError("ripgrep (rg) was not found on PATH")
        return binary

    async def _run_ripgrep(
        self, argv: list[str], cwd: Path,
    ) -> tuple[str, str, int]:
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as error:
            raise ShellToolError(f"Failed to run ripgrep: {error}") from error
        try:
            stdout, stderr = await process.communicate()
        except asyncio.CancelledError:
            _kill_process_group(process)
            await process.communicate()
            raise
        return (
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
            process.returncode,
        )


# ── read ─────────────────────────────────────────────────────────────────────

READ_DESCRIPTION = r"""Read a file or directory from the local filesystem. If the path does not exist, an error is returned.

Usage:
- The file_path parameter should be an absolute path.
- By default, this tool returns up to 2000 lines from the start of the file.
- The offset parameter is the line number to start from (1-indexed).
- To read later sections, call this tool again with a larger offset.
- Use the grep tool to find specific content in large files or files with long lines.
- If you are unsure of the correct file path, use the glob tool to look up filenames by glob pattern.
- Contents are returned with each line prefixed by its line number as `<line>: <content>`. For example, if a file has contents "foo\n", you will receive "1: foo\n". For directories, entries are returned one per line (without line numbers) with a trailing `/` for subdirectories.
- Any line longer than 2000 characters is truncated.
- Call this tool in parallel when you know there are multiple files you want to read.
- Avoid tiny repeated slices (30 line chunks). If you need more context, read a larger window.
- This tool can read image files and PDFs and return them as file attachments."""


class ReadTool(ShellTool):
    name = "read"
    description = READ_DESCRIPTION
    tool_kind = ToolKind.READ

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        file_path: str = Field(
            min_length=1,
            description="The absolute path to the file or directory to read",
        )
        offset: int = Field(
            default=1,
            ge=1,
            description="The 1-based line or directory-entry offset",
        )
        limit: int = Field(
            default=DEFAULT_READ_LIMIT,
            ge=1,
            description="Maximum number of lines or directory entries to return",
        )

    def __init__(
        self,
        workdir: str | os.PathLike[str] | None = None,
        tracker: FileReadTracker | None = None,
    ) -> None:
        super().__init__(workdir)
        self.tracker = tracker or FileReadTracker()

    def build_permission_requests(
        self, args: dict, context: ToolContext,
    ) -> list[PermissionRequest]:
        path = self._resolve(args["file_path"])
        return [self._access_request(self._access_scope(path)), PermissionRequest(
            resources=[ResourcePermission(permission="read", resource=str(path))],
            answer_options=[
                AnswerOption(
                    resource_permissions=[ResourcePermission(
                        permission="read", resource=f"{path.parent}/*",
                    )],
                    metadata={"preview": f"Read files under {path.parent}"},
                ),
            ],
            metadata={"preview": f"Read {path}"},
        )]

    async def _run(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        path = self._resolve(args["file_path"])
        return await asyncio.to_thread(self._read, path, args["offset"], args["limit"])

    def _read(self, path: Path, offset: int, limit: int) -> ExecutionResult:
        if not path.exists():
            raise ShellToolError(self._not_found_message(path))
        if path.is_dir():
            return self._read_directory(path, offset, limit)
        mime = mimetypes.guess_type(path.name)[0]
        if mime in SUPPORTED_IMAGE_MIMES:
            return self._image(path, mime)
        if mime == "application/pdf":
            return self._attachment(path, mime, "PDF read successfully")
        if path.suffix.lower() in KNOWN_BINARY_EXTENSIONS:
            raise ShellToolError(f"Cannot read binary file: {path}")
        with open(path, "rb") as stream:
            sample = stream.read(SAMPLE_BYTES)
        if _looks_binary(sample):
            raise ShellToolError(f"Cannot read binary file: {path}")
        result = self._read_text(path, offset, limit)
        self.tracker.record(path)
        return result

    def _not_found_message(self, path: Path) -> str:
        message = f"File not found: {path}"
        if path.parent.is_dir():
            siblings = difflib.get_close_matches(
                path.name, os.listdir(path.parent), n=3, cutoff=0.6,
            )
            if siblings:
                listed = "\n".join(f"  {path.parent / name}" for name in siblings)
                message += f"\n\nDid you mean one of these?\n{listed}"
        return message

    def _image(self, path: Path, mime: str) -> ExecutionResult:
        """The image itself, so the model actually sees what it asked to read.
        Whether the target provider can receive it is the adapter layer's
        problem, not this tool's."""
        data = path.read_bytes()
        return ExecutionResult(
            content=[
                ImageContent(
                    source=ImageBase64(
                        data=base64.b64encode(data).decode("ascii"),
                        media_type=mime,
                    ),
                    metadata={
                        "name": path.name,
                        "path": str(path),
                        "size_bytes": len(data),
                    },
                ),
            ],
            metadata={"attachment": {"path": str(path), "mime_type": mime}},
        )

    def _attachment(self, path: Path, mime: str, text: str) -> ExecutionResult:
        return ExecutionResult(
            content=[TextContent(text=text)],
            metadata={"attachment": {"path": str(path), "mime_type": mime}},
        )

    def _read_directory(self, path: Path, offset: int, limit: int) -> ExecutionResult:
        names = sorted(os.listdir(path))
        entries = [
            name + "/" if (path / name).is_dir() else name for name in names
        ]
        total = len(entries)
        if offset > total and not (offset == 1 and total == 0):
            raise ShellToolError(
                f"Offset {offset} is out of range for this directory ({total} entries)",
            )
        page = entries[offset - 1 : offset - 1 + limit]
        last = offset - 1 + len(page)
        truncated = last < total
        body = "\n".join(page)
        if truncated:
            body += (
                f"\n\n(Showing entries {offset}-{last} of {total}."
                f" Use offset={last + 1} to continue.)"
            )
        text = f"<path>{path}</path>\n<type>directory</type>\n<entries>\n{body}\n</entries>"
        return ExecutionResult(
            content=[TextContent(text=text)], metadata={"truncated": truncated},
        )

    def _read_text(self, path: Path, offset: int, limit: int) -> ExecutionResult:
        rendered: list[str] = []
        used = 0
        last = offset - 1
        truncated = False
        with open(path, encoding="utf-8", errors="replace") as stream:
            iterator = enumerate(stream, start=1)
            seen = 0
            pending: tuple[int, str] | None = None
            for lineno, line in iterator:
                seen = lineno
                if lineno == offset:
                    pending = (lineno, line)
                    break
            else:
                if offset == 1 and seen == 0:
                    return self._file_result(
                        path, [], "(End of file - total 0 lines)", truncated=False,
                    )
                raise ShellToolError(
                    f"Offset {offset} is out of range for this file ({seen} lines)",
                )
            while True:
                if pending is None:
                    note = f"(End of file - total {last} lines)"
                    break
                lineno, raw = pending
                text_line = raw.rstrip("\n")
                if len(text_line) > MAX_LINE_LENGTH:
                    text_line = (
                        text_line[:MAX_LINE_LENGTH]
                        + f"... (line truncated to {MAX_LINE_LENGTH} chars)"
                    )
                encoded = f"{lineno}: {text_line}"
                size = len(encoded.encode("utf-8")) + 1
                if rendered and used + size > MAX_BYTES:
                    truncated = True
                    note = (
                        f"(Output capped at 50 KB. Showing lines {offset}-{last}."
                        f" Use offset={last + 1} to continue.)"
                    )
                    break
                rendered.append(encoded)
                used += size
                last = lineno
                if lineno - offset + 1 == limit:
                    following = next(iterator, None)
                    if following is None:
                        note = f"(End of file - total {last} lines)"
                    else:
                        truncated = True
                        total = last + 1 + sum(1 for _ in iterator)
                        note = (
                            f"(Showing lines {offset}-{last} of {total}."
                            f" Use offset={last + 1} to continue.)"
                        )
                    break
                pending = next(iterator, None)
        return self._file_result(path, rendered, note, truncated=truncated)

    def _file_result(
        self, path: Path, rendered: list[str], note: str, *, truncated: bool,
    ) -> ExecutionResult:
        body = "\n".join(rendered) + "\n\n" if rendered else ""
        text = f"<path>{path}</path>\n<type>file</type>\n<content>\n{body}{note}\n</content>"
        return ExecutionResult(
            content=[TextContent(text=text)], metadata={"truncated": truncated},
        )


def _looks_binary(sample: bytes) -> bool:
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    allowed = {0x08, 0x09, 0x0A, 0x0C, 0x0D, 0x1B}
    control = sum(1 for byte in sample if byte < 0x20 and byte not in allowed)
    return control / len(sample) > 0.30


# ── glob ─────────────────────────────────────────────────────────────────────

GLOB_DESCRIPTION = """- Fast file pattern matching tool that works with any codebase size
- Supports glob patterns like "**/*.js" or "src/**/*.ts"
- Returns matching file paths
- Use this tool when you need to find files by name patterns
- When you are doing an open-ended search that may require multiple rounds of globbing and grepping, use the Task tool instead
- You have the capability to call multiple tools in a single response. It is always better to speculatively perform multiple searches as a batch that are potentially useful."""


class GlobTool(RipgrepTool):
    name = "glob"
    description = GLOB_DESCRIPTION
    tool_kind = ToolKind.SEARCH

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        pattern: str = Field(
            min_length=1,
            description="The glob pattern to match files against",
        )
        path: str | None = Field(
            default=None,
            description="Directory to search; defaults to the active working directory",
        )

    def build_permission_requests(
        self, args: dict, context: ToolContext,
    ) -> list[PermissionRequest]:
        root = self._resolve(args["path"]) if args.get("path") else self.workdir
        return [self._access_request(root), PermissionRequest(
            resources=[ResourcePermission(permission="glob", resource=str(root))],
            answer_options=[
                AnswerOption(
                    resource_permissions=[ResourcePermission(
                        permission="glob", resource=f"{root}/*",
                    )],
                    metadata={"preview": f"Search files under {root}"},
                ),
            ],
            metadata={"preview": f'Find files matching "{args["pattern"]}" in {root}'},
        )]

    async def _run(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        root = self._resolve(args["path"]) if args.get("path") else self.workdir
        if not root.exists():
            raise ShellToolError(f"glob path does not exist: {root}")
        if not root.is_dir():
            raise ShellToolError(f"glob path must be a directory: {root}")
        argv = [
            self._rg_binary(),
            "--files",
            "--hidden",
            "--glob", "!**/.git/**",
            "--glob", args["pattern"],
        ]
        stdout, stderr, code = await self._run_ripgrep(argv, root)
        if code not in (0, 1):
            raise ShellToolError(
                f"glob failed: {stderr.strip() or f'ripgrep exited with code {code}'}",
            )
        names = [line for line in stdout.splitlines() if line]
        if not names:
            return ExecutionResult(
                content=[TextContent(text="No files found")],
                metadata={"truncated": False, "count": 0},
            )
        matches = [str(root / name) for name in names[:SEARCH_MAX_RESULTS]]
        truncated = len(names) >= SEARCH_MAX_RESULTS
        text = "\n".join(matches)
        if truncated:
            text += (
                f"\n\n(Results are truncated: showing first {SEARCH_MAX_RESULTS}"
                " results. Consider using a more specific path or pattern.)"
            )
        return ExecutionResult(
            content=[TextContent(text=text)],
            metadata={"truncated": truncated, "count": len(matches)},
        )


# ── grep ─────────────────────────────────────────────────────────────────────

GREP_DESCRIPTION = r"""- Fast content search tool that works with any codebase size
- Searches file contents using regular expressions
- Supports full regex syntax (eg. "log.*Error", "function\s+\w+", etc.)
- Filter files by pattern with the include parameter (eg. "*.js", "*.{ts,tsx}")
- Returns file paths and line numbers with matching lines
- Use this tool when you need to find files containing specific patterns
- If you need to identify/count the number of matches within files, use the Bash tool with `rg` (ripgrep) directly. Do NOT use `grep`.
- When you are doing an open-ended search that may require multiple rounds of globbing and grepping, use the Task tool instead"""


class GrepTool(RipgrepTool):
    name = "grep"
    description = GREP_DESCRIPTION
    tool_kind = ToolKind.SEARCH

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        pattern: str = Field(
            min_length=1,
            description="The regex pattern to search for in file contents",
        )
        path: str | None = Field(
            default=None,
            description="File or directory to search; defaults to the active working directory",
        )
        include: str | None = Field(
            default=None,
            description='File pattern to include, such as "*.js" or "*.{ts,tsx}"',
        )

    def build_permission_requests(
        self, args: dict, context: ToolContext,
    ) -> list[PermissionRequest]:
        target = self._resolve(args["path"]) if args.get("path") else self.workdir
        scope = self._access_scope(target)
        return [self._access_request(scope), PermissionRequest(
            resources=[ResourcePermission(permission="grep", resource=str(target))],
            answer_options=[
                AnswerOption(
                    resource_permissions=[ResourcePermission(
                        permission="grep", resource=f"{scope}/*",
                    )],
                    metadata={"preview": f"Search files under {scope}"},
                ),
            ],
            metadata={"preview": f'Search for "{args["pattern"]}" in {target}'},
        )]

    async def _run(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        target = self._resolve(args["path"]) if args.get("path") else self.workdir
        if not target.exists():
            raise ShellToolError(f"grep path does not exist: {target}")
        argv = [
            self._rg_binary(),
            "--json",
            "--hidden",
            "--glob", "!**/.git/**",
        ]
        if args.get("include"):
            argv += ["--glob", args["include"]]
        argv += ["--regexp", args["pattern"], "--", str(target)]
        stdout, stderr, code = await self._run_ripgrep(argv, self.workdir)
        if code not in (0, 1):
            raise ShellToolError(stderr.strip() or f"grep failed with code {code}")
        matches, more = self._parse_matches(stdout)
        if not matches:
            return ExecutionResult(
                content=[TextContent(text="No files found")],
                metadata={"truncated": False, "count": 0},
            )
        count = len(matches)
        header = f"Found {count} match{'es' if count != 1 else ''}"
        if more:
            header += " (more matches available)"
        grouped: dict[str, list[str]] = {}
        for file_path, line_number, preview in matches:
            grouped.setdefault(file_path, []).append(f"  Line {line_number}: {preview}")
        blocks = [
            f"{file_path}:\n" + "\n".join(lines)
            for file_path, lines in grouped.items()
        ]
        text = header + "\n" + "\n\n".join(blocks)
        if more:
            text += "\n\n(Results truncated. Consider using a more specific path or pattern.)"
        return ExecutionResult(
            content=[TextContent(text=text)],
            metadata={"truncated": more, "count": count},
        )

    def _parse_matches(self, stdout: str) -> tuple[list[tuple[str, int, str]], bool]:
        matches: list[tuple[str, int, str]] = []
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "match":
                continue
            if len(matches) >= SEARCH_MAX_RESULTS:
                return matches, True
            data = event["data"]
            text = data["lines"].get("text")
            if text is None:
                continue
            preview = text.rstrip("\r\n")
            if len(preview) > MAX_LINE_LENGTH:
                preview = preview[:MAX_LINE_LENGTH] + "..."
            matches.append((data["path"]["text"], data["line_number"], preview))
        return matches, False


# ── edit ─────────────────────────────────────────────────────────────────────

EDIT_DESCRIPTION = """Performs exact string replacements in files.

Usage:
- You must use your `read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: line number + colon + space (e.g., `1: `). Everything after that space is the actual file content to match. Never include any part of the line number prefix in old_string or new_string.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- The edit will FAIL if old_string is not found in the file with an error "old_string not found in content".
- The edit will FAIL if old_string is found multiple times in the file with an error "Found multiple matches for old_string. Provide more surrounding lines in old_string to identify the correct match." Either provide a larger string with more surrounding context to make it unique or use replace_all to change every instance of old_string.
- Use replace_all for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance."""


class EditTool(ShellTool):
    name = "edit"
    description = EDIT_DESCRIPTION
    tool_kind = ToolKind.EDIT

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        file_path: str = Field(
            min_length=1,
            description="The absolute path to the file to modify",
        )
        old_string: str = Field(description="The text to replace")
        new_string: str = Field(
            description="The text to replace it with; must differ from old_string",
        )
        replace_all: bool = Field(
            default=False,
            description="Replace all occurrences of old_string",
        )

    def __init__(
        self,
        workdir: str | os.PathLike[str] | None = None,
        tracker: FileReadTracker | None = None,
    ) -> None:
        super().__init__(workdir)
        self.tracker = tracker or FileReadTracker()

    def build_permission_requests(
        self, args: dict, context: ToolContext,
    ) -> list[PermissionRequest]:
        path = self._resolve(args["file_path"])
        return [self._access_request(path.parent), PermissionRequest(
            resources=[ResourcePermission(permission="edit", resource=str(path))],
            answer_options=[
                AnswerOption(
                    resource_permissions=[ResourcePermission(
                        permission="edit", resource=f"{path.parent}/*",
                    )],
                    metadata={"preview": f"Edit files under {path.parent}"},
                ),
            ],
            metadata={"preview": f"Edit {path}"},
        )]

    async def _run(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        path = self._resolve(args["file_path"])
        old_string, new_string = args["old_string"], args["new_string"]
        if old_string == new_string:
            raise ShellToolError(
                "No changes to apply: old_string and new_string are identical.",
            )
        if old_string == "":
            if path.exists():
                raise ShellToolError(
                    "old_string cannot be empty when editing an existing file."
                    " Provide the exact text to replace, or use write for an"
                    " intentional full-file replacement.",
                )
            async with _file_lock(path):
                return await asyncio.to_thread(self._create, path, new_string)
        if not path.exists():
            raise ShellToolError(f"File not found: {path}")
        if path.is_dir():
            raise ShellToolError(f"Path is a directory, not a file: {path}")
        if not self.tracker.was_read(path):
            raise ShellToolError(
                f"File has not been read yet: read {path} before editing it.",
            )
        async with _file_lock(path):
            return await asyncio.to_thread(
                self._edit, path, old_string, new_string, args["replace_all"],
            )

    def _create(self, path: Path, content: str) -> ExecutionResult:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode("utf-8"))
        except OSError as error:
            raise ShellToolError(f"Failed to create file: {error}") from error
        self.tracker.record(path)
        diff = _unified_diff("", content, str(path), str(path))
        return ExecutionResult(
            content=[TextContent(text=f"Created file: {path}")],
            metadata={"diff": diff, "created": True},
        )

    def _edit(
        self, path: Path, old_string: str, new_string: str, replace_all: bool,
    ) -> ExecutionResult:
        raw = path.read_bytes()
        bom = raw.startswith(_BOM_BYTES)
        if bom:
            raw = raw[len(_BOM_BYTES):]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ShellToolError(f"File is not valid UTF-8: {path}") from error
        crlf = "\r\n" in text
        working = text.replace("\r\n", "\n")
        try:
            updated, count = replace_text(
                working,
                old_string.replace("\r\n", "\n"),
                new_string.replace("\r\n", "\n"),
                replace_all=replace_all,
            )
        except OldStringNotFound:
            raise ShellToolError(
                "Could not find old_string in the file. It must match exactly,"
                " including whitespace, indentation, and line endings.",
            ) from None
        except OldStringAmbiguous:
            raise ShellToolError(
                "Found multiple matches for old_string. Provide more surrounding"
                " context to make the match unique.",
            ) from None
        diff = _unified_diff(working, updated, str(path), str(path))
        final = updated.replace("\n", "\r\n") if crlf else updated
        data = final.encode("utf-8")
        if bom:
            data = _BOM_BYTES + data
        try:
            path.write_bytes(data)
        except OSError as error:
            raise ShellToolError(f"Failed to write file: {error}") from error
        self.tracker.record(path)
        return ExecutionResult(
            content=[TextContent(text=f"Edited file: {path}")],
            metadata={"diff": diff, "created": False, "replacements": count},
        )


# ── write ────────────────────────────────────────────────────────────────────

WRITE_DESCRIPTION = """Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked."""


class WriteTool(ShellTool):
    name = "write"
    description = WRITE_DESCRIPTION
    tool_kind = ToolKind.EDIT

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        content: str = Field(description="The complete content to write")
        file_path: str = Field(
            min_length=1,
            description="The absolute path to the file to write",
        )

    def __init__(
        self,
        workdir: str | os.PathLike[str] | None = None,
        tracker: FileReadTracker | None = None,
    ) -> None:
        super().__init__(workdir)
        self.tracker = tracker or FileReadTracker()

    def build_permission_requests(
        self, args: dict, context: ToolContext,
    ) -> list[PermissionRequest]:
        path = self._resolve(args["file_path"])
        return [self._access_request(path.parent), PermissionRequest(
            resources=[ResourcePermission(permission="write", resource=str(path))],
            answer_options=[
                AnswerOption(
                    resource_permissions=[ResourcePermission(
                        permission="write", resource=f"{path.parent}/*",
                    )],
                    metadata={"preview": f"Write files under {path.parent}"},
                ),
            ],
            metadata={"preview": f"Write {path}"},
        )]

    async def _run(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        path = self._resolve(args["file_path"])
        existed = path.exists()
        if existed:
            if path.is_dir():
                raise ShellToolError(f"Path is a directory, not a file: {path}")
            if not self.tracker.was_read(path):
                raise ShellToolError(
                    f"File has not been read yet: read {path} before overwriting it.",
                )
        async with _file_lock(path):
            await asyncio.to_thread(self._write, path, args["content"], existed)
        self.tracker.record(path)
        verb = "updated" if existed else "created"
        return ExecutionResult(
            content=[TextContent(text=f"File {verb} successfully at: {path}")],
            metadata={"existed": existed},
        )

    def _write(self, path: Path, content: str, existed: bool) -> None:
        try:
            bom = False
            if existed:
                with open(path, "rb") as stream:
                    bom = stream.read(len(_BOM_BYTES)) == _BOM_BYTES
            if content.startswith(_BOM_CHAR):
                bom = True
                content = content[len(_BOM_CHAR):]
            data = (_BOM_BYTES if bom else b"") + content.encode("utf-8")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        except OSError as error:
            raise ShellToolError(f"Failed to write file: {error}") from error


# ── apply_patch ──────────────────────────────────────────────────────────────

APPLY_PATCH_DESCRIPTION = """Use the `apply_patch` tool to edit files. Your patch language is a stripped-down, file-oriented diff format designed to be easy to parse and safe to apply. You can think of it as a high-level envelope:

*** Begin Patch
[ one or more file sections ]
*** End Patch

Within that envelope, you get a sequence of file operations.
You MUST include a header to specify the action you are taking.
Each operation starts with one of three headers:

*** Add File: <path> - create a new file. Every following line is a + line (the initial contents).
*** Delete File: <path> - remove an existing file. Nothing follows.
*** Update File: <path> - patch an existing file in place (optionally with a rename).

Example patch:

*** Begin Patch
*** Add File: hello.txt
+Hello world
*** Update File: src/app.py
*** Move to: src/main.py
@@ def greet():
-print("Hi")
+print("Hello, world!")
*** Delete File: obsolete.txt
*** End Patch

It is important to remember:

- You must include a header with your intended action (Add/Delete/Update)
- You must prefix new lines with `+` even when creating a new file"""


class ApplyPatchTool(ShellTool):
    name = "apply_patch"
    description = APPLY_PATCH_DESCRIPTION
    tool_kind = ToolKind.EDIT

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        patch_text: str = Field(
            min_length=1,
            description="The full patch text describing all changes",
        )

    def build_permission_requests(
        self, args: dict, context: ToolContext,
    ) -> list[PermissionRequest]:
        try:
            ops = parse_patch(args["patch_text"])
        except PatchError:
            return [PermissionRequest(
                resources=[],
                metadata={"preview": "Apply patch (invalid patch text)"},
            )]
        resources: list[ResourcePermission] = []
        labels: list[str] = []
        directories: list[Path] = []
        for op in ops:
            resolved = self._resolve(op.path)
            resources.append(ResourcePermission(
                permission="apply_patch", resource=str(resolved),
            ))
            labels.append(op.path)
            directories.append(resolved.parent)
            if isinstance(op, UpdateOp) and op.move_to:
                destination = self._resolve(op.move_to)
                resources.append(ResourcePermission(
                    permission="apply_patch", resource=str(destination),
                ))
                labels.append(op.move_to)
                directories.append(destination.parent)
        return [self._access_request(*directories), PermissionRequest(
            resources=resources,
            metadata={"preview": f"Apply patch to {', '.join(labels)}"},
        )]

    async def _run(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        patch_text = args["patch_text"]
        if not patch_text.strip():
            raise ShellToolError("patch_text is required")
        try:
            ops = parse_patch(patch_text)
        except PatchError as error:
            raise ShellToolError(str(error)) from error
        return await asyncio.to_thread(self._apply, ops)

    def _apply(self, ops: list) -> ExecutionResult:
        planned = [self._verify(op) for op in ops]
        summary: list[str] = []
        files: list[dict] = []
        for plan in planned:
            self._commit(plan)
            summary.append(f"{plan['letter']} {plan['display']}")
            diff = _unified_diff(
                plan["before"], plan["after"], plan["path"], plan["display"],
            )
            files.append({
                "path": plan["path"],
                "type": plan["type"],
                "patch": diff,
                "additions": sum(
                    1 for line in diff.splitlines()
                    if line.startswith("+") and not line.startswith("+++")
                ),
                "deletions": sum(
                    1 for line in diff.splitlines()
                    if line.startswith("-") and not line.startswith("---")
                ),
                "move_to": plan["move_to"],
            })
        text = "Success. Updated the following files:\n" + "\n".join(summary)
        return ExecutionResult(
            content=[TextContent(text=text)], metadata={"files": files},
        )

    def _verify(self, op) -> dict:
        source = self._resolve(op.path)
        plan = {
            "path": op.path,
            "display": op.path,
            "source": source,
            "target": source,
            "move_to": None,
            "bom": False,
            "before": "",
            "after": "",
        }
        if isinstance(op, AddOp):
            content = "\n".join(op.lines) + ("\n" if op.lines else "")
            plan.update(type="add", letter="A", after=content)
            return plan
        kind = "Delete" if isinstance(op, DeleteOp) else "Update"
        if not source.exists():
            raise ShellToolError(f"{kind} target not found: {source}")
        if source.is_dir():
            raise ShellToolError(
                f"{kind} target is a directory, not a file: {source}",
            )
        if isinstance(op, DeleteOp):
            before = source.read_text(encoding="utf-8", errors="replace")
            plan.update(type="delete", letter="D", before=before)
            return plan
        raw = source.read_bytes()
        bom = raw.startswith(_BOM_BYTES)
        if bom:
            raw = raw[len(_BOM_BYTES):]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ShellToolError(
                f"Update target is not valid UTF-8: {source}",
            ) from error
        original_lines = text.split("\n")
        if original_lines and original_lines[-1] == "":
            original_lines.pop()
        try:
            new_lines = apply_update(original_lines, op)
        except PatchError as error:
            raise ShellToolError(str(error)) from error
        content = ("\n".join(new_lines) + "\n") if new_lines else ""
        plan.update(
            type="update",
            letter="M",
            before=text,
            after=content,
            bom=bom,
            move_to=op.move_to,
            target=self._resolve(op.move_to) if op.move_to else source,
            display=op.move_to or op.path,
        )
        return plan

    def _commit(self, plan: dict) -> None:
        try:
            if plan["type"] == "add":
                plan["target"].parent.mkdir(parents=True, exist_ok=True)
                plan["target"].write_bytes(plan["after"].encode("utf-8"))
            elif plan["type"] == "delete":
                plan["source"].unlink()
            else:
                data = plan["after"].encode("utf-8")
                if plan["bom"]:
                    data = _BOM_BYTES + data
                plan["target"].parent.mkdir(parents=True, exist_ok=True)
                plan["target"].write_bytes(data)
                if plan["target"] != plan["source"]:
                    plan["source"].unlink()
        except OSError as error:
            raise ShellToolError(
                f"Failed to apply patch to {plan['display']}: {error}",
            ) from error


# ── bash ─────────────────────────────────────────────────────────────────────

BASH_DESCRIPTION_TEMPLATE = """Executes a given bash command in a persistent shell session with optional timeout, ensuring proper handling and security measures.

Be aware: OS: {os}, Shell: {shell}

All commands run in the current working directory by default. Use the `workdir` parameter if you need to run a command in a different directory. AVOID using `cd <directory> && <command>` patterns - use `workdir` instead.

Use `{tmp}` for temporary work outside the workspace. This directory has already been created, already exists, and is pre-approved for external directory access.

IMPORTANT: This tool is for terminal operations like git, npm, docker, etc. DO NOT use it for file operations (reading, writing, editing, searching, finding files) - use the specialized tools for this instead.

Before executing the command, please follow these steps:

1. Directory Verification:
   - If the command will create new directories or files, first use `ls` to verify the parent directory exists and is the correct location
   - For example, before running "mkdir foo/bar", first use `ls foo` to check that "foo" exists and is the intended parent directory

2. Command Execution:
   - Always quote file paths that contain spaces with double quotes (e.g., rm "path with spaces/file.txt")
   - Examples of proper quoting:
     - mkdir "/Users/name/My Documents" (correct)
     - mkdir /Users/name/My Documents (incorrect - will fail)
     - python "/path/with spaces/script.py" (correct)
     - python /path/with spaces/script.py (incorrect - will fail)
   - After ensuring proper quoting, execute the command.
   - Capture the output of the command.

Usage notes:
  - The command argument is required.
  - You can specify an optional timeout in milliseconds. If not specified, commands will time out after {default_timeout_ms}ms.
  - If the output exceeds {max_lines} lines or {max_bytes} bytes, it will be truncated and the full output will be written to a file. You can use Read with offset/limit to read specific sections or Grep to search the full content. Do NOT use `head`, `tail`, or other truncation commands to limit output; the full output will already be captured to a file for more precise searching.

  - Avoid using Bash with the `find`, `grep`, `cat`, `head`, `tail`, `sed`, `awk`, or `echo` commands, unless explicitly instructed or when these commands are truly necessary for the task. Instead, always prefer using the dedicated tools for these commands:
    - File search: Use Glob (NOT find or ls)
    - Content search: Use Grep (NOT grep or rg)
    - Read files: Use Read (NOT cat/head/tail)
    - Edit files: Use Edit (NOT sed/awk)
    - Write files: Use Write (NOT echo >/cat <<EOF)
    - Communication: Output text directly (NOT echo/printf)
  - When issuing multiple commands:
    - If the commands are independent and can run in parallel, make multiple bash tool calls in a single message.
    - If the commands depend on each other and must run sequentially, use a single Bash call with `&&`.
    - Use `;` only when you need to run commands sequentially but do not care if earlier commands fail.
    - DO NOT use newlines to separate commands (newlines are allowed in quoted strings).
  - AVOID using `cd <directory> && <command>`. Use the `workdir` parameter instead.

# Git and GitHub
- Only commit, amend, push, or create PRs when explicitly requested.
- Before committing, inspect `git status`, `git diff`, and `git log --oneline -10`; stage only intended files and never commit secrets.
- Write a concise commit message that matches the repo style.
- Do not update git config, skip hooks, use interactive `-i`, force-push, or create empty commits unless explicitly requested.
- If a commit fails or hooks reject it, fix the issue and create a new commit; do not amend the failed commit.
- Before creating a PR, inspect status, diff, remote tracking, recent commits, and the diff from the base branch.
- Review all commits included in the PR, not just the latest commit.
- Use `gh` for GitHub tasks, including PRs, issues, checks, and releases; return the PR URL when done."""


class BashTool(ShellTool):
    name = "bash"
    description = BASH_DESCRIPTION_TEMPLATE
    tool_kind = ToolKind.EXECUTE

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        command: str = Field(
            min_length=1,
            description="The command to execute",
        )
        timeout: int | None = Field(
            default=None,
            gt=0,
            description="Optional timeout in milliseconds",
        )
        workdir: str | None = Field(
            default=None,
            description="Working directory; defaults to the active working directory",
        )

        @field_validator("command")
        @classmethod
        def _command_not_blank(cls, value: str) -> str:
            if not value.strip():
                raise ValueError("command must not be blank")
            return value

    def __init__(
        self,
        workdir: str | os.PathLike[str] | None = None,
        shell: str | None = None,
        output_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__(workdir)
        self.shell = shell or os.environ.get("SHELL") or "/bin/bash"
        self.output_dir = str(output_dir) if output_dir is not None else None
        self.description = BASH_DESCRIPTION_TEMPLATE.format(
            os=platform.system(),
            shell=self.shell,
            tmp=tempfile.gettempdir(),
            default_timeout_ms=BASH_DEFAULT_TIMEOUT_MS,
            max_lines=BASH_MAX_OUTPUT_LINES,
            max_bytes=BASH_MAX_OUTPUT_BYTES,
        )

    def build_permission_requests(
        self, args: dict, context: ToolContext,
    ) -> list[PermissionRequest]:
        command = args["command"].strip()
        head = command.split()[0]
        workdir = self._resolve(args["workdir"]) if args.get("workdir") else self.workdir
        return [self._access_request(workdir), PermissionRequest(
            resources=[ResourcePermission(permission="bash", resource=command)],
            answer_options=[
                AnswerOption(
                    resource_permissions=[ResourcePermission(
                        permission="bash", resource=f"{head} *",
                    )],
                    metadata={"preview": f"Run any '{head}' command"},
                ),
            ],
            metadata={"preview": f"Run command: {command}"},
        )]

    async def _run(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        workdir = self._resolve(args["workdir"]) if args.get("workdir") else self.workdir
        if not workdir.exists():
            raise ShellToolError(f"workdir does not exist: {workdir}")
        if not workdir.is_dir():
            raise ShellToolError(f"workdir is not a directory: {workdir}")
        timeout_ms = args.get("timeout") or BASH_DEFAULT_TIMEOUT_MS
        try:
            process = await asyncio.create_subprocess_exec(
                self.shell, "-c", args["command"],
                cwd=workdir,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            raise ShellToolError(f"Failed to start shell: {error}") from error
        output, outcome = await self._collect(
            process, timeout_ms, cancellation_token,
        )
        return self._render(output, outcome, process, timeout_ms)

    async def _collect(
        self,
        process: asyncio.subprocess.Process,
        timeout_ms: int,
        cancellation_token: CancellationToken,
    ) -> tuple[str, str]:
        chunks: list[bytes] = []

        async def _pump() -> None:
            while chunk := await process.stdout.read(65_536):
                chunks.append(chunk)

        pump = asyncio.create_task(_pump())
        cancelled = asyncio.create_task(cancellation_token.wait_cancelled())
        outcome = "completed"
        try:
            done, _ = await asyncio.wait(
                {pump, cancelled},
                timeout=timeout_ms / 1_000,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if pump in done:
                await process.wait()
            else:
                outcome = "cancelled" if cancelled in done else "timed_out"
                _kill_process_group(process)
                await process.wait()
                await pump
        except asyncio.CancelledError:
            _kill_process_group(process)
            await process.wait()
            raise
        finally:
            for task in (cancelled, pump):
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        return b"".join(chunks).decode("utf-8", errors="replace"), outcome

    def _render(
        self,
        output: str,
        outcome: str,
        process: asyncio.subprocess.Process,
        timeout_ms: int,
    ) -> ExecutionResult:
        exit_code = process.returncode if outcome == "completed" else None
        text, truncated, output_path = self._truncate(output)
        if not text.strip():
            text = "(no output)"
        if outcome == "timed_out":
            text += (
                f"\n\n<shell_metadata>\nshell tool terminated command after"
                f" exceeding timeout {timeout_ms} ms. If this command is expected"
                " to take longer and is not waiting for interactive input, retry"
                " with a larger timeout value in milliseconds.\n</shell_metadata>"
            )
        elif outcome == "cancelled":
            text += (
                "\n\n<shell_metadata>\nshell tool cancelled the command before"
                " completion; partial output is shown above.\n</shell_metadata>"
            )
        return ExecutionResult(
            content=[TextContent(text=text)],
            metadata={
                "exit": exit_code,
                "truncated": truncated,
                "output_path": output_path,
            },
            is_error=outcome != "completed" or exit_code != 0,
        )

    def _truncate(self, output: str) -> tuple[str, bool, str | None]:
        lines = output.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        if (
            len(lines) <= BASH_MAX_OUTPUT_LINES
            and len(output.encode("utf-8")) <= BASH_MAX_OUTPUT_BYTES
        ):
            return output, False, None
        handle, output_path = tempfile.mkstemp(
            prefix="bash_output_", suffix=".txt", dir=self.output_dir,
        )
        with os.fdopen(handle, "w", encoding="utf-8", errors="replace") as stream:
            stream.write(output)
        preview = "\n".join(lines[-BASH_MAX_OUTPUT_LINES:])
        data = preview.encode("utf-8")
        if len(data) > BASH_MAX_OUTPUT_BYTES:
            preview = data[-BASH_MAX_OUTPUT_BYTES:].decode("utf-8", errors="replace")
        text = (
            "...output truncated...\n\n"
            f"Full output saved to: {output_path}\n\n{preview}"
        )
        return text, True, output_path
