r"""Anthropic's `str_replace_based_edit_tool` (`text_editor_20250728`).

Two layers in one module — the pure string operations exist only to serve
the four commands, and splitting them off would leave a module whose single
consumer is its sibling:

- `view` / `str_replace` / `insert` transform text and know nothing about
  files;
- `execute_text_editor` runs one command dict against a root directory.

Lines are split on `"\n"` and nothing else. `str.splitlines()` would be the
obvious primitive and is the wrong one: it also breaks on `\r`, `\v`, `\f`,
`\x1c`, `\x1d`, `\x1e`, `\x85`, U+2028 and U+2029, so rejoining would replace
each of those with a newline — rewriting content the model never touched —
and its line numbers would disagree with `cat -n`, which `view` claims to
reproduce.

`undo_edit` is gone as of `text_editor_20250728` and is not implemented.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._fs import list_dir, read_text, resolve_within, write_text
from .errors import NativeToolError

LISTING_DEPTH = 2
"""How deep `view` descends into a directory."""


def view(text: str, view_range: Sequence[int] | None = None) -> str:
    """Render `text` `cat -n` style: 1-indexed numbers, six columns, a tab.

    `view_range` is `[start, end]` over those same numbers, `end == -1`
    meaning the last line. A trailing newline does not make a final empty
    line — the numbering matches `insert_line`.
    """
    lines = _split_lines(text)
    if view_range is None:
        return _numbered(lines, 1)

    # `view_range` arrives straight off the wire, so the annotation is the
    # contract and this is the enforcement.
    if not isinstance(view_range, (list, tuple)) or len(view_range) != 2:
        raise NativeToolError(f"view_range must be [start, end]; got {view_range!r}.")
    start = _as_line_number(view_range[0], "view_range start")
    end = _as_line_number(view_range[1], "view_range end")
    if end == -1:
        end = len(lines)
    if not 1 <= start <= len(lines):
        raise NativeToolError(f"view_range start {start} is out of range; the file has {len(lines)} lines.")
    if not start <= end <= len(lines):
        raise NativeToolError(f"view_range end {end} is out of range; expected {start}..{len(lines)} or -1.")
    return _numbered(lines[start - 1 : end], start)


def str_replace(text: str, old_str: str, new_str: str) -> str:
    """Replace `old_str`, which must occur EXACTLY once."""
    matches = text.count(old_str)
    if matches == 0:
        raise NativeToolError(
            "No match found for old_str. It must match the file exactly, whitespace and indentation included."
        )
    if matches > 1:
        raise NativeToolError(
            f"Found {matches} matches for old_str. Add surrounding context so it matches exactly one location."
        )
    return text.replace(old_str, new_str, 1)


def insert(text: str, insert_line: int, insert_text: str) -> str:
    """Insert `insert_text` AFTER line `insert_line`; `0` means the top.

    The upper bound is the line count, which appends. The result keeps the
    original's trailing newline (empty input gains one).
    """
    lines = _split_lines(text)
    insert_line = _as_line_number(insert_line, "insert_line")
    if not 0 <= insert_line <= len(lines):
        raise NativeToolError(f"insert_line {insert_line} is out of range; expected 0..{len(lines)}.")
    merged = lines[:insert_line] + _split_lines(insert_text) + lines[insert_line:]
    trailing = "\n" if text == "" or text.endswith("\n") else ""
    return "\n".join(merged) + trailing


def execute_text_editor(command: Mapping[str, Any], *, root: str | Path) -> str:
    """Run one `str_replace_based_edit_tool` command against `root`.

    `command` is the call's `arguments` — `{"command": ..., "path": ..., …}`.
    Returns the text to send back as the tool result; every failure the model
    can recover from is a `NativeToolError`.
    """
    name = _require(command, "command")
    path = _require(command, "path")
    target = resolve_within(root, path)

    if name == "view":
        if target.is_dir():
            return _list_directory(target)
        return view(read_text(target, display=path), command.get("view_range"))

    if name == "create":
        write_text(target, _require(command, "file_text"), display=path)
        return f"Created {path}"

    if name == "str_replace":
        updated = str_replace(
            read_text(target, display=path),
            _require(command, "old_str"),
            _require(command, "new_str"),
        )
        write_text(target, updated, display=path)
        return f"Updated {path}"

    if name == "insert":
        updated = insert(
            read_text(target, display=path),
            _require(command, "insert_line"),
            _require(command, "insert_text"),
        )
        write_text(target, updated, display=path)
        return f"Updated {path}"

    raise NativeToolError(f"Unsupported text editor command: {name!r}. Expected view, create, str_replace or insert.")


def _split_lines(text: str) -> list[str]:
    """`text` as lines, a trailing newline NOT making a final empty one."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _numbered(lines: Sequence[str], start: int) -> str:
    return "\n".join(f"{number:6}\t{line}" for number, line in enumerate(lines, start))


def _list_directory(target: Path) -> str:
    """Paths relative to `target`, depth-first, directories marked with `/`.

    Hidden entries are skipped and the walk stops at `LISTING_DEPTH`.
    Symlinks are named with a trailing `@` and never followed: descending one
    would disclose paths that `resolve_within` refuses to open, and a link
    back up the tree would make the walk unbounded.
    """
    entries: list[str] = []

    def walk(directory: Path, prefix: str, depth: int) -> None:
        for child in list_dir(directory):
            if child.name.startswith("."):
                continue
            if child.is_symlink():
                entries.append(f"{prefix}{child.name}@")
                continue
            if not child.is_dir():
                entries.append(f"{prefix}{child.name}")
                continue
            entries.append(f"{prefix}{child.name}/")
            if depth < LISTING_DEPTH:
                walk(child, f"{prefix}{child.name}/", depth + 1)

    walk(target, "", 1)
    return "\n".join(entries)


def _as_line_number(value: Any, field: str) -> int:
    # `bool` is an `int` subclass; `true` is never a line number.
    if isinstance(value, bool) or not isinstance(value, int):
        raise NativeToolError(f"{field} must be an integer; got {value!r}.")
    return value


def _require(command: Mapping[str, Any], key: str) -> Any:
    if key not in command:
        raise NativeToolError(f"Missing {key!r} in text editor command.")
    return command[key]


__all__ = ["execute_text_editor", "insert", "str_replace", "view"]
