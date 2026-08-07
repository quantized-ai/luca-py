"""Executor for OpenAI's `apply_patch` operations.

An `ApplyPatchToolCall` carries exactly one operation —
`{"type": "create_file"|"update_file"|"delete_file", "path", "diff"?,
"move_to"?}` — and `execute_apply_patch` performs it against a root
directory. The diff itself is [`v4a.apply_diff`](v4a.py); this module only
adds the filesystem.

The returned strings are the ones OpenAI's own harness returns ("Updated
<path>", the two-line move form, …): the model has seen them.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._fs import read_text, remove_file, resolve_within, write_text
from .errors import NativeToolError
from .v4a import apply_diff


def execute_apply_patch(operation: Mapping[str, Any], *, root: str | Path) -> str:
    """Run one `apply_patch` operation against `root`.

    `operation` is the call's `arguments` (equivalently `call.operation`).
    Returns the text to send back as the tool result; every failure the
    model can recover from is a `NativeToolError`.
    """
    op_type = _require(operation, "type")
    path = _require(operation, "path")
    target = resolve_within(root, path)

    if op_type == "delete_file":
        remove_file(target, display=path)
        return f"Deleted {path}"

    if op_type not in ("create_file", "update_file"):
        raise NativeToolError(f"Unknown operation type: {op_type}")

    diff = operation.get("diff")
    if diff is None:
        raise NativeToolError(f"Missing diff for operation type {op_type} on path {path}")

    if op_type == "create_file":
        write_text(target, apply_diff("", diff, mode="create"), display=path)
        return f"Created {path}"

    updated = apply_diff(read_text(target, display=path), diff)
    move_to = operation.get("move_to")
    if move_to is None:
        write_text(target, updated, display=path)
        return f"Updated {path}"

    moved_target = resolve_within(root, move_to)
    write_text(moved_target, updated, display=move_to)
    if moved_target != target:
        remove_file(target, display=path)
    return f"Updated {path}\nMoved {path} to {move_to}"


def _require(operation: Mapping[str, Any], key: str) -> Any:
    if key not in operation:
        raise NativeToolError(f"Missing {key!r} in apply_patch operation.")
    return operation[key]


__all__ = ["execute_apply_patch"]
