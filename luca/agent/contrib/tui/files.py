"""Workspace file listing + matching for the `@` context picker.

Prefers `git ls-files` (tracked + untracked, exclusions honored) and falls
back to a bounded walk. Token costs are the project's standard estimate
(size / 4). Matching is contiguous and case-insensitive; the matched
substring is wrapped in `[accent]…[/]` spans — the highlight is the only
feedback that the query is doing anything.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

MAX_FILES = 2_000
MAX_ROWS = 8
_SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"}


def list_workspace_files(workspace: str | os.PathLike[str] = ".") -> list[str]:
    root = Path(workspace)
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            files = [line for line in result.stdout.splitlines() if line]
            return sorted(files)[:MAX_FILES]
    except (OSError, subprocess.TimeoutExpired):
        pass
    found: list[str] = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
        for filename in filenames:
            found.append(str((Path(directory) / filename).relative_to(root)))
            if len(found) >= MAX_FILES:
                return sorted(found)
    return sorted(found)


def estimate_tokens(path: Path) -> int:
    try:
        return max(1, path.stat().st_size // 4)
    except OSError:
        return 0


def match_files(
    files: list[str],
    query: str,
    workspace: str | os.PathLike[str] = ".",
    *,
    limit: int = MAX_ROWS,
) -> list[tuple[str, str, int]]:
    """`(path, span-marked path, token estimate)` rows for the picker."""
    root = Path(workspace)
    needle = query.lower()
    rows: list[tuple[str, str, int]] = []
    for path in files:
        if needle:
            position = path.lower().find(needle)
            if position < 0:
                continue
            marked = (
                f"{path[:position]}[accent]{path[position : position + len(needle)]}[/]{path[position + len(needle) :]}"
            )
        else:
            marked = path
        rows.append((path, marked, estimate_tokens(root / path)))
        if len(rows) >= limit:
            break
    return rows
