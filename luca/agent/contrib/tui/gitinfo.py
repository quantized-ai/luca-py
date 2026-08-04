"""Git branch/dirty lookup for the status bar. Subprocess-based and blocking
— call it off the UI thread (`asyncio.to_thread`). Absent git or a non-repo
workspace degrades to no branch segment."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GitInfo:
    branch: str | None = None
    dirty: bool = False


def read_git_info(workspace: str | os.PathLike[str] = ".") -> GitInfo:
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if branch.returncode != 0:
            return GitInfo()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return GitInfo()
    return GitInfo(
        branch=branch.stdout.strip() or None,
        dirty=status.returncode == 0 and bool(status.stdout.strip()),
    )
