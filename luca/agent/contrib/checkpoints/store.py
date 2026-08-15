"""`ShadowGitStore` — workspace snapshots in a git repository the user never sees.

A SHADOW repo: its git dir lives beside the sessions under the luca store, its
work-tree IS the workspace, and every command names both explicitly
(`git --git-dir=… --work-tree=… …`). Nothing is ever written into the
workspace, the user's own `.git` is excluded rather than traversed, and no
command is ever run from inside it. A project that is not a git repository at
all snapshots exactly the same way — the shadow repo does not care.

WHY GIT AND NOT PER-EDIT RECORDS. The alternative is asking each mutating tool
to record what it replaced. That covers `edit` / `write` / `apply_patch` and
fails completely for `bash`, which is a first-class unsandboxed tool: one
`sed -i` or `rm` and the inverse record does not exist. A snapshot is
tool-agnostic — it captures the workspace, however it got that way — and gets
renames and deletions for free.

BLOCKING BY DESIGN, like `contrib/tui/gitinfo.py`: these are subprocesses, they
are called through `asyncio.to_thread` by `CheckpointService`, and every one of
them is bounded by `timeout`. A store that cannot run git (no binary, a failed
init, a command that timed out) reports `available() is False` and every
operation becomes a no-op returning None/False, so checkpoints degrade into an
absent feature rather than a broken turn.

WHAT IS NOT CAPTURED. `add -A` reads the workspace's own `.gitignore`, so an
ignored path is neither snapshotted nor restored. That is deliberate — without
it the first snapshot of any real project would try to swallow `node_modules`
and a virtualenv — and it is the feature's main limitation: edits the agent
makes to ignored files are not undoable. `DEFAULT_EXCLUDES` adds the few paths
that are never source and would be catastrophic to capture even in a project
with no `.gitignore` at all. Build output (`dist/`, `build/`) is deliberately
NOT in that list: it is real content in plenty of projects, and one that wants
it ignored already says so in its `.gitignore`.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Never snapshotted, whatever the workspace's own `.gitignore` says. `.git` is
# mandatory (it is the user's repository metadata, and `add -A` would try to
# record it as a gitlink); the rest are directories that are never source and
# whose size would make the first snapshot unusable.
DEFAULT_EXCLUDES: tuple[str, ...] = (
    ".git/",
    ".luca/",
    "node_modules/",
    ".venv/",
    "venv/",
    "__pycache__/",
    "*.pyc",
)

# The shadow repo commits under its own identity, so a machine with no global
# git identity configured can still snapshot.
_IDENTITY = (("user.name", "luca"), ("user.email", "luca@localhost"))


@dataclass(frozen=True)
class GitResult:
    ok: bool
    stdout: str = ""


class ShadowGitStore:
    """Snapshot and restore one workspace through a private git repository."""

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        git_dir: str | os.PathLike[str],
        *,
        timeout: float = 30.0,
        excludes: tuple[str, ...] = DEFAULT_EXCLUDES,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.git_dir = Path(git_dir).expanduser()
        self.timeout = timeout
        self.excludes = excludes
        # Latches to False the first time git is missing or a command fails in
        # a way that makes the store untrustworthy. Never latches back.
        self._disabled = False
        self._ready = False

    # ── availability ─────────────────────────────────────────────────────────

    def available(self) -> bool:
        """Can this store snapshot? A missing git binary, or any earlier
        failure that disabled it, answers False and keeps answering False."""
        return not self._disabled and shutil.which("git") is not None

    def _disable(self, reason: str) -> None:
        if not self._disabled:
            logger.warning("checkpoints disabled for %s: %s", self.workspace, reason)
        self._disabled = True

    # ── the git boundary ─────────────────────────────────────────────────────

    def _git(self, *args: str, check: bool = True) -> GitResult:
        """Run one git command against the shadow repo. Never raises: a
        failure is logged and reported, because losing a checkpoint must never
        take a turn down with it."""
        command = [
            "git",
            "--git-dir",
            str(self.git_dir),
            "--work-tree",
            str(self.workspace),
            *args,
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._disable(f"git {args[0] if args else ''} failed: {exc}")
            return GitResult(ok=False)
        if completed.returncode != 0:
            logger.warning(
                "git %s failed (%s): %s",
                " ".join(args),
                completed.returncode,
                completed.stderr.strip(),
            )
            if check:
                return GitResult(ok=False)
        return GitResult(ok=completed.returncode == 0, stdout=completed.stdout.strip())

    def ensure(self) -> bool:
        """Create and configure the shadow repo if it is not there yet.
        Idempotent, and cheap after the first call."""
        if self._ready:
            return True
        if not self.available():
            return False
        if not self.git_dir.exists():
            self.git_dir.parent.mkdir(parents=True, exist_ok=True)
            if not self._git("init", "--quiet").ok:
                self._disable("could not initialise the shadow repository")
                return False
            for key, value in _IDENTITY:
                self._git("config", key, value)
            # A signing configuration inherited from the user's global config
            # would make every snapshot prompt or fail.
            self._git("config", "commit.gpgsign", "false")
        exclude_file = self.git_dir / "info" / "exclude"
        try:
            exclude_file.parent.mkdir(parents=True, exist_ok=True)
            exclude_file.write_text("\n".join(self.excludes) + "\n")
        except OSError as exc:
            self._disable(f"could not write {exclude_file}: {exc}")
            return False
        self._ready = True
        return True

    # ── the two operations ───────────────────────────────────────────────────

    def snapshot(self, label: str) -> str | None:
        """Commit the workspace as it stands and return the commit sha, or
        None when the store is unavailable.

        `--allow-empty` because a checkpoint is a POSITION, not a diff: two
        consecutive turns that changed nothing must still produce two distinct
        commits, or restoring the second would silently restore the first."""
        if not self.ensure():
            return None
        if not self._git("add", "--all").ok:
            return None
        if not self._git("commit", "--allow-empty", "--quiet", "--message", label).ok:
            return None
        head = self._git("rev-parse", "HEAD")
        return head.stdout or None

    def restore(self, commit: str) -> bool:
        """Put the workspace back to `commit`. True when it happened.

        Two steps, and both are needed. `read-tree -u --reset` makes every
        TRACKED path match the commit, which covers a modified file and a
        deleted one; `clean -fd` removes what the agent created since, which
        `read-tree` leaves alone because it is untracked.

        `clean` runs WITHOUT `-x`, so it obeys the workspace's `.gitignore` and
        our own excludes: a virtualenv or a build directory is never deleted.
        What it does remove is any untracked, non-ignored file created since the
        snapshot — including one a human made by hand. That is inherent to
        restoring a workspace to an earlier moment, and it is exactly the set
        `add -A` would have captured."""
        if not self.ensure():
            return False
        if not self.has(commit):
            logger.warning("checkpoint commit %s is not in the shadow repository", commit)
            return False
        if not self._git("read-tree", "-u", "--reset", commit).ok:
            return False
        return self._git("clean", "-fd", "--quiet").ok

    def has(self, commit: str) -> bool:
        """Is `commit` a commit object in the shadow repo? Guards a restore
        against an index row whose repository was deleted underneath it."""
        if not self.ensure():
            return False
        return self._git("cat-file", "-e", f"{commit}^{{commit}}", check=False).ok
