"""`ShadowGitStore`: snapshot and restore a real workspace on disk.

Real git, real files, `tmp_path` — a store whose whole job is talking to a
subprocess is not worth testing against a double. Each test is
precondition → one action → postcondition over the workspace.

The suite skips entirely without a git binary, which is also the condition the
store itself degrades under.
"""

import shutil
from pathlib import Path

import pytest

from luca.agent.contrib.checkpoints.store import ShadowGitStore

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="the shadow store needs a git binary",
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    space = tmp_path / "workspace"
    space.mkdir()
    (space / "kept.py").write_text("original\n")
    return space


@pytest.fixture
def store(workspace: Path, tmp_path: Path) -> ShadowGitStore:
    return ShadowGitStore(workspace, tmp_path / "store" / "checkpoints.git")


# ── the round trip ────────────────────────────────────────────────────────────


def test_a_modified_file_is_restored(store: ShadowGitStore, workspace: Path):
    commit = store.snapshot("before")

    (workspace / "kept.py").write_text("agent wrote this\n")
    store.restore(commit)

    assert (workspace / "kept.py").read_text() == "original\n"


def test_a_created_file_is_removed(store: ShadowGitStore, workspace: Path):
    commit = store.snapshot("before")

    (workspace / "invented.py").write_text("agent invented this\n")
    store.restore(commit)

    assert not (workspace / "invented.py").exists()


def test_a_deleted_file_is_brought_back(store: ShadowGitStore, workspace: Path):
    commit = store.snapshot("before")

    (workspace / "kept.py").unlink()
    store.restore(commit)

    assert (workspace / "kept.py").read_text() == "original\n"


def test_a_created_directory_is_removed(store: ShadowGitStore, workspace: Path):
    commit = store.snapshot("before")

    (workspace / "pkg").mkdir()
    (workspace / "pkg" / "mod.py").write_text("new\n")
    store.restore(commit)

    assert not (workspace / "pkg").exists()


def test_restoring_an_older_checkpoint_skips_the_newer_one(store: ShadowGitStore, workspace: Path):
    first = store.snapshot("first")
    (workspace / "kept.py").write_text("second\n")
    store.snapshot("second")
    (workspace / "kept.py").write_text("third\n")

    store.restore(first)

    assert (workspace / "kept.py").read_text() == "original\n"


def test_two_checkpoints_over_an_unchanged_workspace_are_distinct(store: ShadowGitStore):
    """A checkpoint is a position, not a diff — `--allow-empty` is what keeps
    the second one from silently aliasing the first."""
    first = store.snapshot("first")
    second = store.snapshot("second")

    assert first is not None
    assert second is not None
    assert first != second


# ── what is deliberately left alone ───────────────────────────────────────────


def test_gitignored_paths_are_neither_snapshotted_nor_restored(store: ShadowGitStore, workspace: Path):
    """The documented limitation: an ignored file the agent edits is not
    undoable, because capturing ignored paths would swallow every virtualenv."""
    (workspace / ".gitignore").write_text("secret.txt\n")
    (workspace / "secret.txt").write_text("original\n")
    commit = store.snapshot("before")

    (workspace / "secret.txt").write_text("agent wrote this\n")
    store.restore(commit)

    assert (workspace / "secret.txt").read_text() == "agent wrote this\n"


def test_an_ignored_directory_survives_a_restore(store: ShadowGitStore, workspace: Path):
    """`clean` runs without `-x`, so a restore never deletes a virtualenv."""
    (workspace / "node_modules").mkdir()
    (workspace / "node_modules" / "dep.js").write_text("dependency\n")
    commit = store.snapshot("before")

    store.restore(commit)

    assert (workspace / "node_modules" / "dep.js").read_text() == "dependency\n"


def test_the_workspaces_own_git_directory_is_untouched(store: ShadowGitStore, workspace: Path):
    """The shadow repo excludes `.git/` rather than traversing it, so a real
    repository in the workspace keeps its own history."""
    real_git = workspace / ".git"
    real_git.mkdir()
    (real_git / "HEAD").write_text("ref: refs/heads/main\n")
    commit = store.snapshot("before")

    (workspace / "kept.py").write_text("changed\n")
    store.restore(commit)

    assert (real_git / "HEAD").read_text() == "ref: refs/heads/main\n"


def test_nothing_is_written_into_the_workspace(store: ShadowGitStore, workspace: Path):
    store.snapshot("before")

    assert sorted(p.name for p in workspace.iterdir()) == ["kept.py"]


# ── degradation ───────────────────────────────────────────────────────────────


def test_a_store_without_git_is_unavailable_and_snapshots_nothing(store: ShadowGitStore, monkeypatch):
    monkeypatch.setattr("luca.agent.contrib.checkpoints.store.shutil.which", lambda _: None)

    assert store.available() is False
    assert store.snapshot("before") is None
    assert store.restore("deadbeef") is False


def test_restoring_an_unknown_commit_reports_failure(store: ShadowGitStore, workspace: Path):
    store.snapshot("before")
    (workspace / "kept.py").write_text("changed\n")

    assert store.restore("0" * 40) is False
    assert (workspace / "kept.py").read_text() == "changed\n"


def test_the_store_survives_a_workspace_with_no_files(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    store = ShadowGitStore(empty, tmp_path / "store" / "checkpoints.git")

    commit = store.snapshot("before")

    assert commit is not None
    assert store.restore(commit) is True
