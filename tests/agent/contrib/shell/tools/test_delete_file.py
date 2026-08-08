"""Unit tests for the shell `delete_file` tool: removal semantics, the
directory and missing-file refusals, symlinks, the deliberate absence of the
read-first contract, and its permission resource."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from luca.agent.contrib.resource_permissions import ResourcePermission
from luca.agent.contrib.shell import DeleteFileTool


def make_tool(tmp_path) -> DeleteFileTool:
    return DeleteFileTool(workdir=tmp_path)


def body(result) -> str:
    return result.content[0].text


def raise_permission_denied(self) -> None:
    raise OSError(13, "Permission denied")


# ── scenario 1: delete a file ─────────────────────────────────────────────────


async def test_deletes_a_file_that_was_never_read(tmp_path, run):
    target = tmp_path / "obsolete.txt"
    target.write_text("bye")

    result = await run(make_tool(tmp_path), {"file_path": str(target)})

    assert result.is_error is False
    assert body(result) == f"Deleted file: {target}"
    assert result.metadata == {}
    assert target.exists() is False


async def test_relative_path_resolves_from_the_workdir(tmp_path, run):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/f.txt").write_text("x")

    await run(make_tool(tmp_path), {"file_path": "nested/f.txt"})

    assert (tmp_path / "nested/f.txt").exists() is False


async def test_an_empty_file_is_deleted(tmp_path, run):
    target = tmp_path / "empty.txt"
    target.touch()

    result = await run(make_tool(tmp_path), {"file_path": "empty.txt"})

    assert result.is_error is False
    assert target.exists() is False


async def test_the_parent_directory_survives_the_deletion(tmp_path, run):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/f.txt").write_text("x")

    await run(make_tool(tmp_path), {"file_path": "nested/f.txt"})

    assert (tmp_path / "nested").is_dir() is True


# ── scenario 2: a binary file — no read-first contract ────────────────────────


async def test_a_binary_file_the_read_tool_refuses_is_still_deletable(tmp_path, run):
    target = tmp_path / "module.pyc"
    target.write_bytes(b"\x00\x01\x02binary\x00")

    result = await run(make_tool(tmp_path), {"file_path": "module.pyc"})

    assert result.is_error is False
    assert target.exists() is False


# ── scenario 3: missing file ──────────────────────────────────────────────────


async def test_a_missing_file_is_an_error(tmp_path, run):
    result = await run(make_tool(tmp_path), {"file_path": "gone.txt"})

    assert result.is_error is True
    assert body(result) == f"File not found: {tmp_path / 'gone.txt'}"


async def test_a_path_under_a_file_shaped_parent_is_reported_missing(tmp_path, run):
    (tmp_path / "blocker").write_text("i am a file")

    result = await run(make_tool(tmp_path), {"file_path": "blocker/child.txt"})

    assert result.is_error is True
    assert body(result) == f"File not found: {tmp_path / 'blocker/child.txt'}"


# ── scenario 4: directories are refused ───────────────────────────────────────


async def test_a_directory_is_refused_and_left_in_place(tmp_path, run):
    target = tmp_path / "src"
    target.mkdir()
    (target / "keep.txt").write_text("still here")

    result = await run(make_tool(tmp_path), {"file_path": "src"})

    assert result.is_error is True
    assert body(result) == f"Path is a directory, not a file: {target}"
    assert (target / "keep.txt").read_text() == "still here"


# ── scenario 5: symlinks ──────────────────────────────────────────────────────


async def test_a_symlink_is_removed_and_its_target_is_untouched(tmp_path, run):
    target = tmp_path / "real.txt"
    target.write_text("keep me")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    result = await run(make_tool(tmp_path), {"file_path": "link.txt"})

    assert result.is_error is False
    assert link.is_symlink() is False
    assert target.read_text() == "keep me"


async def test_a_symlink_to_a_directory_is_removed_and_the_directory_survives(tmp_path, run):
    target = tmp_path / "real"
    target.mkdir()
    (target / "keep.txt").write_text("still here")
    link = tmp_path / "link"
    link.symlink_to(target)

    result = await run(make_tool(tmp_path), {"file_path": "link"})

    assert result.is_error is False
    assert body(result) == f"Deleted file: {link}"
    assert link.is_symlink() is False
    assert (target / "keep.txt").read_text() == "still here"


async def test_a_broken_symlink_is_removed(tmp_path, run):
    link = tmp_path / "dangling.txt"
    link.symlink_to(tmp_path / "never-existed.txt")

    result = await run(make_tool(tmp_path), {"file_path": "dangling.txt"})

    assert result.is_error is False
    assert link.is_symlink() is False


# ── scenario 6: deletion failure ──────────────────────────────────────────────


async def test_an_undeletable_file_is_an_error_not_a_success(tmp_path, run, monkeypatch):
    target = tmp_path / "f.txt"
    target.write_text("still here")
    monkeypatch.setattr(Path, "unlink", raise_permission_denied)

    result = await run(make_tool(tmp_path), {"file_path": "f.txt"})

    assert result.is_error is True
    assert body(result) == "Failed to delete file: [Errno 13] Permission denied"
    assert target.read_text() == "still here"


# ── arguments ─────────────────────────────────────────────────────────────────


def test_args_require_a_non_empty_file_path_and_forbid_extras():
    with pytest.raises(ValidationError):
        DeleteFileTool.Args.model_validate({})
    with pytest.raises(ValidationError):
        DeleteFileTool.Args.model_validate({"file_path": ""})
    with pytest.raises(ValidationError):
        DeleteFileTool.Args.model_validate({"file_path": "f.txt", "recursive": True})


# ── permission resource ───────────────────────────────────────────────────────


def test_permission_resource_exposes_the_resolved_path(tmp_path, perm):
    [access, request] = perm(make_tool(tmp_path), {"file_path": "nested/deep/file.txt"})

    path = tmp_path / "nested/deep/file.txt"
    assert access.resources == [
        ResourcePermission(
            permission="access_directory",
            resource=str(path.parent),
        ),
    ]
    assert access.metadata["preview"] == f"Access directory {path.parent}"
    assert [(o.resource_permissions, o.metadata["preview"]) for o in access.answer_options] == [
        (
            [
                ResourcePermission(
                    permission="access_directory",
                    resource=str(path.parent),
                ),
                ResourcePermission(
                    permission="access_directory",
                    resource=f"{path.parent}/*",
                ),
            ],
            f"Always allow access to {path.parent}",
        ),
    ]
    assert request.resources == [
        ResourcePermission(permission="delete_file", resource=str(path)),
    ]
    assert request.metadata["preview"] == f"Delete {path}"
    assert [(o.resource_permissions, o.metadata["preview"]) for o in request.answer_options] == [
        (
            [ResourcePermission(permission="delete_file", resource=f"{path.parent}/*")],
            f"Delete files under {path.parent}",
        ),
    ]


def test_a_directory_target_scopes_access_to_itself_not_its_parent(tmp_path, perm):
    (tmp_path / "src").mkdir()

    [access, _] = perm(make_tool(tmp_path), {"file_path": "src"})

    # The call is refused at execution, but the approval it raises first must
    # not ask for a grant over the parent — that is `tmp_path` itself.
    assert access.resources == [
        ResourcePermission(
            permission="access_directory",
            resource=str(tmp_path / "src"),
        ),
    ]
    assert access.metadata["preview"] == f"Access directory {tmp_path / 'src'}"
    assert [(o.resource_permissions, o.metadata["preview"]) for o in access.answer_options] == [
        (
            [
                ResourcePermission(
                    permission="access_directory",
                    resource=str(tmp_path / "src"),
                ),
                ResourcePermission(
                    permission="access_directory",
                    resource=f"{tmp_path / 'src'}/*",
                ),
            ],
            f"Always allow access to {tmp_path / 'src'}",
        ),
    ]
