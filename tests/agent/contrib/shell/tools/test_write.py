"""Unit tests for the shell `write` tool: create/overwrite semantics, parent
creation, the read-first contract, BOM handling, exact content round-trips,
and its permission resource."""

import pytest
from pydantic import ValidationError

from luca.agent.contrib.resource_permissions import ResourcePermission
from luca.agent.contrib.shell import FileReadTracker, WriteTool

BOM = b"\xef\xbb\xbf"


def make_tool(tmp_path, *read_paths) -> WriteTool:
    """A WriteTool whose tracker has already 'read' the given paths."""
    tracker = FileReadTracker()
    for path in read_paths:
        tracker.record(path)
    return WriteTool(workdir=tmp_path, tracker=tracker)


def body(result) -> str:
    return result.content[0].text


# ── scenario 1: create a file ─────────────────────────────────────────────────


async def test_creates_a_missing_file(tmp_path, run):
    target = tmp_path / "newfile.txt"

    result = await run(make_tool(tmp_path), {
        "file_path": str(target), "content": "Hello, World!",
    })

    assert result.is_error is False
    assert body(result) == f"File created successfully at: {target}"
    assert result.metadata == {"existed": False}
    assert target.read_text() == "Hello, World!"


# ── scenario 2: create parent directories ─────────────────────────────────────


async def test_creates_missing_parent_directories(tmp_path, run):
    result = await run(make_tool(tmp_path), {
        "file_path": "nested/deep/file.txt", "content": "data",
    })

    assert result.is_error is False
    assert (tmp_path / "nested/deep/file.txt").read_text() == "data"


# ── scenario 3: relative path ─────────────────────────────────────────────────


async def test_relative_path_resolves_from_the_workdir(tmp_path, run):
    await run(make_tool(tmp_path), {"file_path": "rel.txt", "content": "x"})

    assert (tmp_path / "rel.txt").read_text() == "x"


# ── scenario 4: overwrite existing file ───────────────────────────────────────


async def test_overwrites_an_existing_file_after_a_read(tmp_path, run):
    target = tmp_path / "f.txt"
    target.write_text("old content")

    result = await run(make_tool(tmp_path, target), {
        "file_path": "f.txt", "content": "new content",
    })

    assert body(result) == f"File updated successfully at: {target}"
    assert result.metadata == {"existed": True}
    assert target.read_text() == "new content"


# ── scenario 5: read-first requirement ────────────────────────────────────────


async def test_overwriting_an_unread_file_fails_and_leaves_it_unchanged(tmp_path, run):
    target = tmp_path / "f.txt"
    target.write_text("old content")

    result = await run(make_tool(tmp_path), {
        "file_path": "f.txt", "content": "new content",
    })

    assert result.is_error is True
    assert body(result) == (
        f"File has not been read yet: read {target} before overwriting it."
    )
    assert target.read_text() == "old content"


async def test_a_write_records_the_file_so_a_second_write_is_allowed(tmp_path, run):
    tool = make_tool(tmp_path)
    await run(tool, {"file_path": "f.txt", "content": "first"})

    result = await run(tool, {"file_path": "f.txt", "content": "second"})

    assert result.is_error is False
    assert (tmp_path / "f.txt").read_text() == "second"


# ── scenario 6: preserve BOM ──────────────────────────────────────────────────


async def test_existing_bom_is_retained_when_content_lacks_one(tmp_path, run):
    target = tmp_path / "f.txt"
    target.write_bytes(BOM + b"old")

    await run(make_tool(tmp_path, target), {"file_path": "f.txt", "content": "new"})

    assert target.read_bytes() == BOM + b"new"


async def test_content_with_a_bom_does_not_duplicate_it(tmp_path, run):
    target = tmp_path / "f.txt"
    target.write_bytes(BOM + b"old")

    await run(make_tool(tmp_path, target), {
        "file_path": "f.txt", "content": "﻿new",
    })

    assert target.read_bytes() == BOM + b"new"


async def test_content_with_a_bom_on_a_fresh_file_keeps_exactly_one(tmp_path, run):
    target = tmp_path / "f.txt"

    await run(make_tool(tmp_path), {"file_path": "f.txt", "content": "﻿new"})

    assert target.read_bytes() == BOM + b"new"


# ── scenario 7: empty content ─────────────────────────────────────────────────


async def test_empty_content_creates_or_truncates_to_an_empty_file(tmp_path, run):
    target = tmp_path / "f.txt"
    target.write_text("something")

    result = await run(make_tool(tmp_path, target), {
        "file_path": "f.txt", "content": "",
    })

    assert result.is_error is False
    assert target.read_bytes() == b""


# ── scenario 8: multiline and CRLF content ────────────────────────────────────


async def test_crlf_and_multiline_content_round_trips_exactly(tmp_path, run):
    content = "line1\r\nline2\r\nno final newline"

    await run(make_tool(tmp_path), {"file_path": "f.txt", "content": content})

    assert (tmp_path / "f.txt").read_bytes() == content.encode()


# ── scenario 9: NUL-containing string ─────────────────────────────────────────


async def test_nul_containing_content_is_not_truncated(tmp_path, run):
    await run(make_tool(tmp_path), {
        "file_path": "f.txt", "content": "Hello\x00World",
    })

    assert (tmp_path / "f.txt").read_bytes() == b"Hello\x00World"


# ── scenario 10: write failure ────────────────────────────────────────────────


async def test_an_unwritable_parent_is_an_error_not_a_success(tmp_path, run):
    (tmp_path / "blocker").write_text("i am a file")

    result = await run(make_tool(tmp_path), {
        "file_path": "blocker/child.txt", "content": "x",
    })

    assert result.is_error is True
    assert body(result).startswith("Failed to write file: ")
    assert "successfully" not in body(result)


# ── arguments ─────────────────────────────────────────────────────────────────


def test_args_require_content_and_file_path_and_forbid_extras():
    with pytest.raises(ValidationError):
        WriteTool.Args.model_validate({"file_path": "f.txt"})
    with pytest.raises(ValidationError):
        WriteTool.Args.model_validate({"content": "x", "file_path": ""})
    with pytest.raises(ValidationError):
        WriteTool.Args.model_validate({
            "content": "x", "file_path": "f.txt", "surprise": 1,
        })


# ── permission resource ───────────────────────────────────────────────────────


def test_permission_resource_exposes_the_resolved_path(tmp_path, perm):
    [access, request] = perm(make_tool(tmp_path), {
        "file_path": "nested/deep/file.txt", "content": "data",
    })

    path = tmp_path / "nested/deep/file.txt"
    assert access.resources == [
        ResourcePermission(
            permission="access_directory", resource=str(path.parent),
        ),
    ]
    assert access.metadata["preview"] == f"Access directory {path.parent}"
    assert [
        (o.resource_permissions, o.metadata["preview"])
        for o in access.answer_options
    ] == [
        (
            [
                ResourcePermission(
                    permission="access_directory", resource=str(path.parent),
                ),
                ResourcePermission(
                    permission="access_directory", resource=f"{path.parent}/*",
                ),
            ],
            f"Always allow access to {path.parent}",
        ),
    ]
    assert request.resources == [
        ResourcePermission(permission="write", resource=str(path)),
    ]
    assert request.metadata["preview"] == f"Write {path}"
    assert [
        (o.resource_permissions, o.metadata["preview"])
        for o in request.answer_options
    ] == [
        (
            [ResourcePermission(permission="write", resource=f"{path.parent}/*")],
            f"Write files under {path.parent}",
        ),
    ]
