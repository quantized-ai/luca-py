"""Unit tests for the shell `edit` tool: unique/ambiguous/replace-all
replacement, file creation, the read-first contract, BOM and line-ending
preservation, the OpenCode correction strategies, and its permission
resource."""

import asyncio

import pytest
from pydantic import ValidationError

from luca.agent.contrib.resource_permissions import ResourcePermission
from luca.agent.contrib.shell import EditTool, FileReadTracker

BOM = b"\xef\xbb\xbf"


def make_tool(tmp_path, *read_paths) -> EditTool:
    """An EditTool whose tracker has already 'read' the given paths."""
    tracker = FileReadTracker()
    for path in read_paths:
        tracker.record(path)
    return EditTool(workdir=tmp_path, tracker=tracker)


def body(result) -> str:
    return result.content[0].text


# ── scenario 1: simple unique replacement ─────────────────────────────────────


async def test_unique_replacement(tmp_path, run):
    target = tmp_path / "greeting.txt"
    target.write_text("Hello World\nGoodbye Moon\n")

    result = await run(make_tool(tmp_path, target), {
        "file_path": "greeting.txt", "old_string": "World", "new_string": "Turtle",
    })

    assert result.is_error is False
    assert body(result) == f"Edited file: {target}"
    assert target.read_text() == "Hello Turtle\nGoodbye Moon\n"
    assert result.metadata["replacements"] == 1
    assert result.metadata["created"] is False
    assert "-Hello World" in result.metadata["diff"]
    assert "+Hello Turtle" in result.metadata["diff"]


# ── scenario 2: multiple matches without replace-all ──────────────────────────


async def test_ambiguous_match_fails_and_leaves_the_file_unchanged(tmp_path, run):
    target = tmp_path / "f.txt"
    target.write_text("Hello World\nGoodbye World\n")

    result = await run(make_tool(tmp_path, target), {
        "file_path": "f.txt", "old_string": "World", "new_string": "Turtle",
    })

    assert result.is_error is True
    assert body(result) == (
        "Found multiple matches for old_string. Provide more surrounding"
        " context to make the match unique."
    )
    assert target.read_text() == "Hello World\nGoodbye World\n"


# ── scenario 3: replace all ───────────────────────────────────────────────────


async def test_replace_all_replaces_every_occurrence(tmp_path, run):
    target = tmp_path / "f.txt"
    target.write_text("Hello World\nGoodbye World\n")

    result = await run(make_tool(tmp_path, target), {
        "file_path": "f.txt",
        "old_string": "World",
        "new_string": "Turtle",
        "replace_all": True,
    })

    assert target.read_text() == "Hello Turtle\nGoodbye Turtle\n"
    assert result.metadata["replacements"] == 2


# ── scenario 4: create a new file ─────────────────────────────────────────────


async def test_empty_old_string_creates_the_file_and_parents(tmp_path, run):
    result = await run(make_tool(tmp_path), {
        "file_path": "sub/dir/new.txt", "old_string": "", "new_string": "content\n",
    })

    target = tmp_path / "sub/dir/new.txt"
    assert result.is_error is False
    assert body(result) == f"Created file: {target}"
    assert target.read_text() == "content\n"
    assert result.metadata["created"] is True
    assert "+content" in result.metadata["diff"]


# ── scenario 5: empty old string against an existing file ─────────────────────


async def test_empty_old_string_on_an_existing_file_fails(tmp_path, run):
    target = tmp_path / "f.txt"
    target.write_bytes(b"original bytes")

    result = await run(make_tool(tmp_path, target), {
        "file_path": "f.txt", "old_string": "", "new_string": "replacement",
    })

    assert result.is_error is True
    assert body(result) == (
        "old_string cannot be empty when editing an existing file. Provide the"
        " exact text to replace, or use write for an intentional full-file"
        " replacement."
    )
    assert target.read_bytes() == b"original bytes"


# ── scenario 6: missing file or directory target ──────────────────────────────


async def test_missing_file_reports_not_found(tmp_path, run):
    result = await run(make_tool(tmp_path), {
        "file_path": "missing.txt", "old_string": "a", "new_string": "b",
    })

    assert result.is_error is True
    assert body(result) == f"File not found: {tmp_path / 'missing.txt'}"


async def test_directory_target_reports_not_a_file(tmp_path, run):
    (tmp_path / "adir").mkdir()

    result = await run(make_tool(tmp_path), {
        "file_path": "adir", "old_string": "a", "new_string": "b",
    })

    assert result.is_error is True
    assert body(result) == f"Path is a directory, not a file: {tmp_path / 'adir'}"


# ── scenario 7: missing old string ────────────────────────────────────────────


async def test_unmatched_old_string_fails_and_leaves_the_file_unchanged(tmp_path, run):
    target = tmp_path / "f.txt"
    target.write_text("Hello World\n")

    result = await run(make_tool(tmp_path, target), {
        "file_path": "f.txt", "old_string": "Missing", "new_string": "Anything",
    })

    assert result.is_error is True
    assert body(result) == (
        "Could not find old_string in the file. It must match exactly,"
        " including whitespace, indentation, and line endings."
    )
    assert target.read_text() == "Hello World\n"


async def test_identical_old_and_new_strings_fail(tmp_path, run):
    target = tmp_path / "f.txt"
    target.write_text("Hello\n")

    result = await run(make_tool(tmp_path, target), {
        "file_path": "f.txt", "old_string": "Hello", "new_string": "Hello",
    })

    assert result.is_error is True
    assert body(result) == (
        "No changes to apply: old_string and new_string are identical."
    )


# ── scenario 8: multiline replacement ─────────────────────────────────────────


async def test_multiline_replacement(tmp_path, run):
    target = tmp_path / "f.txt"
    target.write_text("line1\nline2\nline3\n")

    await run(make_tool(tmp_path, target), {
        "file_path": "f.txt",
        "old_string": "line2",
        "new_string": "new line 2\nextra line",
    })

    assert target.read_text() == "line1\nnew line 2\nextra line\nline3\n"


# ── scenario 9: preserve line endings ─────────────────────────────────────────


async def test_lf_arguments_edit_a_crlf_file_and_crlf_survives(tmp_path, run):
    target = tmp_path / "f.txt"
    target.write_bytes(b"Hello World\r\nGoodbye World\r\n")

    await run(make_tool(tmp_path, target), {
        "file_path": "f.txt",
        "old_string": "Hello World\nGoodbye World",
        "new_string": "First\nSecond",
    })

    assert target.read_bytes() == b"First\r\nSecond\r\n"


async def test_crlf_arguments_edit_an_lf_file_and_lf_survives(tmp_path, run):
    target = tmp_path / "f.txt"
    target.write_bytes(b"Hello World\nGoodbye World\n")

    await run(make_tool(tmp_path, target), {
        "file_path": "f.txt",
        "old_string": "Hello World\r\nGoodbye World",
        "new_string": "First\r\nSecond",
    })

    assert target.read_bytes() == b"First\nSecond\n"


async def test_replace_all_over_a_crlf_file_stays_crlf(tmp_path, run):
    target = tmp_path / "f.txt"
    target.write_bytes(b"x World\r\ny World\r\n")

    await run(make_tool(tmp_path, target), {
        "file_path": "f.txt",
        "old_string": "World",
        "new_string": "Turtle",
        "replace_all": True,
    })

    assert target.read_bytes() == b"x Turtle\r\ny Turtle\r\n"


# ── scenario 10: preserve BOM ─────────────────────────────────────────────────


async def test_bom_is_preserved_and_absent_from_the_diff(tmp_path, run):
    target = tmp_path / "f.txt"
    target.write_bytes(BOM + b"Hello World\nSecond\n")

    result = await run(make_tool(tmp_path, target), {
        "file_path": "f.txt", "old_string": "Hello World", "new_string": "Hi",
    })

    assert target.read_bytes() == BOM + b"Hi\nSecond\n"
    assert "﻿" not in result.metadata["diff"]


# ── scenario 11: reject dangerous loose block match ───────────────────────────


async def test_loose_block_anchor_below_the_threshold_is_rejected(tmp_path, run):
    content = (
        "def start():\n"
        "    value = compute(1, 2)\n"
        "    return value * OFFSET\n"
        "def end():\n"
    )
    target = tmp_path / "f.py"
    target.write_text(content)

    result = await run(make_tool(tmp_path, target), {
        "file_path": "f.py",
        "old_string": (
            "def start():\n"
            "    completely unrelated body here\n"
            "    nothing in common with the file\n"
            "def end():"
        ),
        "new_string": "def start():\n    pass\ndef end():",
    })

    assert result.is_error is True
    assert "Could not find old_string" in body(result)
    assert target.read_text() == content


# ── scenario 12: concurrent edits ─────────────────────────────────────────────


async def test_concurrent_edits_to_the_same_file_are_serialized(tmp_path, run):
    target = tmp_path / "f.txt"
    target.write_text("alpha\nbeta\n")
    tool = make_tool(tmp_path, target)

    results = await asyncio.gather(
        run(tool, {"file_path": "f.txt", "old_string": "alpha", "new_string": "ALPHA"}),
        run(tool, {"file_path": "f.txt", "old_string": "beta", "new_string": "BETA"}),
    )

    assert all(result.is_error is False for result in results)
    assert target.read_text() == "ALPHA\nBETA\n"


# ── scenario 13: read-first requirement ───────────────────────────────────────


async def test_editing_an_unread_file_fails_without_changing_it(tmp_path, run):
    target = tmp_path / "f.txt"
    target.write_text("Hello World\n")

    result = await run(make_tool(tmp_path), {
        "file_path": "f.txt", "old_string": "World", "new_string": "Turtle",
    })

    assert result.is_error is True
    assert body(result) == (
        f"File has not been read yet: read {target} before editing it."
    )
    assert target.read_text() == "Hello World\n"


# ── correction strategies ─────────────────────────────────────────────────────


async def test_line_trimmed_match_corrects_wrong_indentation(tmp_path, run):
    target = tmp_path / "f.py"
    target.write_text("def f():\n    return 42\n")

    result = await run(make_tool(tmp_path, target), {
        "file_path": "f.py",
        "old_string": "        return 42",
        "new_string": "    return 43",
    })

    assert result.is_error is False
    assert target.read_text() == "def f():\n    return 43\n"


async def test_whitespace_normalized_match(tmp_path, run):
    target = tmp_path / "f.txt"
    target.write_text("a  b   c\n")

    result = await run(make_tool(tmp_path, target), {
        "file_path": "f.txt", "old_string": "a b c", "new_string": "d e f",
    })

    assert result.is_error is False
    assert target.read_text() == "d e f\n"


async def test_escape_normalized_match(tmp_path, run):
    target = tmp_path / "f.sh"
    target.write_text('say "hi"\n')

    result = await run(make_tool(tmp_path, target), {
        "file_path": "f.sh", "old_string": 'say \\"hi\\"', "new_string": 'say "bye"',
    })

    assert result.is_error is False
    assert target.read_text() == 'say "bye"\n'


async def test_block_anchor_match_with_similar_middle_is_accepted(tmp_path, run):
    target = tmp_path / "f.py"
    target.write_text("first\n    middle = 1\nlast\n")

    result = await run(make_tool(tmp_path, target), {
        "file_path": "f.py",
        "old_string": "first\n    middle = 2\nlast",
        "new_string": "first\n    middle = 3\nlast",
    })

    assert result.is_error is False
    assert target.read_text() == "first\n    middle = 3\nlast\n"


# ── arguments ─────────────────────────────────────────────────────────────────


def test_args_require_paths_and_forbid_extras():
    with pytest.raises(ValidationError):
        EditTool.Args.model_validate({"old_string": "a", "new_string": "b"})
    with pytest.raises(ValidationError):
        EditTool.Args.model_validate({
            "file_path": "f", "old_string": "a", "new_string": "b", "surprise": 1,
        })


# ── permission resource ───────────────────────────────────────────────────────


def test_permission_resource_exposes_the_resolved_path(tmp_path, perm):
    [access, request] = perm(make_tool(tmp_path), {
        "file_path": "greeting.txt", "old_string": "World", "new_string": "Turtle",
    })

    path = tmp_path / "greeting.txt"
    assert access.resources == [
        ResourcePermission(permission="access_directory", resource=str(tmp_path)),
    ]
    assert access.metadata["preview"] == f"Access directory {tmp_path}"
    assert [
        (o.resource_permissions, o.metadata["preview"])
        for o in access.answer_options
    ] == [
        (
            [
                ResourcePermission(
                    permission="access_directory", resource=str(tmp_path),
                ),
                ResourcePermission(
                    permission="access_directory", resource=f"{tmp_path}/*",
                ),
            ],
            f"Always allow access to {tmp_path}",
        ),
    ]
    assert request.resources == [
        ResourcePermission(permission="edit", resource=str(path)),
    ]
    assert request.metadata["preview"] == f"Edit {path}"
    assert [
        (o.resource_permissions, o.metadata["preview"])
        for o in request.answer_options
    ] == [
        (
            [ResourcePermission(permission="edit", resource=f"{tmp_path}/*")],
            f"Edit files under {tmp_path}",
        ),
    ]


def test_permission_resource_for_a_create(tmp_path, perm):
    [access, request] = perm(make_tool(tmp_path), {
        "file_path": "sub/new.txt", "old_string": "", "new_string": "content",
    })

    assert access.resources == [
        ResourcePermission(
            permission="access_directory", resource=str(tmp_path / "sub"),
        ),
    ]
    assert request.resources == [
        ResourcePermission(permission="edit", resource=str(tmp_path / "sub/new.txt")),
    ]
    assert request.answer_options[0].resource_permissions == [
        ResourcePermission(permission="edit", resource=f"{tmp_path / 'sub'}/*"),
    ]
