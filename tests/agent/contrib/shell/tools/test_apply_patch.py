"""Unit tests for the shell `apply_patch` tool: envelope parsing, the four
line-matching passes, verify-before-commit semantics, moves, BOM handling,
the heredoc wrapper, and its permission resource."""

import pytest
from pydantic import ValidationError

from luca.agent.contrib.resource_permissions import ResourcePermission
from luca.agent.contrib.shell import ApplyPatchTool

BOM = b"\xef\xbb\xbf"


def make_tool(tmp_path) -> ApplyPatchTool:
    return ApplyPatchTool(workdir=tmp_path)


def body(result) -> str:
    return result.content[0].text


def patch(*lines: str) -> str:
    return "\n".join(("*** Begin Patch", *lines, "*** End Patch"))


# ── scenario 1: missing or invalid patch ──────────────────────────────────────


async def test_whitespace_only_patch_text_is_rejected(tmp_path, run):
    result = await run(make_tool(tmp_path), {"patch_text": "   \n  "})

    assert result.is_error is True
    assert body(result) == "patch_text is required"


def test_empty_patch_text_fails_args_validation():
    with pytest.raises(ValidationError):
        ApplyPatchTool.Args.model_validate({"patch_text": ""})


async def test_missing_envelope_markers_fail_verification(tmp_path, run):
    result = await run(make_tool(tmp_path), {
        "patch_text": "*** Add File: a.txt\n+hello",
    })

    assert result.is_error is True
    assert body(result) == (
        "Invalid patch: missing '*** Begin Patch' / '*** End Patch' markers"
    )


async def test_an_envelope_with_no_operations_fails(tmp_path, run):
    result = await run(make_tool(tmp_path), {"patch_text": patch()})

    assert result.is_error is True
    assert body(result) == "Invalid patch: the envelope contains no operations"


# ── scenario 2: add, update, and delete together ──────────────────────────────


async def test_add_update_and_delete_in_one_patch(tmp_path, run):
    (tmp_path / "modify.txt").write_text("keep\nchange me\n")
    (tmp_path / "delete.txt").write_text("bye\n")

    result = await run(make_tool(tmp_path), {"patch_text": patch(
        "*** Add File: nested/new.txt",
        "+created line",
        "*** Update File: modify.txt",
        "@@",
        " keep",
        "-change me",
        "+changed",
        "*** Delete File: delete.txt",
    )})

    assert result.is_error is False
    assert body(result) == (
        "Success. Updated the following files:\n"
        "A nested/new.txt\n"
        "M modify.txt\n"
        "D delete.txt"
    )
    assert (tmp_path / "nested/new.txt").read_text() == "created line\n"
    assert (tmp_path / "modify.txt").read_text() == "keep\nchanged\n"
    assert not (tmp_path / "delete.txt").exists()
    assert [(f["path"], f["type"]) for f in result.metadata["files"]] == [
        ("nested/new.txt", "add"),
        ("modify.txt", "update"),
        ("delete.txt", "delete"),
    ]
    add_meta = result.metadata["files"][0]
    assert add_meta["additions"] == 1
    assert add_meta["deletions"] == 0
    assert add_meta["move_to"] is None
    assert "+created line" in add_meta["patch"]


# ── scenario 3: multiple update chunks ────────────────────────────────────────


async def test_two_chunks_apply_without_losing_intervening_content(tmp_path, run):
    (tmp_path / "f.txt").write_text("l1\nl2\nl3\nl4\n")

    result = await run(make_tool(tmp_path), {"patch_text": patch(
        "*** Update File: f.txt",
        "@@",
        " l1",
        "-l2",
        "+L2",
        "@@",
        " l3",
        "-l4",
        "+L4",
    )})

    assert result.is_error is False
    assert (tmp_path / "f.txt").read_text() == "l1\nL2\nl3\nL4\n"


# ── scenario 4: insert-only change ────────────────────────────────────────────


async def test_insert_only_chunk(tmp_path, run):
    (tmp_path / "f.txt").write_text("alpha\nomega\n")

    await run(make_tool(tmp_path), {"patch_text": patch(
        "*** Update File: f.txt",
        "@@",
        " alpha",
        "+beta",
        " omega",
    )})

    assert (tmp_path / "f.txt").read_text() == "alpha\nbeta\nomega\n"


# ── scenario 5: add final newline ─────────────────────────────────────────────


async def test_updated_file_gains_a_final_newline(tmp_path, run):
    (tmp_path / "f.txt").write_bytes(b"a\nb")

    await run(make_tool(tmp_path), {"patch_text": patch(
        "*** Update File: f.txt",
        "@@",
        " a",
        "-b",
        "+B",
    )})

    assert (tmp_path / "f.txt").read_bytes() == b"a\nB\n"


# ── scenario 6: move and update ───────────────────────────────────────────────


async def test_move_writes_the_destination_and_removes_the_source(tmp_path, run):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/app.txt").write_text("print hi\n")

    result = await run(make_tool(tmp_path), {"patch_text": patch(
        "*** Update File: src/app.txt",
        "*** Move to: renamed/dir/name.txt",
        "@@",
        "-print hi",
        "+print hello",
    )})

    assert result.is_error is False
    assert body(result) == (
        "Success. Updated the following files:\nM renamed/dir/name.txt"
    )
    assert (tmp_path / "renamed/dir/name.txt").read_text() == "print hello\n"
    assert not (tmp_path / "src/app.txt").exists()
    assert result.metadata["files"][0]["move_to"] == "renamed/dir/name.txt"
    assert result.metadata["files"][0]["path"] == "src/app.txt"


# ── scenario 7: existing destination ──────────────────────────────────────────


async def test_move_overwrites_an_existing_destination(tmp_path, run):
    (tmp_path / "a.txt").write_text("value\n")
    (tmp_path / "b.txt").write_text("old destination\n")

    await run(make_tool(tmp_path), {"patch_text": patch(
        "*** Update File: a.txt",
        "*** Move to: b.txt",
        "@@",
        "-value",
        "+VALUE",
    )})

    assert (tmp_path / "b.txt").read_text() == "VALUE\n"
    assert not (tmp_path / "a.txt").exists()


async def test_add_overwrites_an_existing_target(tmp_path, run):
    (tmp_path / "a.txt").write_text("old\n")

    result = await run(make_tool(tmp_path), {"patch_text": patch(
        "*** Add File: a.txt",
        "+new",
    )})

    assert result.is_error is False
    assert (tmp_path / "a.txt").read_text() == "new\n"


# ── scenario 8: missing update or delete target ───────────────────────────────


async def test_missing_update_target_fails_verification(tmp_path, run):
    result = await run(make_tool(tmp_path), {"patch_text": patch(
        "*** Update File: missing.txt",
        "@@",
        "-a",
        "+b",
    )})

    assert result.is_error is True
    assert body(result) == f"Update target not found: {tmp_path / 'missing.txt'}"


async def test_missing_delete_target_fails_verification(tmp_path, run):
    result = await run(make_tool(tmp_path), {"patch_text": patch(
        "*** Delete File: missing.txt",
    )})

    assert result.is_error is True
    assert body(result) == f"Delete target not found: {tmp_path / 'missing.txt'}"


# ── scenario 9: directory target ──────────────────────────────────────────────


async def test_update_or_delete_against_a_directory_fails(tmp_path, run):
    (tmp_path / "adir").mkdir()
    (tmp_path / "adir/inner.txt").write_text("safe\n")
    tool = make_tool(tmp_path)

    update = await run(tool, {"patch_text": patch(
        "*** Update File: adir", "@@", "-a", "+b",
    )})
    delete = await run(tool, {"patch_text": patch("*** Delete File: adir")})

    assert update.is_error is True
    assert update.content[0].text == (
        f"Update target is a directory, not a file: {tmp_path / 'adir'}"
    )
    assert delete.is_error is True
    assert (tmp_path / "adir/inner.txt").read_text() == "safe\n"


# ── scenario 10: verification is all-or-nothing ───────────────────────────────


async def test_a_failing_later_update_prevents_an_earlier_add(tmp_path, run):
    (tmp_path / "exists.txt").write_text("real content\n")

    result = await run(make_tool(tmp_path), {"patch_text": patch(
        "*** Add File: created.txt",
        "+should never exist",
        "*** Update File: exists.txt",
        "@@",
        "-line that is not there",
        "+replacement",
    )})

    assert result.is_error is True
    assert not (tmp_path / "created.txt").exists()
    assert (tmp_path / "exists.txt").read_text() == "real content\n"


# ── scenario 11: context header ───────────────────────────────────────────────


async def test_context_header_disambiguates_duplicate_lines(tmp_path, run):
    (tmp_path / "f.txt").write_text("fn a\nx=10\nfn b\nx=10\n")

    await run(make_tool(tmp_path), {"patch_text": patch(
        "*** Update File: f.txt",
        "@@ fn b",
        "-x=10",
        "+x=20",
    )})

    assert (tmp_path / "f.txt").read_text() == "fn a\nx=10\nfn b\nx=20\n"


# ── scenario 12: end-of-file anchor ───────────────────────────────────────────


async def test_end_of_file_marker_selects_the_tail_occurrence(tmp_path, run):
    (tmp_path / "f.txt").write_text("dup\nother\ndup\n")

    await run(make_tool(tmp_path), {"patch_text": patch(
        "*** Update File: f.txt",
        "@@",
        "-dup",
        "+DUP",
        "*** End of File",
    )})

    assert (tmp_path / "f.txt").read_text() == "dup\nother\nDUP\n"


# ── scenario 13: whitespace matching ──────────────────────────────────────────


async def test_patch_without_trailing_spaces_matches_lines_that_have_them(tmp_path, run):
    (tmp_path / "f.txt").write_text("keep me  \nchange me\t\n")

    await run(make_tool(tmp_path), {"patch_text": patch(
        "*** Update File: f.txt",
        "@@",
        " keep me",
        "-change me",
        "+changed",
    )})

    assert (tmp_path / "f.txt").read_text() == "keep me  \nchanged\n"


async def test_unindented_expected_text_matches_indented_file_text(tmp_path, run):
    (tmp_path / "f.txt").write_text("    indented = 1\n")

    await run(make_tool(tmp_path), {"patch_text": patch(
        "*** Update File: f.txt",
        "@@",
        "-indented = 1",
        "+indented = 2",
    )})

    assert (tmp_path / "f.txt").read_text() == "indented = 2\n"


# ── scenario 14: unicode punctuation matching ─────────────────────────────────


async def test_ascii_punctuation_matches_unicode_punctuation(tmp_path, run):
    (tmp_path / "f.txt").write_text("it’s “quoted” — dashed\n")

    await run(make_tool(tmp_path), {"patch_text": patch(
        "*** Update File: f.txt",
        "@@",
        "-it's \"quoted\" - dashed",
        "+plain replacement",
    )})

    assert (tmp_path / "f.txt").read_text() == "plain replacement\n"


# ── scenario 15: preserve BOM ─────────────────────────────────────────────────


async def test_bom_is_retained_and_absent_from_the_diff(tmp_path, run):
    (tmp_path / "f.txt").write_bytes(BOM + b"first\nsecond\n")

    result = await run(make_tool(tmp_path), {"patch_text": patch(
        "*** Update File: f.txt",
        "@@",
        " first",
        "-second",
        "+SECOND",
    )})

    assert (tmp_path / "f.txt").read_bytes() == BOM + b"first\nSECOND\n"
    diff = result.metadata["files"][0]["patch"]
    assert "﻿" not in diff
    assert "-first" not in diff


# ── scenario 16: heredoc wrapper ──────────────────────────────────────────────


async def test_cat_heredoc_wrapper_is_parsed(tmp_path, run):
    text = "cat <<'EOF'\n" + patch("*** Add File: a.txt", "+hello") + "\nEOF"

    result = await run(make_tool(tmp_path), {"patch_text": text})

    assert result.is_error is False
    assert (tmp_path / "a.txt").read_text() == "hello\n"


async def test_bare_heredoc_wrapper_is_parsed(tmp_path, run):
    text = "<<EOF\n" + patch("*** Add File: b.txt", "+hi") + "\nEOF"

    result = await run(make_tool(tmp_path), {"patch_text": text})

    assert result.is_error is False
    assert (tmp_path / "b.txt").read_text() == "hi\n"


# ── permission resource ───────────────────────────────────────────────────────


def test_permission_resource_lists_all_touched_paths(tmp_path, perm):
    [access, request] = perm(make_tool(tmp_path), {"patch_text": patch(
        "*** Add File: hello.txt",
        "+Hello world",
        "*** Update File: src/app.py",
        "*** Move to: src/main.py",
        "@@",
        "-a",
        "+b",
        "*** Delete File: obsolete.txt",
    )})

    assert access.resources == [
        ResourcePermission(permission="access_directory", resource=str(tmp_path)),
        ResourcePermission(
            permission="access_directory", resource=str(tmp_path / "src"),
        ),
    ]
    assert access.metadata["preview"] == (
        f"Access directories {tmp_path}, {tmp_path / 'src'}"
    )
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
        (
            [
                ResourcePermission(
                    permission="access_directory", resource=str(tmp_path / "src"),
                ),
                ResourcePermission(
                    permission="access_directory", resource=f"{tmp_path / 'src'}/*",
                ),
            ],
            f"Always allow access to {tmp_path / 'src'}",
        ),
    ]
    assert request.resources == [
        ResourcePermission(
            permission="apply_patch", resource=str(tmp_path / "hello.txt"),
        ),
        ResourcePermission(
            permission="apply_patch", resource=str(tmp_path / "src/app.py"),
        ),
        ResourcePermission(
            permission="apply_patch", resource=str(tmp_path / "src/main.py"),
        ),
        ResourcePermission(
            permission="apply_patch", resource=str(tmp_path / "obsolete.txt"),
        ),
    ]
    assert request.metadata["preview"] == (
        "Apply patch to hello.txt, src/app.py, src/main.py, obsolete.txt"
    )
    assert request.answer_options == []


def test_permission_resource_for_unparseable_patch_text(tmp_path, perm):
    [request] = perm(make_tool(tmp_path), {"patch_text": "not a patch"})

    assert request.resources == []
    assert request.metadata["preview"] == "Apply patch (invalid patch text)"
