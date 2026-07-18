"""Unit tests for the shell `read` tool: text paging, directory listings,
binary/media classification, and its permission resource."""

import pytest
from pydantic import ValidationError

from luca.agent.contrib.resource_permissions import ResourcePermission
from luca.agent.contrib.shell import FileReadTracker, ReadTool


def make_tool(tmp_path, tracker=None) -> ReadTool:
    return ReadTool(workdir=tmp_path, tracker=tracker)


def body(result) -> str:
    return result.content[0].text


# ── scenario 1: small text file ───────────────────────────────────────────────


async def test_small_text_file_returns_numbered_lines_and_total(tmp_path, run):
    (tmp_path / "notes.txt").write_text("alpha\nbeta\n")

    result = await run(make_tool(tmp_path), {"file_path": "notes.txt"})

    assert result.is_error is False
    assert f"<path>{tmp_path / 'notes.txt'}</path>" in body(result)
    assert "<type>file</type>" in body(result)
    assert "1: alpha\n2: beta" in body(result)
    assert "(End of file - total 2 lines)" in body(result)
    assert result.metadata == {"truncated": False}


async def test_absolute_file_path_is_used_directly(tmp_path, run):
    (tmp_path / "notes.txt").write_text("alpha\n")

    result = await run(make_tool("/somewhere/else"), {
        "file_path": str(tmp_path / "notes.txt"),
    })

    assert "1: alpha" in body(result)


async def test_successful_read_records_the_file_on_the_tracker(tmp_path, run):
    (tmp_path / "notes.txt").write_text("alpha\n")
    tracker = FileReadTracker()

    await run(make_tool(tmp_path, tracker), {"file_path": "notes.txt"})

    assert tracker.was_read(tmp_path / "notes.txt")


# ── scenario 2: offset and limit ──────────────────────────────────────────────


async def test_offset_and_limit_return_only_the_requested_page(tmp_path, run):
    (tmp_path / "lines.txt").write_text(
        "".join(f"line{n}\n" for n in range(1, 21)),
    )

    result = await run(make_tool(tmp_path), {
        "file_path": "lines.txt", "offset": 10, "limit": 5,
    })

    assert "10: line10" in body(result)
    assert "14: line14" in body(result)
    assert "9: line9" not in body(result)
    assert "15: line15" not in body(result)
    assert "(Showing lines 10-14 of 20. Use offset=15 to continue.)" in body(result)
    assert result.metadata == {"truncated": True}


# ── scenario 3: offset outside the file ───────────────────────────────────────


async def test_offset_past_the_end_is_an_error(tmp_path, run):
    (tmp_path / "three.txt").write_text("a\nb\nc\n")

    result = await run(make_tool(tmp_path), {"file_path": "three.txt", "offset": 4})

    assert result.is_error is True
    assert body(result) == "Offset 4 is out of range for this file (3 lines)"


async def test_empty_file_is_valid_at_offset_1_and_invalid_at_offset_2(tmp_path, run):
    (tmp_path / "empty.txt").write_text("")
    tool = make_tool(tmp_path)

    ok = await run(tool, {"file_path": "empty.txt"})
    bad = await run(tool, {"file_path": "empty.txt", "offset": 2})

    assert ok.is_error is False
    assert "(End of file - total 0 lines)" in body(ok)
    assert bad.is_error is True
    assert body(bad) == "Offset 2 is out of range for this file (0 lines)"


# ── scenario 4: line-count truncation ─────────────────────────────────────────


async def test_limit_truncates_and_points_at_the_next_offset(tmp_path, run):
    (tmp_path / "big.txt").write_text(
        "".join(f"row{n}\n" for n in range(1, 101)),
    )

    result = await run(make_tool(tmp_path), {"file_path": "big.txt", "limit": 10})

    assert "1: row1" in body(result)
    assert "10: row10" in body(result)
    assert "11: row11" not in body(result)
    assert "(Showing lines 1-10 of 100. Use offset=11 to continue.)" in body(result)
    assert result.metadata == {"truncated": True}


# ── scenario 5: byte truncation ───────────────────────────────────────────────


async def test_50kib_output_cap_stops_the_page_early(tmp_path, run):
    (tmp_path / "wide.txt").write_text(
        "".join(f"{'x' * 100}\n" for _ in range(1_000)),
    )

    result = await run(make_tool(tmp_path), {"file_path": "wide.txt"})

    assert result.is_error is False
    assert result.metadata == {"truncated": True}
    assert "(Output capped at 50 KB. Showing lines 1-" in body(result)
    assert "Use offset=" in body(result)
    assert "1000: " not in body(result)
    assert len(body(result).encode()) < 60 * 1024


# ── scenario 6: long line ─────────────────────────────────────────────────────


async def test_long_line_is_cut_at_2000_chars_with_a_marker(tmp_path, run):
    (tmp_path / "long.txt").write_text("x" * 3_000 + "\n")

    result = await run(make_tool(tmp_path), {"file_path": "long.txt"})

    assert f"1: {'x' * 2_000}... (line truncated to 2000 chars)" in body(result)
    assert "x" * 2_001 not in body(result)


# ── scenario 7: directory listing ─────────────────────────────────────────────


async def test_directory_listing_is_sorted_with_trailing_slash(tmp_path, run):
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "src").mkdir()

    result = await run(make_tool(tmp_path), {"file_path": str(tmp_path)})

    assert f"<path>{tmp_path}</path>" in body(result)
    assert "<type>directory</type>" in body(result)
    assert "<entries>\na.txt\nb.txt\nsrc/\n</entries>" in body(result)
    assert result.metadata == {"truncated": False}


async def test_directory_paging_marks_intermediate_pages_only(tmp_path, run):
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "src").mkdir()
    tool = make_tool(tmp_path)

    middle = await run(tool, {"file_path": ".", "offset": 2, "limit": 1})
    final = await run(tool, {"file_path": ".", "offset": 3, "limit": 1})

    assert "b.txt" in body(middle)
    assert "(Showing entries 2-2 of 3. Use offset=3 to continue.)" in body(middle)
    assert middle.metadata == {"truncated": True}
    assert "src/" in body(final)
    assert "Use offset=" not in body(final)
    assert final.metadata == {"truncated": False}


# ── scenario 8: missing path ──────────────────────────────────────────────────


async def test_missing_path_suggests_similar_siblings(tmp_path, run):
    (tmp_path / "config.yaml").write_text("")

    result = await run(make_tool(tmp_path), {"file_path": "config.yml"})

    assert result.is_error is True
    assert f"File not found: {tmp_path / 'config.yml'}" in body(result)
    assert "Did you mean one of these?" in body(result)
    assert str(tmp_path / "config.yaml") in body(result)


async def test_missing_path_without_similar_siblings_is_a_plain_error(tmp_path, run):
    result = await run(make_tool(tmp_path), {"file_path": "zzz.txt"})

    assert result.is_error is True
    assert body(result) == f"File not found: {tmp_path / 'zzz.txt'}"


# ── scenario 9: binary detection ──────────────────────────────────────────────


async def test_known_binary_extension_is_rejected_even_with_text_content(tmp_path, run):
    (tmp_path / "mod.wasm").write_text("this looks like text")

    result = await run(make_tool(tmp_path), {"file_path": "mod.wasm"})

    assert result.is_error is True
    assert body(result) == f"Cannot read binary file: {tmp_path / 'mod.wasm'}"


async def test_text_extension_with_nul_byte_is_rejected(tmp_path, run):
    (tmp_path / "data.txt").write_bytes(b"abc\x00def")

    result = await run(make_tool(tmp_path), {"file_path": "data.txt"})

    assert result.is_error is True
    assert body(result) == f"Cannot read binary file: {tmp_path / 'data.txt'}"


# ── scenario 10: media attachments ────────────────────────────────────────────


async def test_png_is_returned_as_an_image_attachment(tmp_path, run):
    (tmp_path / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n....")

    result = await run(make_tool(tmp_path), {"file_path": "pic.png"})

    assert result.is_error is False
    assert body(result) == "Image read successfully"
    assert result.metadata == {
        "attachment": {"path": str(tmp_path / "pic.png"), "mime_type": "image/png"},
    }


async def test_pdf_is_returned_as_a_pdf_attachment(tmp_path, run):
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.7 ....")

    result = await run(make_tool(tmp_path), {"file_path": "doc.pdf"})

    assert body(result) == "PDF read successfully"
    assert result.metadata["attachment"]["mime_type"] == "application/pdf"


# ── arguments ─────────────────────────────────────────────────────────────────


def test_args_reject_zero_offset_zero_limit_and_extras():
    with pytest.raises(ValidationError):
        ReadTool.Args.model_validate({"file_path": "a.txt", "offset": 0})
    with pytest.raises(ValidationError):
        ReadTool.Args.model_validate({"file_path": "a.txt", "limit": 0})
    with pytest.raises(ValidationError):
        ReadTool.Args.model_validate({"file_path": "a.txt", "surprise": 1})


# ── permission resource ───────────────────────────────────────────────────────


def test_permission_resource_exposes_the_resolved_path(tmp_path, perm):
    [access, request] = perm(make_tool(tmp_path), {"file_path": "notes.txt"})

    path = tmp_path / "notes.txt"
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
        ResourcePermission(permission="read", resource=str(path)),
    ]
    assert request.metadata["preview"] == f"Read {path}"
    assert [
        (o.resource_permissions, o.metadata["preview"])
        for o in request.answer_options
    ] == [
        (
            [ResourcePermission(permission="read", resource=f"{tmp_path}/*")],
            f"Read files under {tmp_path}",
        ),
    ]


def test_permission_resource_for_a_directory_target(tmp_path, perm):
    (tmp_path / "src").mkdir()

    [access, request] = perm(make_tool(tmp_path), {"file_path": str(tmp_path / "src")})

    assert access.resources == [
        ResourcePermission(
            permission="access_directory", resource=str(tmp_path / "src"),
        ),
    ]
    assert request.resources == [
        ResourcePermission(permission="read", resource=str(tmp_path / "src")),
    ]
    assert request.answer_options[0].resource_permissions == [
        ResourcePermission(permission="read", resource=f"{tmp_path}/*"),
    ]
