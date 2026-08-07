"""The apply_patch executor: the three operation types plus move_to."""

import pytest

from luca.client.native import NativeToolError, execute_apply_patch

NOTES = "alpha\nbeta\ngamma\n"
UPDATE_DIFF = "\n".join(["@@ alpha", "-beta", "+BETA", " gamma"])


def test_create_file_writes_the_diffs_lines_and_its_parents(tmp_path):
    operation = {"type": "create_file", "path": "src/hello.txt", "diff": "+hello\n+world\n+"}
    assert execute_apply_patch(operation, root=tmp_path) == "Created src/hello.txt"
    assert (tmp_path / "src" / "hello.txt").read_text() == "hello\nworld\n"


def test_update_file_applies_the_diff_in_place(tmp_path):
    (tmp_path / "notes.txt").write_text(NOTES)
    operation = {"type": "update_file", "path": "notes.txt", "diff": UPDATE_DIFF}
    assert execute_apply_patch(operation, root=tmp_path) == "Updated notes.txt"
    assert (tmp_path / "notes.txt").read_text() == "alpha\nBETA\ngamma\n"


def test_update_file_keeps_the_files_crlf_line_endings_on_disk(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"line1\r\nline2\r\nline3\r\n")
    diff = "\n".join(["@@ line1", "-line2", "+updated", " line3"])
    operation = {"type": "update_file", "path": "notes.txt", "diff": diff}
    assert execute_apply_patch(operation, root=tmp_path) == "Updated notes.txt"
    assert (tmp_path / "notes.txt").read_bytes() == b"line1\r\nupdated\r\nline3\r\n"


def test_update_file_with_move_to_writes_the_new_path_and_drops_the_old(tmp_path):
    (tmp_path / "notes.txt").write_text(NOTES)
    operation = {
        "type": "update_file",
        "path": "notes.txt",
        "diff": UPDATE_DIFF,
        "move_to": "docs/notes.txt",
    }
    assert execute_apply_patch(operation, root=tmp_path) == ("Updated notes.txt\nMoved notes.txt to docs/notes.txt")
    assert (tmp_path / "docs" / "notes.txt").read_text() == "alpha\nBETA\ngamma\n"
    assert not (tmp_path / "notes.txt").exists()


def test_delete_file_removes_it(tmp_path):
    (tmp_path / "notes.txt").write_text(NOTES)
    operation = {"type": "delete_file", "path": "notes.txt", "diff": None}
    assert execute_apply_patch(operation, root=tmp_path) == "Deleted notes.txt"
    assert not (tmp_path / "notes.txt").exists()


def test_delete_file_that_does_not_exist_is_an_error(tmp_path):
    with pytest.raises(NativeToolError) as excinfo:
        execute_apply_patch({"type": "delete_file", "path": "nope.txt"}, root=tmp_path)
    assert str(excinfo.value) == "File not found: nope.txt"


def test_delete_file_pointed_at_a_directory_is_an_error(tmp_path):
    (tmp_path / "adir").mkdir()
    # The reason after the colon is the OS's own strerror and differs by
    # platform ("Is a directory" on Linux, "Operation not permitted" on macOS).
    with pytest.raises(NativeToolError, match=r"^Could not delete adir: .+\.$"):
        execute_apply_patch({"type": "delete_file", "path": "adir"}, root=tmp_path)
    assert (tmp_path / "adir").is_dir()


def test_create_file_over_an_existing_directory_is_an_error(tmp_path):
    (tmp_path / "adir").mkdir()
    operation = {"type": "create_file", "path": "adir", "diff": "+hello"}
    with pytest.raises(NativeToolError) as excinfo:
        execute_apply_patch(operation, root=tmp_path)
    assert str(excinfo.value) == "Could not write adir: Is a directory."


def test_update_file_that_does_not_exist_is_an_error(tmp_path):
    operation = {"type": "update_file", "path": "nope.txt", "diff": UPDATE_DIFF}
    with pytest.raises(NativeToolError) as excinfo:
        execute_apply_patch(operation, root=tmp_path)
    assert str(excinfo.value) == "File not found: nope.txt"


def test_update_file_whose_diff_does_not_apply_leaves_the_file_untouched(tmp_path):
    (tmp_path / "notes.txt").write_text(NOTES)
    operation = {
        "type": "update_file",
        "path": "notes.txt",
        "diff": "\n".join(["@@ alpha", "-delta", "+DELTA", " gamma"]),
    }
    with pytest.raises(NativeToolError, match="Invalid Context"):
        execute_apply_patch(operation, root=tmp_path)
    assert (tmp_path / "notes.txt").read_text() == NOTES


def test_update_file_without_a_diff_is_an_error(tmp_path):
    (tmp_path / "notes.txt").write_text(NOTES)
    with pytest.raises(NativeToolError) as excinfo:
        execute_apply_patch({"type": "update_file", "path": "notes.txt"}, root=tmp_path)
    assert str(excinfo.value) == "Missing diff for operation type update_file on path notes.txt"


def test_unknown_operation_type_is_an_error(tmp_path):
    with pytest.raises(NativeToolError) as excinfo:
        execute_apply_patch({"type": "rename_file", "path": "notes.txt"}, root=tmp_path)
    assert str(excinfo.value) == "Unknown operation type: rename_file"


def test_a_path_outside_the_root_is_refused(tmp_path):
    (tmp_path / "workspace").mkdir()
    (tmp_path / "outside.txt").write_text("secret\n")
    operation = {"type": "delete_file", "path": "../outside.txt"}
    with pytest.raises(NativeToolError, match="resolves outside the root directory"):
        execute_apply_patch(operation, root=tmp_path / "workspace")
    assert (tmp_path / "outside.txt").read_text() == "secret\n"
