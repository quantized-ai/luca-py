"""The pure text-editor operations and the filesystem executor."""

import pytest

from luca.client.native import NativeToolError, execute_text_editor
from luca.client.native.text_editor import insert, str_replace, view

NOTES = "alpha\nbeta\ngamma\n"


def test_view_numbers_every_line_cat_n_style():
    assert view(NOTES) == "     1\talpha\n     2\tbeta\n     3\tgamma"


def test_view_range_clips_to_the_requested_lines():
    assert view(NOTES, [2, 2]) == "     2\tbeta"


def test_view_range_ending_in_minus_one_runs_to_the_last_line():
    assert view(NOTES, [2, -1]) == "     2\tbeta\n     3\tgamma"


def test_view_of_empty_text_is_empty():
    assert view("") == ""


def test_view_range_on_empty_text_is_out_of_range():
    with pytest.raises(NativeToolError) as excinfo:
        view("", [1, -1])
    assert str(excinfo.value) == "view_range start 1 is out of range; the file has 0 lines."


def test_view_counts_a_form_feed_as_content_not_a_line_break():
    assert view("a\x0cb\n") == "     1\ta\x0cb"


def test_view_range_past_the_end_is_an_error():
    with pytest.raises(NativeToolError) as excinfo:
        view(NOTES, [2, 9])
    assert str(excinfo.value) == "view_range end 9 is out of range; expected 2..3 or -1."


def test_view_range_starting_before_the_first_line_is_an_error():
    with pytest.raises(NativeToolError) as excinfo:
        view(NOTES, [0, 2])
    assert str(excinfo.value) == "view_range start 0 is out of range; the file has 3 lines."


def test_view_range_with_the_wrong_arity_is_an_error():
    with pytest.raises(NativeToolError) as excinfo:
        view(NOTES, [1])
    assert str(excinfo.value) == "view_range must be [start, end]; got [1]."


def test_view_range_that_is_not_a_pair_at_all_is_an_error():
    with pytest.raises(NativeToolError) as excinfo:
        view(NOTES, 2)
    assert str(excinfo.value) == "view_range must be [start, end]; got 2."


def test_view_range_of_non_integers_is_an_error():
    with pytest.raises(NativeToolError) as excinfo:
        view(NOTES, ["1", "2"])
    assert str(excinfo.value) == "view_range start must be an integer; got '1'."


def test_view_range_of_booleans_is_an_error():
    with pytest.raises(NativeToolError) as excinfo:
        view(NOTES, [True, 2])
    assert str(excinfo.value) == "view_range start must be an integer; got True."


def test_str_replace_replaces_the_single_match():
    assert str_replace(NOTES, "beta", "BETA") == "alpha\nBETA\ngamma\n"


def test_str_replace_without_a_match_is_an_error():
    with pytest.raises(NativeToolError) as excinfo:
        str_replace(NOTES, "delta", "DELTA")
    assert str(excinfo.value) == (
        "No match found for old_str. It must match the file exactly, whitespace and indentation included."
    )


def test_str_replace_with_several_matches_is_an_error():
    with pytest.raises(NativeToolError) as excinfo:
        str_replace("x\ny\nx\n", "x", "z")
    assert str(excinfo.value) == (
        "Found 2 matches for old_str. Add surrounding context so it matches exactly one location."
    )


def test_insert_at_line_zero_goes_to_the_top():
    assert insert(NOTES, 0, "first") == "first\nalpha\nbeta\ngamma\n"


def test_insert_at_the_line_count_appends():
    assert insert(NOTES, 3, "delta") == "alpha\nbeta\ngamma\ndelta\n"


def test_insert_into_empty_text_gains_a_trailing_newline():
    assert insert("", 0, "first") == "first\n"


def test_insert_past_the_line_count_is_an_error():
    with pytest.raises(NativeToolError) as excinfo:
        insert(NOTES, 4, "delta")
    assert str(excinfo.value) == "insert_line 4 is out of range; expected 0..3."


def test_insert_line_that_is_not_an_integer_is_an_error():
    with pytest.raises(NativeToolError) as excinfo:
        insert(NOTES, "1", "delta")
    assert str(excinfo.value) == "insert_line must be an integer; got '1'."


def test_insert_keeps_a_form_feed_as_content():
    assert insert("a\x0cb\nc\n", 0, "top") == "top\na\x0cb\nc\n"


def test_execute_view_renders_the_file(tmp_path):
    (tmp_path / "notes.txt").write_text(NOTES)
    command = {"command": "view", "path": "notes.txt", "view_range": [1, 2]}
    assert execute_text_editor(command, root=tmp_path) == "     1\talpha\n     2\tbeta"


def test_execute_view_lists_a_directory_two_levels_deep_without_hidden_entries(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / ".hidden").write_text("h")
    (tmp_path / "sub" / "deep").mkdir(parents=True)
    (tmp_path / "sub" / "b.txt").write_text("b")
    (tmp_path / "sub" / "deep" / "c.txt").write_text("c")
    assert execute_text_editor({"command": "view", "path": "."}, root=tmp_path) == ("a.txt\nsub/\nsub/b.txt\nsub/deep/")


def test_execute_view_names_a_symlink_without_following_it(tmp_path):
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "a.txt").write_text("a")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.txt").write_text("secret\n")
    (tmp_path / "workspace" / "linkdir").symlink_to(tmp_path / "outside")
    command = {"command": "view", "path": "."}
    assert execute_text_editor(command, root=tmp_path / "workspace") == "a.txt\nlinkdir@"


def test_execute_view_of_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(NativeToolError) as excinfo:
        execute_text_editor({"command": "view", "path": "nope.txt"}, root=tmp_path)
    assert str(excinfo.value) == "File not found: nope.txt"


def test_execute_create_writes_the_file_and_its_parents(tmp_path):
    command = {"command": "create", "path": "src/new.txt", "file_text": "hello\n"}
    assert execute_text_editor(command, root=tmp_path) == "Created src/new.txt"
    assert (tmp_path / "src" / "new.txt").read_text() == "hello\n"


def test_execute_create_overwrites_an_existing_file(tmp_path):
    (tmp_path / "notes.txt").write_text(NOTES)
    command = {"command": "create", "path": "notes.txt", "file_text": "replaced\n"}
    assert execute_text_editor(command, root=tmp_path) == "Created notes.txt"
    assert (tmp_path / "notes.txt").read_text() == "replaced\n"


def test_execute_create_under_a_parent_that_is_a_file_is_an_error(tmp_path):
    (tmp_path / "notes.txt").write_text(NOTES)
    command = {"command": "create", "path": "notes.txt/nested.txt", "file_text": "x"}
    with pytest.raises(NativeToolError) as excinfo:
        execute_text_editor(command, root=tmp_path)
    assert str(excinfo.value) == "Could not write notes.txt/nested.txt: File exists."


def test_execute_str_replace_rewrites_the_file(tmp_path):
    (tmp_path / "notes.txt").write_text(NOTES)
    command = {
        "command": "str_replace",
        "path": "notes.txt",
        "old_str": "beta",
        "new_str": "BETA",
    }
    assert execute_text_editor(command, root=tmp_path) == "Updated notes.txt"
    assert (tmp_path / "notes.txt").read_text() == "alpha\nBETA\ngamma\n"


def test_execute_str_replace_leaves_the_file_untouched_when_it_fails(tmp_path):
    (tmp_path / "notes.txt").write_text(NOTES)
    command = {
        "command": "str_replace",
        "path": "notes.txt",
        "old_str": "delta",
        "new_str": "DELTA",
    }
    with pytest.raises(NativeToolError, match="No match found for old_str"):
        execute_text_editor(command, root=tmp_path)
    assert (tmp_path / "notes.txt").read_text() == NOTES


def test_execute_insert_rewrites_the_file(tmp_path):
    (tmp_path / "notes.txt").write_text(NOTES)
    command = {
        "command": "insert",
        "path": "notes.txt",
        "insert_line": 3,
        "insert_text": "delta",
    }
    assert execute_text_editor(command, root=tmp_path) == "Updated notes.txt"
    assert (tmp_path / "notes.txt").read_text() == "alpha\nbeta\ngamma\ndelta\n"


def test_execute_insert_keeps_a_form_feed_on_disk(tmp_path):
    (tmp_path / "ff.py").write_bytes(b"def a():\n    pass\n\x0cdef b():\n    pass\n")
    command = {"command": "insert", "path": "ff.py", "insert_line": 0, "insert_text": "# top"}
    assert execute_text_editor(command, root=tmp_path) == "Updated ff.py"
    assert (tmp_path / "ff.py").read_bytes() == b"# top\ndef a():\n    pass\n\x0cdef b():\n    pass\n"


def test_execute_view_shows_crlf_line_endings_as_the_content_they_are(tmp_path):
    (tmp_path / "crlf.txt").write_bytes(b"line1\r\nline2\r\n")
    command = {"command": "view", "path": "crlf.txt"}
    assert execute_text_editor(command, root=tmp_path) == "     1\tline1\r\n     2\tline2\r"


def test_execute_insert_leaves_the_untouched_crlf_lines_byte_identical(tmp_path):
    # The inserted line itself carries LF: `insert` joins with "\n" and does no
    # newline detection, so a CRLF file gains one mixed line and loses nothing.
    (tmp_path / "crlf.txt").write_bytes(b"line1\r\nline2\r\n")
    command = {"command": "insert", "path": "crlf.txt", "insert_line": 0, "insert_text": "first"}
    assert execute_text_editor(command, root=tmp_path) == "Updated crlf.txt"
    assert (tmp_path / "crlf.txt").read_bytes() == b"first\nline1\r\nline2\r\n"


def test_execute_str_replace_keeps_the_files_crlf_line_endings(tmp_path):
    (tmp_path / "crlf.txt").write_bytes(b"line1\r\nline2\r\n")
    command = {"command": "str_replace", "path": "crlf.txt", "old_str": "line2", "new_str": "LINE2"}
    assert execute_text_editor(command, root=tmp_path) == "Updated crlf.txt"
    assert (tmp_path / "crlf.txt").read_bytes() == b"line1\r\nLINE2\r\n"


def test_execute_accepts_an_absolute_path_inside_the_root(tmp_path):
    (tmp_path / "notes.txt").write_text(NOTES)
    command = {"command": "view", "path": str(tmp_path / "notes.txt"), "view_range": [3, 3]}
    assert execute_text_editor(command, root=tmp_path) == "     3\tgamma"


def test_execute_refuses_a_path_outside_the_root(tmp_path):
    (tmp_path / "workspace").mkdir()
    (tmp_path / "outside.txt").write_text("secret\n")
    command = {"command": "view", "path": "../outside.txt"}
    with pytest.raises(NativeToolError, match="resolves outside the root directory"):
        execute_text_editor(command, root=tmp_path / "workspace")


def test_execute_rejects_the_dropped_undo_edit_command(tmp_path):
    with pytest.raises(NativeToolError) as excinfo:
        execute_text_editor({"command": "undo_edit", "path": "notes.txt"}, root=tmp_path)
    assert str(excinfo.value) == (
        "Unsupported text editor command: 'undo_edit'. Expected view, create, str_replace or insert."
    )


def test_execute_reports_a_missing_argument(tmp_path):
    with pytest.raises(NativeToolError) as excinfo:
        execute_text_editor({"command": "create", "path": "new.txt"}, root=tmp_path)
    assert str(excinfo.value) == "Missing 'file_text' in text editor command."
