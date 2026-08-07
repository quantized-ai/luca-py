"""The V4A diff grammar: create, update, context matching, newline style.

Several cases are ported from openai-agents-python, tests/test_apply_diff.py
— the best available spec of the format. That code is MIT licensed,
Copyright (c) 2025 OpenAI; the full notice, permission grant and warranty
disclaimer included, is in the module under test,
`luca/client/native/v4a.py`.
"""

import pytest

from luca.client.native import NativeToolError, apply_diff


def test_create_mode_takes_plus_prefixed_lines():
    diff = "\n".join(["+hello", "+world", "+"])
    assert apply_diff("", diff, mode="create") == "hello\nworld\n"


def test_create_mode_drops_the_diffs_own_trailing_blank_line():
    assert apply_diff("", "+hello\n+world\n+\n", mode="create") == "hello\nworld\n"


def test_create_mode_rejects_a_line_without_the_plus_prefix():
    with pytest.raises(NativeToolError, match="Invalid Add File Line: plain line"):
        apply_diff("", "plain line", mode="create")


def test_create_mode_follows_the_diffs_newline_style():
    diff = "\r\n".join(["+hello", "+world", "+"])
    assert apply_diff("", diff, mode="create") == "hello\r\nworld\r\n"


def test_update_replaces_a_line_found_by_context():
    diff = "\n".join(["@@ line1", "-line2", "+updated", " line3"])
    assert apply_diff("line1\nline2\nline3\n", diff) == "line1\nupdated\nline3\n"


def test_update_applies_two_chunks_from_one_section():
    diff = "\n".join(["@@ a", "-b", "+B", " c", "-d", "+D"])
    assert apply_diff("a\nb\nc\nd\ne\n", diff) == "a\nB\nc\nD\ne\n"


def test_update_appends_at_the_end_of_file_marker():
    diff = "\n".join([" two", "+three", "*** End of File"])
    assert apply_diff("one\ntwo", diff) == "one\ntwo\nthree"


def test_update_anchors_an_end_of_file_section_at_the_last_match():
    diff = "\n".join([" a", " b", "+TAIL", "*** End of File"])
    assert apply_diff("a\nb\na\nb", diff) == "a\nb\na\nb\nTAIL"


def test_update_prefers_a_right_stripped_match_over_a_fully_stripped_one():
    diff = "\n".join(["@@", " x", "+INSERTED"])
    assert apply_diff("a\n x \nb\nc\nd\nx  \ne\n", diff) == "a\n x \nb\nc\nd\nx  \nINSERTED\ne\n"


def test_update_matches_context_ignoring_surrounding_whitespace():
    diff = "\n".join(["@@ line1", "-line2", "+updated", " line3"])
    assert apply_diff("line1\n  line2  \nline3\n", diff) == "line1\nupdated\nline3\n"


def test_update_inserts_a_floating_hunk_into_empty_text():
    diff = "\n".join(["@@", "+hello", "+world"])
    assert apply_diff("", diff) == "hello\nworld\n"


def test_update_raises_when_the_context_is_not_in_the_text():
    diff = "\n".join(["@@ -1,2 +1,2 @@", " x", "-two", "+2"])
    with pytest.raises(NativeToolError, match="Invalid Context 0:\nx\ntwo"):
        apply_diff("one\ntwo\n", diff)


def test_update_keeps_the_inputs_crlf_newlines():
    diff = "\n".join(["@@ line1", "-line2", "+updated", " line3"])
    assert apply_diff("line1\r\nline2\r\nline3\r\n", diff) == "line1\r\nupdated\r\nline3\r\n"


def test_update_keeps_the_inputs_lf_newlines_against_a_crlf_diff():
    diff = "\r\n".join(["@@ line1", "-line2", "+updated", " line3"])
    assert apply_diff("line1\nline2\nline3\n", diff) == "line1\nupdated\nline3\n"
