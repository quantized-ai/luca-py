"""Workspace listing and the `@` picker's fuzzy matcher.

Matching is subsequence, so the tests are mostly RANKING tests: which file
comes first for a given query. The bonus weights exist to produce these
orderings, so a weight change that breaks one shows up here.
"""

import pytest

from luca.agent.contrib.tui.files import (
    list_workspace_files,
    mark_path,
    match_files,
    score_path,
)

TREE = [
    "README.md",
    "design_handoff_luca_tui/README.md",
    "design_handoff_luca_tui/prompt.md",
    "docs/agent/contrib/tui/README.md",
    "luca/agent/contrib/tui/app.py",
    "luca/agent/contrib/tui/prompt.py",
    "luca/client/providers.py",
    "main.py",
]


def ranked(query, files=TREE, limit=8) -> list[str]:
    return [path for path, _marked, _tokens in match_files(files, query, limit=limit)]


# ── subsequence, not substring ────────────────────────────────────────────────


def test_a_query_spanning_a_directory_and_a_filename_matches():
    # the motivating case: "design/prompt" is not a substring of the path, but
    # its characters appear in order
    assert ranked("design/prompt") == ["design_handoff_luca_tui/prompt.md"]


def test_an_abbreviation_scattered_across_the_path_matches():
    assert "luca/agent/contrib/tui/app.py" in ranked("lactui")


def test_characters_out_of_order_do_not_match():
    assert ranked("tpmorp") == []


def test_score_path_returns_the_matched_indices():
    assert score_path("mp", "main.py") == (pytest.approx(31.3), [0, 5])


def test_the_scan_prefers_the_later_of_two_viable_positions():
    # "tui" must land on "/tui/", not on the t in "agent" — a left-to-right
    # greedy scan takes the wrong t and scores the file far too low
    _score, positions = score_path("tui", "luca/agent/contrib/tui/app.py")

    assert positions == [19, 20, 21]


def test_score_path_returns_none_when_a_character_is_missing():
    assert score_path("zq", "main.py") is None


# ── ranking ───────────────────────────────────────────────────────────────────


def test_a_basename_hit_outranks_a_directory_hit():
    assert ranked("tui")[0] == "luca/agent/contrib/tui/app.py"


def test_a_contiguous_run_outranks_scattered_characters():
    files = ["luca/agent/contrib/tui/prompt.py", "p/r/o/m/p/t/x.py"]

    assert ranked("prompt", files)[0] == "luca/agent/contrib/tui/prompt.py"


def test_the_shorter_path_wins_an_otherwise_equal_match():
    assert ranked("readme") == [
        "README.md",
        "docs/agent/contrib/tui/README.md",
        "design_handoff_luca_tui/README.md",
    ]


def test_a_camel_case_hump_counts_as_a_word_boundary():
    files = ["src/parsequery.py", "src/parseQuery.py"]

    assert ranked("pq", files)[0] == "src/parseQuery.py"


def test_matching_is_case_insensitive():
    assert ranked("README") == ranked("readme")


def test_an_empty_query_lists_the_files_as_given():
    assert ranked("", limit=3) == TREE[:3]


def test_the_row_limit_caps_the_display_only():
    assert len(ranked("p", limit=2)) == 2


# ── the truncation bug ────────────────────────────────────────────────────────


def test_a_late_alphabet_file_is_reachable_in_a_large_workspace():
    # listing used to be `sorted(files)[:2000]`, which made every file past the
    # alphabetical cutoff permanently unmatchable
    files = [f"a{index:05d}.py" for index in range(5_000)] + ["zzz_service.py"]

    assert ranked("zzz", files) == ["zzz_service.py"]


# ── span marking ──────────────────────────────────────────────────────────────


def test_a_contiguous_match_is_marked_as_one_span():
    assert mark_path("main.py", [0, 1, 2, 3]) == "[accent]main[/].py"


def test_a_scattered_match_is_marked_as_several_spans():
    assert mark_path("main.py", [0, 5]) == "[accent]m[/]ain.[accent]p[/]y"


def test_a_match_running_to_the_end_leaves_no_trailing_text():
    assert mark_path("app", [0, 1, 2]) == "[accent]app[/]"


def test_match_files_marks_the_row_it_returns():
    [(path, marked, _tokens)] = match_files(["main.py"], "mp", limit=1)

    assert (path, marked) == ("main.py", "[accent]m[/]ain.[accent]p[/]y")


# ── listing ───────────────────────────────────────────────────────────────────


def test_listing_a_plain_directory_walks_it(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "b.py").write_text("b")
    (tmp_path / "a.py").write_text("a")

    assert list_workspace_files(tmp_path) == ["a.py", "pkg/b.py"]


def test_the_walk_skips_noise_directories(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("x")
    (tmp_path / "a.py").write_text("a")

    assert list_workspace_files(tmp_path) == ["a.py"]
