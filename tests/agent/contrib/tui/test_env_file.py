"""`.env`: what it accepts, what it refuses, and who wins against a real
environment variable.

The refusals matter more than the accepts here. This parser exists because
`python-dotenv` reports an unreadable line as a `logging` warning and carries
on, and the TUI paints over stderr — so a single stray quote dropped a real
credential with nothing on screen to say so.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from luca.agent.contrib.tui.config import LucaConfigError
from luca.agent.contrib.tui.env_file import (
    ENV_ENV_PATH,
    apply_env_file,
    load_env_file,
    parse_env,
    resolve_env_path,
)

# ── the grammar ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Case:
    name: str
    text: str
    expected: dict


ACCEPTED = [
    Case("bare", "KEY=value", {"KEY": "value"}),
    Case("double_quoted", 'KEY="value"', {"KEY": "value"}),
    Case("single_quoted", "KEY='value'", {"KEY": "value"}),
    Case("export_prefix", "export KEY=value", {"KEY": "value"}),
    Case("comment_and_blanks", "# note\n\nKEY=value\n\n", {"KEY": "value"}),
    Case("crlf", "A=1\r\nB=2\r\n", {"A": "1", "B": "2"}),
    Case("surrounding_space", "  KEY = value  ", {"KEY": "value"}),
    Case("empty_value", "KEY=", {"KEY": ""}),
    Case("empty_quoted_value", 'KEY=""', {"KEY": ""}),
    # An AWS bearer token is base64 and ends in '=' padding; a URL has '/' and
    # ':'. None of them are special once we are past the first '='.
    Case("equals_inside_value", 'KEY="abc123="', {"KEY": "abc123="}),
    Case("equals_inside_bare_value", "KEY=abc123==", {"KEY": "abc123=="}),
    Case("url_value", "KEY=https://host:443/a/b", {"KEY": "https://host:443/a/b"}),
    # A '#' only starts a comment when whitespace precedes it, which is the
    # rule python-dotenv uses. A file written for the usual tool must not
    # quietly mean something else here.
    Case("hash_inside_bare_value", "KEY=a#b", {"KEY": "a#b"}),
    Case("trailing_comment_dropped", "KEY=value # note", {"KEY": "value"}),
    Case("tab_before_comment_dropped", "KEY=value\t# note", {"KEY": "value"}),
    Case("hash_first_is_part_of_the_value", "KEY=#abc", {"KEY": "#abc"}),
    Case("comment_not_stripped_inside_quotes", 'KEY="a # b"', {"KEY": "a # b"}),
    Case("quote_inside_single_quotes", "KEY='a\"b'", {"KEY": 'a"b'}),
    Case("several", 'A=1\nB="2"\nexport C=3', {"A": "1", "B": "2", "C": "3"}),
    Case("empty_file", "", {}),
]


@pytest.mark.parametrize("case", ACCEPTED, ids=lambda c: c.name)
def test_the_grammar_is_read_as_expected(case):
    assert parse_env(case.text) == case.expected


# ── the refusals ─────────────────────────────────────────────────────────────


def test_the_doubled_closing_quote_that_started_all_this():
    # Verbatim shape of the real `.env` line that cost an afternoon:
    #   AWS_BEARER_TOKEN_BEDROCK="ABSK...=""
    # python-dotenv drops the whole statement and logs a warning nobody sees.
    with pytest.raises(LucaConfigError) as exc_info:
        parse_env('AWS_BEARER_TOKEN_BEDROCK="abc123=""\n')

    assert str(exc_info.value) == (
        '.env line 1: AWS_BEARER_TOKEN_BEDROCK has trailing characters after the closing " quote'
    )


def test_an_unterminated_quote_names_the_variable():
    with pytest.raises(LucaConfigError, match='line 1: KEY has an unterminated " quote'):
        parse_env('KEY="value')


def test_a_line_with_no_equals_is_refused():
    with pytest.raises(LucaConfigError, match="line 2: no '=' in 'JUST_A_NAME'"):
        parse_env("A=1\nJUST_A_NAME\n")


def test_a_line_with_no_name_is_refused():
    with pytest.raises(LucaConfigError, match="line 1: no name before the '='"):
        parse_env("=orphan")


def test_the_reported_source_is_the_file_that_was_read(tmp_path):
    path = tmp_path / ".env"
    path.write_text('KEY="oops""')

    with pytest.raises(LucaConfigError, match=f"{path} line 1: KEY has trailing"):
        load_env_file(path)


# ── loading ──────────────────────────────────────────────────────────────────


def test_a_missing_file_is_simply_nothing_to_load(tmp_path):
    # Running off exported variables or auth.json is the ordinary case.
    assert load_env_file(tmp_path / "nope") == {}
    assert load_env_file(None) == {}


def test_applying_sets_only_the_names_that_are_absent(tmp_path, monkeypatch):
    # The real environment wins: exporting is deliberate, a checked-out file is
    # a default, and a stale default must not shadow it.
    path = tmp_path / ".env"
    path.write_text("ALREADY_SET=from-file\nNOT_SET=from-file\n")
    monkeypatch.setenv("ALREADY_SET", "from-shell")
    monkeypatch.delenv("NOT_SET", raising=False)

    applied = apply_env_file(path)

    assert applied == {"NOT_SET": "from-file"}
    assert (os.environ["ALREADY_SET"], os.environ["NOT_SET"]) == ("from-shell", "from-file")


# ── discovery ────────────────────────────────────────────────────────────────


def test_the_env_var_names_a_file_outright(monkeypatch):
    monkeypatch.setenv(ENV_ENV_PATH, "~/elsewhere.env")

    assert resolve_env_path() == Path.home() / "elsewhere.env"


def test_an_explicit_path_beats_the_env_var(monkeypatch):
    monkeypatch.setenv(ENV_ENV_PATH, "/from/env")

    assert resolve_env_path("/from/the/caller") == Path("/from/the/caller")


def test_it_is_found_from_a_subdirectory(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".env").write_text("KEY=value")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)

    assert resolve_env_path(cwd=nested) == (tmp_path / ".env").resolve()


def test_the_walk_stops_at_the_repo_root(tmp_path):
    # A `.env` above the project must not silently apply inside it.
    (tmp_path / ".env").write_text("KEY=outside")
    project = tmp_path / "project"
    project.mkdir()
    (project / ".git").mkdir()

    assert resolve_env_path(cwd=project) is None
