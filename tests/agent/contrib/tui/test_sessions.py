"""Session persistence helpers: the store layout, and the resume listing."""

import os
from pathlib import Path

import pytest
from pydantic_core import PydanticSerializationError

from luca.agent.contrib.tui.sessions import (
    DEFAULT_STORE,
    SessionSummary,
    encode_project_path,
    fork_session,
    list_sessions,
    load_session,
    resolve_session_directory,
    save_session,
    session_path,
    summarize_session,
)
from tests.agent.scenarios import (
    main_conversation,
)

from .helpers import fresh_session, with_user_message


def test_session_path_joins_directory(tmp_path):
    assert session_path("abc", tmp_path) == tmp_path / "abc.json"


def test_save_load_round_trip(tmp_path):
    session = fresh_session()

    save_session(session, tmp_path)

    assert load_session(session.id, tmp_path) == session


def test_fork_gets_a_fresh_id_and_keeps_the_conversation():
    session = fresh_session()

    forked = fork_session(session)

    assert forked.id != session.id
    assert main_conversation(forked) == main_conversation(session)
    assert forked.entries == session.entries


def test_save_replaces_the_file_atomically_and_leaves_no_temporaries(tmp_path):
    # a crash mid-write must never leave truncated JSON: the write goes to a
    # temporary in the same directory and is renamed into place
    session = fresh_session()
    save_session(session, tmp_path)

    save_session(session, tmp_path)  # the same path, written again

    assert [p.name for p in tmp_path.iterdir()] == [f"{session.id}.json"]
    assert load_session(session.id, tmp_path) == session


def test_a_failed_save_leaves_the_previous_file_intact(tmp_path):
    session = fresh_session()
    save_session(session, tmp_path)
    before = session_path(session.id, tmp_path).read_text()
    session.entries["broken"] = object()  # a value model_dump_json cannot write

    with pytest.raises(PydanticSerializationError, match="Unable to serialize unknown type"):
        save_session(session, tmp_path)

    assert session_path(session.id, tmp_path).read_text() == before
    assert [p.name for p in tmp_path.iterdir()] == [f"{session.id}.json"]


def test_save_creates_a_project_directory_that_does_not_exist_yet(tmp_path):
    session = fresh_session()
    store = tmp_path / "store" / "-a-project"

    save_session(session, store)

    assert load_session(session.id, store) == session


# ── the project path encoding ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "encoded"),
    [
        ("/a/b", "-a-b"),
        ("/Users/me/MY/Luca/luca-py", "-Users-me-MY-Luca-luca-py"),
        ("/", "-"),  # the filesystem root is one separator, so one dash
        ("C:/Users/me/project", "C--Users-me-project"),  # a Windows drive letter
    ],
)
def test_a_project_path_becomes_one_directory_name(path, encoded, monkeypatch):
    # resolve() is a no-op on these, since they are already absolute
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)

    assert encode_project_path(path) == encoded


def test_the_encoding_is_lossy_and_that_is_accepted(monkeypatch):
    # `a/b` and a directory literally named `a-b` collide, exactly as they do
    # in ~/.claude/projects. A collision only means two projects share a
    # directory, and sessions are keyed by id inside it.
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)

    assert encode_project_path("/x/a/b") == encode_project_path("/x/a-b")


def test_a_relative_path_is_resolved_before_encoding(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert encode_project_path(".") == encode_project_path(tmp_path)


# ── where a project's sessions live ──────────────────────────────────────────


def test_the_default_store_is_under_the_home_directory(tmp_path):
    resolved = resolve_session_directory(tmp_path)

    assert resolved == Path(DEFAULT_STORE).expanduser() / encode_project_path(tmp_path)
    # the conftest points HOME at a tmp dir; if that ever stops working this
    # test writes into the contributor's real home
    assert str(resolved).startswith(os.environ["HOME"])


def test_the_configured_root_replaces_the_default(tmp_path):
    assert resolve_session_directory(tmp_path, str(tmp_path / "elsewhere")) == (
        tmp_path / "elsewhere" / encode_project_path(tmp_path)
    )


def test_a_tilde_in_the_configured_root_is_expanded(tmp_path):
    resolved = resolve_session_directory(tmp_path, "~/store")

    assert resolved == Path.home() / "store" / encode_project_path(tmp_path)


def test_the_workspace_decides_the_directory_not_the_process_cwd(tmp_path, monkeypatch):
    workspace = tmp_path / "project"
    workspace.mkdir()
    monkeypatch.chdir(tmp_path)

    assert resolve_session_directory(workspace) == resolve_session_directory(workspace, None)
    assert resolve_session_directory(workspace) != resolve_session_directory(tmp_path)


# ── the resume listing ───────────────────────────────────────────────────────


def test_a_missing_directory_lists_nothing(tmp_path):
    assert list_sessions(tmp_path / "nope") == []


def test_an_empty_directory_lists_nothing(tmp_path):
    assert list_sessions(tmp_path) == []


def test_a_summary_carries_the_first_user_message_and_the_model(tmp_path):
    session = with_user_message("fix the parser")
    path = save_session(session, tmp_path)

    assert summarize_session(path) == SessionSummary(
        id=session.id,
        path=path,
        modified=summarize_session(path).modified,  # the file's own mtime
        title="fix the parser",
        turns=1,
        model=session.session_config.llm_config.model,
    )


def test_a_session_with_no_message_yet_still_summarizes(tmp_path):
    path = save_session(fresh_session(), tmp_path)

    assert summarize_session(path).title == "(empty)"


def test_only_the_first_line_of_a_long_message_becomes_the_title(tmp_path):
    path = save_session(with_user_message("first line\nsecond line"), tmp_path)

    assert summarize_session(path).title == "first line"


def test_the_newest_session_is_listed_first(tmp_path):
    older = save_session(with_user_message("older"), tmp_path)
    newer = save_session(with_user_message("newer"), tmp_path)
    os.utime(older, (1, 1))

    assert [summary.title for summary in list_sessions(tmp_path)] == ["newer", "older"]
    assert [summary.path for summary in list_sessions(tmp_path)] == [newer, older]


def test_an_unreadable_session_is_skipped_and_its_neighbours_still_load(tmp_path):
    save_session(with_user_message("good"), tmp_path)
    (tmp_path / "broken.json").write_text("{not json")

    assert [summary.title for summary in list_sessions(tmp_path)] == ["good"]


def test_summarizing_an_unreadable_file_returns_nothing(tmp_path):
    (tmp_path / "broken.json").write_text("{not json")

    assert summarize_session(tmp_path / "broken.json") is None
