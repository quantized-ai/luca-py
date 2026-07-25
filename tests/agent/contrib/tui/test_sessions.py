"""Session persistence helpers."""

import pytest

from luca.agent.contrib.tui.sessions import (
    fork_session,
    load_session,
    save_session,
    session_path,
)

from .helpers import fresh_session


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
    assert forked.active_conversation == session.active_conversation
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

    with pytest.raises(Exception):
        save_session(session, tmp_path)

    assert session_path(session.id, tmp_path).read_text() == before
    assert [p.name for p in tmp_path.iterdir()] == [f"{session.id}.json"]
