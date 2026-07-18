"""Session persistence helpers."""

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
