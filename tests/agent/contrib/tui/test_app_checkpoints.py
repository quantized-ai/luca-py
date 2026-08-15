"""Checkpoints through the app: `/undo` and `/rewind` with a real shadow git
store over a real workspace.

Marked `checkpoints` so the conftest leaves the store switched on — every other
TUI test runs with it disabled, because the feature shells out to git and
writes a shadow repository beside the session.

Core owns the rewind mechanics (`tests/agent/test_runner_rewind.py`) and the
store owns the file mechanics (`tests/agent/contrib/test_checkpoints_store.py`).
What is under test here is the WIRING: that a turn is checkpointed before it
opens, that restoring puts both halves back, and that the transcript on screen
follows the conversation rather than the one that was archived.
"""

import shutil

import pytest

from luca.agent.contrib.checkpoints import read_index
from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.blocks import NoticeLine
from luca.agent.contrib.tui.shells import OverlayListView
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text
from tests.agent.scenarios import main_conversation

from .helpers import fresh_session, idle_again, submit, wait_until

pytestmark = [
    pytest.mark.checkpoints,
    pytest.mark.skipif(shutil.which("git") is None, reason="checkpoints need a git binary"),
]


def scripted(*responses) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return provider


def build_app(tmp_path, faux, **over) -> AgentApp:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "kept.py").write_text("original\n")
    return AgentApp(
        fresh_session(),
        provider=faux,
        workspace=workspace,
        session_dir=tmp_path / "store",
        skills=False,
        instructions=False,
        **over,
    )


# ── taking them ───────────────────────────────────────────────────────────────


async def test_a_turn_is_checkpointed_before_it_opens(tmp_path):
    app = build_app(tmp_path, scripted(faux_assistant_message([faux_text("Answer.")])))

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "do the thing")
        await wait_until(pilot, lambda: idle_again(app))

        [checkpoint] = read_index(app.runner.session).checkpoints
        # anchored BEFORE the turn: an empty conversation had no leaf
        assert checkpoint.anchor_entry_id is None
        assert checkpoint.label == "do the thing"


async def test_the_second_turn_anchors_at_the_first_turns_last_node(tmp_path):
    app = build_app(
        tmp_path,
        scripted(
            faux_assistant_message([faux_text("One.")]),
            faux_assistant_message([faux_text("Two.")]),
        ),
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "first")
        await wait_until(pilot, lambda: idle_again(app))
        after_first = list(main_conversation(app.runner.session).nodes)

        await submit(pilot, "second")
        await wait_until(pilot, lambda: idle_again(app))

        second = read_index(app.runner.session).checkpoints[1]
        assert second.anchor_entry_id == after_first[-1]


async def test_checkpoints_are_off_without_git(tmp_path, monkeypatch):
    monkeypatch.setattr("luca.agent.contrib.checkpoints.store.shutil.which", lambda _: None)
    app = build_app(tmp_path, scripted(faux_assistant_message([faux_text("Answer.")])))

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "do the thing")
        await wait_until(pilot, lambda: idle_again(app))

        assert read_index(app.runner.session).checkpoints == []


# ── restoring them ────────────────────────────────────────────────────────────


async def test_undo_reverts_the_files_and_rewinds_the_conversation(tmp_path):
    app = build_app(tmp_path, scripted(faux_assistant_message([faux_text("Answer.")])))
    workspace = tmp_path / "workspace"

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "edit the file")
        await wait_until(pilot, lambda: idle_again(app))
        # stand in for the edit the agent would have made
        (workspace / "kept.py").write_text("the agent changed this\n")

        await submit(pilot, "/undo")
        await wait_until(pilot, lambda: main_conversation(app.runner.session).nodes == [])

        assert (workspace / "kept.py").read_text() == "original\n"
        assert app.runner.idle()


async def test_undo_archives_the_turn_rather_than_deleting_it(tmp_path):
    app = build_app(tmp_path, scripted(faux_assistant_message([faux_text("Answer.")])))

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "edit the file")
        await wait_until(pilot, lambda: idle_again(app))
        entries_before = set(app.runner.session.entries)

        await submit(pilot, "/undo")
        await wait_until(pilot, lambda: main_conversation(app.runner.session).nodes == [])

        session = app.runner.session
        assert entries_before <= set(session.entries)
        assert main_conversation(session).previous_conversation_id is not None


async def test_undo_clears_the_rewound_turn_from_the_transcript(tmp_path):
    app = build_app(tmp_path, scripted(faux_assistant_message([faux_text("Answer.")])))

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "edit the file")
        await wait_until(pilot, lambda: idle_again(app))

        await submit(pilot, "/undo")
        await wait_until(pilot, lambda: main_conversation(app.runner.session).nodes == [])
        await pilot.pause()

        rendered = app.transcript.query("*")
        assert "edit the file" not in " ".join(str(getattr(w, "renderable", "")) for w in rendered)


async def test_undo_with_nothing_to_undo_says_so(tmp_path):
    app = build_app(tmp_path, scripted())

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/undo")
        await pilot.pause()

        assert ("nothing to undo", True) in [(n.text, n.error) for n in app.query(NoticeLine)]


async def test_rewind_offers_one_row_per_checkpoint(tmp_path):
    app = build_app(
        tmp_path,
        scripted(
            faux_assistant_message([faux_text("One.")]),
            faux_assistant_message([faux_text("Two.")]),
        ),
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "first")
        await wait_until(pilot, lambda: idle_again(app))
        await submit(pilot, "second")
        await wait_until(pilot, lambda: idle_again(app))

        await submit(pilot, "/rewind")
        await pilot.pause()

        # newest first, so the second turn's checkpoint leads
        assert [row.primary for row in app._menu_rows] == ["second", "first"]


async def test_picking_an_earlier_checkpoint_drops_every_turn_after_it(tmp_path):
    """The picker's own callback, and the case `/undo` cannot reach: restoring
    a checkpoint two turns back rewinds BOTH of them in one cut."""
    app = build_app(
        tmp_path,
        scripted(
            faux_assistant_message([faux_text("One.")]),
            faux_assistant_message([faux_text("Two.")]),
        ),
    )
    workspace = tmp_path / "workspace"

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "first")
        await wait_until(pilot, lambda: idle_again(app))
        (workspace / "kept.py").write_text("after the first turn\n")
        await submit(pilot, "second")
        await wait_until(pilot, lambda: idle_again(app))
        (workspace / "kept.py").write_text("after the second turn\n")

        await submit(pilot, "/rewind")
        await wait_until(pilot, lambda: bool(app.query(OverlayListView)))
        view = app.query_one(OverlayListView)
        view.post_message(OverlayListView.Committed(view, 1))  # "first" — the older one
        await wait_until(pilot, lambda: main_conversation(app.runner.session).nodes == [])

        assert (workspace / "kept.py").read_text() == "original\n"
        assert app.runner.idle()
