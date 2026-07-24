"""Compaction through the app: `/compact`, auto-compact at the threshold, the
failure-leaves-source path, and the context bar. Driven headless via Pilot with
a scripted FauxProvider.
"""

from luca.agent.contrib.compaction import Compactor
from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.cells import NoticeCell
from luca.agent.contrib.tui.commands import COMMANDS
from luca.agent.contrib.tui.context_bar import ContextBar
from luca.agent.core.models import CompactionEntry
from luca.client.exceptions import ProviderAPIError
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_error,
    faux_text,
)

from .helpers import fresh_session, idle_again, submit, wait_until


def scripted(*responses) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return provider


def _notices(app) -> list[str]:
    return [cell.text for cell in app.query(NoticeCell)]


def _first_entry(app):
    session = app.runner.session
    return session.entries[session.active_conversation.nodes[0]]


def test_compact_is_a_registered_command():
    assert any(command.name == "compact" for command in COMMANDS)


async def test_manual_compact_swaps_to_a_compacted_session(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(
            faux_assistant_message([faux_text("Answer.")]),
            faux_assistant_message([faux_text("A DENSE SUMMARY")]),
        ),
        workspace=tmp_path, session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))
        old_id = app.runner.session.id

        await submit(pilot, "/compact")
        await wait_until(pilot, lambda: app.runner.session.id != old_id and idle_again(app))

        first = _first_entry(app)
        assert isinstance(first, CompactionEntry)
        assert first.summary == "A DENSE SUMMARY"
        assert (tmp_path / f"{app.runner.session.id}.json").exists()
        assert (tmp_path / f"{old_id}.json").exists()  # source preserved
        # the replayed transcript shows the compaction marker for the new session
        assert any("context compacted" in note for note in _notices(app))


async def test_auto_compaction_fires_when_utilization_crosses_the_threshold(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(
            faux_assistant_message([faux_text("Answer.")]),
            faux_assistant_message([faux_text("AUTO SUMMARY")]),
        ),
        workspace=tmp_path, session_dir=tmp_path,
        compactor=Compactor(default_window=1, threshold=0.5),  # any content trips it
    )

    async with app.run_test() as pilot:
        old_id = app.runner.session.id
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: app.runner.session.id != old_id and idle_again(app))

        first = _first_entry(app)
        assert isinstance(first, CompactionEntry)
        assert first.summary == "AUTO SUMMARY"


async def test_compaction_failure_leaves_the_source_session_intact(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(
            faux_assistant_message([faux_text("Answer.")]),
            faux_assistant_message(
                [], finish_reason="stop",
                error=faux_error("summary boom", error_class=ProviderAPIError),
            ),
        ),
        workspace=tmp_path, session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))
        old_id = app.runner.session.id

        await submit(pilot, "/compact")
        await wait_until(
            pilot,
            lambda: idle_again(app)
            and any("compaction failed" in note for note in _notices(app)),
        )

        assert app.runner.session.id == old_id  # nothing swapped


async def test_the_context_bar_is_mounted_and_renders(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message([faux_text("Hello.")])),
        workspace=tmp_path, session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))
        bar = app.query_one("#context-bar", ContextBar)
        assert bar.text.startswith("context ")


async def test_the_context_bar_turns_danger_and_fills_when_over_the_threshold(tmp_path):
    from textual.widgets import ProgressBar

    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message([faux_text("Hello.")])),
        workspace=tmp_path, session_dir=tmp_path,
        # a tiny window forces 100%; disabled so it does not auto-compact away
        compactor=Compactor(default_window=1, threshold=0.5, enabled=False),
    )

    async with app.run_test() as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))
        bar = app.query_one("#context-bar", ContextBar)
        assert bar.has_class("-danger")
        assert app.query_one("#ctx-progress", ProgressBar).percentage == 1.0
