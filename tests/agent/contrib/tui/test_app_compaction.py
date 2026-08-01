"""Compaction through the app with a real SummarizingContextManager: /compact
schedules and drives, the transition archives the old conversation, and the new
one leads with the summary. Core owns the mechanics (tested in tests/agent)."""

from luca.agent.contrib.simple_context_manager import SummarizingContextManager
from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.cells import CompactionCell, NoticeCell
from luca.agent.core.models import CompactionEntry, TextContent
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text
from tests.agent.scenarios import (
    main_conversation,
)

from .helpers import fresh_session, idle_again, submit, wait_until


def scripted(*responses) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return provider


def _active_first_entry(app) -> object:
    session = app.runner.session
    return session.entries[main_conversation(session).nodes[0]]


async def test_manual_compact_archives_the_old_conversation_and_leads_with_the_summary(tmp_path):
    faux = scripted(
        faux_assistant_message([faux_text("Answer.")]),
        faux_assistant_message([faux_text("A DENSE SUMMARY")]),  # the /compact call
    )
    app = AgentApp(
        fresh_session(),
        provider=faux,
        workspace=tmp_path,
        session_dir=tmp_path,
        context_manager=SummarizingContextManager(provider=faux, enabled=False),
    )

    async with app.run_test() as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))
        old_entry_ids = set(app.runner.session.entries)

        await submit(pilot, "/compact")
        await wait_until(
            pilot,
            lambda: idle_again(app) and main_conversation(app.runner.session).previous_conversation_id is not None,
        )

        session = app.runner.session
        first = _active_first_entry(app)
        assert isinstance(first, CompactionEntry)
        assert first.parts == [TextContent(text="A DENSE SUMMARY")]
        assert old_entry_ids <= set(session.entries)  # immutable: originals preserved
        assert app.query(CompactionCell)  # the summary rendered a transcript cell


async def test_compact_surfaces_a_turn_error_when_the_manager_cannot_compact(tmp_path):
    # `context_manager=None` falls back to core's default, which accounts but
    # never compacts: /compact schedules, and the base `compact()` raises on the
    # drive, which the app renders as an ordinary turn failure.
    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message([faux_text("hi")])),
        workspace=tmp_path,
        session_dir=tmp_path,
        context_manager=None,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "/compact")
        await wait_until(pilot, lambda: idle_again(app))
        assert any("turn failed" in cell.text for cell in app.query(NoticeCell))
