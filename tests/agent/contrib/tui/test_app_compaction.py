"""Compaction through the app with a real SummarizingCompactionPolicy: /compact
schedules and drives, the transition archives the old conversation, and the new
one leads with the summary. Core owns the mechanics (tested in tests/agent)."""

from luca.agent.contrib.compaction import SummarizingCompactionPolicy
from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.cells import CompactionCell, NoticeCell
from luca.agent.core.models import CompactionEntry, TextContent
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text

from .helpers import fresh_session, idle_again, submit, wait_until


def scripted(*responses) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return provider


def _active_first_entry(app) -> object:
    session = app.runner.session
    return session.entries[session.active_conversation.nodes[0]]


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
        compaction_policy=SummarizingCompactionPolicy(provider=faux, enabled=False),
    )

    async with app.run_test() as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))
        old_entry_ids = set(app.runner.session.entries)

        await submit(pilot, "/compact")
        await wait_until(
            pilot,
            lambda: idle_again(app) and len(app.runner.session.conversation_history) == 1,
        )

        session = app.runner.session
        first = _active_first_entry(app)
        assert isinstance(first, CompactionEntry)
        assert first.parts == [TextContent(text="A DENSE SUMMARY")]
        assert old_entry_ids <= set(session.entries)  # immutable: originals preserved
        assert app.query(CompactionCell)  # the summary rendered a transcript cell


async def test_compact_reports_when_no_policy_is_configured(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message([faux_text("hi")])),
        workspace=tmp_path,
        session_dir=tmp_path,
        compaction_policy=None,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "/compact")
        await wait_until(pilot, lambda: idle_again(app))
        assert any("no compaction policy" in cell.text for cell in app.query(NoticeCell))
