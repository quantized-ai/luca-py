"""Compaction through the app with a real SummarizingContextManager: /compact
schedules and drives, the transition archives the old conversation, the new one
leads with the summary, and the transcript states it as a compaction notice.
Core owns the mechanics (tested in tests/agent)."""

from luca.agent.contrib.simple_context_manager import SummarizingContextManager
from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.blocks import NoticeLine
from luca.agent.core.models import CompactionEntry, TextContent
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text, faux_tool_call
from tests.agent.scenarios import main_conversation

from .helpers import fresh_session, idle_again, submit, wait_until


def scripted(*responses) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return provider


async def test_manual_compact_renders_the_notice_and_leads_with_the_summary(tmp_path):
    faux = scripted(
        faux_assistant_message([faux_text("Answer.")]),
        faux_assistant_message([faux_text("A DENSE SUMMARY")]),  # the /compact call
    )
    app = AgentApp(
        fresh_session(),
        provider=faux,
        workspace=tmp_path,
        session_dir=tmp_path,
        skills=False,
        instructions=False,
        context_manager=SummarizingContextManager(provider=faux, enabled=False),
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))
        old_entry_ids = set(app.runner.session.entries)

        await submit(pilot, "/compact")
        await wait_until(
            pilot,
            lambda: idle_again(app) and main_conversation(app.runner.session).previous_conversation_id is not None,
        )

        session = app.runner.session
        first = session.entries[main_conversation(session).nodes[0]]
        assert isinstance(first, CompactionEntry)
        assert first.parts == [TextContent(text="A DENSE SUMMARY")]
        assert len(first.compacted_nodes) == 4  # TurnStart, user, assistant, TurnFinish
        assert old_entry_ids <= set(session.entries)  # immutable: originals preserved
        assert [(notice.text, notice.error) for notice in app.query(NoticeLine)] == [
            ("compacting the conversation…", False),
            ("context compacted · 4 entries summarized", False),
        ]


async def test_compaction_moves_the_memory_stores_onto_the_new_conversation(tmp_path):
    # A transition installs a NEW conversation id and both memory stores are
    # filed under the old one. The app owns those dicts — it handed them to the
    # plugin — so the app is what moves them, or the agent resumes a compacted
    # conversation with no list and numbering back at #1.
    faux = scripted(
        faux_assistant_message([faux_tool_call("update_todos", {"todos": [{"content": "T1", "status": "pending"}]})]),
        faux_assistant_message([faux_text("Planned.")]),
        faux_assistant_message([faux_text("A DENSE SUMMARY")]),  # the /compact call
    )
    app = AgentApp(
        fresh_session(),
        provider=faux,
        workspace=tmp_path,
        session_dir=tmp_path,
        skills=False,
        instructions=False,
        context_manager=SummarizingContextManager(provider=faux, enabled=False),
    )

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "plan it")
        await wait_until(pilot, lambda: idle_again(app))
        outgoing = app.runner.main_conversation_id

        await submit(pilot, "/compact")
        await wait_until(
            pilot,
            lambda: idle_again(app) and main_conversation(app.runner.session).previous_conversation_id is not None,
        )

        session = app.runner.session
        assert session.extras["todos"] == {
            session.main_conversation_id: {
                "todos": [{"id": 1, "content": "T1", "status": "pending"}],
                "next_id": 2,
            }
        }
        assert outgoing not in session.extras["todos"]
