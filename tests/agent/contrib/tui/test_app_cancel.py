"""Escape during a live run: `run.cancel()` and the rendered wind-down."""

from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.cells import NoticeCell
from luca.agent.core.models import ConversationStatus, TurnFinish, TurnOutcome
from luca.client.testing import FauxProvider, faux_assistant_message, faux_hang

from .helpers import fresh_session, idle_again, submit, wait_until


async def test_escape_cancels_a_hung_completion(tmp_path):
    provider = FauxProvider()
    provider.set_responses([faux_assistant_message([faux_hang()])])
    session = fresh_session()
    app = AgentApp(
        session, provider=provider, workspace=tmp_path, session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "hang please")
        await wait_until(pilot, lambda: app.current_run is not None)
        await pilot.press("escape")
        await wait_until(pilot, lambda: idle_again(app))

        assert "cancelling — winding down the turn" in [
            cell.text for cell in app.query(NoticeCell)
        ]
        outcomes = [
            entry.outcome for entry in app.runner.session.entries.values()
            if isinstance(entry, TurnFinish)
        ]
        assert outcomes == [TurnOutcome.CANCELLED]
        assert app.runner.status is ConversationStatus.IDLE


async def test_escape_when_idle_is_a_no_op(tmp_path):
    app = AgentApp(
        fresh_session(), provider=FauxProvider(),
        workspace=tmp_path, session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()

        assert app.runner.status is ConversationStatus.IDLE
        assert list(app.query(NoticeCell)) == []
