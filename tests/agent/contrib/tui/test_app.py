"""End-to-end Pilot tests: `AgentApp` driven by a scripted `FauxProvider`.

Each test scripts the provider, runs the app headless (`run_test`), submits
a prompt, waits for the drive worker to settle, and asserts on transcript
cell state (`.text`, `.status`, …) plus the persisted session.
"""

from textual.widgets import Input

from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.cells import (
    AssistantCell,
    NoticeCell,
    ReasoningCell,
    UserCell,
)
from luca.agent.contrib.tui.sessions import load_session
from luca.agent.core.models import ConversationStatus
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_error,
    faux_text,
    faux_thinking,
)

from .helpers import fresh_session, idle_again, submit, wait_until


def scripted(*responses) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return provider


async def test_text_turn_renders_user_and_assistant_cells(tmp_path):
    session = fresh_session()
    app = AgentApp(
        session,
        provider=scripted(faux_assistant_message([faux_text("Hello there!")])),
        workspace=tmp_path, session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))

        assert [cell.text for cell in app.query(UserCell)] == ["hi"]
        assert [cell.text for cell in app.query(AssistantCell)] == ["Hello there!"]
        assert app.runner.status is ConversationStatus.IDLE
        assert (tmp_path / f"{session.id}.json").exists()


async def test_streaming_thinking_renders_once(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message(
            [faux_thinking("pondering the greeting"), faux_text("Hey!")],
        )),
        workspace=tmp_path, session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))

        assert [cell.text for cell in app.query(ReasoningCell)] == [
            "pondering the greeting",
        ]
        assert [cell.text for cell in app.query(AssistantCell)] == ["Hey!"]


async def test_non_streaming_renders_the_same_transcript(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message(
            [faux_thinking("pondering the greeting"), faux_text("Hey!")],
        )),
        workspace=tmp_path, session_dir=tmp_path,
        streaming=False,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))

        assert [cell.text for cell in app.query(ReasoningCell)] == [
            "pondering the greeting",
        ]
        assert [cell.text for cell in app.query(AssistantCell)] == ["Hey!"]


async def test_blank_text_renders_no_assistant_cell(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message(
            [faux_thinking("deciding"), faux_text(" ")],
        )),
        workspace=tmp_path, session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))

        assert [cell.text for cell in app.query(ReasoningCell)] == ["deciding"]
        assert [cell.text for cell in app.query(AssistantCell)] == []


async def test_blank_text_renders_no_assistant_cell_non_streaming(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message([faux_text("   ")])),
        workspace=tmp_path, session_dir=tmp_path,
        streaming=False,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))

        assert [cell.text for cell in app.query(AssistantCell)] == []


async def test_resume_skips_blank_assistant_text(tmp_path):
    session = fresh_session()
    app = AgentApp(
        session,
        provider=scripted(faux_assistant_message(
            [faux_thinking("deciding"), faux_text(" ")],
        )),
        workspace=tmp_path, session_dir=tmp_path,
    )
    async with app.run_test() as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))

    reloaded = load_session(session.id, tmp_path)
    resumed = AgentApp(
        reloaded, provider=scripted(), workspace=tmp_path, session_dir=tmp_path,
    )
    async with resumed.run_test() as pilot:
        await pilot.pause()

        assert [cell.text for cell in resumed.query(ReasoningCell)] == ["deciding"]
        assert [cell.text for cell in resumed.query(AssistantCell)] == []


async def test_llm_failure_shows_an_error_notice_and_recovers(tmp_path):
    session = fresh_session()
    app = AgentApp(
        session,
        provider=scripted(faux_assistant_message([], error=faux_error("boom"))),
        workspace=tmp_path, session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "hi")
        await wait_until(
            pilot,
            lambda: bool(app.query(NoticeCell))
            and not app.query_one("#prompt", Input).disabled,
        )

        [notice] = app.query(NoticeCell)
        assert "turn failed" in notice.text
        assert app.runner.status is ConversationStatus.PENDING  # retry-ready
        assert (tmp_path / f"{session.id}.json").exists()


async def test_resume_replays_the_transcript(tmp_path):
    session = fresh_session()
    app = AgentApp(
        session,
        provider=scripted(faux_assistant_message(
            [faux_thinking("resumable pondering"), faux_text("First answer.")],
        )),
        workspace=tmp_path, session_dir=tmp_path,
    )
    async with app.run_test() as pilot:
        await submit(pilot, "first question")
        await wait_until(pilot, lambda: idle_again(app))

    reloaded = load_session(session.id, tmp_path)
    resumed = AgentApp(
        reloaded, provider=scripted(), workspace=tmp_path, session_dir=tmp_path,
    )
    async with resumed.run_test() as pilot:
        await pilot.pause()

        assert [cell.text for cell in resumed.query(UserCell)] == ["first question"]
        assert [cell.text for cell in resumed.query(ReasoningCell)] == [
            "resumable pondering",
        ]
        assert [cell.text for cell in resumed.query(AssistantCell)] == ["First answer."]
        assert resumed.runner.status is ConversationStatus.IDLE


async def test_ctrl_d_saves_and_quits(tmp_path):
    session = fresh_session()
    app = AgentApp(
        session, provider=scripted(), workspace=tmp_path, session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await pilot.press("ctrl+d")
        await pilot.pause()

    assert app.is_running is False
    assert (tmp_path / f"{session.id}.json").exists()
