"""The approval gate through the modal: approve, deny, abandon.

The demo math tools are resourceless, so every gated call presents the
synthesized three-option prompt (approve once / deny / abandon). Answers are
selected by digit key, exactly as a user would.
"""

from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.cells import AssistantCell, NoticeCell, ToolCallCell
from luca.agent.contrib.tui.screens import ApprovalScreen
from luca.agent.contrib.tui.sessions import load_session
from luca.agent.core.models import ConversationStatus, ExecutionStatus
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
)

from .helpers import fresh_session, idle_again, submit, wait_until


def scripted(*responses) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return provider


def tool_turn(reaction: str) -> list:
    return [
        faux_assistant_message(
            [faux_tool_call("multiply", {"a": 6, "b": 7}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text(reaction)]),
    ]


async def test_approve_once_runs_the_tool(tmp_path):
    session = fresh_session()
    app = AgentApp(
        session,
        provider=scripted(*tool_turn("It is 42.")),
        workspace=tmp_path, session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "what is 6 times 7?")
        await wait_until(pilot, lambda: isinstance(app.screen, ApprovalScreen))

        assert app.screen.prompt.tool_name == "multiply"
        assert app.screen.prompt.preview == "Run multiply"

        await pilot.press("1")  # Approve once
        await wait_until(pilot, lambda: idle_again(app))

        [cell] = app.query(ToolCallCell)
        assert cell.status is ExecutionStatus.COMPLETED
        assert cell.result_text == "42.0"
        assert cell.is_error is False
        assert [c.text for c in app.query(AssistantCell)] == ["It is 42."]

    # the reloaded session replays the finished tool round
    replayed = AgentApp(
        load_session(session.id, tmp_path),
        provider=scripted(), workspace=tmp_path, session_dir=tmp_path,
    )
    async with replayed.run_test() as pilot:
        await pilot.pause()

        [cell] = replayed.query(ToolCallCell)
        assert cell.status is ExecutionStatus.COMPLETED
        assert cell.result_text == "42.0"


async def test_deny_rejects_the_call_and_the_model_reacts(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(*tool_turn("Understood, I won't compute it.")),
        workspace=tmp_path, session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "what is 6 times 7?")
        await wait_until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await pilot.press("2")  # Deny
        await wait_until(pilot, lambda: idle_again(app))

        [cell] = app.query(ToolCallCell)
        assert cell.status is ExecutionStatus.REJECTED
        assert cell.is_error is True
        assert [c.text for c in app.query(AssistantCell)] == [
            "Understood, I won't compute it.",
        ]


async def test_abandon_cancels_the_turn(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(*tool_turn("never sent")[:1]),
        workspace=tmp_path, session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "what is 6 times 7?")
        await wait_until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await pilot.press("3")  # Abandon turn
        await wait_until(pilot, lambda: idle_again(app))

        [cell] = app.query(ToolCallCell)
        assert cell.status is ExecutionStatus.CANCELLED
        assert "turn abandoned — flushing" in [
            cell.text for cell in app.query(NoticeCell)
        ]
        assert app.runner.status is ConversationStatus.IDLE


async def test_escape_on_the_modal_abandons(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(*tool_turn("never sent")[:1]),
        workspace=tmp_path, session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "what is 6 times 7?")
        await wait_until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await pilot.press("escape")
        await wait_until(pilot, lambda: idle_again(app))

        assert app.runner.status is ConversationStatus.IDLE
        assert "turn abandoned — flushing" in [
            cell.text for cell in app.query(NoticeCell)
        ]
