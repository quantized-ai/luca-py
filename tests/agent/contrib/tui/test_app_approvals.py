"""The approval gate through the inline prompt: approve, deny, cancel turn.

The demo math tools are resourceless, so every gated call presents the
synthesized three-option prompt (Approve once / Deny / Cancel turn — no
"Approve always" without a suggested widening). The prompt replaces the
composer in `#dock`; answers are picked by digit key, exactly as a user
would.
"""

from luca.agent.contrib.app.sessions import load_session
from luca.agent.contrib.tui import AgentApp, state as vm
from luca.agent.contrib.tui.blocks import AssistantText, NoticeLine, ToolBlockView
from luca.agent.contrib.tui.shells import ApprovalPromptView
from luca.agent.core.models import ConversationStatus, ExecutionStatus, ToolExecution
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
)

from .helpers import fresh_session, idle_again, submit, wait_until

GATED_PROMPT = vm.ApprovalState(
    question="Run multiply?",
    options=[
        vm.ApprovalOption(label="Approve once"),
        vm.ApprovalOption(label="Deny — tell Luca what to do instead"),
        vm.ApprovalOption(label="Cancel turn", key_hint="esc"),
    ],
    selected=0,
)


def scripted(*responses) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return provider


def make_app(session, provider, tmp_path) -> AgentApp:
    return AgentApp(
        session,
        provider=provider,
        workspace=tmp_path,
        session_dir=tmp_path,
        skills=False,
        instructions=False,
    )


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
    app = make_app(session, scripted(*tool_turn("It is 42.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "what is 6 times 7?")
        await wait_until(pilot, lambda: bool(app.query(ApprovalPromptView)))

        [prompt] = app.query(ApprovalPromptView)
        assert prompt.model == GATED_PROMPT

        await pilot.press("1")  # Approve once
        await wait_until(pilot, lambda: idle_again(app))

        assert [view.model for view in app.query(ToolBlockView)] == [
            vm.ToolBlock(
                tool="multiply",
                arg="a=6, b=7",
                status="ok",
                result=vm.ToolResult(summary="42.0"),
            ),
        ]
        assert [text.model for text in app.query(AssistantText)] == [
            vm.TextBlock(text="It is 42.", streaming=False),
        ]

    # the reloaded session replays the finished tool round
    replayed = make_app(load_session(session.id, tmp_path), scripted(), tmp_path)
    async with replayed.run_test(size=(105, 35)) as pilot:
        await pilot.pause()

        assert [view.model for view in replayed.query(ToolBlockView)] == [
            vm.ToolBlock(
                tool="multiply",
                arg="a=6, b=7",
                status="ok",
                result=vm.ToolResult(summary="42.0"),
            ),
        ]


async def test_deny_rejects_the_call_and_the_model_reacts(tmp_path):
    app = make_app(fresh_session(), scripted(*tool_turn("Understood, I won't compute it.")), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "what is 6 times 7?")
        await wait_until(pilot, lambda: bool(app.query(ApprovalPromptView)))
        await pilot.press("2")  # Deny — tell Luca what to do instead
        await wait_until(pilot, lambda: idle_again(app))

        assert [view.model for view in app.query(ToolBlockView)] == [
            vm.ToolBlock(
                tool="multiply",
                arg="a=6, b=7",
                status="denied",
                result=vm.ToolResult(summary="denied", note="by you"),
            ),
        ]
        [execution] = [e for e in app.runner.session.entries.values() if isinstance(e, ToolExecution)]
        assert execution.status is ExecutionStatus.REJECTED
        assert [text.model for text in app.query(AssistantText)] == [
            vm.TextBlock(text="Understood, I won't compute it.", streaming=False),
        ]


async def test_cancel_turn_by_digit_cancels_the_turn(tmp_path):
    app = make_app(fresh_session(), scripted(*tool_turn("never sent")[:1]), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "what is 6 times 7?")
        await wait_until(pilot, lambda: bool(app.query(ApprovalPromptView)))
        await pilot.press("3")  # Cancel turn
        await wait_until(pilot, lambda: idle_again(app))

        assert "turn cancelled — winding down" in [notice.text for notice in app.query(NoticeLine)]
        [execution] = [e for e in app.runner.session.entries.values() if isinstance(e, ToolExecution)]
        assert execution.status is ExecutionStatus.CANCELLED
        assert app.runner.status is ConversationStatus.IDLE


async def test_escape_on_the_prompt_cancels_the_turn(tmp_path):
    app = make_app(fresh_session(), scripted(*tool_turn("never sent")[:1]), tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "what is 6 times 7?")
        await wait_until(pilot, lambda: bool(app.query(ApprovalPromptView)))
        await pilot.press("escape")  # esc = the last option, Cancel turn
        await wait_until(pilot, lambda: idle_again(app))

        assert app.runner.status is ConversationStatus.IDLE
        assert "turn cancelled — winding down" in [notice.text for notice in app.query(NoticeLine)]
