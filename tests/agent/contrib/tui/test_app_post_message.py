"""Mid-turn input: the prompt stays enabled while the agent works, a submit
posts into the open turn and renders immediately, and every rejection —
subagents active, a slash command mid-turn — keeps the draft in the input.
Input is never silently discarded."""

from luca.agent.contrib.subagents import SPAWN_TOOL_NAME
from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.cells import SubagentPanel, UserCell
from luca.agent.contrib.tui.prompt import PromptInput
from luca.agent.core.models import RuntimeConfig, UserMessage
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_hang,
    faux_text,
    faux_tool_call,
)

from .helpers import fresh_session, idle_again, submit, wait_until

SUBAGENTS = RuntimeConfig(subagents_enabled=True, subagents_max_depth=1)


def scripted(*responses) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return provider


def spawn(description: str, prompt: str, *, call_id: str, task_id: str):
    return faux_tool_call(
        SPAWN_TOOL_NAME,
        {"prompt": prompt, "description": description, "task_id": task_id},
        id=call_id,
    )


async def test_typing_mid_run_posts_into_the_open_turn(tmp_path):
    provider = FauxProvider()
    provider.set_responses([faux_assistant_message([faux_hang()])])
    app = AgentApp(
        fresh_session(),
        provider=provider,
        workspace=tmp_path,
        session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "hang please")
        await wait_until(pilot, lambda: app.current_run is not None)
        prompt = app.query_one("#prompt", PromptInput)
        assert not prompt.disabled  # the prompt stays enabled while busy

        await submit(pilot, "and steer this way")
        await pilot.pause()

        # the post landed INSIDE the open turn, rendered immediately, and the
        # input was cleared for the next message
        session = app.runner.session
        nodes = session.conversations[session.main_conversation_id].nodes
        posted = [entry.parts[0].text for n in nodes if isinstance(entry := session.entries[n], UserMessage)]
        assert posted == ["hang please", "and steer this way"]
        assert [cell.text for cell in app.query(UserCell)] == ["hang please", "and steer this way"]
        assert prompt.text == ""

        await pilot.press("escape")
        await wait_until(pilot, lambda: idle_again(app))


async def test_a_post_while_subagents_run_is_rejected_and_the_draft_kept(tmp_path):
    app = AgentApp(
        fresh_session(SUBAGENTS),
        provider=scripted(
            faux_assistant_message(
                [
                    faux_text("Delegating."),
                    spawn("read alpha.txt", "Read alpha.txt.", call_id="c1", task_id="t1"),
                ]
            ),
            faux_assistant_message([faux_hang()]),  # the child hangs mid-work
        ),
        workspace=tmp_path,
        session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "go")
        await wait_until(pilot, lambda: len(list(app.query(SubagentPanel))) == 1)

        await submit(pilot, "hurry up")
        await pilot.pause()

        # rejected: no new message anywhere, the draft still in the input
        session = app.runner.session
        posted = [entry.parts[0].text for entry in session.entries.values() if isinstance(entry, UserMessage)]
        assert "hurry up" not in posted
        assert [cell.text for cell in app.query(UserCell)] == ["go"]
        assert app.query_one("#prompt", PromptInput).text == "hurry up"

        await pilot.press("escape")  # unwind the hung child; everything settles
        await wait_until(pilot, lambda: idle_again(app))


async def test_a_command_submitted_mid_run_is_refused_and_the_draft_kept(tmp_path):
    provider = FauxProvider()
    provider.set_responses([faux_assistant_message([faux_hang()])])
    app = AgentApp(
        fresh_session(),
        provider=provider,
        workspace=tmp_path,
        session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "hang please")
        await wait_until(pilot, lambda: app.current_run is not None)

        await submit(pilot, "/model other:model")
        await pilot.pause()

        # commands stay idle-only: nothing dispatched, draft preserved
        assert app.runner.session.session_config.llm_config.model == "fake-model"
        assert app.query_one("#prompt", PromptInput).text == "/model other:model"

        await pilot.press("escape")
        await wait_until(pilot, lambda: idle_again(app))
