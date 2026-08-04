"""Mid-turn input: the composer stays enabled while the agent works, a submit
posts into the open turn and renders immediately — subagents running included,
where the post wakes the parked parent and the turn answers it — and a slash
command mid-run is refused with a notice, keeping the draft. Input is never
silently discarded."""

from luca.agent.contrib.subagents import SPAWN_TOOL_NAME
from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.blocks import NoticeLine, UserTurn
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


def make_app(session, provider, tmp_path) -> AgentApp:
    return AgentApp(
        session,
        provider=provider,
        workspace=tmp_path,
        session_dir=tmp_path,
        skills=False,
        instructions=False,
    )


def spawn(description: str, prompt: str, *, call_id: str, task_id: str):
    return faux_tool_call(
        SPAWN_TOOL_NAME,
        {"prompt": prompt, "description": description, "task_id": task_id},
        id=call_id,
    )


async def test_typing_mid_run_posts_into_the_open_turn(tmp_path):
    provider = scripted(faux_assistant_message([faux_hang()]))
    app = make_app(fresh_session(), provider, tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "hang please")
        await wait_until(pilot, lambda: app.current_run is not None)
        prompt = app.query_one(PromptInput)
        assert not prompt.disabled  # the composer stays enabled while busy

        await submit(pilot, "and steer this way")
        await pilot.pause()

        # the post landed INSIDE the open turn, rendered immediately, and the
        # input was cleared for the next message
        session = app.runner.session
        nodes = session.conversations[session.main_conversation_id].nodes
        posted = [entry.parts[0].text for n in nodes if isinstance(entry := session.entries[n], UserMessage)]
        assert posted == ["hang please", "and steer this way"]
        assert [turn.text for turn in app.query(UserTurn)] == ["hang please", "and steer this way"]
        assert prompt.text == ""

        await pilot.press("escape")
        await wait_until(pilot, lambda: idle_again(app))


async def test_a_post_while_subagents_run_wakes_the_parent_and_is_answered(tmp_path):
    provider = scripted(
        faux_assistant_message(
            [
                faux_text("Delegating."),
                spawn("read alpha.txt", "Read alpha.txt.", call_id="c1", task_id="t1"),
            ]
        ),
        faux_assistant_message([faux_hang()]),  # the child hangs mid-work
        # the accepted post wakes the parked parent for one round
        faux_assistant_message([faux_text("Noted — still waiting on the task.")]),
    )
    app = make_app(fresh_session(SUBAGENTS), provider, tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "go")
        await wait_until(pilot, lambda: len(app._tasks) == 1)

        await submit(pilot, "hurry up")
        await pilot.pause()

        # accepted: the message landed in the parent's open turn, rendered
        # immediately, and the input cleared for the next message
        posted = [e.parts[0].text for e in app.runner.session.entries.values() if isinstance(e, UserMessage)]
        assert "hurry up" in posted
        assert [turn.text for turn in app.query(UserTurn)] == ["go", "hurry up"]
        assert app.query_one(PromptInput).text == ""

        # …and the woken parent's next call SAW it, while the child kept going
        await wait_until(pilot, lambda: len(provider.requests) >= 3)
        assert provider.requests[-1].messages[-1].content[0].text == "hurry up"

        await pilot.press("escape")  # unwind the hung child; everything settles
        await wait_until(pilot, lambda: idle_again(app))


async def test_a_command_submitted_mid_run_is_refused_and_the_draft_kept(tmp_path):
    provider = scripted(faux_assistant_message([faux_hang()]))
    app = make_app(fresh_session(), provider, tmp_path)

    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "hang please")
        await wait_until(pilot, lambda: app.current_run is not None)

        await submit(pilot, "/model other:model")
        await pilot.pause()

        # commands stay idle-only: nothing dispatched, the refusal is stated,
        # and the draft is preserved
        assert app.runner.session.session_config.llm_config.model == "fake-model"
        assert [(notice.text, notice.error) for notice in app.query(NoticeLine)] == [
            ("commands are available when the agent is idle", True),
        ]
        assert app.query_one(PromptInput).text == "/model other:model"

        await pilot.press("escape")
        await wait_until(pilot, lambda: idle_again(app))
