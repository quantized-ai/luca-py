"""Slash command behaviour, driven headless through the Pilot.

`/model` drills down (pick a provider, then one of its models); `/reasoning`
opens a single picker. Both switch directly when given an argument. Picker
tests mirror the approval-modal pattern: submit the command, wait for the
`PickerScreen`, drive it with arrow/enter/esc, then assert the whole
`LLMConfig`.
"""

from textual.widgets import Input

from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.cells import AssistantCell, NoticeCell, UserCell
from luca.agent.contrib.tui.commands import COMMANDS
from luca.agent.contrib.tui.screens import PickerScreen
from luca.agent.contrib.tui.wiring import RECOMMENDED_MODELS
from luca.agent.core.models import LLMConfig
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text

from .helpers import fresh_session, idle_again, submit, wait_until


def scripted(*responses) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return provider


def _notices(app) -> list[str]:
    return [cell.text for cell in app.query(NoticeCell)]


def _config(app) -> LLMConfig:
    return app.runner.session.session_config.llm_config


async def _picker(pilot, app):
    await wait_until(pilot, lambda: isinstance(app.screen, PickerScreen))


async def _picker_titled(pilot, app, needle):
    await wait_until(
        pilot,
        lambda: isinstance(app.screen, PickerScreen) and needle in app.screen._title,
    )


async def _picker_closed(pilot, app):
    await wait_until(pilot, lambda: not isinstance(app.screen, PickerScreen))


# ── /help ────────────────────────────────────────────────────────────────────


async def test_help_lists_every_command(tmp_path):
    app = AgentApp(fresh_session(), workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/help")
        await pilot.pause()

        text = _notices(app)[-1]
        for command in COMMANDS:
            assert f"/{command.name}" in text


# ── /model ───────────────────────────────────────────────────────────────────


async def test_model_drills_down_provider_then_model(tmp_path):
    # provider step highlights index 0 (anthropic); model step index 0.
    provider = list(RECOMMENDED_MODELS)[0]
    model = RECOMMENDED_MODELS[provider][0]
    app = AgentApp(fresh_session(), workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/model")
        await _picker_titled(pilot, app, "provider")
        await pilot.press("enter")
        await _picker_titled(pilot, app, "model")
        await pilot.press("enter")
        await _picker_closed(pilot, app)

        assert _config(app) == LLMConfig(model=model, provider=provider)


async def test_model_arrows_through_both_steps(tmp_path):
    # provider index 1, then model index 1 within that provider.
    provider = list(RECOMMENDED_MODELS)[1]
    model = RECOMMENDED_MODELS[provider][1]
    app = AgentApp(fresh_session(), workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/model")
        await _picker_titled(pilot, app, "provider")
        await pilot.press("down")
        await pilot.press("enter")
        await _picker_titled(pilot, app, "model")
        await pilot.press("down")
        await pilot.press("enter")
        await _picker_closed(pilot, app)

        assert _config(app) == LLMConfig(model=model, provider=provider)


async def test_model_back_returns_to_provider_step_then_reselects(tmp_path):
    # "end" jumps to the last model-step option, which is "← Back to providers".
    # Choosing it lands back on the provider step; then pick provider 1, model 0.
    provider = list(RECOMMENDED_MODELS)[1]
    model = RECOMMENDED_MODELS[provider][0]
    app = AgentApp(fresh_session(), workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/model")
        await _picker_titled(pilot, app, "provider")
        await pilot.press("enter")  # provider 0
        await _picker_titled(pilot, app, "model")
        await pilot.press("end")  # ← Back to providers (last)
        await pilot.press("enter")
        await _picker_titled(pilot, app, "provider")
        await pilot.press("down")  # provider 1
        await pilot.press("enter")
        await _picker_titled(pilot, app, "model")
        await pilot.press("enter")  # model 0
        await _picker_closed(pilot, app)

        assert _config(app) == LLMConfig(model=model, provider=provider)


async def test_model_esc_at_provider_step_changes_nothing(tmp_path):
    session = fresh_session()
    before = session.session_config.llm_config.model_copy(deep=True)
    app = AgentApp(session, workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/model")
        await _picker_titled(pilot, app, "provider")
        await pilot.press("escape")
        await _picker_closed(pilot, app)

        assert _config(app) == before


async def test_model_esc_at_model_step_changes_nothing(tmp_path):
    # The provider is only applied together with the model, so bailing at the
    # model step leaves the provider untouched too.
    session = fresh_session()
    before = session.session_config.llm_config.model_copy(deep=True)
    app = AgentApp(session, workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/model")
        await _picker_titled(pilot, app, "provider")
        await pilot.press("enter")
        await _picker_titled(pilot, app, "model")
        await pilot.press("escape")
        await _picker_closed(pilot, app)

        assert _config(app) == before


async def test_model_arg_switches_directly_without_a_picker(tmp_path):
    app = AgentApp(fresh_session(), workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/model anthropic:claude-sonnet-5")
        await pilot.pause()

        assert not isinstance(app.screen, PickerScreen)
        assert _config(app) == LLMConfig(model="claude-sonnet-5", provider="anthropic")


async def test_model_arg_with_a_bare_id_keeps_the_provider(tmp_path):
    app = AgentApp(fresh_session(), workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/model gpt-5.4")
        await pilot.pause()

        assert _config(app) == LLMConfig(model="gpt-5.4", provider="faux")


async def test_model_arg_rejects_an_empty_half(tmp_path):
    session = fresh_session()
    before = session.session_config.llm_config.model_copy(deep=True)
    app = AgentApp(session, workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/model openai:")
        await pilot.pause()

        assert _config(app) == before
        assert "invalid model spec" in _notices(app)[-1]


# ── /reasoning ───────────────────────────────────────────────────────────────


async def test_reasoning_picker_sets_the_level(tmp_path):
    # current is None -> "provider-default" (index 0) is highlighted; down -> "none".
    app = AgentApp(fresh_session(), workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/reasoning")
        await _picker(pilot, app)
        await pilot.press("down")
        await pilot.press("enter")
        await wait_until(pilot, lambda: not isinstance(app.screen, PickerScreen))

        assert _config(app) == LLMConfig(
            model="fake-model", provider="faux", reasoning="none",
        )


async def test_reasoning_arg_rejects_an_unknown_level_and_leaves_config_untouched(tmp_path):
    session = fresh_session()
    before = session.session_config.llm_config.model_copy(deep=True)
    app = AgentApp(session, workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/reasoning bogus")
        await pilot.pause()

        assert _config(app) == before
        assert "unknown reasoning level" in _notices(app)[-1]


# ── /new, /quit, dispatch ────────────────────────────────────────────────────


async def test_new_starts_a_fresh_session_keeping_the_model_and_clearing_history(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message([faux_text("hi back")])),
        workspace=tmp_path, session_dir=tmp_path,
    )
    async with app.run_test() as pilot:
        await submit(pilot, "hello")
        await wait_until(pilot, lambda: idle_again(app))
        old_id = app.runner.session.id

        await submit(pilot, "/new")
        await pilot.pause()

        assert app.runner.session.id != old_id
        assert _config(app) == LLMConfig(model="fake-model", provider="faux")
        assert len(app.query(UserCell)) == 0
        assert len(app.query(AssistantCell)) == 0
        assert (tmp_path / f"{old_id}.json").exists()


async def test_new_preserves_the_runtime_config(tmp_path):
    from luca.agent.core.models import RuntimeConfig

    runtime = RuntimeConfig(hard_max_steps=3)
    app = AgentApp(fresh_session(runtime), workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/new")
        await pilot.pause()

        assert app.runner.session.session_config.runtime_config == runtime


async def test_an_unknown_command_is_sent_as_a_normal_message(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message([faux_text("ok")])),
        workspace=tmp_path, session_dir=tmp_path,
    )
    async with app.run_test() as pilot:
        await submit(pilot, "/nope")
        await wait_until(pilot, lambda: idle_again(app))

        assert [cell.text for cell in app.query(UserCell)] == ["/nope"]


async def test_a_path_like_message_is_not_swallowed(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message([faux_text("ok")])),
        workspace=tmp_path, session_dir=tmp_path,
    )
    async with app.run_test() as pilot:
        await submit(pilot, "/etc/hosts is missing an entry")
        await wait_until(pilot, lambda: idle_again(app))

        assert [cell.text for cell in app.query(UserCell)] == [
            "/etc/hosts is missing an entry",
        ]


async def test_quit_saves_and_exits(tmp_path):
    session = fresh_session()
    app = AgentApp(session, workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/quit")

    assert (tmp_path / f"{session.id}.json").exists()
