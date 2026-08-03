"""Slash command behaviour, driven headless through the Pilot.

`/model` drills down (pick a provider, then one of its models); `/reasoning`
opens a single picker; `/theme` opens Textual's native theme palette. Model and
reasoning both switch directly when given an argument. Picker tests mirror the
approval-modal pattern: submit the command, wait for the `PickerScreen`, drive
it with arrow/enter/esc, then assert the whole `LLMConfig`.
"""

import os

from textual.command import CommandPalette

from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.cells import (
    AssistantCell,
    CompactionCell,
    NoticeCell,
    UserCell,
)
from luca.agent.contrib.tui.screens import PickerScreen
from luca.agent.contrib.tui.sessions import save_session
from luca.agent.contrib.tui.wiring import RECOMMENDED_MODELS
from luca.agent.core.compaction import CompactionPlan
from luca.agent.core.models import LLMConfig, TextContent
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text
from tests.agent.scenarios import (
    GATED_SESSION,
    FakeContextManager,
    main_conversation,
)

from .helpers import fresh_session, idle_again, submit, wait_until, with_user_message

# The block `/help` renders, spelled out rather than re-derived from COMMANDS:
# a command added, renamed, or re-summarized shows up here as a diff.
HELP_TEXT = "\n".join(
    (
        "/help                    show this help",
        "/model [provider:model]  pick a provider then a model",
        "/reasoning [level]       pick or set the reasoning level",
        "/theme                   choose the Textual theme",
        "/compact                 summarize the history and continue",
        "/new                     save and start a fresh conversation",
        "/resume                  switch to another session in this project",
        "/quit                    save and exit",
    ),
)


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

        assert _notices(app) == [HELP_TEXT]


async def test_theme_opens_textual_native_theme_palette(tmp_path):
    app = AgentApp(fresh_session(), workspace=tmp_path, session_dir=tmp_path)

    async with app.run_test() as pilot:
        await submit(pilot, "/theme")
        await wait_until(pilot, lambda: CommandPalette.is_open(app))

        assert type(app.screen) is CommandPalette


# ── /model ───────────────────────────────────────────────────────────────────


async def test_model_drills_down_provider_then_model(tmp_path):
    # provider step highlights index 0 (anthropic); model step index 0.
    provider = next(iter(RECOMMENDED_MODELS))
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
        assert _notices(app) == ["invalid model spec 'openai:'; use provider:model"]


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
            model="fake-model",
            provider="faux",
            reasoning="none",
        )


async def test_reasoning_arg_rejects_an_unknown_level_and_leaves_config_untouched(tmp_path):
    session = fresh_session()
    before = session.session_config.llm_config.model_copy(deep=True)
    app = AgentApp(session, workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/reasoning bogus")
        await pilot.pause()

        assert _config(app) == before
        assert _notices(app) == [
            "unknown reasoning level 'bogus'. Valid: provider-default, none, minimal, low, medium, high, xhigh",
        ]


# ── /new, /quit, dispatch ────────────────────────────────────────────────────


async def test_new_starts_a_fresh_session_keeping_the_model_and_clearing_history(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message([faux_text("hi back")])),
        workspace=tmp_path,
        session_dir=tmp_path,
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


# ── /resume ──────────────────────────────────────────────────────────────────


async def test_resume_switches_to_the_picked_session_and_replays_it(tmp_path):
    stored = with_user_message("the older conversation")
    save_session(stored, tmp_path)
    app = AgentApp(fresh_session(), workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/resume")
        await _picker_titled(pilot, app, "Resume")
        await pilot.press("enter")
        await _picker_closed(pilot, app)

        assert app.runner.session.id == stored.id
        # the transcript, not just the id: a swap that left the pane blank
        # would pass an id-only assertion while showing the user nothing
        assert [cell.text for cell in app.query(UserCell)] == ["the older conversation"]


async def test_resume_lists_the_sessions_newest_first_with_their_first_message(tmp_path):
    older = with_user_message("older")
    newer = with_user_message("newer")
    save_session(older, tmp_path)
    save_session(newer, tmp_path)
    os.utime(tmp_path / f"{older.id}.json", (1, 1))
    app = AgentApp(fresh_session(), workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/resume")
        await _picker_titled(pilot, app, "Resume")

        assert app.screen._options == [newer.id, older.id]
        assert [row.split("  ", 1)[1] for row in app.screen._labels] == [
            "newer  (1 turns)",
            "older  (1 turns)",
        ]


async def test_the_session_being_left_is_marked_current_in_the_picker(tmp_path):
    # the row reads as the label, but the value behind it is still the id
    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message([faux_text("hi back")])),
        workspace=tmp_path,
        session_dir=tmp_path,
    )
    async with app.run_test() as pilot:
        await submit(pilot, "hello")
        await wait_until(pilot, lambda: idle_again(app))

        await submit(pilot, "/resume")
        await _picker_titled(pilot, app, "Resume")

        assert app.screen._options == [app.runner.session.id]
        assert app.screen._labels[0].endswith("hello  (1 turns)")


async def test_escaping_the_resume_picker_changes_nothing(tmp_path):
    save_session(with_user_message("stored"), tmp_path)
    app = AgentApp(fresh_session(), workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        current = app.runner.session.id
        await submit(pilot, "/resume")
        await _picker_titled(pilot, app, "Resume")
        await pilot.press("escape")
        await _picker_closed(pilot, app)

        assert app.runner.session.id == current


async def test_resume_with_nothing_stored_notices_instead_of_opening_a_picker(tmp_path):
    app = AgentApp(fresh_session(), workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/resume")
        await pilot.pause()

        assert not isinstance(app.screen, PickerScreen)
        assert _notices(app) == ["no saved sessions for this project yet"]


async def test_resume_does_not_write_the_untouched_session_it_opened_over(tmp_path):
    # `--resume` opens over a fresh empty session; saving that would drop a
    # blank file into the store on every launch
    save_session(with_user_message("stored"), tmp_path)
    app = AgentApp(fresh_session(), workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        empty_id = app.runner.session.id
        await submit(pilot, "/resume")
        await _picker_titled(pilot, app, "Resume")
        await pilot.press("escape")
        await _picker_closed(pilot, app)

        assert not (tmp_path / f"{empty_id}.json").exists()


async def test_resume_saves_the_conversation_it_is_leaving(tmp_path):
    save_session(with_user_message("stored"), tmp_path)
    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message([faux_text("hi back")])),
        workspace=tmp_path,
        session_dir=tmp_path,
    )
    async with app.run_test() as pilot:
        await submit(pilot, "hello")
        await wait_until(pilot, lambda: idle_again(app))
        leaving = app.runner.session.id

        await submit(pilot, "/resume")
        await _picker_titled(pilot, app, "Resume")
        await pilot.press("escape")
        await _picker_closed(pilot, app)

        assert (tmp_path / f"{leaving}.json").exists()


async def test_resume_picks_up_a_session_that_was_left_mid_turn(tmp_path, monkeypatch):
    # quitting at an approval modal persists a non-idle session. `--conversation`
    # drives it on mount; `/resume` reaching the same file must too, or the turn
    # sits frozen with no modal and no way forward but sending a new message.
    started: list[str] = []
    monkeypatch.setattr(AgentApp, "_start_drive", lambda self: started.append("drive"))
    gated = GATED_SESSION.model_copy(deep=True)
    save_session(gated, tmp_path)
    app = AgentApp(fresh_session(), workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/resume")
        await _picker_titled(pilot, app, "Resume")
        await pilot.press("enter")
        await _picker_closed(pilot, app)

        assert app.runner.session.id == gated.id
        assert app.runner.idle() is False
        assert started == ["drive"]


async def test_resume_into_an_idle_session_just_focuses_the_prompt(tmp_path, monkeypatch):
    # a stored session with nothing pending — `with_user_message` would leave a
    # trailing unanswered message, which the runner rightly wants to drive
    started: list[str] = []
    monkeypatch.setattr(AgentApp, "_start_drive", lambda self: started.append("drive"))
    save_session(fresh_session(), tmp_path)
    app = AgentApp(fresh_session(), workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/resume")
        await _picker_titled(pilot, app, "Resume")
        await pilot.press("enter")
        await _picker_closed(pilot, app)

        assert started == []


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
        workspace=tmp_path,
        session_dir=tmp_path,
    )
    async with app.run_test() as pilot:
        await submit(pilot, "/nope")
        await wait_until(pilot, lambda: idle_again(app))

        assert [cell.text for cell in app.query(UserCell)] == ["/nope"]


async def test_a_path_like_message_is_not_swallowed(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message([faux_text("ok")])),
        workspace=tmp_path,
        session_dir=tmp_path,
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


# ── /compact ─────────────────────────────────────────────────────────────────


def _fold_everything(session, nodes, entry):
    return CompactionPlan(
        entry=entry.model_copy(
            update={"parts": [TextContent(text="the story so far")]},
        ),
        nodes=[entry.id],
    )


async def test_compact_schedules_a_compaction_and_drives_it(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message([faux_text("ok")])),
        workspace=tmp_path,
        session_dir=tmp_path,
        context_manager=FakeContextManager(plan=_fold_everything),
    )
    async with app.run_test() as pilot:
        await submit(pilot, "hello")
        await wait_until(pilot, lambda: idle_again(app))

        await submit(pilot, "/compact")
        await wait_until(pilot, lambda: idle_again(app))

        assert [cell.text for cell in app.query(CompactionCell)] == [
            "the story so far",
        ]
        # the bracket stayed behind; only the summary carried over
        installed = main_conversation(app.runner.session)
        archived = app.runner.session.conversations[installed.previous_conversation_id]
        assert installed.nodes == [archived.nodes[-2]]


async def test_compact_with_a_manager_that_cannot_compact_reports_a_turn_failure(tmp_path):
    # No pre-check is possible — a `ContextManager` always exists. /compact
    # schedules, the bracket is written, and core's base `compact()` raises on
    # the drive, which the app reports like any other turn failure.
    app = AgentApp(fresh_session(), workspace=tmp_path, session_dir=tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/compact")
        await wait_until(pilot, lambda: idle_again(app))

        assert _notices(app) == [
            "compacting the conversation…",
            "compaction errored",
            "turn failed: ",
        ]
