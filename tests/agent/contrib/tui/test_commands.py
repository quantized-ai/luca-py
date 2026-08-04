"""Slash commands: the registry, `dispatch`, the palette, and the live
modal-state builders.

Pickers are inline overlays now — `OverlayListView` replaces the composer in
`#dock` (never a stock modal) — and `/model` drills down provider → model
through two menu states of the same overlay. The sessions and settings
screens are full-frame modals whose states come from `build_sessions_state`
and `build_settings_state`, and `adjust_setting` answers the settings
screen's `← →` intents.
"""

import os
import time

from luca.agent.contrib.tui import state as vm
from luca.agent.contrib.tui.app import AgentApp
from luca.agent.contrib.tui.blocks import ListBlockView, NoticeLine, UserTurn
from luca.agent.contrib.tui.commands import (
    COMMANDS,
    build_sessions_state,
    build_settings_state,
    dispatch,
    palette_rows,
    run_palette_choice,
)
from luca.agent.contrib.tui.modals import SettingsScreen
from luca.agent.contrib.tui.prompt import PromptInput
from luca.agent.contrib.tui.sessions import save_session
from luca.agent.contrib.tui.shells import OverlayListView
from luca.agent.core.models import LLMConfig, RuntimeConfig
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text

from .helpers import fresh_session, with_user_message


def scripted(*responses) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return provider


def agent_app(tmp_path, provider=None) -> AgentApp:
    return AgentApp(
        fresh_session(),
        provider=provider,
        workspace=tmp_path,
        session_dir=tmp_path,
        skills=False,
        instructions=False,
    )


def _notices(app) -> list[tuple[str, bool]]:
    return [(line.text, line.error) for line in app.query(NoticeLine)]


def _config(app) -> LLMConfig:
    return app.runner.session.session_config.llm_config


def _idle_again(app) -> bool:
    return app.runner.idle() and not app._driving


async def submit(pilot, text: str) -> None:
    prompt = pilot.app.query_one(PromptInput)
    prompt.load_text(text)
    prompt.focus()
    await pilot.pause()
    await pilot.press("enter")


async def wait_until(pilot, condition, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            await pilot.pause()
            return
        await pilot.pause(0.02)
    raise AssertionError("condition not met within timeout")


# ── the registry ─────────────────────────────────────────────────────────────


def test_the_registry_lists_the_fourteen_commands():
    # Spelled out rather than re-derived: a command added, renamed or
    # re-summarized shows up here as a diff.
    assert [(c.name, c.usage, c.summary, c.insert) for c in COMMANDS] == [
        ("skill", "[name]", "load a skill into this session", True),
        ("session", "", "resume, fork or list sessions", False),
        ("context", "", "add files to the working set", False),
        ("cost", "", "token and spend breakdown", False),
        ("settings", "", "model, theme, permissions", False),
        ("clear", "", "start a fresh session", False),
        ("model", "[provider:model]", "pick a provider then a model", False),
        ("reasoning", "[level]", "pick or set the reasoning level", False),
        ("theme", "", "choose the color theme", False),
        ("compact", "", "summarize the history and continue", False),
        ("resume", "", "switch to another session in this project", False),
        ("new", "", "save and start a fresh conversation", False),
        ("help", "", "show every command", False),
        ("quit", "", "save and exit", False),
    ]


# ── dispatch ─────────────────────────────────────────────────────────────────


async def test_dispatch_returns_false_for_an_unknown_or_empty_command(tmp_path):
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)):
        assert await dispatch(app, "/nope") is False
        assert await dispatch(app, "/") is False


async def test_dispatch_splits_the_name_from_the_arg_and_strips_it(tmp_path):
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)):
        assert await dispatch(app, "/reasoning   high  ") is True

        assert _config(app) == LLMConfig(model="fake-model", provider="faux", reasoning="high")


async def test_an_unknown_command_is_sent_as_a_normal_message(tmp_path):
    app = agent_app(tmp_path, provider=scripted(faux_assistant_message([faux_text("ok")])))
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/nope")
        await wait_until(pilot, lambda: _idle_again(app))

        assert [turn.text for turn in app.query(UserTurn)] == ["/nope"]


async def test_a_path_like_message_is_not_swallowed(tmp_path):
    app = agent_app(tmp_path, provider=scripted(faux_assistant_message([faux_text("ok")])))
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/etc/hosts is missing an entry")
        await wait_until(pilot, lambda: _idle_again(app))

        assert [turn.text for turn in app.query(UserTurn)] == ["/etc/hosts is missing an entry"]


# ── /model ───────────────────────────────────────────────────────────────────


async def test_model_arg_switches_directly_without_a_menu(tmp_path):
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/model anthropic:claude-sonnet-5")
        await pilot.pause()

        assert not app.query(OverlayListView)
        assert _config(app) == LLMConfig(model="claude-sonnet-5", provider="anthropic")
        assert _notices(app) == [("model set to claude-sonnet-5", False)]


async def test_model_arg_with_a_bare_id_keeps_the_provider(tmp_path):
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/model gpt-5.4")
        await pilot.pause()

        assert _config(app) == LLMConfig(model="gpt-5.4", provider="faux")


async def test_model_arg_rejects_an_empty_half(tmp_path):
    session = fresh_session()
    before = session.session_config.llm_config.model_copy(deep=True)
    app = AgentApp(session, workspace=tmp_path, session_dir=tmp_path, skills=False, instructions=False)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/model openai:")
        await pilot.pause()

        assert _config(app) == before
        assert _notices(app) == [("invalid model spec 'openai:'; use provider:model", True)]


async def test_model_with_no_arg_drills_down_provider_then_model(tmp_path):
    # menu 1 lists providers (anthropic first); menu 2 that provider's models.
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/model")
        await wait_until(pilot, lambda: bool(app.query(OverlayListView)))
        await pilot.press("enter")  # provider 0: anthropic
        await pilot.pause()
        await pilot.press("enter")  # model 0 of anthropic
        await wait_until(pilot, lambda: not app.query(OverlayListView))

        assert _config(app) == LLMConfig(model="claude-opus-4-8", provider="anthropic")
        assert _notices(app) == [("model set to anthropic:claude-opus-4-8", False)]


async def test_model_menu_arrows_pick_by_index(tmp_path):
    # provider index 1 (openrouter), then its model index 1.
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/model")
        await wait_until(pilot, lambda: bool(app.query(OverlayListView)))
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("down")
        await pilot.press("enter")
        await wait_until(pilot, lambda: not app.query(OverlayListView))

        assert _config(app) == LLMConfig(model="openai/gpt-5.4", provider="openrouter")


async def test_escaping_the_model_menu_changes_nothing(tmp_path):
    session = fresh_session()
    before = session.session_config.llm_config.model_copy(deep=True)
    app = AgentApp(session, workspace=tmp_path, session_dir=tmp_path, skills=False, instructions=False)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/model")
        await wait_until(pilot, lambda: bool(app.query(OverlayListView)))
        await pilot.press("escape")
        await wait_until(pilot, lambda: not app.query(OverlayListView))

        assert _config(app) == before


# ── /reasoning ───────────────────────────────────────────────────────────────


async def test_reasoning_menu_sets_the_level(tmp_path):
    # the menu lists the levels in order; down picks index 1 ("none")
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/reasoning")
        await wait_until(pilot, lambda: bool(app.query(OverlayListView)))
        await pilot.press("down")
        await pilot.press("enter")
        await wait_until(pilot, lambda: not app.query(OverlayListView))

        assert _config(app) == LLMConfig(model="fake-model", provider="faux", reasoning="none")


async def test_reasoning_arg_rejects_an_unknown_level(tmp_path):
    session = fresh_session()
    before = session.session_config.llm_config.model_copy(deep=True)
    app = AgentApp(session, workspace=tmp_path, session_dir=tmp_path, skills=False, instructions=False)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/reasoning bogus")
        await pilot.pause()

        assert _config(app) == before
        assert _notices(app) == [
            (
                "unknown reasoning level 'bogus'. Valid: provider-default, none, minimal, low, medium, high, xhigh",
                True,
            ),
        ]


# ── /clear and /new ──────────────────────────────────────────────────────────


async def test_clear_saves_the_old_session_and_starts_a_fresh_one(tmp_path):
    app = agent_app(tmp_path, provider=scripted(faux_assistant_message([faux_text("hi back")])))
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "hello")
        await wait_until(pilot, lambda: _idle_again(app))
        old_id = app.runner.session.id

        await submit(pilot, "/clear")
        await pilot.pause()

        assert app.runner.session.id != old_id
        assert _config(app) == LLMConfig(model="fake-model", provider="faux")
        assert list(app.query(UserTurn)) == []
        assert (tmp_path / f"{old_id}.json").exists()
        assert _notices(app) == [
            (f"saved {old_id}, started fresh session {app.runner.session.id}", False),
        ]


async def test_new_is_an_alias_that_preserves_the_runtime_config(tmp_path):
    runtime = RuntimeConfig(hard_max_steps=3)
    app = AgentApp(
        fresh_session(runtime),
        workspace=tmp_path,
        session_dir=tmp_path,
        skills=False,
        instructions=False,
    )
    async with app.run_test(size=(105, 35)) as pilot:
        old_id = app.runner.session.id
        await submit(pilot, "/new")
        await pilot.pause()

        assert app.runner.session.id != old_id
        assert app.runner.session.session_config.runtime_config == runtime


# ── /help ────────────────────────────────────────────────────────────────────


async def test_help_mounts_the_command_list_block(tmp_path):
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/help")
        await pilot.pause()

        [view] = list(app.query(ListBlockView))
        assert view.model == vm.ListBlock(
            label="commands · 14",
            column=24,
            rows=[
                vm.ListRow(glyph="none", text="/skill [name]", annotation="load a skill into this session"),
                vm.ListRow(glyph="none", text="/session", annotation="resume, fork or list sessions"),
                vm.ListRow(glyph="none", text="/context", annotation="add files to the working set"),
                vm.ListRow(glyph="none", text="/cost", annotation="token and spend breakdown"),
                vm.ListRow(glyph="none", text="/settings", annotation="model, theme, permissions"),
                vm.ListRow(glyph="none", text="/clear", annotation="start a fresh session"),
                vm.ListRow(glyph="none", text="/model [provider:model]", annotation="pick a provider then a model"),
                vm.ListRow(glyph="none", text="/reasoning [level]", annotation="pick or set the reasoning level"),
                vm.ListRow(glyph="none", text="/theme", annotation="choose the color theme"),
                vm.ListRow(glyph="none", text="/compact", annotation="summarize the history and continue"),
                vm.ListRow(glyph="none", text="/resume", annotation="switch to another session in this project"),
                vm.ListRow(glyph="none", text="/new", annotation="save and start a fresh conversation"),
                vm.ListRow(glyph="none", text="/help", annotation="show every command"),
                vm.ListRow(glyph="none", text="/quit", annotation="save and exit"),
            ],
        )


# ── the palette ──────────────────────────────────────────────────────────────


def test_palette_rows_mirror_the_registry():
    # the contents are pinned by test_the_registry_lists_the_fourteen_commands;
    # this pins the projection: `/name` primary, summary secondary, nothing else
    assert palette_rows() == [vm.OverlayRow(primary=f"/{c.name}", secondary=c.summary) for c in COMMANDS]


async def test_a_palette_pick_of_an_insert_command_loads_the_composer(tmp_path):
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await run_palette_choice(app, "/skill")
        await pilot.pause()

        assert app.query_one(PromptInput).text == "/skill "


async def test_a_palette_pick_of_a_plain_command_runs_it(tmp_path):
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await run_palette_choice(app, "/help")
        await pilot.pause()

        assert len(app.query(ListBlockView)) == 1


# ── /quit ────────────────────────────────────────────────────────────────────


async def test_quit_saves_and_exits(tmp_path):
    session = fresh_session()
    app = AgentApp(session, workspace=tmp_path, session_dir=tmp_path, skills=False, instructions=False)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/quit")

    assert (tmp_path / f"{session.id}.json").exists()


# ── the settings state ───────────────────────────────────────────────────────


def test_build_settings_state_reflects_the_runner_and_the_app(tmp_path):
    app = agent_app(tmp_path)

    assert build_settings_state(app) == vm.SettingsState(
        selected=0,
        swatch_label="luca-dark",
        groups=[
            vm.SettingsGroup(
                label="model",
                rows=[
                    vm.SettingRow(name="model", value="fake-model"),
                    vm.SettingRow(name="provider", value="faux"),
                    vm.SettingRow(name="reasoning", value="provider-default"),
                    vm.SettingRow(name="streaming", value="on"),
                ],
            ),
            vm.SettingsGroup(
                label="permissions",
                rows=[vm.SettingRow(name="approval mode", value="ask", color="muted")],
            ),
            vm.SettingsGroup(
                label="appearance",
                rows=[
                    vm.SettingRow(name="theme", value="luca-dark"),
                    vm.SettingRow(name="show token counter", value="on"),
                ],
            ),
        ],
    )


async def test_settings_right_on_model_cycles_into_the_recommended_list(tmp_path):
    # ("faux", "fake-model") is not in the list, so the first step lands on
    # the first recommended pair
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/settings")
        await wait_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
        await pilot.press("right")
        await pilot.pause()

        assert _config(app) == LLMConfig(model="claude-opus-4-8", provider="anthropic")
        assert app.screen.state.groups[0].rows[0] == vm.SettingRow(name="model", value="claude-opus-4-8")


async def test_settings_right_on_reasoning_cycles_through_the_levels(tmp_path):
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/settings")
        await wait_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
        await pilot.press("down")
        await pilot.press("down")  # row 2: reasoning
        await pilot.press("right")
        await pilot.pause()
        assert _config(app) == LLMConfig(model="fake-model", provider="faux", reasoning="provider-default")

        await pilot.press("right")
        await pilot.pause()
        assert _config(app) == LLMConfig(model="fake-model", provider="faux", reasoning="none")


async def test_settings_right_on_streaming_toggles_it(tmp_path):
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/settings")
        await wait_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("down")  # row 3: streaming
        await pilot.press("right")
        await pilot.pause()

        assert app._streaming is False
        assert app.screen.state.groups[0].rows[3] == vm.SettingRow(name="streaming", value="off")


async def test_settings_right_on_the_token_counter_toggles_it(tmp_path):
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/settings")
        await wait_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
        await pilot.press("up")  # wraps to the last row: show token counter
        await pilot.press("right")
        await pilot.pause()

        assert app._show_counter is False
        assert app.screen.state.groups[2].rows[1] == vm.SettingRow(name="show token counter", value="off")


async def test_settings_leaves_the_approval_mode_untouched(tmp_path):
    # display-only until mode switching is real
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/settings")
        await wait_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("down")  # row 4: approval mode
        await pilot.press("right")
        await pilot.pause()

        assert app._mode == "ask"
        assert app.screen.state.groups[1].rows == [
            vm.SettingRow(name="approval mode", value="ask", color="muted"),
        ]


# ── the sessions state ───────────────────────────────────────────────────────


def test_build_sessions_state_lists_newest_first_with_the_count_line(tmp_path):
    older = with_user_message("older")
    newer = with_user_message("newer")
    save_session(older, tmp_path)
    save_session(newer, tmp_path)
    now = time.time()
    os.utime(tmp_path / f"{older.id}.json", (now - 600, now - 600))
    os.utime(tmp_path / f"{newer.id}.json", (now - 300, now - 300))
    total = sum(path.stat().st_size for path in tmp_path.glob("*.json"))
    app = agent_app(tmp_path)

    state, summaries = build_sessions_state(app)

    assert state == vm.SessionsState(
        count_line=f"2 sessions in this project · {total / 1000:.0f} KB · {tmp_path.name}",
        rows=[
            vm.SessionRow(id=newer.id, when="5m ago", first_message="newer", turns="1", tokens="0", cost="—"),
            vm.SessionRow(id=older.id, when="10m ago", first_message="older", turns="1", tokens="0", cost="—"),
        ],
        selected=0,
        preview=["[accent]›[/] newer"],
    )
    assert [summary.id for summary in summaries] == [newer.id, older.id]


def test_build_sessions_state_with_an_empty_store_returns_nothing(tmp_path):
    app = agent_app(tmp_path)

    assert build_sessions_state(app) == (None, [])
