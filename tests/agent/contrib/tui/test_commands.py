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
from functools import partial

from luca.agent.contrib.tui import state as vm
from luca.agent.contrib.tui.app import AgentApp
from luca.agent.contrib.tui.auth import AuthEntry
from luca.agent.contrib.tui.blocks import ListBlockView, NoticeLine, UserTurn
from luca.agent.contrib.tui.commands import (
    COMMANDS,
    build_sessions_state,
    dispatch,
    palette_rows,
    pickable_models,
    recent_models,
    run_palette_choice,
)
from luca.agent.contrib.tui.config import LucaConfig, apply_model_options
from luca.agent.contrib.tui.format import fmt_tokens, short_model
from luca.agent.contrib.tui.modals import SettingsScreen
from luca.agent.contrib.tui.prompt import PromptInput
from luca.agent.contrib.tui.sessions import list_sessions, save_session
from luca.agent.contrib.tui.shells import OverlayListView, QueryLine
from luca.agent.contrib.tui.wiring import default_model
from luca.agent.core.models import LLMConfig, RuntimeConfig
from luca.client import catalog
from luca.client.catalog._data import cache_path
from luca.client.providers import PROVIDERS
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


def test_the_registry_lists_the_seventeen_commands():
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
        ("undo", "", "revert the last turn's edits and rewind", False),
        ("rewind", "", "pick an earlier checkpoint to restore", False),
        ("resume", "", "switch to another session in this project", False),
        ("new", "", "save and start a fresh conversation", False),
        ("mcp", "[server]", "MCP servers: status, tools, sign in, reconnect", False),
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

        assert _config(app) == LLMConfig(model="fake-model", provider="faux", model_options={"reasoning": "high"})


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
    # Both come from the catalog, so the expectation is derived from the same
    # door rather than restating a list that now has hundreds of entries.
    offered = pickable_models()
    provider = next(iter(offered))
    model = offered[provider][0]
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/model")
        await wait_until(pilot, lambda: bool(app.query(OverlayListView)))
        await pilot.press("enter")  # provider 0
        await pilot.pause()
        await pilot.press("enter")  # model 0 of that provider
        await wait_until(pilot, lambda: not app.query(OverlayListView))

        assert _config(app) == LLMConfig(model=model, provider=provider)
        assert _notices(app) == [(f"model set to {provider}:{model}", False)]


async def test_model_menu_arrows_pick_by_index(tmp_path):
    # provider index 1, then its model index 1 — both read off the catalog
    offered = pickable_models()
    provider = list(offered)[1]
    model = offered[provider][1]
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

        assert _config(app) == LLMConfig(model=model, provider=provider)


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


# ── /model + configured options ──────────────────────────────────────────────

OPTIONS_CONFIG = LucaConfig.model_validate(
    {
        "providers": {
            "anthropic": {
                "models": {"claude-sonnet-5": {"options": {"max_tokens": 6000, "reasoning": "high"}}},
            },
        },
    }
)


def options_app(tmp_path) -> AgentApp:
    return AgentApp(
        fresh_session(),
        workspace=tmp_path,
        session_dir=tmp_path,
        skills=False,
        instructions=False,
        model_options=partial(apply_model_options, config=OPTIONS_CONFIG),
    )


async def test_switching_to_a_configured_model_resolves_its_options(tmp_path):
    app = options_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/model anthropic:claude-sonnet-5")
        await pilot.pause()

        assert _config(app) == LLMConfig(
            model="claude-sonnet-5",
            provider="anthropic",
            model_options={"max_tokens": 6000, "reasoning": "high"},
        )


async def test_switching_away_again_clears_them(tmp_path):
    app = options_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/model anthropic:claude-sonnet-5")
        await pilot.pause()
        await submit(pilot, "/model anthropic:claude-opus-4-8")
        await pilot.pause()

        assert _config(app) == LLMConfig(
            model="claude-opus-4-8",
            provider="anthropic",
        )


async def test_switching_provider_re_points_the_credential(tmp_path):
    # A key is per PROVIDER, so a switch that leaves it behind would send
    # openrouter's key to anthropic.
    app = AgentApp(
        fresh_session(),
        auth={
            "faux": AuthEntry(type="api", key="sk-faux"),
            "anthropic": AuthEntry(type="api", key="sk-ant"),
        },
        workspace=tmp_path,
        session_dir=tmp_path,
        skills=False,
        instructions=False,
    )
    async with app.run_test(size=(105, 35)) as pilot:
        before = app.runner.api_key
        await submit(pilot, "/model anthropic:claude-sonnet-5")
        await pilot.pause()

        assert (before, app.runner.api_key) == ("sk-faux", "sk-ant")


async def test_switching_to_a_provider_with_no_entry_clears_the_credential(tmp_path):
    app = AgentApp(
        fresh_session(),
        auth={"faux": AuthEntry(type="api", key="sk-faux")},
        workspace=tmp_path,
        session_dir=tmp_path,
        skills=False,
        instructions=False,
    )
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/model anthropic:claude-sonnet-5")
        await pilot.pause()

        assert app.runner.api_key is None


async def test_reasoning_alone_does_not_re_resolve_the_model_block(tmp_path):
    # `/reasoning` after a switch must stand: re-resolving on every _apply
    # would silently put the configured level back.
    app = options_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/model anthropic:claude-sonnet-5")
        await pilot.pause()
        await submit(pilot, "/reasoning low")
        await pilot.pause()

        assert _config(app) == LLMConfig(
            model="claude-sonnet-5",
            provider="anthropic",
            model_options={"max_tokens": 6000, "reasoning": "low"},
        )


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

        assert _config(app) == LLMConfig(model="fake-model", provider="faux", model_options={"reasoning": "none"})


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

        # Scoped to the transcript: the docked plan panel is a `ListBlockView`
        # too, mounted from the start and hidden until there are todos.
        [view] = list(app.transcript.query(ListBlockView))
        assert view.model == vm.ListBlock(
            label="commands · 17",
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
                vm.ListRow(glyph="none", text="/undo", annotation="revert the last turn's edits and rewind"),
                vm.ListRow(glyph="none", text="/rewind", annotation="pick an earlier checkpoint to restore"),
                vm.ListRow(glyph="none", text="/resume", annotation="switch to another session in this project"),
                vm.ListRow(glyph="none", text="/new", annotation="save and start a fresh conversation"),
                vm.ListRow(
                    glyph="none", text="/mcp [server]", annotation="MCP servers: status, tools, sign in, reconnect"
                ),
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

        assert len(app.transcript.query(ListBlockView)) == 1


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

    assert app.settings_state() == vm.SettingsState(
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


async def test_settings_right_on_model_cycles_within_the_providers_recent_models(tmp_path):
    # cycling stays on the current provider and walks its newest handful; the
    # catalog holds hundreds, which is not something you step through
    newest = recent_models("anthropic")
    app = agent_app(tmp_path)
    app.runner.session.session_config.llm_config = LLMConfig(model=newest[0], provider="anthropic")
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/settings")
        await wait_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
        await pilot.press("right")
        await pilot.pause()

        assert _config(app) == LLMConfig(model=newest[1], provider="anthropic")
        assert app.screen.state.groups[0].rows[0] == vm.SettingRow(name="model", value=short_model(newest[1]))


async def test_settings_model_cycling_does_nothing_for_a_provider_nothing_lists(tmp_path):
    # `faux` is in no catalog and no config, so there is nothing to cycle to
    app = agent_app(tmp_path)
    before = _config(app)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/settings")
        await wait_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
        await pilot.press("right")
        await pilot.pause()

        assert _config(app) == before


async def test_settings_right_on_reasoning_cycles_through_the_levels(tmp_path):
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/settings")
        await wait_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
        await pilot.press("down")
        await pilot.press("down")  # row 2: reasoning
        await pilot.press("right")
        await pilot.pause()
        assert _config(app) == LLMConfig(
            model="fake-model", provider="faux", model_options={"reasoning": "provider-default"}
        )

        await pilot.press("right")
        await pilot.pause()
        assert _config(app) == LLMConfig(model="fake-model", provider="faux", model_options={"reasoning": "none"})


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

    summaries = list_sessions(tmp_path)
    state = build_sessions_state(summaries, directory_name=tmp_path.name)

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
    assert build_sessions_state(list_sessions(tmp_path), directory_name=tmp_path.name) is None


# ── what the model pickers may offer ─────────────────────────────────────────


def test_the_picker_offers_only_providers_luca_can_route_to():
    offered = set(pickable_models())

    assert offered <= set(PROVIDERS)
    assert "anthropic" in offered
    assert "openrouter" in offered
    # models.dev carries google; luca has no google transport, so offering it
    # would be a dead end
    assert "google" not in offered


def test_a_configured_host_is_unioned_in_even_though_models_dev_lacks_it():
    offered = pickable_models({"quantized": ["anthropic/claude-opus-4.6"]})

    assert offered["quantized"] == ["anthropic/claude-opus-4.6"]


def test_configured_models_extend_a_provider_rather_than_replacing_it():
    offered = pickable_models({"anthropic": ["claude-unreleased"]})

    assert "claude-unreleased" in offered["anthropic"]
    assert "claude-fable-5" in offered["anthropic"]


def test_recent_models_are_newest_first_and_one_per_family():
    # bedrock publishes `us.`/`eu.`/`jp.`/`global.` copies of one model; four
    # rows for one choice is not a choice
    newest = recent_models("bedrock", limit=6)

    assert len(newest) == len(set(newest))
    families = [catalog.get("bedrock", model).family for model in newest]
    assert len(families) == len(set(families))


def test_recent_models_is_bounded():
    assert len(recent_models("openrouter", limit=3)) == 3


def test_recent_models_falls_back_to_the_configured_list():
    # `ollama` serves whatever you pulled; the catalog cannot know it
    assert recent_models("ollama", configured={"ollama": ["llama3", "qwen"]}) == ["llama3", "qwen"]


def test_recent_models_is_empty_for_a_host_nothing_lists():
    assert recent_models("faux") == []


def test_the_default_model_is_one_the_catalog_knows():
    # a default models.dev no longer lists should fail the build, not the user
    default = default_model()

    assert catalog.get(default.provider, default.model) is not None


# ── picking from a filtered menu ─────────────────────────────────────────────


async def test_a_filtered_pick_applies_the_row_that_was_highlighted(tmp_path):
    # THE risk of this feature. The overlay reports the index of the row among
    # those left after the query, so resolving it against the unfiltered list
    # would apply a different model than the one under the cursor.
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/model")
        await wait_until(pilot, lambda: bool(app.query(OverlayListView)))
        await pilot.press("down", "down", "down", "down", "down", "enter")  # openrouter
        await pilot.pause()

        view = app.query_one(OverlayListView)
        view.post_message(OverlayListView.QueryChanged(view, "sonnet"))
        await pilot.pause()
        await pilot.pause()
        highlighted = app._menu_rows[1].primary
        unfiltered = app._menu_all_rows[1].primary

        view.post_message(OverlayListView.Committed(view, 1))
        await wait_until(pilot, lambda: not app.query(OverlayListView))

        assert highlighted != unfiltered  # or the test proves nothing
        assert _config(app) == LLMConfig(model=highlighted, provider="openrouter")


async def test_every_model_is_offered_and_the_list_scrolls(tmp_path):
    # the whole line-up is reachable: the stylesheet bounds what is VISIBLE and
    # the container scrolls, rather than the row list being trimmed
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/model")
        await wait_until(pilot, lambda: bool(app.query(OverlayListView)))
        await pilot.press("down", "down", "down", "down", "down", "enter")  # openrouter
        await pilot.pause()
        await pilot.pause()
        options = app.query_one(OverlayListView).query_one(".overlay-options")

        assert len(app._menu_rows) == len(pickable_models()["openrouter"])
        assert options.virtual_size.height > options.size.height  # scrollable


async def test_the_query_line_stays_on_screen_under_a_long_list(tmp_path):
    # without a ceiling the overlay grows past the terminal and takes the
    # query line with it, leaving no way to narrow the list
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/model")
        await wait_until(pilot, lambda: bool(app.query(OverlayListView)))
        await pilot.press("down", "down", "down", "down", "down", "enter")  # openrouter
        await pilot.pause()
        await pilot.pause()

        assert app.query_one(OverlayListView).query_one(QueryLine).region.y < app.screen.size.height


async def test_arrowing_past_the_visible_rows_scrolls_the_caret_into_view(tmp_path):
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/model")
        await wait_until(pilot, lambda: bool(app.query(OverlayListView)))
        await pilot.press("down", "down", "down", "down", "down", "enter")  # openrouter
        await pilot.pause()
        options = app.query_one(OverlayListView).query_one(".overlay-options")
        assert options.scroll_offset.y == 0

        for _ in range(25):
            await pilot.press("down")
        await pilot.pause()

        assert options.scroll_offset.y > 0


async def test_a_model_far_down_the_list_is_reachable_by_typing(tmp_path):
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/model")
        await wait_until(pilot, lambda: bool(app.query(OverlayListView)))
        await pilot.press("down", "down", "down", "down", "down", "enter")  # openrouter
        await pilot.pause()

        view = app.query_one(OverlayListView)
        view.post_message(OverlayListView.QueryChanged(view, "qwen3.8-max"))
        await pilot.pause()
        await pilot.pause()

        assert [row.primary for row in app._menu_rows] == ["qwen/qwen3.8-max"]


async def test_a_model_row_shows_its_context_window(tmp_path):
    # a menu row renders `primary` and `secondary`; `annotation` is the @
    # picker's field and would never appear here
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/model")
        await wait_until(pilot, lambda: bool(app.query(OverlayListView)))
        await pilot.press("enter")  # anthropic
        await pilot.pause()

        row = next(row for row in app._menu_rows if row.primary == "claude-sonnet-5")
        assert row.secondary == fmt_tokens(catalog.get("anthropic", "claude-sonnet-5").context_window)


async def test_a_model_the_catalog_does_not_list_is_still_selectable(tmp_path):
    # the catalog is metadata, never a gate
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/model openrouter:some/unreleased-model")
        await pilot.pause()

        assert catalog.get("openrouter", "some/unreleased-model") is None
        assert _config(app) == LLMConfig(model="some/unreleased-model", provider="openrouter")


async def test_a_model_id_containing_a_colon_survives_the_argument_form(tmp_path):
    # bedrock ids look like `amazon.nova-2-lite-v1:0`
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/model bedrock:amazon.nova-2-lite-v1:0")
        await pilot.pause()

        assert _config(app) == LLMConfig(model="amazon.nova-2-lite-v1:0", provider="bedrock")


def test_the_tests_read_an_isolated_model_catalog_cache():
    # `--refresh-models` writes `$XDG_CACHE_HOME/luca/models.json` and the
    # catalog layers it over the vendored records. Without the conftest
    # pinning that variable, a contributor who has refreshed would be testing
    # against different models than CI.
    assert str(cache_path()).startswith(os.environ["XDG_CACHE_HOME"])
