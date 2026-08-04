"""Slash commands: the palette registry and the live modal-state builders.

A flat, data-driven registry of the 14 commands the palette lists. `dispatch`
is the single entry point for typed `/name arg` input; `run_palette_choice`
handles a palette pick (a command that wants an argument is inserted into the
composer instead of run). Pickers (`/model`, `/reasoning`, `/theme`) use the
overlay menu — the same shell as the palette — never a stock modal.

The live builders for the sessions / settings / cost screens live here too,
so `app.py` stays event wiring and the states stay derivable (and testable)
from a runner + config.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, get_args

from luca.agent.core.models import LLMConfig, TextContent
from luca.agent.core.runner import AgentSessionRunner
from luca.client.types import Reasoning

from . import state as vm
from .format import fmt_cost, fmt_tokens, fmt_when, short_model
from .sessions import (
    SessionSummary,
    delete_session,
    fork_session,
    load_session,
    save_session,
)
from .wiring import RECOMMENDED_MODELS

if TYPE_CHECKING:
    from .app import AgentApp
    from .modals import SessionsScreen, SettingsScreen


@dataclass(frozen=True)
class SlashCommand:
    name: str
    usage: str  # argument hint shown by /help, e.g. "[provider:model]"
    summary: str
    handler: Callable[[AgentApp, str], Awaitable[None]]
    insert: bool = False  # palette pick inserts "/name " instead of running


# ── handlers ──────────────────────────────────────────────────────────────────


def _apply(app: AgentApp, **updates: str | None) -> None:
    """Reassign the session's next-turn config; the runner reads it fresh at
    the top of each turn."""
    config = app.runner.session.session_config.llm_config
    app.runner.session.session_config.llm_config = config.model_copy(update=updates)
    app._refresh_status()


async def _cmd_help(app: AgentApp, arg: str) -> None:
    rows = [vm.ListRow(glyph="none", text=f"/{c.name} {c.usage}".rstrip(), annotation=c.summary) for c in COMMANDS]
    await app.mount_block(vm.ListBlock(label=f"commands · {len(COMMANDS)}", column=24, rows=rows))


async def _cmd_model(app: AgentApp, arg: str) -> None:
    if arg:
        if ":" in arg:
            provider, model = arg.split(":", 1)
            if not provider or not model:
                await app._notice(f"invalid model spec {arg!r}; use provider:model", error=True)
                return
            _apply(app, provider=provider, model=model)
        else:
            _apply(app, model=arg)
        await app._notice(f"model set to {app.runner.session.session_config.llm_config.model}")
        return

    models = app.recommended_models or RECOMMENDED_MODELS

    async def picked_provider(index: int) -> None:
        provider = list(models)[index]

        async def picked_model(model_index: int) -> None:
            await app._restore_composer()
            _apply(app, provider=provider, model=models[provider][model_index])
            await app._notice(f"model set to {provider}:{models[provider][model_index]}")

        await app.open_menu(
            [vm.OverlayRow(primary=model) for model in models[provider]],
            picked_model,
            column=40,
        )

    await app.open_menu(
        [vm.OverlayRow(primary=provider) for provider in models],
        picked_provider,
        column=40,
    )


async def _cmd_reasoning(app: AgentApp, arg: str) -> None:
    levels = list(get_args(Reasoning))
    if arg:
        if arg not in levels:
            await app._notice(f"unknown reasoning level {arg!r}. Valid: {', '.join(levels)}", error=True)
            return
        _apply(app, reasoning=arg)
        await app._notice(f"reasoning set to {arg}")
        return

    async def picked(index: int) -> None:
        await app._restore_composer()
        _apply(app, reasoning=levels[index])
        await app._notice(f"reasoning set to {levels[index]}")

    await app.open_menu([vm.OverlayRow(primary=level) for level in levels], picked)


async def _cmd_theme(app: AgentApp, arg: str) -> None:
    themes = list(app.available_themes)

    async def picked(index: int) -> None:
        await app._restore_composer()
        app.theme = themes[index]
        await app._notice(f"theme set to {themes[index]}")

    await app.open_menu([vm.OverlayRow(primary=name) for name in themes], picked, column=30)


async def _cmd_clear(app: AgentApp, arg: str) -> None:
    old_id = app.runner.session.id
    save_session(app.runner.session, app._session_dir)
    config = app.runner.session.session_config
    new = AgentSessionRunner.new_session(
        config.llm_config,
        runtime_config=config.runtime_config.model_copy(deep=True),
    )
    await app._reset_session(new)
    await app._notice(f"saved {old_id}, started fresh session {new.id}")


async def _cmd_sessions(app: AgentApp, arg: str) -> None:
    session = app.runner.session
    if session.conversations[session.main_conversation_id].nodes:
        save_session(session, app._session_dir)
    await app.open_sessions_screen()


async def _cmd_skill(app: AgentApp, arg: str) -> None:
    if not arg:
        composer = app.composer()
        if composer is not None:
            composer.input.load_text("/skill ")
            composer.input.move_cursor(composer.input.document.end)
            composer.input.focus()
        return
    text = f"/skill {arg}"
    app.runner.post_message([TextContent(text=text)])
    await app.mount_block(vm.UserBlock(text=text))
    if not app._driving:
        app._start_drive()


async def _cmd_context(app: AgentApp, arg: str) -> None:
    await app.open_context_picker(arg)


async def _cmd_cost(app: AgentApp, arg: str) -> None:
    await app.open_cost_screen()


async def _cmd_settings(app: AgentApp, arg: str) -> None:
    await app.open_settings_screen()


async def _cmd_compact(app: AgentApp, arg: str) -> None:
    app.runner.schedule_compaction()
    await app._notice("compacting the conversation…")
    if not app._driving:
        app._start_drive()


async def _cmd_quit(app: AgentApp, arg: str) -> None:
    await app._quit()


COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("skill", "[name]", "load a skill into this session", _cmd_skill, insert=True),
    SlashCommand("session", "", "resume, fork or list sessions", _cmd_sessions),
    SlashCommand("context", "", "add files to the working set", _cmd_context),
    SlashCommand("cost", "", "token and spend breakdown", _cmd_cost),
    SlashCommand("settings", "", "model, theme, permissions", _cmd_settings),
    SlashCommand("clear", "", "start a fresh session", _cmd_clear),
    SlashCommand("model", "[provider:model]", "pick a provider then a model", _cmd_model),
    SlashCommand("reasoning", "[level]", "pick or set the reasoning level", _cmd_reasoning),
    SlashCommand("theme", "", "choose the color theme", _cmd_theme),
    SlashCommand("compact", "", "summarize the history and continue", _cmd_compact),
    SlashCommand("resume", "", "switch to another session in this project", _cmd_sessions),
    SlashCommand("new", "", "save and start a fresh conversation", _cmd_clear),
    SlashCommand("help", "", "show every command", _cmd_help),
    SlashCommand("quit", "", "save and exit", _cmd_quit),
)

_BY_NAME = {c.name: c for c in COMMANDS}


async def dispatch(app: AgentApp, text: str) -> bool:
    """Run `/name arg` if `name` is registered. False leaves the text for the
    caller to send as a normal message, so `/etc/hosts` is never swallowed."""
    parts = text[1:].split(maxsplit=1)
    if not parts:
        return False
    command = _BY_NAME.get(parts[0])
    if command is None:
        return False
    arg = parts[1].strip() if len(parts) > 1 else ""
    await command.handler(app, arg)
    return True


# ── palette ───────────────────────────────────────────────────────────────────


def palette_rows() -> list[vm.OverlayRow]:
    return [vm.OverlayRow(primary=f"/{c.name}", secondary=c.summary) for c in COMMANDS]


async def run_palette_choice(app: AgentApp, primary: str) -> None:
    command = _BY_NAME.get(primary.lstrip("/"))
    if command is None:
        return
    if command.insert:
        composer = app.composer()
        if composer is not None:
            composer.input.load_text(f"/{command.name} ")
            composer.input.move_cursor(composer.input.document.end)
            composer.input.focus()
        return
    await command.handler(app, "")


# ── live modal states ─────────────────────────────────────────────────────────


def build_sessions_state(
    summaries: list[SessionSummary],
    *,
    directory_name: str,
    now: datetime | None = None,
    selected: int = 0,
) -> vm.SessionsState | None:
    """The sessions screen from its rows. Takes summaries rather than a
    directory — the live app lists a store, the catalog hands over sessions it
    already holds, and `now` is injectable so `18m ago` is reproducible."""
    if not summaries:
        return None
    now = now or datetime.now()
    total_bytes = sum(summary.size_bytes for summary in summaries)
    rows = [
        vm.SessionRow(
            id=summary.id,
            when=fmt_when(summary.modified, now),
            first_message=summary.title,
            turns=str(summary.turns),
            tokens=fmt_tokens(summary.tokens),
            cost=fmt_cost(summary.cost) if summary.cost is not None else "—",
        )
        for summary in summaries
    ]
    size = f"{total_bytes / 1_000_000:.1f} MB" if total_bytes >= 1_000_000 else f"{total_bytes / 1000:.0f} KB"
    noun = "session" if len(summaries) == 1 else "sessions"
    return vm.SessionsState(
        count_line=f"{len(summaries)} {noun} in this project · {size} · {directory_name}",
        rows=rows,
        selected=selected,
        preview=summaries[selected].preview or [],
    )


def session_preview(summary: SessionSummary) -> list[str]:
    return summary.preview or []


async def resume_session(app: AgentApp, screen: SessionsScreen, row: vm.SessionRow) -> None:
    screen.dismiss(None)
    if row.id is None or row.id == app.runner.session.id:
        return
    await app._reset_session(load_session(row.id, app._session_dir))
    await app._notice(f"resumed session {row.id}")


async def fork_session_row(app: AgentApp, screen: SessionsScreen, row: vm.SessionRow) -> None:
    screen.dismiss(None)
    if row.id is None:
        return
    forked = fork_session(load_session(row.id, app._session_dir))
    save_session(forked, app._session_dir)
    await app._reset_session(forked)
    await app._notice(f"forked {row.id} into {forked.id}")


async def delete_session_row(app: AgentApp, screen: SessionsScreen, row: vm.SessionRow) -> None:
    screen.dismiss(None)
    if row.id is None:
        return
    delete_session(row.id, app._session_dir)
    await app._notice(f"deleted session {row.id}")
    await app.open_sessions_screen()


# ── settings ──────────────────────────────────────────────────────────────────

_MODE_COLORS: dict[str, vm.SettingColor] = {"ask": "muted", "yolo": "accent", "auto": "muted"}


def build_settings_state(
    config: LLMConfig,
    *,
    theme: str,
    streaming: bool,
    mode: str,
    show_counter: bool,
    selected: int = 0,
) -> vm.SettingsState:
    """The settings screen from the model config plus the ambient app state.
    Takes data, not the app, so a catalogued world renders the same screen the
    live app does."""
    return vm.SettingsState(
        selected=selected,
        swatch_label=theme,
        groups=[
            vm.SettingsGroup(
                label="model",
                rows=[
                    vm.SettingRow(name="model", value=short_model(config.model)),
                    vm.SettingRow(name="provider", value=config.provider),
                    vm.SettingRow(name="reasoning", value=config.reasoning or "provider-default"),
                    vm.SettingRow(name="streaming", value="on" if streaming else "off"),
                ],
            ),
            vm.SettingsGroup(
                label="permissions",
                rows=[
                    vm.SettingRow(
                        name="approval mode",
                        value=mode,
                        color=_MODE_COLORS.get(mode, "muted"),
                    ),
                ],
            ),
            vm.SettingsGroup(
                label="appearance",
                rows=[
                    vm.SettingRow(name="theme", value=theme),
                    vm.SettingRow(name="show token counter", value="on" if show_counter else "off"),
                ],
            ),
        ],
    )


def adjust_setting(app: AgentApp, screen: SettingsScreen, row: vm.SettingRow, delta: int) -> None:
    """`← →` on a settings row. Model/reasoning/theme/streaming/counter are
    live; the approval mode is display-only until mode switching is real."""
    if row.name == "model":
        models = app.recommended_models or RECOMMENDED_MODELS
        flat = [(provider, model) for provider, names in models.items() for model in names]
        config = app.runner.session.session_config.llm_config
        current = next(
            (i for i, (p, m) in enumerate(flat) if p == config.provider and m == config.model),
            -1,
        )
        provider, model = flat[(current + delta) % len(flat)]
        _apply(app, provider=provider, model=model)
    elif row.name == "provider":
        return
    elif row.name == "reasoning":
        levels = [None, *get_args(Reasoning)]
        config = app.runner.session.session_config.llm_config
        current = levels.index(config.reasoning) if config.reasoning in levels else 0
        _apply(app, reasoning=levels[(current + delta) % len(levels)])
    elif row.name == "streaming":
        app._streaming = not app._streaming
    elif row.name == "theme":
        themes = list(app.available_themes)
        current = themes.index(app.theme) if app.theme in themes else 0
        app.theme = themes[(current + delta) % len(themes)]
    elif row.name == "show token counter":
        app._show_counter = not app._show_counter
        app._refresh_status()
    else:
        return
    screen.set_state(app.settings_state(selected=screen.state.selected))


# ── skills (^s) ───────────────────────────────────────────────────────────────


def skills_block(app: AgentApp) -> vm.ListBlock:
    try:
        from luca.agent.contrib.skills import default_locations, discover_skills

        locations = default_locations(app._workspace)
        for extra in app._extra_skill_locations or []:
            locations.append(Path(extra).expanduser())
        skills = discover_skills(locations)
    except Exception:
        skills = {}
    rows = [
        vm.ListRow(glyph="included", text=name, annotation=str(skill.path.parent))
        for name, skill in sorted(skills.items())
    ]
    if not rows:
        rows = [vm.ListRow(glyph="none", text="no skills discovered")]
    return vm.ListBlock(label=f"skills · {len(skills)}", column=24, rows=rows)
