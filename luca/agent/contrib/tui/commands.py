"""Slash commands: the palette registry and the live modal-state builders.

A flat, data-driven registry of the 14 built-in commands the palette lists.
`dispatch` is the single entry point for typed `/name arg` input;
`run_palette_choice` handles a palette pick (a command that wants an argument
is inserted into the composer instead of run). Pickers (`/model`,
`/reasoning`, `/theme`) use the overlay menu — the same shell as the palette —
never a stock modal.

A user's own commands (`.md` files, `custom_commands.py`) are appended to that
registry per app rather than baked into it, which is why the lookups here take
the app: `COMMANDS` is what this module ships, `commands_for(app)` is what a
running TUI answers to. Built-ins always win a name collision.

The live builders for the sessions / settings / cost screens live here too,
so `app.py` stays event wiring and the states stay derivable (and testable)
from a runner + config.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, get_args

from luca.agent.contrib.checkpoints import Checkpoint
from luca.agent.core.exceptions import AgentError
from luca.agent.core.models import LLMConfig, TextContent
from luca.agent.core.runner import AgentSessionRunner
from luca.client import catalog
from luca.client.providers import PROVIDERS
from luca.client.types import Reasoning

from . import state as vm
from .custom_commands import (
    CustomCommand,
    discover_commands,
    expand,
    resolve_locations,
)
from .format import fmt_cost, fmt_tokens, fmt_when, short_model
from .sessions import (
    SessionSummary,
    delete_session,
    fork_session,
    load_session,
    save_session,
)

if TYPE_CHECKING:
    from .app import AgentApp
    from .modals import SessionsScreen, SettingsScreen

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlashCommand:
    name: str
    usage: str  # argument hint shown by /help, e.g. "[provider:model]"
    summary: str
    handler: Callable[[AgentApp, str], Awaitable[None]]
    insert: bool = False  # palette pick inserts "/name " instead of running


# ── what the model pickers may offer ─────────────────────────────────────────

# `← →` on the settings row and the retry-on-failure offer want a handful, not
# every model a provider hosts.
RECENT_LIMIT = 5


def pickable_models(configured: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Provider → models for `/model`.

    The catalog says what exists; the provider registry says what luca can
    actually reach, and offering a host with no transport would be a dead end.
    `configured` is the `models` map from luca.json, unioned on top — the only
    route for a local or custom host, since neither is in models.dev.

    Never a gate: `/model provider:model` still switches to anything, which
    matters because a provider's line-up moves faster than a catalog does."""
    models: dict[str, list[str]] = {}
    for record in catalog.list():
        if record.provider in PROVIDERS and record.model is not None:
            models.setdefault(record.provider, []).append(record.model)
    for provider, listed in (configured or {}).items():
        models[provider] = sorted(set(models.get(provider, [])) | set(listed))
    return {provider: sorted(entries) for provider, entries in sorted(models.items())}


def recent_models(
    provider: str,
    limit: int = RECENT_LIMIT,
    configured: dict[str, list[str]] | None = None,
) -> list[str]:
    """The newest models one provider offers, one per family.

    Newest first, because an alphabetical list buries this year's releases. One
    per `family` because a host publishes the same model many times — bedrock
    carries `us.`, `eu.`, `jp.` and `global.` copies of one model, and cycling
    through four of them is not a choice. A model with no family stands alone.

    Falls back to the configured list for a host the catalog has never heard
    of, which is how `ollama` and custom providers still cycle."""
    records = [record for record in catalog.list(provider=provider) if record.model is not None]
    records.sort(key=lambda record: (record.release_date or "", record.model or ""), reverse=True)
    seen: set[str] = set()
    newest: list[str] = []
    for record in records:
        family = record.family or record.model
        if family in seen:
            continue
        seen.add(family)
        newest.append(record.model)
        if len(newest) == limit:
            break
    return newest or list((configured or {}).get(provider, ()))[:limit]


def model_context_note(provider: str, model: str) -> str | None:
    """The context window, when the catalog knows it — the one fact that
    actually tells rows apart in a list this long. Rendered as the row's
    SECONDARY: a menu row draws primary and secondary, never `annotation`,
    which only the `@` picker's row uses."""
    record = catalog.get(provider, model)
    if record is None or record.context_window is None:
        return None
    return fmt_tokens(record.context_window)


# ── handlers ──────────────────────────────────────────────────────────────────


_UNSET = object()
"""`reasoning` is nullable, so "leave it alone" and "clear it" need different
values and `None` is already taken by the second."""


def _apply(
    app: AgentApp,
    *,
    provider: str | None = None,
    model: str | None = None,
    reasoning: str | object | None = _UNSET,
) -> None:
    """Reassign the session's next-turn config; the runner reads it fresh at
    the top of each turn.

    The ONE mutation point for the session's llm_config, which is what lets a
    model switch re-resolve that model's options and re-point its credential in
    a single place. Only a provider/model change does either: re-resolving on a
    bare `/reasoning` would undo the level the user just picked.

    `reasoning` is written into `model_options`, where it now lives — the level
    is an ordinary client kwarg like `max_tokens`, and the core has no field for
    it."""
    config = app.runner.session.session_config.llm_config
    updates = {key: value for key, value in (("provider", provider), ("model", model)) if value is not None}
    updated = config.model_copy(update=updates) if updates else config
    if updates:
        updated = app.resolve_model_options(updated)
        app.repoint_credential(updated.provider)
    if reasoning is not _UNSET:
        options = {**updated.model_options}
        if reasoning is None:
            options.pop("reasoning", None)
        else:
            options["reasoning"] = reasoning
        updated = updated.model_copy(update={"model_options": options})
    app.runner.session.session_config.llm_config = updated
    app._refresh_status()


async def _warn_if_unbuildable(app: AgentApp) -> None:
    """Report that the new provider cannot be built. Never a gate: the switch
    has already happened and stays. Without this the model reads as set and
    only the next message discovers it."""
    error = app.provider_error()
    if error:
        await app._notice(error, error=True)


async def _cmd_help(app: AgentApp, arg: str) -> None:
    everything = commands_for(app)
    rows = [vm.ListRow(glyph="none", text=f"/{c.name} {c.usage}".rstrip(), annotation=c.summary) for c in everything]
    await app.mount_block(vm.ListBlock(label=f"commands · {len(everything)}", column=24, rows=rows))


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
        await _warn_if_unbuildable(app)
        return

    models = pickable_models(app.recommended_models)

    # Both steps read the choice back out of `app._menu_rows`, which is what the
    # overlay is actually showing. The index the overlay reports counts the rows
    # left after the query narrowed them, so indexing the unfiltered list would
    # apply a different model than the one under the cursor — and with hundreds
    # of models, typing to narrow is the only way to use this picker.
    async def picked_provider(index: int) -> None:
        provider = app._menu_rows[index].primary

        async def picked_model(model_index: int) -> None:
            model = app._menu_rows[model_index].primary
            await app._restore_composer()
            _apply(app, provider=provider, model=model)
            await app._notice(f"model set to {provider}:{model}")
            await _warn_if_unbuildable(app)

        await app.open_menu(
            [vm.OverlayRow(primary=model, secondary=model_context_note(provider, model)) for model in models[provider]],
            picked_model,
            column=40,
        )

    await app.open_menu(
        [vm.OverlayRow(primary=provider, secondary=f"{len(models[provider])} models") for provider in models],
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
    app._save()
    config = app.runner.session.session_config
    new = AgentSessionRunner.new_session(
        config.llm_config,
        runtime_config=config.runtime_config.model_copy(deep=True),
    )
    await app._reset_session(new)
    await app._notice(f"saved {old_id}, started fresh session {new.id}")


async def _restore_checkpoint(app: AgentApp, checkpoint: Checkpoint) -> None:
    """The shared tail of `/undo` and `/rewind`: restore, re-render, persist.

    `_reset_view` and not `_reset_session`: a rewind keeps the runner and only
    changes which conversation it drives, so there is nothing to rebuild — but
    every widget on screen still describes the path that was just archived."""
    try:
        restored = await app.checkpoints.restore(app.runner, checkpoint)
    except AgentError as exc:
        await app._notice(str(exc), error=True)
        return
    if not restored:
        await app._notice("could not restore that checkpoint", error=True)
        return
    await app._reset_view()
    app._save()
    label = f" to {checkpoint.label!r}" if checkpoint.label else ""
    await app._notice(f"restored the workspace and rewound the conversation{label}")


def _checkpoint_refusal(app: AgentApp) -> str | None:
    """Why neither checkpoint command can run right now, or None."""
    if not app.checkpoints.available:
        return "checkpoints are off — no git binary, or --no-checkpoints"
    if app._driving:
        return "cancel the current turn first (esc)"
    return None


async def _cmd_undo(app: AgentApp, arg: str) -> None:
    if refusal := _checkpoint_refusal(app):
        await app._notice(refusal, error=True)
        return
    points = app.checkpoints.checkpoints(app.runner.session)
    if not points:
        await app._notice("nothing to undo", error=True)
        return
    await _restore_checkpoint(app, points[0])


async def _cmd_rewind(app: AgentApp, arg: str) -> None:
    if refusal := _checkpoint_refusal(app):
        await app._notice(refusal, error=True)
        return
    points = app.checkpoints.checkpoints(app.runner.session)
    if not points:
        await app._notice("no checkpoints in this session", error=True)
        return

    async def picked(index: int) -> None:
        await app._restore_composer()
        await _restore_checkpoint(app, points[index])

    now = datetime.now()
    rows = [
        vm.OverlayRow(
            primary=point.label or "(the start of the session)",
            secondary=fmt_when(datetime.fromtimestamp(point.created_at / 1000), now) if point.created_at else None,
        )
        for point in points
    ]
    await app.open_menu(rows, picked, column=44)


async def _cmd_sessions(app: AgentApp, arg: str) -> None:
    session = app.runner.session
    if session.conversations[session.main_conversation_id].nodes:
        app._save()
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
    SlashCommand("undo", "", "revert the last turn's edits and rewind", _cmd_undo),
    SlashCommand("rewind", "", "pick an earlier checkpoint to restore", _cmd_rewind),
    SlashCommand("resume", "", "switch to another session in this project", _cmd_sessions),
    SlashCommand("new", "", "save and start a fresh conversation", _cmd_clear),
    SlashCommand("help", "", "show every command", _cmd_help),
    SlashCommand("quit", "", "save and exit", _cmd_quit),
)

BUILTIN_NAMES = frozenset(c.name for c in COMMANDS)


def _custom_handler(command: CustomCommand) -> Callable[[AgentApp, str], Awaitable[None]]:
    """Turn a discovered file into a handler. The body goes out as an ordinary
    user message — the same path `/skill` takes, so the transcript shows what
    was actually sent rather than the `/name` that stood for it."""

    async def handler(app: AgentApp, arg: str) -> None:
        text = expand(command.body, arg)
        app.runner.post_message([TextContent(text=text)])
        await app.mount_block(vm.UserBlock(text=text))
        if not app._driving:
            app._start_drive()

    return handler


def to_slash_commands(commands: dict[str, CustomCommand]) -> tuple[SlashCommand, ...]:
    """Discovered files as palette entries, sorted by name. One that declares an
    `argument-hint` is inserted rather than run when picked, exactly as `/skill`
    is — the hint is the file saying it expects to be given something."""
    return tuple(
        SlashCommand(
            name=command.name,
            usage=command.argument_hint,
            summary=command.description,
            handler=_custom_handler(command),
            insert=bool(command.argument_hint),
        )
        for _, command in sorted(commands.items())
    )


def load_custom_commands(
    workspace: str | Path,
    extra_locations: list[str] | None = None,
) -> tuple[SlashCommand, ...]:
    """Every user-defined command for `workspace`. Never raises: a broken
    commands directory costs the user their own commands, not their session.

    An unreadable individual file is already dropped by `load_command`, so what
    reaches here is the directory-level failure — an unresolvable `~user` in a
    configured location, or a root that cannot be listed."""
    try:
        locations = resolve_locations(workspace, extra_locations)
        return to_slash_commands(discover_commands(locations, reserved=BUILTIN_NAMES))
    except (OSError, RuntimeError, ValueError):
        logger.warning("could not read the user-defined commands; continuing with the built-ins", exc_info=True)
        return ()


def commands_for(app: AgentApp | None = None) -> tuple[SlashCommand, ...]:
    """The built-ins plus whatever `app` loaded from disk. `None` is the bare
    registry, which is what the component gallery renders."""
    if app is None:
        return COMMANDS
    return COMMANDS + app.custom_commands


async def dispatch(app: AgentApp, text: str) -> bool:
    """Run `/name arg` if `name` is registered. False leaves the text for the
    caller to send as a normal message, so `/etc/hosts` is never swallowed."""
    parts = text[1:].split(maxsplit=1)
    if not parts:
        return False
    command = next((c for c in commands_for(app) if c.name == parts[0]), None)
    if command is None:
        return False
    arg = parts[1].strip() if len(parts) > 1 else ""
    await command.handler(app, arg)
    return True


# ── palette ───────────────────────────────────────────────────────────────────


def palette_rows(app: AgentApp | None = None) -> list[vm.OverlayRow]:
    return [vm.OverlayRow(primary=f"/{c.name}", secondary=c.summary) for c in commands_for(app)]


async def run_palette_choice(app: AgentApp, primary: str) -> None:
    name = primary.lstrip("/")
    command = next((c for c in commands_for(app) if c.name == name), None)
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
                    vm.SettingRow(
                        name="reasoning",
                        value=config.model_options.get("reasoning") or "provider-default",
                    ),
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
        # Within the current provider only, and only its newest handful: this
        # row cycles one keypress at a time, and the catalog holds hundreds.
        config = app.runner.session.session_config.llm_config
        models = recent_models(config.provider, configured=app.recommended_models)
        if not models:
            return
        current = models.index(config.model) if config.model in models else -1
        _apply(app, model=models[(current + delta) % len(models)])
    elif row.name == "provider":
        return
    elif row.name == "reasoning":
        levels = [None, *get_args(Reasoning)]
        config = app.runner.session.session_config.llm_config
        reasoning = config.model_options.get("reasoning")
        current = levels.index(reasoning) if reasoning in levels else 0
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
