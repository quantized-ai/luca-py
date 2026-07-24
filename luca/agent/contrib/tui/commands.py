"""Slash commands for the TUI.

A flat, data-driven registry: one `SlashCommand` per command, so `/help`
renders itself and nothing drifts. `dispatch` is the single entry point the
app calls from `on_input_submitted`; it returns True when it handled a
command and False when the text was not a command, so the caller can send it
to the agent as a normal message.

`/model`, `/provider`, and `/reasoning` open an arrow-key picker when given no
argument, and switch directly when given one (`/model anthropic:claude-sonnet-5`).
Pickers use `push_screen` with a callback rather than `push_screen_wait`, which
would require a worker; command handlers run from an event handler, not one.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, get_args

from luca.agent.core.models import LLMConfig
from luca.agent.core.runner import AgentSessionRunner
from luca.client.types import Reasoning

from .screens import PickerScreen
from .sessions import save_session
from .wiring import RECOMMENDED_MODELS

if TYPE_CHECKING:
    from .app import AgentApp

# The `/model` model step ends with this entry; choosing it returns to the
# provider step instead of setting a model.
_BACK = "← Back to providers"


@dataclass(frozen=True)
class SlashCommand:
    name: str
    usage: str  # argument hint shown by /help, e.g. "[provider:model]"
    summary: str
    handler: Callable[["AgentApp", str], Awaitable[None]]


def _current_model_line(cfg: LLMConfig) -> str:
    return f"{cfg.provider}:{cfg.model} (reasoning: {cfg.reasoning or 'provider-default'})"


def _apply(app: "AgentApp", **updates: str) -> None:
    """Reassign the session's next-turn config. The runner reads it fresh at the
    top of each turn, so a change while idle takes effect on the next message."""
    cfg = app.runner.session.session_config.llm_config
    app.runner.session.session_config.llm_config = cfg.model_copy(update=updates)
    app._refresh_status()


async def _cmd_help(app: "AgentApp", arg: str) -> None:
    width = max(len(f"/{c.name} {c.usage}".rstrip()) for c in COMMANDS)
    lines = [
        f"{f'/{c.name} {c.usage}'.rstrip():<{width}}  {c.summary}"
        for c in COMMANDS
    ]
    await app._notice("\n".join(lines))


async def _cmd_model(app: "AgentApp", arg: str) -> None:
    cfg = app.runner.session.session_config.llm_config
    if arg:
        if ":" in arg:
            provider, model = arg.split(":", 1)
            if not provider or not model:
                await app._notice(
                    f"invalid model spec {arg!r}; use provider:model", error=True,
                )
                return
            _apply(app, provider=provider, model=model)
        else:
            _apply(app, model=arg)
        await app._notice(
            f"model set to {_current_model_line(app.runner.session.session_config.llm_config)}",
        )
        return

    # No arg: drill down. Pick a provider, then one of its models — the two are
    # applied together at the end, so a provider is never left with a mismatched
    # model, and Esc at either step changes nothing. The model step carries a
    # "back" entry that returns to the provider step with that provider still
    # highlighted, so the provider choice can be changed without starting over.
    def open_provider_step(highlight: str | None) -> None:
        app.push_screen(
            PickerScreen(
                "Select a provider", list(RECOMMENDED_MODELS), current=highlight,
            ),
            picked_provider,
        )

    def picked_provider(provider: str | None) -> None:
        if provider is None:
            return

        async def picked_model(choice: str | None) -> None:
            if choice is None:
                return
            if choice == _BACK:
                open_provider_step(provider)
                return
            _apply(app, provider=provider, model=choice)
            await app._notice(
                f"model set to {_current_model_line(app.runner.session.session_config.llm_config)}",
            )

        app.push_screen(
            PickerScreen(
                f"Select a model ({provider})",
                [*RECOMMENDED_MODELS[provider], _BACK],
                current=cfg.model,
            ),
            picked_model,
        )

    open_provider_step(cfg.provider)


async def _cmd_reasoning(app: "AgentApp", arg: str) -> None:
    levels = list(get_args(Reasoning))
    cfg = app.runner.session.session_config.llm_config
    if arg:
        if arg not in levels:
            await app._notice(
                f"unknown reasoning level {arg!r}. Valid: {', '.join(levels)}",
                error=True,
            )
            return
        _apply(app, reasoning=arg)
        await app._notice(f"reasoning set to {arg}")
        return

    async def chosen(level: str | None) -> None:
        if level is None:
            return
        _apply(app, reasoning=level)
        await app._notice(f"reasoning set to {level}")

    app.push_screen(
        PickerScreen(
            "Select a reasoning level", levels,
            current=cfg.reasoning or "provider-default",
        ),
        chosen,
    )


async def _cmd_new(app: "AgentApp", arg: str) -> None:
    old_id = app.runner.session.id
    save_session(app.runner.session, app._session_dir)
    # Carry both halves of the session config forward, not just the model: the
    # runtime knobs (timeouts, step limits) are session-global and persisted.
    config = app.runner.session.session_config
    new = AgentSessionRunner.new_session(
        config.llm_config,
        runtime_config=config.runtime_config.model_copy(deep=True),
    )
    await app._reset_session(new)
    await app._notice(f"saved {old_id}, started new session {new.id}")


async def _cmd_quit(app: "AgentApp", arg: str) -> None:
    await app._quit()


COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("help", "", "show this help", _cmd_help),
    SlashCommand("model", "[provider:model]", "pick a provider then a model", _cmd_model),
    SlashCommand("reasoning", "[level]", "pick or set the reasoning level", _cmd_reasoning),
    SlashCommand("new", "", "save and start a fresh conversation", _cmd_new),
    SlashCommand("quit", "", "save and exit", _cmd_quit),
)

_BY_NAME = {c.name: c for c in COMMANDS}


async def dispatch(app: "AgentApp", text: str) -> bool:
    """Run `/name arg` if `name` is registered. Returns True when handled;
    False leaves the text for the caller to send as a normal message, so a
    path like `/etc/hosts` or a typo is never swallowed."""
    parts = text[1:].split(maxsplit=1)
    if not parts:
        return False
    command = _BY_NAME.get(parts[0])
    if command is None:
        return False
    arg = parts[1].strip() if len(parts) > 1 else ""
    await command.handler(app, arg)
    return True
