"""Slash commands over ACP: what to advertise, and what to do when one arrives.

ACP has no method for invoking a command. The agent advertises a list through
`available_commands_update`, and the client sends the invocation back as
ORDINARY PROMPT TEXT beginning with `/name`. So a command is a parse of the
first text block, not a separate entry point — and a client that never saw the
advertisement can still type one, which is why `dispatch` re-parses rather than
trusting that a `/` means something.

WHICH COMMANDS EXIST HERE is a deliberate subset of the TUI's sixteen. Most of
those are Textual: `/theme` and `/quit` are meaningless to a client that owns
its own window, `/session` and `/clear` belong to whatever manages threads, and
`/rewind` and `/context` are pickers. Three carry over because they are
front-end-neutral, and the user's own `.md` commands carry over because a
custom command is a saved prompt and nothing else.

`/model` and `/reasoning` are deliberately NOT here. They are session settings,
and ACP has `session/set_config_option` for exactly that; putting them behind a
slash would be the wrong shape in a client that wants to render a picker.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from acp.schema import AvailableCommand, AvailableCommandInput, UnstructuredCommandInput

from luca.agent.contrib.app import (
    AgentApplication,
    CustomCommand,
    discover_commands,
    estimated_cost,
    expand,
    resolve_locations,
    usage_totals,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Invocation:
    """A parsed `/name args`. `name` is without the slash."""

    name: str
    args: str


@dataclass(frozen=True)
class Prompt:
    """Send this to the model as an ordinary user message. What every custom
    command produces, and the reason a command needs no new machinery: the
    transcript shows the expanded text, so a reader sees what was actually
    asked."""

    text: str


@dataclass(frozen=True)
class Reply:
    """Answer the client directly, with no model call. For a command that
    reports rather than asks."""

    text: str


@dataclass(frozen=True)
class Command:
    name: str
    description: str
    hint: str | None
    run: Callable[[AgentApplication, str], Prompt | Reply]


_INVOCATION = re.compile(r"^/([A-Za-z0-9][A-Za-z0-9_-]*)(?:\s+(.*))?$", re.DOTALL)
"""A bare word after the slash, then whitespace, then anything.

The name class is what disarms a path: `/tmp/notes.txt` fails because the
character after `tmp` is a slash rather than whitespace, so the whole line is
prose. `//comment` fails on the first character. A LEADING SPACE also disarms
it, which is the escape hatch clients tell users about."""


def parse(text: str) -> Invocation | None:
    """`/name args` as an invocation, or None for ordinary prose.

    The argument may span lines — a client's composer is multi-line, and
    `/review\nthese three files` is one invocation with a three-word argument,
    not a bare `/review`."""
    match = _INVOCATION.match(text.lstrip("\n"))
    if match is None:
        return None
    return Invocation(name=match.group(1), args=(match.group(2) or "").strip())


# ── the three portable built-ins ─────────────────────────────────────────────


def _compact(application: AgentApplication, args: str) -> Reply:
    """Schedule a compaction. The runner performs it at the top of the next
    drive, so this only has to ask."""
    application.runner.schedule_compaction()
    return Reply("Compacting the conversation. The next reply will run against the summary.")


def _cost(application: AgentApplication, args: str) -> Reply:
    session = application.session
    totals = usage_totals(session)
    cost = estimated_cost(session)
    lines = [
        f"**{session.session_config.llm_config.model}** on {session.session_config.llm_config.provider}",
        "",
        f"- input: {totals.input:,}",
        f"- output: {totals.output:,}",
        f"- cache read: {totals.cache_read:,}",
        f"- cache write: {totals.cache_write:,}",
        f"- total: {totals.total:,}",
    ]
    # None means the catalog does not price this model, which is not the same
    # as free. Saying "$0.00" for an unpriced model would be a lie.
    lines.append(f"- estimated cost: ${cost:.4f}" if cost is not None else "- estimated cost: not priced")
    return Reply("\n".join(lines))


def render_help(commands: tuple[Command, ...]) -> str:
    """Most clients render their own list from the advertisement, but not all
    do, and a user who cannot find the list has no way to discover the
    commands at all."""
    if not commands:
        return "No commands are available in this session."
    # The padding goes OUTSIDE the backticks: inside, a client renders the
    # trailing spaces as part of the code span.
    width = max(len(command.name) for command in commands) + 2
    rows = [
        f"- `/{command.name}`".ljust(width + 6) + " ".join(filter(None, [command.hint, command.description]))
        for command in commands
    ]
    return "\n".join(["Available commands:", "", *rows])


# `help` is not here: it has to name the whole registry, including the user's
# own commands, so it is bound in `build()` once that registry exists.
BUILTINS: tuple[Command, ...] = (
    Command("compact", "summarize the history and continue", None, _compact),
    Command("cost", "token and spend totals for this session", None, _cost),
)

HELP_NAME = "help"

BUILTIN_NAMES = frozenset({*(command.name for command in BUILTINS), HELP_NAME})


# ── the user's own ───────────────────────────────────────────────────────────


def _custom(command: CustomCommand) -> Command:
    return Command(
        name=command.name,
        description=command.description,
        hint=command.argument_hint or None,
        run=lambda _application, args, body=command.body: Prompt(expand(body, args)),
    )


def discover(workspace, extra_locations: list[str] | None = None) -> tuple[Command, ...]:
    """The user's `.md` commands, as commands.

    NEVER RAISES: a broken commands directory costs the user their own
    commands, not their session. `reserved=` is what stops a file shadowing a
    built-in, exactly as the TUI does it — the two front ends reserve different
    names because they offer different built-ins, which is precisely why this
    set is not shared."""
    try:
        found = discover_commands(resolve_locations(workspace, extra_locations), reserved=BUILTIN_NAMES)
    except (OSError, RuntimeError, ValueError):
        logger.warning("could not read the user-defined commands; continuing with the built-ins", exc_info=True)
        return ()
    return tuple(_custom(command) for command in found.values())


# ── the registry, per session ────────────────────────────────────────────────


def build(workspace, extra_locations: list[str] | None = None, *, enabled: bool = True) -> tuple[Command, ...]:
    """Every command this session offers, built-ins first.

    `help` closes over the finished registry — including itself and the user's
    own commands, which is what makes it list everything rather than only what
    was defined before it."""
    if not enabled:
        return ()
    registry: list[Command] = []
    help_command = Command(
        HELP_NAME,
        "list the available commands",
        None,
        lambda _application, _args: Reply(render_help(tuple(registry))),
    )
    # Built-ins first, `help` closing them, then the user's own. A stable order
    # matters: it is what a client renders in its palette, and one that
    # reshuffles as files appear on disk is disorienting.
    registry.extend([*BUILTINS, help_command, *discover(workspace, extra_locations)])
    return tuple(registry)


def advertisement(commands: tuple[Command, ...]) -> list[AvailableCommand]:
    """The list a client renders in its own command palette."""
    return [
        AvailableCommand(
            name=command.name,
            description=command.description,
            input=AvailableCommandInput(UnstructuredCommandInput(hint=command.hint)) if command.hint else None,
        )
        for command in commands
    ]


def dispatch(
    application: AgentApplication,
    commands: tuple[Command, ...],
    invocation: Invocation,
) -> Prompt | Reply | None:
    """Run the named command, or None when nothing has that name.

    None is NOT an error here. A user can type `/anything`, and prose that
    happens to start with a slash should reach the model rather than being
    refused — the client already refuses the names it knows nothing about."""
    for command in commands:
        if command.name == invocation.name:
            return command.run(application, invocation.args)
    return None
