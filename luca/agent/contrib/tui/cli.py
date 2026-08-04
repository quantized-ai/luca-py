"""Command-line entry point for the TUI.

    uv run python -m luca.agent.contrib.tui                     # fresh session
    uv run python -m luca.agent.contrib.tui --faux              # offline, scripted
    uv run python -m luca.agent.contrib.tui --no-subagents      # no parallel subagents
    uv run python -m luca.agent.contrib.tui --subagents-max-depth 1   # no nesting
    uv run python -m luca.agent.contrib.tui --subagents-max-per-turn 5
    uv run python -m luca.agent.contrib.tui --subagents-max-workers 3
    uv run python -m luca.agent.contrib.tui --resume            # pick a past session
    uv run python -m luca.agent.contrib.tui --resume <id>       # resume it by id
    uv run python -m luca.agent.contrib.tui --resume <id> --fork
    uv run python -m luca.agent.contrib.tui --no-streaming      # block-level events
    uv run python -m luca.agent.contrib.tui --theme nord        # Textual theme
    uv run python -m luca.agent.contrib.tui --config ./ci.json  # use THIS config
    uv run python -m luca.agent.contrib.tui --no-skills         # ignore SKILL.md skills
    uv run python -m luca.agent.contrib.tui --no-instructions   # ignore AGENTS.md
    uv run python -m luca.agent.contrib.tui \
        --model moonshotai/kimi-k2.7-code --reasoning high
    uv run python -m luca.agent.contrib.tui \
        --resume <id> --pretty-print                            # print and exit

`--pretty-print` replaces the TUI entirely: it loads `<id>.json`, writes the
`pretty_print` transcript to stdout and exits without starting the app, so it
requires `--resume <id>` and ignores every other flag — config included, since
a transcript does not depend on it.

Configuration layers, highest precedence first: CLI flags, then the nearest
`luca.json` at or above the cwd (repo policy), then `~/.config/luca/luca.json`
(personal defaults), then the persisted session, then built-in defaults.
`--config <path>` (or the `LUCA_CONFIG_PATH` env var, which the flag overrides)
REPLACES both file layers with the one named file. See `config.py` and the docs.

The system prompt is assembled by `contrib/prompts`: a base prompt chosen for
the model's family, an environment block, and the project's instruction files
(`LUCA.md` / `AGENTS.md` / `CLAUDE.md`, one per directory from the git root down
to the workspace). `--no-instructions` withholds the last of those.

Sessions persist to `~/.luca/projects/<encoded-project-path>/<session-id>.json`
after every run — one directory per project, so nothing lands next to your code.
`sessions.directory` in the config moves that root; the per-project
subdirectory is always applied under it. A bare `--resume` opens the picker on
start, `/resume` opens it mid-session, and `--resume <id>` goes straight to
one. A real session needs a provider key (OPENROUTER_API_KEY by default) in
the environment; `--faux` needs nothing — it plays back the scripted demo
conversation (one turn: a gated `multiply` call, then the wrap-up).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

from luca.agent.contrib.prompts import InstructionsError
from luca.agent.core import AgentSessionRunner, Inf, RuntimeConfig, pretty_print
from luca.agent.core.models import AgentSession
from luca.client.types import Reasoning

from .app import DEFAULT_THEME, AgentApp
from .config import (
    LucaConfig,
    LucaConfigError,
    build_context_manager,
    build_permission_rules,
    load_luca_config,
    pick,
    register_config_providers,
    resolve_config_path,
    resolve_llm_config,
    resolve_read_limits,
    resolve_runtime_config,
)
from .sessions import fork_session, load_session, resolve_session_directory
from .wiring import build_faux_provider, default_model, faux_model

PICKER = ""
"""`--resume` given without an id: open the picker instead of loading one session."""


def resume_id(args: argparse.Namespace) -> str | None:
    """The session id named by `--resume <id>`. None for a bare `--resume`
    (which opens the picker) and for no flag at all (a fresh session)."""
    return args.resume or None


def resume_picker(args: argparse.Namespace) -> bool:
    """Whether `--resume` was given bare, i.e. asking for the picker."""
    return args.resume == PICKER


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="luca.agent Textual TUI")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=PICKER,
        default=None,
        metavar="ID",
        help="Resume a session: `--resume <id>` loads <id>.json directly, bare "
        "`--resume` opens the picker for this project (the same as /resume).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a config file to use INSTEAD of the discovered luca.json "
        "and ~/.config/luca/luca.json. Also settable as LUCA_CONFIG_PATH; the flag wins.",
    )
    parser.add_argument(
        "--fork",
        action="store_true",
        help="Fork the loaded session into a new id.",
    )
    parser.add_argument(
        "--pretty-print",
        action="store_true",
        help="Print the loaded conversation as a text transcript and exit "
        "instead of starting the TUI. Requires --resume <id>.",
    )
    parser.add_argument(
        "--gallery",
        nargs="?",
        const="all",
        default=None,
        metavar="STATE",
        help="Boot the design-system gallery instead of a live agent: a catalog "
        "entry by name (e.g. chat/subagents), a bundled fixture by name (e.g. "
        "1a_agent_loop), a path to a fixture or a stored session, or no value "
        "to browse them all.",
    )
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Live token deltas (--no-streaming for block-level events).",
    )
    parser.add_argument(
        "--no-subagents",
        dest="subagents",
        action="store_false",
        default=True,
        help="Stop the agent from spawning subagents (parallel subagents are on by default).",
    )
    parser.add_argument(
        "--subagents-max-depth",
        type=int,
        default=3,
        help="How many levels of subagents may nest below the main conversation (default 3; the library default is 1).",
    )
    parser.add_argument(
        "--subagents-max-per-turn",
        type=int,
        default=None,
        help="How many subagents one conversation may spawn in one turn (default: no limit).",
    )
    parser.add_argument(
        "--subagents-max-workers",
        type=int,
        default=None,
        help="How many subagents may be doing work at the same time, across the whole session (default: no limit).",
    )
    parser.add_argument(
        "--no-skills",
        dest="skills",
        action="store_false",
        default=True,
        help="Do not load SKILL.md skills (they are read from .claude/skills, .agents/skills "
        "and the ~ equivalents by default).",
    )
    parser.add_argument(
        "--no-instructions",
        dest="instructions",
        action="store_false",
        default=True,
        help="Do not read the project's LUCA.md / AGENTS.md / CLAUDE.md into the system prompt.",
    )
    parser.add_argument(
        "--faux",
        action="store_true",
        help="No network: drive the scripted offline demo conversation.",
    )
    parser.add_argument(
        "--model",
        help="Model id for the session (e.g. moonshotai/kimi-k2.7-code). "
        "Overrides luca.json and the resumed session's model.",
    )
    parser.add_argument(
        "--provider",
        help="Provider name for the session (e.g. openrouter, anthropic).",
    )
    parser.add_argument(
        "--reasoning",
        choices=list(get_args(Reasoning)),
        help="Reasoning level for the model.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Shell workspace root (default: the current directory).",
    )
    parser.add_argument(
        "--theme",
        default=None,
        help=f"Textual theme name (default: {DEFAULT_THEME}).",
    )
    parser.add_argument(
        "--mode",
        choices=["ask", "yolo", "auto"],
        default=None,
        help="Tool-approval mode.",
    )
    parser.add_argument(
        "--autocompact",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Automatic compaction (--no-autocompact to disable; /compact stays).",
    )
    parser.add_argument(
        "--compact-threshold",
        type=float,
        default=None,
        help="Auto-compact when context utilization reaches this fraction.",
    )
    parser.add_argument(
        "--compact-keep-turns",
        type=int,
        default=None,
        help="Keep the last N exchanges verbatim when compacting (0 = summary only).",
    )
    return parser


def build_session(
    args: argparse.Namespace,
    config: LucaConfig | None = None,
    session_dir: Path | None = None,
) -> AgentSession:
    config = config or LucaConfig()
    session_id = resume_id(args)
    if session_id:
        session = load_session(session_id, session_dir or ".")
        if args.fork:
            session = fork_session(session)
    else:
        model = faux_model() if args.faux else default_model()
        session = AgentSessionRunner.new_session(model)
    cli = {"model": args.model, "provider": args.provider, "reasoning": args.reasoning}
    session.session_config.llm_config = resolve_llm_config(
        session.session_config.llm_config,
        config,
        cli,
    )
    session.session_config.runtime_config = resolve_runtime_config(
        session.session_config.runtime_config,
        config,
    )
    # The flags set the CAPABILITY on this session, durably — including on a
    # resumed one, so `--no-subagents` turns a session that had them off;
    # wiring the plugin only makes the tools available to a session that asked.
    # Same rule for the subagent limits: every launch writes them, so the flags
    # always describe the run you are starting now. Written through
    # `model_validate` rather than attribute assignment: an invalid flag value
    # (`--subagents-max-workers 0`) must fail loudly here, not wedge the first
    # spawn and poison the saved session file.
    per_turn = getattr(args, "subagents_max_per_turn", None)
    max_workers = getattr(args, "subagents_max_workers", None)
    try:
        session.session_config.runtime_config = RuntimeConfig.model_validate(
            {
                **session.session_config.runtime_config.model_dump(),
                "subagents_enabled": getattr(args, "subagents", True),
                "subagents_max_depth": getattr(args, "subagents_max_depth", 3),
                "subagents_max_per_turn": Inf if per_turn is None else per_turn,
                "subagents_max_workers": Inf if max_workers is None else max_workers,
            }
        )
    except ValidationError as exc:
        raise LucaConfigError(f"invalid subagent flag value: {exc}") from exc
    return session


def main(argv: list[str] | None = None) -> None:
    parser = arg_parser()
    args = parser.parse_args(argv)
    if args.gallery is not None:
        from .gallery import FixtureError, run_gallery

        try:
            run_gallery(args.gallery)
        except FixtureError as exc:
            sys.stderr.write(f"luca: {exc}\n")
            raise SystemExit(1) from exc
        return
    if args.pretty_print and not resume_id(args):
        parser.error("--pretty-print requires --resume <id>.")
    # Building the app is inside the try: composing it resolves the config's
    # `instructions` paths, and a typo there has to exit readably like any
    # other bad config value rather than traceback.
    try:
        config = load_luca_config(path=resolve_config_path(args.config))
        # Resolved once and threaded everywhere: `build_session` loads before
        # the app exists, and `--pretty-print` never builds one at all.
        store = resolve_session_directory(
            pick(args.workspace, config.workspace, "."),
            config.sessions.directory,
        )
        if args.pretty_print:
            print(pretty_print(load_session(resume_id(args), store)))
            return
        register_config_providers(config)
        session = build_session(args, config, store)
        provider = build_faux_provider() if args.faux else None
        config_mode = config.permissions.mode.value if config.permissions.mode is not None else None
        app = AgentApp(
            session,
            provider=provider,
            session_dir=store,
            resume=resume_picker(args),
            read_limits=resolve_read_limits(config),
            theme=pick(args.theme, config.theme.name, DEFAULT_THEME),
            streaming=pick(args.streaming, config.streaming, True),
            workspace=pick(args.workspace, config.workspace, "."),
            mode=pick(args.mode, config_mode, "ask"),
            context_manager=build_context_manager(
                config,
                provider=provider,
                enabled=args.autocompact,
                threshold=args.compact_threshold,
                keep_turns=args.compact_keep_turns,
            ),
            additional_directories=config.additional_directories or None,
            permission_rules=build_permission_rules(config) or None,
            recommended_models=config.models or None,
            subagents=args.subagents,
            skills=args.skills,
            extra_skill_locations=config.extra_skill_locations or None,
            instructions=args.instructions,
            extra_instructions=config.instructions or None,
        )
    except (LucaConfigError, InstructionsError) as exc:
        sys.stderr.write(f"luca: {exc}\n")
        raise SystemExit(1) from exc

    app.run()
    print(f"Goodbye! Pick this session back up with `python main.py --resume` ({app.runner.session.id})")


if __name__ == "__main__":
    main()
