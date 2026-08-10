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
    uv run python -m luca.agent.contrib.tui --no-use-native     # no provider-native tools
    uv run python -m luca.agent.contrib.tui --no-streaming      # block-level events
    uv run python -m luca.agent.contrib.tui --theme nord        # Textual theme
    uv run python -m luca.agent.contrib.tui --config ./ci.json  # use THIS config
    uv run python -m luca.agent.contrib.tui --log-level DEBUG   # verbose session log
    uv run python -m luca.agent.contrib.tui --no-skills         # ignore SKILL.md skills
    uv run python -m luca.agent.contrib.tui --no-commands       # ignore .md slash commands
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

luca logs to `<session dir>/logs/<session-id>.log` at INFO — errors carry the
traceback the session file cannot keep. `--log-level` (or `LUCA_LOG_LEVEL`, or
`logging.level` in the config) changes it, `OFF` writes nothing, and
`--log-file` moves it. Nothing is ever logged to stderr: the TUI is drawing
there.

Sessions persist to `~/.luca/projects/<encoded-project-path>/<session-id>.json`
after every run — one directory per project, so nothing lands next to your code.
`sessions.directory` in the config moves that root; the per-project
subdirectory is always applied under it. A bare `--resume` opens the picker on
start, `/resume` opens it mid-session, and `--resume <id>` goes straight to
one.

A real session needs a key for its provider. `~/.local/share/luca/auth.json`
(`$XDG_DATA_HOME` honored, `LUCA_AUTH_PATH` overrides) names one per provider:
`{"openrouter": {"type": "api", "key": "sk-or-…"}}`. It is read at boot, passed
to the runner, and never written to the session. A provider with no entry gets
no key and falls back to whatever env var `luca.client` knows for it
(OPENROUTER_API_KEY by default). `--faux` needs nothing — it plays back the
scripted demo conversation (one turn: a gated `multiply` call, then the
wrap-up).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from functools import partial
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

from luca.agent.contrib.prompts import InstructionsError
from luca.agent.core import AgentSessionRunner, Inf, RuntimeConfig, pretty_print
from luca.agent.core.models import AgentSession
from luca.client.catalog.refresh import main as refresh_catalog
from luca.client.types import Reasoning

from .app import DEFAULT_THEME, AgentApp
from .auth import api_key_for, credentials_for, load_auth, resolve_auth_path
from .config import (
    LucaConfig,
    LucaConfigError,
    apply_model_options,
    build_context_manager,
    build_permission_rules,
    check_provider_buildable,
    load_luca_config,
    pick,
    picker_models,
    resolve_config_path,
    resolve_llm_config,
    resolve_read_limits,
    resolve_runtime_config,
    validate_provider,
)
from .env_file import apply_env_file
from .sessions import fork_session, load_session, resolve_session_directory
from .wiring import build_faux_provider, default_model, faux_model

PICKER = ""
"""`--resume` given without an id: open the picker instead of loading one session."""

ENV_LOG_LEVEL = "LUCA_LOG_LEVEL"
"""Level for the session log, below `--log-level` and above `luca.json`."""

DEFAULT_LOG_LEVEL = "INFO"

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"

LOG_MAX_BYTES = 5_000_000
LOG_BACKUPS = 3

LOG_HANDLER_NAME = "luca-session-log"
"""Marks the handler this module owns, so re-running `setup_logging` replaces
its own and never a handler the embedding program installed."""


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
        "--refresh-models",
        action="store_true",
        help="Pull the model catalog from models.dev into the local cache, then exit.",
    )
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
        "--log-level",
        default=None,
        metavar="LEVEL",
        help=f"How much luca writes to the session log: DEBUG, INFO, WARNING, ERROR, "
        f"or OFF for no log file (default {DEFAULT_LOG_LEVEL}). Also settable as "
        f"{ENV_LOG_LEVEL}; the flag wins.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="Write the log HERE instead of <session dir>/logs/<session-id>.log.",
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
        "--use-native",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Offer the provider's own native tools where the model supports them "
        "(apply_patch + shell on OpenAI, text_editor + bash on Anthropic). On by "
        "default; --no-use-native keeps every model on the generic shell tools.",
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
        "--no-commands",
        dest="commands",
        action="store_false",
        default=True,
        help="Do not load user-defined slash commands (they are read from .claude/commands, "
        ".agents/commands and the ~ equivalents by default).",
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


def log_path(session_id: str, session_dir: Path, configured: str | None = None) -> Path:
    """Where this session's log goes: `<session dir>/logs/<session-id>.log`,
    beside the `<session-id>.json` it belongs to, or the configured path."""
    if configured is not None:
        return Path(configured).expanduser()
    return session_dir / "logs" / f"{session_id}.log"


def setup_logging(path: Path, level: str) -> None:
    """Point the `luca` logger at one rotating file.

    The TUI OWNS the stderr the app is drawing on, so nothing may log to it: the
    handler is attached to `luca` alone and `propagate` is turned off, which
    keeps luca's records away from a root handler the embedding program may have
    installed. `level="OFF"` writes no file at all.

    Idempotent — a second call REPLACES the handler the first one installed
    rather than stacking a second file on the logger."""
    _remove_log_handlers()
    if level.upper() == "OFF":
        return
    levels = logging.getLevelNamesMapping()
    if level.upper() not in levels:
        raise LucaConfigError(
            f"unknown log level {level!r}; expected one of {', '.join(sorted(levels))} or OFF.",
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS)
    except OSError as exc:
        raise LucaConfigError(f"cannot write the log to {path}: {exc}") from exc
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.set_name(LOG_HANDLER_NAME)
    log = logging.getLogger("luca")
    log.setLevel(levels[level.upper()])
    log.addHandler(handler)
    log.propagate = False


def _remove_log_handlers() -> None:
    """Detach and close the handlers THIS module installed, leaving the
    `NullHandler` and anything an embedding program added in place."""
    log = logging.getLogger("luca")
    for handler in [h for h in log.handlers if h.get_name() == LOG_HANDLER_NAME]:
        log.removeHandler(handler)
        handler.close()
    log.propagate = True


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
    # Native tools are the same kind of durable capability flag: every launch
    # writes it, so `--no-use-native` turns a resumed session that had them on.
    # It is read fresh at the top of every drive iteration, so it also decides
    # each individual call — a session can be flipped mid-run.
    session.session_config.use_native_tools = pick(
        getattr(args, "use_native", None),
        config.use_native_tools,
        True,
    )
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
    if args.refresh_models:
        # An explicit empty argv: the refresh has its own parser, and letting it
        # fall through to sys.argv would hand it this command's flags.
        raise SystemExit(refresh_catalog([]))
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
        # Before the config: a `.env` may carry `LUCA_CONFIG_PATH`. Skipped
        # under `--faux`, the offline path that needs no credential at all.
        if not args.faux:
            apply_env_file()
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
        session = build_session(args, config, store)
        # `--faux` injects a provider INSTANCE and never resolves a name, so it
        # needs neither a credential nor the reachability check. Otherwise both
        # run against the provider this launch will actually call — `--provider`,
        # the file and the resumed session are all folded in by `build_session`.
        auth: dict = {}
        if not args.faux:
            auth = load_auth(resolve_auth_path())
            llm_config = session.session_config.llm_config
            validate_provider(config, llm_config.provider)
            # And then BUILD it: a missing region or a half-written credential
            # lives in the provider's constructor, which otherwise runs first
            # on the opening message of the session.
            try:
                check_provider_buildable(
                    llm_config,
                    api_key=api_key_for(auth, llm_config.provider),
                    credentials=credentials_for(auth, llm_config.provider),
                )
            except LucaConfigError as exc:
                # Naming the way out matters most on a resume, where refusing
                # to start otherwise locks you out of your own transcript.
                raise LucaConfigError(
                    f"{exc}\n       (this session is on {llm_config.provider!r}; "
                    "--provider/--model opens it elsewhere, --faux works offline)"
                ) from exc
        # After build_session: the filename is the session's, and a resumed
        # session appends to the log it already has.
        setup_logging(
            log_path(session.id, store, pick(args.log_file, config.logging.file, None)),
            pick(
                args.log_level,
                os.environ.get(ENV_LOG_LEVEL) or config.logging.level,
                DEFAULT_LOG_LEVEL,
            ),
        )
        provider = build_faux_provider() if args.faux else None
        config_mode = config.permissions.mode.value if config.permissions.mode is not None else None
        app = AgentApp(
            session,
            provider=provider,
            auth=auth,
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
                api_key=api_key_for(auth, session.session_config.llm_config.provider),
                credentials=credentials_for(auth, session.session_config.llm_config.provider),
                enabled=args.autocompact,
                threshold=args.compact_threshold,
                keep_turns=args.compact_keep_turns,
            ),
            additional_directories=config.additional_directories or None,
            permission_rules=build_permission_rules(config) or None,
            recommended_models=picker_models(config) or None,
            model_options=partial(apply_model_options, config=config),
            subagents=args.subagents,
            skills=args.skills,
            extra_skill_locations=config.extra_skill_locations or None,
            commands=args.commands,
            extra_command_locations=config.extra_command_locations or None,
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
