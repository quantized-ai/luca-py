"""Getting a session and its surroundings ready, before any front end exists.

Three things, in the order a launch needs them:

- `boot()` resolves the ENVIRONMENT — the `luca.json` in force, where this
  project's sessions live, the credential map, and the session log.
- `build_session()` produces the `AgentSession` itself: a fresh one, or one
  loaded off disk, with the configured and command-line values folded in.
- the logging helpers, which point the `luca` logger at ONE rotating file and
  nowhere else.

Nothing here parses a command line and nothing here draws. `cli.py` owns
argparse and hands the values in; the ACP server calls the same functions with
values off its own flags. Logging to a file rather than to a stream is not a
preference: the TUI is drawing on stderr and the ACP server is speaking
JSON-RPC on stdout, so a stream handler corrupts one or the other.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

from pydantic import ValidationError

from luca.agent.core import AgentSessionRunner, Inf, RuntimeConfig
from luca.agent.core.models import AgentSession, LLMConfig

from .auth import load_auth, resolve_auth_path
from .config import (
    LucaConfig,
    LucaConfigError,
    load_luca_config,
    pick,
    resolve_config_path,
    resolve_llm_config,
    resolve_runtime_config,
    validate_provider,
)
from .sessions import fork_session, load_session, resolve_session_directory
from .wiring import default_model, faux_model

ENV_LOG_LEVEL = "LUCA_LOG_LEVEL"
"""Level for the session log, below the command line and above `luca.json`."""

DEFAULT_LOG_LEVEL = "INFO"

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"

LOG_MAX_BYTES = 5_000_000
LOG_BACKUPS = 3

LOG_HANDLER_NAME = "luca-session-log"
"""Marks the handler this module owns, so re-running `setup_logging` replaces
its own and never a handler the embedding program installed."""


# ── the session log ──────────────────────────────────────────────────────────


def log_path(session_id: str, session_dir: Path, configured: str | None = None) -> Path:
    """Where this session's log goes: `<session dir>/logs/<session-id>.log`,
    beside the `<session-id>.json` it belongs to, or the configured path."""
    if configured is not None:
        return Path(configured).expanduser()
    return session_dir / "logs" / f"{session_id}.log"


def setup_logging(path: Path, level: str) -> None:
    """Point the `luca` logger at one rotating file.

    A FRONT END OWNS ITS STREAMS, so nothing may log to them: the handler is
    attached to `luca` alone and `propagate` is turned off, which keeps luca's
    records away from a root handler the embedding program may have installed.
    `level="OFF"` writes no file at all.

    Idempotent — a second call REPLACES the handler the first one installed
    rather than stacking a second file on the logger."""
    remove_log_handlers()
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


def remove_log_handlers() -> None:
    """Detach and close the handlers THIS module installed, leaving the
    `NullHandler` and anything an embedding program added in place."""
    log = logging.getLogger("luca")
    for handler in [h for h in log.handlers if h.get_name() == LOG_HANDLER_NAME]:
        log.removeHandler(handler)
        handler.close()
    log.propagate = True


# ── the session ──────────────────────────────────────────────────────────────


def build_session(
    *,
    config: LucaConfig | None = None,
    session_dir: str | os.PathLike[str] = ".",
    resume: str | None = None,
    fork: bool = False,
    faux: bool = False,
    model: str | None = None,
    provider: str | None = None,
    reasoning: str | None = None,
    use_native: bool | None = None,
    subagents: bool = True,
    subagents_max_depth: int = 3,
    subagents_max_per_turn: int | None = None,
    subagents_max_workers: int | None = None,
) -> AgentSession:
    """A session ready to be driven: fresh, or `resume`d off disk (optionally
    `fork`ed onto a new id), with `luca.json` and the caller's overrides folded
    into its `SessionConfig`.

    The `model` / `provider` / `reasoning` triple wins over the file, which
    wins over what a resumed session recorded — the same precedence
    `resolve_llm_config` applies everywhere."""
    config = config or LucaConfig()
    if resume:
        session = load_session(resume, session_dir)
        if fork:
            session = fork_session(session)
    else:
        llm: LLMConfig = faux_model() if faux else default_model()
        session = AgentSessionRunner.new_session(llm)
    session.session_config.llm_config = resolve_llm_config(
        session.session_config.llm_config,
        config,
        {"model": model, "provider": provider, "reasoning": reasoning},
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
    session.session_config.use_native_tools = pick(use_native, config.use_native_tools, True)
    try:
        session.session_config.runtime_config = RuntimeConfig.model_validate(
            {
                **session.session_config.runtime_config.model_dump(),
                "subagents_enabled": subagents,
                "subagents_max_depth": subagents_max_depth,
                "subagents_max_per_turn": Inf if subagents_max_per_turn is None else subagents_max_per_turn,
                "subagents_max_workers": Inf if subagents_max_workers is None else subagents_max_workers,
            }
        )
    except ValidationError as exc:
        raise LucaConfigError(f"invalid subagent flag value: {exc}") from exc
    return session


# ── the environment around it ────────────────────────────────────────────────


@dataclass(frozen=True)
class BootResult:
    """What a launch resolves before it has a session: the config in force,
    the workspace it applies to, and where that project's sessions live."""

    config: LucaConfig
    session_dir: Path
    workspace: str


def boot(*, workspace: str | None = None, config_path: str | None = None) -> BootResult:
    """Resolve the launch environment: `luca.json` and the session directory.

    Deliberately does NOT resolve credentials. The provider to validate comes
    off the session's `LLMConfig`, and the session cannot be built until this
    has run, so the two are separate calls in a fixed order:

        env = boot(workspace=..., config_path=...)
        session = build_session(config=env.config, session_dir=env.session_dir, ...)
        auth = credentials(env.config, session.session_config.llm_config)

    Raises `LucaConfigError` for anything a user can fix by editing a file, so
    a caller reports one readable line instead of a traceback."""
    config = load_luca_config(path=resolve_config_path(config_path))
    resolved_workspace = pick(workspace, config.workspace, ".")
    session_dir = resolve_session_directory(resolved_workspace, config.sessions.directory)
    return BootResult(config=config, session_dir=session_dir, workspace=resolved_workspace)


def credentials(config: LucaConfig, llm_config: LLMConfig | None, *, faux: bool = False) -> dict:
    """The credential map from `auth.json`, with this session's provider
    checked against the catalog.

    Empty for `faux`, which injects a provider INSTANCE and never resolves a
    name, so it needs neither a credential nor the reachability check."""
    if faux:
        return {}
    auth = load_auth(resolve_auth_path())
    if llm_config is not None:
        validate_provider(config, llm_config.provider)
    return auth
