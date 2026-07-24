"""Command-line entry point for the TUI.

    uv run python -m luca.agent.contrib.tui                     # fresh session
    uv run python -m luca.agent.contrib.tui --faux              # offline, scripted
    uv run python -m luca.agent.contrib.tui --conversation <id> # resume <id>.json
    uv run python -m luca.agent.contrib.tui --conversation <id> --fork
    uv run python -m luca.agent.contrib.tui --no-streaming      # block-level events
    uv run python -m luca.agent.contrib.tui \
        --model moonshotai/kimi-k2.7-code --reasoning high

Configuration layers, highest precedence first: CLI flags, then `./luca.json`
(repo policy), then `~/.config/luca/luca.json` (personal defaults), then the
persisted session, then built-in defaults. See `config.py` and the docs.

Sessions persist to `<session-id>.json` in the working directory after every
run. A real session needs a provider key (OPENROUTER_API_KEY by default) in
the environment; `--faux` needs nothing — it plays back the scripted demo
conversation (one turn: a gated `multiply` call, then the wrap-up).
"""

from __future__ import annotations

import argparse
import sys
from typing import get_args

from luca.agent.core import AgentSessionRunner
from luca.agent.core.models import AgentSession
from luca.client.types import Reasoning

from .app import AgentApp
from .config import (
    LucaConfig,
    LucaConfigError,
    build_compactor,
    build_permission_rules,
    load_luca_config,
    pick,
    register_config_providers,
    resolve_llm_config,
    resolve_runtime_config,
)
from .sessions import fork_session, load_session
from .wiring import build_faux_provider, default_model, faux_model


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="luca.agent Textual TUI")
    parser.add_argument("--conversation", help="Session id to load (<id>.json).")
    parser.add_argument(
        "--fork", action="store_true",
        help="Fork the loaded session into a new id.",
    )
    parser.add_argument(
        "--streaming", action=argparse.BooleanOptionalAction, default=None,
        help="Live token deltas (--no-streaming for block-level events).",
    )
    parser.add_argument(
        "--faux", action="store_true",
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
        "--workspace", default=None,
        help="Shell workspace root (default: the current directory).",
    )
    parser.add_argument(
        "--mode", choices=["ask", "yolo", "auto"], default=None,
        help="Tool-approval mode.",
    )
    parser.add_argument(
        "--autocompact", action=argparse.BooleanOptionalAction, default=None,
        help="Automatic compaction (--no-autocompact to disable; /compact stays).",
    )
    parser.add_argument(
        "--compact-threshold", type=float, default=None,
        help="Auto-compact when context utilization reaches this fraction.",
    )
    parser.add_argument(
        "--compact-keep-turns", type=int, default=None,
        help="Keep the last N exchanges verbatim when compacting (0 = summary only).",
    )
    return parser


def build_session(args: argparse.Namespace, config: LucaConfig | None = None) -> AgentSession:
    config = config or LucaConfig()
    if args.conversation:
        session = load_session(args.conversation)
        if args.fork:
            session = fork_session(session)
    else:
        model = faux_model() if args.faux else default_model()
        session = AgentSessionRunner.new_session(model)
    cli = {"model": args.model, "provider": args.provider, "reasoning": args.reasoning}
    session.session_config.llm_config = resolve_llm_config(
        session.session_config.llm_config, config, cli,
    )
    session.session_config.runtime_config = resolve_runtime_config(
        session.session_config.runtime_config, config,
    )
    return session


def main(argv: list[str] | None = None) -> None:
    args = arg_parser().parse_args(argv)
    try:
        config = load_luca_config()
        register_config_providers(config)
        session = build_session(args, config)
    except LucaConfigError as exc:
        sys.stderr.write(f"luca: {exc}\n")
        raise SystemExit(1)

    provider = build_faux_provider() if args.faux else None
    config_mode = (
        config.permissions.mode.value if config.permissions.mode is not None else None
    )
    app = AgentApp(
        session,
        provider=provider,
        streaming=pick(args.streaming, config.streaming, True),
        workspace=pick(args.workspace, config.workspace, "."),
        mode=pick(args.mode, config_mode, "ask"),
        compactor=build_compactor(
            config,
            enabled=args.autocompact,
            threshold=args.compact_threshold,
            keep_turns=args.compact_keep_turns,
        ),
        additional_directories=config.additional_directories or None,
        permission_rules=build_permission_rules(config) or None,
        recommended_models=config.models or None,
    )
    app.run()
    print(
        f"Goodbye! Resume session with "
        f"`python main.py --conversation {app.runner.session.id}`"
    )


if __name__ == "__main__":
    main()
