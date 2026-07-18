"""Command-line entry point for the TUI.

    uv run python -m luca.agent.contrib.tui                     # fresh session
    uv run python -m luca.agent.contrib.tui --faux              # offline, scripted
    uv run python -m luca.agent.contrib.tui --conversation <id> # resume <id>.json
    uv run python -m luca.agent.contrib.tui --conversation <id> --fork
    uv run python -m luca.agent.contrib.tui --no-streaming      # block-level events
    uv run python -m luca.agent.contrib.tui \
        --model moonshotai/kimi-k2.7-code --reasoning-effort high

`--model` / `--provider` / `--reasoning-effort` update the session's
`LLMConfig` (defaults: openrouter, or faux under `--faux`); the config
persists with the session, and passing any of these flags on a resume
overrides the stored value.

Sessions persist to `<session-id>.json` in the working directory after every
run. A real session needs a provider key (OPENROUTER_API_KEY by default) in
the environment; `--faux` needs nothing — it plays back the scripted demo
conversation (one turn: a gated `multiply` call, then the wrap-up).
"""

from __future__ import annotations

import argparse

from luca.agent.core import AgentSessionRunner
from luca.agent.core.models import AgentSession

from .app import AgentApp
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
        "--no-streaming", action="store_true",
        help="Consume block-level events instead of live token deltas.",
    )
    parser.add_argument(
        "--faux", action="store_true",
        help="No network: drive the scripted offline demo conversation.",
    )
    parser.add_argument(
        "--model",
        help="Model id for the session (e.g. moonshotai/kimi-k2.7-code). "
             "Persists with the session; on --conversation it overrides the "
             "resumed session's model.",
    )
    parser.add_argument(
        "--provider",
        help="Provider name for the session (e.g. openrouter, anthropic). "
             "Passed to the LLMConfig as-is. Persists like --model.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "auto"],
        help="Reasoning effort for the model. Persists like --model.",
    )
    return parser


def build_session(args: argparse.Namespace) -> AgentSession:
    if args.conversation:
        session = load_session(args.conversation)
        if args.fork:
            session = fork_session(session)
    else:
        model = faux_model() if args.faux else default_model()
        session = AgentSessionRunner.new_session(model)
    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.provider:
        overrides["provider"] = args.provider
    if args.reasoning_effort:
        overrides["reasoning_effort"] = args.reasoning_effort
    if overrides:
        session.session_config.llm_config = (
            session.session_config.llm_config.model_copy(update=overrides)
        )
    return session


def main(argv: list[str] | None = None) -> None:
    args = arg_parser().parse_args(argv)
    session = build_session(args)
    provider = build_faux_provider() if args.faux else None
    app = AgentApp(
        session,
        provider=provider,
        streaming=not args.no_streaming,
    )
    app.run()
    print(
        f"Goodbye! Resume session with "
        f"`python main.py --conversation {app.runner.session.id}`"
    )


if __name__ == "__main__":
    main()
