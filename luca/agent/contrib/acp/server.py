"""The stdio entry point: argparse in, a JSON-RPC conversation out.

Mirrors the TUI's flags where they mean the same thing, and drops every flag
that only makes sense with a screen attached. Nothing here writes to stdout —
that is the protocol's channel — and nothing writes to stderr either, since a
client commonly surfaces an agent's stderr as an error.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys

from luca.agent.contrib.app import DEFAULT_LOG_LEVEL, ENV_LOG_LEVEL, LucaConfigError

from .agent import LucaAgent


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m luca.agent.contrib.acp",
        description="Serve luca over the Agent Client Protocol on stdio.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Override the client's cwd. Rarely right: ACP says the client's cwd wins.",
    )
    parser.add_argument("--config", default=None, help="Use THIS luca.json instead of discovering one.")
    parser.add_argument("--log-level", default=None, help=f"Session log level, or OFF (default {DEFAULT_LOG_LEVEL}).")
    parser.add_argument(
        "--faux",
        action="store_true",
        help="Play the scripted offline conversation. No key, no network.",
    )
    parser.add_argument("--no-checkpoints", dest="checkpoints", action="store_false", default=True)
    parser.add_argument("--no-subagents", dest="subagents", action="store_false", default=True)
    parser.add_argument("--no-skills", dest="skills", action="store_false", default=True)
    parser.add_argument("--no-instructions", dest="instructions", action="store_false", default=True)
    parser.add_argument(
        "--no-commands",
        dest="commands",
        action="store_false",
        default=True,
        help="Ignore user-defined .md slash commands.",
    )
    return parser


def build_agent(args: argparse.Namespace) -> LucaAgent:
    provider = None
    if args.faux:
        from luca.agent.contrib.app import build_faux_provider

        provider = build_faux_provider()
    return LucaAgent(
        config_path=args.config,
        workspace=args.workspace,
        provider=provider,
        faux=args.faux,
        checkpoints=args.checkpoints,
        subagents=args.subagents,
        skills=args.skills,
        instructions=args.instructions,
        commands=args.commands,
        log_level=args.log_level,
    )


async def serve(agent: LucaAgent) -> None:
    """Speak the protocol on this process's stdin and stdout until the client
    goes away."""
    from acp import run_agent

    await run_agent(agent)


def main(argv: list[str] | None = None) -> None:
    import os

    args = arg_parser().parse_args(argv)
    if args.log_level is None:
        args.log_level = os.environ.get(ENV_LOG_LEVEL)
    try:
        agent = build_agent(args)
    except LucaConfigError as exc:
        # The one place a message may reach stderr: the protocol has not
        # started, so there is no stream to corrupt and no client to tell.
        print(f"luca: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(agent))


__all__ = ["arg_parser", "build_agent", "main", "serve"]
