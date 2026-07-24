"""Console entry point for the `luca` command.

Loads a local `.env` (if `python-dotenv` is installed), then launches the
Textual TUI in the current directory. Registered as the `luca` script and as
`python -m luca`. Kept deliberately thin: the argument parsing and the app
live in `luca.agent.contrib.tui`.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if "--version" in args:
        # Answered without importing the TUI, so it works without the [tui] extra.
        from luca import __version__

        print(f"luca {__version__}")
        return

    _load_dotenv()
    try:
        from luca.agent.contrib.tui.cli import main as tui_main
    except ModuleNotFoundError as exc:
        if exc.name == "textual":
            sys.stderr.write(
                "The luca TUI needs Textual. Install it with:\n"
                '    uv tool install "luca-ai[tui]"   # or: pipx install "luca-ai[tui]"\n'
            )
            raise SystemExit(1) from exc
        raise
    tui_main(args)


def _load_dotenv() -> None:
    """Pick up keys from a `.env` in the current directory. A no-op when
    `python-dotenv` is not installed — real environment variables still work."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(find_dotenv(usecwd=True))
