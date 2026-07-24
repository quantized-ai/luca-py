"""Dev launcher for the luca TUI: `uv run python main.py [flags]`.

Delegates to the same entry the installed `luca` command uses
(`luca.cli:main`): load a local `.env`, then launch the Textual TUI. Run
`uv run python main.py --help` for the flags, or install the command with
`uv tool install ".[tui]"` and run `luca` in any directory.
"""

from luca.cli import main

if __name__ == "__main__":
    main()
