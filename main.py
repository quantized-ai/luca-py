"""Interactive Textual TUI demo for luca.agent.

A thin launcher over `luca.agent.contrib.tui` — the full-screen agent chat:
transcript, live streaming, modal tool approvals, Esc cancellation, per-run
session persistence. The agent wiring (shell + memory plugins, the demo math
tools, one shared permission strategy) lives in the tui package.

Usage:
    uv run python main.py                          # start a fresh session
    uv run python main.py --faux                   # offline scripted demo, no key
    uv run python main.py --conversation <id>      # resume <id>.json
    uv run python main.py --conversation <id> --fork  # branch into a new session
    uv run python main.py --no-streaming           # block-level rendering

Requires a provider key (OPENROUTER_API_KEY by default) in env or .env,
except with `--faux`. Sessions persist to `<session-id>.json` in the current
directory after every run. Requires the `tui` dependency group (installed by
default with `uv sync`).
"""

from dotenv import load_dotenv

from luca.agent.contrib.tui import main

load_dotenv()

if __name__ == "__main__":
    main()
