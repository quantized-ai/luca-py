"""Interactive Textual TUI demo for luca.agent.

A thin launcher over `luca.agent.contrib.tui` — the full-screen agent chat:
transcript, live streaming, modal tool approvals, Esc cancellation, per-run
session persistence. The agent wiring (shell + memory plugins, the demo math
tools, one shared permission strategy) lives in the tui package. The subagent
tools are wired and the capability turned on for the session by default;
`--no-subagents` withholds both.

Usage:
    uv run python main.py                          # start a fresh session
    uv run python main.py --faux                   # offline scripted demo, no key
    uv run python main.py --conversation <id>      # resume <id>.json
    uv run python main.py --conversation <id> --fork  # branch into a new session
    uv run python main.py --no-subagents           # no parallel subagents
    uv run python main.py --subagents-max-depth 1  # no nesting (default 3)
    uv run python main.py --subagents-max-per-turn 5   # per-turn spawn budget
    uv run python main.py --subagents-max-workers 3    # how many work at once
    uv run python main.py --no-streaming           # block-level rendering
    uv run python main.py --theme nord             # override the Textual theme
    uv run python main.py --config ./ci.json       # use THIS config, skip discovery
    uv run python main.py --no-skills              # ignore SKILL.md skills
    uv run python main.py --conversation <id> --pretty-print  # transcript, then exit

Requires a provider key (OPENROUTER_API_KEY by default) in env or .env,
except with `--faux`. Sessions persist to `<session-id>.json` in the current
directory after every run. Requires the `tui` dependency group (installed by
default with `uv sync`).

Configuration comes from `./luca.json` over `~/.config/luca/luca.json`, unless
`--config <path>` (or `LUCA_CONFIG_PATH`) names one file to use instead of both.
"""

from dotenv import load_dotenv

from luca.agent.contrib.tui import main

load_dotenv()

if __name__ == "__main__":
    main()
