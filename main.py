"""Interactive Textual TUI demo for luca.agent.

A thin launcher over `luca.agent.contrib.tui` — the full-screen agent chat:
transcript, live streaming, modal tool approvals, Esc cancellation, per-run
session persistence. The agent wiring (shell + memory plugins, the demo math
tools, one shared permission strategy) lives in the tui package. The subagent
tools are wired and the capability turned on for the session by default;
`--no-subagents` withholds both. Provider-native tools are on by default too:
a model that supports them gets its provider's own `apply_patch`/`shell` or
`text_editor`/`bash` in place of the generic shell tools those replace, and
`--no-use-native` keeps every model on the generic set.

Usage:
    uv run python main.py                          # start a fresh session
    uv run python main.py --faux                   # offline scripted demo, no key
    uv run python main.py --resume                 # pick a past session
    uv run python main.py --resume <id>            # resume that session by id
    uv run python main.py --resume <id> --fork     # branch into a new session
    uv run python main.py --no-subagents           # no parallel subagents
    uv run python main.py --subagents-max-depth 1  # no nesting (default 3)
    uv run python main.py --subagents-max-per-turn 5   # per-turn spawn budget
    uv run python main.py --subagents-max-workers 3    # how many work at once
    uv run python main.py --no-use-native          # generic shell tools only
    uv run python main.py --no-streaming           # block-level rendering
    uv run python main.py --theme luca-dark        # the design theme (default)
    uv run python main.py --gallery                # browse the design-system catalog
    uv run python main.py --gallery chat/subagents # one catalog entry
    uv run python main.py --gallery 1c_diff_approval   # one hand-authored fixture
    uv run python main.py --config ./ci.json       # use THIS config, skip discovery
    uv run python main.py --log-level DEBUG        # verbose session log
    uv run python main.py --no-skills              # ignore SKILL.md skills
    uv run python main.py --no-instructions        # ignore AGENTS.md
    uv run python main.py --resume <id> --pretty-print  # transcript, then exit

Requires a provider credential, except with `--faux`: `auth.json`, or an
exported environment variable (OPENROUTER_API_KEY by default). A `.env` is NOT
read. The provider is built at boot, so a missing region or a half-written
credential is a startup error rather than a failed first message. Sessions
persist to
`~/.luca/projects/<encoded-project-path>/<session-id>.json` after every run —
one directory per project, so nothing is written next to your code. Requires the `tui` dependency group (installed by
default with `uv sync`).

Configuration comes from `./luca.json` over `~/.config/luca/luca.json`, unless
`--config <path>` (or `LUCA_CONFIG_PATH`) names one file to use instead of both.

Each session logs to `~/.luca/projects/<encoded-project-path>/logs/<id>.log` at
INFO; `--log-level DEBUG` (or `LUCA_LOG_LEVEL`) turns it up, `OFF` turns it off.

The system prompt is picked for the model's family and carries the project's
`LUCA.md` / `AGENTS.md` / `CLAUDE.md`; `--no-instructions` drops the latter.
"""

from luca.agent.contrib.tui import main

if __name__ == "__main__":
    main()
