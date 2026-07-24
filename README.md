A small framework for building AI Agents

Luca's main focus is on an extensible core, a robust data model and extensibility.

## Install the `luca` command

Install as a tool so `luca` is on your `PATH` and works in any directory (needs
the `[tui]` extra for the terminal UI):

```bash
uv tool install ".[tui]"                 # from a clone of this repo
uv tool install "luca-ai[tui] @ git+https://github.com/quantized-ai/luca-py"   # from git
# pipx works too: pipx install ".[tui]"
```

`uv tool` and `pipx` both install into an isolated environment, the same way
`claude` or `opencode` install. Once published to PyPI the command is just
`uv tool install "luca-ai[tui]"`.

## Run

```bash
luca                       # open the TUI in the current directory
luca --faux                # offline demo, no API key needed
luca --model anthropic:claude-sonnet-5
luca --conversation <id>   # resume a saved session
luca --version
```

`luca` reads provider keys from the environment or a `.env` in the current
directory (add `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, … there). Sessions
persist to `<session-id>.json` in the directory you run it from.

### Developing in this repo

Without installing the tool, run it straight from the checkout:

```bash
uv run luca --help          # or: uv run python main.py --help
uv run python -m luca --faux
```
