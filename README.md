A small framework for building AI Agents

Luca's main focus is on an extensible core, a robust data model and extensibility.

### TUI
The TUI agent is a work in progress and just a demonstration of Luca's architecture. Start with:

```bash
uv run python main.py --help
```

To use it, export a provider key and then run:

```bash
export OPENROUTER_API_KEY=sk-or-...
uv run python main.py --model moonshotai/kimi-k2.7-code --reasoning high --provider openrouter
```

Nothing reads a `.env` — export the variables yourself, or put the key in
`~/.local/share/luca/auth.json`.

### Contributing

See CONTRIBUTING.md.

Set up your env:

```bash
$ uv sync
$ uv run pre-commit install
```

Run tests
```bash
$ uv run py.test tests/
# or with pytest-xdist
$ uv run py.test tests/ -n {n} # make sure `n < auto` as some tests are time-bound
```

