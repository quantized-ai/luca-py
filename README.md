A small framework for building AI Agents

Luca's main focus is on an extensible core, a robust data model and extensibility.

### TUI
The TUI agent is a work in progress and just a demonstration of Luca's architecture. Start with:

```bash
uv run python main.py --help
```

To use it first add your keys to your `.env` and then run:

```bash
uv run python main.py --model moonshotai/kimi-k2.7-code --reasoning-effort high --provider openrouter
```
