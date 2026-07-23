# Installation

`luca` targets **Python 3.11+** and is managed with
[`uv`](https://docs.astral.sh/uv/). The only runtime dependencies are
[`httpx`](https://www.python-httpx.org/) and
[`pydantic`](https://docs.pydantic.dev/) — the SDK does not wrap any vendor
SDK.

## In this repo

```bash
# install deps + project (uv reads pyproject.toml)
uv sync

# run the test suite
uv run py.test tests/

# run the demo agent loop
uv run python main.py
uv run python main.py --streaming
```

`uv run <cmd>` is the easiest way to invoke anything inside the project's
virtualenv. Direct `python` calls also work after `uv sync`.

## Environment variables

Providers read their API keys from environment variables by default. A
`.env` file in the project root is picked up by `python-dotenv` in the demo
scripts; the SDK itself reads `os.environ` directly.

| Provider | Env var |
|---|---|
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `bedrock` | `AWS_BEARER_TOKEN_BEDROCK` (+ `BEDROCK_AWS_REGION`) |
| `groq` | `GROQ_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `ollama` | (none — local) |

Per-call overrides always win: pass `api_key=` to a helper or construct a
provider instance directly with `api_key="…"`.

## Dependencies

The runtime install graph is intentionally tiny:

- `httpx>=0.28.1` — every transport talks directly over HTTPS.
- `pydantic>=2` — every DTO is a Pydantic model.

The dev group adds `pytest`, `pytest-asyncio`, and `python-dotenv` for the
demos. See [`pyproject.toml`](../../pyproject.toml) for the exact constraints.

If you ever find yourself adding a heavy dependency, push back — a new
provider should be a one-line `PROVIDERS` entry or a small transport
subclass, not a new package.
