# AGENTS.md

## What this project is

`luca` is an AI agent framework written in Python.

It has two layers:

- **`luca.agent`** — the primary product. A full-featured, durable agent: a single serializable `AgentSession` that records the whole conversation, an async agent loop that drives it, a permission model for tool approvals, and an event API for rendering.
- **`luca.client`** — a supporting package. A thin, unified LLM SDK ("small, simple LiteLLM") that gives the agent one API across providers (OpenAI, Anthropic, OpenRouter, …). Deliberately minimal and stable; exists to serve the agent. Only runtime deps: `httpx` + `pydantic`.

Most new feature work happens in `luca.agent`. Changes to `luca.client` are usually in service of an agent need.

Package boundaries are sharp and must stay that way: `luca/client` is the LLM client; `luca/agent/core` is the core of the agent (data model, main runner, main abstractions); **everything else goes in `luca/agent/contrib`** — optional packages that consume only core's public surface, exactly like application code would. Core never imports from contrib. Contrib→contrib dependencies ARE allowed (e.g. `contrib/plugins` builds on `contrib/simple_tool_registry`).

This project is a library. We always have to think first about our developer users and give them the possibility to extend and customize the behavior. That's why Middleware and other architectural decisions are key. We don't know how our library will be used so we must always keep it extensible and open while keeping a very tight Data Model.

## Non-goals (project-wide)

- Server / proxy mode.
- Guardrails / moderation pipelines.
- Automatic retries, multi-model fallback, cross-provider message transformation.
- Wrapping vendor SDKs.
- Batch APIs, re-ranking.

## Repo layout (top level)

```
luca/
├── __init__.py                    # just __version__
├── agent/                         # THE AGENT FRAMEWORK (primary) — see AGENTS.agent.md
│   ├── core/                      # the agent core: data model, runner, main abstractions
│   └── contrib/                   # everything else — optional packages built on core
└── client/                        # the supporting LLM SDK — see AGENTS.client.md

docs/                              # user-facing docs — docs/agent/ + docs/client/
│                                  #   docs/llm.txt = how to write/update these docs

tests/
├── agent/                         # tests for luca.agent (contrib tests under tests/agent/contrib/)
└── client/                        # mirrors luca/client/ layout
main.py                            # runnable agent demo — launches the contrib TUI
api_prd.md                         # client public API contract
architecture.md                    # client internal design
pyproject.toml                     # uv-managed
```

## Which file to read next

| You're working on… | Read… |
|---|---|
| `luca/agent/` or `tests/agent/` | **[AGENTS.agent.md](AGENTS.agent.md)** |
| `luca/client/` or `tests/client/` | **[AGENTS.client.md](AGENTS.client.md)** |
| Writing or updating anything under `docs/` | **[docs/llm.txt](docs/llm.txt)** |

Read the relevant layer file before making any changes to that layer.

## Running tests

```bash
uv run py.test tests/
```

`pyproject.toml` configures pytest with `filterwarnings = ["error"]` and `-W error::ResourceWarning`. Any warning fails the build — unclosed streams or connections surface as test failures. Fix them; don't suppress them.

## Running the agent demo

Use `uv run`, not bare `python`. `main.py` is a thin launcher over the Textual TUI in `luca/agent/contrib/tui` (streaming by default).

```bash
uv run python main.py                              # fresh session
uv run python main.py --faux                       # offline scripted demo — no key, no network
uv run python main.py --conversation <id>          # resume <id>.json
uv run python main.py --conversation <id> --fork   # branch into a new session
uv run python main.py --no-streaming               # block-level events instead of deltas
uv run python main.py --model <id> --reasoning <level>  # override the session's LLMConfig
```

The demo needs `OPENROUTER_API_KEY` (or whichever model you swap in) in env or `.env` — except with `--faux`. Sessions persist to `<session-id>.json` in the working directory.

## Code style (project-wide)

- Match the existing module style: focused docstrings at the top, minimal inline comments, type hints throughout.
- Pydantic v2 idioms only: `model_config = ConfigDict(...)`, `model_validate`, `model_copy(deep=True)`, discriminated unions via `Annotated[Union[...], Field(discriminator="type")]`, `str, Enum` for JSON-clean enums. `extra="forbid"` on every Pydantic model.
- No new runtime dependencies without explicit user approval — raise it first.
- This is V1, not released — edit freely, no backwards-compat shims.
- No speculative hooks or extension points before a real second case exists. Polymorphism on demand.

## Communication style (project-wide)

How to respond in design and architecture discussions:

- Be concise. Prefer a few sentences or bullets over long explanations. Cut to the points that matter.
- Take a position. Give a recommendation, not a menu of options.
- Check the premise before agreeing. If the user's framing or proposal is wrong, say so and explain why. Do not validate by default.
- Stay at the architecture level unless asked for code or mechanics.
- Answer the underlying question, not just the literal last message.
- Use plain prose. No hyperbole, no analogies, no marketing phrasing.
