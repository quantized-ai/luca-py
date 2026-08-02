# AGENTS.md

## What this project is

`luca` is an AI agent framework written in Python.

It has two layers:

- **`luca.agent`** — the primary product. A full-featured, durable agent: a single serializable `AgentSession` that records every conversation in it (the main one plus any parallel subagents), an async agent loop that drives them, a permission model for tool approvals, and an event API for rendering.
- **`luca.client`** — a supporting package. A thin, unified LLM SDK ("small, simple LiteLLM") that gives the agent one API across providers (OpenAI, Anthropic, OpenRouter, …). Deliberately minimal and stable; exists to serve the agent. Only runtime deps: `httpx` + `pydantic`.

Most new feature work happens in `luca.agent`. Changes to `luca.client` are usually in service of an agent need.

Package boundaries are sharp and must stay that way: `luca/client` is the LLM client; `luca/agent/core` is the core of the agent (data model, main runner, main abstractions); **everything else goes in `luca/agent/contrib`** — optional packages that consume only core's public surface, exactly like application code would. Core never imports from contrib. Contrib→contrib dependencies ARE allowed (e.g. `contrib/plugins` builds on `contrib/simple_tool_registry`, and `contrib/subagents` — the spawn + result tools behind the parallel-subagent capability — builds on both).

This project is a library. We always have to think first about our developer users and give them the possibility to extend and customize the behavior. That's why Middleware and other architectural decisions are key. We don't know how our library will be used so we must always keep it extensible and open while keeping a very tight Data Model.

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
pyproject.toml                     # uv-managed
```

## Which file to read next

| You're working on… | Read… |
|---|---|
| `luca/agent/` or `tests/agent/` | AGENTS.agent.md |
| `luca/client/` or `tests/client/` | AGENTS.client.md |
| Writing or updating anything under `docs/` | docs/llm.txt |

Read the relevant layer file before making any changes to that layer.

## Running tests

```bash
uv run py.test tests/
```

`pyproject.toml` configures pytest with `filterwarnings = ["error"]` and `-W error::ResourceWarning`. Any warning fails the build — unclosed streams or connections surface as test failures. Fix them; don't suppress them.

## Lint and format

After changing code, run:

```bash
uv run ruff check --fix
uv run ruff format
```

Config is in `ruff.toml`. Never use `--unsafe-fixes`.

## Test style (project-wide)

**Assert on the full object, not on individual properties.** `assert block == ThinkingBlock(text=…, signature=…, redacted=False)` rather than three separate attribute checks. A field added later shows up as a diff instead of passing unnoticed, and the expected value doubles as documentation of the shape. The same goes for a whole payload dict, a complete event list, and the resulting `AgentSession`.

Reach for a partial assertion only when the object genuinely carries noise the test does not own — a wire payload with unrelated keys, for instance — and say so in a comment.

Tests are declarative: precondition → one action → postcondition. No logic, no helpers in the test body.

## Running the agent demo

Use `uv run`, not bare `python`. `main.py` is a thin launcher over the Textual TUI in `luca/agent/contrib/tui` (streaming by default). Run `uv run python --help` for more.

The demo needs `OPENROUTER_API_KEY` (or whichever model you swap in) in env or `.env` — except with `--faux`. Sessions persist to `<session-id>.json` in the working directory.

Skills (`<name>/SKILL.md`) are read from `.claude/skills`, `.agents/skills` and the `~` equivalents, plus any `extra_skill_locations` in the config; `--no-skills` turns that off. See [docs/agent/contrib/skills/README.md](docs/agent/contrib/skills/README.md).

Configuration is read from the nearest `luca.json` at or above the cwd (bounded by the repo) over `~/.config/luca/luca.json`. `--config <path>` (or `LUCA_CONFIG_PATH`, which the flag overrides) names one file to use INSTEAD of both — see [docs/agent/contrib/tui/config.md](docs/agent/contrib/tui/config.md).

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
- Use plain prose. No hyperbole, no analogies, no marketing phrasing, don't try to sound cool.
- When explaining a problem, start from the top assuming no previous knowledge of the user. Think of your user as a fresh new employee with a lot of seniority that has no context of the codebase, so before explaining the problem you have to give context. Context > problem > why it happens > my suggestions next.
