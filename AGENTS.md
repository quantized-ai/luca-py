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

Model facts — context windows, pricing, capabilities — come from [models.dev](https://models.dev), vendored in `luca/client/catalog/_data/models.json` and refreshable with `--refresh-models`. Metadata, never a gate: an unlisted model still runs. See [docs/client/10-catalog.md](docs/client/10-catalog.md).

The demo needs a key for whichever provider it talks to — except with `--faux`. Either name it in `~/.local/share/luca/auth.json` (`{"openrouter": {"type": "api", "key": "sk-or-…"}}`), which the TUI reads at boot and hands to the runner, or leave that file out and let `luca.client` read the provider's own env var (`OPENROUTER_API_KEY`, …) from the environment. NOTHING reads a `.env` — not the library, not the TUI; export the variables yourself. Never in `luca.json`, and never on the session: `LLMConfig` is persisted and copied onto every assistant message. Sessions persist to `~/.luca/projects/<encoded-project-path>/<session-id>.json`, one directory per project; `--resume` (or `/resume`) picks one back up, and `sessions.directory` in the config moves the root.

Skills (`<name>/SKILL.md`) are read from `.claude/skills`, `.agents/skills` and the `~` equivalents, plus any `extra_skill_locations` in the config; `--no-skills` turns that off. See [docs/agent/contrib/skills/README.md](docs/agent/contrib/skills/README.md).

User-defined SLASH COMMANDS are read the same way, from `.claude/commands` / `.agents/commands` (`extra_command_locations`, `--no-commands`). One `<name>.md` per command: its body is sent as an ordinary user message with `$ARGUMENTS` and `$1`…`$9` filled in, so a command is a saved prompt, not a tool. A built-in name is never shadowed. See [docs/agent/contrib/tui/README.md](docs/agent/contrib/tui/README.md#31-your-own-commands).

Provider-native tools are ON by default: a model that supports them is offered its provider's own `apply_patch`/`shell` or `text_editor`/`bash` instead of the generic shell tools those replace. `--no-use-native` (or `use_native_tools` in the config) keeps every model on the generic set. Support is per MODEL, and a session stays valid either way — the tool set is re-derived before every call. Anthropic's native `bash` is a PERSISTENT session, as its wire contract promises: one live `/bin/bash` per conversation, so `cd` and `export` carry between calls, and a restart (timeout, cancel, resume) tells the model its shell is empty rather than rebuilding it. See [docs/agent/contrib/shell/README.md](docs/agent/contrib/shell/README.md#6-provider-native-tools).

The agent can ASK THE USER. `ask_user` (`luca/agent/contrib/questions`) is the first DEFERRED tool: the model asks up to four questions, the tool returns `ExecutionDeferred()`, the runner parks the open turn at `AWAITING_RESULT` and the drive returns; the TUI renders the set in its own dock, and the answers resolve the same call on the next drive. Outstanding questions live in `<session dir>/<session-id>.tui.json` — the TUI's own sidecar, so a parked question comes back on resume. See [docs/agent/contrib/questions/README.md](docs/agent/contrib/questions/README.md).

The system prompt is assembled per model: a base coding-agent prompt plus an addendum for the model's family, an environment block, and the project's instruction files (`LUCA.md`, then `AGENTS.md`, then `CLAUDE.md` — one per directory from the git root down to the workspace). `--no-instructions` turns the last part off; `instructions` in the config names extra files. See [docs/agent/contrib/prompts/README.md](docs/agent/contrib/prompts/README.md).

CHECKPOINTS are on by default: the workspace is snapshotted before each turn into a private git repo beside the session (`<session dir>/checkpoints.git`), never into the workspace, and `/undo` (the last turn) and `/rewind` (a picker) put BOTH halves back — the files with git, the conversation with the core primitive `AgentSessionRunner.rewind_to`, which archives the conversation and installs a successor over the truncated prefix. Nothing is deleted, `.gitignore`d paths are not captured and therefore not restored, and no git binary means the feature switches itself off. `--no-checkpoints` (or `checkpoints` in the config) turns it off. See [docs/agent/15-rewind.md](docs/agent/15-rewind.md) and [docs/agent/contrib/checkpoints/README.md](docs/agent/contrib/checkpoints/README.md).

EXTERNAL MCP SERVERS are configured under `mcp.servers` in `luca.json` and their tools offered to the model as `mcp__<server>__<tool>`, gated by the same `PermissionStrategy` as everything else (`--no-mcp` turns it off). The client is FIRST-PARTY, written on `httpx` and `asyncio` like every `luca.client` transport; the only dependency is `mcp-types` (pydantic only). It speaks the stateless 2026-07-28 revision and falls back to the `initialize` handshake for older servers, deciding per server with a `server/discover` probe. One connection per server lasts the whole process: a stdio server is one subprocess with requests multiplexed by JSON-RPC id, an HTTP server is one POST per request over a shared client. THE SERVICE IS APP-SCOPED, built beside `CheckpointService` in `AgentApp.__init__`, because `/clear`, `/new`, `/resume` and fork rebuild the runner and a plugin holding connections would re-discover every server and re-open OAuth mid-session; the plugin and registry are stateless views and the plugin deliberately has no `aclose`. The tool list is a DURABLE cache under `~/.local/share/luca/mcp/`, refreshed out of band on the server's own `ttlMs`, because the registry contract forbids listing inside `get_tools` and an in-memory cache would leave the model toolless on the first turn of every run. The first turn of a genuinely cold run waits on that listing, bounded, because the alternative is a first message the model answers with none of the tools it was told about. `${VAR}` in `env` and `headers` values is expanded from the exported environment. OAuth tokens are keyed by ISSUER with a `resources` index beside them mapping each server's canonical URI to its issuer, so a cold process finds the token it already has instead of demanding a fresh browser login. A server that has never been authorized is NEEDS_AUTH, not a failure, and is never listed (so it cannot answer 401 and be reported red); a cached listing whose refresh failed is STALE, because its tools are still being offered and calling that either "connected" or "inactive" would be a lie about one half of it. `/mcp` (or `/mcp <server>`) opens a server list with a state per row and the actions on it — open its tools, authenticate, reconnect, disable for the session — and the browser only opens from there, never inside a turn; the screen stays up while an action runs and `esc` cancels it. See [docs/agent/contrib/mcp/README.md](docs/agent/contrib/mcp/README.md).

Configuration is read from the nearest `luca.json` at or above the cwd (bounded by the repo) over `~/.config/luca/luca.json`. `--config <path>` (or `LUCA_CONFIG_PATH`, which the flag overrides) names one file to use INSTEAD of both — see [docs/agent/contrib/tui/config.md](docs/agent/contrib/tui/config.md).

The library EMITS log records and configures nothing — module loggers under `luca`, no handlers, no levels. Failure sites log at ERROR with the traceback, because the runner converts exceptions into durable state and only `str(exc)` reaches the session. Records carry `conv=<id>` in the message text; there is no `extra` dict and no adapter. The demo writes each session's log to `<session dir>/logs/<session-id>.log` at INFO (`--log-level`, `LUCA_LOG_LEVEL`, `logging.level`; `OFF` disables) and never to stderr, which the TUI is drawing on. See [docs/agent/14-logging.md](docs/agent/14-logging.md).

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
