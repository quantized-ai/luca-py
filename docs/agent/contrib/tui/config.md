# `luca.json`

The TUI reads a JSON config from two places and deep-merges them, project over
home:

- `~/.config/luca/luca.json` (or `$XDG_CONFIG_HOME/luca/luca.json`) — your
  personal defaults across every repo.
- `luca.json` — repo policy, committed with the project. The nearest one at or
  above the current directory wins, so it applies from any subdirectory. The
  search stops at the directory holding `.git`, so a `luca.json` outside the
  repo never leaks in.

Precedence, highest first: **CLI flag > `./luca.json` > `~/.config/luca/luca.json`
> the persisted session > built-in default.** So the file behaves like sticky
CLI flags: it overrides a resumed session's model, and a `--model` flag still
overrides the file. Every field is optional; unknown keys are rejected, and a
malformed file exits with a one-line error.

API keys are deliberately NOT here — they live in
[`auth.json`](#credentials), one user-global file that never touches a repo.

## Naming a file directly

`--config <path>` uses that file and **replaces both locations above** — neither
`./luca.json` nor `~/.config/luca/luca.json` is read. It is not an extra layer:
"use this config" means this one, not this one on top of whatever the repo
carries.

```bash
uv run python main.py --config ./configs/ci.json
LUCA_CONFIG_PATH=~/luca-profiles/review.json uv run python main.py
```

The `LUCA_CONFIG_PATH` environment variable is the same channel, and `--config`
overrides it. `~` is expanded in both. CLI flags still win over whatever the
named file says.

Unlike the two discovered locations — which are simply empty when absent — a
path you name that does not resolve is an error and exits `1`:

```
luca: /configs/ci.json: not a readable config file
```

That asymmetry is deliberate. Naming a file is a statement that it exists, and
silently falling back to an empty config would run the agent with the settings
you thought you had overridden.

`--pretty-print` ignores config entirely (a transcript does not depend on it).

Point your editor at [`luca.schema.json`](../../../../luca.schema.json) via the
`$schema` key for autocomplete.

## Every field

```jsonc
{
  "$schema": "./luca.schema.json",

  "model": {                     // defaults for the session's LLMConfig
    "provider": "anthropic",
    "model": "claude-sonnet-5",
    "reasoning": "high"          // provider-default|none|minimal|low|medium|high|xhigh
  },

  "theme": {
    "name": "luca-dark"          // a registered theme (luca-dark ships)
  },

  "runtime": {                   // any RuntimeConfig knob (ms; -1 = disabled)
    "hard_max_steps": 40,
    "soft_max_steps": 30,
    "doom_loop_threshold": 5,
    "tool_execution_timeout_in_ms": 120000,
    "client_completion_timeout_in_ms": -1
  },

  "compaction": {
    "enabled": true,
    "threshold": 0.8,            // auto-compact at this context fraction
    "keep_turns": 2,             // 0 = summarize everything; N = keep last N exchanges
    "default_window": 200000     // fallback context window when the model is uncatalogued
  },

  "permissions": {
    "mode": "ask",               // ask | yolo | auto
    "match_mode": "relaxed",     // relaxed | strict
    "rules": [                    // allow/deny, last match wins
      { "decision": "allow", "tool_kind": "read" },
      { "decision": "deny",  "permission": "bash", "resource": "/etc/*" }
    ]
  },

  "providers": {                 // where hosts are, and how models on them are called
    "mycompany": {               // base_url + transport = a host luca does not know
      "base_url": "https://llm.mycompany.com/v1",
      "transport": "luca.client.transports.OpenAITransport",
      "options": { "temperature": 0 }
    },
    "openrouter": {              // a provider that already exists: settings only
      "options": { "max_tokens": 8000 },        // → LLMConfig.model_options
      "provider_options": {                     // → LLMConfig.provider_options
        "mycustom_param": 1
      },
      "models": {
        "moonshotai/kimi-k2:free": {            // wins per key over the two above
          "options": {
            "max_tokens": 6000,
            "temperature": 0.2,
            "reasoning": "high"
          },
          "provider_options": {                 // raw, straight to the provider
            "provider": { "order": ["baseten", "together"] },
            "transforms": ["middle-out"]
          }
        }
      }
    }
  },

  "models": {                    // override the /model picker list
    "anthropic": ["claude-sonnet-5", "claude-opus-4-8"]
  },

  "workspace": ".",              // shell root
  "additional_directories": [],  // extra roots the shell tools may touch
  "extra_skill_locations": [     // more places to find <name>/SKILL.md
    ".opencode/skills/",
    "~/.config/opencode/skills/"
  ],
  "extra_command_locations": [   // more places to find <name>.md slash commands
    "~/.config/opencode/commands/"
  ],
  "instructions": [              // extra instruction files, on top of AGENTS.md
    "docs/conventions.md"
  ],
  "sessions": {
    "directory": "~/.luca/projects"  // the session store ROOT
  },
  "logging": {
    "level": "INFO",               // DEBUG|INFO|WARNING|ERROR, or OFF for no log
    "file": "~/luca.log"           // default: <session dir>/logs/<session-id>.log
  },
  "streaming": true,
  "checkpoints": true,           // snapshot the workspace before each turn (/undo, /rewind)
  "use_native_tools": true,      // provider-native tools where the model has them
  "websearch": {                 // provider-hosted web search — ABSENT = OFF, {} = on
    "enabled": true,             // flip false to disable TEMPORARILY, block kept in place
    "openai":    { "options": { "search_context_size": "high" } },
    "anthropic": { "search": { "max_uses": 5 },
                   "fetch":  { "max_content_tokens": 20000 } }   // fetch is opt-in
  }
}
```

The `websearch` block maps 1:1 onto
[`WebSearchPlugin`](../websearch/README.md)'s constructor — the per-provider
options ARE the client's own tool declarations. The TUI merges its own
defaults UNDER the user's block, per key (`include_results` +
`include_sources` on OpenAI; `allowed_callers: ["direct"]` on Anthropic's
tools — the fetch half only when the block enables fetch). Nothing about it
is persisted: it is runtime wiring, re-applied every launch.

## Notes

- A `runtime` block sets those fields over the session's persisted runtime; the
  rest of the session's runtime is untouched.
- A `providers` entry says where a host is and how models on it are called.
  Nothing is registered globally: `base_url` and `transport` are passed on
  every call, so pointing a built-in like `openai` at a proxy is as ordinary as
  configuring a host luca has never heard of. A provider the client does not
  know needs BOTH — they are what let it build a generic provider for that name
  ([providers](../../../client/09-providers-and-transports.md)) — and naming an
  unreachable provider fails at startup rather than on the first turn.
  `transport` is a dotted path to the transport CLASS, e.g.
  `"luca.client.transports.OpenAITransport"`.
- `options` and `provider_options` are both per provider and per model, and the
  model's own block wins per key over the provider-wide one. See
  [Model options](#model-options).
- There is no api key in this file. Credentials live in
  [`auth.json`](#credentials), which is a separate file for a reason.
- `provider` (singular) is accepted as an alias for `providers`, since that
  reads naturally in a file that configures one. The two are the same key: a
  file carrying both is an error. `providers` is canonical and the only
  spelling in the JSON schema.
- Permission `rules` are re-applied every launch (approval is runtime state,
  never persisted). A rule with `tool_kind` matches every call of that kind; a
  rule with `permission` (+ optional `resource` glob) matches a
  `(permission, resource)` pair. `resource` is an fnmatch glob (`"*"`,
  `"/etc/*"`).
- `extra_skill_locations` adds roots to scan for `<name>/SKILL.md`, on top of
  `.claude/skills`, `.agents/skills` and the `~` equivalents. `~` is expanded per
  entry; see [`skills/`](../skills/README.md).
- `extra_command_locations` adds roots to scan for `<name>.md` slash commands,
  on top of `.claude/commands`, `.agents/commands` and the `~` equivalents. `~`
  is expanded per entry, the first location wins a name collision, and a
  built-in command is never shadowed; see [the TUI page](README.md#31-your-own-commands).
- `instructions` adds files to the discovered `LUCA.md` / `AGENTS.md` /
  `CLAUDE.md`, read last so they win. `~` is expanded and a relative entry
  resolves against the workspace. An entry that does not resolve to a readable
  file is an error, not a silent skip; see [`prompts/`](../prompts/README.md).
- `sessions.directory` is the store ROOT, not the final directory: the encoded
  project path is always appended under it, so two projects never share a
  session list. Defaults to `~/.luca/projects`; see
  [`tui/README.md`](README.md).
- `models` ADDS to what `/model` offers rather than replacing it. The picker's
  own list comes from the model catalog, so this key is for hosts models.dev
  does not know: a custom provider, or a local `ollama`. See
  [`10-catalog.md`](../../../client/10-catalog.md). Anything under
  `providers.<name>.models` is added too, so configuring a model's options is
  enough to make it pickable and you never list it twice.
- `checkpoints` (default true) snapshots the workspace before each turn into a
  private git repository at `<session dir>/checkpoints.git`, which is what
  `/undo` and `/rewind` restore. Nothing is written into the workspace and the
  user's own `.git` is excluded; `.gitignore`d paths are not captured, so edits
  to them are not undoable. No git binary switches the feature off on its own.
- `use_native_tools` (default true) offers the provider's own tools where the
  ACTIVE model supports them — `apply_patch` + `shell` on OpenAI,
  `text_editor` + `bash` on Anthropic — and the generic shell tools they
  replace drop out of that request. It is an adaptation input, not a session
  property: the same session is valid either way, the set is re-derived before
  every call, and a model with no natives (or reached through OpenRouter) is
  unaffected. See [`shell/`](../shell/README.md#6-provider-native-tools).
- The file is pure data. Nothing in it is executed, unlike some other agents'
  configs.

## Model options

`model` picks WHICH model runs. Two blocks say HOW it is invoked, and they
mirror the two dicts on the session's `LLMConfig`
([data model](../../02-data-model.md)) exactly:

| Block | Becomes | Holds |
|---|---|---|
| `options` | `LLMConfig.model_options` | arguments `luca.client.acompletion` takes — `max_tokens`, `temperature`, `top_p`, `reasoning`, `seed`, `stop`, … |
| `provider_options` | `LLMConfig.provider_options` | `base_url`, `transport`, and any raw wire field the provider itself documents |

**Which block a key is written in IS the routing.** Nothing inspects a key
name, nothing is renamed, and nothing guesses: `"provider": {"order": [...]}`
belongs in `provider_options` because it is OpenRouter's wire field, and
`"max_tokens"` belongs in `options` because it is the client's argument.

Inside `options`, four keys are typed and validated — `max_tokens`,
`temperature`, `top_p`, `reasoning` — because `max_tokens: 0` and
`reasoning: "huge"` are worth catching at startup. Every other key passes
through, since the client takes a dozen more and refusing them would mean a
luca release for every client one. The cost is that a key that is NOT an
`acompletion` argument (a typo like `"max_token"`) is a `TypeError` on the
first turn rather than a startup error.

Resolution for the model you are running, lowest precedence first:

```
built-in default
  < ~/.config/luca/luca.json
  < ./luca.json
  < model.reasoning
  < providers.<provider>.options / .provider_options
  < providers.<provider>.models.<model>.options / .provider_options
  < CLI flag (--reasoning)
```

Both levels merge per key rather than replacing wholesale, and nested objects
merge deeply (scalars and lists replace). So a provider-wide `provider.order`
survives a model that only sets `transforms`. Switching model with `/model`
re-resolves from scratch: a model with no block of its own runs with none of
the previous model's settings.

Nothing is renamed on the way out. Write the field the provider documents, not
an approximation of it: OpenRouter's is `"transforms": ["middle-out"]`, plural
and an array. A near miss like `"transform": "middle-out"` is sent exactly as
you wrote it and the provider ignores it, which is the cost of a passthrough
that does not pretend to know the field.

One thing worth knowing before you reach for `provider_options`: its raw keys
are merged into the request LAST and beat everything luca derived, including
`messages`, `tools`, `model` and `stream`. A stray `"tools"` key in your config
will silently turn off tool calling. That is the deal with an escape hatch — it
is not validated because validating it would close it.

## Credentials

API keys are not in `luca.json`. They live in one user-global file:

```
$XDG_DATA_HOME/luca/auth.json      # default: ~/.local/share/luca/auth.json
```

```jsonc
{
  "openrouter":         { "type": "api", "key": "sk-or-..." },
  "my_custom_provider": { "type": "api", "key": "sk-..." }
}
```

Any provider name works, including one the client has never heard of — pair it
with a `providers` entry giving `base_url` and `transport` and the host is
reachable. `type` is `"api"` today; `"oauth"` is coming.

A separate file because a config is the kind of thing you commit to a repo or
paste into an issue and a key is not, and because nothing in the file ever
reaches the session: it is read once at startup and handed to the runner as a
runtime argument, so no key is written to `~/.luca/projects/…`.

A provider with **no entry** is not an error. No key is passed for it and the
client falls back to whatever environment variable it knows for that provider
(`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, …) — which is how luca worked
before this file existed, and still the shortest path to a running agent. The
TUI never knows which variable that is; that is the client's business, and a
missing one surfaces as the provider's own authentication error.

`LUCA_AUTH_PATH` names a different file, for a sandbox or a test.

## Logging

Each session writes a rotating log next to the session file it belongs to:

```
~/.luca/projects/<encoded-project-path>/
├── a1b2c3d4.json          # the session
└── logs/
    └── a1b2c3d4.log       # what happened while it ran
```

Default level `INFO`. Errors carry the traceback, which the session file cannot
keep — so this is where you look when a tool blew up and the transcript only
showed you `KeyError: 'path'`. Precedence is `--log-level` > `LUCA_LOG_LEVEL` >
`logging.level` > `INFO`; `OFF` writes no file, and `--log-file` (or
`logging.file`) moves it.

Nothing is ever written to stderr — the TUI is drawing there. The `luca` logger
is given the file handler and `propagate` is turned off, so luca's records also
stay out of any root handler your own program installed. See
[`14-logging.md`](../../14-logging.md) for the records themselves.

## CLI flags that override it

`--model`, `--provider`, `--reasoning`, `--theme`, `--workspace`, `--mode`,
`--log-level`, `--log-file`,
`--streaming` / `--no-streaming`, `--autocompact` / `--no-autocompact`,
`--compact-threshold`, `--compact-keep-turns`,
`--use-native` / `--no-use-native`,
`--checkpoints` / `--no-checkpoints`,
`--websearch` / `--no-websearch` (a bare `--websearch` with no block is the
minimal enable, `≡ "websearch": {}`; forcing it on also overrides a block's
temporary `"enabled": false`).

`--no-skills`, `--no-commands` and `--no-instructions` withhold skills,
user-defined slash commands and the project's instruction files entirely,
whatever the config says.
