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

  "providers": {                 // register hosts, configure how models are called
    "mycompany": {               // base_url = a host registration
      "base_url": "https://llm.mycompany.com/v1",
      "api_key_env": "MYCOMPANY_API_KEY",
      "transport": "openai",     // openai | anthropic | openrouter | bedrock
      "options": { "temperature": 0 }
    },
    "openrouter": {              // no base_url = settings for a provider that exists
      "options": { "max_tokens": 8000 },        // every model on this provider
      "models": {
        "moonshotai/kimi-k2:free": {
          "options": {
            "max_tokens": 6000,                 // known keys: typed + validated
            "temperature": 0.2,
            "reasoning": "high",
            "provider": {                       // unknown keys: raw, to the provider
              "order": ["baseten", "together"],
              "allow_fallbacks": true
            },
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
  "use_native_tools": true       // provider-native tools where the model has them
}
```

## Notes

- A `runtime` block sets those fields over the session's persisted runtime; the
  rest of the session's runtime is untouched.
- A `providers` entry does one or both of two jobs. With `base_url` it
  REGISTERS a host in the client's provider registry
  ([providers](../../../client/09-providers-and-transports.md)); set
  `model.provider` to the key to route through it. Without one it only carries
  settings for a provider that already exists, which is how you configure a
  built-in like `openrouter` — registering over a built-in is still refused, so
  a custom host needs a distinct name. A settings-only entry naming a provider
  luca cannot reach is an error rather than a silent no-op.
- `options` is per provider and per model, and the model's own block wins per
  key over the provider-wide one. See [Model options](#model-options).
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

`model` picks WHICH model runs. An `options` block says HOW it is invoked.

Four keys are known and validated: `max_tokens`, `temperature`, `top_p` and
`reasoning`. Every other key is passed to the provider verbatim as its own wire
field. That split is why this is the one block in the file that accepts unknown
keys: a provider's options move faster than a schema does, and OpenRouter's
`provider.order` has no portable equivalent to be typed as.

Resolution for the model you are running, lowest precedence first:

```
built-in default
  < ~/.config/luca/luca.json
  < ./luca.json
  < providers.<provider>.options
  < providers.<provider>.models.<model>.options
  < CLI flag (--reasoning)
```

The two `options` levels merge per key rather than replacing wholesale, and raw
passthrough keys merge deeply (nested objects merge, scalars and lists replace).
So a provider-wide `provider.order` survives a model that only sets
`transforms`. Switching model with `/model` re-resolves from scratch: a model
with no block of its own runs with none of the previous model's settings.

Nothing is renamed on the way out. Write the field the provider documents, not
an approximation of it: OpenRouter's is `"transforms": ["middle-out"]`, plural
and an array. A near miss like `"transform": "middle-out"` is sent exactly as
you wrote it and the provider ignores it, which is the cost of a passthrough
that does not pretend to know the field.

Two things worth knowing before you reach for the raw keys.

They are merged into the request LAST and beat everything luca derived,
including `messages`, `tools`, `model` and `stream`. A stray `"tools"` key in
your config will silently turn off tool calling. That is the deal with an
escape hatch: it is not validated because validating it would close it.

A middleware that routes a turn to a different model (`build_model_string`,
see [middleware](../../07-middleware.md)) carries the configured model's
settings. `max_tokens` and friends still apply; the raw block does not, because
it is keyed by provider name and the transport finds nothing under its own. It
is dropped rather than sent to the wrong provider.

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
`--use-native` / `--no-use-native`.

`--no-skills` and `--no-instructions` withhold skills and the project's
instruction files entirely, whatever the config says.
