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

  "providers": {                 // register OpenAI-compatible (or other) hosts
    "mycompany": {
      "base_url": "https://llm.mycompany.com/v1",
      "api_key_env": "MYCOMPANY_API_KEY",
      "transport": "openai"      // openai | anthropic | openrouter | bedrock
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
- A `providers` entry uses the client's existing host registry
  ([providers](../../../client/09-providers-and-transports.md)); set
  `model.provider` to the key to route through it.
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
  [`10-catalog.md`](../../../client/10-catalog.md).
- `use_native_tools` (default true) offers the provider's own tools where the
  ACTIVE model supports them — `apply_patch` + `shell` on OpenAI,
  `text_editor` + `bash` on Anthropic — and the generic shell tools they
  replace drop out of that request. It is an adaptation input, not a session
  property: the same session is valid either way, the set is re-derived before
  every call, and a model with no natives (or reached through OpenRouter) is
  unaffected. See [`shell/`](../shell/README.md#6-provider-native-tools).
- The file is pure data. Nothing in it is executed, unlike some other agents'
  configs.

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
