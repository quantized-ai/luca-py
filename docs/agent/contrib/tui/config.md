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
    "name": "nord"               // Textual theme name
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
  "streaming": true
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
- The file is pure data. Nothing in it is executed, unlike some other agents'
  configs.

## CLI flags that override it

`--model`, `--provider`, `--reasoning`, `--theme`, `--workspace`, `--mode`,
`--streaming` / `--no-streaming`, `--autocompact` / `--no-autocompact`,
`--compact-threshold`, `--compact-keep-turns`.
