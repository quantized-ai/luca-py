# `luca.json`

The TUI reads a JSON config from two places and deep-merges them, project over
home:

- `~/.config/luca/luca.json` (or `$XDG_CONFIG_HOME/luca/luca.json`) — your
  personal defaults across every repo.
- `./luca.json` — repo policy, committed with the project.

Precedence, highest first: **CLI flag > `./luca.json` > `~/.config/luca/luca.json`
> the persisted session > built-in default.** So the file behaves like sticky
CLI flags: it overrides a resumed session's model, and a `--model` flag still
overrides the file. Every field is optional; unknown keys are rejected, and a
malformed file exits with a one-line error.

Point your editor at [`luca.schema.json`](../../../luca.schema.json) via the
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
- The file is pure data. Nothing in it is executed, unlike some other agents'
  configs.

## CLI flags that override it

`--model`, `--provider`, `--reasoning`, `--workspace`, `--mode`,
`--streaming` / `--no-streaming`, `--autocompact` / `--no-autocompact`,
`--compact-threshold`, `--compact-keep-turns`.
