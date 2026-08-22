# `luca.agent.contrib.app` — the headless application layer

Everything a front end needs to put a configured agent on the road, with no
front end in it. The TUI is one consumer; the [ACP server](../acp/README.md) is
another.

```python
from luca.agent.contrib.app import AgentApplication, boot, build_session, credentials

env = boot(workspace=".")
session = build_session(config=env.config, session_dir=env.session_dir)
auth = credentials(env.config, session.session_config.llm_config)

application = AgentApplication(session, auth=auth, workspace=".", session_dir=env.session_dir)
application.runner.post_message("Summarize the repo.")
```

`application.runner`, `.strategy` and `.questions` are the three references a
driver works from. What it does with them — rendering, asking, answering — is
its own business; see [04-runner.md](../../04-runner.md).

> **Nothing in this package may import Textual**, directly or transitively.
> That is the whole point of it, and
> `tests/agent/contrib/app/test_no_textual.py` enforces it in a subprocess with
> the import blocked.

## 1. What is in it

| Module | Topic |
|---|---|
| `config` | the `luca.json` loader and everything derived from it |
| `auth` | `auth.json`, and the key for one provider |
| `boot` | the launch sequence: config, session directory, credentials, the session log, `build_session` |
| `application` | `AgentApplication` — the composition, persistence and credential re-pointing |
| `wiring` | `build_runner`: the plugin composition, plus `build_faux_provider` |
| `sessions` | the `~/.luca/projects/<encoded-path>/` store, the sidecar, and picker rows |
| `approvals` | a gated `ToolExecution` as display-ready prompts and options |
| `custom_commands` | user-defined `.md` slash commands |
| `prompt_files` | `@`-mention handling and `ReadLimits` |
| `usage` | token totals and the dollar estimate |

## 2. `AgentApplication`

Owns the runner, the permission strategy, the `ask_user` plugin, the checkpoint
service, and both halves of persistence.

```python
application.use(session)          # drive a different session; rebuilds the runner
application.save()                # the conversation AND the sidecar, together
application.repoint_api_key(p)    # after a model switch moves providers
application.move_memory_stores(old, new)   # after a compaction installs a successor
```

The checkpoint service outlives any single session — `/clear` and `/resume`
swap the runner, not the workspace — so it is built once and `use()` carries it
over.

Two things it deliberately does NOT hold. The DRIVE LOOP: advancing the agent
means rendering, asking and collecting, which look nothing alike between a
terminal and a socket. And CUSTOM COMMANDS: discovery is one call, but the
`reserved` set that stops a user shadowing a built-in is the front end's own
vocabulary, as is what a command becomes afterwards.

## 3. The sidecar

Beside each `<session-id>.json` sits an optional `<session-id>.app.json`, for
state that belongs to the interface rather than to the conversation. Today it
holds exactly one thing: the `ask_user` question store, so a question parked
when the process died comes back parked.

It is a separate file rather than a key under `AgentSession.extras` because
`extras` is the SESSION's state, read by anything that loads it. Losing the
sidecar costs a parked question, which `ask_user` re-seeds and asks again.

## 4. Boot, in order

`boot()` deliberately does not resolve credentials, because the provider to
validate comes off the session's `LLMConfig` and the session cannot exist until
`boot()` has run. Three calls, one order:

```python
env = boot(workspace=..., config_path=...)
session = build_session(config=env.config, session_dir=env.session_dir, ...)
auth = credentials(env.config, session.session_config.llm_config, faux=False)
```

Every failure a user can fix by editing a file raises `LucaConfigError`, so a
caller prints one readable line instead of a traceback.

`setup_logging` points the `luca` logger at ONE rotating file and turns
`propagate` off. Not a preference: the TUI draws on stderr and the ACP server
speaks JSON-RPC on stdout, so a stream handler corrupts one or the other.

Next: [`acp`](../acp/README.md) — the front end that exists because of this
package.
