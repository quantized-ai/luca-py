# MCP

Connect the agent to external [Model Context Protocol](https://modelcontextprotocol.io) servers and
expose their tools to the model, namespaced by server, gated by the same permission strategy as every
other tool. Optional: needs the `mcp` dependency-group (`uv sync --group mcp`); without it the TUI
still runs, MCP simply contributes nothing.

Built on the official `mcp` Python SDK. v1 is **tools only** (resources and prompts are additive),
over **stdio** and **Streamable HTTP**, with **OAuth** for remote servers.

## Configure servers in `luca.json`

```jsonc
{
  "mcp": {
    "files": {                                 // a local subprocess server
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "env": { "FOO": "bar" }
    },
    "sentry": {                                // a remote server, static auth
      "type": "http",
      "url": "https://mcp.sentry.dev/mcp",
      "headers": { "Authorization": "Bearer <your-token>" }
    },
    "linear": {                                // a remote server, OAuth
      "type": "http",
      "url": "https://mcp.linear.app/mcp",
      "oauth": true
    }
  }
}
```

Each server's tools appear to the model as `label__toolname` (e.g. `files__read_file`). Set
`"enabled": false` to keep a server in the file but off.

Config is read from `~/.config/luca/luca.json` (personal) and `./luca.json` (project), deep-merged
with the project winning per key. The two `mcp` blocks combine, so servers from both files are
available; two servers under the same label merge field by field. See [`luca.json`](../tui/config.md).

## Secrets and environment

Config values are literal. There is no `${VAR}` expansion, so a header or `env` value is sent exactly
as written. Two consequences:

- A stdio server inherits only a safe allowlist from your shell (`PATH`, `HOME`, `SHELL`, `USER`,
  `LOGNAME`). Any API key it needs goes in its `env` block explicitly; a variable you only exported in
  the shell will not reach it.
- Because the token lives in the file, keep servers that carry secrets in your personal
  `~/.config/luca/luca.json`, not the committed `./luca.json`.

## How it fits

- Connections are per call, so there is no long-lived actor and no lifecycle for anyone to own.
  `session.py` opens a short-lived `ClientSession`, uses it, and closes it, all inside one task. That
  satisfies the SDK's rule that a session be entered and exited in the same task, which is why the
  async `get_tools` plus the `prepare()` callable make the old manager unnecessary.
- `McpToolRegistry` is a `ToolRegistry`. `get_tools` **waits** for the listing before answering, and
  returns the servers' tools (namespaced `label__tool`, each carrying the server's `inputSchema` as
  `ToolSpec.input_schema`). It is async precisely so a registry fronting a remote tool server can do
  that I/O — the contract's "must not block" rule binds `prepare()`, not `get_tools` — and the runner
  races the whole `build_tool_list` step against the run's cancellation token, so waiting can never
  make `cancel()` a no-op. `create_execution` waits on the same listing, because the `ToolSpec` it
  produces is data only the server has. `prepare()` does not wait and does not need to: it routes on
  the `label__tool` prefix against static server config, so a cold resume dispatches without a round
  trip. It returns a callable that opens a fresh session, calls the tool, and closes. The registry is
  one child of the TUI's `ProxyToolRegistry`, via `McpPlugin`, sharing the one `PermissionStrategy`.
- The TUI still lists at startup in a worker (`AgentApp.on_mount`), so any OAuth flow and the
  connection notice happen up front rather than inside your first turn. The listing is shared, so the
  first turn does not pay for it twice — and because `get_tools` waits, the first turn has the tools
  whether that worker finished or not. There is nothing to close on exit, because execution is per
  call.
- Listing state is tracked per server: a label is either listed or still pending. A server that fails
  to list contributes no tools, is recorded in `failures` (the startup notice flags it), and is
  **retried on the next ask** — so a machine that was offline at launch picks its servers up on a
  later turn instead of staying dead for the session. Healthy servers are never re-listed.
- Each server's listing runs under its own deadline, bounding the request — connect, handshake and
  every pagination round trip. Expiry is recorded like any other listing failure and retried the same
  way; it never aborts the run, because one unreachable MCP server must not take down the agent.
  Servers list concurrently, so a slow one costs its siblings nothing. Resolution, most specific wins:

  ```
  server.timeout_in_ms  >  LUCA_DEFAULT_MCP_TIMEOUT_MS  >  30_000
  ```

  with one exception: a server with `oauth: true` that sets no `timeout_in_ms` takes a 330s ceiling
  ahead of the env var. Its browser flow runs *inside* the listing and waits on a human, so a short
  global default set for some other reason would otherwise make it impossible to connect. The 330s
  deliberately exceeds the 300s `oauth._AUTH_TIMEOUT`, so that inner wait expires first with its own
  message. A malformed `LUCA_DEFAULT_MCP_TIMEOUT_MS` raises when the registry is built rather than
  silently falling back.

  The deadline bounds the request, not the teardown that follows it. When a stdio listing expires,
  the SDK takes up to its own 2s `PROCESS_TERMINATION_TIMEOUT` to SIGTERM, wait for and finally
  SIGKILL the subprocess, and we wait that out rather than orphan a child — so a 30s bound can cost
  ~32s in the worst case. A leaked subprocess is worse than a late return.

- A tool **call** is bounded separately, by `call_timeout_in_ms`. It is stamped onto every spec as
  `ToolSpec.timeout_in_ms`, which the runner already enforces at dispatch — expiry records
  `TIMED_OUT`, resultless, and the turn runs to completion. Same cascade, one fewer rung:

  ```
  server.call_timeout_in_ms  >  LUCA_DEFAULT_MCP_CALL_TIMEOUT_MS  >  unset
  ```

  Unset, not a number, is deliberate: a tool body's default deadline is a framework decision
  (`RuntimeConfig.tool_execution_timeout_in_ms`), and any figure picked for arbitrary third-party
  tools would silently kill someone's legitimately slow one. What the knob buys is bounding **one**
  server's calls without bounding every tool in the agent. It is separate from `timeout_in_ms`
  because a listing is a quick metadata fetch and a call may legitimately run for minutes.

  It must stay a pure function of static server config. A `ToolSpec` that varies per call mints a
  fresh `session.tool_specs` row every time and silently defeats normalization.
- Results map to `ExecutionResult` content (text → `TextContent`, image → `ImageContent`).
- The trade-off: each tool call re-spawns the stdio subprocess and re-runs `initialize`, and any
  per-session state the server holds is gone between calls — a working directory, an open document, an
  open transaction or cursor. Most tool servers are stateless per call, so this is invisible; it is not
  for a stateful one. Startup pays the listing connect (the notice worker), not each turn. The fix, if
  a stateful server ever needs it, is a redesign rather than a patch: one long-lived task per server
  owning the connection and its cancel scope, with a request queue. Warm persistent connections are
  that path, deferred until a stateful server actually shows up.

## Permissions

An MCP tool is `ToolKind.OTHER`. The framework cannot know what a server tool does, so it does not
claim `READ`/`EDIT`/`EXECUTE`. In `ask` mode every MCP tool prompts by its namespaced name; in
`yolo`/`auto` it is allowed unless a `deny` rule blocks it. Pre-authorize with a config
[`PermissionRule`](../tui/config.md), e.g. `{ "decision": "allow", "permission": "files__read_file" }`.

## OAuth

For an `http` server with `"oauth": true`, the SDK's `OAuthClientProvider` runs the flow: a browser
opens to authorize, a one-shot `localhost` server captures the redirect, and the token persists to
`~/.config/luca/mcp-auth.json`. The flow runs on the first connection (the first tool listing) and
later calls reuse the stored token. Static-header auth needs none of this.

## Not in v1

Resources and prompts, per-agent tool toggles, warm persistent connections (per call for now), MCP
Apps / Tasks / sampling, and reconnect-on-drop (a downed server contributes no tools; relaunch to
retry).

Next: back to the [contrib index](../README.md).
