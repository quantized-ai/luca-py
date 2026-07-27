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

- Each server is an **actor** (`connection.py`): its whole `ClientSession` lifecycle lives in one
  task, so the SDK's async context managers are entered and exited in the same task. Callers dispatch
  a `call_tool` over a queue and await a future. This sidesteps the "cancel scope in a different task"
  failure that bites naive integrations.
- `McpManager` owns the connections. A server that fails to start contributes no tools and is
  reported, never crashing the app.
- `McpToolRegistry` is a `ToolRegistry` whose `get_tools` reads the live tool list (dynamic, so a
  server that comes up mid-session is picked up on the next turn). It is one child of the TUI's
  `ProxyToolRegistry`, via `McpPlugin`, sharing the one `PermissionStrategy`.
- Results map to `ExecutionResult` content (text → `TextContent`, image → `ImageContent`).
- The TUI owns the lifecycle: connections open in `AgentApp.on_mount` (a worker) and close in
  `on_unmount` (every exit path).

## Permissions

An MCP tool is `ToolKind.OTHER`. The framework cannot know what a server tool does, so it does not
claim `READ`/`EDIT`/`EXECUTE`. In `ask` mode every MCP tool prompts by its namespaced name; in
`yolo`/`auto` it is allowed unless a `deny` rule blocks it. Pre-authorize with a config
[`PermissionRule`](../tui/config.md), e.g. `{ "decision": "allow", "permission": "files__read_file" }`.

## OAuth

For an `http` server with `"oauth": true`, the SDK's `OAuthClientProvider` runs the flow: a browser
opens to authorize, a one-shot `localhost` server captures the redirect, and the token persists to
`~/.config/luca/mcp-auth.json`, so the browser flow runs only once. Static-header auth needs none of
this.

## Not in v1

Resources and prompts, per-agent tool toggles, MCP Apps / Tasks / sampling, and reconnect-on-drop
(a downed server is marked out; relaunch to retry).
