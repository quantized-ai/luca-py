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
- `McpToolRegistry` is a `ToolRegistry`. `get_tools` lists each server once and caches the specs
  (namespaced `label__tool`), each carrying the server's `inputSchema` as the `ToolSpec.input_schema`.
  Its `prepare()` returns a callable that opens a fresh session, calls the tool, and closes. The
  registry is one child of the TUI's `ProxyToolRegistry`, via `McpPlugin`, sharing the one
  `PermissionStrategy`.
- The TUI lists the servers once at startup in a background worker (`AgentApp.on_mount`): it warms the
  cache, runs any OAuth flow up front, and posts a notice of the connected servers and tool count.
  There is still nothing to close on exit, because execution is per call.
- A server that fails to list contributes no tools and is recorded, and the startup notice flags it.
- Results map to `ExecutionResult` content (text → `TextContent`, image → `ImageContent`).
- The trade-off: each tool call re-spawns the stdio subprocess and re-runs `initialize`, and a server
  that keeps per-session state is reset between calls. Most tool servers are stateless per call.
  Startup pays the listing connect (the notice worker), not each turn. Warm persistent connections are
  a future option if that latency bites.

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
