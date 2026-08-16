# MCP

Connect the agent to external [Model Context Protocol](https://modelcontextprotocol.io) servers and offer their tools to the model, namespaced per server and gated by the same permission strategy as every other tool.

```jsonc
// luca.json
{
  "mcp": {
    "servers": {
      "files": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."] },
      "github": { "url": "https://api.githubcopilot.com/mcp/", "headers": { "Authorization": "Bearer ${GITHUB_TOKEN}" } },
      "linear": { "url": "https://mcp.linear.app/mcp", "oauth": true }
    }
  }
}
```

The model then sees `mcp__files__read_file`, `mcp__github__create_issue`, and so on. `/mcp` says what is connected.

## 1. Using it from your own agent

The TUI reads the config above. A library user builds the same three objects: the servers, one `McpService` that owns the connections, and an `McpPlugin` that hands the runner a registry over it.

```python
import asyncio

from luca.agent.contrib.mcp import McpSettings
from luca.agent.contrib.mcp.plugin import McpPlugin
from luca.agent.contrib.mcp.service import McpService
from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.simple_tool_registry import YoloPermissionPolicy
from luca.agent.core import AgentSessionRunner, LLMConfig


async def main() -> None:
    servers = McpSettings.model_validate(
        {"servers": {"files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]}}}
    ).build()

    service = McpService(servers, catalog_path=None)
    await service.start()
    print([spec.name for spec in service.specs()])

    session = AgentSessionRunner.new_session(LLMConfig(model="openai/gpt-4o-mini", provider="openrouter"))
    runner = PluginAgentSessionRunner(session, plugins=[McpPlugin(service, YoloPermissionPolicy())])
    await runner.run("Read note.txt and summarize it.")

    await service.aclose()


asyncio.run(main())
```

`start()` connects every server and lists its tools; it is single-flight, so a
second caller awaits the same work. `specs()` is a local read and never touches
the network, which is what lets the registry answer `get_tools` synchronously.

Three arguments carry the decisions:

| Argument | Effect |
| --- | --- |
| `catalog_path` | Where the tool list is cached between runs. `None` keeps it in memory, so every process starts cold |
| `token_path` | Where OAuth tokens go. `None` keeps them in memory, so every process logs in again |
| `permission_policy` | The gate. Pass the same `PermissionStrategy` your other registries use so one approval model covers everything; `YoloPermissionPolicy()` allows everything |

> ⚠️ **Build the service once and keep it.** It owns live subprocesses and
> negotiated protocol state. Rebuilding it per session reconnects every server
> and re-runs OAuth — see section 7.

## 2. Which protocol it speaks

Both eras, decided per server on the first connection.

The `2026-07-28` revision removed the `initialize` handshake, protocol-level sessions and the `Mcp-Session-Id` header. A server that needs state across calls now has to mint an explicit handle and pass it back as an ordinary tool argument. Most servers deployed today have not migrated, so the client probes with `server/discover` and reads the answer three ways, exactly as the spec prescribes:

| The server answers | It is | luca does |
| --- | --- | --- |
| a `DiscoverResult` | modern | picks a mutually supported version and continues |
| `-32020`, `-32021` or `-32022` | modern, wrong version | retries in a version it advertises |
| anything else, or nothing | pre-2026 | falls back to the `initialize` handshake |

That last row is why the fallback is not keyed to one error code: a legacy server answers an unknown pre-`initialize` request with whatever its implementation happens to return, commonly `-32601`.

The client is written here rather than taken from the official SDK, for the same reason [luca/client/](../../../../luca/client/) writes its own provider transports. The only new dependency is `mcp-types` (pydantic and typing-extensions), for the generated wire models and the per-version surface map. Install it with the `mcp` dependency group, which `uv sync` picks up by default.

## 3. One connection per server, for the life of the process

A stdio server is a subprocess spawned on first use and kept. Requests are multiplexed over its one channel and correlated by JSON-RPC id, so several tool calls can be in flight at once and a slow one never blocks a fast one. Cancelling a call sends `notifications/cancelled`.

If the server dies, every in-flight call fails with a clear error and the next call spawns a fresh process. The spec makes that safe: the protocol is stateless, so a restarted server is indistinguishable from the original.

An HTTP server has no process. Each request is one POST carrying `MCP-Protocol-Version`, `Mcp-Method` and, where the spec requires it, `Mcp-Name`. The answer is either one JSON object or an SSE stream carrying progress notifications before the result. Cancelling closes the stream, which is itself the cancellation signal in this revision.

**Retries are deliberate and narrow.** A dropped connection retries the listings, which change nothing. A `tools/call` is never retried: a tool may have side effects, and "the server restarted mid-call" is an honest failure the model can reason about, where running someone's `create_issue` twice is not.

## 4. The tool list is cached, and the cache is durable

The framework's registry contract is explicit that a registry fronting a remote tool server "keeps a cached tool list refreshed out of band and does all of its network work inside the callable". So listing inside `get_tools` is out. But an in-memory cache is empty at every boot, which would mean the model silently has no MCP tools on the first turn of every run.

So the catalog is written to `~/.local/share/luca/mcp/catalog.json` and read back at startup. A slice is dropped when the server's command line or URL changed, or when the listing was marked `cacheScope: "private"` and the credentials are different. An expired slice is still served while it refreshes, because losing tools mid-conversation over a lapsed freshness hint is worse than a slightly stale description.

The only genuinely cold moment is the first run after a server is configured. The TUI makes that one visible and bounded rather than silent.

Refreshing happens on the server's own `ttlMs`, and a stdio server that sends `notifications/tools/list_changed` invalidates its slice immediately. The equivalent over HTTP needs a second long-lived `subscriptions/listen` stream and is not implemented.

## 5. Permissions

MCP tools ride the one `PermissionStrategy` the shell plugin builds, so they prompt in `ask` mode like anything else and one "always allow" answer behaves the same way.

They are `ToolKind.OTHER` and stay that way. A server's `readOnlyHint` annotation is carried for display and kept out of the decision: mapping it to `ToolKind.READ` would let a remote tool talk itself past a `tool_kind: read` allow-rule on its own unverified claim.

Rules use the existing vocabulary, with no new schema:

```jsonc
{
  "permissions": {
    "rules": [
      { "decision": "allow", "permission": "mcp", "resource": "files/*" },
      { "decision": "deny",  "permission": "mcp", "resource": "github/delete_*" }
    ]
  }
}
```

## 6. Authentication

`headers` covers every server behind a static token. Values may reference an exported environment variable as `${VAR}`, expanded when the server is built, so a token never has to be written into a file people commit. An undefined variable is a startup error rather than a silent empty header.

`"oauth": true` turns on the browser flow: authorization code with PKCE, a loopback redirect on an OS-assigned port, and RFC 9207 `iss` validation before the code is redeemed. Tokens go to `~/.local/share/luca/mcp/mcp-auth.json`, keyed by **issuer** rather than by label, so two servers behind one authorization server share a login and renaming a label does not orphan a token.

The browser only ever opens from startup or an explicit `/mcp login <server>`, never from inside a turn. A call that finds no usable token fails telling you to run that command.

Registration is dynamic by default, declaring `application_type: "native"` as the revision requires. Set `client_id` to use a pre-registered client instead.

## 7. Where the state lives, and why

The service that owns the connections, the catalog and the tokens is built once in `AgentApp.__init__`, beside `CheckpointService`, and it outlives every session. `/clear`, `/new`, `/resume` and fork all rebuild the runner through `_reset_session`; none of them touch MCP. The plugin and the registry are stateless views holding two references each.

This is load-bearing rather than tidy. When the connections lived on the plugin, `/clear` re-discovered every server and could open a browser in the middle of someone's next message. The plugin deliberately has no `aclose` for the same reason: `_close_plugins()` runs on `/clear` too, and only `_quit` closes the service.

## 8. Configuration

| Field | Meaning |
| --- | --- |
| `command`, `args`, `env` | a local server over stdio. `env` values may use `${VAR}` |
| `url`, `headers` | a remote server over Streamable HTTP. `headers` values may use `${VAR}` |
| `oauth`, `client_id`, `redirect_port` | browser authorization; `redirect_port` pins the loopback port for servers that violate RFC 8252 §7.3 |
| `enabled` | default true |
| `connect_timeout_in_ms` | bounds the probe |
| `list_timeout_in_ms` | bounds one `tools/list` page |
| `call_timeout_in_ms` | bounds one tool call. Unset inherits `runtime.tool_execution_timeout_in_ms` |

Give either `command` or `url`, never both. A server defined one way in `~/.config/luca/luca.json` and the other way in the project file is rejected by name, because the two files are merged field by field and the result would otherwise be a server that is neither.

`mcp.enabled: false` or `--no-mcp` turns the whole thing off.

## 9. Not implemented

**Resources and prompts.** Only tools are exposed. `resources/read` and `prompts/get` are spoken by the client but nothing surfaces them yet.

**Multi Round-Trip Requests.** A server that answers `resultType: "input_required"` gets a completed result marked `is_error`, explaining that it asked for input luca cannot supply, with the requests in `structured_content`. The plumbing fits the framework's deferred-tool machinery exactly, but a deferral with nothing to answer it is a hang, and the existing `ask_user` question model is deliberately too narrow to carry an arbitrary JSON Schema.

**`subscriptions/listen`.** See section 4.

**Sampling, roots and logging.** Deprecated in this revision; luca advertises no capabilities it does not implement, so a server has no reason to ask.

Next: back to the [contrib index](../README.md).
