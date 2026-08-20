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

The model then sees `mcp__files__read_file`, `mcp__github__create_issue`, and so on; the transcript shows them as `files:read_file`. `/mcp` lists the servers, opens the tools any one of them is offering, and lets you sign in, reconnect or switch one off. `/mcp <server>` opens it on that row.

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
from luca.agent.core.events import TextBlock


async def main() -> None:
    servers = McpSettings.model_validate(
        {"servers": {"files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]}}}
    ).build()

    service = McpService(servers, catalog_path=None)
    await service.start()
    print([spec.name for spec in service.specs()])

    session = AgentSessionRunner.new_session(LLMConfig(model="openai/gpt-4o-mini", provider="openrouter"))
    runner = PluginAgentSessionRunner(session, plugins=[McpPlugin(service, YoloPermissionPolicy())])

    runner.post_message("Read note.txt and summarize it.")
    async with runner.run() as run:
        async for event in run:
            match event:
                case TextBlock(text=text):
                    print(text)

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
> and re-runs OAuth — see section 8.

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

The only genuinely cold moment is the first run after a server is configured. The TUI holds that turn — and only that turn — until the listing lands or ten seconds pass, so the alternative is not a faster first message but one where the model silently has none of the tools it was told about.

Refreshing happens on the server's own `ttlMs`, and a stdio server that sends `notifications/tools/list_changed` invalidates its slice and cuts the refresh loop's sleep short, so the new listing arrives in the time one round trip takes rather than at the next TTL. The equivalent over HTTP needs a second long-lived `subscriptions/listen` stream and is not implemented.

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

Beside the records sits a small `resources` index, mapping each MCP server's canonical URI to the issuer that guards it. The issuer takes a network round trip to learn and a fresh process has not made it yet, so without the index a token written under `auth.linear.app` was looked for under `mcp.linear.app` and never found — every relaunch demanding a browser login for a token already on disk.

Two details the spec makes MUST-level, and both fail the same silent way when you skip them:

**The `resource` parameter** (RFC 8707) goes on the authorization request and both token grants, carrying the server's canonical URI. It is what binds the token to this MCP server, and the server is required to reject a token that is not bound to it. Omit it and the browser says Authorized, the token arrives, and the very next request comes back 401 with nothing to explain it.

**Scopes come from the protected-resource document**, not the authorization server's. Both publish `scopes_supported` and they mean different things: the resource's list is what this MCP server needs. A `WWW-Authenticate` challenge overrides it, since the spec makes the challenged scopes authoritative for the operation that failed.

A server you have not signed into is `not authenticated`, never a failure, and is not even listed — it would only answer 401. The browser opens when you ask for it from `/mcp` (or, if a stored token can be refreshed silently, at startup). It never opens from inside a turn: a call that finds no usable token fails telling you to run `/mcp <server>`.

A token that lapsed between listings is renewed on the call path, silently, because that renewal needs no human. One that comes back 401 anyway flips the server to `needs_auth` and records the scopes the challenge named, so the row you go to next is already telling you what to do about it. And a login that produced a token the server then refuses is reported as the failure it is, rather than as the authorization that technically succeeded.

One interop trap, met in the wild: the `Authorization` scheme is sent as the canonical `Bearer`, not as whatever case the token response used. RFC 6749 §7.1 makes `token_type` case-insensitive and Linear answers `"bearer"`, but its own resource server accepts only `Bearer` — so echoing the server's spelling back at it turns a login that worked into a 401 on every request after it.

Registration is dynamic by default, declaring `application_type: "native"` as the revision requires. Set `client_id` to use a pre-registered client instead.

## 7. What `connected` means

That the model can call the server's tools, which is true as soon as there is a
listing for it. Not that a process is running.

The distinction is load-bearing rather than pedantic. Transports are spawned
lazily and the catalog is durable, so a relaunch inside the TTL serves a
server's tools straight off disk and never opens a connection at all. Asking
the transport whether it is alive would report that server as broken while its
tools work perfectly.

| State | Means | Enter does |
| --- | --- | --- |
| `connected` | its tools are available to the model | opens them |
| `stale` | its tools are still available, from a listing that outlived a failed refresh | opens them |
| `connecting` | a listing is in flight | nothing yet |
| `needs_auth` | an `oauth` server nobody has signed into | authenticate |
| `inactive` | it could not be reached, with the reason | retry |
| `disabled` | switched off for this session | enable |

`stale` exists because the two halves of the answer disagree. A refresh that fails does not withdraw the tools — losing them mid-conversation over one bad round trip is what the durable catalog is for — so the model goes on calling a server that may have gone away. Reporting that as `connected` hides it, and reporting it as `inactive` contradicts the tools still on offer. Naming it is the only honest option, and the row carries the failure alongside the count.

`a` authenticates, `r` reconnects, `d` toggles, `enter` does whatever the row's own state makes useful, and `esc` walks back out: out of a tool list, then out of an action in flight, then out of the screen. An action does not dismiss the screen — the row says `authorizing…` while it runs, and `esc` cancels it, which closes the loopback listener with it.

## 8. Where the state lives, and why

The service that owns the connections, the catalog and the tokens is built once in `AgentApp.__init__`, beside `CheckpointService`, and it outlives every session. `/clear`, `/new`, `/resume` and fork all rebuild the runner through `_reset_session`; none of them touch MCP. The plugin and the registry are stateless views holding two references each.

This is load-bearing rather than tidy. When the connections lived on the plugin, `/clear` re-discovered every server and could open a browser in the middle of someone's next message. The plugin deliberately has no `aclose` for the same reason: `_close_plugins()` runs on `/clear` too, and only `_quit` closes the service.

## 9. Configuration

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

## 10. Not implemented

**Resources and prompts.** Only tools are exposed. `resources/read` and `prompts/get` are spoken by the client but nothing surfaces them yet.

**Multi Round-Trip Requests.** A server that answers `resultType: "input_required"` gets a completed result marked `is_error`, explaining that it asked for input luca cannot supply, with the requests in `structured_content`. The plumbing fits the framework's deferred-tool machinery exactly, but a deferral with nothing to answer it is a hang, and the existing `ask_user` question model is deliberately too narrow to carry an arbitrary JSON Schema.

**`subscriptions/listen`.** See section 4.

**Sampling, roots and logging.** Deprecated in this revision; luca advertises no capabilities it does not implement, so a server has no reason to ask.

Next: back to the [contrib index](../README.md).
