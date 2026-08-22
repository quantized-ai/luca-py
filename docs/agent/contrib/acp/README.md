# `luca.agent.contrib.acp` — luca as an ACP agent

The [Agent Client Protocol](https://agentclientprotocol.com) is JSON-RPC 2.0
over stdio. A client (Zed, [Nori](https://github.com/tilework-tech/nori-cli),
[Pool](https://github.com/poolsideai/pool),
[acp-ui](https://github.com/formulahendry/acp-ui)) spawns the agent as a
subprocess and talks to it on stdin and stdout. This package is the adapter.

```bash
uv run python -m luca.agent.contrib.acp
```

It drives the same [`AgentApplication`](../app/README.md) the TUI does, so the
two front ends compose identically and cannot drift. Requires the `acp`
dependency group (`agent-client-protocol`, whose only runtime dependency is
pydantic).

## 1. Point a client at it

Zed, in `settings.json`:

```json
{
  "agent_servers": {
    "luca": {
      "type": "custom",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/luca-py", "python", "-m", "luca.agent.contrib.acp"],
      "env": {}
    }
  }
}
```

Credentials work exactly as they do for the TUI: `~/.local/share/luca/auth.json`,
or the provider's own environment variable. There are no ACP `authMethods` —
nothing here has anything for a client to log into.

Flags mirror the TUI's, minus everything that needs a screen: `--workspace`,
`--config`, `--log-level`, `--faux`, `--no-checkpoints`, `--no-subagents`,
`--no-skills`, `--no-instructions`.

> ⚠️ **stdout is the protocol.** Logging goes to `<session dir>/logs/<id>.log`
> and nowhere else. One stray `print` corrupts the stream and the client
> disconnects.

## 2. What it advertises

| Capability | Value | Why |
|---|---|---|
| `protocolVersion` | `1` | v2 exists in the schema repo but is still unstable |
| `loadSession` | `true` | sessions already persist per project; §5 |
| `promptCapabilities` | image, audio, embeddedContext | `ContentPart` is text \| image \| audio \| file |
| `mcpCapabilities` | absent | nothing in luca speaks MCP yet (#25); servers passed to `session/new` are ignored, with a warning |
| `authMethods` | `[]` | see above |

Client capabilities it never uses: `fs/read_text_file` and
`fs/write_text_file` (edits go straight to disk, so unsaved editor buffers are
invisible), and `terminal/*` (it runs its own subprocesses, so a long command
reports only when it finishes). Both are optional in the protocol and clients
degrade cleanly.

## 3. The mapping

The two vocabularies line up almost exactly, which is most of why this package
is small. `stream.py` holds it all.

| luca | ACP |
|---|---|
| `TextDelta` / `TextBlock` | `agent_message_chunk` |
| `ReasoningDelta` / `ReasoningBlock` | `agent_thought_chunk` (a redacted one is dropped) |
| `ToolCallReceived` | `tool_call`, status `pending` |
| `ToolExecutionStarted` | `tool_call_update`, `in_progress` |
| `ToolExecuted` | `tool_call_update`, `completed` / `failed`, with diffs and locations |
| `update_todos` | `plan` |
| `ToolKind` | identical, except ours says `web_fetch` where ACP says `fetch` |
| `ExecutionStatus` (13) | `pending` / `in_progress` / `completed` / `failed` |
| `TurnOutcome.COMPLETED` / `CANCELLED` | `end_turn` / `cancelled` |

`ERRORED` and `TIMED_OUT` have no ACP stop reason, so they raise a JSON-RPC
error. A client shows a failed turn rather than a successful one with nothing
in it, and the TUI's retry prompt has no counterpart here.

Two tests guard the tables rather than the behaviour: a `ToolKind` or an
`ExecutionStatus` added to core without a mapping fails
`tests/agent/contrib/acp/test_stream.py`.

## 4. Subagents, in a protocol that has none

An ACP session is ONE stream. A luca run yields events for the main
conversation and every child at once, tagged by `conversation_id`.

So a child's output is folded into the tool call that spawned it: `Translator`
maps each child conversation to its `spawn_subagent` call and emits the child's
text as `tool_call_update` content on it. A client shows nested work as
progress on the spawn, which is the only reading ACP has room for. A child's
*thinking* is dropped outright — one nested voice is enough, and folding two
buries the output.

A child whose spawn call cannot be resolved is dropped and logged. That happens
on a resumed session, where the spawn was in a previous process.

## 5. Approvals, questions, load

**Approvals.** `session/request_permission`, one request per approval STEP —
an execution can need several (directory access, then the tool's own verb) and
ACP carries one flat option list per request. Our four prompt options map to
`allow_once` / `allow_always` / `reject_once`; there is no cancel option
because ACP models "stop everything" as the request's *outcome*.

The gate is read from `runner.pending_approvals()` after the run drains, not
from the `ApprovalRequired` event — the event can be superseded, the durable
read cannot. Same reason the TUI ignores it.

**Questions.** A parked `ask_user` becomes `elicitation/create` when the client
advertises elicitation. When it does not — Zed did not, when this was written —
the call is answered with a note telling the model to ask in prose instead. The
handler must always resolve something: there is no framework backoff, so
returning without answering spins.

**Load.** `session/load` replays the main conversation's path as
`session/update` notifications before answering, per the spec. The ACP session
id IS the luca session id, so a load is a lookup in
`~/.luca/projects/<encoded-project-path>/`, which already keys on the workspace
ACP passes as `cwd`.

## 6. Testing it

```bash
uv run py.test tests/agent/contrib/acp/
```

Three layers, none of which need a key or a network:

- `test_stream.py` — the mapping, one event at a time.
- `test_agent.py` — the protocol surface driven in-process against `--faux`,
  whose scripted conversation covers thinking, a gated call, a subagent spawn
  and the wrap-up in one turn.
- `test_stdio.py` — a real subprocess over a real pipe, which is the only thing
  that catches a stray write to stdout.

## Not implemented

MCP (#25), the client filesystem, terminals, `session/list`,
`session/set_config_option`, and ACP v2.

Next: [`app`](../app/README.md) — the headless layer this package drives.
