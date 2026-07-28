# Tools

To the core, a tool is **data**: a `ToolSpec` — a name, a description, and the
arguments as a JSON Schema. Nothing in `luca.agent.core` references a Python
tool class. The **tool registry** the runner is constructed with
([`05-permissions.md`](05-permissions.md)) advertises specs, mints one
execution record per call, decides approval, and prepares the dispatch; the
core records what happened.

Python authors rarely hand-write a spec — the ergonomic `Tool` base class in
[`contrib/tools/`](contrib/tools/README.md) derives one from a Pydantic `Args`
model. This page is what the core itself knows, and all a registry fronting a
remote tool server needs.

```python
from luca.agent.core import ToolKind, ToolSpec
```

## 1. `ToolSpec` — the core's only tool type

| Field | Meaning |
|---|---|
| `name` | the name the model calls |
| `description` | required — the client's wire tool type rejects null |
| `input_schema` | required — the arguments as a JSON Schema dict |
| `metadata` | free-form, registry-owned; never interpreted by the core |
| `tool_kind` / `namespace` / `version` / `timeout_in_ms` | identity and deadline — §2 |

```python
READ_FILE = ToolSpec(
    name="read_file",
    description="Read a file from disk and return its contents.",
    input_schema={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Absolute path."}},
        "required": ["path"],
    },
    tool_kind=ToolKind.READ,
)
READ_FILE.spec_id()   # 'caded1dc…' — sha256 of the canonical JSON, 64 hex chars (§4)
```

One type plays two roles: the **advertisement** sent to the model
(`name` / `description` / `input_schema`) and the **identity snapshot** kept on
a past execution, so a session whose tools were deleted from the codebase years
ago still renders what it called. It carries no arguments — those live on
`ToolExecution.raw_tool_call`.

> ⚠️ **`input_schema` is required and never `None`.** A tool that takes no
> arguments advertises the empty object schema —
> `{"type": "object", "properties": {}}`. An absent schema and an empty schema
> mean different things to a provider.

## 2. Kind, namespace, version — and the deadline

| Field | What it does |
|---|---|
| `tool_kind` | classification permission policies select on: `read` `search` `web_fetch` `edit` `move` `delete` `execute` `switch_mode` `other` (default) |
| `namespace` | the owning tool group, e.g. `"builtin.fs"` |
| `version` | the tool's version at call time, e.g. `"1.0.0"` |
| `timeout_in_ms` | this tool's execution deadline; beats `RuntimeConfig.tool_execution_timeout_in_ms` ([`08`](08-runtime-config.md)), `None` defers to it, `-1` disables |

The deadline is read from the spec **recorded at birth**, so redefining the
tool mid-run never moves an in-flight call's deadline.

> ⚠️ **`timeout_in_ms` bounds the BODY only.** A tool configured with
> `timeout_in_ms=5000` is not bounded end to end: listing tools, minting the
> record, deciding approval and `prepare()` all run outside it, under no
> deadline at all. Expiry hard-cancels the body and records `TIMED_OUT`,
> resultless.

## 3. A registry advertises specs

`get_tools` is a query, re-run fresh before every LLM call; the adapter
projects each spec onto the wire tool the provider sees
(`input_schema` → `parameters`, verbatim). Because a spec is plain data, a
registry backed by a remote tool server hands back the JSON Schema it already
has — no Python tool class anywhere:

```python
from luca.agent.core import (
    AgentSession, ApprovalDecision, ApprovalOption, CancellationToken,
    ExecutionResult, TextContent, ToolCall, ToolExecution, ToolNotFound,
    ToolRegistry, ToolSpec,
)

SPECS = {READ_FILE.name: READ_FILE}

class RemoteToolRegistry(ToolRegistry):
    async def get_tools(self, session: AgentSession) -> list[ToolSpec]:
        return list(SPECS.values())

    async def create_execution(self, session: AgentSession, call: ToolCall) -> ToolExecution:
        return ToolExecution(              # a birth DRAFT: no id, no timestamps
            tool_call_id=call.id,
            raw_tool_call=call,
            tool_spec=SPECS.get(call.name),
        )

    async def decide(self, session: AgentSession, execution: ToolExecution) -> ApprovalDecision:
        return ApprovalDecision(decision=ApprovalOption.ALLOW)

    async def prepare(self, session: AgentSession, execution: ToolExecution):
        call = execution.raw_tool_call
        spec = SPECS.get(call.name)
        if spec is None:
            raise ToolNotFound(f"Unknown tool: {call.name!r}.")

        async def run(*, cancellation_token: CancellationToken) -> ExecutionResult:
            reply = await tool_server.call(spec.name, call.arguments)   # your transport
            return ExecutionResult(content=[TextContent(text=reply)])

        return run
```

The four-method contract, the approval gate and the rules for registry authors
are [`05-permissions.md`](05-permissions.md). One of them belongs here:
**the core never validates arguments against `input_schema`** — it knows the
schema but will never check a call against it, because a registry may delegate
to a remote server that validates on its own side and double validation would
break it. Validation is the registry's.

## 4. Where specs live: `session.tool_specs`

Specs are stored once per session, not once per call:
`AgentSession.tool_specs` maps `spec_id()` → `ToolSpec`, append-only, written
only by the framework's write doors. Each execution keeps the durable
reference, `tool_spec_id`.

```python
execution.tool_spec.name              # works in memory AND after a reload
execution.tool_spec_id                # the durable ref into session.tool_specs
session.tool_specs[execution.tool_spec_id] is execution.tool_spec   # True
```

Both are `None` when the tool never resolved. `tool_spec` is a restorable
cache: serializing a **session** strips the inline copies and writes the shared
store once; serializing a standalone `ToolExecution` keeps its spec inline,
which is what makes a tool lifecycle event self-describing to a consumer
holding no session. Constructing an `AgentSession` restores every `tool_spec`
from its id, and refuses a session it cannot fully restore — a dangling
`tool_spec_id`, or a `tool_spec` with no id (a pre-normalization file), raises
on load. There is no migration: regenerate old session files.

> ⚠️ **A spec must be a pure function of the tool definition.** Put anything
> call-scoped in `metadata` — a timestamp, a request id — and every call mints
> a new stored row, silently defeating the normalization. No error, no warning.

Registries never compute a `spec_id` and never touch `session.tool_specs`: hand
back a draft with `tool_spec` populated and the framework files it.

## 5. The one-output invariant

**Every tool call produces exactly one tool execution, and exactly one tool
output.** A call that never reaches its body is born or made terminal, and
`started_at` / `dispatched` say so — they mean "the body was dispatched",
always:

| Where it ended | `status` | `dispatched` |
|---|---|---|
| registry authored a terminal birth | `NOT_FOUND` / `INVALID` / `FAILED` | `False` |
| `create_execution` raised (or the runner has no registry) | `FAILED` / `NOT_FOUND` | `False` |
| `decide` returned DENY | `REJECTED` | `False` |
| `prepare` raised `ToolNotFound` | `NOT_FOUND` | `False` |
| `prepare` raised `InvalidToolArguments` or a pydantic `ValidationError` | `INVALID` | `False` |
| `prepare` raised anything else, or returned a non-callable | `FAILED` | `False` |
| cancelled up to and including `prepare()` settling | `CANCELLED` | `False` |
| the body returned an `ExecutionResult` | `COMPLETED` | `True` |
| the body raised (any exception type) | `FAILED` | `True` |
| the deadline expired | `TIMED_OUT` | `True` |
| the cancel grace expired, or the process died mid-body | `INTERRUPTED` | `True` |

`FAILED` / `NOT_FOUND` / `INVALID` carry a structured `ToolExecutionError`;
the other terminal statuses are complete facts on their own (resultless,
errorless). Every registry- or tool-owned raise stamps `details["phase"]` —
`"create_execution"`, `"prepare"` or `"execution"` — so a failure is
attributable to the phase that produced it.

You never have to handle these paths: the
[`ConversationProjector`](10-projection.md) derives a correlated tool message
for every terminal status, so the model always sees exactly one output per call
it made. Failures are isolated — one bad call never touches its siblings.
Next: [`04-runner.md`](04-runner.md).
