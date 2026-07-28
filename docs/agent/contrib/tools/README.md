# Tools — the `Tool` base class

The ergonomic way to write a tool **in Python**. The core's only tool type is
`ToolSpec`, plain JSON-serializable data ([`03-tools.md`](../../03-tools.md));
this package is a convenience for tools that live in this process: subclass
`Tool`, declare a Pydantic `Args` model, override one method, and
`get_tool_spec()` derives the spec. Nothing in the core depends on it, and a
registry fronting a remote tool server never needs it —
[`resource_permissions`](../resource_permissions/README.md),
[`shell`](../shell/README.md) and `memory` all build on it.

```python
from luca.agent.contrib.tools import Tool, tool, tool_class
```

## 1. The simplest tool

Three class vars — `name`, `description`, `Args` — and one `_execute`:

```python
from pydantic import BaseModel, Field
from luca.agent.core import AgentSession, CancellationToken

class ReadFileArgs(BaseModel):
    path: str = Field(description="Absolute path of the file to read.")

class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a file from disk and return its contents."
    Args = ReadFileArgs

    async def _execute(
        self, args: dict, session: AgentSession,
        *, cancellation_token: CancellationToken,
    ) -> str:
        with open(args["path"]) as f:
            return f.read()
```

Wrap **instances** in a registry, dispatched by `name`:

```python
from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry, YoloPermissionPolicy

registry = SimpleToolRegistry(tools=[ReadFileTool()], permission_policy=YoloPermissionPolicy())
runner = AgentSessionRunner(session, tool_registry=registry)
```

`SimpleToolRegistry` validates the model's arguments against `Args` before you
are ever called ([`simple_tool_registry`](../simple_tool_registry/README.md)) —
the core never does that for anyone.

## 2. What the body receives

| Argument | What it is |
|---|---|
| `args` | the **validated** arguments, dumped from `Args` to a plain dict |
| `session` | the live `AgentSession` — read `session.id`, `session.session_config.llm_config` |
| `cancellation_token` | keyword-only, always passed — see §6 |

> ⚠️ **The session is read-only.** The runner owns every write to session
> state; a tool that mutates what it was handed corrupts the ledger. Per-run
> application state is not the framework's concern — a tool is application
> code, so hold your own references or read a `contextvars.ContextVar`.

## 3. Approval context

Tools don't decide whether they may run — the registry does
([`05-permissions.md`](../../05-permissions.md)). Define
`get_approval_context` to hand the permission policy whatever it needs to
decide: the resources touched, a preview, suggested "always allow" grants. It
receives the validated args and the session; `SimpleToolRegistry` stores the
returned dict under `ToolExecution.extras["approval_context"]` and the core
never reads it — its shape is a private contract between your tool and your
policy.

```python
class ReadFileTool(Tool):
    ...
    async def get_approval_context(self, args: dict, session: AgentSession) -> dict:
        return {
            "resources": [args["path"]],
            "preview": f"Read file {args['path']}",
        }
```

> ⚠️ **A convention, not a base-class method.** `Tool` doesn't declare
> `get_approval_context` — it's duck-typed: `SimpleToolRegistry` calls it iff
> your tool defines it (a raise there makes the birth `FAILED`). A custom
> registry may read, or ignore, anything else.

> ⚠️ **Don't block here.** This is awaited on the event loop, inside the
> registry's `create_execution`, under no deadline. Stat a path or read a file
> through `asyncio.to_thread` — a blocking syscall can't be interrupted by
> cancellation, so one hung network mount stalls the whole run.
> `ResourcePermissionToolMixin`
> ([`contrib/resource_permissions/`](../resource_permissions/README.md) §6)
> does this for you: its override point is a plain `def` it runs in a thread.

## 4. Rich results

`_execute → str` is the easy path. For a failure flag, metadata, or multi-block
output, override `execute` and return an `ExecutionResult` instead:

```python
from luca.agent.core import ExecutionResult, TextContent

class RunSqlTool(Tool):
    name = "run_sql"
    description = "Execute a read-only SQL query."
    Args = SqlArgs

    async def execute(
        self, args: dict, session: AgentSession,
        *, cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        try:
            rows = await db.fetch(args["query"])
        except QueryError as e:
            return ExecutionResult(content=[TextContent(text=str(e))], is_error=True)
        return ExecutionResult(
            content=[TextContent(text=format(rows))],
            metadata={"row_count": len(rows)},
        )
```

- `is_error=True` is **your verdict about the returned result** — the execution
  still records `COMPLETED` (the framework received a result). Raising is what
  records `FAILED`, with a structured `ToolExecutionError` instead of a result.
- Timing lives on the execution (`started_at` / `ended_at`), stamped by the
  runner — the result carries none.

`content` takes the same `ContentPart` union a user message does, so a tool can
return an image — a screenshot tool, or the shell `read` tool on a png:

```python
return ExecutionResult(content=[
    ImageContent(source=ImageBase64(data=b64, media_type="image/png")),
])
```

> ⚠️ **Return what the tool actually produced.** Whether the target provider
> can receive it is the adapter layer's problem
> ([`10-projection.md`](../../10-projection.md)), not the tool's. Today an
> image in a tool result reaches Anthropic and raises on the OpenAI
> chat-completions API.

## 5. What lands in the `ToolSpec`

`get_tool_spec()` snapshots the class vars into the `ToolSpec` the registry
advertises and records on every execution, with
`input_schema=Args.model_json_schema()` verbatim:

```python
from luca.agent.core import ToolKind

class ReadFileTool(Tool):
    name = "read_file"
    description = "..."
    Args = ReadFileArgs
    tool_kind = ToolKind.READ           # read | search | web_fetch | edit | move | delete | execute | switch_mode | other
    namespace = "builtin.fs"            # optional owning group
    version = "1.0.0"                   # optional
    timeout_in_ms = 30_000              # optional per-tool deadline — see §6
```

A spec must stay a pure function of the tool *definition*: it is stored once
per session under a content hash ([`03-tools.md`](../../03-tools.md) §4), so
anything call-scoped in it mints a new stored row on every call.

## 6. Cancellation and timeouts

The runner races the body against the run's cancellation token and an optional
deadline — `timeout_in_ms`, else `RuntimeConfig.tool_execution_timeout_in_ms`
([`08-runtime-config.md`](../../08-runtime-config.md)). By default a cancelled
or timed-out tool is **hard-cancelled** and recorded resultless
(`INTERRUPTED` / `TIMED_OUT`) — you write nothing extra.

> ⚠️ **The deadline bounds the body only.** A tool with `timeout_in_ms=5000`
> is *not* bounded end to end: resolution, argument validation and the approval
> preflight all happen outside it, under no deadline at all.

A **cooperative** tool can watch the token and return partial output within the
cancellation grace window (`RuntimeConfig.tool_cancellation_grace_period`,
default 0) — whatever it returns becomes its real result:

```python
    async def _execute(self, args, session, *, cancellation_token):
        for chunk in stream:
            if cancellation_token.cancelled:
                return "…cut short by cancellation."
            ...
```

Tools that spawn processes **must** kill their process group on
`asyncio.CancelledError` (`start_new_session=True` + `os.killpg`) — the hard
cancel is identical for cancellation and timeout; blocking sync work belongs in
`asyncio.to_thread`.

## 7. `tool()` / `tool_class()` — tools built at runtime

Build a `Tool` from plain callables when the tool is assembled at runtime.
`tool()` returns an instance ready for a registry, `tool_class()` the class:

```python
async def list_files(args: dict, session: AgentSession) -> str:
    return "\n".join(os.listdir(args["path"]))

ls = tool(
    name="list_files",
    description="List the files in a directory.",
    arguments={"path": (str, Field(default="."))},   # or a ready BaseModel subclass
    execute=list_files,
    tool_kind=ToolKind.READ,
    class_attrs={"namespace": "builtin.fs", "timeout_in_ms": 5_000},
)
```

| Parameter | Notes |
|---|---|
| `arguments` | a `BaseModel` subclass used as `Args` as-is, or a `create_model` field spec compiled into an `extra="forbid"` model |
| `execute` | becomes `_execute`, the simple text path: async `(args, session) -> str`; per-instance configuration goes in the closure |
| `get_approval_context` | optional async `(args, session) -> dict`; overrides an inherited one — passing both is almost certainly a mistake |
| `bases` | must contain a `Tool` subclass; MRO order is yours (mixins before `Tool`) |
| `class_attrs` | extra class attributes — mixin requirements, or the remaining ClassVars (`namespace`, `version`, `timeout_in_ms`); colliding with a factory-managed name raises |

> ⚠️ **Anonymous classes.** Every call builds a fresh type with no importable
> qualname (instances don't pickle) and no stable identity. Hand-write the
> class if you need either — subclassing stays the recommended mechanism, and
> is the escape hatch for rich `ExecutionResult` output, validators, and
> per-instance `__init__` state.

Next: [`simple_tool_registry/README.md`](../simple_tool_registry/README.md).
