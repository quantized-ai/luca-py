# Tools

A tool is a Python class that describes itself to the model and does work. You
subclass `Tool`, set three class vars, and override one method. Everything
around the body — resolving the call, validating arguments, approval, dispatch
— is owned by the **tool registry** the runner is constructed with
([`05-permissions.md`](05-permissions.md)); the `Tool` class is the execution
contract only.

```python
from luca.agent.core import CancellationToken, Tool, ToolContext
```

## 1. The simplest tool

Three class vars — `name`, `description`, `Args` — and one `_execute`. `Args` is
a Pydantic model; the registry turns it into the JSON schema the model sees, and
validates the model's arguments against it before you're ever called.

```python
from pydantic import BaseModel, Field

class ReadFileArgs(BaseModel):
    path: str = Field(description="Absolute path of the file to read.")

class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a file from disk and return its contents."
    Args = ReadFileArgs

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        with open(args["path"]) as f:
            return f.read()
```

`_execute` is `async`, receives the **validated** args as a dict plus the
keyword-only cancellation token (see §6), and returns a string for the model.
Wrap **instances** in a registry, dispatched by `name`:

```python
from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry, YoloPermissionPolicy

registry = SimpleToolRegistry(tools=[ReadFileTool()], permission_policy=YoloPermissionPolicy())
runner = AgentSessionRunner(session, tool_registry=registry)
```

## 2. The context

Every execution receives a `ToolContext` — information-only, built fresh per
run, never persisted:

```python
context.session_id          # str
context.model               # LLMConfig — the active model
```

The cancellation token is **not** on the context — it arrives as the
keyword-only `cancellation_token` argument.

## 3. Approval context

Tools don't decide whether they may run — the registry does
([`05-permissions.md`](05-permissions.md)). Define `get_approval_context` to
hand `SimpleToolRegistry`'s permission policy whatever it needs to decide: the
resources touched, a preview, suggested "always allow" grants. It receives the
validated args; the registry stores the returned dict under
`ToolExecution.extras["approval_context"]` and the core never reads it — its
shape is a private contract between your tool and your policy.

```python
class ReadFileTool(Tool):
    ...
    async def get_approval_context(self, args: dict, context: ToolContext) -> dict:
        return {
            "resources": [args["path"]],
            "preview": f"Read file {args['path']}",
        }
```

> ⚠️ **A convention, not a base-class method.** `Tool` doesn't declare
> `get_approval_context` — it's duck-typed: `SimpleToolRegistry` calls it iff
> your tool defines it. A custom registry may read (or ignore) anything else.

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
        self, args: dict, context: ToolContext,
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
> can receive it is the adapter layer's problem ([10](10-projection.md)), not
> the tool's. Today an image in a tool result reaches Anthropic and raises on
> the OpenAI chat-completions API.

## 5. Tool identity — `tool_kind`, `namespace`, `version`, `timeout_in_ms`

These class vars are snapshotted into the `ToolSpec` recorded on every execution
at birth, so a saved session stays identifiable even after the tool changes or
is removed. `tool_kind` also classifies the call for permission policies.

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

## 6. Cancellation and timeouts

The runner races every execution against the run's cancellation token and an
optional deadline. By default a cancelled or timed-out tool is **hard-cancelled**
and recorded resultless (`INTERRUPTED` / `TIMED_OUT`) — you write nothing extra.

Set a per-tool deadline via the `timeout_in_ms` class var (snapshotted into the
birth `ToolSpec`; beats the global `tool_execution_timeout_in_ms`, see
[`08-runtime-config.md`](08-runtime-config.md)):

```python
class SlowTool(Tool):
    timeout_in_ms = 30_000   # 30s; None defers to RuntimeConfig; -1 disables
```

A **cooperative** tool can watch the token and return partial output within the
cancellation grace window — whatever it returns becomes its real result:

```python
    async def _execute(self, args, context, *, cancellation_token):
        for chunk in stream:
            if cancellation_token.cancelled:
                return "…cut short by cancellation."
            ...
```

Tools that spawn processes **must** kill their process group on
`asyncio.CancelledError` (`start_new_session=True` + `os.killpg`); blocking sync
work belongs in `asyncio.to_thread`.

## 7. The one-output invariant

**Every tool call produces exactly one tool output.** A call that fails the
registry's preflight is born terminal — with a structured
`ToolExecutionError` authored by the registry — and never reaches the
approval decision (`SimpleToolRegistry`'s outcomes):

| Birth failure | `status` | `error.error_type` |
|---|---|---|
| Unknown tool name | `NOT_FOUND` (`tool_spec=None`) | `ToolNotFound` |
| Arguments fail `Args` | `INVALID` (pydantic errors under `error.details["errors"]`) | `InvalidToolArguments` |
| `get_approval_context` raises | `FAILED` | the exception's class name |

A denied call becomes `REJECTED`; a cancelled-before-start call `CANCELLED`;
grace expiry `INTERRUPTED`; deadline expiry `TIMED_OUT` — all resultless and
errorless (the status is the whole fact). You never have to handle these paths:
the [`ConversationProjector`](10-projection.md) derives a correlated tool
message for every terminal status, so the model always sees exactly one output
per call it made. Birth failures are isolated — one bad call never touches
its siblings. Next: [`04-runner.md`](04-runner.md).
