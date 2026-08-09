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
        self, args: dict, session: AgentSession, conversation_id: str,
        *, tool_name: str, tool_call_id: str, cancellation_token: CancellationToken,
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
| `conversation_id` | which conversation is calling. One tool instance serves the whole tree, so per-conversation state is keyed by this ([`13-subagents.md`](../../13-subagents.md)) |
| `tool_name` | keyword-only — the **effective** name the registry resolved this call under. Not necessarily the class's `name`: one class can be registered under several, and middleware may rewrite the call |
| `tool_call_id` | keyword-only — **which call this is.** The key for per-call state, and for `session.get_tool_execution(tool_call_id)`, which reads this call's own durable `ToolExecution` (its `attempts`, its `raw_tool_call.extras`) |
| `cancellation_token` | keyword-only, always passed — see §8 |

```python
class NoteTool(Tool):
    def __init__(self) -> None:
        self.notes: dict[str, str] = {}          # keyed by conversation, never flat

    async def _execute(self, args, session, conversation_id,
                       *, tool_name, tool_call_id, cancellation_token) -> str:
        self.notes[conversation_id] = args["text"]
        return "noted"
```

> ⚠️ **Unkeyed state is a bug that only appears with subagents.** One instance
> is shared by the main agent and every subagent running in parallel; a flat
> dict means they overwrite each other, silently. Dispatch within one
> conversation is sequential, so a per-conversation slot needs no lock.

`tool_call_id` is the *narrower* key, and a deferring tool cannot work without
it (§10): a call that defers is re-dispatched from scratch on a later drive, so
the id is the only thing connecting the two invocations. It is also the identity
that travels to a remote tool body over a wire.

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
    async def get_approval_context(
        self, args: dict, session: AgentSession, conversation_id: str
    ) -> dict:
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
        self, args: dict, session: AgentSession, conversation_id: str,
        *, tool_name: str, tool_call_id: str, cancellation_token: CancellationToken,
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
- Timing lives on the execution, stamped by the runner — the result carries
  none. `created_at` / `finished_at` bracket the whole call; one
  `ExecutionAttempt` per body invocation carries that invocation's own
  `started_at` / `ended_at` ([`02-data-model.md`](../../02-data-model.md) §3).

`execute` is also the only place a tool can say **"not yet"** — see §10.

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
    title = "Read file"                 # optional UI label — see below
    tool_kind = ToolKind.READ           # read | search | web_fetch | edit | move | delete | execute | switch_mode | other
    namespace = "builtin.fs"            # optional owning group
    version = "1.0.0"                   # optional
    timeout_in_ms = 30_000              # optional per-tool deadline — see §8
    output_schema = ReadFileResult      # optional output model — see §6
    is_private = False                  # optional — keep it off the wire, see §7
```

`title` is presentation only: a UI reads `spec.display_name` (the title when
set, `name` otherwise), while `name` stays the identity for resolution,
approvals and middleware. Reach for it when the internal name is not what a
person should read — `openai_apply_patch` rendering as "Apply patch".

A spec must stay a pure function of the tool *definition*: it is stored once
per session under a content hash ([`03-tools.md`](../../03-tools.md) §4), so
anything call-scoped in it mints a new stored row on every call.

## 6. Structured output

A tool can return a machine-readable payload next to its text. That is two
separate acts: **declaring** the shape, and **producing** the payload.

Declare it by binding a model to the `output_schema` ClassVar. Define the model
at module level — then it stays importable, and a consumer can validate a
payload back through it instead of indexing raw keys:

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict
from luca.agent.core import AgentSession, CancellationToken, ExecutionResult, TextContent, ToolKind
from luca.agent.contrib.tools import Tool


class WeatherReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    degrees_in_celsius: int
    wind_direction: Literal["north", "south", "east", "west"]
    conditions: str


class GetWeatherTool(Tool):
    name = "get_weather"
    description = "Get the current weather for a city."
    Args = CityArgs
    output_schema = WeatherReport          # declares the shape
    tool_kind = ToolKind.WEB_FETCH

    async def execute(                     # produces the payload
        self, args: dict, session: AgentSession, conversation_id: str,
        *, tool_name: str, tool_call_id: str, cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        reading: WeatherReport = await fetch_weather(args["city"])
        return ExecutionResult(
            content=[TextContent(text=f"{reading.degrees_in_celsius}°C, {reading.conditions}.")],
            structured_content=reading.model_dump(),
        )
```

Note `execute`, not `_execute` — the text path has nowhere to put a payload.

`get_tool_spec()` derives `output_schema` from the ClassVar exactly as
`input_schema` comes from `Args`:

```python
ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    input_schema={...},                    # from Args
    output_schema={                        # from WeatherReport
        "additionalProperties": False,
        "properties": {
            "degrees_in_celsius": {"title": "Degrees In Celsius", "type": "integer"},
            "wind_direction": {
                "enum": ["north", "south", "east", "west"],
                "title": "Wind Direction",
                "type": "string",
            },
            "conditions": {"title": "Conditions", "type": "string"},
        },
        "required": ["degrees_in_celsius", "wind_direction", "conditions"],
        "title": "WeatherReport",
        "type": "object",
    },
    tool_kind=ToolKind.WEB_FETCH,
)
```

One call, two channels, two readers:

| Reader | Gets | How |
|---|---|---|
| the **model** | `content` only — `ToolMessage(content=[TextBlock(text="25°C, clear.")])`, identical to the same tool without any of this | `ConversationProjector` ([`10-projection.md`](../../10-projection.md)) |
| your **app** | `execution.result.structured_content` | off the session, or straight off `ToolExecuted` — no session needed |

```python
from luca.agent.core.events import ToolExecuted

match event:
    case ToolExecuted(execution=ex) if ex.result and ex.result.structured_content:
        report = WeatherReport.model_validate(ex.result.structured_content)
        widget.render(report.degrees_in_celsius, report.wind_direction)
```

Importing the model works in-process. Across a boundary — a serialized session,
an MCP server, a web UI — `ex.tool_spec.output_schema` is the contract.

> ⚠️ **Nothing is validated.** A tool returning `{"degrees": "warm"}` still
> records `COMPLETED` and the payload is stored verbatim. Declaring a schema is
> an advertisement, not an enforcement, and `output_schema` never reaches the
> model — no provider has a field for it.

> ⚠️ **Same name, two types.** `Tool.output_schema` is a Pydantic model
> *class*; `ToolSpec.output_schema` is the JSON Schema *dict* derived from it.
> (`Args → input_schema` differ in name because they differ in type; these
> don't.)

## 7. Private tools

`is_private = True` keeps a tool off the wire: the model never sees it and
cannot call it, while the runtime still can
([`03-tools.md`](../../03-tools.md) §6). Everything else — approval, dispatch,
the durable `ToolExecution` — is unchanged.

```python
class SummarizeChildTool(Tool):
    name = "summarize_child"
    description = "Turn a finished subagent's transcript into one result."
    Args = ChildArgs
    is_private = True
```

## 8. Cancellation and timeouts

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
    async def _execute(self, args, session, conversation_id,
                       *, tool_name, tool_call_id, cancellation_token):
        for chunk in stream:
            if cancellation_token.cancelled:
                return "…cut short by cancellation."
            ...
```

Tools that spawn processes **must** kill their process group on
`asyncio.CancelledError` (`start_new_session=True` + `os.killpg`) — the hard
cancel is identical for cancellation and timeout; blocking sync work belongs in
`asyncio.to_thread`.

## 9. `tool()` / `tool_class()` — tools built at runtime

Build a `Tool` from plain callables when the tool is assembled at runtime.
`tool()` returns an instance ready for a registry, `tool_class()` the class:

```python
async def list_files(args: dict, session: AgentSession, conversation_id: str) -> str:
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
| `execute` | becomes `_execute`, the simple text path: async `(args, session, conversation_id) -> str`; per-instance configuration goes in the closure |
| `output` | optional — becomes `output_schema` (§6). Same two forms as `arguments`; a dict is a field spec here too, never a raw JSON Schema |
| `get_approval_context` | optional async `(args, session, conversation_id) -> dict`; overrides an inherited one — passing both is almost certainly a mistake |
| `is_private` | optional — keeps the tool off the wire (§7) |
| `bases` | must contain a `Tool` subclass; MRO order is yours (mixins before `Tool`) |
| `class_attrs` | extra class attributes — mixin requirements, or the remaining ClassVars (`namespace`, `version`, `timeout_in_ms`); colliding with a factory-managed name raises |

> ⚠️ **Anonymous classes.** Every call builds a fresh type with no importable
> qualname (instances don't pickle) and no stable identity. Hand-write the
> class if you need either — subclassing stays the recommended mechanism, and
> is the escape hatch for rich `ExecutionResult` output, validators, and
> per-instance `__init__` state.

> ⚠️ **The factories only wire the text path.** So `output=` lets a
> factory-built tool *declare* an output schema but never populate
> `structured_content` — that needs an `execute` override. Hand-write the class,
> or pass a `bases=` mixin that overrides `execute`, when you need both.

> ⚠️ **`tool_name` and `tool_call_id` are accepted and dropped.** The wrapped
> callable keeps the three-argument shape `(args, session, conversation_id)` the
> factory advertises, so a factory-built tool can neither see which call it is
> nor defer (§10). Hand-write the class for either.

## 10. Deferred results — "not yet"

`execute` may return `ExecutionDeferred()` instead of an `ExecutionResult`: *I
cannot produce the final result yet*. The execution parks at `AWAITING_RESULT`,
the drive returns, and the application resolves whatever the tool is waiting on
before driving again ([`03-tools.md`](../../03-tools.md) §7,
[`04-runner.md`](../../04-runner.md) §9).

```python
from luca.agent.core import ExecutionDeferred, ExecutionResult, TextContent

class RenderVideoTool(Tool):
    name = "render_video"
    description = "Queue a render and return the finished file."
    Args = RenderArgs

    def __init__(self, store: dict) -> None:
        self.store = store                     # keyed by tool_call_id, JSON-shaped

    async def execute(
        self, args: dict, session: AgentSession, conversation_id: str,
        *, tool_name: str, tool_call_id: str, cancellation_token: CancellationToken,
    ) -> ExecutionResult | ExecutionDeferred:
        job = self.store.setdefault(tool_call_id, {"path": None})
        if job["path"] is None:
            return ExecutionDeferred()         # the driver will fill it in
        return ExecutionResult(content=[TextContent(text=f"Rendered to {job['path']}.")])
```

**There is no resume, only re-dispatch.** When a tool defers, nothing about that
call stays alive — no callable, no coroutine, no runner state. The next drive
runs the identical dispatch path from scratch: `before_tool_execution` →
`ToolRegistry.prepare()` → a brand-new callable → invoke. So:

| Fact | Consequence |
|---|---|
| `execute` must be a **pure predicate** | read state, answer *ready / not yet*, never wait. No future, no event, no UI |
| it is invoked once per drive, forever | `setdefault` makes the first dispatch the seeding one and every later one a read |
| the call is one call | one `tool_call_id`, one `ToolExecution`, **one approval** — never re-asked per dispatch — and one `ExecutionAttempt` per invocation |
| whatever state that needs is yours | `AgentSession.extras` is where it goes if it must survive a restart ([`02-data-model.md`](../../02-data-model.md) §7) |

The driver discovers parked calls by reading the session — no event announces
one, because `ToolExecutionStarted` fires per dispatch attempt and `ToolExecuted`
only at the end:

```python
for execution in runner.pending_deferred_tool_executions():
    await resolve_somehow(execution)          # your tool, your protocol
```

> ⚠️ **Nothing bounds a deferral, so the handler must block until it has made
> progress.** Re-dispatching produces no model round, so no step limit or
> doom-loop check trips; a handler that returns without resolving anything
> causes an immediate re-drive, another deferral, and a spin as fast as Python
> runs. The driver owns the cadence — and, since nothing cleans up after a
> cancelled or crashed call, it owns abandonment too.

`timeout_in_ms` still bounds each **body invocation**, never the parked period:
a poll that hangs past the deadline records `TIMED_OUT` like any other, while a
call sitting parked across drives is not aged out by anything.

[`contrib/questions/`](../questions/README.md) is the worked example — the model
asks the user up to four questions and the turn parks until they answer.

Next: [`simple_tool_registry/README.md`](../simple_tool_registry/README.md).
