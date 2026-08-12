# Tools

The SDK supports three input styles for a tool's parameter schema, all
first-class at the API boundary:

1. A raw **JSON Schema** `dict` — low-level escape hatch.
2. A **Pydantic `BaseModel`** subclass — idiomatic Python.
3. A `TypeAdapter[...]` wrapping a `TypedDict` or any other type.

Each transport normalizes to whatever the provider expects (JSON Schema on
the wire for OpenAI/Anthropic/etc.). The original is preserved on the
`Tool` instance so you can validate inbound `ToolCall.arguments` against it
afterwards.

## `Tool`

```python
from pydantic import BaseModel
from luca.client.types import Tool

class WeatherArgs(BaseModel):
    location: str
    units: str = "celsius"

tool = Tool(
    name="get_weather",
    description="Get the current weather for a location.",
    parameters=WeatherArgs,    # BaseModel subclass
)
```

Equivalent forms:

```python
# Raw JSON Schema
Tool(name="get_weather", description="...", parameters={
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "units": {"type": "string", "default": "celsius"},
    },
    "required": ["location"],
})

# TypeAdapter
from typing import TypedDict
from pydantic import TypeAdapter

class WeatherTD(TypedDict):
    location: str
    units: str

Tool(name="get_weather", description="...", parameters=TypeAdapter(WeatherTD))
```

## Tool choice

`tool_choice=` controls how aggressively the model picks tools:

| Value | Meaning |
|---|---|
| `"auto"` | Model decides (default if `tools=` is passed). |
| `"required"` | Must call at least one tool. |
| `"none"` | Forbid tool calls. |
| `{"name": "..."}` | Force one tool, provider-agnostic: resolved against the declared tools and projected to the provider's forcing shape (native tools included). |
| `{"type": "function", "function": {"name": "..."}}` (or similar provider shape) | Force a specific tool, provider shape passed through verbatim. |

## Calling tools in a loop (sync)

This is the canonical agent loop. See [`main.py`](../../main.py) for a runnable
version with three math tools.

```python
from collections.abc import Callable
from pydantic import BaseModel, Field

from luca.client import completion
from luca.client.types import (
    Tool, ToolCall, ToolMessage, UserMessage, TextBlock,
)

class BinaryOp(BaseModel):
    a: float = Field(description="First operand.")
    b: float = Field(description="Second operand.")

def add(a, b): return a + b
def multiply(a, b): return a * b

TOOLS = [
    Tool(name="add", description="Add two numbers.", parameters=BinaryOp),
    Tool(name="multiply", description="Multiply two numbers.", parameters=BinaryOp),
]
REGISTRY: dict[str, Callable] = {"add": add, "multiply": multiply}

def execute(tc: ToolCall) -> str:
    fn = REGISTRY[tc.name]
    return str(fn(**tc.arguments))

messages = [UserMessage(content=[TextBlock(text="What is (15+25)*4?")])]

while True:
    response = completion(
        model="anthropic:claude-3-5-sonnet-latest",
        messages=messages,
        system_message="Use the tools for any arithmetic.",
        tools=TOOLS,
    )
    messages.extend(response.messages)
    answer = response.messages[-1]

    if answer.finish_reason != "tool_use":
        break

    for tc in answer.tool_calls:
        result = execute(tc)
        messages.append(ToolMessage(
            tool_call_id=tc.id,
            content=[TextBlock(text=result)],
        ))
```


Two things to notice:

- `response.messages` is appended directly. Each entry is already an
  `AssistantMessage` with `tool_calls` inside `content` — no manual
  reconstruction. The tool calls to execute are on the LAST one.
- `message.tool_calls` is a **filter** of `message.content` (same
  instances). Mutating a `ToolCall` from either view mutates both.

## Parsing arguments against a schema

If you want validated, typed arguments rather than the raw dict:

```python
for tc in response.messages[-1].tool_calls:
    args = tc.parse_arguments(BinaryOp)  # → BinaryOp instance
    result = add(args.a, args.b)
```

`parse_arguments` accepts a Pydantic `BaseModel` subclass or a `TypeAdapter`.
On validation failure it raises `StructuredOutputError`.

## Streaming tool calls

In a streaming response, tool calls arrive in pieces. The accumulator
mutates a single `ToolCall` instance per call:

- `tool_call_start` — block created, `complete=False`, `arguments={}`.
- `tool_call_delta` — `partial_arguments` accumulates raw JSON fragments.
- `tool_call_end` — `arguments` parsed from `partial_arguments`,
  `complete=True`, `partial_arguments=""`.

`stream.tool_calls` and `partial.tool_calls` are filter views over the same
instances, so they always reflect the live state. See
[`08-streaming.md`](08-streaming.md) for the full event vocabulary and
[`main.py`](../../main.py) (run with `--streaming`) for a streaming agent loop.

## Provider-native tools

Some tools are built into the provider: the model is trained against them and
the declaration is a type marker, not a schema. Declare them as instances
alongside your own tools — the loop above works unchanged, and the client
still only *describes* tools; **you execute** (apply the diffs, own the shell
session, report honest failures). For the two file-editing tools you don't
have to write that yourself: see
[`luca.client.native`](#executing-them-lucaclientnative).

| Tool | Provider | Call arrives as | Result you return |
|---|---|---|---|
| `ApplyPatchTool()` | OpenAI (Responses) | `ApplyPatchToolCall`, name `"apply_patch"` | plain `ToolMessage` (`is_error` → wire status) |
| `LocalShellTool()` | OpenAI (Responses) | `ShellToolCall`, name `"shell"` | `ShellToolMessage` (structured, required) |
| `TextEditorTool(max_characters=...)` | Anthropic | base `ToolCall`, name `"str_replace_based_edit_tool"` | plain `ToolMessage` |
| `BashTool()` | Anthropic | base `ToolCall`, name `"bash"` | plain `ToolMessage` |

Every typed class above has an equivalent canonical-types-only form — see
[the generic form](#the-generic-form-extras).

### OpenAI: apply_patch + shell

Calls surface as **typed `ToolCall` subclasses** with synthesized names (the
wire items carry none) and a typed accessor over `arguments`:

```python
from luca.client.providers.openai import (
    ApplyPatchTool, ApplyPatchToolCall,
    LocalShellTool, ShellToolCall, ShellToolMessage,
    ShellCommandResult, ShellExitOutcome, ShellTimeoutOutcome,
)
from luca.client.types import TextBlock, ToolMessage

def handle_apply_patch(tc: ApplyPatchToolCall) -> ToolMessage:
    op = tc.operation      # {"type": "create_file"|"update_file"|"delete_file", "path", "diff"?}
    ok, detail = apply_v4a(op)
    return ToolMessage(tool_call_id=tc.id, name=tc.name,
                       content=[TextBlock(text=detail)], is_error=not ok)

def handle_shell(tc: ShellToolCall) -> ShellToolMessage:
    action = tc.action     # {"commands": [...], "timeout_ms"?, "max_output_length"?}
    results = [
        ShellCommandResult(
            stdout=r.stdout, stderr=r.stderr,
            outcome=ShellExitOutcome(exit_code=r.code) if not r.timed_out
                    else ShellTimeoutOutcome(),
        )
        for r in (run(cmd, timeout_ms=action.get("timeout_ms")) for cmd in action["commands"])
    ]
    return ShellToolMessage(tool_call_id=tc.id, name=tc.name, results=results)
```

> ⚠️ **Shell results are structured.** Per-command stdout/stderr/exit codes
> cannot ride prose — answering a shell call with a plain `ToolMessage`
> raises `BadRequestError` at projection.

### Anthropic: text editor + bash

Only the declaration is special. Calls are ordinary `ToolCall`s (wire names
kept), results ordinary `ToolMessage`s:

```python
from luca.client.providers.anthropic import BashTool, TextEditorTool
from luca.client.types import TextBlock, ToolCall, ToolMessage

def handle_bash(tc: ToolCall) -> ToolMessage:
    if tc.arguments.get("restart"):          # {"restart": true} — no command key
        SHELL.restart()
        return ToolMessage(tool_call_id=tc.id, name=tc.name,
                           content=[TextBlock(text="shell session restarted")])
    r = SHELL.run(tc.arguments["command"])   # caller owns the persistent session
    return ToolMessage(tool_call_id=tc.id, name=tc.name,
                       content=[TextBlock(text=r.stdout + r.stderr)],
                       is_error=r.exit_code != 0)
```

The text editor's `arguments` carry the `text_editor_20250728` command set —
each command has `path` plus its own fields:

| `command` | Fields | What the caller returns |
|---|---|---|
| `view` | `view_range?: [start, end]` (`-1` = last line) | file contents `cat -n` style, or the directory listing |
| `create` | `file_text` | a confirmation |
| `str_replace` | `old_str`, `new_str` | a confirmation; `is_error` when `old_str` does not match exactly once |
| `insert` | `insert_line` (0 = top of file), `insert_text` | a confirmation |

> ⚠️ `insert` carries **`insert_text`**, not `new_str` — only `str_replace`
> uses `new_str`. Verified live against claude-sonnet-4-5.

Runnable end to end (all four commands, plus bash):
[`anthropic_example_non_streaming.py`](../../specs/0009-provider-native-tools/examples/anthropic_example_non_streaming.py).

### Executing them: `luca.client.native`

The two file-editing tools ship with a working caller-side implementation:
`luca.client.native`, opt-in (nothing else in the SDK imports it) and
standard-library-only. Give it a root directory and the call's `arguments`:

```python
from luca.client.native import NativeToolError, execute_apply_patch, execute_text_editor
from luca.client.types import TextBlock, ToolCall, ToolMessage

WORKSPACE = "./workspace"
HANDLERS = {"apply_patch": execute_apply_patch, "str_replace_based_edit_tool": execute_text_editor}

def handle_edit(tc: ToolCall) -> ToolMessage:
    try:
        output, failed = HANDLERS[tc.name](tc.arguments, root=WORKSPACE), False
    except NativeToolError as exc:
        output, failed = str(exc), True     # the message is written for the model
    return ToolMessage(tool_call_id=tc.id, name=tc.name,
                       content=[TextBlock(text=output)], is_error=failed)
```

| Function | Runs | Returns |
|---|---|---|
| `execute_apply_patch(operation, *, root)` | one `apply_patch` operation | `Created/Updated/Deleted <path>`, plus a second `Moved <path> to <move_to>` line |
| `execute_text_editor(command, *, root)` | one `str_replace_based_edit_tool` command | the `view` render, else `Created <path>` / `Updated <path>` |

Underneath sit the pure transforms — no filesystem, no IO, useful on their own:

| Function | Does |
|---|---|
| `native.apply_diff(text, diff, mode="default")` | the V4A grammar an `apply_patch` `diff` carries; `mode="create"` for the add-file syntax |
| `native.text_editor.view(text, view_range=None)` | `cat -n` numbering; `view_range=[start, end]`, `-1` = last line |
| `native.text_editor.str_replace(text, old_str, new_str)` | replace the one match |
| `native.text_editor.insert(text, insert_line, insert_text)` | insert after that line; `0` = top |

```python
from luca.client.native import apply_diff

apply_diff("alpha\nbeta\n", "@@ alpha\n-beta\n+BETA")   # -> "alpha\nBETA\n"
```

> ⚠️ **`root` is the boundary, and the only one.** A model path — relative or
> absolute — is fully resolved and then must be inside `root`, so a symlink
> pointing out of it is refused. `view` on a directory names symlinks with a
> trailing `@` and never descends into them, for the same reason. `root="/"`
> turns confinement off. Nothing else here is a sandbox: `create` overwrites,
> `delete_file` deletes.

Everything a model can recover from — a missing file, an `old_str` matching
zero or three times, a diff whose context is gone — raises `NativeToolError`,
whose message is the text to send back. It sits deliberately **outside** the
[`ClientError` hierarchy](11-exceptions.md): those mean transport or network
failure, and none of this code touches the network.

### The generic form: `extras`

Every native call and result has a second, equivalent representation that uses
**only the canonical types**. `ToolCall.extras` and `ToolMessage.extras` are
free-form dicts with one reserved key, `custom_type`, naming the native type;
every other key is a field of that class.

```python
from luca.client.types import ToolCall, ToolMessage

# identical to ApplyPatchToolCall(item_id="apc_1", status="completed", …)
ToolCall(
    id="call_1",
    name="apply_patch",
    arguments={"type": "create_file", "path": "hello.txt", "diff": "+hi\n"},
    extras={"custom_type": "apply_patch_call", "item_id": "apc_1", "status": "completed"},
)

# identical to ShellToolMessage(results=[ShellCommandResult(…)])
ToolMessage(
    tool_call_id="call_2",
    content="",
    extras={
        "custom_type": "shell_call_output",
        "results": [{"stdout": "5\n", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}],
    },
)
```

The transports normalize before projecting, so the two forms produce the
**same wire payload** — including the shell-result requirement, the
`max_output_length` echo, and the foreign-drop policy. Converting between them
is one call each way, and lossless:

```python
call.as_generic()      # ApplyPatchToolCall -> base ToolCall + extras
call.as_native()       # base ToolCall + extras -> ApplyPatchToolCall
```

Use the typed classes when you are writing Python against a known tool — they
give you `tc.operation`, `tc.action`, and validation. Use the generic form
when you would rather not import a provider class at all: storing a session,
crossing a process or language boundary, or writing tool-agnostic plumbing.
Nothing else changes: **responses always come back as the typed subclass**,
whichever form you sent.

An unregistered `custom_type` raises `BadRequestError` at projection — the
generic form removes the need to *import* a native class, not the need for one
to exist.

### Compatibility and provider switches

A native tool declared on the wrong transport fails **before any HTTP** with
`BadRequestError` — `BashTool()` on OpenAI, `ApplyPatchTool()` on Anthropic
(or on chat completions: OpenAI native tools exist only on the Responses
wire).

> ⚠️ **Switching providers mid-conversation.** A native call already in
> history (and its result) is silently dropped when the history is replayed
> to another provider — same policy as foreign thinking-block attestations:
> one lost exchange beats a 400 that kills the conversation.

Forcing works through the provider-agnostic `tool_choice={"name": ...}` form:
`{"name": "apply_patch"}` projects to `{"type": "apply_patch"}` on OpenAI;
`{"name": "bash"}` to `{"type": "tool", "name": "bash"}` on Anthropic.

### Streaming semantics

Anthropic native calls stream like any tool call — full
`tool_call_delta` JSON fragments. OpenAI native calls stream **start → end
only**: the in-progress payload arrives as raw text (not JSON), so there are
no public deltas; `tool_call_end.tool_call` always carries the complete
typed call.

### Persistence

Native calls and results survive a JSON round trip: `model_dump()` keeps the
subclass fields, and validating against the base classes
(`AssistantMessage` / `ToolMessage`) restores the typed subclasses —
provided `luca.client` is imported, which registers every first-party native
type. Store `as_generic()` instead and that import stops mattering: the
generic form reloads as a plain `ToolCall` / `ToolMessage` and projects the
same.

### Extending: your own native tool

The same machinery is public: subclass `BaseTool` and return a
`ToolProjector` (subclassing the target transport's projector base) from
`get_projector()`; a typed call is a `ToolCall` subclass with a distinct
`type` literal, which registers itself for parsing and persistence at import.
The four first-party tools in
`luca/client/transports/*/native_tools.py` are the reference
implementations.
