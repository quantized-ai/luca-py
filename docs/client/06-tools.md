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

`stream.tool_calls` and `stream.message.tool_calls` are filter views over the
same instances, so they always reflect the live state. See
[`08-streaming.md`](08-streaming.md) for the full event vocabulary and
[`main.py`](../../main.py) (run with `--streaming`) for a streaming agent loop.

## Provider-native tools

### Quick intro

Native tools are defined by a provider instead of a JSON schema. You still
run them: the client declares them, gives you the model's calls, and sends
your results back on the next request.

The examples below leave execution unimplemented on purpose. Replace the
`NotImplementedError` functions with code that applies patches, owns the
shell session, enforces permissions, and reports failures honestly.

### OpenAI

OpenAI provides `apply_patch` and a local `shell` on the Responses API. Calls
arrive as typed subclasses: `ApplyPatchToolCall` and `ShellToolCall`.
`apply_patch` returns a regular `ToolMessage`; `shell` must return the
structured `ShellToolMessage`.

#### Non-streaming

```python
from luca.client import completion
from luca.client.providers.openai import (
    ApplyPatchTool,
    ApplyPatchToolCall,
    LocalShellTool,
    ShellCommandResult,
    ShellExitOutcome,
    ShellTimeoutOutcome,
    ShellToolCall,
    ShellToolMessage,
)
from luca.client.types import TextBlock, ToolMessage, UserMessage

MODEL = "openai:gpt-5.1"
SYSTEM = "Use apply_patch for file edits and shell for shell commands."
TOOLS = [ApplyPatchTool(), LocalShellTool()]


def apply_v4a(operation: dict) -> tuple[bool, str]:
    raise NotImplementedError


def run_shell(action: dict) -> list[dict]:
    raise NotImplementedError


def handle_apply_patch(tc: ApplyPatchToolCall) -> ToolMessage:
    operation = tc.operation
    # {"type": "create_file"|"update_file"|"delete_file",
    #  "path": str, "diff"?: str}
    ok, detail = apply_v4a(operation)
    return ToolMessage(
        tool_call_id=tc.id,
        name=tc.name,
        content=[TextBlock(text=detail)],
        is_error=not ok,
    )


def handle_shell(tc: ShellToolCall) -> ShellToolMessage:
    action = tc.action
    # {"commands": list[str], "timeout_ms"?: int,
    #  "max_output_length"?: int}
    results = []
    for result in run_shell(action):
        outcome = (
            ShellTimeoutOutcome()
            if result["timed_out"]
            else ShellExitOutcome(exit_code=result["exit_code"])
        )
        results.append(
            ShellCommandResult(
                stdout=result["stdout"],
                stderr=result["stderr"],
                outcome=outcome,
            )
        )
    return ShellToolMessage(
        tool_call_id=tc.id,
        name=tc.name,
        results=results,
    )


def execute(tc: ApplyPatchToolCall | ShellToolCall) -> ToolMessage:
    if isinstance(tc, ApplyPatchToolCall):
        return handle_apply_patch(tc)
    if isinstance(tc, ShellToolCall):
        return handle_shell(tc)
    raise ValueError(f"unknown tool: {tc.name}")


messages = []

while True:
    prompt = input("You: ")
    if prompt.strip().lower() in {"q", "quit"}:
        break

    messages.append(UserMessage(content=[TextBlock(text=prompt)]))

    while True:
        response = completion(
            MODEL,
            messages,
            system_message=SYSTEM,
            tools=TOOLS,
        )
        messages.extend(response.messages)
        answer = response.messages[-1]

        for block in answer.content:
            if isinstance(block, TextBlock):
                print(block.text, end="", flush=True)
        print()

        if answer.finish_reason != "tool_use":
            break

        for tc in answer.tool_calls:
            messages.append(execute(tc))
```

> ⚠️ **Shell results are structured.** A plain `ToolMessage` cannot carry
> the per-command stdout, stderr, and outcome required by OpenAI. Returning
> one for a shell call raises `BadRequestError` before the request is sent.

#### Streaming

This is the same conversation loop using `completion_stream`. The event
dispatch includes every public event so you can see which ones carry display
text, tool arguments, usage, and terminal state.

```python
from luca.client import completion_stream
from luca.client.providers.openai import (
    ApplyPatchTool,
    ApplyPatchToolCall,
    LocalShellTool,
    ShellCommandResult,
    ShellExitOutcome,
    ShellTimeoutOutcome,
    ShellToolCall,
    ShellToolMessage,
)
from luca.client.types import (
    FinishEvent,
    TextBlock,
    ToolMessage,
    UserMessage,
)

MODEL = "openai:gpt-5.1"
SYSTEM = "Use apply_patch for file edits and shell for shell commands."
TOOLS = [ApplyPatchTool(), LocalShellTool()]


def apply_v4a(operation: dict) -> tuple[bool, str]:
    raise NotImplementedError


def run_shell(action: dict) -> list[dict]:
    raise NotImplementedError


def handle_apply_patch(tc: ApplyPatchToolCall) -> ToolMessage:
    operation = tc.operation
    # {"type": "create_file"|"update_file"|"delete_file",
    #  "path": str, "diff"?: str}
    ok, detail = apply_v4a(operation)
    return ToolMessage(
        tool_call_id=tc.id,
        name=tc.name,
        content=[TextBlock(text=detail)],
        is_error=not ok,
    )


def handle_shell(tc: ShellToolCall) -> ShellToolMessage:
    action = tc.action
    # {"commands": list[str], "timeout_ms"?: int,
    #  "max_output_length"?: int}
    results = []
    for result in run_shell(action):
        outcome = (
            ShellTimeoutOutcome()
            if result["timed_out"]
            else ShellExitOutcome(exit_code=result["exit_code"])
        )
        results.append(
            ShellCommandResult(
                stdout=result["stdout"],
                stderr=result["stderr"],
                outcome=outcome,
            )
        )
    return ShellToolMessage(
        tool_call_id=tc.id,
        name=tc.name,
        results=results,
    )


def execute(tc: ApplyPatchToolCall | ShellToolCall) -> ToolMessage:
    if isinstance(tc, ApplyPatchToolCall):
        return handle_apply_patch(tc)
    if isinstance(tc, ShellToolCall):
        return handle_shell(tc)
    raise ValueError(f"unknown tool: {tc.name}")


messages = []

while True:
    prompt = input("You: ")
    if prompt.strip().lower() in {"q", "quit"}:
        break

    messages.append(UserMessage(content=[TextBlock(text=prompt)]))

    while True:
        finish: FinishEvent | None = None

        with completion_stream(
            MODEL,
            messages,
            system_message=SYSTEM,
            tools=TOOLS,
        ) as stream:
            for event in stream:
                match event.type:
                    case "start":
                        pass
                    case "text_start":
                        pass
                    case "text_delta":
                        print(event.delta, end="", flush=True)
                    case "text_end":
                        print()
                    case "thinking_start":
                        pass
                    case "thinking_delta":
                        pass
                    case "thinking_end":
                        pass
                    case "tool_call_start":
                        print(f"\n[tool] {event.name}")
                    case "tool_call_delta":
                        print(event.arguments_delta, end="", flush=True)
                    case "tool_call_end":
                        print(event.tool_call.arguments)
                    case "refusal_start":
                        pass
                    case "refusal_delta":
                        print(event.delta, end="", flush=True)
                    case "refusal_end":
                        print()
                    case "usage":
                        pass
                    case "finish":
                        finish = event
                    case "error":
                        raise event.error

        if finish is None:
            raise RuntimeError("stream ended without a finish event")

        messages.append(finish.message)

        if finish.finish_reason != "tool_use":
            break

        for tc in finish.tool_calls:
            messages.append(execute(tc))
```

OpenAI native calls normally emit `tool_call_start` and `tool_call_end`
without argument deltas between them. The completed typed call is always on
`tool_call_end.tool_call` and the terminal `finish.tool_calls`.

### Anthropic

Anthropic provides a text editor and a persistent bash session. Only their
declarations are provider-specific: calls are regular `ToolCall` objects and
results are regular `ToolMessage` objects.

The editor's `arguments` use these commands:

| `command` | Other fields |
|---|---|
| `view` | `path`, optional `view_range: [start, end]` (`-1` means the last line) |
| `create` | `path`, `file_text` |
| `str_replace` | `path`, `old_str`, `new_str` |
| `insert` | `path`, `insert_line` (`0` means the top), `insert_text` |

#### Non-streaming

```python
from luca.client import completion
from luca.client.providers.anthropic import BashTool, TextEditorTool
from luca.client.types import TextBlock, ToolCall, ToolMessage, UserMessage

MODEL = "anthropic:claude-sonnet-4-5"
SYSTEM = "Use the text editor for file contents and bash for shell commands."
TOOLS = [TextEditorTool(max_characters=10_000), BashTool()]


def run_text_editor(arguments: dict) -> tuple[bool, str]:
    raise NotImplementedError


def run_bash(arguments: dict) -> tuple[bool, str]:
    raise NotImplementedError


def handle_text_editor(tc: ToolCall) -> ToolMessage:
    arguments = tc.arguments
    # {"command": "view"|"create"|"str_replace"|"insert", "path": str,
    #  "view_range"?: list[int], "file_text"?: str, "old_str"?: str,
    #  "new_str"?: str, "insert_line"?: int, "insert_text"?: str}
    ok, detail = run_text_editor(arguments)
    return ToolMessage(
        tool_call_id=tc.id,
        name=tc.name,
        content=[TextBlock(text=detail)],
        is_error=not ok,
    )


def handle_bash(tc: ToolCall) -> ToolMessage:
    arguments = tc.arguments
    # {"command": str} or {"restart": True}
    ok, detail = run_bash(arguments)
    return ToolMessage(
        tool_call_id=tc.id,
        name=tc.name,
        content=[TextBlock(text=detail)],
        is_error=not ok,
    )


HANDLERS = {
    TextEditorTool.name: handle_text_editor,
    BashTool.name: handle_bash,
}


def execute(tc: ToolCall) -> ToolMessage:
    return HANDLERS[tc.name](tc)


messages = []

while True:
    prompt = input("You: ")
    if prompt.strip().lower() in {"q", "quit"}:
        break

    messages.append(UserMessage(content=[TextBlock(text=prompt)]))

    while True:
        response = completion(
            MODEL,
            messages,
            system_message=SYSTEM,
            tools=TOOLS,
            max_tokens=2_048,
        )
        messages.extend(response.messages)
        answer = response.messages[-1]

        for block in answer.content:
            if isinstance(block, TextBlock):
                print(block.text, end="", flush=True)
        print()

        if answer.finish_reason != "tool_use":
            break

        for tc in answer.tool_calls:
            messages.append(execute(tc))
```

The caller owns one persistent bash session and must implement
`{"restart": True}`. For `str_replace`, return an error unless `old_str`
matches exactly once. For `insert`, the text field is `insert_text`, not
`new_str`.

#### Streaming

Anthropic streams native calls exactly like schema-defined tools: start, JSON
argument deltas, then a completed regular `ToolCall`.

```python
from luca.client import completion_stream
from luca.client.providers.anthropic import BashTool, TextEditorTool
from luca.client.types import (
    FinishEvent,
    TextBlock,
    ToolCall,
    ToolMessage,
    UserMessage,
)

MODEL = "anthropic:claude-sonnet-4-5"
SYSTEM = "Use the text editor for file contents and bash for shell commands."
TOOLS = [TextEditorTool(max_characters=10_000), BashTool()]


def run_text_editor(arguments: dict) -> tuple[bool, str]:
    raise NotImplementedError


def run_bash(arguments: dict) -> tuple[bool, str]:
    raise NotImplementedError


def handle_text_editor(tc: ToolCall) -> ToolMessage:
    arguments = tc.arguments
    # {"command": "view"|"create"|"str_replace"|"insert", "path": str,
    #  "view_range"?: list[int], "file_text"?: str, "old_str"?: str,
    #  "new_str"?: str, "insert_line"?: int, "insert_text"?: str}
    ok, detail = run_text_editor(arguments)
    return ToolMessage(
        tool_call_id=tc.id,
        name=tc.name,
        content=[TextBlock(text=detail)],
        is_error=not ok,
    )


def handle_bash(tc: ToolCall) -> ToolMessage:
    arguments = tc.arguments
    # {"command": str} or {"restart": True}
    ok, detail = run_bash(arguments)
    return ToolMessage(
        tool_call_id=tc.id,
        name=tc.name,
        content=[TextBlock(text=detail)],
        is_error=not ok,
    )


HANDLERS = {
    TextEditorTool.name: handle_text_editor,
    BashTool.name: handle_bash,
}


def execute(tc: ToolCall) -> ToolMessage:
    return HANDLERS[tc.name](tc)


messages = []

while True:
    prompt = input("You: ")
    if prompt.strip().lower() in {"q", "quit"}:
        break

    messages.append(UserMessage(content=[TextBlock(text=prompt)]))

    while True:
        finish: FinishEvent | None = None

        with completion_stream(
            MODEL,
            messages,
            system_message=SYSTEM,
            tools=TOOLS,
            max_tokens=2_048,
        ) as stream:
            for event in stream:
                match event.type:
                    case "start":
                        pass
                    case "text_start":
                        pass
                    case "text_delta":
                        print(event.delta, end="", flush=True)
                    case "text_end":
                        print()
                    case "thinking_start":
                        pass
                    case "thinking_delta":
                        pass
                    case "thinking_end":
                        pass
                    case "tool_call_start":
                        print(f"\n[tool] {event.name} ", end="", flush=True)
                    case "tool_call_delta":
                        print(event.arguments_delta, end="", flush=True)
                    case "tool_call_end":
                        print()
                    case "refusal_start":
                        pass
                    case "refusal_delta":
                        print(event.delta, end="", flush=True)
                    case "refusal_end":
                        print()
                    case "usage":
                        pass
                    case "finish":
                        finish = event
                    case "error":
                        raise event.error

        if finish is None:
            raise RuntimeError("stream ended without a finish event")

        messages.append(finish.message)

        if finish.finish_reason != "tool_use":
            break

        for tc in finish.tool_calls:
            messages.append(execute(tc))
```

A native tool declared on the wrong transport raises `BadRequestError` before
any HTTP request. OpenAI native tools require the OpenAI Responses transport;
Anthropic native tools require the Anthropic transport.

Native calls and results survive `model_dump()` and validation. OpenAI's typed
calls and shell results also have lossless `as_generic()` / `as_native()`
forms for storage or tool-agnostic plumbing.

**Next:** [Structured output](07-structured-output.md)
