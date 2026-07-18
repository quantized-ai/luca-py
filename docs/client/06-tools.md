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
| `{"type": "function", "function": {"name": "..."}}` (or similar provider shape) | Force a specific tool. |

The dict form is passed through verbatim per provider's shape.

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
    messages.append(response.message)

    if response.finish_reason != "tool_use":
        break

    for tc in response.tool_calls:
        result = execute(tc)
        messages.append(ToolMessage(
            tool_call_id=tc.id,
            content=[TextBlock(text=result)],
        ))
```

Two things to notice:

- `response.message` is appended directly. It's already an
  `AssistantMessage` with `tool_calls` inside `content` — no manual
  reconstruction.
- `response.tool_calls` is a **filter** of `response.message.content` (same
  instances). Mutating a `ToolCall` from either view mutates both.

## Parsing arguments against a schema

If you want validated, typed arguments rather than the raw dict:

```python
for tc in response.tool_calls:
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
