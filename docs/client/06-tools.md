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

## Provider-defined tools

Some providers define tools themselves and train the model on the schema.
Anthropic ships a text editor and a bash tool; OpenAI's Responses API ships
`apply_patch` and `shell`. You do not describe these — you name the type, and
the provider supplies the schema.

```python
from luca.client.types import Tool

editor = Tool(
    name="str_replace_based_edit_tool",
    description="Never sent; the provider owns the schema.",
    parameters=EditorArgs,            # never sent either — see below
    provider_type="text_editor_20250728",
)
```

With `provider_type` set, the transport sends the provider's own form and
drops `description` and `parameters` entirely:

```json
{"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}
```

`parameters` stays useful locally. It is what you validate the incoming
`ToolCall.arguments` against, and it is still a complete description of the
tool for anything that inspects it later. It is simply not the advertisement.

**You still execute the call.** These are client tools: the model asks, your
code runs it, you send the result back. Nothing about the response path
changes for Anthropic — the call arrives as an ordinary `ToolCall`.

### What each transport serves

| Transport | Types | Wire form |
|---|---|---|
| `AnthropicTransport` | `text_editor_20250728`, `text_editor_20250124`, `bash_20250124` | `{"type": …, "name": …}` |
| `OpenAIResponsesTransport` | `apply_patch`, `shell` | `{"type": "apply_patch"}`, `{"type": "shell", "environment": {"type": "local"}}` |
| everything else | none | refused |

> ⚠️ **Refusing is the default.** A transport that does not know a
> `provider_type` raises `UnsupportedParameterError` before the request. Chat
> completions has no provider-defined tools at all, and Bedrock serves Claude
> over Converse, which has no equivalent — so both refuse. Shipping the tool as
> an ordinary function instead would advertise your local schema as the
> contract, under a description saying it is never sent.

Anthropic pairs the type with the NAME: `text_editor_20250728` goes with
`str_replace_based_edit_tool`, `text_editor_20250124` with the older
`str_replace_editor`. Changing one without the other is a 400.

### OpenAI's two are a different shape

`apply_patch` and `shell` do not arrive as function calls. The model emits
`apply_patch_call` / `shell_call` items and expects
`apply_patch_call_output` / `shell_call_output` back. The transport hides
this: it parses them into ordinary `ToolCall`s on the way in and rebuilds the
matching item on the way out, streaming and buffered alike.

The item carries no tool name, so an incoming call is resolved from whichever
tool was offered with that provider type on the same request. Replaying a
recorded one reads `ToolCall.provider_type` instead:

```python
for tc in response.tool_calls:
    tc.provider_type    # "apply_patch" | "shell" | None for a function call
```

> ⚠️ **Do not infer the item type from the name.** A name is not a durable
> identifier — you may well ship your own `apply_patch`. A call recorded
> before you enabled provider tools has `provider_type=None` and must replay
> as a `function_call`; sending it back as an `apply_patch_call` produces an
> operation with no `type`, `path` or `diff`, and the API rejects the whole
> input array. Keep the field with the call.

A native call whose type is not among the tools on the current request
replays as an ordinary `function_call`, so turning provider tools off mid
conversation does not poison the history.

## Streaming tool calls

In a streaming response, tool calls arrive in pieces. The accumulator
mutates a single `ToolCall` instance per call:

- `tool_call_start` — block created, `complete=False`, `arguments={}`.
- `tool_call_delta` — `partial_arguments` accumulates raw JSON fragments.
- `tool_call_end` — `arguments` parsed from `partial_arguments`,
  `complete=True`, `partial_arguments=""`.

A provider-defined call streams through the same three events. OpenAI's
`apply_patch_call` / `shell_call` carry their arguments whole on the item's
done frame rather than in fragments, so you get one `tool_call_delta` with the
complete JSON instead of several.

`stream.tool_calls` and `partial.tool_calls` are filter views over the same
instances, so they always reflect the live state. See
[`08-streaming.md`](08-streaming.md) for the full event vocabulary and
[`main.py`](../../main.py) (run with `--streaming`) for a streaming agent loop.
