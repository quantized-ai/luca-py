# Quickstart

Every example below assumes you've set the relevant API key in the
environment (see [`01-installation.md`](01-installation.md)).

## Sync, non-streaming

```python
from luca.client import completion

response = completion(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "Say hello in three languages."}],
)

for block in response.message.content:
    if block.type == "text":
        print(block.text)

print(f"finish_reason={response.finish_reason}")
print(f"used {response.usage.total_tokens} tokens")
```

`response` is a `ChatCompletionResponse`. Everything you commonly read
(`finish_reason`, `usage`, `provider`, `tool_calls`, …) forwards transparently
to `response.message`, so `response.finish_reason` and
`response.message.finish_reason` are the same field — no drift.

## Async

```python
import asyncio
from luca.client import acompletion

async def main():
    response = await acompletion(
        model="anthropic:claude-3-5-sonnet-latest",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.message.content[0].text)

asyncio.run(main())
```

## Streaming

`completion_stream()` (sync) and `acompletion_stream()` (async) are
**separate functions** — not a `stream=True` flag — so each helper has one
unambiguous return type.

```python
from luca.client import completion_stream

with completion_stream(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "Tell me a story."}],
) as s:
    for event in s:
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)
        elif event.type == "finish":
            print(f"\n[finish reason={event.finish_reason}]")
```

`acompletion_stream()` returns the stream **synchronously** (HTTP fires on the
first `__aiter__`), so the idiom is `async with acompletion_stream(...) as s:`
— **no `await` on creation**.

```python
import asyncio
from luca.client import acompletion_stream

async def main():
    async with acompletion_stream(
        model="anthropic:claude-3-5-sonnet-latest",
        messages=[{"role": "user", "content": "Tell me a story."}],
    ) as s:
        async for event in s:
            if event.type == "text_delta":
                print(event.delta, end="", flush=True)

asyncio.run(main())
```

## System prompts are request-scoped

There is **no** `SystemMessage` class and **no** `"system"` role in `messages`.
The system prompt rides on `system_message=`:

```python
completion(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    system_message="You are a concise, technical assistant.",
)
```

Each transport projects `system_message` into the host's expected wire shape
— for OpenAI-compatible hosts that means a wire-level `{role: "system",
content: ...}` entry; for Anthropic, it populates the top-level `system`
field.

If you pass `{"role": "system", …}` inside `messages`, the helper raises
`BadRequestError` with a hint to move it to `system_message=`.

## Tools (one-shot)

```python
from pydantic import BaseModel, Field
from luca.client import completion
from luca.client.types import Tool

class BinaryOp(BaseModel):
    a: float = Field(description="First operand.")
    b: float = Field(description="Second operand.")

tools = [
    Tool(name="add", description="Add two numbers.", parameters=BinaryOp),
]

response = completion(
    model="anthropic:claude-3-5-sonnet-latest",
    messages=[{"role": "user", "content": "What is 21 + 21?"}],
    tools=tools,
)

if response.finish_reason == "tool_use":
    for tc in response.tool_calls:
        print(f"{tc.name}({tc.arguments})")
```

For the full agent loop pattern see [`06-tools.md`](06-tools.md) and the
runnable [`main.py`](../../main.py) at the repo root.
