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

message = response.messages[-1]
for block in message.content:
    if block.type == "text":
        print(block.text)

print(f"finish_reason={message.finish_reason}")
print(f"used {message.usage.total_tokens} tokens")
```

`response` is a `ChatCompletionResponse`. It holds `messages`, a non-empty
list of `AssistantMessage` in wire order — today always one element, because
every transport parses a response into a single message. Everything you
commonly read (`finish_reason`, `usage`, `provider`, `tool_calls`, …) lives on
the messages themselves; the terminal state is on the last one. The response
forwards nothing, so `response.finish_reason` raises `AttributeError`.

## Async

```python
import asyncio
from luca.client import acompletion

async def main():
    response = await acompletion(
        model="anthropic:claude-3-5-sonnet-latest",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.messages[-1].content[0].text)

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

answer = response.messages[-1]
if answer.finish_reason == "tool_use":
    for tc in answer.tool_calls:
        print(f"{tc.name}({tc.arguments})")
```

For the full agent loop pattern see [`06-tools.md`](06-tools.md) and the
runnable [`main.py`](../../main.py) at the repo root.

## Complete chatbot loop

Replace the two tool stubs with real implementations:

```python
import asyncio
from collections.abc import Callable

from pydantic import BaseModel

from luca.client import acompletion
from luca.client.types import Tool, ToolMessage, UserMessage


class SearchWebArgs(BaseModel):
    query: str


class GetWeatherArgs(BaseModel):
    city: str


def search_web(query: str) -> str: ...


def get_weather(city: str) -> str: ...


TOOLS = [
    Tool(name="search_web", description="Search the web.", parameters=SearchWebArgs),
    Tool(name="get_weather", description="Get the weather in a city.", parameters=GetWeatherArgs),
]
HANDLERS: dict[str, Callable[..., str]] = {
    "search_web": search_web,
    "get_weather": get_weather,
}


async def main():
    messages = []

    while True:
        prompt = input("You: ")
        if prompt.lower() in {"quit", "exit"}:
            break
        messages.append(UserMessage(content=prompt))

        while True:
            response = await acompletion(
                model="openai:gpt-4o",
                messages=messages,
                tools=TOOLS,
            )
            messages.extend(response.messages)
            answer = response.messages[-1]

            if answer.finish_reason != "tool_use":
                for block in answer.content:
                    if block.type == "text":
                        print(f"Assistant: {block.text}")
                break

            for call in answer.tool_calls:
                result = HANDLERS[call.name](**call.arguments)
                messages.append(ToolMessage(tool_call_id=call.id, content=result))


asyncio.run(main())
```
