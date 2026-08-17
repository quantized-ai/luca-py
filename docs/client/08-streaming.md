# Streaming

Stream a completion token by token instead of waiting for the whole
response. Everything on this page builds on
[chat completion](04-chat-completion.md) — same request kwargs, same
`AssistantMessage` at the end. The difference is that you watch it being
built.

## 1. The two functions

Streaming is **two dedicated functions**, not a `stream=True` flag, so each
one has a single unambiguous return type:

| Function | Returns | Iterate with |
|---|---|---|
| `completion_stream(...)` | `ChatCompletionStream` | `for event in s:` |
| `acompletion_stream(...)` | `AsyncChatCompletionStream` | `async for event in s:` |

```python
from luca.client import completion_stream

with completion_stream(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "Tell me a story."}],
) as s:
    for event in s:
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)
```

The async twin returns the stream **synchronously** — there is no `await` on
the call itself, because the HTTP request has not been made yet:

```python
import asyncio
from luca.client import acompletion_stream

async def main():
    async with acompletion_stream(
        model="openai:gpt-4o",
        messages=[{"role": "user", "content": "Tell me a story."}],
    ) as s:
        async for event in s:
            if event.type == "text_delta":
                print(event.delta, end="", flush=True)

asyncio.run(main())
```

Both accept every kwarg `completion()` does — `tools`, `system_message`,
`response_format`, `temperature`, `max_tokens`, `reasoning`, `provider`,
`api_key`, `timeout`, and the rest. See
[chat completion](04-chat-completion.md) for the full list.
`acompletion_stream()` accepts one extra: `total_timeout=` (§11).

> ⚠️ **The request opens lazily, on the first iteration.** Creating the stream
> makes no network call. Always use `with` / `async with` — a stream that is
> garbage-collected while open emits a `ResourceWarning`.

## 2. What a stream actually emits

Every stream produces the same shape: one `start`, then a block-by-block
narration of the message, then usage, then exactly one terminal. Here is a
complete real sequence for a response containing reasoning, text, and a tool
call:

```python
StartEvent          type='start'
ThinkingStartEvent  type='thinking_start'  index=0
ThinkingDeltaEvent  type='thinking_delta'  index=0  delta='Let me think about that.'
ThinkingEndEvent    type='thinking_end'    index=0  content='Let me think about that.'
TextStartEvent      type='text_start'      index=1
TextDeltaEvent      type='text_delta'      index=1  delta='Checking the weather.'
TextEndEvent        type='text_end'        index=1  content='Checking the weather.'
ToolCallStartEvent  type='tool_call_start' index=2  id='call_1' name='get_weather'
ToolCallDeltaEvent  type='tool_call_delta' index=2
ToolCallEndEvent    type='tool_call_end'   index=2
UsageEvent          type='usage'
FinishEvent         type='finish'          finish_reason='tool_use'
```

Three rules hold for every stream, on every provider:

- **`start` is always first**, before any block.
- **`index` is dense and ordered.** It's the position in
  `message.content`, so `index=0` is the first block, `index=1` the second.
  Blocks open and close in order; a `*_delta` always falls between its
  block's `*_start` and `*_end`.
- **Exactly one terminal**, either `finish` or `error`, and nothing after it.

## 3. Reading events

Each event is a Pydantic model with a `type` string. Three equivalent styles:

```python
# by discriminator
for event in s:
    if event.type == "text_delta":
        print(event.delta, end="")

# by class
from luca.client.types import TextDeltaEvent, FinishEvent

for event in s:
    if isinstance(event, TextDeltaEvent):
        print(event.delta, end="")
    elif isinstance(event, FinishEvent):
        print(event.finish_reason)

# structural match
for event in s:
    match event.type:
        case "text_delta" | "thinking_delta":
            print(event.delta, end="")
        case "tool_call_end":
            dispatch(event.tool_call)
        case "finish":
            print(event.finish_reason)
```

Every event class imports from `luca.client.types`:

```python
from luca.client.types import (
    StartEvent,
    TextStartEvent, TextDeltaEvent, TextEndEvent,
    ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent,
    ToolCallStartEvent, ToolCallDeltaEvent, ToolCallEndEvent,
    RefusalStartEvent, RefusalDeltaEvent, RefusalEndEvent,
    UsageEvent, FinishEvent, ErrorEvent,
    StreamEvent,          # the discriminated union of all of the above
)
```

## 4. The message as it grows

Events narrate; the **stream object holds the state**. `s.message` is the
`AssistantMessage` being built — one live object, updated in place as chunks
arrive:

```python
with completion_stream(...) as s:
    for event in s:
        if event.type == "text_delta":
            print(len(s.message.content), "blocks so far")
```

Because it is live, it keeps mutating under you. To store what you saw at a
given event, copy it yourself:

```python
if event.type == "text_delta":
    frozen = s.message.model_copy(deep=True)
```

The terminal `FinishEvent` carries `message`, a deep snapshot — safe to keep
as-is. After the terminal, `s.message` holds the final state
(`finish_reason`, `usage`, and `error_message` included).

## 5. Every event type

```python
class TextDeltaEvent(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    index: int
    delta: str
```

| Class | `type` | Fields (besides `type`) | Emitted when |
|---|---|---|---|
| `StartEvent` | `start` | — | Once, before any block. |
| `TextStartEvent` | `text_start` | `index` | A text block opens. |
| `TextDeltaEvent` | `text_delta` | `index`, `delta` | A text chunk arrives. |
| `TextEndEvent` | `text_end` | `index`, `content` | The text block closes; `content` is the whole block. |
| `ThinkingStartEvent` | `thinking_start` | `index` | A reasoning block opens. |
| `ThinkingDeltaEvent` | `thinking_delta` | `index`, `delta` | A reasoning chunk arrives. |
| `ThinkingEndEvent` | `thinking_end` | `index`, `content` | The reasoning block closes. |
| `ToolCallStartEvent` | `tool_call_start` | `index`, `id`, `name` | A tool call opens; the name is known, arguments are not. |
| `ToolCallDeltaEvent` | `tool_call_delta` | `index`, `arguments_delta` | A fragment of the arguments JSON. |
| `ToolCallEndEvent` | `tool_call_end` | `index`, `tool_call` | Arguments parsed; `tool_call` is ready to dispatch. |
| `RefusalStartEvent` | `refusal_start` | `index` | A refusal block opens (OpenAI strict mode). |
| `RefusalDeltaEvent` | `refusal_delta` | `index`, `delta` | A refusal chunk. |
| `RefusalEndEvent` | `refusal_end` | `index`, `content` | The refusal block closes. |
| `UsageEvent` | `usage` | `usage` | Token counts arrive, near the end. |
| `FinishEvent` | `finish` | `message`, `finish_reason`, `provider_finish_reason`, `cancelled`, `usage`, `tool_calls` | **Terminal.** The model produced a turn. |
| `ErrorEvent` | `error` | `error`, `usage` | **Terminal.** The stream broke after opening. |

A model that produces no reasoning emits no `thinking_*` events. A model that
calls no tools emits no `tool_call_*` events. Only `start` and a terminal are
guaranteed.

## 6. The two terminals

```python
with completion_stream(...) as s:
    for event in s:
        if event.type == "finish":
            print("model finished:", event.finish_reason)
        elif event.type == "error":
            raise event.error
```

The split is on **source of failure**, not on outcome:

- **`FinishEvent`** — the *model* produced a terminal. This is the terminal
  whenever the wire closed normally, **including** `finish_reason="error"`
  (a safety filter or content filter outcome) and cancellation
  (`cancelled=True`).
- **`ErrorEvent`** — the *stream* broke. A mid-stream HTTP failure, malformed
  JSON, a timeout. Carries `error: ClientError`; whatever had streamed before
  the break stays readable on `s.message`.

```python
# A safety refusal is NOT an ErrorEvent:
FinishEvent(finish_reason="error", ...)   # message.error_message explains it
```

`FinishEvent` also carries `provider_finish_reason`, the raw upstream string
(`"tool_use"`, `"max_tokens"`, `"incomplete:content_filter"`, …) next to the
SDK-canonical `finish_reason`.

> ⚠️ **A rejected request raises instead of emitting.** If the provider
> refuses before the stream opens — bad key, no credit, rate limit — the first
> iteration raises the mapped `ClientError`. There is no stream to terminate,
> so no `ErrorEvent`. See [exceptions](11-exceptions.md#streaming).

## 7. Live accessors on the stream

The stream object exposes the state it is building, at any point:

```python
with completion_stream(...) as s:
    for event in s:
        s.message                 # AssistantMessage being built
        s.text                    # all text-block content concatenated so far
        s.tool_calls              # the ToolCall blocks (same instances, not copies)
        s.usage                   # Usage | None — set when `usage` arrives
        s.finish_reason           # SDK-canonical; set at the terminal
        s.provider_finish_reason  # raw upstream string
        s.cancelled               # bool
```

Handy for progress reporting without accumulating your own buffer:

```python
for event in s:
    if event.type == "text_delta":
        print(f"\r{len(s.text)} chars", end="")
```

## 8. `collect()` — skip the loop

When you want streaming's latency but not its events, `collect()` drains the
stream and returns the same `ChatCompletionResponse` that `completion()`
would have:

```python
with completion_stream(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "Say hello."}],
) as s:
    response = s.collect()

print(response.messages[-1].content[0].text)
print(response.messages[-1].finish_reason)
```

A stream always builds exactly one message, so `messages` has one element —
the same deep copy the terminal `FinishEvent` carries.

On an `ErrorEvent`, `collect()` re-raises the underlying `ClientError`.
`collect()` consumes the stream; iterating afterwards raises `StreamError`.
The async form is `await s.collect()`.

## 9. Structured output

Pass `response_format=` as usual, then parse at the terminal:

```python
from pydantic import BaseModel
from luca.client import completion_stream

class Weather(BaseModel):
    city: str
    celsius: int

with completion_stream(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "Weather in Madrid?"}],
    response_format=Weather,
) as s:
    for event in s:
        if event.type == "text_delta":
            print(event.delta, end="")     # watch the JSON arrive
        elif event.type == "finish":
            weather = event.parse()        # -> Weather(city='Madrid', celsius=21)
```

`FinishEvent.parse()` validates the concatenated text against the
`response_format` from the originating request. Calling it on a stream that
had no `response_format` raises `ValueError`. See
[structured output](07-structured-output.md).

## 10. Cancellation

```python
with completion_stream(...) as s:
    for event in s:
        if event.type == "text_delta" and "STOP" in event.delta:
            s.cancel()
```

`cancel()` closes the underlying HTTP response. The next read fails, and the
stream converts that into a terminal `FinishEvent` with:

- `cancelled=True`
- `finish_reason=None` (no terminal arrived from the provider)
- the partial message and whatever usage had already been reported

Cancellation is **not** an error — you still get a `FinishEvent`, never an
`ErrorEvent`. The async form is `await s.cancel()`.

> ⚠️ **Cancelling a stream that already finished does nothing.** If the
> provider's terminal had already arrived, you get a normal
> `FinishEvent(cancelled=False)`. `cancelled=True` requires an in-flight read
> to interrupt.

## 11. Timeouts

Two independent knobs:

```python
# per-request HTTP timeout — same as completion(); applies to the connection
with completion_stream(model="openai:gpt-4o", messages=[...], timeout=30.0) as s:
    ...

# wall-clock deadline over the WHOLE stream — async only
async with acompletion_stream(
    model="openai:gpt-4o",
    messages=[...],
    total_timeout=60.0,
) as s:
    async for event in s:
        ...
```

`total_timeout=` arms a deadline when the stream opens and enforces it on
every chunk read. On expiry you get exactly one terminal `ErrorEvent`
carrying the SDK `TimeoutError`, then the stream closes:

```python
['start', 'text_start', 'text_delta', 'text_end', 'error']
#                                                   ^ ErrorEvent(error=TimeoutError(...))
```

It is async-only: the sync path has no event loop to enforce a deadline on.
`timeout=` works on both.

## 12. Tool calls while streaming

The three tool events map onto the three things you learn, in order — that a
call started, what its arguments look like so far, and the finished call:

```python
from luca.client import completion_stream

with completion_stream(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "Weather in Madrid?"}],
    tools=[get_weather_tool],
) as s:
    for event in s:
        if event.type == "tool_call_start":
            print(f"calling {event.name}...")
        elif event.type == "tool_call_end":
            result = dispatch(event.tool_call)   # arguments parsed, ready to run
    calls = s.tool_calls
```

`ToolCallDeltaEvent.arguments_delta` is a **fragment of JSON**, not valid
JSON on its own. It accumulates on `ToolCall.partial_arguments`; the parse
happens once at `tool_call_end`, which is the first point `tool_call.arguments`
is populated and `tool_call.complete` is `True`. Malformed JSON from the
provider surfaces as an `ErrorEvent`, not a silent empty dict.

`FinishEvent.tool_calls` contains complete calls only.

> ⚠️ **OpenAI native tools stream start → end only.** An
> [OpenAI native call](06-tools.md#provider-native-tools) (`apply_patch`,
> `shell`) emits no `tool_call_delta` — its in-progress payload is raw text,
> not JSON — and `tool_call_end.tool_call` carries the complete typed call.
> Anthropic native tools stream full JSON deltas like any other tool.

## 13. Lower-level entry points

`completion_stream()` resolves a provider, builds a `ChatCompletionRequest`,
and hands it down. You can enter at either lower layer with the same request
object — see [providers and transports](09-providers-and-transports.md).

**Provider level** — you own the provider lifecycle, the SDK still picks the
transport:

```python
from luca.client.providers import OpenAIProvider
from luca.client.types import ChatCompletionRequest, UserMessage

with OpenAIProvider(api_key="sk-…", timeout=30.0) as prov:
    request = ChatCompletionRequest(
        model="gpt-4o",
        messages=[UserMessage(content="Tell me a story.")],
    )
    with prov.completion_stream(request) as s:
        for event in s:
            ...
```

**Transport level** — you pick the wire protocol yourself:

```python
import httpx
from luca.client.transports import OpenAIResponsesTransport
from luca.client.types import ChatCompletionRequest, UserMessage

transport = OpenAIResponsesTransport(
    provider="openai",
    base_url="https://api.openai.com/v1",
    api_key="sk-…",
    timeout=30.0,
    http_client=httpx.Client(),
)

request = ChatCompletionRequest(
    model="gpt-4o",
    messages=[UserMessage(content="Tell me a story.")],
)
with transport.completion_stream(request) as s:
    for event in s:
        ...
```

Both layers expose `completion_stream(request)` and
`acompletion_stream(request)`, returning the same stream classes. Note that
`total_timeout=` is a helper-level convenience; at these layers, arm it with
`stream._set_total_timeout(seconds)` before the first iteration.

The stream classes themselves are importable for type annotations:

```python
from luca.client.types import ChatCompletionStream, AsyncChatCompletionStream

def render(s: ChatCompletionStream) -> str:
    with s:
        for event in s:
            ...
        return s.text
```

## 14. Testing streams

The [faux provider](12-testing.md) scripts an assistant message and streams it
back through the real stream machinery — same events, same ordering, no
network:

```python
from luca.client import completion_stream
from luca.client.testing import (
    FauxProvider, faux_assistant_message, faux_text, faux_tool_call,
)

prov = FauxProvider()
prov._transport.set_responses([
    faux_assistant_message(
        blocks=[
            faux_text("Checking the weather."),
            faux_tool_call("get_weather", {"city": "Madrid"}, id="call_1"),
        ],
        finish_reason="tool_use",
    )
])

with completion_stream(
    model="faux:test",
    messages=[{"role": "user", "content": "Weather in Madrid?"}],
    provider=prov,
) as s:
    events = list(s)

assert [e.type for e in events] == [
    "start",
    "text_start", "text_delta", "text_end",
    "tool_call_start", "tool_call_delta", "tool_call_end",
    "finish",
]
```

`tokens_per_second=` on `FauxProvider` paces the deltas if you need to test
timing. `faux_error(...)` scripts a mid-stream break into an `ErrorEvent`.

## 15. Internal: `RawStreamEvent`

Only relevant if you are **writing a transport**. Transports don't build
public events — they emit a small dataclass vocabulary that
`ChatCompletionStream` translates:

```python
from luca.client.types import (
    RawBlockStart, RawBlockStop,
    RawTextDelta, RawThinkingDelta, RawToolArgumentsDelta, RawRefusalDelta,
    RawFinish, RawUsage,
)

def parse_chunks(self):
    yield RawBlockStart(index=0, block_type="text")
    yield RawTextDelta(index=0, text="Hello")
    yield RawBlockStop(index=0)
    yield RawUsage(usage=Usage(input_tokens=10, output_tokens=2, total_tokens=12))
    yield RawFinish(reason="stop")
```

The stream is the single mutator of the message and the single emitter of
public events, which is what makes the rules in §2 and §4 hold identically on
every provider. Transports supply the wire format and nothing else.

Two Responses-API details worth knowing when reading a transcript: a
reasoning item's summary parts are joined into **one** thinking block
separated by a blank line, and its `encrypted_content` only arrives at
`response.output_item.done`, so the block's signature is set at the end rather
than as it streams.

**Next:** [Providers and transports](09-providers-and-transports.md)
