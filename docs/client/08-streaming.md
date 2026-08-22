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
| `completion_stream(...)` | a sync stream object | `for event in s:` |
| `acompletion_stream(...)` | an async stream object | `async for event in s:` |

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
the call itself, because the HTTP request only fires at `async with`:

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

> ⚠️ **The request opens at `with` / `async with`.** Creating the stream makes
> no network call; entering the context manager sends the request and reads
> the response headers, so a rejected request raises there (§6). Iterating a
> stream that was never entered raises `StreamError`, and a stream that is
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
| `TextAnnotationEvent` | `text_annotation` | `index`, `annotation` | A citation landed on the text block at `index`; complete on arrival. |
| `WebStartEvent` | `web_start` | `id` | A hosted web operation opened. |
| `WebSearchEvent` | `web_search` | `id`, `queries` | The operation is a search; the queries are known. |
| `WebSearchResultEvent` | `web_search_result` | `id`, `results` | The search's results, batched as the provider sent them. |
| `WebFetchEvent` | `web_fetch` | `id`, `urls` | The operation opens these URLs. |
| `WebFindEvent` | `web_find` | `id`, `url`, `pattern` | The operation searches a page for a pattern (OpenAI). |
| `WebEndEvent` | `web_end` | `id` | The operation finished; its blocks are on the final message. |
| `UsageEvent` | `usage` | `usage` | Token counts arrive, near the end. |
| `FinishEvent` | `finish` | `message`, `finish_reason`, `provider_finish_reason`, `cancelled`, `usage`, `tool_calls` | **Terminal.** The model produced a turn. |
| `ErrorEvent` | `error` | `error`, `usage` | **Terminal.** The stream broke after opening. |

A model that produces no reasoning emits no `thinking_*` events. A model that
calls no tools emits no `tool_call_*` events. Only `start` and a terminal are
guaranteed.

### Web operations

The `web_*` events narrate [hosted web tools](06-tools.md#hosted-web-tools)
as direct, user-facing facts — one `id` links every event of the same
operation, and there is no nested provider action to inspect:

```python
match event:
    case WebStartEvent():
        show_status("Using the web…")
    case WebSearchEvent(queries=queries):
        show_status(f"Searching: {', '.join(queries)}")
    case WebSearchResultEvent(results=results):
        show_found_results(results)
    case WebFetchEvent(urls=urls):
        show_status(f"Opening: {', '.join(urls)}")
    case WebFindEvent(url=url, pattern=pattern):
        show_status(f"Searching {url} for {pattern!r}")
    case WebEndEvent():
        clear_status()
```

No result event is emitted when the provider sends no results. The completed
canonical blocks ride the final message; `web_end` does not duplicate them.

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
SDK-canonical `finish_reason`. A long-running hosted tool pausing the
response (Anthropic's `pause_turn`) finishes the stream normally with
canonical `finish_reason="pause"` — re-send the recorded assistant content
as-is to continue.

> ⚠️ **A rejected request raises instead of emitting.** If the provider
> refuses the request — bad key, no credit, rate limit — the `with` /
> `async with` line raises the mapped `ClientError`. There is no stream to
> terminate, so no `ErrorEvent`. See [exceptions](11-exceptions.md#streaming).

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

`cancel()` interrupts the stream — it closes the underlying HTTP response
(unblocking a read in flight), and after any events already buffered from
the last chunk drain, the stream terminates with a `FinishEvent` carrying:

- `cancelled=True`
- `finish_reason=None` (no terminal arrived from the provider)
- the partial message and whatever usage had already been reported

Cancellation is **not** an error — you still get a `FinishEvent`, never an
`ErrorEvent`. The async form is `await s.cancel()`. The one exception is a
cancel that lands while `async with` is still opening the request: there is
no stream to terminate yet, so the open raises `StreamError` instead.

> ⚠️ **Cancelling a stream that already finished does nothing.** If the
> provider's terminal had already arrived, you get a normal
> `FinishEvent(cancelled=False)`. `cancelled=True` requires an in-flight read
> to interrupt.

## 11. Timeouts

`timeout=` is one knob: a **total wall-clock deadline for the whole call**,
on all four helpers, sync and async. `None` (the default) means no deadline.

```python
async with acompletion_stream(
    model="openai:gpt-4o",
    messages=[...],
    timeout=60.0,
) as s:
    async for event in s:
        ...
```

Expiry after the stream opened emits exactly one terminal `ErrorEvent`
carrying the SDK `TimeoutError`, then the stream closes; expiry during the
open raises it at the `with` line:

```python
['start', 'text_start', 'text_delta', 'text_end', 'error']
#                                                   ^ ErrorEvent(error=TimeoutError(...))
```

Async enforcement can interrupt a read in flight (a timer cancels the
streamer's own read task). Sync enforcement is **cooperative and
best-effort**: the clock is checked before each read, so a read already
blocked on a live socket runs to completion before expiry is detected — a
platform limit, not a bug. The connect phase is always bounded (a black-holed
host fails in seconds), and a shared client's default read timeout never
applies to a stream, so a slow model cannot be killed mid-answer by client
configuration.

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

Both layers expose `completion_stream(request, timeout=None)` and
`acompletion_stream(request, timeout=None)`, returning the same streamer
objects the helpers do — `timeout=` travels per call at every layer.

The streamer base class is importable for type annotations:

```python
from luca.client.transports.streamer import BaseStreamer

def render(s: BaseStreamer) -> str:
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
prov.set_responses([
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

`faux_error(...)` scripts a mid-stream break into an `ErrorEvent`, and
`faux_hang()` parks an async stream until it is cancelled or times out.

## 15. Internal: the streamer classes

Only relevant if you are **writing a transport**. Each wire protocol is a
STREAMER CLASS that owns its wire end-to-end — request building, chunk
parsing, event creation, error mapping — as methods, overridable by
subclassing. Sync/async iteration and the deadline machinery live in two
mixins written once; a concrete streamer is an empty mixin × wire
combination:

```python
from luca.client.transports.streamer import (
    AsyncStreamerMixin, BaseStreamer, SyncStreamerMixin,
)
from luca.client.transports.anthropic.streamer import (
    AnthropicStreamer,        # the wire class: handlers, no iteration
    SyncAnthropicStreamer,    # SyncStreamerMixin  x AnthropicStreamer
    AsyncAnthropicStreamer,   # AsyncStreamerMixin x AnthropicStreamer
)
```

A wire class fills `HANDLERS_BY_TYPE` — wire event type → handler method
name — and each handler turns one raw wire event into a list of public
events while mutating `self.message`:

```python
class AnthropicStreamer(AnthropicWireMixin, BaseStreamer):
    PROVIDER = "anthropic"
    HANDLERS_BY_TYPE = {
        "content_block_delta": "handle_content_block_delta",
        # ...
    }

    def handle_content_block_delta(self, raw_event: dict) -> list:
        ...
```

The pieces a new wire implements or overrides:

| Method | Job | Override when |
|---|---|---|
| `parse(chunk) -> list` | One wire chunk → raw events. Default: SSE `data:` lines. | The framing differs (Bedrock buffers binary frames; CC maps `[DONE]` to a marker). |
| `handle(raw) -> list` | Dispatch via `HANDLERS_BY_TYPE`. | The wire has no event types (CC routes on chunk shape). |
| `handle_<event>(raw)` | One handler per wire event type. | Always — this IS the wire knowledge. |
| `handle_wire_end() -> list` | The wire closed; `[]` means premature end → `ErrorEvent`. | Usage arrives after the finish marker (CC, Bedrock) so the `FinishEvent` can only be built here. |
| `iter_chunks` / `aiter_chunks` | The chunk source over the response. | The wire is not line-based (Bedrock: bytes). |
| `open_wire` / `aopen_wire` | Send the request, return the chunk source. | There is no HTTP at all (the faux installs a scripted source). |

Everything the wire shares with the non-streaming path — payload projection,
the HTTP error mapper, `_classify_finish` — is inherited from the same
`<X>WireMixin` the transport uses, never duplicated. The transport itself is
a pure factory: two class attributes name the combos, and
`completion_stream()` forwards data only.

```python
class AnthropicTransport(BaseTransport, ChatCompletionTransportMixin, AnthropicWireMixin):
    STREAMER = SyncAnthropicStreamer
    ASYNC_STREAMER = AsyncAnthropicStreamer
```

The streamer is the single mutator of its message and the single emitter of
its events, which is what makes the rules in §2 and §4 hold identically on
every provider.

Two Responses-API details worth knowing when reading a transcript: a
reasoning item's summary parts are joined into **one** thinking block
separated by a blank line, and its `encrypted_content` only arrives at
`response.output_item.done`, so the block's signature is set at the end rather
than as it streams.

**Next:** [Providers and transports](09-providers-and-transports.md)
