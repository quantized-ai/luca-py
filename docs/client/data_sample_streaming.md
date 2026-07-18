# Data sample — `while True` tool loop with **streaming**, event by event

A streaming version of the same `"What's the weather like in Madrid?"` walk-through from `data_sample.md`. Shows the per-event timeline of two `completion_stream()` calls (one per turn), how `event.partial` evolves as fragments arrive, and the terminal `FinishEvent` at the end of each turn. Uses the actual event types from `api_prd.md` §12.6 — unified `ToolCall` blocks carry their streaming state (`partial_arguments`, `complete`) inline, the terminal carries both the SDK-canonical `finish_reason` and the raw upstream string on `provider_finish_reason`, and the system prompt rides on the **request-scoped `system_message=` kwarg** rather than as a message in `messages`.

## The loop

```python
import sys
from py import completion_stream
from py.types import (
    UserMessage, AssistantMessage, ToolMessage,
    TextBlock, ThinkingBlock, ToolCall, Usage,
)

SYSTEM_PROMPT = "You're a helpful assistant."

messages = [
    UserMessage(content=[TextBlock(text="What's the weather like in Madrid?")]),
]

while True:
    with completion_stream(
        model="anthropic:claude-3-5-sonnet-latest",
        messages=messages,
        system_message=SYSTEM_PROMPT,                 # request-scoped; not a message in `messages`
        tools=[get_weather_tool],
    ) as s:
        for event in s:
            match event.type:
                case "thinking_delta" | "text_delta":
                    sys.stdout.write(event.delta); sys.stdout.flush()
                case "tool_call_start":
                    sys.stdout.write(f"\n→ {event.name}(")
                case "tool_call_delta":
                    sys.stdout.write(event.arguments_delta); sys.stdout.flush()
                case "tool_call_end":
                    sys.stdout.write(")\n")
            # The terminal FinishEvent / ErrorEvent ends iteration; after the for-loop
            # exits, s.message is the assembled AssistantMessage.

    # `with` closed the HTTP connection. `s` is still alive: its live accessors
    # (s.message, s.finish_reason, s.tool_calls, s.usage) reflect the assembled state.
    messages.append(s.message)

    if s.finish_reason != "tool_use":            # SDK-canonical — see api_prd.md §12.1
        break

    # s.tool_calls — same ToolCall instances filtered out of s.message.content
    for tc in s.tool_calls:
        result = execute_tool(tc)
        messages.append(ToolMessage(
            tool_call_id=tc.id,
            content=[TextBlock(text=result)],
        ))
```

The `with` block guarantees the upstream HTTP connection is closed even if the consumer `break`s or raises. After the `for event in s:` loop exits, `s.message` is the assembled `AssistantMessage` (equivalent to the `message` field on the terminal `FinishEvent`).

---

## Turn 1 — timeline of events

The stream opens with `StartEvent`, walks through two content blocks (`thinking` then `tool_call`), each bracketed by `_start` / `_delta` / `_end` events, emits a `UsageEvent`, and terminates with exactly one `FinishEvent` (or `ErrorEvent`, if the *stream* itself broke).

```
[tick]  [event]                                                [partial.content[] snapshot]
─────────────────────────────────────────────────────────────────────────────────────────────────────────
   1    StartEvent                                             []
   2    ThinkingStartEvent(index=0)                            [ThinkingBlock(text="", signature=None)]
   3    ThinkingDeltaEvent(0, delta="The user wants")          [ThinkingBlock(text="The user wants")]
   4    ThinkingDeltaEvent(0, delta=" current weather in ")    [ThinkingBlock(text="The user wants current weather in ")]
   5    ThinkingDeltaEvent(0, delta="Madrid. I should call ")  [ThinkingBlock(text="...Madrid. I should call ")]
   6    ThinkingDeltaEvent(0, delta="get_weather.")            [ThinkingBlock(text="...call get_weather.")]
   7    ThinkingEndEvent(0, content="<full text>",             [ThinkingBlock(text="The user wants current weather in
        signature="eyJhbGciOi...opaque...")                     Madrid. I should call get_weather.",
                                                                signature="eyJhbGciOi...opaque...")]
   8    ToolCallStartEvent(index=1,                            [ThinkingBlock(...),
                           id="toolu_01ABcdEFghIJklMN",         ToolCall(id="toolu_01AB...", name="get_weather",
                           name="get_weather")                           arguments={}, partial_arguments="",
                                                                         complete=False)]
   9    ToolCallDeltaEvent(1, arguments_delta='{"city"')       [..., ToolCall(arguments={},
                                                                              partial_arguments='{"city"',
                                                                              complete=False)]
  10    ToolCallDeltaEvent(1, arguments_delta=': "Mad')        [..., ToolCall(arguments={},
                                                                              partial_arguments='{"city": "Mad',
                                                                              complete=False)]
  11    ToolCallDeltaEvent(1, arguments_delta='rid"}')         [..., ToolCall(arguments={},
                                                                              partial_arguments='{"city": "Madrid"}',
                                                                              complete=False)]
  12    ToolCallEndEvent(1, tool_call=                         [..., ToolCall(arguments={"city": "Madrid"},
                              ToolCall(arguments={"city":                     partial_arguments="",
                              "Madrid"}, complete=True, ...))                 complete=True)]
  13    UsageEvent(usage=Usage(input_tokens=87,                (content unchanged; partial.usage now populated)
                               output_tokens=64, total=151))
  14    FinishEvent(finish_reason="tool_use",                  (final deep snapshot — see "Zooming in" below)
                    provider_finish_reason="tool_use",
                    cancelled=False, ...)
```

Three load-bearing rules from `api_prd.md` §5.4 to keep in mind while reading the table:

- **`partial` is a SHARED REFERENCE on `*_delta` events.** Snapshot tick #5's `event.partial` and tick #6 mutates it underneath you. Read fields immediately, or `event.partial.model_copy(deep=True)` if you must store.
- **`partial` is a DEEP SNAPSHOT on `*_start`, `*_end`, `UsageEvent`, `FinishEvent`, `ErrorEvent`.** Safe to store, serialize, or send across threads.
- **A streaming `ToolCall` block in `partial.content[i]` has `complete=False`** and accumulates raw JSON fragments into `partial_arguments`. Only `ToolCallEndEvent` resolves the buffer: `arguments` becomes the parsed dict, `partial_arguments` resets to `""`, `complete=True`. Same `ToolCall` object throughout — two phases, never two classes.

### Zooming in — three representative events

**`StartEvent`** — provenance is already populated; content is empty:

```python
StartEvent(
    partial=AssistantMessage(
        content=[],
        finish_reason=None,                  # not set until FinishEvent
        provider_finish_reason=None,
        cancelled=False,
        error_message=None,
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        response_id="msg_01XY...",
        usage=None,
        timestamp=1748361000123,
    ),
)
```

**`ToolCallEndEvent`** — the `tool_call` field IS the same `ToolCall` instance that lives in `partial.content[1]` (one object, two views):

```python
ToolCallEndEvent(
    index=1,
    tool_call=ToolCall(
        id="toolu_01ABcdEFghIJklMN",
        name="get_weather",
        arguments={"city": "Madrid"},       # parsed
        partial_arguments="",               # reset
        complete=True,
        thought_signature=None,
    ),
    partial=AssistantMessage(               # deep snapshot
        content=[
            ThinkingBlock(
                text="The user wants current weather in Madrid. I should call get_weather.",
                signature="eyJhbGciOi...opaque...",
            ),
            ToolCall(                       # same instance as ToolCallEndEvent.tool_call above
                id="toolu_01ABcdEFghIJklMN",
                name="get_weather",
                arguments={"city": "Madrid"},
                partial_arguments="",
                complete=True,
            ),
        ],
        finish_reason=None,                 # not yet — only FinishEvent sets these
        provider_finish_reason=None,
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        response_id="msg_01XY...",
        usage=None,                         # populated by the upcoming UsageEvent
        timestamp=1748361000123,
    ),
)
```

**`FinishEvent`** (turn 1 terminal):

```python
FinishEvent(
    message=AssistantMessage(               # deep snapshot of the fully assembled message
        content=[
            ThinkingBlock(
                text="The user wants current weather in Madrid. I should call get_weather.",
                signature="eyJhbGciOi...opaque...",
            ),
            ToolCall(
                id="toolu_01ABcdEFghIJklMN",
                name="get_weather",
                arguments={"city": "Madrid"},
                partial_arguments="",
                complete=True,
            ),
        ],
        finish_reason="tool_use",                       # SDK-canonical
        provider_finish_reason="tool_use",              # raw Anthropic terminal — equal here
        cancelled=False,
        error_message=None,
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        response_id="msg_01XY...",
        usage=Usage(input_tokens=87, output_tokens=64, total_tokens=151),
        timestamp=1748361000123,
    ),
    finish_reason="tool_use",
    provider_finish_reason="tool_use",
    cancelled=False,
    usage=Usage(input_tokens=87, output_tokens=64, total_tokens=151),
    tool_calls=[                                        # same ToolCall instance as message.content[1]
        ToolCall(
            id="toolu_01ABcdEFghIJklMN",
            name="get_weather",
            arguments={"city": "Madrid"},
            partial_arguments="",
            complete=True,
        ),
    ],
)
```

After the `for` loop exits, `s.message` is byte-for-byte equivalent to `FinishEvent.message`. The outer loop appends it to `messages` and proceeds to run the tool.

---

## Between turns — append `ToolMessage`

Identical to the non-streaming version: the loop runs `execute_tool(tc)`, gets the string `"18°C, sunny, light breeze from the SW."`, and appends:

```python
ToolMessage(
    tool_call_id="toolu_01ABcdEFghIJklMN",     # matches the streamed ToolCall.id
    content=[TextBlock(text="18°C, sunny, light breeze from the SW.")],
    name="get_weather",
    is_error=False,
    timestamp=1748361000456,
)
```

`messages` now has 4 entries: system, user, the streamed `AssistantMessage`, and this `ToolMessage`.

---

## Turn 2 — timeline of events

The second stream runs against the 4-message conversation. The model emits a short thinking block and then the answer text.

```
[tick]  [event]                                                [partial.content[] snapshot]
─────────────────────────────────────────────────────────────────────────────────────────────────────────
   1    StartEvent                                             []
   2    ThinkingStartEvent(index=0)                            [ThinkingBlock(text="", signature=None)]
   3    ThinkingDeltaEvent(0, delta="I have the weather       [ThinkingBlock(text="I have the weather data. ")]
                                     data. ")
   4    ThinkingDeltaEvent(0, delta="I'll summarize           [ThinkingBlock(text="...summarize concisely.")]
                                     concisely.")
   5    ThinkingEndEvent(0, content="<full text>",             [ThinkingBlock(text="I have the weather data. I'll
        signature="eyJhbGciOi...opaque-2...")                  summarize concisely.",
                                                               signature="eyJhbGciOi...opaque-2...")]
   6    TextStartEvent(index=1)                                [..., TextBlock(text="")]
   7    TextDeltaEvent(1, delta="It's 18 °C ")                 [..., TextBlock(text="It's 18 °C ")]
   8    TextDeltaEvent(1, delta="and sunny in Madrid ")        [..., TextBlock(text="It's 18 °C and sunny in Madrid ")]
   9    TextDeltaEvent(1, delta="right now, with a light ")    [..., TextBlock(text="...with a light ")]
  10    TextDeltaEvent(1, delta="breeze from the southwest.")  [..., TextBlock(text="...southwest.")]
  11    TextEndEvent(1, content="It's 18 °C and sunny in       [..., TextBlock(text="It's 18 °C and sunny in Madrid
        Madrid right now, with a light breeze from              right now, with a light breeze from the southwest.")]
        the southwest.")
  12    UsageEvent(usage=Usage(input_tokens=158,                (content unchanged; partial.usage populated)
                               output_tokens=42, total=200))
  13    FinishEvent(finish_reason="stop",                       (final deep snapshot — see below)
                    provider_finish_reason="end_turn",
                    cancelled=False, ...)
```

### Turn 2 `FinishEvent`

```python
FinishEvent(
    message=AssistantMessage(
        content=[
            ThinkingBlock(
                text="I have the weather data. I'll summarize concisely.",
                signature="eyJhbGciOi...opaque-2...",
            ),
            TextBlock(
                text="It's 18 °C and sunny in Madrid right now, with a light breeze from the southwest.",
            ),
        ],
        finish_reason="stop",                          # SDK-canonical
        provider_finish_reason="end_turn",             # raw Anthropic terminal — differs from canonical here
        cancelled=False,
        error_message=None,
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        response_id="msg_01ZW...",
        usage=Usage(input_tokens=158, output_tokens=42, total_tokens=200),
        timestamp=1748361001789,
    ),
    finish_reason="stop",
    provider_finish_reason="end_turn",
    cancelled=False,
    usage=Usage(input_tokens=158, output_tokens=42, total_tokens=200),
    tool_calls=[],                                     # no tool calls in this turn
)
```

`s.finish_reason == "stop"`, so the outer loop breaks. The final `messages` list is identical to `data_sample.md` Step 3 — same shape, same field values, just assembled incrementally rather than in one shot.

---

## What if the stream ends in a refusal / safety filter?

Per `api_prd.md` §12.1, refusals and content filters arrive as **data on `FinishEvent`** (not as `ErrorEvent`). For example, if Anthropic had returned `"refusal"` on the second turn:

```python
FinishEvent(
    message=AssistantMessage(
        content=[
            RefusalBlock(text="I can't help with that."),
        ],
        finish_reason="error",                         # SDK-canonical (load-bearing normalization)
        provider_finish_reason="refusal",              # raw Anthropic terminal
        cancelled=False,
        error_message="Anthropic refusal stop reason", # short human-readable description
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        ...
    ),
    finish_reason="error",
    provider_finish_reason="refusal",
    cancelled=False,
    usage=Usage(...),
    tool_calls=[],
)
```

The model successfully produced a turn — the turn just happens to be a refusal. `ErrorEvent` is reserved for transport / parser failures (HTTP errors, malformed JSON mid-stream, parser bugs). Refusals streamed via OpenAI's separate refusal channel arrive as `refusal_start` / `refusal_delta` / `refusal_end` events first, then terminate with the same `FinishEvent(finish_reason="error", ...)` shape.

## What if the user cancels mid-stream?

`s.cancel()` (sync) or `await s.cancel()` (async) tears down the HTTP connection and causes the iterator to yield one final `FinishEvent(cancelled=True)` carrying the partial message and best-known usage. Cancellation is a `FinishEvent`, not an `ErrorEvent` — nothing failed, the user terminated. `s.message` is then *potentially* resumable: push it into `messages` and continue (the SDK does not retry or resume automatically). If the *task* is cancelled via `task.cancel()` / `asyncio.timeout()` instead, the iterator re-raises `CancelledError` rather than emitting a terminal event. See `api_prd.md` §5.4.1 for the full sync/async table.

---

## Things worth noticing (streaming-specific)

- **The system prompt rides on the request, not in `messages`.** There is no `SystemMessage` class and no `"system"` role; the `system_message=` kwarg on `completion_stream()` (or `ChatCompletionRequest.system_message` at the DTO level) is what the model sees as its system prompt. The transport projects it into Anthropic's top-level `system` field on the wire, or OpenAI's prepended `role: "system"` message, etc. Streaming behaviour is otherwise identical: the prompt is sent once at the start of the call, and never appears in the per-event stream — `StartEvent.partial.content` is empty and grows from there.
- **One `ToolCall` class, two phases.** While arguments stream in, `partial_arguments` accumulates fragments and `complete=False`. `ToolCallEndEvent` flips it to the resolved state: `arguments` is the parsed dict, `partial_arguments=""`, `complete=True`. The same `ToolCall` object lives in `partial.content[i]` throughout — and is the same instance handed to you in `ToolCallEndEvent.tool_call` and in `FinishEvent.tool_calls`. Mutating it through any one view mutates it through all of them.
- **`partial` snapshot rules are deliberate** (see the three bullets after the turn-1 table). The shared-reference vs deep-snapshot distinction is the difference between O(N) and O(N²) memory pressure on long streams; the SDK splits at the natural read/write boundaries (`_start`/`_end`/`finish`/`error` = stable, `_delta` = hot loop).
- **`FinishEvent` for a refusal/safety outcome is still a `FinishEvent`.** Check `s.finish_reason == "error"` (one canonical check across every provider) and read `s.message.error_message` for the human-readable description; the raw upstream string is on `s.message.provider_finish_reason` for telemetry / branching. `ErrorEvent` means the stream itself broke — HTTP error, malformed JSON mid-stream, parser bug.
- **Cancellation is a `FinishEvent(cancelled=True)`** when invoked via `s.cancel()` / `await s.cancel()`; it's `CancelledError` (re-raised) when invoked via `task.cancel()` / `asyncio.timeout()`. The two paths are intentional; see `api_prd.md` §5.4.1.
- **`s.finish_reason`, `s.tool_calls`, `s.message`, `s.usage`** are live accessors on the stream. They reflect the current accumulator state during iteration, and after the iterator terminates they hold the final assembled values. After the `with` block exits, those live accessors still work — what's closed is the HTTP connection, not the stream's Python state.
- **`s.finish_reason` is the SDK-canonical value** (`"stop"` / `"length"` / `"tool_use"` / `"error"`) — same vocabulary as the non-streaming `response.finish_reason`. The raw upstream string is always preserved on `s.message.provider_finish_reason` (and on the `FinishEvent.provider_finish_reason` field). Per the worked-examples table in `api_prd.md` §12.1: Anthropic's `"end_turn"` → canonical `"stop"`; Anthropic's `"tool_use"` → canonical `"tool_use"`; OpenAI's `"tool_calls"` → canonical `"tool_use"`; OpenAI `"content_filter"` / Anthropic `"refusal"` / Gemini `"SAFETY"` / etc. → canonical `"error"`.
- **The two streams produce the SAME `messages` list as the non-streaming sample.** The terminal `FinishEvent.message` is field-for-field equivalent to what `response.message` would have been on `completion()`. The same conversation can be replayed across streaming/non-streaming boundaries without reshaping.
