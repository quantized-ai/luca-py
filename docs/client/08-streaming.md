# Streaming

Streaming is exposed as **two dedicated functions**, not a `stream=True`
flag:

| Function | Returns |
|---|---|
| `completion_stream(...)` | `ChatCompletionStream` (sync iterator) |
| `acompletion_stream(...)` | `AsyncChatCompletionStream` (async iterator) |

The HTTP request opens lazily on the first iteration, so the idiom is:

```python
# sync
with completion_stream(model="...", messages=[...]) as s:
    for event in s:
        ...

# async — no `await` on the stream creation
async with acompletion_stream(model="...", messages=[...]) as s:
    async for event in s:
        ...
```

Failing to close the stream emits a `ResourceWarning` from `__del__`.

## The event union

Every event is a Pydantic model with `type` as discriminator. Every event
except `error` carries `partial: AssistantMessage` — the live message being
built. `*_delta` events share the same reference; `*_start` / `*_end` /
`usage` / `finish` events carry a deep snapshot so you can save it
mid-stream without races.

| Event | Fields (besides `type` / `partial`) | Emitted when |
|---|---|---|
| `start` | — | Exactly once, before any block. |
| `text_start` | `index` | New text block. |
| `text_delta` | `index`, `delta` | Text chunk. |
| `text_end` | `index`, `content` | Text block closed. |
| `thinking_start` | `index` | New thinking block. |
| `thinking_delta` | `index`, `delta` | Reasoning chunk. |
| `thinking_end` | `index`, `content` | Thinking block closed. |
| `tool_call_start` | `index`, `id`, `name` | New tool call. |
| `tool_call_delta` | `index`, `arguments_delta` | JSON fragment. |
| `tool_call_end` | `index`, `tool_call` | Tool call complete, `arguments` parsed. |
| `refusal_start` | `index` | New refusal block (strict-mode OpenAI). |
| `refusal_delta` | `index`, `delta` | Refusal chunk. |
| `refusal_end` | `index`, `content` | Refusal block closed. |
| `usage` | `usage` | Token usage available. |
| **`finish`** | `message`, `finish_reason`, `provider_finish_reason`, `cancelled`, `usage`, `tool_calls` | **Exactly one terminal** when the model produced a turn. |
| **`error`** | `error`, `partial_message`, `usage` | **Terminal** when the stream itself broke (HTTP error, malformed metadata). |

## Two terminals, on a strict split

- `FinishEvent` — the **model** produced a terminal. Always the terminal
  when the wire closed normally, even when `finish_reason="error"` (a
  refusal / safety filter / content filter outcome). Cancellation is also a
  `FinishEvent` with `cancelled=True`.
- `ErrorEvent` — the **stream** broke. HTTP error, malformed JSON
  mid-stream, parser bug. Carries `error: ClientError` and
  `partial_message`.

The split is on **source of failure**, not on outcome. A safety refusal is
still `FinishEvent(finish_reason="error", error_message="...")`, not an
`ErrorEvent`.

Every stream emits **exactly one** terminal event by structure.

## Live accessors

The stream exposes the message under construction:

```python
with completion_stream(...) as s:
    for event in s:
        ...
        # These reflect live state:
        s.message              # AssistantMessage being built
        s.text                 # concatenated text-block contents so far
        s.tool_calls           # filter of message.content (same instances)
        s.finish_reason        # SDK-canonical, set at the terminal
        s.provider_finish_reason  # raw upstream string, set on RawFinish
        s.usage                # Usage | None
        s.cancelled            # bool
```

## Cancellation

```python
with completion_stream(...) as s:
    for event in s:
        if event.type == "text_delta" and "STOP" in event.delta:
            s.cancel()        # next iteration emits FinishEvent(cancelled=True)
```

Cancellation is **not** an error. The stream still emits a `FinishEvent` —
with `cancelled=True`, `finish_reason=None` if no terminal arrived, and the
partial message + partial usage preserved.

## `collect()`

Skip the event loop and get a `ChatCompletionResponse` directly:

```python
with completion_stream(...) as s:
    response = s.collect()

print(response.message.content[0].text)
print(response.finish_reason)
```

On `ErrorEvent`, `collect()` re-raises the underlying `ClientError`.
`collect()` consumes the stream — calling iteration afterwards raises
`StreamError`.

## Async mirror

`AsyncChatCompletionStream` is the structural twin: `__aenter__` /
`__aexit__`, `async for`, `await s.cancel()`, `await s.collect()`.

```python
async with acompletion_stream(...) as s:
    async for event in s:
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)
```

## Internal: `RawStreamEvent` (transport-internal)

Transport implementations communicate with `ChatCompletionStream` through a
small dataclass-based vocabulary: `RawBlockStart`, `RawTextDelta`,
`RawThinkingDelta`, `RawToolArgumentsDelta`, `RawRefusalDelta`,
`RawBlockStop`, `RawFinish`, `RawUsage`.

These are not consumed by end users — only by people writing a new
transport. The block-start/stop pairing, dense indices, and "raw finish
string passed through unchanged" rules are documented in
[`architecture.md`](../../architecture.md) §10.
