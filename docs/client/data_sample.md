# Data sample — `while True` tool loop, step by step

A complete walk-through of a `while True` tool loop for a single-question interaction (`"What's the weather like in Madrid?"`), showing what the `messages` list looks like at every step. Uses the actual DTOs from `api_prd.md` §12 — unified `ToolCall` (one class, lives both inside `AssistantMessage.content` and as `message.tool_calls` / `response.tool_calls`; field name is `arguments`), canonical `finish_reason` with raw upstream string preserved on `provider_finish_reason`, and the **system prompt as a request-scoped `system_message=` kwarg** (not a message in `messages`). `content` on user/tool messages is `str | list[Block]` — the list form is used throughout.

## The loop

```python
SYSTEM_PROMPT = "You're a helpful assistant."

messages = [
    UserMessage(content=[TextBlock(text="What's the weather like in Madrid?")]),
]

while True:
    response = completion(
        model="anthropic:claude-3-5-sonnet-latest",
        messages=messages,
        system_message=SYSTEM_PROMPT,                 # request-scoped; not a message in `messages`
        tools=[get_weather_tool],
    )
    messages.append(response.message)

    if response.finish_reason != "tool_use":          # SDK-canonical — covers OpenAI "tool_calls", Anthropic "tool_use", …
        for block in response.message.content:
            if block.type == "text":
                print(block.text)
        break

    # response.tool_calls forwards to message.tool_calls — same ToolCall instances,
    # filtered out of message.content, never copied.
    for tc in response.tool_calls:
        result = execute_tool(tc)
        messages.append(ToolMessage(
            tool_call_id=tc.id,
            content=[TextBlock(text=result)],
        ))
```

The system prompt rides on every `completion()` call as a kwarg — it never enters `messages`. Each call replays the same `system_message="You're a helpful assistant."`; the transport projects it into Anthropic's top-level `system` field on the wire. (OpenAI would prepend a wire-level `role: "system"` message; Gemini would set `systemInstruction`.) None of that shape leaks into the SDK's `messages` list.

---

## Step 0 — Initial state (before first `completion()`)

```python
messages = [
    UserMessage(
        content=[TextBlock(text="What's the weather like in Madrid?")],
    ),
]
# system_message="You're a helpful assistant." is passed to completion(...) on each call.
```

## Step 1 — After first turn: assistant returns thinking + tool call

`completion()` returns; `response.finish_reason == "tool_use"`, so we append `response.message` and stay in the loop.

```python
messages = [
    UserMessage(
        content=[TextBlock(text="What's the weather like in Madrid?")],
    ),
    AssistantMessage(
        content=[
            ThinkingBlock(
                text="The user wants current weather in Madrid. I should call get_weather.",
                signature="eyJhbGciOi...opaque...",   # Anthropic replay token; preserved verbatim next turn
            ),
            ToolCall(
                id="toolu_01ABcdEFghIJklMN",
                name="get_weather",
                arguments={"city": "Madrid"},
                complete=True,            # non-streaming response; arguments arrived whole
                partial_arguments="",
            ),
        ],
        finish_reason="tool_use",                  # SDK-canonical
        provider_finish_reason="tool_use",         # raw Anthropic terminal — equal here, since Anthropic already uses "tool_use"
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        response_id="msg_01XY...",
        usage=Usage(input_tokens=87, output_tokens=64, total_tokens=151),
        timestamp=1748361000123,
    ),
]
```

## Step 2 — After tool execution: append `ToolMessage` with the result

The loop runs `execute_tool(tc)`, gets the string `"18°C, sunny, light breeze from the SW."`, and wraps it in a `ToolMessage` whose `tool_call_id` matches the `ToolCall.id` above.

```python
messages = [
    UserMessage(
        content=[TextBlock(text="What's the weather like in Madrid?")],
    ),
    AssistantMessage(
        content=[
            ThinkingBlock(
                text="The user wants current weather in Madrid. I should call get_weather.",
                signature="eyJhbGciOi...opaque...",
            ),
            ToolCall(
                id="toolu_01ABcdEFghIJklMN",
                name="get_weather",
                arguments={"city": "Madrid"},
                complete=True,
            ),
        ],
        finish_reason="tool_use",
        provider_finish_reason="tool_use",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        response_id="msg_01XY...",
        usage=Usage(input_tokens=87, output_tokens=64, total_tokens=151),
        timestamp=1748361000123,
    ),
    ToolMessage(
        tool_call_id="toolu_01ABcdEFghIJklMN",      # must match the ToolCall.id above
        content=[TextBlock(text="18°C, sunny, light breeze from the SW.")],
        name="get_weather",                          # optional; some OpenAI-compat providers want it
        is_error=False,
        timestamp=1748361000456,
    ),
]
```

## Step 3 — After second turn: assistant produces the final text answer

The second `completion()` call sees the full conversation (including the tool result), and the model now answers in plain text. `response.finish_reason == "stop"`, so the loop iterates `response.message.content` to print the text and `break`s.

```python
messages = [
    UserMessage(
        content=[TextBlock(text="What's the weather like in Madrid?")],
    ),
    AssistantMessage(
        content=[
            ThinkingBlock(
                text="The user wants current weather in Madrid. I should call get_weather.",
                signature="eyJhbGciOi...opaque...",
            ),
            ToolCall(
                id="toolu_01ABcdEFghIJklMN",
                name="get_weather",
                arguments={"city": "Madrid"},
                complete=True,
            ),
        ],
        finish_reason="tool_use",
        provider_finish_reason="tool_use",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        response_id="msg_01XY...",
        usage=Usage(input_tokens=87, output_tokens=64, total_tokens=151),
        timestamp=1748361000123,
    ),
    ToolMessage(
        tool_call_id="toolu_01ABcdEFghIJklMN",
        content=[TextBlock(text="18°C, sunny, light breeze from the SW.")],
        name="get_weather",
        is_error=False,
        timestamp=1748361000456,
    ),
    AssistantMessage(
        content=[
            ThinkingBlock(
                text="I have the weather data. I'll summarize concisely.",
                signature="eyJhbGciOi...opaque-2...",
            ),
            TextBlock(
                text="It's 18 °C and sunny in Madrid right now, with a light breeze from the southwest.",
            ),
        ],
        finish_reason="stop",                      # SDK-canonical
        provider_finish_reason="end_turn",         # raw Anthropic terminal — differs from canonical here
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        response_id="msg_01ZW...",
        usage=Usage(input_tokens=158, output_tokens=42, total_tokens=200),
        timestamp=1748361001789,
    ),
]
```

---

## Things worth noticing

- **The system prompt lives on the request, not in `messages`.** There is intentionally no `SystemMessage` class and no `"system"` role in this SDK. Pass `system_message="..."` (string) or `system_message=[TextBlock(...), ...]` (for prompt-cache markers / multi-segment system prompts) on the helper or on the `ChatCompletionRequest` DTO; the transport projects it into whatever shape the host expects (Anthropic top-level `system`, OpenAI wire-level `role: "system"` prepend, Gemini `systemInstruction`). This is why `messages` in every step above starts with `UserMessage`, not a system message.
- **Tool-call/tool-result pairing is by `id`.** `ToolCall.id` (`"toolu_01AB..."`) on turn 1 must equal `ToolMessage.tool_call_id` on the appended tool result. The transport projects this into whatever shape the wire wants (OpenAI: `tool_call_id` on a `role: "tool"` message; Anthropic: `tool_use_id` on a `tool_result` content block inside a synthetic user message — that's the Anthropic *wire* name; in our SDK the field on `ToolResultBlock` is `tool_call_id`).
- **`ThinkingBlock.signature` is replayed verbatim.** Per `api_prd.md` §5.9, those opaque bytes ride along on every subsequent turn to the *same* provider. The SDK forwards them as-is; if you switch providers mid-loop, the destination will 400 (that's the caller's job to clean up).
- **`ToolCall.arguments` is always a parsed dict**, never a JSON string — even on OpenAI where the wire format stringifies it. The transport parses on the way in. The streaming-state fields (`partial_arguments`, `complete`) ride along on the same `ToolCall` object; they're inert here (`partial_arguments=""`, `complete=True`) because this is a non-streamed response.
- **`finish_reason` is the SDK-canonical value, with the raw upstream string on `provider_finish_reason`.** Per `api_prd.md` §12.1, the loop discriminator is the canonical value (`"tool_use"` here — never the provider's raw `"tool_calls"` on OpenAI or `"tool_use"` on Anthropic). In this Anthropic example the two happen to be equal for the tool-call turn (`"tool_use"` / `"tool_use"`) and differ for the final turn (`"stop"` / `"end_turn"`). The `"error"` axis is the load-bearing normalization — every provider's safety/refusal/content-filter vocabulary collapses to `finish_reason="error"` with a populated `message.error_message`; see `api_prd.md` §12.1's worked-examples table.
- **`response.tool_calls`** is the unified `ToolCall` content blocks filtered out of `message.content` — same instances either way, never copied. There is no separate `ToolCall` ↔ `ToolUseBlock` rename to worry about anymore: one class, used in both places.
- **`response.message` *is* the `AssistantMessage`** you append to `messages` — same object, not a copy. `response.finish_reason`, `response.provider_finish_reason`, `response.tool_calls`, `response.usage`, `response.provider`, `response.model`, `response.error_message` all reach the same instances on `response.message` via `__getattr__` forwarding — no duplicated storage, no possibility of drift. The canonical way to read the model's *output* is to iterate `response.message.content` (there is intentionally no `response.content` / `response.text` / `response.refusal` shortcut — those would flatten the typed content-block sequence into a string and hide thinking blocks, refusals, and ordering).
