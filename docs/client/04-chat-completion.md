# Chat Completion API

Four public helpers, all imported from `luca.client`:

| Helper | Sync/Async | Returns |
|---|---|---|
| `completion(...)` | sync | `ChatCompletionResponse` |
| `acompletion(...)` | async | `ChatCompletionResponse` |
| `completion_stream(...)` | sync | `ChatCompletionStream` |
| `acompletion_stream(...)` | sync (returns async stream) | `AsyncChatCompletionStream` |

Plus `get_provider(model_or_pair)` for grabbing the cached provider instance
behind a given model string.

> Note: `acompletion_stream(...)` is **a regular `def`**, not `async def`.
> The HTTP request fires on first iteration, so the idiom is
> `async with acompletion_stream(...) as s:` — no `await` on creation.

## Signature

All four helpers share the same kwargs. Required positional:

| Param | Type | Notes |
|---|---|---|
| `model` | `str` | `"openai:gpt-4o"` (preferred) or `"gpt-4o"` + `provider="openai"` |
| `messages` | `list[dict \| Message]` | User/assistant/tool messages. **No `"system"` role.** |

Common keyword args:

| Param | Type | Notes |
|---|---|---|
| `provider` | `str \| BaseProvider \| None` | String name, prefix override, or a pre-built provider instance. |
| `system_message` | `str \| list[TextBlock] \| None` | Request-scoped system prompt. |
| `tools` | `list[Tool \| dict] \| None` | See [`06-tools.md`](06-tools.md). |
| `tool_choice` | `"auto" \| "required" \| "none" \| dict` | Force or block tool use. |
| `response_format` | `dict \| type \| TypeAdapter \| None` | Structured output schema. See [`07-structured-output.md`](07-structured-output.md). |
| `temperature`, `top_p`, `top_k`, `max_tokens`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `logprobs`, `top_logprobs` | various | Standard generation knobs. `None` means "not sent". |
| `reasoning_effort` | `"none" \| "minimal" \| "low" \| "medium" \| "high" \| "xhigh" \| "auto"` | For reasoning-capable models. |
| `thinking_budgets` | `dict[str, int] \| None` | Provider-specific thinking budget. |
| `cache_retention` | `"none" \| "short" \| "long" \| None` | Prompt-cache hint. |
| `session_id` | `str \| None` | Routing affinity hint where supported. |
| `parallel_tool_calls` | `bool \| None` | OpenAI knob. |
| `user` | `str \| None` | End-user identifier. |
| `metadata` | `dict \| None` | Provider-level metadata. |
| `extra_args` | `dict \| None` | Escape hatch — merged into the outbound JSON. |
| `model_info` | `ModelInfo \| dict \| None` | Override the catalog lookup for this call. |
| `api_key` | `str \| None` | Override env. |
| `base_url` | `str \| None` | Override host URL. |
| `timeout` | `float \| None` | Per-call timeout (sec). Default 60. |
| `total_timeout` | `float \| None` | Wall-clock deadline over the whole call (sec). **Async helpers only** (`acompletion` / `acompletion_stream` — the sync helpers have no loop to enforce it). Expiry raises the SDK `TimeoutError`; on a stream it follows the streaming contract: exactly one terminal `ErrorEvent` carrying `TimeoutError`, then close. |
| `transport_class` | `type \| None` | Bypass `PROVIDERS` and use a specific transport class. |
| `transport` | `BaseTransport \| None` | Wrap a pre-built transport instance. |

## Coercion at the boundary

`messages` accepts dict shape or typed instances. Dicts are validated into
`UserMessage` / `AssistantMessage` / `ToolMessage` based on `role`. A dict
with `role="system"` raises `BadRequestError` — system prompts live on
`system_message=`.

`tools` accepts dicts or `Tool` instances. Dicts are coerced via
`Tool.model_validate`.

## The response

`completion()` / `acompletion()` return `ChatCompletionResponse`:

```python
class ChatCompletionResponse(BaseModel):
    message: AssistantMessage
    raw: Any  # raw provider payload, excluded from .model_dump()
    # plus a private _response_format for .parse()
```

There is **no** `response.content` / `response.text` / `response.refusal`
shortcut. The canonical way to read output is to iterate
`response.message.content` and dispatch on `block.type`.

Everything callers commonly read forwards from `response` to
`response.message` via `__getattr__`:

```python
response.finish_reason          # == response.message.finish_reason
response.provider_finish_reason # == response.message.provider_finish_reason
response.usage                  # == response.message.usage
response.provider               # == response.message.provider
response.model                  # == response.message.model
response.tool_calls             # == response.message.tool_calls (filter, not copy)
response.error_message          # == response.message.error_message
response.cancelled              # == response.message.cancelled
```

`response.message.tool_calls` is a `@property` that filters `ToolCall`
instances out of `self.content` — same objects, never copied.

## Finish reasons

There are **two** finish-reason fields on the message:

- `finish_reason` — SDK-canonical, one of `"stop"`, `"length"`,
  `"tool_use"`, `"error"`, or `None` (cancelled before any terminal arrived).
- `provider_finish_reason` — the **raw upstream string**, preserved verbatim
  (`"end_turn"`, `"content_filter"`, `"refusal"`, `"max_tokens"`, …).

Each transport implements `_classify_finish(provider_value, message)` which
maps the raw string + the assembled message to the canonical pair
`(finish_reason, error_message)`. The classifier inspects the assembled
content blocks too — so a strict-mode OpenAI response with a `RefusalBlock`
but a benign `"stop"` upstream lifts to canonical `"error"`.

The canonical mapping summary:

| Canonical | What it means |
|---|---|
| `"stop"` | Model produced a complete turn. |
| `"length"` | Hit `max_tokens` / token budget. |
| `"tool_use"` | Model wants you to execute tool calls and reply. |
| `"error"` | LLM-side refusal / safety / content filter. Check `error_message`. |
| `None` | Stream was cancelled before any terminal arrived (streaming only). |

LLM-side moderation outcomes (refusals, safety filters) are **not**
exceptions — they arrive as a normal response with `finish_reason="error"`
and an `error_message`. Exceptions (`ClientError` and subclasses) are
reserved for transport / SDK / configuration failures. See
[`11-exceptions.md`](11-exceptions.md) for the split.

## Provider caching

The helper caches one provider instance per
`(name, api_key, base_url, transport_class, timeout)` tuple — so repeated
calls reuse the same `httpx.Client` and its connection pool. The cache is
process-local and never evicts.

If you pass a pre-built `provider=` instance the cache is bypassed and you
own the lifecycle.

## Direct provider use

For long-lived services it can be cleaner to hold the provider directly:

```python
from luca.client.providers import OpenAIProvider
from luca.client.types import ChatCompletionRequest, UserMessage

with OpenAIProvider(api_key="sk-…") as prov:
    request = ChatCompletionRequest(
        model="gpt-4o",
        messages=[UserMessage(content="Hello")],
    )
    response = prov.completion(request)
```

See [`09-providers-and-transports.md`](09-providers-and-transports.md) for
the full provider/transport surface.
