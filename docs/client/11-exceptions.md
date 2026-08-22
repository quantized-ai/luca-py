# Exceptions

Every exception raised by the SDK inherits from `ClientError`. They all
live in `luca.client.exceptions`.

```python
from luca.client.exceptions import (
    ClientError,             # base
      ConfigurationError,
        AuthenticationError,   # 401
      BadRequestError,         # 400
        ContextLengthExceededError,
        InvalidModelError,
        UnsupportedParameterError,
      ProviderNotFoundError,   # unknown provider name
      ModelNotFoundError,      # 404 from upstream
      RateLimitError,          # 429 (carries retry_after)
      ProviderAPIError,        # 5xx
      ConnectionError,         # network failure
      TimeoutError,            # httpx timeout
      StructuredOutputError,   # JSON/schema validation failure
      StreamError,             # mid-stream protocol violation
)
```

Every `ClientError` carries:

- `provider: str | None` — host name (e.g. `"openai"`, `"groq"`).
- `original_exception: BaseException | None` — the underlying cause when
  applicable.

Extra fields per subclass:

- `RateLimitError.retry_after: float | None` — seconds, when the upstream
  provides it.

## What is and isn't an exception

This is the **load-bearing split** to internalize:

| Category | How it surfaces |
|---|---|
| Transport / SDK / configuration failures (4xx, 5xx, timeouts, malformed responses, missing API keys, registry misses) | **Raised** as `ClientError` subclasses. |
| LLM-side moderation outcomes (refusals, safety filters, content filters) | **Returned** as a normal `ChatCompletionResponse` / `FinishEvent` with `finish_reason="error"` and an `error_message`. |
| User cancellation of a stream | **Returned** as a `FinishEvent(cancelled=True)`. Not an error. |

The point: callers always know that an exception means "the call did not
succeed at the protocol level". A refusal is the model speaking — it's data
to render or branch on, not a bug to catch.

## Mapping (per transport)

Each transport's `_map_chat_completion_http_error` decides which subclass to
raise. Both OpenAI transports share one mapping (`OpenAIErrorMappingMixin` —
same error envelope on `/v1/responses` and `/chat/completions`), and every
OpenAI-compatible host inherits it:

| HTTP / error | Raised as |
|---|---|
| 401 | `AuthenticationError` |
| 400 + `error.type` mentions context length | `ContextLengthExceededError` |
| 400 (other) | `BadRequestError` (or a subclass) |
| 404 | `ModelNotFoundError` |
| 429 | `RateLimitError(retry_after=...)` |
| 5xx | `ProviderAPIError` |
| `httpx.TimeoutException` | `TimeoutError` |
| `httpx.NetworkError` | `ConnectionError` |
| anything else | `ProviderAPIError` |

`AnthropicTransport` has its own mapping with the same shape — refer to
its `_map_chat_completion_http_error` for specifics.

## Streaming

Streams fail in two places, and the place decides the shape.

**Rejected before the stream opens** — the provider answered the HTTP request
with a 4xx/5xx, so no event was ever emitted. The mapped `ClientError` is
**raised** at the `with` / `async with` line, exactly as `completion()` raises
it. The same rejection must not change shape just because you asked for a
stream:

```python
try:
    with completion_stream(model="openrouter:anthropic/claude-sonnet-5", messages=[...]) as stream:
        for event in stream:
            ...
except ProviderAPIError as e:
    print(e)  # "This request requires more credits, or fewer max_tokens…"
```

**Broken after it opened** — a transport error or protocol violation mid-flight
flows through `_handle_iter_exception` and surfaces as a terminal `ErrorEvent`
carrying the typed `ClientError`. By then a stream exists to terminate, so
there is an event to do it with; everything received before the break stays
readable on `stream.message`. Iteration ends after that event — the stream is
single-use.

Calling `stream.collect()` re-raises the `error` from the `ErrorEvent`
directly, so `collect()` raises in both cases.

## Typical handler

```python
from luca.client import completion
from luca.client.exceptions import (
    RateLimitError, AuthenticationError, ClientError,
)

try:
    response = completion(model="openai:gpt-4o", messages=[...])
except AuthenticationError:
    # check OPENAI_API_KEY
    raise
except RateLimitError as e:
    sleep_for = e.retry_after or 1.0
    ...
except ClientError as e:
    # everything else SDK-related
    log.error("provider=%s err=%s", e.provider, e)
    raise

answer = response.messages[-1]
if answer.finish_reason == "error":
    # LLM-side refusal / safety — render to the user, don't retry
    print("model refused:", answer.error_message)
```
