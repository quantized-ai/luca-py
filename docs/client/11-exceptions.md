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
- `StreamError.partial_message: AssistantMessage | None` — everything
  received before the protocol violation.

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
raise. For `OpenAITransport` (and OpenAI-compatible hosts):

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

Inside a stream, a transport error or protocol violation flows through
`_handle_iter_exception` and surfaces as a terminal `ErrorEvent` carrying
the typed `ClientError`. Iteration ends after that event — the stream is
single-use.

Calling `stream.collect()` re-raises the `error` from the `ErrorEvent`
directly.

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

if response.finish_reason == "error":
    # LLM-side refusal / safety — render to the user, don't retry
    print("model refused:", response.error_message)
```
