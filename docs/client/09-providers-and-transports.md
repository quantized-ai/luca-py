# Providers and Transports

The helper layer is the easy entry point; the **provider** and **transport**
layers are what every helper call eventually reaches. You can use either
directly for finer control over `httpx` lifecycle, headers, or registration
of new hosts.

## Built-in providers (V1)

| Name | Class | Default base URL | Env var | Transport |
|---|---|---|---|---|
| `openai` | `OpenAIProvider` | `https://api.openai.com/v1` | `OPENAI_API_KEY` | `OpenAITransport` |
| `anthropic` | `AnthropicProvider` | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` | `AnthropicTransport` |
| `openrouter` | `OpenRouterProvider` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `OpenRouterTransport` |
| `groq` | `GenericProvider` (from dict) | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` | `OpenAITransport` |
| `deepseek` | `GenericProvider` (from dict) | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` | `OpenAITransport` |
| `ollama` | `GenericProvider` (from dict) | `http://localhost:11434/v1` | (none) | `OpenAITransport` |
| `faux` | `FauxProvider` | — | (none) | `FauxTransport` |

`PROVIDERS` lives in `luca/client/providers/__init__.py`.

## Model strings

`provider:model` is the canonical form. The string is split at the **first**
colon; everything after is the wire model id, sent verbatim:

| Input | Resolved as |
|---|---|
| `"openai:gpt-4o"` | `("openai", "gpt-4o")` |
| `"openrouter:openai/gpt-4o"` | `("openrouter", "openai/gpt-4o")` |
| `"together:meta-llama/Llama-3.1-70B"` | `("together", "meta-llama/Llama-3.1-70B")` |
| `"gpt-4o"` + `provider="openai"` | `("openai", "gpt-4o")` |
| `"openai:gpt-4o"` + `provider="azure"` | `("azure", "gpt-4o")` — kw wins, prefix stripped |
| `"gpt-4o"` (no `provider=`) | `ValueError` |

There is **no** automatic model translation. Sending
`openrouter:gpt-4o` (without the `openai/` author prefix OpenRouter wants)
will likely 400 — that's the caller's problem.

## Using providers directly

```python
from luca.client.providers import OpenAIProvider
from luca.client.types import ChatCompletionRequest, UserMessage

with OpenAIProvider(api_key="sk-…", timeout=30.0) as prov:
    request = ChatCompletionRequest(
        model="gpt-4o",
        messages=[UserMessage(content="Hello")],
    )
    response = prov.completion(request)
```

`BaseProvider` exposes `.completion`, `.acompletion`, `.completion_stream`,
`.acompletion_stream`, and a `.transport` accessor. Constructor kwargs:

| Kwarg | Notes |
|---|---|
| `api_key` | Overrides env var lookup. |
| `base_url` | Overrides the class default. |
| `transport_class` | Force a specific transport (rare). |
| `transport` | Wrap a pre-built transport instance — provider just delegates. |
| `timeout` | Per-call httpx timeout (default 60s). |
| `http_client` / `async_http_client` | Pre-built httpx clients (for proxies, custom transports, etc.). |

Both `__enter__`/`__exit__` and `__aenter__`/`__aexit__` are implemented so
you can use providers as context managers in sync or async code.

## Adding a host (one-line)

Most OpenAI-compatible hosts are a one-line `PROVIDERS` entry — no new
class, no new transport.

```python
from luca.client.providers import register_provider
from luca.client.transports import OpenAITransport

register_provider("together", {
    "default_base_url": "https://api.together.xyz/v1",
    "default_api_key_env_var": "TOGETHER_API_KEY",
    "default_transport_class": OpenAITransport,
})

# Now:
completion(model="together:meta-llama/Llama-3.1-70B", messages=[...])
```

`register_provider` accepts:

- A config **dict** (as above) → spawns a `GenericProvider` at resolve time.
- A `BaseProvider` **subclass** → instantiated directly for custom behavior.

The escape hatch for a one-off host without registering at all:

```python
completion(
    model="some-vendor:some-model",
    base_url="https://api.some-vendor.example/v1",
    api_key="...",
    transport_class=OpenAITransport,  # tells GenericProvider the wire format
)
```

## Per-vendor quirks: subclass, don't flag

When an OpenAI-compatible host needs different behavior (e.g. `max_tokens`
vs `max_completion_tokens`), the right move is a **transport subclass**
that overrides the relevant private method. We do **not** read flags from
`ModelInfo` and we do **not** add `if self._provider == "..."` branches in
the base transport. See [`architecture.md`](../../architecture.md) §6.3 for
the worked example.

### Reasoning content is read data-driven, not per-vendor

Reasoning ("thinking") models behind OpenAI-compatible hosts — OpenRouter
primarily, but also DeepSeek, Fireworks, Together, … — return the model's
reasoning text on the response (`message.reasoning`, or `reasoning_content`
for DeepSeek-style hosts; `delta.reasoning` / `delta.reasoning_content`
while streaming). The base `OpenAITransport` reads it **if the wire carries
it** and surfaces a `ThinkingBlock` (prepended before the visible text, and
as `thinking_*` stream events). Pure OpenAI responses simply omit the field,
so the branch is inert for them — this is data-driven, **not** an
`if self._provider == "openrouter"` check, and it benefits every
OpenAI-compatible reasoning host at once. Sending an assistant turn back
**drops** the `ThinkingBlock` (round-trip replay via OpenRouter's
`reasoning_details` + `signature` is a future enhancement).

## Lower level: transports

If you need direct access to the wire layer (custom `httpx.Client`, custom
headers, integration tests against a recorded fixture, …):

```python
import httpx
from luca.client.transports import OpenAITransport
from luca.client.types import ChatCompletionRequest, UserMessage

transport = OpenAITransport(
    provider="openai",
    base_url="https://api.openai.com/v1",
    api_key="sk-…",
    timeout=30.0,
    http_client=httpx.Client(proxies="..."),
)

request = ChatCompletionRequest(
    model="gpt-4o",
    messages=[UserMessage(content="Hello")],
)
response = transport.completion(request)
```

`BaseTransport` owns the httpx lifecycle (`close` / `aclose` / context
managers) and the `_headers()` defaults. Each transport subclass implements
the hook methods defined in `ChatCompletionTransportMixin`:

- `_build_chat_completion_payload(request, *, stream=False) -> dict`
- `_parse_chat_completion_response(response, request) -> ChatCompletionResponse`
- `_classify_finish(provider_value, message) -> (canonical, error_message)`
- `_map_chat_completion_http_error(exc) -> ClientError`
- `_chat_completion_stream_class()`, `_async_chat_completion_stream_class()`
- Optional: `_chat_completion_url()`, `_build_chat_completion_httpx_request()`

The `transport=` kwarg on `BaseProvider.__init__` lets you wrap a
pre-configured transport in a provider for a uniform call surface.

## Caching

The helper functions cache one provider instance per
`(name, api_key, base_url, transport_class, timeout)` tuple. Repeated calls
to `completion(...)` reuse the same provider, the same transport, and the
same `httpx.Client` connection pool. Passing a pre-built `provider=` or
`transport=` bypasses the cache and gives you the lifecycle.
