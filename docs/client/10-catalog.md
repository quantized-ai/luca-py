# Model Catalog

The catalog is an in-memory dict of `(provider, model) → ModelInfo` records
covering pricing, context window, and capability flags. It lives behind
exactly **one** door — the helper — so the request path is never gated by
catalog state.

`luca/client/catalog/` ships with a small curated set of records
generated from
[`models.dev`](https://models.dev) covering OpenAI / Anthropic /
OpenRouter.

## Public surface

```python
from luca.client import catalog
from luca.client.types import ModelInfo, ModelCost

# Lookup (provider, model) → ModelInfo | None
info = catalog.get("openai", "gpt-4o")

# Filter
catalog.list(provider="openai")
catalog.list(supports="tools")
catalog.list(supports="reasoning")
catalog.list(supports="prompt_caching")
catalog.list(supports="structured_output_strict")

# Register or override
catalog.register(
    provider="custom-host",
    model="my-model",
    info=ModelInfo(
        provider="custom-host",
        model="my-model",
        context_window=128_000,
        max_tokens=4096,
        supports_tools=True,
        cost=ModelCost(
            input_per_million_tokens=0.50,
            output_per_million_tokens=1.50,
        ),
    ),
)
```

`catalog.list()` filters by `supports="..."` against the boolean flags on
`ModelInfo` (`supports_image_input` → `"vision"`, `supports_audio_input` →
`"audio"`, etc.). See `_matches_supports` in
`luca/client/catalog/_store.py` for the exact mapping.

## `ModelInfo` shape

```python
class ModelInfo(BaseModel):
    model: str | None = None
    provider: str | None = None
    display_name: str | None = None
    aliases: list[str] = []

    context_window: int | None = None
    max_tokens: int | None = None

    supports_text_input: bool = True
    supports_image_input: bool = False
    supports_audio_input: bool = False
    supports_pdf_input: bool = False
    supports_video_input: bool = False

    supports_tools: bool = False
    supports_parallel_tool_calls: bool = False
    supports_structured_output: Literal["strict", "loose", "none"] = "none"
    supports_reasoning: bool = False
    reasoning_signature_format: Literal["anthropic", "gemini", "openai", "none"] = "none"
    supports_prompt_caching: bool = False
    supports_streaming: bool = True

    cost: ModelCost | None = None
    compat: dict = {}
```

```python
class ModelCost(BaseModel):
    input_per_million_tokens: float | None = None
    output_per_million_tokens: float | None = None
    cached_input_per_million_tokens: float | None = None
    cache_write_per_million_tokens: float | None = None
    reasoning_per_million_tokens: float | None = None
```

All fields are optional — the SDK uses what's present and falls back
gracefully.

## Cost computation

When a transport finishes a response, it reads
`request.model_info.cost` (populated by the helper from
`catalog.get(provider, model)`) and computes `UsageCost`:

```python
class UsageCost(BaseModel):
    input: float = 0.0
    output: float = 0.0
    cached_input: float = 0.0
    cache_write: float = 0.0
    reasoning: float = 0.0
    total: float = 0.0
```

If the catalog has no entry and you didn't pass `model_info=`,
`response.usage.cost` is `None`.

## Overriding per-call

`model_info=` on any helper overrides the catalog lookup for that call:

```python
from luca.client.types import ModelInfo, ModelCost

completion(
    model="openai:gpt-4o",
    messages=[...],
    model_info=ModelInfo(
        cost=ModelCost(
            input_per_million_tokens=1.00,
            output_per_million_tokens=4.00,
        ),
    ),
)
```

## Important non-rule

The catalog is **informational, not load-bearing**. The SDK does **not**
gate requests on capability flags — passing `tools=` to a model whose
`supports_tools=False` will still be attempted; if the upstream rejects it,
you get a `BadRequestError` back. The catalog exists so callers can make
informed decisions, not so the SDK can second-guess them.

Providers and transports never import
`luca.client.catalog` directly. They only read
`request.model_info`. This is enforced by convention (and by code review on
new transports).
