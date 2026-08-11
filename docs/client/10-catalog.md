# Model Catalog

The catalog is an in-memory dict of `(provider, model) → ModelInfo` records
covering pricing, context window, and capability flags. It lives behind
exactly **one** door — the helper — so the request path is never gated by
catalog state.

## Where the records come from

[`models.dev`](https://models.dev), narrowed twice. Its one endpoint carries
~180 providers and ~6,000 models; luca keeps the providers it has a transport
for, and within those the models an agent can actually drive — `tool_call` plus
text in and text out, which drops the image, embedding and speech models. That
is **450 records across 6 providers**.

Two files, layered:

| | |
|---|---|
| `luca/client/catalog/_data/models.json` | generated and shipped in the package. The offline floor — no import ever touches the network |
| `$XDG_CACHE_HOME/luca/models.json` | written by a refresh, layered on top by `(provider, model)` |

The cache adds and updates; it never subtracts. A missing, corrupt or
partly-unreadable cache costs only the records it would have added, because
`catalog.get` backs the context window that compaction, the cost screen, the
status gauge and the `@`-mention size cap all read.

```bash
python -m luca.client.catalog.refresh            # update the local cache
python -m luca.client.catalog.refresh --vendor   # regenerate the shipped file
uv run python main.py --refresh-models           # the same, from the TUI
```

> **The catalog is metadata, never a gate.** A provider's line-up moves faster
> than a release does, and `ollama` and custom hosts are not in models.dev at
> all — so `catalog.get` returning `None` means "no metadata", not "cannot be
> used". A request for an unlisted model runs on provider defaults.

models.dev is the default source, not the only one. `catalog.register(...)`
takes a record from anywhere, which is how a local Ollama gets real numbers:
`transports.ollama.discover()` reads what the daemon has pulled and the caller
registers it. For that provider the metadata is load-bearing rather than
decorative — `context_window` becomes the `num_ctx` the transport asks for, so
the figure the compactor reads and the window the server runs are the same
number. See
[09-providers-and-transports.md](09-providers-and-transports.md#ollama-runs-on-the-native-api-not-v1).

`release_date` and `family` come along for a reason: the TUI sorts "newest
first" by the former and collapses a host's near-duplicates by the latter,
which is how bedrock's `us.`/`eu.`/`jp.` copies of one model stop being four
separate choices.

Prices are per `(provider, model)`. A handful of models price by context tier —
above a threshold the rate can double — and `ModelCost` carries one flat rate,
so a very long session under-reports.

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
`response.messages[-1].usage.cost` is `None`.

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
