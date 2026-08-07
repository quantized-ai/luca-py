Guidance for the `luca.client` layer. Read this file whenever you're working in `luca/client/` or `tests/client/`.

## What `luca.client` is

A thin, unified LLM SDK — a deliberately minimal "small, simple LiteLLM" — that exposes one API across providers (OpenAI, Anthropic, OpenRouter, …). It exists to serve `luca.agent`; most new feature work happens in the agent layer. Changes here are usually in service of an agent need.

PyPI distribution name: `luca-ai`.

**Runtime dependencies: `httpx` and `pydantic` only. No vendor SDKs are wrapped.**

## Goals

- One canonical request DTO (`ChatCompletionRequest`) and one canonical response DTO (`ChatCompletionResponse`) across all providers.
- Direct `httpx` calls to each provider's HTTP API.
- Per-vendor wire-format logic isolated inside a single **transport** class per provider.
- Streaming exposed as **separate functions** (`completion_stream` / `acompletion_stream`), never as a `stream=True` flag on the non-streaming call.
- A self-describing `AssistantMessage` (carries `finish_reason`, `usage`, `provider`, `model`, …) so a saved conversation can reload with full context.

## Non-goals

- Server / proxy mode.
- Automatic retries, multi-model fallback, or cross-provider message transformation.
- Wrapping vendor SDKs.
- Batch APIs, re-ranking, embeddings, image, or audio (described in `api_prd.md` but not yet implemented in V1).
- Guardrails or moderation pipelines.

## Authoritative specs

| Document | Authority |
|---|---|
| `api_prd.md` | Public API contract: DTO shapes, kwargs, error semantics, streaming events. |
| `architecture.md` | Internal design: provider/transport layering, streaming subsystem, polymorphism-not-flags rule, finish-reason classification. |
| `docs/client/` | User-facing documentation. Start at `docs/client/README.md`. |
| `docs/client/13-roadmap.md` | What V1 implements vs. the full spec. |

**Conflict resolution:** when code disagrees with a doc, the code wins (the doc needs updating). When a doc disagrees with its spec, the spec wins unless we are consciously evolving it.

## File layout

```
luca/client/                       # the supporting LLM SDK
├── __init__.py                    # public API: completion, acompletion, ...
├── _client.py                     # helper functions; model-string parsing; provider cache
├── exceptions.py                  # ClientError hierarchy
├── testing.py                     # FauxProvider + builder re-exports
│
├── types/                         # canonical DTOs (Pydantic)
│   ├── catalog.py                 # ModelInfo, ModelCost
│   ├── completion.py              # ChatCompletionRequest, ChatCompletionResponse, Usage, UsageCost
│   ├── content.py                 # TextBlock, ImageBlock, ThinkingBlock, ToolCall, RefusalBlock, ...
│   ├── media.py                   # MediaURL, MediaBase64, MediaFileId
│   ├── messages.py                # UserMessage, AssistantMessage, ToolMessage
│   ├── reasoning.py               # Reasoning literal
│   ├── streaming.py               # StreamEvent union, BaseStream, accumulator
│   ├── structured.py              # ResponseFormat, parse_structured_output
│   └── tools.py                   # Tool, ToolChoice, JSON-schema normalization
│
├── providers/
│   ├── __init__.py                # PROVIDERS registry + resolve_provider
│   ├── base.py                    # BaseProvider + ChatCompletionMixin
│   ├── openai.py                  # OpenAI provider (→ OpenAIResponsesTransport)
│   ├── anthropic.py               # Anthropic provider
│   ├── openrouter.py              # OpenRouter provider
│   ├── bedrock.py                 # Bedrock provider (region → base_url)
│   ├── generic.py                 # GenericProvider (used by PROVIDERS dict entries)
│   └── faux.py                    # FauxProvider for tests
│
├── transports/
│   ├── __init__.py                # TRANSPORTS registry
│   ├── base.py                    # BaseTransport + ChatCompletionTransportMixin
│   ├── openai/                    # chat completions: transport.py + stream.py
│   │                              #   + errors.py (envelope shared with responses)
│   ├── openai_responses/          # /v1/responses: transport.py + stream.py
│   ├── anthropic/                 # transport.py + stream.py
│   ├── openrouter/                # subclass of OpenAITransport (only overrides _headers)
│   ├── bedrock/                   # Converse translation; binary eventstream decoder
│   └── faux/                      # scripted responses; no httpx
│
└── catalog/                       # in-memory ModelInfo store
    ├── __init__.py                # public facade: get / list / register
    ├── _store.py                  # dict + load-on-first-access; vendored
    │                              #   records, then the refresh cache over them
    ├── _source.py                 # models.dev -> ModelInfo: the routable-provider
    │                              #   map, the agent-usability filter, the mapping
    ├── refresh.py                 # python -m luca.client.catalog.refresh [--vendor]
    └── _data/                     # models.json (generated) + its loader

tests/client/                      # mirrors luca/client/ layout exactly
api_prd.md                         # client public API contract
architecture.md                    # client internal design spec
testing_architecture.md            # testing strategy for the client
```

## Tests

The project-wide test style in [`AGENTS.md`](AGENTS.md) applies here: **assert
on the full object, not on individual properties.** For this layer that means
the whole outbound payload dict, the whole parsed `AssistantMessage`, the whole
content block — `assert block == ThinkingBlock(...)`, not three attribute
checks. `tests/client/` mirrors `luca/client/` exactly.

## Design principles (non-negotiable — internalize before changing structure)

From `architecture.md`. These are architectural constraints, not preferences.

### 1. Three explicit layers: helper → provider → transport

Arrows point inward only.

- The **helper** (`_client.py`, `__init__.py`) imports providers.
- **Providers** import transports.
- **Transports** never import providers or the catalog.

### 2. Encapsulation — per-vendor logic lives on the transport class

Build, parse, and error-map logic lives as **methods on the transport class**. Per-vendor quirks are added as a **subclass** that overrides the relevant private method.

Never add:
- A `compat=` flag.
- An `if self._provider == "..."` branch inside a base class.

The rule targets branching on *provider identity*. Reading a field that the wire happens to carry is data-driven (not a quirk) and needs no subclass. Example: `OpenAITransport` parses `reasoning` / `reasoning_content` into a `ThinkingBlock` in the base — inert for pure OpenAI (no such field on the wire), live for OpenRouter / DeepSeek / others that carry it.

### 3. Single DTO at every layer

The helper builds a `ChatCompletionRequest` **once**. Provider and transport receive that same instance. There is no intermediate "wire-prep" DTO.

### 4. The catalog lives behind one door — the helper

Providers and transports do **not** import `luca.client.catalog`. They read `request.model_info` only, which the helper has already populated.

### 5. The catalog is metadata, never a gate

Records are generated from [models.dev](https://models.dev), which cannot be
complete — a provider's line-up moves faster than a release does, and `ollama`
and custom hosts are not there at all. So `catalog.get` returning `None` means
"no metadata", never "unusable": the request goes out on provider defaults.
Nothing may start refusing a model because the catalog has not heard of it.

Regenerate the vendored file with `python -m luca.client.catalog.refresh
--vendor`; never hand-edit `_data/models.json`.

### 6. No automatic model translation

`provider:author/model` is the canonical model string, split at the first `:`. Everything after the colon is the wire model id, passed verbatim. No alias expansion, no name mapping.

### 6. Provider is mandatory

Every call must specify a provider — via the `provider:model` prefix or an explicit `provider=` kwarg. Bare model-name lookups are not supported.

### 7. Duck-typed composition, no ABC / Protocol

Base classes are concrete. Hook methods `raise NotImplementedError`. Subclasses override specific hooks; no abstract base classes or Protocol declarations.

### 8. httpx + pydantic only

No vendor SDKs are wrapped. The only runtime dependencies are `httpx` and `pydantic`.

### 9. Simple over magic

No retries, no fallback, no synthesized tools, no cross-provider message transformation, no "just in case" hooks.

### 10. YAGNI on hooks

Extract a hook method only when a real second case requires it. Pre-extracted hooks are invariably the wrong granularity.

## Key facts

### System prompts are request-scoped

There is no `SystemMessage` class and no `"system"` role in the message list. The `system_message=` parameter rides on the `ChatCompletionRequest`, and each transport projects it into the provider's expected wire shape. Passing a dict with `role="system"` inside the `messages` list raises `BadRequestError`.

### `ToolCall` is one class, two views

The same `ToolCall` instances live inside `AssistantMessage.content` and are also surfaced via `message.tool_calls`, `response.tool_calls`, `stream.tool_calls`, and `FinishEvent.tool_calls`. These are **filter views**, never copies. Mutating a `ToolCall` through one view mutates it through all others.

### Provider-native calls have two equivalent forms

A native call/result is either a typed subclass (`ApplyPatchToolCall`,
`ShellToolMessage`, …) or a canonical `ToolCall` / `ToolMessage` whose
`extras` carry `{"custom_type": "<the native type>", …the subclass's own
fields}`. `as_native()` / `as_generic()` convert either way, losslessly.

Equivalence is **by construction, not by parallel code**:
`BaseTransport._resolve_call` normalizes through `as_native()` before any
projector runs, and each transport's result projection starts with
`msg = msg.as_native()`. No projector, and no native class, knows `extras`
exists. Keep it that way — a projector that starts reading `extras` directly
is the bug this design exists to prevent.

Parsing is unaffected: the wire always builds the typed subclass. `extras` is
an input and storage form.

### Two finish-reason fields on `AssistantMessage`

| Field | Value |
|---|---|
| `finish_reason` | SDK-canonical: `"stop"` \| `"length"` \| `"tool_use"` \| `"error"` \| `None` |
| `provider_finish_reason` | Raw upstream string, verbatim. |

Each transport implements `_classify_finish(provider_value, message)` to compute the canonical pair `(finish_reason, error_message)`. It receives the fully assembled message so it can inspect content — for example, strict-mode OpenAI returns a raw `"stop"` reason alongside a `RefusalBlock`, which classifies to canonical `"error"`.

**LLM-side refusals, safety blocks, and content filters are not exceptions.** They arrive as a normal response with `finish_reason="error"` and an `error_message`. `ClientError` subclasses are reserved for transport or network failures.

### Two OpenAI wire protocols

`openai` the PROVIDER runs on `/v1/responses` (`OpenAIResponsesTransport`).
`openai` the TRANSPORT is chat completions, and serves every OpenAI-compatible
host (`groq`, `deepseek`, `ollama`, and `openrouter` by subclassing). They are
siblings, not a base and an override: `input` items vs `messages`, `output`
items vs `choices`, named SSE events vs `delta` chunks, `status` +
`incomplete_details` vs `finish_reason`. The only genuinely shared piece is the
HTTP error envelope, which lives in `transports/openai/errors.py` as
`OpenAIErrorMappingMixin` and is mixed into both.

`OpenAIErrorMappingMixin` must come BEFORE `ChatCompletionTransportMixin` in
the bases — the latter defines the same hook raising `NotImplementedError`, and
MRO order decides which one answers.

Responses specifics: `store: false` always (the whole conversation goes up
every turn), `reasoning: {effort, summary: "auto"}` plus
`include: ["reasoning.encrypted_content"]` when reasoning is requested, and
`stop` / `seed` / `presence_penalty` / `frequency_penalty` / `logprobs` /
`top_logprobs` raise `UnsupportedParameterError` because the endpoint has no
equivalent.

`reasoning.context` is deliberately NOT sent. Only the newest model family
accepts `all_turns`; `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5.1`, `gpt-5.2`,
`o3` and `o4-mini` reject the whole request. It buys nothing either, since
replay under `store: false` runs on the `encrypted_content` in `input`. Do not
re-add it without a per-model gate.

### Reasoning ("thinking") on OpenAI-compatible hosts

`OpenAITransport` parses provider reasoning text into a `ThinkingBlock`,
prepended before visible text. The resulting content order is
`[thinking, text, refusal?, tool_calls…]`.

- **Non-streaming:** `_parse_assistant_message` reads `message.reasoning`,
  falling back to `message.reasoning_content`.
- **Streaming:** `_process_chunk` in `stream.py` emits a `thinking` block from
  `delta.reasoning` / `delta.reasoning_content`. The thinking block claims the
  earliest index (reasoning streams before text), stays open while text
  streams, and closes at finish.
- **Usage:** `reasoning_tokens` flows through `_parse_usage` from
  `usage.completion_tokens_details.reasoning_tokens`.
- **Send-back:** `_project_assistant_message` deliberately **drops**
  `ThinkingBlock` — chat completions has no replay surface at all.

`OpenRouterTransport` inherits all of this behavior because it subclasses
`OpenAITransport` and only overrides `_headers`.

### Replaying a provider-owned attestation

A `ThinkingBlock` carries two opaque halves: `signature` (Anthropic's
signature, OpenAI's `encrypted_content`) and `id` (OpenAI's `rs_…` item id).
Both are minted by ONE (provider, model) pair and rejected by every other with
a 400 that makes the conversation permanently unusable.

`BaseTransport._attestation_is_replayable(message, request)` is the single
rule: replay when `message.provider` / `message.model` match the pair being
called, or when they are absent (a hand-built message means the caller is
driving). The Responses, Anthropic and Bedrock transports each consult it
before projecting a thinking block and DROP the block otherwise — one turn's
visible reasoning is a cheaper loss than a dead conversation.

This is reachable in one keystroke: the agent TUI's `/model` rewrites the
session model between turns, and the whole history is replayed to the new one.
For that to work the projected assistant message has to carry its provenance,
which is why `ConversationProjector` copies `provider` / `model` (see
`AGENTS.agent.md`).

Related: every transport sets `AssistantMessage.model` to the model that was
REQUESTED and `response_model` to whatever the provider reported. Keeping the
requested string in `model` is what makes the comparison above sound — an
Anthropic alias resolves to a dated id on the wire, and comparing that against
`request.model` would drop every attestation.

## Common tasks

### Add an OpenAI-compatible host (e.g. Groq, Together)

Edit `luca/client/providers/__init__.py` and append to `PROVIDERS`:

```python
PROVIDERS["together"] = {
    "default_base_url": "https://api.together.xyz/v1",
    "default_api_key_env_var": "TOGETHER_API_KEY",
    "default_transport_class": OpenAITransport,
}
```

Add an entry to the table in `docs/client/09-providers-and-transports.md`. No new class is needed.

### Add a host whose wire format differs from OpenAI

1. Create a transport subclass that overrides only the specific private method(s) that differ (e.g. `_max_tokens_field`).
2. Register the new class in the `TRANSPORTS` registry (`luca/client/transports/__init__.py`).
3. Register the new provider in the `PROVIDERS` registry (`luca/client/providers/__init__.py`).

Do **not** add `if self._provider == "…"` branches inside `OpenAITransport`. See `architecture.md` §6.3 for a worked example.

### Add a new content block type

1. Add the Pydantic class in `luca/client/types/content.py`.
2. Extend the `ContentBlock` union.
3. Update each transport's `_project_messages` to handle the new type on send.
4. Update each transport's response parser to produce the new type on receive.

Tests go in `tests/client/test_types/test_content_blocks.py`.

## When in doubt

- **DTO shapes, kwargs, return types, error semantics** → `api_prd.md`
- **Layer responsibilities, where logic belongs** → `architecture.md`
- **What V1 implements vs. the full spec** → `docs/client/13-roadmap.md`
- **Provider/transport registration patterns** → `docs/client/09-providers-and-transports.md`
