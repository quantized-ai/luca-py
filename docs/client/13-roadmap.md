# Roadmap and Scope

The current code is **V1** of `luca.client`. The full public-API
spec in [`api_prd.md`](../../api_prd.md) describes a larger surface;
[`architecture.md`](../../architecture.md) describes the internal design that
realizes it.

This page maps what V1 actually ships against what the spec covers, so
nobody has to grep to find out whether a feature is real.

## In V1 (implemented and tested)

**Chat completion** — the entire surface.

- `completion`, `acompletion`, `completion_stream`, `acompletion_stream`,
  `get_provider`.
- Content blocks: `TextBlock`, `ImageBlock`, `AudioBlock`, `FileBlock`,
  `ThinkingBlock`, `ToolCall`, `ToolResultBlock`, `RefusalBlock` — defined.
  ImageBlock with URL + base64 sources fully exercised; audio/file/pdf
  blocks defined but per-transport projection has thin coverage outside
  ImageBlock.
- Tools: dict / Pydantic `BaseModel` / `TypeAdapter` parameter forms; tool
  choice (incl. the provider-agnostic `{"name": ...}` forcing form);
  parallel tool calls (where the host supports it); provider-native tools
  (OpenAI apply_patch + local shell, Anthropic text editor + bash) on the
  per-tool projector mechanism — see [06-tools.md](06-tools.md).
- Structured output: `response_format=` + `response.parse()` /
  `FinishEvent.parse()` with the same three input styles, projected onto the
  provider's own field (OpenAI `text.format` / `response_format`, Anthropic
  `output_config.format`) and strictified on the way out. Bedrock refuses it —
  Converse has no equivalent. See
  [07-structured-output.md](07-structured-output.md).
- Reasoning: `ThinkingBlock`, `reasoning=`, signature and item-id
  preservation per-transport, including replay across a mid-session model
  switch (a foreign attestation is dropped rather than sent).
- Streaming: full `StreamEvent` union, live accessors on the stream object
  (`s.message`, `s.text`, …), cancellation as `FinishEvent(cancelled=True)`,
  terminal `FinishEvent` / `ErrorEvent` split, `ResourceWarning` safety net.
- Typed error hierarchy under `ClientError`.
- Catalog: `catalog.get`, `catalog.list`, `catalog.register`. Ships with a
  curated set of records.
- `provider_options=` and `model_info=` escape hatches on every request.

**Providers**: `OpenAIProvider`, `AnthropicProvider`, `OpenRouterProvider`,
`BedrockProvider`, `FauxProvider`, plus `GenericProvider`-backed entries for
`groq`, `deepseek`, `ollama`, `quantized`.

**Transports**: `OpenAIResponsesTransport` (provider `openai`),
`OpenAITransport` (chat completions — every OpenAI-compatible host),
`AnthropicTransport`, `OpenRouterTransport`, `BedrockTransport`,
`FauxTransport`.

**Testing**: `luca.client.testing` — `FauxProvider`,
`FauxTransport`, scripted-response builders.

## Not in V1 (deferred — `api_prd.md` describes these)

**Other product surfaces** (whole sections of `api_prd.md`):

- **Embeddings** — `embedding` / `aembedding`, `EmbeddingRequest` /
  `EmbeddingResponse`.
- **Image Generation** — `image` / `aimage` / `image_stream` /
  `aimage_stream`, `ImageRequest` / `ImageResponse`.
- **Audio** — `transcribe`, `speech`, request/response models.

The provider/transport layering already has placeholders for the surface
mixins (see `architecture.md` §4.2 — `EmbeddingMixin`,
`ImageGenerationMixin`, `TranscriptionMixin`, `SpeechMixin`). Adding a new
surface means: (a) define the request/response DTOs, (b) add the surface
mixin pair (provider + transport), (c) override the hook methods on the
relevant transport classes.

**Providers** present in `api_prd.md` / `architecture.md` but not yet
registered in `PROVIDERS`:

- `together`, `cerebras`, `fireworks`, `xai`, `parasail`, `mistral`,
  Vertex / Gemini, Azure.

For OpenAI-compatible hosts, "register" is a one-line dict entry. For
Vertex it's a new transport class (different auth scheme, very different
wire shape) — Bedrock is the worked example of that path, now shipped.

**Agent module** — `luca.agent` is a sibling submodule scheduled
for after the LLM client stabilizes.

**Operational features explicitly out of scope**:

- Server / proxy mode (this is an SDK, not LiteLLM).
- Guardrails / moderation pipelines.
- Automatic multi-model fallback.
- Automatic retries.
- Cross-provider message rewriting (you switch providers, you own the
  cleanup).
- Batch APIs.
- Re-ranking.

## Where to find the boundary

- `luca/client/__init__.py` lists the exact public re-exports.
- `luca/client/types/__init__.py` lists every public DTO.
- `luca/client/providers/__init__.py` lists every registered
  provider via the `PROVIDERS` dict.
- `luca/client/transports/__init__.py` lists every transport.

Anything outside those lists isn't in V1.
