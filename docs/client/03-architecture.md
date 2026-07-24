# Architecture (at a glance)

`luca.client` is layered so that vendor differences live in exactly
one place: a per-vendor **transport** class. Above that, everything is
canonical and provider-agnostic.

```
┌──────────────────────────────────────────────────────────┐
│  Helpers     (luca/client/_client.py)           │  ← most users
│  completion / acompletion                                │
│  completion_stream / acompletion_stream                  │
│  get_provider                                            │
└──────────────────────────────────────────────────────────┘
                       │ parses model string, builds DTO,
                       │ resolves provider, dispatches
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Providers   (luca/client/providers/)           │  ← power users
│  OpenAIProvider, AnthropicProvider, OpenRouterProvider,  │
│  BedrockProvider, GenericProvider, FauxProvider          │
│  • holds default base_url, env var, transport class      │
│  • forwards .completion / .*_stream to the transport     │
└──────────────────────────────────────────────────────────┘
                       │ self._transport.completion(request)
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Transports  (luca/client/transports/)          │  ← advanced
│  OpenAITransport, AnthropicTransport, OpenRouterTransport│
│  BedrockTransport, FauxTransport                         │
│  • builds wire JSON  • parses SSE/JSON stream            │
│  • maps errors to typed ClientError subclasses           │
└──────────────────────────────────────────────────────────┘
                       │ httpx.Client / httpx.AsyncClient
                       ▼
              Upstream HTTP API
```

## Three rules to keep in mind

1. **Provider is mandatory.** Either via prefix (`openai:gpt-4o`) or via
   `provider="openai"`. The SDK never guesses a provider from a bare model
   name.
2. **One canonical DTO per layer.** The helper builds a
   `ChatCompletionRequest` exactly once; the provider and transport both
   receive that same DTO. There's no intermediate "wire-prep" object.
3. **The catalog lives behind one door — the helper.** Providers and
   transports never import `luca.client.catalog`. Anything they need
   from it arrives on `request.model_info`.

## Provider vs Transport

- A **provider** is a vendor identity: a host name (`"openai"`,
  `"groq"`), a default base URL, an env var for the API key, and a default
  transport class. It owns a transport instance.
- A **transport** is a wire protocol: it builds payloads, parses the response
  stream (SSE for most transports, a binary `vnd.amazon.eventstream` for
  Bedrock), and maps HTTP errors. The same transport class can serve many
  providers — `OpenAITransport` is the wire format for OpenAI, Groq,
  DeepSeek, Together, Cerebras, Fireworks, Ollama, … via `PROVIDERS`
  config-dict entries that point at it.

So adding **Groq** is a one-line `PROVIDERS` entry (it reuses
`OpenAITransport`); adding **Anthropic** is its own transport class because
the wire shape differs enough that subclassing would mean overriding
everything.

## Where each kind of edit goes

| Task | Where to edit |
|---|---|
| Add a new OpenAI-compatible host | One line in `PROVIDERS` (`providers/__init__.py`) |
| Add a host with custom auth/wire shape | New file in `providers/` + new folder in `transports/` |
| Tweak how OpenAI builds payloads | `transports/openai/transport.py` |
| Add a new content block type | `types/content.py` + transport projections |
| Add a new finish-reason mapping | `_classify_finish` on the relevant transport |

For the long-form rationale (the polymorphism-not-flags rule, the
per-vendor quirks discussion, the streaming subsystem layering) see the
internal [`architecture.md`](../../architecture.md) at the repo root.
