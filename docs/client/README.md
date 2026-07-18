# `luca` — Documentation

`luca` is a thin, unified Python SDK for talking to multiple LLM
providers (OpenAI, Anthropic, OpenRouter, Groq, DeepSeek, Ollama, …) through a
single typed interface. Think of it as a smaller, simpler LiteLLM: no
guardrails, no proxy server, no cross-provider message rewriting — just one
canonical request DTO, one canonical response DTO, and per-vendor transport
classes that speak each wire format directly via `httpx`.

The SDK ships as the top-level package `luca`. The LLM client lives
under `luca.client`; a future `luca.agent` submodule will
host higher-level agent primitives.

## What's in this folder

Start at the top and walk down — each page builds on the previous ones.

| Page | Topic |
|---|---|
| [`01-installation.md`](01-installation.md) | Install, environment variables, `uv` workflow |
| [`02-quickstart.md`](02-quickstart.md) | First completion in sync, async, and streaming flavors |
| [`03-architecture.md`](03-architecture.md) | The three-layer model (helper → provider → transport) at a glance |
| [`04-chat-completion.md`](04-chat-completion.md) | `completion()` / `acompletion()` — every kwarg, return shape, finish reasons |
| [`05-messages-and-content.md`](05-messages-and-content.md) | `UserMessage` / `AssistantMessage` / `ToolMessage` and the content-block union |
| [`06-tools.md`](06-tools.md) | Defining tools, calling them in a loop, parsing arguments |
| [`07-structured-output.md`](07-structured-output.md) | `response_format=` with dicts, Pydantic models, and `TypeAdapter` |
| [`08-streaming.md`](08-streaming.md) | The `StreamEvent` union, terminal events, cancellation, `collect()` |
| [`09-providers-and-transports.md`](09-providers-and-transports.md) | Provider registry, `PROVIDERS` config, adding hosts, lower-level entry points |
| [`10-catalog.md`](10-catalog.md) | The model catalog, `ModelInfo`, cost computation |
| [`11-exceptions.md`](11-exceptions.md) | The `ClientError` hierarchy and what triggers each one |
| [`12-testing.md`](12-testing.md) | `FauxProvider` / `FauxTransport` + scripted-response builders |
| [`13-roadmap.md`](13-roadmap.md) | What V1 ships, what is deferred, where the full API spec lives |

The companion files in the repo root are the deeper, internal references:

- [`api_prd.md`](../../api_prd.md) — the **full** public API spec, including the
  surfaces (embeddings, image generation, audio) that V1 does **not** yet
  implement. Authoritative for DTO shapes.
- [`architecture.md`](../../architecture.md) — the **internal** design: how
  providers and transports are wired, the streaming subsystem, the
  finish-reason classification rules.
- [`testing_architecture.md`](../../testing_architecture.md) — design notes for
  the test suite (already implemented under `tests/`).

If you only read one file in this folder, read
[`02-quickstart.md`](02-quickstart.md). If you only read two, add
[`04-chat-completion.md`](04-chat-completion.md).
