# Local Ollama on its native API

## Why

`ollama` is already a `PROVIDERS` dict entry pointing `OpenAITransport` at
`http://localhost:11434/v1`, and it works: completion, streaming and tool
calling were all verified live against a local daemon (v0.32.6).

The problem is not the wire, it is the context window.

`catalog.get("ollama", …)` returns `None` — `_source.py` excludes ollama
deliberately, because models.dev cannot know what you pulled. So
`get_context_window_size` falls back to `DEFAULT_WINDOW = 200_000` and
compaction waits for 160k tokens. Measured on one machine, for one model:

| number | source |
|---|---|
| 32768 | `qwen2.context_length` in `/api/show` — the architectural max |
| 4096 | `context_length` in `/api/ps` — what Ollama actually loaded |
| 200000 | what luca believes |

Ollama truncates past its window in silence. A ~4000-token prompt sent with
`num_ctx: 512` reported `prompt_eval_count: 258`, with no `truncated` flag and
no warning field anywhere in the response. The agent forgets, nothing says so,
and the compaction trigger is never reached.

**The OpenAI-compatible endpoint cannot fix this.** Measured: `/v1/chat/completions`
carrying `options.num_ctx: 16384` left the loaded window at 4096. The same
options on `/api/chat` moved it to 16384. That is the whole argument for a
native transport.

## Objectives

1. Talk to `/api/chat` so `num_ctx` can be set.
2. Discover local models and their real facts, and register them in the
   catalog so the window luca reports is the window Ollama runs.
3. Make `/model` list what is actually pulled, with real context figures.
4. Say something useful when the daemon is not running.

## Non-objectives

Embeddings (`/api/embed`), pulling models from inside luca, `keep_alive`
tuning, image input, and provider-native tools — Ollama has none, and
`supported_native_tools` already returns nothing for it, correctly.

## The loop that keeps the window honest

1. `/api/tags` lists local models; `/api/show` per model gives `capabilities`
   and `<family>.context_length`.
2. Those map to `ModelInfo` and go in via `catalog.register(...)` — an
   existing public API that nothing in `luca/` currently calls.
3. The helper already stamps `request.model_info` from the catalog.
4. `OllamaTransport` reads `model_info.context_window` and sends it as
   `options.num_ctx`.

The number luca reports to the compactor and the number Ollama runs are the
same number, because one produced the other. They cannot drift.

This makes `OllamaTransport` the first transport to read a `model_info` field
other than `cost`, which `AGENTS.client.md` currently rules out. The rule
changes: for a local server the context window is a request parameter, not
metadata.

## num_ctx policy

`min(architectural context_length, ceiling)`, ceiling defaulting to 32768 and
overridable per model via `provider_options`.

Two reasons for a ceiling. `llama3.2` advertises 131072, and allocating that
on a laptop either fails to load or spills to CPU. And changing `num_ctx`
**forces a model reload** (measured), so the value has to be one stable number
per model rather than something recomputed per request.

When `/api/show` reports no `context_length` at all — which happens, one of
the four local models has an empty `model_info` — use a conservative 8192 and
register that. Because luca sets the value, it is true either way.

## What is discovered, and what is skipped

From `/api/show`:

| `ModelInfo` field | source |
|---|---|
| `context_window` | `min(<family>.context_length, ceiling)`, else 8192 |
| `supports_tools` | `"tools" in capabilities` |
| `supports_reasoning` | `"thinking" in capabilities` |
| `supports_image_input` | `"vision" in capabilities` |
| `family` | `details.family` |
| `cost` | none — local inference is free |

A model whose `capabilities` lack `completion` is not registered. The local
`nomic-embed-text` is `["embedding"]` only and is not a chat model, despite
carrying a `context_length` that would otherwise look valid.

## Failure behaviour

Discovery runs at boot, only when the session's provider is `ollama`, and is
never fatal: a daemon that is not running logs and the launch continues, the
same way a missing `auth.json` is not an error.

A dead daemon on an actual request currently surfaces as
`ConnectionError: [Errno 61] Connection refused`, naming neither Ollama nor
how to start it. It should name both. A 404 for a model that was never pulled
should say `ollama pull <model>`.
