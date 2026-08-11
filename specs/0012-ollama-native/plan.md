# 0012 — implementation plan

Execution order for the PRD. Each step leaves `uv run py.test tests/` green,
then `uv run ruff check --fix && uv run ruff format`. No new runtime
dependencies.

**Standing gate:** the suite must pass with the Ollama daemon STOPPED. Every
hermetic test uses `httpx.MockTransport`; only the `live`-marked ones touch a
daemon, and those skip when `/api/version` does not answer.

Branches from `bedrock-sigv4-and-env-loading`.

---

## Step 1 — discovery

**File**: `luca/client/transports/ollama/discovery.py` (new)

- `model_info_from_show(name, payload, *, ceiling) -> ModelInfo | None` — pure,
  no I/O. `None` for a model without the `completion` capability.
- `discover(base_url, *, ceiling, timeout) -> list[ModelInfo]` — `/api/tags`
  then `/api/show` per model.
- Imports `types.catalog` and httpx only. **Never** `luca.client.catalog`:
  transports do not import the catalog, so this returns data and the caller
  registers it.

**Tests** (`tests/client/test_transports/test_ollama/test_discovery.py`): the
four real `/api/show` payloads captured from a live daemon as fixtures —
normal + tools (131072, gets capped), normal + tools (32768, under the cap),
embedding-only (skipped), and `thinking` with an entirely empty `model_info`
(falls back to 8192). Plus `/api/tags` → the full list, and a daemon that is
down raising `ConnectionError`.

## Step 2 — the transport

**Files**: `luca/client/transports/ollama/{__init__,transport,stream}.py` (new),
`luca/client/transports/__init__.py`

- `_chat_completion_url` → `{base_url}/api/chat` for both operations; the
  streaming variant differs by the `stream` field, not the path.
- Payload: `model`, `messages`, `tools`, `stream`, `think`, and `options`
  carrying `num_ctx`, `temperature`, `top_p`, `num_predict`.
- `num_ctx` from `request.model_info.context_window`, capped;
  `provider_options["options"]["num_ctx"]` overrides.
- System prompt is a `role: "system"` message, not a top-level field.
- `OllamaToolProjector` bound to `TOOL_PROJECTOR_BASE`. Tool arguments are a
  real JSON object on this wire, like Bedrock Converse and unlike OpenAI.
- `stream.py`: NDJSON line parser. `done: true` terminates; the final frame
  carries `prompt_eval_count` / `eval_count` → `Usage`.
- Errors: 404 → `ModelNotFoundError` naming `ollama pull <model>`;
  `httpx.ConnectError` → `ConnectionError` naming the daemon and the base URL.
- Register in `TRANSPORTS`.

**Tests**: `test_payload_building.py` (num_ctx source, capping, override,
system role, tool projection, `think` only when the model supports it),
`test_stream.py` (one line per read, several per read, a line split across
reads, a trailing partial, usage off the `done` frame), `test_errors.py`.

## Step 3 — the provider

**Files**: `luca/client/providers/ollama.py` (new),
`luca/client/providers/__init__.py`

First-class, replacing the dict entry — it needs somewhere to hold the
`num_ctx` ceiling, the same reason `BedrockProvider` is a class.
`default_api_key_env_var` stays `None`.

**Tests** (`tests/client/test_providers/test_ollama_provider.py`): defaults,
an explicit `base_url`, the ceiling, and no auth header on the wire.

## Step 4 — agent wiring

**Files**: `luca/agent/contrib/tui/cli.py`, `.../config.py`

Run discovery at boot when the session provider is `ollama`, register each
result, log and continue when the daemon is down. The existing picker and
compaction paths then work unchanged; a `models` list in `luca.json` still
unions on top.

**Tests**: discovery results reach the catalog and the `/model` picker; a dead
daemon still boots.

## Step 5 — live tests

**File**: `tests/client/test_transports/test_ollama/test_live.py` (new)

Marked `live`, skipped when `/api/version` does not answer. Free and fast,
unlike Bedrock's.

1. **`num_ctx` is honored** — request a window, then assert `/api/ps` reports
   it. The direct regression test for this whole change.
2. Discovery against the real daemon returns registrable `ModelInfo`.
3. A tool round trip on `llama3.2:latest`.
4. A prompt larger than the window does not silently truncate below it.

## Step 6 — docs

`docs/client/09-providers-and-transports.md` (the ollama row now names
`OllamaTransport` and `/api/chat`), `10-catalog.md` (a provider can register
its own models; the catalog is no longer models.dev-only), `AGENTS.client.md`
(the `model_info`-to-wire rule and the layout tree).
