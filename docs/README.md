# `luca` — Documentation

`luca` (PyPI: `luca-ai`) is an AI agent framework in two layers. The docs are
split the same way:

| Folder | Layer | Start here |
|---|---|---|
| [**`agent/`**](agent/README.md) | The agent framework — durable `AgentSession`, the async runner, tools, permissions, prompts, middleware. **The primary product.** | [`agent/README.md`](agent/README.md) |
| [**`client/`**](client/README.md) | The supporting LLM SDK — one typed API (`completion`/`acompletion`) across OpenAI, Anthropic, OpenRouter, … Only deps: `httpx` + `pydantic`. | [`client/README.md`](client/README.md) |

Most application work lives in the **agent** layer; it uses the **client**
underneath to talk to models. If you're building an agent, start in
[`agent/`](agent/README.md). If you only need a thin multi-provider chat-completion
SDK, start in [`client/`](client/README.md).

Both layers install as the single `luca-ai` package — see
[`client/01-installation.md`](client/01-installation.md).
