# Provider-hosted web tools — `luca.agent.contrib.websearch`

The provider executes web search (and, on Anthropic, page fetch) SERVER-SIDE,
mid-response: the agent declares the tool, the model searches while sampling,
and the operations arrive as content — there is nothing to dispatch and no
result to send back. This plugin is the whole integration:

- **advertise** the hosted declarations when the ACTIVE (provider, model)
  pair supports them (config ∩ a per-model support table);
- **record** one durable, private, terminal `ToolExecution` per operation —
  the structural index that survives reload and provider switches. The truth
  stays in the assistant entry's parts; the synthetic never duplicates them.

```python
from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.websearch import WebSearchPlugin

runner = PluginAgentSessionRunner(session, plugins=[WebSearchPlugin({
    "openai":    {"options": {"search_context_size": "high"}},
    "anthropic": {"search": {"max_uses": 5},
                  "fetch":  {"max_content_tokens": 20_000}},   # fetch is opt-in
})])
```

In the TUI demo the same dict is `luca.json`'s top-level `websearch` block
(absent = off, `{}` = minimal enable; `--websearch` / `--no-websearch` —
[tui/config.md](../tui/config.md)).

## 1. The config IS the client declaration

The per-provider options are not a schema this plugin invents: they validate
into `luca.client`'s own tool classes
([client tools](../../../client/06-tools.md#hosted-web-tools)), so every knob
the provider documents is available under its own name.

| Key | Validates into | Notes |
|---|---|---|
| `enabled` | — | `false` disables everything, block kept in place |
| `openai.options` | `luca.client.providers.openai.WebSearchTool` | one tool covers search, open_page, find_in_page |
| `anthropic.search` | `luca.client.providers.anthropic.WebSearchTool` | |
| `anthropic.fetch` | `luca.client.providers.anthropic.WebFetchTool` \| absent | OPT-IN: no key, no fetch tool |

At request time the middleware intersects with the per-model support table
(`support.py`, a dated snapshot of the vendor model pages) — an enabled
provider on an unsupported model advertises nothing, and an unknown model id
simply has no web capability. The gate reads the ACTIVE `session.llm_config`
re-stamped every drive iteration, so a mid-session `/model` flip re-derives
the declarations on the very next call.

## 2. What a web operation leaves in the session

One response's search lands as parts + a synthetic execution:

```
AssistantMessage(parts=[
    PrivateProviderContent(format="anthropic.messages", data={…verbatim…}),   # replay
    WebSearchContent(queries=[…], results=[…], extras={"id": "srvtoolu_…"}),  # meaning
    TextContent(text="…", annotations=[URLCitation(…)]),                      # cited answer
])
ToolExecution(tool_spec=web:web_search, private, synthetic, terminal)         # the index
```

- The **private parts replay verbatim** on the producing wire and are omitted
  by every other ([02](../../02-data-model.md)) — switching models never
  breaks the conversation, and switching back restores exact replay.
- The **synthetic execution** (spec `web:web_search` / `web:web_fetch`,
  `is_private` + `is_synthetic`) is minted through middleware hook #14
  ([07](../../07-middleware.md)): `tool_call_id` = the provider's operation
  id, a one-line summary as `result.content`, the portable part as
  `structured_content`, `extras={"websearch": {"message_entry_id": …}}`.
  COMPLETED, or FAILED with the provider's own error. It fires NO tool
  lifecycle events — web activity renders through the dedicated web events
  ([04](../../04-runner.md) §4–5) — and never wakes a parked parent.
- **Usage** carries the provider's hosted-tool counters:
  `session.usages[cid][eid].tool_requests == {"web_search": 2}`.

Long-running Anthropic operations can pause the response mid-turn; the
runner's generic pause-and-replay continues it ([04](../../04-runner.md)
§14).

## 3. No approval gate — by design

The provider executes mid-sampling, so the permission pipeline cannot
intercept an operation. The control surface is advertisement on/off (the
config, the flag) plus the declaration's own limits — `allowed_domains`,
`blocked_domains`, `max_uses`, `max_content_tokens`. The TUI defaults also
pin `allowed_callers: ["direct"]` on Anthropic's tools.

## 4. Overriding the summaries

`WebSearchMiddleware._summary` is the one-line posterity wording
(`Web search: "query" — 9 results (cnn.com, …, +4 more)`), deterministic and
overridable by subclass; pass your subclass through your own plugin's
`get_middleware`. The default `ContextManager` can also prune a whole
web-heavy assistant entry down to summary + verbatim answer
([11](../../11-context-and-usage.md) §5).
