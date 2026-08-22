# Agent integration of native web tools — PRD / design state

Status: **design settled** (2026-08-21, rev 7 — resolution log in §5; no
open items). This document is
self-contained: it assumes no knowledge of the codebase or of the
conversation that produced it. It records what we are trying to do, what has
been decided (with rationale), what is drafted but not yet argued, and what
remains open.

---

## 1. Background

### 1.1 The project, in two layers

`luca` is an AI agent framework with a sharp two-layer split:

- **`luca.client`** — a thin, unified LLM SDK (one API across OpenAI,
  Anthropic, OpenRouter, …). It owns everything wire-shaped: transports,
  payload building, streaming. Deliberately minimal and stable.
- **`luca.agent`** — the product. Its central artifact is a single
  serializable `AgentSession`: a flat, append-only store of **entries**
  (`UserMessage`, `AssistantMessage`, `ToolExecution`, turn markers, …)
  addressed by id, with each `Conversation` holding an ordered path of entry
  ids. The wire payload is never stored — a `ConversationProjector`
  recomputes the LLM message list from the path on every call, and the
  transport builds the actual request from that. Extensibility comes from
  duck-typed **middleware hooks** (13 today) and contrib **plugins**; the
  core stays provider-blind.

Two existing mechanisms matter for this design:

- **Provenance-gated thinking replay.** Assistant entries store the model's
  reasoning (`ThinkingContent`) with provider-opaque attestations. The
  projector re-emits them with `provider`/`model` provenance stamped on the
  message; the **transport** decides whether to replay them (an attestation
  minted by one provider is refused by every other). This is what makes
  mid-session model switching safe today.
- **Private tools.** `ToolSpec.is_private=True` marks a tool that is never
  advertised to the model and whose executions never project onto the wire
  (`project_private_execution` renders nothing). The subagents feature
  already mints such executions at runtime (`CreateConversationResult`), so
  "a `ToolExecution` with no corresponding tool-call block on the path" is
  an established, wire-legal shape.

### 1.2 What the client just shipped (commit `8e28e22`)

Hosted web tools are tools the **provider executes server-side, mid-response**.
The caller declares them; there is nothing to dispatch and no result to send
back. The client added:

**Declarations** (per provider, passed through `tools=`):

```python
# OpenAI — one tool covers search, page opening, find-in-page
from luca.client.providers.openai import WebSearchTool
WebSearchTool(search_context_size=..., filters=..., user_location=...,
              include_sources=..., include_results=...)

# Anthropic — search and fetch are separate tools
from luca.client.providers.anthropic import WebSearchTool, WebFetchTool
WebSearchTool(max_uses=..., allowed_domains=..., user_location=...)
WebFetchTool(max_content_tokens=..., citations=...)
```

**Content blocks — every operation is stored twice, side by side**, in
`AssistantMessage.content`:

```python
AssistantMessage(content=[
    PrivateProviderBlock(               # the exact wire item(s), for replay
        format="openai.responses",      # or "anthropic.messages", ...
        data={"id": "ws_01", "type": "web_search_call", ...},
    ),
    WebSearchBlock(                     # the portable meaning, for the caller
        queries=["Apple latest quarterly results"],
        results=[WebPagePart(url=..., title=..., content=...)],  # or None
    ),
    TextBlock(
        text="Apple's latest quarterly report says...",
        annotations=[URLCitationAnnotation(url=..., title=...,
                                           start_index=15, end_index=37)],
    ),
])
```

**The replay contract** (the part everything below leans on):

- `PrivateProviderBlock` is authoritative for replay **by the wire format
  that produced it** (`format`). The transport for that format replays it
  verbatim; **every other transport omits it — and never removes it**, so
  switching providers back and forth preserves exact replay.
- `WebSearchBlock` / `WebFetchBlock` are portable observations. **No
  transport ever sends them.**
- Cited text is normalized: Anthropic's split cited spans are merged into
  one `TextBlock` with derived ranges; the original split blocks are kept as
  `PrivateProviderBlock`s after it (their citations carry provider-only
  fields Anthropic wants back). Replay to Anthropic sends the privates and
  skips the merged block; every other provider gets the merged block.
- Anthropic search-result *content* is encrypted (readable only by the
  model, lives in the private block); a fetch's page text is readable.

**Streaming events**: `web_start`, `web_search` (queries),
`web_search_result`, `web_fetch`, `web_find`, `web_end`, `text_annotation`.

**Usage**: `Usage.tool_requests` (normalized `{"web_search": 2}`) +
`provider_tool_usage` (the provider's payload, verbatim).

### 1.3 The problem

The agent knows none of this. Today its inbound adapter
(`message_to_parts`) converts client blocks into agent parts and **silently
drops** anything it does not recognize — private blocks, web blocks, and
annotations would all vanish from the session. The integration must:

1. Let the agent search the web (declare the tools when the active model
   supports them).
2. Preserve everything durably so **switching provider/model mid-session
   never breaks** — the same conversation must keep replaying exactly on the
   original provider and cleanly on any other.
3. Give a reloaded session a **structural view** of what happened (not just
   opaque content blocks).

---

## 2. Design frame

A hosted web tool is **three things** to the agent — and *not* an ordinary
tool (no registry resolution, no approval, no dispatch, no result to send
back):

1. **A request feature** — declare it on the call when active.
2. **Assistant content** — store what came back, losslessly, in order.
3. **A synthetic structural index** — one private, terminal `ToolExecution`
   per operation, so the session records "a web search happened here" in the
   same vocabulary as every other action, surviving reload and provider
   switches.

---

## 3. DECIDED

### D1 — The agent stays naive about the wire; the client owns the filter

The agent stores the new blocks verbatim (as parts) and re-emits them
verbatim, in order. It never decides which private blocks reach the wire.

**Rationale.** The filter key is not `(provider, model)` — it is the **wire
format of the transport the call will actually use**. Example: the same
OpenAI model called through OpenRouter goes over chat-completions, not the
Responses API, so `format="openai.responses"` privates must be dropped there
too. Transport selection (`base_url` / `transport` resolution) is client
knowledge by design; the agent core explicitly treats those as "how the
client is reached". An agent-side filter would duplicate that rule — and the
client must keep its own filter regardless (it is a standalone SDK) — so
duplication would only add drift risk. This is also not a new kind of
naivety: it is the exact contract thinking replay already has (store with
provenance; transport decides).

**The switch walkthrough:**

```
session parts: [Private(fmt=openai.responses), WebSearchContent, Text(annotated)]

drive on openai:     projector emits all three as client blocks
                     → transport replays the Private VERBATIM, never sends the portable
drive on anthropic:  projector emits the same three
                     → transport OMITS the foreign-format Private, keeps the text
switch back:         nothing was ever deleted from the session → exact replay again
```

### D2 — New client API: `effective_messages`

"Effective" = the subset of a conversation that will actually be serialized
onto the wire for a given target, returned in client types instead of wire
dicts.

```python
def effective_messages(
    model: str,                      # same resolution as acompletion()
    messages: list[Message],
    *,
    provider: str | None = None,
    transport: str | None = None,
) -> list[Message]
```

**Contract:**

```
payload(messages, t)  ==  payload(effective(messages, t), t)     # same bytes
effective(effective(x, t), t)  ==  effective(x, t)               # idempotent
```

**The rules are exactly what transports already do at payload build:**

```
PrivateProviderBlock(format=f)   → keep iff f == transport.WIRE_FORMAT
WebSearchBlock | WebFetchBlock   → always drop (no transport sends them)
ThinkingBlock                    → keep iff attestation provenance accepted by target
merged cited TextBlock           → drop when its split privates replay in its place
everything else                  → keep;  non-assistant messages pass through
```

**No-drift constraint** — the selection logic is factored so there is ONE
copy of the rule:

```python
class Transport:
    WIRE_FORMAT: ClassVar[str]
    def select_assistant_blocks(self, m: AssistantMessage) -> list[ContentBlock]:
        """build_payload() serializes exactly these;
        effective_message() returns exactly these."""
```

**Scope and fine print (settled, seventh round).** The purpose is
measurement, for exactly one consumer: the `ContextManager` (D3). It is
never in the request path (transports keep filtering at payload build, as
today), and it exists to remove the *categorical* counting error — blocks
stored but never sent for the current target (the concrete failure: switch
away from Anthropic and the encrypted blobs would otherwise still count,
firing compaction spuriously). Counting stays the agent's; the invariant
above is kept strict not for consumers but because it forces both code
paths through `select_assistant_blocks`. With that scope:

- **A message whose blocks all drop is returned as
  `AssistantMessage(content=[])`, never omitted** (user decision):
  omitting would change the list shape, and if something needs to ride on
  the message later, dropping it would break the API. Token-wise the two
  are equivalent; the invariant still holds because transports skip an
  empty assistant message at build time (they must — an empty content
  array is a wire error anyway).
- The `WIRE_FORMAT` strings (`"openai.responses"`, `"anthropic.messages"`)
  stay transport-internal — documented as data (the values of
  `PrivateProviderBlock.format`), never exported as constants/enums.
  Nothing outside the client compares them (D1).
- Media down-conversion inside user messages is a documented non-goal:
  `effective` models message and assistant-block survival only; media is
  counted by flat constants regardless.

### D3 — Context accounting: the agent attributes, the client measures

The agent's `ContextManager` estimates per-entry `context_tokens`
(chars/4 heuristic today), provider-blind. Encrypted Anthropic payloads make
that wrong in one direction or the other after a switch. Fix: measure the
*effective* view, keep all attribution agent-side.

```python
# ContextManager.calculate_context — provider-aware now
projected = projector.project_entry(entry, entries)          # agent knows entry → messages
wire_view = effective_messages(cfg.model, projected, provider=cfg.provider)
entry.context_tokens = self._estimate_tokens(wire_view)      # agent's estimator, agent's ledger
```

- Attribution (which context belongs to a tool, a skill, the system prompt)
  is **structural knowledge the agent already holds** — the projector builds
  the message list entry by entry, tool declarations come from specs it
  resolved, each prompt part is contributed by a named plugin. The client is
  never asked to judge it; it only answers "what survives the wire".
- A model switch triggers `runner.recalculate_context_tokens()` (the door
  already exists).
- Any local count remains an estimate; the exact number stays the
  provider-reported input tokens in `AgentSession.usages`, which the context
  gauge keeps anchoring on.

### D4 — Synthetic executions: web operations get a durable structural record

Without this, a reloaded session shows web activity only as opaque content
blocks inside an assistant message. Decision: each web operation also mints
a `ToolExecution` entry — fake/synthetic, and honest about it:

```
TurnStart
AssistantMessage(parts=[Private, WebSearchContent, Text(annotated)])
ToolExecution(tool="web_search", private, terminal)        ← the structural index
TurnFinish
```

**Two rules keep it honest:**

1. **Single-source, never mirror.** The parts remain the truth (privates =
   replay, `WebSearchContent` = meaning). The execution carries a summary
   `result.content`, the portable payload as `structured_content`, and
   `extras={"message_entry_id": …}` — it never duplicates private data, so
   there is nothing to drift.
2. **An index, not a lifecycle.** Born terminal (`COMPLETED`, or `FAILED`
   from the provider's error), zero attempts, `approval_status=None`.

**Why it is safe:** the `is_private` + runtime-minted-execution shape already
exists (subagents' result executions) — private executions project as
nothing, so replay is untouched.

**Events:** synthetic executions are silent — no `ToolCallReceived` /
`ToolExecuted` fires for them (user decision, §5). Web operations are
rendered through the dedicated web events (P4); the synthetic is purely the
durable structural record.

### D5 — The minting door: middleware hook #14, `synthesize_executions`

Nothing today lets an extension append entries: executions are born from the
model's tool calls via the registry, and middleware transforms in-flight
values, never persists. That gap is generic (any future hosted tool, MCP
server-side operations, an app recording retrievals or redactions as
structural history), so the door is generic — **one new middleware hook**,
using the same template-in / runner-stamps-identity contract that
`prune_entry` already uses. Middleware proposes durable facts; the runner
remains the only writer.

```python
# core/middleware.py — hook #14
def synthesize_executions(
    self,
    session: AgentSession,
    conversation_id: str,
    entry: AssistantMessage,          # the COMMITTED entry — durable id available for linking
) -> list[ToolExecution]:
    """Executions to append after `entry`, derived from what the response
    carried. Templates: no id/created_at/conversation_id — the runner stamps
    them. Must be TERMINAL and resolve to a PRIVATE spec. Default: []."""
    return []
```

```python
# runner — right after _record_assistant commits the assistant entry:
templates = concat(mw.synthesize_executions(session, conversation_id, entry)
                   for mw in middleware)                  # middleware order, like every list hook
for t in templates:
    refuse if t.status in NONTERMINAL_STATUSES            # "never project a nonterminal
                                                          #  execution" stays unbreakable
    refuse if not (t.tool_spec and t.tool_spec.is_private)  # no ToolCall block exists on the
                                                          #  path → private is the only
                                                          #  wire-legal shape, by construction
    stamp(t)          # id=generate_id(), created_at=now_ms(), conversation_id
    normalize(t)      # tool_spec → session.tool_specs[spec_id()], as any spec
    ledger.append(conversation_id, t)                     # path: assistant entry, then these,
                                                          # in order, BEFORE tool-call births
    # NO tool lifecycle event fires here (user decision, §5): synthetics are
    # a silent structural record; web ops render through P4's web events
```

**Why middleware and not the registry:** no prepare/decide/dispatch
happened; forcing a recording through the registry's four-method lifecycle
would falsify it.

**Accepted trade-off:** a crash between the assistant commit and the
synthetic appends loses the synthetics. Acceptable — they are an index, the
truth stays in the parts, and a re-derivation utility can rebuild them.

### D6 — Pruning private payloads: whole-assistant-message pruning

The way to shed heavy web payloads from the wire is to prune the **whole
`AssistantMessage` entry** that carries them, using the pruning machinery
that already exists — not per-part surgery inside the message.

Background: pruning replaces one path node with a `PrunedEntry` (the
original stays in `entries`, untouched); the ledger door and
`project_pruned` already accept `AssistantMessage` referents (the
replacement projects under the original's role, with `provider`/`model`
provenance). Only the default `ContextManager.prune_entry` template policy
refuses non-`ToolExecution`s today.

```python
# a heavy Anthropic web-search assistant entry (thinking + 4 server_tool_use/
# result pairs, ~all bytes in encrypted_content blobs + a cited answer), pruned:
PrunedEntry(
    pruned_entry_type="assistant_message",
    pruned_entry_id="am1",
    content=[TextContent(text=
        "Searched the web: 'Apple stock price today' (9 results: cnn.com, "
        "tradingview.com, ...), 'NVIDIA stock price today' (7 results: ...), "
        "... Answered: <the final text, kept verbatim>")],
)
# projects as AssistantMessage(content=[TextBlock(...)], provider=..., model=...)
# original stays in entries; ~all the size (the encrypted_content blobs) leaves the wire
```

Why whole-message replacement is the right unit:

- **Always wire-legal.** Anthropic's `server_tool_use` / result pairs live
  *inside* the one message, so replacing the whole message removes both
  halves of every pair — the pair-legality question from the old "drop a
  private block" idea disappears.
- **No new machinery.** `PrunedEntry.content` stays `ContentPart`; ledger
  door and projection need nothing.

Policy rules for the replacement:

1. **Keep the final answer text verbatim** — the summary condenses only the
   search machinery (queries + result URLs), which is where the bytes are.
   Annotations drop with the original; acceptable.
2. **Only prune assistant messages with no client-executed `ToolCall`
   parts** — or prune their executions in the same pass. A replayed
   `ToolMessage` whose call vanished from the wire is a 400 on every
   provider. A pure web-search message has no such calls.
3. The D4 synthetic executions are separate path nodes that project as
   nothing, so they **survive the prune** — the structural record remains,
   and the summary text can be built from their `structured_content`.

**Scope boundary (user decision):** this PRD defines and implements the
prune *behavior* for such an entry — the template policy and the
replacement wording. **Triggering is out of scope**: nothing in the
framework triggers pruning for any entry type today (there is no framework
call site), and integrating a trigger into the agent is not this PRD's
responsibility.

### D7 — Configuration ergonomics: plugin constructor + `luca.json`

**The plugin** takes one JSON-shaped dict (or the typed config). The
per-provider `options` are not a schema we invent — they validate into the
client's own declaration classes, so the config surface IS the client tool,
single source:

```python
from luca.agent.contrib.websearch import WebSearchPlugin

WebSearchPlugin({
    "enabled": True,                       # default; False disables everything
    "openai": {
        "enabled": True,
        "options": {                       # ≡ luca.client.providers.openai.WebSearchTool
            "search_context_size": "high",
            "filters": {"allowed_domains": ["apple.com"]},
            "user_location": {"city": "Cordoba", "country": "AR"},
            "include_results": True,
        },
    },
    "anthropic": {                         # two client tools → search/fetch split
        "enabled": True,
        "search": {"max_uses": 5},         # ≡ anthropic WebSearchTool
        "fetch": {"max_content_tokens": 20_000},   # ≡ anthropic WebFetchTool; fetch is opt-in
    },
})
```

```python
# contrib/websearch/config.py — thin wrappers, extra="forbid"
class OpenAIWebConfig(BaseModel):
    enabled: bool = True
    options: OpenAIWebSearchTool = OpenAIWebSearchTool()

class AnthropicWebConfig(BaseModel):
    enabled: bool = True
    search: AnthropicWebSearchTool = AnthropicWebSearchTool()
    fetch: AnthropicWebFetchTool | None = None       # opt-in

class WebSearchConfig(BaseModel):
    enabled: bool = True
    openai: OpenAIWebConfig = OpenAIWebConfig()
    anthropic: AnthropicWebConfig = AnthropicWebConfig()
```

Semantics: instantiating the plugin = feature on, with provider defaults; a
provider block customizes or disables its side; at request time the
middleware still intersects with the per-model support table — an enabled
provider on an unsupported model advertises nothing.

**`luca.json`**: a top-level `websearch` feature block (the
`compaction`/`permissions` precedent — NOT `providers.<name>`, which
configures how models are called and feeds the persisted `LLMConfig`). The
block's value maps 1:1 onto the plugin constructor; the TUI wiring does
`WebSearchPlugin(config["websearch"])` and nothing else.

```jsonc
{
  "websearch": {
    "enabled": true,                 // flip to false to disable TEMPORARILY —
                                     // the rest of the block stays in place
    "openai":    { "options": { "search_context_size": "high" } },
    "anthropic": { "search": { "max_uses": 5 },
                   "fetch":  { "max_content_tokens": 20000 } }
  }
}
```

- **Absent block = off.** Deliberately breaking with `use_native_tools`'
  default-on: natives replace equivalent generic tools, while web search is
  a new capability with per-search cost and an injection surface — opt-in.
  `"websearch": {}` is the minimal enable; `--no-websearch` for CLI parity;
  the JSON schema gets the block so unknown keys keep failing loudly.
- Deep-merges home → project per key like every other block.
- **Nothing is persisted.** The config is runtime wiring, re-applied every
  launch — the `permissions` precedent. This closes the old "config home"
  open item: no new core session field, no `use_native_tools` overload;
  `AgentSession.extras` would come back only if a durable mid-session
  toggle is ever wanted (none is planned).

**Opinionated TUI defaults** (user decision — applied by the TUI wiring
under the user's block, per-key overridable; a library user instantiating
the plugin directly gets neutral client/provider defaults):

```python
TUI_DEFAULTS = {
    "openai":    {"options": {"include_results": True, "include_sources": True}},
    "anthropic": {"search": {"allowed_callers": ["direct"]},
                  "fetch":  {"allowed_callers": ["direct"]}},  # when fetch is enabled
}
# precedence per key: luca.json block > TUI_DEFAULTS > client/provider defaults
```

Non-goal for now: per-model option overrides (the `providers.<name>.models`
pattern) — per-provider options are enough until a real need shows up.

### D8 — Generic pause-and-replay: how `pause_turn` is handled

Anthropic's long-running server tools can end a response with
`stop_reason: "pause_turn"`; the contract is "re-send the recorded assistant
content as-is and let the model continue". The mechanism is GENERIC — the
runner understands "this stop condition means replay and continue", and
Anthropic's `pause_turn` is merely its first member.

**Client** — normalize to one generic reason (the anchor: `_classify_finish`
currently lets `pause_turn` fall through unnormalized):

```python
# anthropic _classify_finish gains one line:
if provider_value == "pause_turn":
    return ("pause", None)
# → finish_reason="pause", provider_finish_reason="pause_turn"
# any future provider condition meaning "replay and continue" maps to "pause"
```

**Runner-level config** — rides on the persisted `RuntimeConfig`, so
`luca.json`'s existing `runtime` block configures it for free:

```python
class RuntimeConfig:
    resume_finish_reasons: list[str] = ["pause"]   # normalized client vocabulary
# [] = feature off: a pause closes the turn COMPLETED, as today
```

**The drive predicate** — checked after recording the assistant entry,
before the close sites:

```python
if entry.stop_reason in config.resume_finish_reasons:
    emit(ResponsePaused(...))
    continue          # loop to the top: project → call the model again
# else: existing rules (tool calls → dispatch; nothing → close COMPLETED)
```

The replay needs ZERO special code: projection re-derives the history, the
path now ends with the paused assistant entry, its same-format privates
replay verbatim (D1), and a trailing assistant message is exactly the
provider's continuation shape. A pause round is a real model round —
`hard_max_steps` / `soft_max_steps` count it, which is also the
infinite-pause protection; an unconsumed cancel wins before the next call,
as everywhere.

**Data model (user-ratified): the marker is a field value, not an entry.**
`AssistantMessage.stop_reason` already exists (`"stop" | "tool_use"`) and
gains `"pause"` (normalized vocabulary). "Paused" is durably recorded on the
entry; "re-played" is the path signature — two assistant entries
back-to-back in one turn with nothing between, which occurs in no other
flow. No marker ENTRY: it would duplicate that information while needing its
own projection/status/compaction rules — turn structure is derived, not
stored. The durable field is also what makes this crash-safe for free:
reload after recording the paused entry → open turn, nothing nonterminal →
`BUSY` → the next `run()` reads `stop_reason` and continues.

**Events** — a simple lifecycle pair:

```python
class ResponsePaused(AgentEventBase):
    type: Literal["response_paused"] = "response_paused"
    finish_reason: str                      # "pause"
    provider_finish_reason: str | None      # "pause_turn"

class ResponseResumed(AgentEventBase):
    type: Literal["response_resumed"] = "response_resumed"
```

Named `Response*`, not `Turn*` — the logical turn never pauses (same
`TurnStart`/`TurnFinish` bracket throughout); the provider's RESPONSE did.
`Paused` fires right after the entry is persisted (events follow
persistence); `Resumed` fires on the continuation round immediately before
the LLM call. Two events because they can land in different drives: a lazy
consumer can exit after `Paused`, and after a crash/reload only `Resumed`
fires in the new process.

Fine print: pause and tool calls are mutually exclusive by provider contract
(a pause means the model did NOT yield for tool execution) — the predicate
runs first and the round is continuation-only. A user post landing exactly
in the pause window is the already-documented in-flight window; the
continuation request then simply ends with the user message instead, which
is legal.

### D9 — Synthetic execution shape: the specs and the executions

**Enabling client tweak** (part of this work): the portable blocks don't
carry the operation id or the provider's error today — both live only in
the adjacent private block. The transports stamp them into the portable
block's `extras` (`{"id": <op id>}`, plus `{"error": …}` when the provider
reported one) — one line each, and every consumer becomes adjacency-free.

**Spec granularity: one spec per operation kind, provider-independent.**
The *definition* ("a hosted web search: queries → results") is
provider-independent; which provider ran it is provenance — call-scoped,
living on the execution and the linked entry's `llm_config`. That is
exactly the split `spec_id()`'s purity rule wants: anything call-scoped in
the spec would mint a row per call.

```python
WEB_SEARCH_SPEC = ToolSpec(
    name="web_search",
    namespace="web",
    version="1",
    tool_kind="web",
    is_private=True,                     # required by D5
    timeout_in_ms=None,                  # never dispatched
    description=(
        "Provider-hosted web search, recorded for posterity by the websearch "
        "plugin. Never advertised and never dispatched: executions are "
        "synthesized from response content; the exact wire items live in the "
        "adjacent assistant message's private blocks."
    ),
    input_schema={                       # the synthetic call's arguments shape
        "type": "object",
        "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
        "required": ["queries"],
    },
    output_schema=WebSearchContent.model_json_schema(),   # = structured_content
    metadata={},                         # stays empty — definition-pure
)

WEB_FETCH_SPEC = ToolSpec(
    name="web_fetch", namespace="web", version="1", tool_kind="web", is_private=True,
    description="Provider-hosted page fetch (OpenAI open_page / Anthropic web_fetch), recorded ...",
    input_schema={"type": "object", "properties": {"url": {"type": "string"}},
                  "required": ["url"]},
    output_schema=WebFetchContent.model_json_schema(),
)
```

`queries` is plural on both providers (Anthropic = list of one), so the
argument shape is uniform.

**The COMPLETED execution** (from a real Anthropic example):

```python
ToolExecution(
    # runner stamps: id, created_at, conversation_id, updated_at,
    # finished_at = created_at (born terminal)
    tool_call_id="srvtoolu_01V6vsJNQSpfd8xCAiHDo8cd",     # ← the PROVIDER's op id
    raw_tool_call=ToolCall(
        id="srvtoolu_01V6vsJNQSpfd8xCAiHDo8cd",
        name="web_search",
        arguments={"queries": ["Apple stock price today"]},
    ),
    tool_spec=WEB_SEARCH_SPEC,           # runner normalizes → tool_spec_id
    status=ExecutionStatus.COMPLETED,
    approval_status=None,                # no decision ever processed it
    attempts=[],                         # never dispatched
    result=ExecutionResult(
        content=[TextContent(text=
            'Web search: "Apple stock price today" — 9 results '
            "(cnn.com, tradingview.com, coinbase.com, cnbc.com, finance.yahoo.com, +4 more)"
        )],
        structured_content=part.model_dump(),   # the WebSearchContent, verbatim
        is_error=False,
    ),
    extras={"websearch": {"message_entry_id": "am1"}},     # namespaced; single-source link
)
```

**op_id rule:** `tool_call_id = raw_tool_call.id =` the provider's
operation id (`srvtoolu_…`, `ws_…`), read from the part's `extras["id"]`;
`generate_id()` only if absent. No `provider` key in `extras` — derivable
from the linked entry's `llm_config`.

**Summary wording** — the plugin's `_summary`, overridable by subclass,
deterministic, one line:

```
search:  Web search: "<query>" — N results (host1, host2, …up to 5, +K more)
         results=None → 'Web search: "<query>" (result metadata not returned)'
         results=[]   → 'Web search: "<query>" — no results'
fetch:   Web fetch: <url> ("<title>", 18,203 chars)
failed:  Web search: "<query>" — failed: <error_code>
```

**The FAILED execution** — from the provider's own error report (Anthropic:
result content is an error object; OpenAI: `web_search_call.status ==
"failed"`):

```python
ToolExecution(
    tool_call_id="srvtoolu_01Xx...",
    raw_tool_call=ToolCall(id="srvtoolu_01Xx...", name="web_search",
                           arguments={"queries": ["..."]}),
    tool_spec=WEB_SEARCH_SPEC,
    status=ExecutionStatus.FAILED,
    error=ToolExecutionError(
        error_type="web_operation_error",
        error_message="max_uses_exceeded",                 # the provider's code/message
        details={"provider_error": {"type": "web_search_tool_result_error",
                                    "error_code": "max_uses_exceeded"}},
    ),
    extras={"websearch": {"message_entry_id": "am1"}},
)
```

**Status mapping is two-state:** provider says completed → `COMPLETED`;
provider reports an operation error → `FAILED`. Nothing else — never a
nonterminal, never `CANCELLED`/`REFUSED`. Fetch's COMPLETED twin is
symmetric (`arguments={"url": …}`, `structured_content=WebFetchContent`,
summary line as above).

---

## 4. PROPOSED — drafted, not yet argued

### P1 — Core data model: agent parts mirroring the client blocks

Agent-owned duplicates (the house pattern — the agent already duplicates
`MediaSource` rather than importing client types into the session schema):

```python
# core/models.py
class URLCitation(BaseModel):            # mirrors client URLCitationAnnotation
    url: str
    title: str
    start_index: int | None = None
    end_index: int | None = None

class WebPageContent(BaseModel):
    url: str
    title: str | None = None
    content: str | None = None
    extras: dict = Field(default_factory=dict)

class TextContent(BaseModel):
    ...
    annotations: list[URLCitation] = Field(default_factory=list)   # NEW

class PrivateProviderContent(BaseModel):
    """Verbatim wire item(s). Byte-for-byte — middleware never rewrites it
    (same doctrine as ThinkingContent)."""
    type: Literal["private_provider"] = "private_provider"
    format: str            # "openai.responses" | "anthropic.messages" | ...
    data: dict

class WebSearchContent(BaseModel):
    type: Literal["web_search"] = "web_search"
    queries: list[str]
    results: list[WebPageContent] | None = None    # None = not requested; [] = empty
    extras: dict = Field(default_factory=dict)

class WebFetchContent(BaseModel):
    type: Literal["web_fetch"] = "web_fetch"
    web_page: WebPageContent
    extras: dict = Field(default_factory=dict)

AssistantContentPart = Annotated[
    TextContent | ThinkingContent | ToolCall
    | PrivateProviderContent | WebSearchContent | WebFetchContent,
    Field(discriminator="type"),
]
```

### P2 — The round-trip code

```python
# core/adapter.py — message_to_parts(): client block → part, 1:1, ORDER
# PRESERVED (the client's positional contracts — private-before-portable on
# OpenAI, split-citation privates AFTER the merged TextBlock on Anthropic —
# ride through the session untouched)
elif block.type == "private_provider":
    parts.append(PrivateProviderContent(format=block.format, data=block.data))
elif block.type == "web_search":
    parts.append(WebSearchContent(queries=block.queries, results=..., extras=block.extras))
elif block.type == "web_fetch":
    parts.append(WebFetchContent(web_page=..., extras=block.extras))
# plus TextBlock.annotations → TextContent.annotations

# core/projection.py — project_assistant_message(): exact inverses, back to
# client blocks, provenance stamped as today
```

Implementation note: while adding the branches, the adapter's unknown-block
fall-through changes from silently skipping to raising, so the next client
block type cannot leak out of the session unnoticed.

### P3 — Advertisement + the first D5 consumer: `contrib/websearch/`

The shell-native plugin pattern, minus the slots web does not need (no
adoption, no call upgrade, no result synthesis):

```python
class WebSearchMiddleware:
    def adapt_tool_declarations(self, session, conversation_id, tools):
        """APPEND the provider's client-native declarations when the ACTIVE
        (provider, model) pair supports them. No ToolSpec behind them."""
        if not self._active(session.llm_config):        # config (D7) ∩ per-model support table
            return tools
        return tools + self._declarations(session.llm_config)
        # openai    → [config.openai.options]           # the declaration IS the config (D7)
        # anthropic → [config.anthropic.search, config.anthropic.fetch?]

    def synthesize_executions(self, session, conversation_id, entry):
        out = []
        for part in entry.parts:
            if isinstance(part, WebSearchContent):
                out.append(ToolExecution(
                    status=ExecutionStatus.COMPLETED,          # or FAILED, from private data
                    raw_tool_call=ToolCall(id=op_id(part), name="web_search",
                                           arguments={"queries": part.queries}),
                    tool_spec=WEB_SEARCH_SPEC,                 # the D9 spec
                    result=ExecutionResult(
                        content=[TextContent(text=self._summary(part))],   # posterity view
                        structured_content=part.model_dump(),              # typed payload
                    ),
                    extras={"message_entry_id": entry.id},
                ))
            # same for WebFetchContent
        return out

class WebSearchPlugin(BasePlugin):
    def __init__(self, config: dict | WebSearchConfig | None = None) -> None:
        self.config = WebSearchConfig.model_validate(config or {})   # see D7

    def get_middleware(self, session):
        return [WebSearchMiddleware(session, self.config)]
```

### P4 — Events and usage

Web operations get their **own event vocabulary, on both tiers** (user
decision, §5 — the synthetic executions fire no tool lifecycle events):

- **Always tier**, derived from the recorded parts: `WebSearchBlock`,
  `WebFetchBlock`.
- **Streaming tier** (mirrored in the runner's `_to_delta_event`, like
  text/thinking deltas): `WebOperationStart`, `WebSearchQueries`,
  `WebSearchResults`, `WebOperationEnd`, `TextAnnotation`.
- **Agent `Usage`** grows the client's normalized counter:
  `tool_requests: dict[str, int]` (e.g. `{"web_search": 2}`).

---

## 5. Resolution log (user decisions, 2026-08-20)

- **Events for synthetics.** Synthetic executions fire NO tool lifecycle
  events (`ToolCallReceived` / `ToolExecuted`) — they are a silent
  structural record. Web operations are rendered through the dedicated web
  events (P4), on both tiers.
- **Meaning loss after a provider switch: not an issue for now.** (Replay is
  safe by construction; the new provider sees the grounded text with no
  trace of the searches. If that ever matters, the adaptation door already
  exists: projector subclass / plugin `before_llm_call`.)
- **No approval gate: not an issue.** The provider executes mid-sampling, so
  the permission pipeline cannot intercept a search; advertisement on/off
  plus the declaration's own limits (`allowed_domains`, `max_uses`) are the
  control surface. Document it.
- **The former "shedding private payloads" item was conflating three
  separate points** (disk size / pruning / compaction). Split into separate
  items. Not framed as a problem: pruning doesn't cover private
  blocks today simply because they are a new feature.
- **No inline compression of encrypted payloads.** Investigated and
  rejected: the payloads are ciphertext, so gzip cannot compress them — it
  only reclaims the base64 packing, and re-encoding the result to be
  JSON-safe adds the packing right back. Measured on 100KB of random bytes:
  b64 133,336 → gzip+b64 134,676 (101% — slightly worse than doing
  nothing). Wire and context cost are unaffected by storage encoding in any
  case (verbatim replay; the provider counts the tokens). If disk size ever
  matters, the option is compressing the whole session file at rest — an
  application/persistence concern with zero data-model impact.
- **Pruning private payloads → decided as D6** (second review round):
  whole-assistant-message pruning through the existing machinery, behavior
  only — triggering/integration explicitly out of this PRD's scope.
- **Configuration ergonomics → decided as D7** (third round): plugin
  constructor dict + top-level `websearch` block in `luca.json` with its
  own `enabled` (temporary disable without deleting the block), opinionated
  TUI defaults (OpenAI: `include_results` + `include_sources`; Anthropic:
  `allowed_callers=["direct"]`). Also closes the old "config home" item:
  runtime wiring re-applied per launch, nothing persisted, no `extras` use.
- **`pause_turn` → decided as D8** (fourth round): generic pause-and-replay
  keyed on the normalized `"pause"` finish reason,
  `RuntimeConfig.resume_finish_reasons`, `AssistantMessage.stop_reason`
  as the durable marker (a field value, NOT a marker entry — user-ratified),
  `ResponsePaused`/`ResponseResumed` events.
- **Execution shape → decided as D9** (fifth round, 2026-08-21): one
  provider-independent spec per operation kind (`web:web_search`,
  `web:web_fetch`), `tool_call_id` = the provider's op id via
  client-stamped block `extras`, deterministic one-line summaries,
  two-state COMPLETED/FAILED mapping, namespaced execution `extras`.
- **Sixth round (2026-08-21) — four items discussed and closed explicitly,
  so they are not reopened later:**
  - *Disk size of sessions*: we don't care for now. Nothing to do. (Private
    blocks live in the append-only session file forever; accepted.
    File-at-rest compression stays a future option if it ever matters.)
  - *Compaction × private blocks*: nothing to do. A compacted path stops
    projecting the old assistant entries, so their privates fall off the
    wire with them.
  - *Generic fallback tool*: there is none, and none is planned. If the
    provider is not OpenAI or Anthropic, OR the tool is disabled, the agent
    simply has no web tool available. No in-tree fallback, no
    `REPLACES`-style hook for app tools.
  - *Support table scope*: nothing to do. Supported = OpenAI and Anthropic
    direct only; no OpenRouter passthrough, no other host.
- **Seventh round (2026-08-21) — `effective_messages` fine print settled
  (folded into D2):** purpose re-stated as measurement-only for the
  `ContextManager`; a fully-filtered message returns as
  `AssistantMessage(content=[])` rather than being omitted (API stability —
  user decision); `WIRE_FORMAT` strings stay internal, documented as data;
  media down-conversion is a non-goal. No open items remain.

## 6. OPEN

Nothing. Every item raised in this design discussion is either decided
(D1–D9, P1–P4 as drafted) or explicitly closed in the §5 resolution log.

Related doctrine to state on the field docstrings when P1 lands: the
byte-for-byte rule extends to annotated/cited text — Anthropic's split
privates are index-linked to the merged `TextContent`, so middleware
rewriting cited text desyncs the ranges (uncontested).

---

## 7. Suggested build order

1. **D2** — `effective_messages` in the client (self-contained, testable
   against existing transports).
2. **P1 + P2** — core parts + adapter/projector round-trip (incl. the loud
   adapter fall-through).
3. **D5** — the `synthesize_executions` hook + runner minting site.
4. **D3** — provider-aware context accounting.
5. **P3 + P4 + D7 + D9** — the `contrib/websearch` plugin with its config
   surface and the D9 specs/synthesis (incl. the client extras-stamping
   tweak), the `luca.json` block + TUI wiring/defaults, streaming events,
   usage.
6. **D6** — the assistant-message prune behavior (template policy +
   replacement wording; no trigger). Any time after P1 + P2.
7. **D8** — pause-and-replay (client `"pause"` mapping, the runner
   predicate + `resume_finish_reasons`, the event pair) before
   default-enabling the Anthropic tools.
