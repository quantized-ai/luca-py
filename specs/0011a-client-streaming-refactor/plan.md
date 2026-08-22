# Implementation plan — PRD 0011a

Companion to `prd.md` (the contract) and `test_audit.md` (the frozen/deleted
lists). The PoC is normative for shape, naming, and altitude; §2 below lists the
four places the real wires force a deviation, each verified against the code.

## 1. Module layout

```
luca/client/transports/
├── streamer.py                  NEW — BaseStreamer, SyncStreamerMixin, AsyncStreamerMixin
├── base.py                      wire-shared bits extracted into WireFormatMixin;
│                                ChatCompletionTransportMixin streaming methods → pure factory
├── openai/
│   ├── transport.py             payload/projection/classify moved into OpenAIWireMixin (same file)
│   ├── errors.py                unchanged (OpenAIErrorMappingMixin, already a mixin)
│   ├── streamer.py              NEW — OpenAIChatCompletionsStreamer + Sync/Async combos
│   └── stream.py                DELETED
├── openai_responses/            same pattern: OpenAIResponsesWireMixin, streamer.py, stream.py deleted
├── openrouter/
│   ├── transport.py             sets STREAMER/ASYNC_STREAMER
│   └── streamer.py              NEW — OpenRouterChatCompletionsStreamer(PROVIDER/URL only) + combos
├── anthropic/                   AnthropicWireMixin (error mapping moves into it from transport.py),
│                                streamer.py NEW, stream.py DELETED
├── bedrock/                     BedrockWireMixin, streamer.py NEW, stream.py DELETED
└── faux/
    ├── transport.py             keeps scripted queue, _pop, .requests, builders; stream classes DELETED
    └── streamer.py              NEW — FauxStreamer + combos

luca/client/types/streaming.py   trimmed to the public event models + StreamEvent union
luca/client/types/__init__.py    drop the Raw*/stream-class re-exports and __all__ rows
luca/client/_client.py           timeout unification (§4), total_timeout gone
luca/client/providers/base.py    forwarders gain timeout= pass-through
```

## 2. Deviations from the PoC — each forced by a verified wire fact

1. **`parse()` returns a `list` of raw events, not payload-or-None.** Bedrock is
   not SSE: one `iter_bytes` chunk can hold zero, one, or several CRC-framed
   events, and a frame can straddle chunks (bedrock/stream.py:75–93 today). A
   single-payload `parse` cannot express that. SSE wires return `[]` or
   `[payload]`; Bedrock buffers bytes and returns every completed frame. The
   mixin loop becomes `for raw in self.parse(chunk): events += self.handle(raw)`.
   Same altitude, same names, one plural.

2. **`handle_wire_end()` — `FinishEvent` is emitted at wire end, with a default
   premature-end error.** Verified: on the CC wire the usage chunk arrives
   *after* the `finish_reason` chunk (and may not arrive at all — finish must
   still be emitted, with `Usage()`); on Bedrock, `metadata` (usage) arrives
   after `messageStop`. The PoC's emit-finish-inside-a-handler cannot survive
   either wire; today's implementation already builds the terminal at wire end
   for exactly this reason. Shape: wires latch `stop_reason` in handlers and
   `BaseStreamer.handle_wire_end()` returns `[]` (→ the mixin emits the
   premature-end `ErrorEvent`); CC and Bedrock override it to return
   `[FinishEvent]` when a stop reason was latched. Anthropic and Responses keep
   emitting finish from their terminal handlers (their wires put usage inside
   the terminal event) and keep the default. CC's `[DONE]` line is wire
   knowledge: its `parse` maps it to a `{"done": true}` marker and its `handle`
   routes that to `handle_wire_end()`; source exhaustion is the fallback for
   servers that just close.

3. **`open_wire()` / `aopen_wire()` — the one factored seam in the mixins.**
   Each mixin's `__enter__` is: arm the deadline, `self.lines =
   self.open_wire()`, return self. The HTTP default (build_request → send →
   status check → error mapping → `iter_chunks(response)`) lives on the mixin;
   `FauxStreamer` overrides `open_wire`/`aopen_wire` to install its scripted
   source, so the faux combos stay empty and the deadline/iteration machinery is
   reused verbatim — which is what makes the async timeout tests run on faux.
   Bedrock overrides only `iter_chunks`/`aiter_chunks` (bytes, not lines). Both
   overrides have a concrete second case today; nothing else is factored out.

4. **The constructor gains `provider=`.** `OpenAITransport` serves DeepSeek and
   every generic OpenAI-compatible host; `message.provider` and `provider=` on
   raised errors carry the *instance* label ("deepseek"), not the wire class
   identity (verified: providers/base.py:63–64 constructs transports with
   `provider=self.name`). `provider=` is data, defaults to the class `PROVIDER`,
   and the factory passes `provider=self._provider`.

## 3. The classes

### 3.1 `BaseStreamer` (transports/streamer.py)

```python
class BaseStreamer:
    PROVIDER = None
    URL = None                       # default endpoint; explicit base_url wins
    HANDLERS_BY_TYPE = {}
    CONNECT_TIMEOUT = 5.0            # the one httpx phase kept alive (§4)

    def __init__(self, request, *, provider=None, httpx_client=None,
                 api_key=None, base_url=None, timeout=None):
        self._provider = provider or self.PROVIDER    # underscored: the wire
        self._api_key = api_key                       # mixins read these names
        self._base_url = base_url or self.URL
        self.request = request
        self.timeout = timeout
        self.client = httpx_client
        self.owns_client = httpx_client is None
        self.message = AssistantMessage(content=[], provider=self._provider,
                                        model=request.model)
        self.usage = None            # last Usage seen; rides ErrorEvent too
        self.stop_reason = None
        self.response = None
        self.lines = None            # None until entered → StreamError on iteration
        self.pending = []
        self.finished = False
        self.cancelled = False
        self.expired = False
```

Methods: `parse(chunk) -> list` (SSE default: `data:` prefix, `[DONE]` → `[]`,
json.loads with `StreamError("... non-JSON data ...")` on failure), `handle(raw)
-> list` (dispatch via `HANDLERS_BY_TYPE`), `handle_wire_end() -> list` (default
`[]`), `iter_chunks(response)` / `aiter_chunks(response)` (iter_lines/aiter_lines
defaults), `finish_event(cancelled=False)` (the single terminal builder — runs
the inherited `_classify_finish(self.stop_reason, self.message)`, stamps
`finish_reason` / `provider_finish_reason` / `error_message` / `cancelled` /
`usage` onto the message, deep-copies it into `FinishEvent`, attaches
`request.response_format` as `_response_format`), `error_event(exc)` (stamps
`message.error_message`, carries `usage=self.usage`), `timeout_event()`, the
live accessors (`message`, `text`, `tool_calls`, `usage`, `finish_reason`,
`provider_finish_reason`, `cancelled` — properties over the state above), and
`__del__` (ResourceWarning + sync force-close when the response is still open,
as today).

Two invariants stamped here, pinned by frozen full-object tests: streaming never
sets `response_id`/`response_model` (they stay `None` — frozen event-equality
tests assert the whole `FinishEvent.message`), and `Usage` is built exactly as
each wire builds it today (no cost computation added).

### 3.2 The mixins

`SyncStreamerMixin`: `__enter__` (deadline = monotonic + timeout, then
`open_wire()`), `open_wire()` (build_request → `client.send(request,
stream=True)` → on ≥400 read body then map through the inherited
`_map_chat_completion_http_error`; connect errors likewise; expiry check →
`TimeoutError` raise), `__exit__` (close response; close owned client),
`__iter__` (raises `StreamError` if `finished` — the consumed-once guard),
`__next__` (PoC loop + `for raw in parse(chunk)` + wire-end handling +
`lines is None` → `StreamError` never-entered guard), `expired()`, `cancel()`
(set flag, close the response — unblocks a blocked read; the resulting httpx
error is routed to `finish_event(cancelled=True)`), `collect()`.

`AsyncStreamerMixin`: the PoC machinery verbatim — `call_later` timer +
`expire()` setting `expired` *before* cancelling, task-per-read, the
three-way `CancelledError` disambiguation (`expired` → timeout event,
`cancelled` → cancelled finish, else re-raise untouched) — plus `aopen_wire()`,
`await cancel()`, `collect()`, and `aclose()`: idempotent, closes the response,
cancels/awaits any in-flight read task, best-effort `aclose()` on the source
iterator; `__aexit__` = cancel timer + `aclose()` + close owned client. The
agent runner's pattern (external task-cancel of `__anext__` → `aclose()` →
`__aexit__`) is the acceptance test for this method.

Mid-stream exception policy (both mixins, in `__next__`/`__anext__`):
`StreamError` → `error_event`; `httpx.HTTPError` → mapped then `error_event`;
source exhaustion → `handle_wire_end()` or premature-end
`StreamError("Stream ended without a terminal event — the provider closed the
wire without sending a finish reason")` → `error_event`; everything else
re-raises. Exactly one terminal, ever.

The agreed loop shape (sync; async adds the task-per-read and the three-way
`CancelledError` disambiguation; httpx error mapping elided):

```python
def __next__(self):
    if self.lines is None:
        raise StreamError("stream was never entered — use 'with'")
    if self.finished:
        raise StopIteration
    if self.pending:
        event = self.pending.pop(0)
        if isinstance(event, (FinishEvent, ErrorEvent)):
            self.finished = True
        return event

    while True:
        if self.expired():
            return self.timeout_event()
        try:
            chunk = next(self.lines)
        except StopIteration:
            events = self.handle_wire_end()
            if not events:
                return self.error_event(StreamError(
                    "Stream ended without a terminal event — ..."))
        else:
            events = []
            for raw in self.parse(chunk):
                events += self.handle(raw)
            if not events:
                continue

        event, *rest = events
        self.pending += rest
        if isinstance(event, (FinishEvent, ErrorEvent)):
            self.finished = True
        return event
```

`finish_event`/`error_event`/`timeout_event` set `finished` themselves; the
`isinstance` tail covers terminals produced by handlers. `handle_wire_end` is
sync in both mixins — latched state only, no I/O, called at most once.

### 3.3 Wire classes — what each must reproduce

**`OpenAIChatCompletionsStreamer`** (shape-routed `handle()`, synthesized block
starts/ends, as the PoC): tool-arg buffering until id+name both arrived,
`StreamError` at finish if a tool never resolved; `finish_reason` chunk closes
all open blocks and latches the stop reason; usage chunk → `UsageEvent`;
`handle_wire_end` emits the finish; mid-stream `{"error": …}` chunk (checked
first, empty choices) → `StreamError` carrying the wire's message —
`tests/agent/test_provider_stream_failure.py` pins this; `reasoning` /
`reasoning_content` deltas → thinking blocks (stays in this base class, per
PRD §6). **`OpenRouterChatCompletionsStreamer`**: `PROVIDER`/`URL` only.

**`OpenAIResponsesStreamer`**: typed-event dispatch; unknown types ignored;
reasoning items keep the wire `rs_…` id (→ `ThinkingBlock.id`), summary parts
joined `"\n\n"`, `encrypted_content` lands as signature at `output_item.done`;
native `apply_patch_call`/`shell_call`: prebuilt typed call at
`output_item.added` (complete=False, no delta events, command deltas ignored),
complete replacement swapped in at `output_item.done`, duplicate-done guarded;
terminal events close still-open blocks (truncated native call → `arguments={}`,
`complete=True`, subclass preserved); hosted-tool items ignored; `"error"` event
→ `StreamError`; `provider_finish_reason` composed as `"incomplete:<reason>"`
exactly like the non-streaming path.

**`AnthropicStreamer`**: event-envelope SSE (`event:`/`data:` blocks —
its `parse` handles the envelope); text / thinking (+`signature_delta`,
redacted-whole-on-start) / `tool_use` with `input_json_delta`; usage from
`message_start` + `message_delta`; finish emitted from `message_stop` (wire
puts usage before it). **Index mapping decision**: the streamer keeps its own
wire-index → content-index dict; unknown `content_block_start` types go into an
ignored set and their deltas/stops are dropped, so a future server-tool block
degrades to "ignored" instead of today's dense-index `StreamError`. No frozen
test pins the old failure mode. The ignored set is a placeholder, not a policy:
`server_tool_use`, `web_search_tool_result`, etc. are planned and each lands
later as one more branch in the start/delta handlers routing through
`self.indices` (plus new public event/block models if they surface publicly —
separate scope).

**`BedrockStreamer`**: `iter_chunks`/`aiter_chunks` → bytes; `parse` = frame
buffer (prelude/message CRC checks → `StreamError` "…CRC mismatch",
`:message-type` exception/error frames → `StreamError` with the server's
message), and each completed frame is returned with its `:event-type` header
stamped in as `"type"` — Converse payloads carry no `type` key, so the base
`handle()` dispatch works unchanged; synthesized starts for text/reasoning (Converse omits
`contentBlockStart` for them); `messageStop` latches, `metadata` → `UsageEvent`
(cache tokens included), `handle_wire_end` emits the finish; partial trailing
frame at wire end is dropped silently (today's behavior).

**`FauxStreamer`**: constructor gains `scripted=` (data). `open_wire` /
`aopen_wire` yield scripted raw dicts through the same `handle()` machinery: one
start/delta/end triple per block, then the scripted error (always
`StreamError(message)` — `error_class` is ignored in streaming, as today), then
usage/finish. `faux_hang`: async source awaits `asyncio.Event().wait()` at the
marker (cancellable by timer, `cancel()`, or external cancel); sync source
raises `RuntimeError("faux_hang() is async-only; use acompletion_stream")`.
Classification stays the faux passthrough (canonical values verbatim, refusal →
`"Faux refusal: …"`). `FauxTransport` keeps `set_responses`, `.requests`,
`_pop()` called *after* `_record_request()` and eagerly at call time — the
agent's `ConversationScript` subclasses `_pop`, so that contract is frozen.
The `faux_refusal`-not-in-`faux/__init__` asymmetry stays as-is.

### 3.4 Shared wire code — the mixin extraction

Verified: every payload/projection/classify/error-mapping method across all four
packages reads only `self._provider`, `self._base_url`, `self._api_key`, plus
ClassVars (`TOOL_PROJECTOR_BASE`, `THINKING_DISPLAY`) and the BaseTransport
helpers `_headers`, `_chat_completion_url`, `_attestation_is_replayable`,
`_resolve_projector`/`_resolve_call`/`_native_projector_for_item`. None touch
clients or `_timeout`. So:

- `transports/base.py`: extract those helpers + the two ClassVar declarations
  into `WireFormatMixin`; `BaseTransport` inherits it.
- Per package, **in the same file**, move the wire methods from the transport
  class into `<X>WireMixin(WireFormatMixin, …)` — `OpenAIWireMixin` also
  inherits the existing `OpenAIErrorMappingMixin`; Anthropic's and Bedrock's
  inline `_map_chat_completion_http_error` move into their wire mixins.
  `<X>Transport(BaseTransport, ChatCompletionTransportMixin, <X>WireMixin)`;
  streamer wire class `<X>Streamer(<X>WireMixin, BaseStreamer)`.

This step is a pure refactor with the full suite green before any demolition.
The streamer's `build_request()` then reads top-to-bottom:
`payload = self._build_chat_completion_payload(self.request, stream=True)`,
`url = self._chat_completion_url(self.request, stream=True)`, headers from
`self._headers()`, and `client.build_request(..., timeout=httpx.Timeout(None,
connect=self.CONNECT_TIMEOUT))`.

## 4. Timeout mechanics (verified against httpx 0.28.1)

- `build_request(timeout=…)` sets `request.extensions["timeout"]`, and
  `send(stream=True)` injects the client default **only when the extension is
  absent** — the per-request override wins over shared-client config, which is
  what makes injected/shared clients safe. `httpx.Timeout(None, connect=5.0)`
  is exactly "connect 5s, read/write/pool disabled".
- With `build_request+send(stream=True)` we own closing (no CM): the mixins'
  `__exit__`/`aclose`/`__del__` are the close path, including the ≥400 route
  (read body → close → raise mapped error — the OpenRouter-402-body test pins
  reading before raising).
- `_client.py`: `timeout=` becomes per-call on all four helpers; drop `timeout`
  from `_provider_cache`'s key and from provider construction. `acompletion`
  keeps the `asyncio.timeout` wrap (message: `"completion exceeded
  timeout={t}s"` — the agent test respell tracks this wording). `completion`
  forwards it as the per-request httpx timeout (best-effort, documented).
  Transports' `completion`/`acompletion` gain `timeout=None`; providers forward.
- Stream expiry wording: open → raise `TimeoutError("request did not open
  within timeout={t}s")`; mid-stream → terminal
  `ErrorEvent(TimeoutError("stream exceeded timeout={t}s"))`.

## 5. Wiring

`ChatCompletionTransportMixin.completion_stream(request, *, timeout=None)` =
`self.STREAMER(request=request, provider=self._provider,
httpx_client=self._client, api_key=self._api_key, base_url=self._base_url,
timeout=timeout)`; async twin via `_ensure_aclient()`. Delete the
`_chat_completion_stream_class` hooks. Faux overrides the two methods to
record/pop and pass `scripted=`. Providers pass `timeout=` through verbatim.

Agent edits (in scope, per PRD §5.2): `runner.py:3497–3506` and `:3540–3550`
pass `timeout=` only (from `client_completion_timeout_in_ms`); delete
`RuntimeConfig.builtin_client_completion_timeout_in_ms` and its plumbing;
respell the compaction message at `runner.py:2875`; respell
`test_runner_failures.py:1620` and `scenarios.py:870`; update
`docs/agent/08-runtime-config.md`.

## 6. New tests (PRD §8.5 — designed from these layers, not ported)

```
tests/client/test_transports/test_streamer/    the shared machinery, driven through faux
├── test_iteration.py      pending fan-out, exactly-one-terminal, consumed-once,
│                          never-entered StreamError, post-exit accessor reads
├── test_timeout_async.py  timer expiry mid-read and between events; expiry during open raises
├── test_timeout_sync.py   cooperative deadline (net-new behavior)
├── test_cancel.py         cancel → FinishEvent(cancelled=True, finish_reason=None,
│                          partial message, last usage); cancel-after-finish no-op
└── test_lifecycle.py      aclose idempotence, external-cancel → aclose → __aexit__,
                           abandonment — all under -W error::ResourceWarning

per package: test_<x>/test_streamer_handlers.py
    Instantiate the combo class, never enter, feed raw payloads to handle()/named
    handlers; assert the returned event list AND the message state as whole
    objects. This is where the per-handler "given this payload → exactly these
    events / this raise" tests the PRD asks for live. Includes CC tool-buffering
    and mid-stream error chunks, Responses native prebuilt/replacement +
    truncation-close + duplicate-done, Anthropic redacted/signature + unknown-
    block-ignored, Bedrock synthesized starts.
test_bedrock/test_streamer_parse.py
    Frame decoding at the parse() seam: straddled/concatenated/byte-at-a-time
    frames, CRC corruption → StreamError, exception/error frames. The old
    file's CRC'd _frame byte-builder is reused as wire data.
end-to-end gaps closed: Bedrock through transport.completion_stream (sync+async),
    OpenRouter streaming, Anthropic async + streaming error mapping,
    collect() sync+async at the helper layer, the accessor set.
```

Frozen-test respells (from `test_audit.md` §3): the two `"RawFinish"`
substrings → `"without a terminal event"`; drop the `_http_response is None`
line; the four `total_timeout=` renames; the agent timeout-message respell.

## 7. Order of work

1. **Wire-mixin extraction** (§3.4) — pure refactor, full suite green.
2. **Delete the 22 internal tests** (audit §2) + the stale `.pyc`.
3. **Demolition**: `types/streaming.py` trim, `transports/*/stream.py`,
   factory hooks, `total_timeout` plumbing, `types/__init__` re-exports.
   Frozen streaming tests are red from here until their wire lands.
4. **Core + faux + helper plumbing**: `streamer.py`, `FauxStreamer`, the
   transport factory, helper `timeout=` pass-through. Gate: the mixin test
   suite (new), `test_faux/*`, `test_api/test_total_timeout.py` (respelled),
   and the **entire agent suite** — the strongest early signal, since every
   agent streaming test runs on faux.
5. **OpenAI CC + OpenRouter streamers.** Gate: frozen `test_openai/*`,
   `tests/agent/test_provider_stream_failure.py`, new handler tests.
6. **Responses streamer.** Gate: frozen `test_openai_responses/*` incl. native
   captures; new handler tests.
7. **Anthropic streamer.** Gate: frozen `test_anthropic/*` + new async/error
   tests.
8. **Bedrock streamer.** Gate: new parse/handler/end-to-end tests.
9. **Agent-side edits** (§5) + remaining respells.
10. **Full suite + `uv run ruff check --fix && uv run ruff format`** — no
    warnings (they are errors).
11. **Docs**: rewrite `docs/client/08-streaming.md` (lifecycle at `with`,
    single `timeout=`, streamer architecture replacing §15's RawStreamEvent,
    writing-a-transport guidance); touch `02-quickstart`, `04-chat-completion`,
    `09-providers-and-transports`, `11-exceptions`, `12-testing`,
    `docs/agent/08-runtime-config.md`. Per `docs/llm.txt`: validate every
    snippet with `uv run`, keep README index/Next chains, code wins.

## 8. Risks

- **Frozen full-object equality is the sharpest edge**: any newly stamped field
  (`response_id`, usage cost) or reordered event breaks event-list-equality
  tests that are not allowed to change. Match today's field values exactly.
- **Async teardown under the runner's grace/shield dance**: a read task that
  outlives the `wait_for` that awaited it, then is killed, then `aclose()`,
  then `__aexit__` — any leaked task or response is a suite failure. Build
  step 4's lifecycle tests first.
- **MRO discipline**: wire mixins and iteration mixins must stay
  method-disjoint; the combo classes must need no bodies (faux included, via
  the `open_wire` seam).
- **CC `[DONE]`-then-open-socket**: finish fires on the marker via
  `handle_wire_end`, not on socket close, so a server that lingers cannot hang
  the consumer an extra read.
- **Anthropic index remapping** changes an unpinned failure mode (unknown block
  types now ignored, not `StreamError`) — deliberate, noted in §3.3.
- **Sync stall**: with read timeouts off, a stalled sync socket blocks until
  `cancel()` from another thread or TCP death — documented best-effort
  behavior, connect phase still bounded.
