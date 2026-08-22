# PRD 0011a — Rewrite of `luca.client` streaming

## 1. What we are building

`luca.client` is a thin, unified LLM SDK (`httpx` + `pydantic` only). Its streaming
surface is two public functions — `completion_stream()` and `acompletion_stream()` —
that yield a typed event sequence (`StartEvent`, `TextDeltaEvent`, …, `FinishEvent` /
`ErrorEvent`) while an assistant message is being generated over a provider's wire
protocol.

We are rewriting the entire streaming implementation from scratch as a polymorphic
class hierarchy in which **each transport is a streamer class that owns its wire
end-to-end**: request building, SSE parsing, event creation, and error mapping are
methods on the class, overridable by subclassing. Sync/async iteration and timeout
mechanics live in two mixins written once and shared by every transport.

**None of the existing streaming implementation survives.** The current code is
imperative, tightly coupled (stream objects call back into their transport through a
`transport=self` reference), and has no override points. It is replaced wholesale,
not migrated. Delete first, then build.

The public API is preserved except for three deliberate changes listed in §5.

## 2. The reference implementation

`specs/0011a-client-streaming-refactor/poc_streamer.py` is a working proof of
concept of the full design, validated live against OpenAI (`/v1/responses` and
`/v1/chat/completions`), OpenRouter, and Anthropic, in both sync and async, with and
without timeouts. **It is normative for shape, naming, and altitude**: the real
implementation adds the contract behaviors in §6 and integrates with the existing
provider/transport layers, but a reader must be able to lay the final code next to
the PoC and recognize every class. When in doubt about how much abstraction to add,
add none — match the PoC.

Four deviations from the PoC were agreed during planning and are recorded, with
the wire facts that force them, in `plan.md` §2: `parse()` returns a **list** of
raw events (Bedrock frames), `handle_wire_end()` emits the terminal at wire end
(the CC and Bedrock wires deliver usage after their finish marker), the mixins
expose one `open_wire()`/`aopen_wire()` seam (faux reuses the machinery with
empty combo classes), and the constructor takes a `provider=` label (generic
hosts ride `OpenAITransport`). Where the PoC and `plan.md` disagree, `plan.md`
wins.

## 3. The design

### 3.1 Class layout

```
BaseStreamer                     shared state, SSE line parsing, handler dispatch
AsyncStreamerMixin               async iteration + deadline (call_later timer cancels the read task)
SyncStreamerMixin                sync iteration + deadline (monotonic clock, cooperative)

OpenAIResponsesStreamer          wire knowledge for /v1/responses
OpenAIChatCompletionsStreamer    wire knowledge for /v1/chat/completions
OpenRouterChatCompletionsStreamer  subclass: only PROVIDER and URL change
AnthropicStreamer                wire knowledge for /v1/messages
BedrockStreamer                  wire knowledge for the Bedrock Converse stream
FauxStreamer                     scripted messages for tests, same event machinery

AsyncOpenAIResponsesStreamer(AsyncStreamerMixin, OpenAIResponsesStreamer): pass
SyncOpenAIResponsesStreamer(SyncStreamerMixin, OpenAIResponsesStreamer): pass
... one empty sync/async combination class per wire class ...
```

Two orthogonal axes. A new provider is one wire class plus two empty combination
classes. A new iteration strategy would be one new mixin serving every provider.
Specialization happens by **inheritance, never by injection**: no strategy objects,
no callbacks passed into constructors, no collaborators.

### 3.2 `BaseStreamer` — the shared contract

```python
class BaseStreamer:
    PROVIDER = None            # "openai", "anthropic", ... — stamped on messages and errors
    URL = None
    HANDLERS_BY_TYPE = {}      # wire event type -> handler method name

    def __init__(self, request, httpx_client=None, api_key=None, base_url=None, timeout=None):
        ...
        self.message = AssistantMessage(content=[], provider=self.PROVIDER, model=request.model)
        self.pending = []      # one wire line can produce several public events
        self.finished = False

    def parse(self, chunk):
        """One wire chunk -> a LIST of raw events ([] for non-data lines,
        several for a bytes chunk holding multiple Bedrock frames)."""

    def handle(self, raw_event):
        """One wire event -> a LIST of public events (possibly empty)."""
        handler = self.HANDLERS_BY_TYPE.get(raw_event["type"])
        if handler is None:
            return []
        return getattr(self, handler)(raw_event)

    def handle_wire_end(self):
        """The wire closed before a terminal was emitted. Default [] means the
        provider hung up -> premature-end ErrorEvent. CC and Bedrock override:
        their wires deliver usage AFTER the finish marker, so their FinishEvent
        can only be built here. Latched state only — no I/O, at most once."""
        return []
```

The constructor receives **data only**: the `ChatCompletionRequest`, an optional
httpx client (owned and closed by the streamer when it creates one itself),
`api_key`, `base_url`, and `timeout`. Never a transport, provider object, or any
callable. Everything behavioral is a method on the class.

Note this differs from the PoC constructor (which took `messages, model, tools`
directly): the real streamer takes the `ChatCompletionRequest` and its
`build_request()` builds the full payload from it — tools, system message,
temperature, everything the non-streaming path sends. The request object is also
needed after the wire closes (`response_format` for `FinishEvent.parse()`).

### 3.3 The mixins — iteration and timeout

Copy the PoC. The essential shape of the async loop:

```python
class AsyncStreamerMixin:
    async def __aenter__(self):
        request = self.build_request()
        if self.timeout is not None:
            self.timer = asyncio.get_running_loop().call_later(self.timeout, self.expire)
        self.last_task = asyncio.create_task(self.client.send(request, stream=True))
        self.response = await self.last_task          # CancelledError -> TimeoutError if expired
        ...status check -> typed exception...
        self.lines = self.response.aiter_lines()
        return self

    def expire(self):                                  # timer callback
        self.expired = True
        if self.last_task is not None and not self.last_task.done():
            self.last_task.cancel()

    async def __anext__(self):
        if self.finished: raise StopAsyncIteration
        if self.pending: return self.pending.pop(0)    # + terminal bookkeeping
        while True:
            if self.expired: return self.timeout_event()
            self.last_task = asyncio.create_task(anext(self.lines))
            line = await self.last_task                # CancelledError disambiguated by flags
            events = self.handle(self.parse(line))     # skip None / empty
            event, *rest = events
            self.pending += rest
            return event                               # + terminal bookkeeping
```

Why this exact machinery (settled decisions — do not redesign):

- **Object iterators with explicit `__next__`/`__anext__`, never generator-based
  `__iter__`/`__aiter__`.** Every await lives in a method frame the library
  controls, so cancellation is always catchable there; there is no async-generator
  lifecycle (`aclose()`, loop-bound finalization, `GeneratorExit` interplay); and
  methods — not a closed generator body — are the override surface.
- **Task-per-read plus one `call_later` timer** is the async deadline. Wrapping each
  read in a task the streamer owns means the timer only ever cancels library code,
  never the consumer's frame. `asyncio.timeout()` around the whole iteration is
  impossible here: the cancellation would land in the consumer's frame while the
  generator/method is suspended.
- **The `expired` flag, set by the timer before cancelling**, covers deadline expiry
  while no read is in flight (the consumer holds control between events) and
  disambiguates the timer's `CancelledError` from external cancellation, which must
  re-raise untouched.
- **The sync deadline is cooperative**: `time.monotonic()` checked before every
  read. A read already blocked on the socket runs to completion before expiry is
  detected. This is a platform limit, not a bug; it is documented behavior.
- **`pending` list**: `handle()` returns a list because one wire line can produce
  several public events (usage + finish, synthesized block ends). `__next__` returns
  the first and buffers the rest. `__anext__` cannot yield, so a buffer is the only
  correct fan-out.
- **`finished` flag**: after a terminal event (`FinishEvent` or `ErrorEvent`) is
  returned, the next `__next__` raises `StopIteration`. Exactly one terminal, ever.

### 3.4 Wire classes — handlers create events

Each wire class fills `HANDLERS_BY_TYPE` and implements one public, named handler
method per wire event type, returning a list of public events. From the PoC:

```python
class AnthropicStreamer(BaseStreamer):
    PROVIDER = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"
    HANDLERS_BY_TYPE = {
        "message_start": "handle_message_start",
        "content_block_start": "handle_content_block_start",
        ...
    }

    def handle_content_block_start(self, raw_event):
        index, block = raw_event["index"], raw_event["content_block"]
        if block["type"] == "text":
            self.message.content.append(TextBlock(text=""))
            return [TextStartEvent(index=index)]
        ...
```

The dispatch table is the primary override point, `handle()` itself is the escape
hatch: `/v1/chat/completions` chunks carry no `"type"` field, so that wire class
overrides `handle()` with shape-based routing and *synthesizes* the block start/end
events its wire never sends (see the PoC). Both patterns are legitimate; the
contract is only `handle(raw_event) -> list[StreamEvent]`.

Wire classes accumulate state on `self` (`self.message`, stashed usage, stop
reason). The streamer is the single mutator of its message and the single emitter of
its events. Finish-reason canonicalization is a dict or a few lines inside the
terminal handler — not a separate classifier object.

The public event contract every wire class must uphold (it is documented in
`docs/client/08-streaming.md` and asserted by the surviving tests): `start` first;
block indices dense and ordered, matching positions in `message.content`; every
delta between its block's start and end; exactly one terminal.

Bedrock's stream is not SSE (AWS event-stream framing). Its wire classes override
the read/parse seam as methods on the class, same pattern, no new abstractions.
The faux transport gets `FauxStreamer` classes that walk a scripted
`AssistantMessage` and emit the same events through the same base machinery — the
test suite streams through it.

### 3.5 Wiring into the existing layers

The helper and provider layers keep their jobs. The transport keeps owning the
non-streaming path; for streaming it becomes a pure factory:

```python
# layer 1 — helper (no network)
def completion_stream(*, model, messages, timeout=None, **kwargs):
    provider, model_id = ...resolve "openai:gpt-4o"...
    request = ...build ChatCompletionRequest...
    return provider.completion_stream(request, timeout=timeout)

# layer 2 — provider: picks the transport for the model, delegates
# layer 3 — transport: names the class, forwards data
class OpenAIResponsesTransport:
    STREAMER = SyncOpenAIResponsesStreamer
    ASYNC_STREAMER = AsyncOpenAIResponsesStreamer

    def completion_stream(self, request, *, timeout=None):
        return self.STREAMER(
            request=request,
            httpx_client=self._client,
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=timeout,
        )
```

The two class attributes replace today's stream-class-factory methods. The
`transport=self` back-reference is gone: everything the old stream borrowed from the
transport at runtime (payload building, URL, headers, error mapping, finish
classification) is now reachable as methods on the streamer. `timeout=` travels per
call, never through the transport constructor — the constructor's `timeout=` keeps
its current meaning, the httpx default for the transport-owned clients serving the
non-streaming path.

Wire knowledge that serves BOTH paths stays shared — by inheritance, never by
moving or duplicating it. The message/tool payload projection, the HTTP error
mapper (`OpenAIErrorMappingMixin` and its Anthropic/Bedrock equivalents), and
finish classification live in the transport package and are inherited or imported
by the streamer exactly as the transport uses them for `completion()`/
`acompletion()`. The streamer's `build_request()` and error-mapping methods are
thin wrappers adding only the stream-only deltas (`stream: true`,
`stream_options`, the `converse-stream` URL). Nothing is duplicated; nothing is
injected.

## 4. User-visible lifecycle

```python
with completion_stream(model="openai:gpt-4o", messages=[...]) as s:
    for event in s:
        ...
```

| User code | What fires |
|---|---|
| `completion_stream(...)` | resolve provider, build request objects — nothing on the wire |
| `with s:` / `async with s:` | HTTP request sent, status + headers read; a rejected request raises a typed exception here |
| `for event in s:` | one wire read → `parse()` → `handle()` → one event out |
| block exit | response closed; owned client closed |

## 5. Deliberate public API changes

Exactly three. Everything else — function signatures, event types and ordering,
the `FinishEvent`/`ErrorEvent` split, `collect()`, `cancel()`, the live accessors
(`s.message`, `s.text`, `s.tool_calls`, `s.usage`, `s.finish_reason`,
`s.provider_finish_reason`, `s.cancelled`) — is preserved and must keep passing its
existing black-box tests.

1. **The request opens at `__enter__`/`__aenter__`, not on first iteration.**
   Creating the stream still makes no network call, but entering the context manager
   sends the request and reads the response headers. Pre-open failures (401, 429,
   unknown model, …) therefore raise at the `with` line instead of the first loop
   pass. Two consequences, both accepted: iterating a stream that was never entered
   raises `StreamError` (the lazy-open path is gone); and the open moves out from
   under the agent runner's cancellation race — today the first read performs the
   open and is raced against the CancellationToken, after the change a cancel
   during connect waits for the open (bounded by the connect timeout kept in
   change 2). Deliberate.

2. **`timeout=` is a total wall-clock deadline for the call — on all four helpers
   (`completion`, `acompletion`, `completion_stream`, `acompletion_stream`), sync
   and async. `total_timeout=` is removed everywhere** (parameters, plumbing,
   `_set_total_timeout`, and `acompletion`'s variant).
   - *Plumbing.* `timeout=` travels per call — helper →
     `provider.completion_stream(request, timeout=…)` → transport → streamer — and
     no longer feeds provider construction or the provider cache key in
     `_client.py`.
   - *Streaming.* Expiry after the stream opened emits one terminal `ErrorEvent`
     carrying the SDK `TimeoutError`; expiry during open raises `TimeoutError`.
     Async enforcement can interrupt an in-flight read. Sync enforcement is
     cooperative (checked before each read) and best-effort by nature — httpx read
     timeouts reset whenever new data arrives, so a wall clock cannot be enforced
     on a live sync socket. Document it as best-effort.
   - *Non-streaming.* `acompletion` enforces the deadline with the same
     `asyncio.timeout` wrap `total_timeout=` uses today; `completion` passes it as
     the per-request httpx timeout — best-effort, documented.
   - *httpx timeouts on stream requests.* The streamer builds every stream request
     with a per-request override: connect timeout kept, read/write/pool disabled.
     A shared transport client's 60s default must not kill a slow stream, and a
     black-holed connect must still fail in bounded time. Same policy when the
     streamer owns its client. Caller-injected `http_client`/`async_http_client`
     keep working — the per-request override is what makes that safe.
   - *Agent layer, in scope of this change.* The two runner call sites passing
     `total_timeout=`, deleting
     `RuntimeConfig.builtin_client_completion_timeout_in_ms`, repointing
     `client_completion_timeout_in_ms` at `timeout=`, the compaction error string,
     and `docs/agent/08-runtime-config.md`.
   Deliberate.

3. **Events no longer carry `partial` (nor `ErrorEvent.partial_message`).** Already
   landed in `luca/client/types/streaming.py` — the event models there are current
   and are NOT part of this rewrite's demolition. The streamer still accumulates
   `self.message` internally; `FinishEvent.message` remains a deep copy. Deliberate.

## 6. Behavior the rewrite must implement (beyond the PoC)

The PoC proves the design; production adds the contract edges. All of this is
existing, documented, black-box-tested behavior — reimplement it inside the new
shape:

- **Exactly one terminal per opened stream, no exceptions leak from iteration.**
  Mid-stream httpx errors, malformed JSON, malformed tool-argument JSON, and the
  wire closing without a terminal all become one `ErrorEvent` (carrying a
  `StreamError` / mapped `ClientError`), then the stream stops.
- **Pre-open error mapping.** A non-200 at open maps status + provider error body to
  the typed exception hierarchy (`AuthenticationError`, `RateLimitError` with
  retry-after, `BadRequestError`, …) as a method on the wire class. Connection
  failures map to the SDK's `ConnectionError`.
- **`cancel()`** (sync) / **`await cancel()`** (async): interrupts the stream and
  produces a terminal `FinishEvent(cancelled=True, finish_reason=None)` with the
  partial message and any usage already seen — a finish, not an error. In the async
  mixin this is the third source of `CancelledError` on the read await; a
  `cancelled` flag disambiguates it exactly like `expired` does for the timer.
  External cancellation (the consumer's own task being cancelled) still re-raises.
- **`collect()`**: drains the stream and returns the same `ChatCompletionResponse`
  the non-streaming call would produce; re-raises on `ErrorEvent`; a consumed stream
  raises `StreamError` on re-iteration.
- **Full block vocabulary.** Thinking and refusal blocks stream on the wires that
  have them; provider-native tool calls keep their current streaming behavior
  (OpenAI native calls stream start→end with a complete typed call; Anthropic native
  tools stream JSON deltas). The surviving tests define the exact expectations.
- **Resource hygiene.** The suite runs with `filterwarnings = ["error"]`: no
  unclosed responses or clients, ever, including on cancellation, timeout, and
  abandonment. `__del__` on an open stream warns, as today.
- **Iterator lifecycle contract.** The async streamer exposes `aclose()` — the
  agent runner calls `iterator.aclose()` in a `finally` and hard-cancels in-flight
  `__anext__` tasks; external cancellation → `aclose()` → `__aexit__` must not
  leak a response, client, or task. Iterating a stream that was never entered
  raises `StreamError`; re-iterating a consumed stream keeps raising `StreamError`.
- **`ErrorEvent.usage`** carries the last usage seen before the failure, as today.
- **Exception identity.** The SDK `TimeoutError` stays a `ClientError` subclass
  and must NOT become a subclass of builtin `TimeoutError` — the agent classifies
  timeouts by `isinstance` and its bare `except TimeoutError` handlers must not
  catch the SDK's. `timeout=None` is always accepted and means no deadline; the
  agent passes it explicitly on every call.
- **OpenRouter quirks stay in the base.** Mid-stream `{"error": …}` chunks and
  `reasoning`/`reasoning_content` thinking deltas are handled in
  `OpenAIChatCompletionsStreamer` — they serve DeepSeek-style OpenAI-compatible
  hosts too and are inert on pure OpenAI. The OpenRouter subclass changes only
  `PROVIDER` and `URL`.

## 7. Demolition list

Delete — do not adapt, do not keep "for reference":

- The stream class hierarchy and the accumulator in
  `luca/client/types/streaming.py` (`BaseStream`, `AsyncBaseStream`,
  `ChatCompletionStream`, `AsyncChatCompletionStream`, `_ChatCompletionAccumulator`)
  and the entire `RawStreamEvent` dataclass vocabulary (`RawBlockStart`,
  `RawTextDelta`, …). The transport-emits-raw-events / accumulator-translates
  architecture is gone. Only the public event models in that file survive.
- Every `transports/*/stream.py`.
- The streaming halves of the transport mixin (`completion_stream` /
  `acompletion_stream` plumbing that passes `transport=self`, the stream-class
  factory methods).
- `total_timeout` parameters and plumbing in `_client.py` (`acompletion`'s
  included), `_set_total_timeout`, and the per-pull `asyncio.timeout` machinery.

If a piece of old streaming code seems worth keeping, it is not. Rewrite it inside
the new shape or drop the behavior only if it is one of the three deliberate changes
in §5. New streamer code lives in the transport packages beside the wire knowledge
it belongs to; the public event models stay where they are.

## 8. Implementation order

Work in exactly this order:

1. **Inventory the tests before touching anything.** Classify every streaming test
   as either **black-box** (drives `completion_stream`/`acompletion_stream` or a
   provider/transport surface and asserts only on the public event sequence,
   terminals, accessors, `collect()`, `cancel()`, exceptions) or **internal**
   (imports or asserts on `RawStreamEvent`, the accumulator, `parse_chunks`,
   `_open_http`, stream-class internals, `total_timeout` plumbing). Write the two
   lists down first.
2. **The black-box list is frozen.** Those tests survive the rewrite and are the
   acceptance criteria. They may receive only the minimal edits forced by §5 —
   where an error was expected at first iteration and now raises at `with`, where
   `total_timeout=` is spelled `timeout=`, where a test touched `event.partial`.
   Nothing else in them changes: not assertions, not scenarios, not structure.
3. **Delete every internal test.** They test code that will not exist. They are
   not a checklist for new tests either — nothing is ported.
4. **Delete the old implementation** (§7), then build the new one (§3, §6).
5. **Write the new tests from the new implementation's contracts.**
   Behavior-driven, layer by layer, each layer tested as a black box at its own
   surface: every handler (given *this* raw wire payload, `handle_<event>` on
   *this* wire class returns exactly these public events, or raises like this),
   `parse` per framing (SSE lines, event-stream frames, CRC failures), the mixins
   (iteration, pending fan-out, exactly one terminal, deadline expiry, cancel,
   consumed-once, never-entered), pre-open error mapping per wire, and the full
   streamer surface through the transport factories. Assert whole objects and
   whole event lists, per the project test style.
6. **The suite passes.** The frozen black-box tests and the new tests green,
   `uv run ruff check --fix && uv run ruff format` clean, no warnings (they are
   errors).
7. **Update `docs/client/08-streaming.md`** (and any doc referencing
   `total_timeout`, lazy opening, `partial`, or `RawStreamEvent` — that includes
   `02-quickstart.md`, `04-chat-completion.md`, `09-providers-and-transports.md`,
   `11-exceptions.md`, `12-testing.md`, and `docs/agent/08-runtime-config.md`) to
   match — follow `docs/llm.txt` for how docs are written.

## 9. Code style — read before writing any code

This rewrite exists because the old implementation was imperative and coupled.
Do not rebuild that in new clothes:

- **No underscore-helper soup.** Behavior lives in a small number of public, named
  methods on the class hierarchy — `build_request`, `parse`, `handle`,
  `handle_<wire_event>`, `handle_wire_end`, `open_wire`, `timeout_event`,
  `expire`, `cancel`, `collect`. If you are
  about to write a module-level `_do_x()` or a three-line private method called from
  one place, inline it.
- **No injected collaborators.** No parser objects, emitter objects, strategy
  callables, or config dataclasses threaded through constructors. State on `self`,
  behavior on the class, specialization by subclass — the PoC is the ceiling of
  abstraction, not the floor.
- **No speculative hooks.** Add an override point only where a second concrete case
  already exists in this rewrite. Polymorphism on demand.
- **Flat over clever.** A handler method that reads top-to-bottom beats a dispatch
  pyramid. `if`/`isinstance` chains inside a handler are fine — they are the wire
  knowledge, and the wire is finite.
- Match the repo's existing style: module docstring stating the design, type hints,
  Pydantic v2 idioms, minimal inline comments (only for constraints the code cannot
  show — e.g. why `expired` is set before cancelling).
- Tests follow the project rule: assert on whole objects and whole event lists,
  declarative precondition → action → postcondition.
