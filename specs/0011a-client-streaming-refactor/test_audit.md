# Test audit — PRD 0011a, §8 step 1

Every streaming-related test function in `tests/client/` and `tests/agent/` was
read and classified. Two lists, as the PRD requires: **black-box** (frozen — the
acceptance criteria) and **internal** (deleted, not ported). A third short section
records the four places a black-box test is pinned to a name the demolition
removes, and the last section records contract behaviors whose only coverage dies
with the internal tests — input for §8 step 5, not a porting checklist.

## 1. Black-box — FROZEN

These survive the rewrite untouched except where a forced edit is listed.

### tests/client

| File | Tests | Forced edits |
|---|---|---|
| `test_transports/test_openai/test_completion_stream_sync.py` | both (2; the `CASES` table at 40–195 is the data source for the async file too) | none |
| `test_transports/test_openai/test_completion_stream_async.py` | `test_openai_transport_acompletion_stream` | none |
| `test_transports/test_openai/test_errors.py` | `test_openai_transport_stream_http_error_mapping` (111–122), `test_openai_transport_astream_http_error_mapping` (125–140) | none — `pytest.raises` already wraps the `with` line, so the §5(a) raise-site move is invisible. Docstring narrating `_open()/_aopen()` goes stale (prose only) |
| `test_transports/test_openai/test_stream_cancellation.py` | `test_premature_finish_emits_error_event` (24–39) | ⚠ conflict #1 below |
| `test_transports/test_openai_responses/test_completion_stream_sync.py` | 1 test, 5 cases (`CASES` at 55–292 shared with the async file) | none |
| `test_transports/test_openai_responses/test_completion_stream_async.py` | `test_responses_transport_acompletion_stream` | none |
| `test_transports/test_openai_responses/test_stream_protocol.py` | 8 of 9 (open-block-at-terminal closes ×2, malformed tool JSON, non-JSON data, wire error frame, missing status fallback, duplicate terminal ignored, cancel-without-iterating/no-ResourceWarning) | none; `test_a_stream_that_ends_without_a_terminal_emits_an_error_event` is ⚠ conflict #2 |
| `test_transports/test_openai_responses/test_errors.py` | 2 streaming tests (41–54, 57–74) | none — same `pytest.raises`-wraps-`with` shape. Stale docstrings |
| `test_transports/test_openai_responses/test_native_streaming.py` | all 3 (apply_patch start→end, shell ignores command deltas, truncated native call closes empty+complete) | none — reads `captures/*.sse`; captures stay |
| `test_transports/test_anthropic/test_completion_stream_sync.py` | all 4 (text, tool_use, thinking+signature, redacted thinking) | none — the redacted test's side call to `transport._project_assistant_message` is transport projection, which survives |
| `test_transports/test_anthropic/test_native_tools.py` | the 2 streaming tests (209–221, 224–235) | none — reads `captures/*.sse`; captures stay |
| `test_transports/test_bedrock/test_payload_building.py` | `test_the_model_id_is_only_in_the_url_never_the_body` (line 70: `_chat_completion_url(request, stream=True)` → `/converse-stream`) | none — the URL hook survives on the transport; the new Bedrock streamer must keep calling it |
| `test_api/test_total_timeout.py` | all 4 | `total_timeout=` → `timeout=` at lines 30, 46, 67, 92 (§5(b)); module docstring's "async-only" premise reworded; ⚠ conflict #3 on line 78 |
| `test_api/test_completion_async.py` | `test_acompletion_stream_returns_synchronously` | none — pins that the helper is a plain function |
| `test_faux/test_faux_transport.py` | the 2 streaming tests (120–131, 134–148) | none — post-exit `s.message` read must stay valid |
| `test_faux/test_async_streaming.py` | both | none — never-entered stream must not warn on GC |
| `test_providers/test_chat_completion_mixin.py` | the 2 streaming forwarding tests (45–58) | none — pins `provider.completion_stream(request)` sentinel pass-through |
| `test_integration/test_smoke.py` | `test_openai_streaming_smoke` (78–116) | none — end-to-end through the helper; only httpx is mocked, so it runs the new streamer whole |
| `test_exceptions/test_hierarchy.py` | `test_subclass_relationship` (the `TimeoutError`/`StreamError` → `ClientError` rows) | none |

**Surviving fixtures and helpers** (nothing on the demolition list): the four
transport `conftest.py` factories (openai, openai_responses, anthropic, bedrock),
`test_api/conftest.py` (`StubProvider`), `_helpers/httpx_mocks.py`,
`_helpers/stub_transports.py`, and both `captures/` directories.

### tests/agent

All agent tests touching streaming go through public seams (faux helpers, the
`acompletion_stream` monkeypatch point, real provider over a mock socket) and
survive frozen. Notable pins the rewrite must honor:

| File | Pin |
|---|---|
| `test_native_tools/conftest.py` (`_FakeStream`, `MockLLM`) | `FinishEvent` constructor shape; `async with` + `async for` consumption; the monkeypatch seam is the helper *name* in `runner.py` |
| `test_provider_stream_failure.py` (2 tests) | Real OpenRouter provider+transport over `httpx.MockTransport` — runs the new streamer end-to-end. Pins: the mid-stream `{"error": …, choices: []}` chunk becomes `StreamError` carrying the wire's own message ("Stream ended before a terminal response event" is OpenRouter's text, not ours) |
| `test_runner_failures.py` | ⚠ conflict #4 below; also pins faux contracts: mid-stream `faux_error` → `StreamError` *after* the preceding block's deltas; scripted `error_class` surfaces as that exact type |
| `test_runner.py`, `test_runner_cancellation.py`, TUI tests, `test_integration_full_stack.py` | faux async streaming end-to-end; `faux_hang` parks until cancel; `ConversationScript` subclasses `FauxTransport` and overrides `_pop()` — the rewritten faux transport keeps `_pop` as the single per-call script hook and keeps appending to `.requests` before answering |
| `scenarios.py:870` | hand-built string `"completion exceeded total_timeout=0.05s"` — inert data, respell alongside conflict #4 for consistency |

## 2. Internal — DELETE

| File | Scope |
|---|---|
| `tests/client/test_types/test_streaming_events.py` | whole file (4 tests) — drives `_ChatCompletionAccumulator` + `Raw*` directly |
| `tests/client/test_types/test_streaming_native.py` | whole file (5 tests) — accumulator `prebuilt=`/`replacement=` protocol |
| `tests/client/test_transports/test_bedrock/test_stream.py` | whole file (12 tests) — `__new__`-built stream, fake `_http_response`, `parse_chunks()`, `Raw*` assertions. The `_frame` helper (31–46) builds real CRC'd eventstream frames — legitimate to reuse as wire *data* when designing the new Bedrock streamer tests |
| `tests/client/test_transports/test_openai/test_stream_cancellation.py::test_cancellation_path_via_accumulator` | one function (42–70) |

Total: 22 test functions — three whole files plus one function. Also delete the
stale `tests/client/_helpers/__pycache__/stream_iteration.cpython-314.pyc`
(its source module is already gone).

## 3. Conflicts — black-box shape pinned to a dying name (forced respells)

These four are frozen tests whose scenario and assertions are public, but one
line names something the demolition removes. The edits are forced in the §5
sense — the names cannot exist afterwards:

1. `test_stream_cancellation.py:37` — asserts `"RawFinish" in str(error)`.
   Respell to the new premature-termination `StreamError` message.
2. `test_stream_protocol.py:117` — same `"RawFinish"` substring, same respell.
3. `test_total_timeout.py:78` — asserts `s._http_response is None` as a leak
   side-check. Replace with a public closed-ness signal or drop the line; the
   suite's `-W error::ResourceWarning` already polices leaks.
4. `test_runner_failures.py:1620` — asserts the turn error text
   `"completion exceeded total_timeout=0.05s"`, which is the client's expiry
   message flowing verbatim into `TurnFinish.error`. Respell to the new
   `timeout=` wording (the new message is a freeze decision the implementation
   makes once).

Error-text substrings that are **semantic contract**, to keep in the new
implementation as-is: `'malformed JSON'`, `'non-JSON data'`, the
`'OpenAI refusal: '` prefix, `retry_after` on 429, `.provider` /
`.original_exception` on every mapped error, and the wire-supplied OpenRouter
message passthrough.

## 4. Coverage that dies with the internal tests

Not a porting checklist — §8 step 5 designs tests from the new layers' own
contracts. But these documented behaviors lose their only coverage on deletion,
so the new suite must own them from its own surfaces:

- **Bedrock wire** — frame decode with CRC validation (`StreamError` on
  corruption), a frame straddling two reads, multiple frames in one read,
  byte-at-a-time delivery, synthesized starts for text/reasoning blocks,
  `exception`/`error` frames → `StreamError` with the server's message, cache
  read/write tokens in usage.
- **`cancel()`** → `FinishEvent(cancelled=True, finish_reason=None)` with the
  partial message — today asserted only at the accumulator level.
- **Native prebuilt/replacement mechanics** at unit level (the black-box
  `test_native_streaming.py` keeps the end-to-end half).
- **Finish canonicalization** and subclass-field survival in event dumps at
  unit level.

And these have **zero coverage today** in any list: the sync deadline (net-new
feature), OpenRouter streaming, Anthropic async streaming, `collect()`, and the
accessors `s.text` / `s.tool_calls` / `s.usage` / `s.finish_reason` /
`s.provider_finish_reason`.
