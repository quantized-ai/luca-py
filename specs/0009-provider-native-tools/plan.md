# 0009 — implementation plan

Execution order for the refined PRD. Each step leaves the suite green
(`uv run py.test tests/client/`), then `uv run ruff check --fix && uv run
ruff format`. No new runtime dependencies. Wire shapes and pseudocode live in
`examples.md`; this file is the order, the file list, and the test list.

**Two standing gates:**

1. **Byte-for-byte**: existing transport tests must pass UNCHANGED through
   every step. Moving standard logic into default projectors may not alter a
   single payload byte. If an existing test needs editing, the step is wrong.
2. **`filterwarnings = error`** is the serialization canary: any pydantic
   "unexpected serialization" warning (missing `SerializeAsAny`) fails the
   suite. Run the full suite from step 3 on, not just the touched files.

---

## Step 1 — `BaseTool` + `ToolProjector` (types)

**Files**: `luca/client/types/tools.py`, `luca/client/types/__init__.py`

- `BaseTool(BaseModel)`: `extra="forbid"`, `get_projector() -> ToolProjector
  | None` returning `None`.
- `Tool(BaseTool)`: existing fields/config unchanged.
- `ToolProjector`: concrete class, five hooks raising `NotImplementedError`
  (`project_tool_to_llm`, `build_tool_call`, `project_tool_call_to_llm`,
  `project_tool_message_to_llm(msg, call)`, `project_tool_choice_to_llm`).
  House style: no ABC/Protocol.
- Export `BaseTool`, `ToolProjector` from `types/__init__.py` (public surface
  for third-party tools).

**Tests** (`tests/client/test_types/test_tools.py`): `Tool` validation
unchanged; `BaseTool.get_projector()` defaults to `None`; every projector
hook raises `NotImplementedError`.

## Step 2 — `ToolCall` registry + validation dispatch

**Files**: `luca/client/types/content.py`

- `NATIVE_TOOL_CALL_TYPES: dict[str, type[ToolCall]]` module-level.
- `ToolCall.projector_class: ClassVar[type | None] = None`.
- `__pydantic_init_subclass__`: register when the `type` literal default ≠
  `"tool_call"`; duplicate key from a different class → `TypeError`.
- `model_validator(mode="wrap")` `_dispatch_native`: only when `cls is
  ToolCall` and value is a dict; registry hit → subclass validation.

**Tests** (`tests/client/test_types/test_content_blocks.py`): subclass
registers under its literal; re-registering same class is a no-op, different
class raises; base-class validation of a native payload returns the subclass
(full-object assert); `{"type": "tool_call"}` payloads validate as base;
direct subclass validation unaffected; `parse_arguments` still works on a
subclass.

## Step 3 — `ToolMessage` registry + `SerializeAsAny` on content

**Files**: `luca/client/types/messages.py`

- `NATIVE_TOOL_MESSAGE_TYPES` + `__pydantic_init_subclass__` (register only
  subclasses that declare a `type` field; same duplicate guard) + wrap
  validator (dispatch only when the dict HAS `"type"`; base gains NO field).
- `AssistantMessage.content`: `ToolCall` member becomes
  `SerializeAsAny[ToolCall]`.

**Tests** (`tests/client/test_types/test_messages.py`): dump/validate round
trip of an `AssistantMessage` holding a native-call subclass — subclass
fields survive `model_dump()` and revalidate into the subclass; `ToolMessage`
payload without `type` → base (existing serialized data untouched);
`Message` union (`role="tool"` + `type`) dispatches; base `ToolCall` dump
byte-identical to before. Run the FULL suite here — this is where stray
serializer warnings surface.

## Step 4 — streaming vocabulary + accumulator + event containers

**Files**: `luca/client/types/streaming.py`

- `RawBlockStart.prebuilt: ToolCall | None = None`,
  `RawBlockStop.replacement: ToolCall | None = None`.
- Accumulator: on start with `prebuilt`, append it (`complete=False`) and
  emit `ToolCallStartEvent`; on stop with `replacement`, swap it in and emit
  `ToolCallEndEvent` skipping the JSON parse; stop without replacement keeps
  today's path (prebuilt block closes with empty arguments).
- `ToolCallEndEvent.tool_call` → `SerializeAsAny[ToolCall]`;
  `FinishEvent.tool_calls` → `list[SerializeAsAny[ToolCall]]`.

**Tests** (`tests/client/test_types/test_streaming_events.py`): prebuilt
start → correct `ToolCallStartEvent` id/name; stop with replacement → final
arguments, `complete=True`, no parse of `partial_arguments`; stop without
replacement on a prebuilt block → empty-arguments close; event `model_dump()`
keeps subclass fields.

## Step 5 — request DTO + coercion

**Files**: `luca/client/types/completion.py`, `luca/client/_client.py`

- `ChatCompletionRequest.tools: list[BaseTool] | None`.
- `_coerce_tools`: `BaseTool` instances pass through; dicts validate as
  `Tool`, `ValidationError` wraps into `BadRequestError` whose message points
  at native-tools-are-instances; other types → `BadRequestError`.

**Tests**: coercion cases incl. `{"type": "apply_patch"}` dict → clear
`BadRequestError` (behavior fix: today a bad dict leaks raw
`ValidationError` — assert the new wrapping).

## Step 6 — Responses transport refactor (no native tools yet)

**Files**: `luca/client/transports/openai_responses/transport.py`

- `OpenAIResponsesToolProjector`: today's five standard behaviors moved
  verbatim — declaration, `build_tool_call` (absorbs the `_parse_arguments`
  staticmethod), replay item, `function_call_output` (absorbs the
  text-only/`_output_text` logic incl. the images-refused `BadRequestError`),
  forcing shape `{"type": "function", "name": …}`.
- Transport: `TOOL_PROJECTOR_BASE` ClassVar, `_default_tool_projector`,
  `_resolve_projector` (isinstance check → `BadRequestError` before HTTP),
  `_project_tools` via resolver.
- Parse: `function_call` branch delegates to the default projector; registry
  branch with the parse-side isinstance compat check (incompatible/miss →
  ignore).
- `_project_messages` walk with `lineage: dict[str, tuple[projector, call] |
  None]`; `_projector_for_call` (plain call → default; registered call →
  its projector if compatible, else `None` = drop call AND its result);
  `_project_tool_message(msg, lineage)` with default fallback (`call=None`).
- `_project_tool_choice(choice, tools)`: `{"name"}` dict resolves against
  declared tools (first match on `.name`) → projector; not found → today's
  literal; strings/raw dicts unchanged.

**Tests** (`tests/client/test_transports/test_openai_responses/`): existing
files pass untouched (gate 1). New: `_resolve_projector` rejects a foreign
projector before HTTP; lineage-miss result falls back to
`function_call_output`; `tool_choice={"name": …}` for a declared standard
tool is byte-identical to today.

## Step 7 — Responses native tools

**Files**: `luca/client/transports/openai_responses/native_tools.py` (new),
`luca/client/transports/openai_responses/__init__.py`

- Per `examples.md`: `ApplyPatchToolCall` + `ApplyPatchProjector` +
  `ApplyPatchTool`; `ShellExitOutcome` / `ShellTimeoutOutcome` /
  `ShellCommandResult` / `ShellToolCall` / `ShellToolMessage` /
  `ShellProjector` / `LocalShellTool`. `projector_class` bound after each
  projector definition.
- `__init__.py`: import `native_tools` AFTER the transport import
  (native_tools imports the projector base from `.transport`; the reverse
  import never happens — no cycle).

**Tests** (`test_openai_responses/test_native_tools.py`, new): declaration
payloads; parse of the observed wire items → full-object typed calls
(synthesized names, `item_id`, `status`); replay round trip (verbatim item,
id + status); apply_patch result (`is_error` → status); shell result
(structured array, `max_output_length` echoed from the call via lineage,
absent without it); plain `ToolMessage` for a shell call →
`BadRequestError` naming `ShellToolMessage`; `tool_choice={"name":
"apply_patch"}` → `{"type": "apply_patch"}`; foreign-drop: a
`ShellToolCall` + its result projected by a non-Responses transport
disappear together (test lands with step 9/10 transports too); JSON
round trip of a conversation containing both native calls.

## Step 8 — Responses streaming

**Files**: `luca/client/transports/openai_responses/stream.py`

- `_NATIVE_CALL_SLOT = -3`; `_item_added`: registry hit (module can check
  `isinstance(projector, OpenAIResponsesToolProjector)` directly — same
  package) → build call, `complete=False`, emit `RawBlockStart` with
  `prebuilt`; `_item_done`: rebuild from the complete item, emit
  `RawBlockStop` with `replacement`. Diff-delta events
  (`response.apply_patch_call_operation_diff.delta` / `.done`) fall through
  unhandled. Terminal mid-call: existing close loop emits stop without
  replacement.
- Guard the skeleton like the `function_call` branch does: a native item
  without `call_id` at added-time raises `StreamError`, not `KeyError`.
  (Live capture shows `call_id` IS present at added — plus a partial
  `operation` with `type`/`path` known and `diff: ""` — but the guard is
  the same courtesy the standard branch already extends.)
- Accumulator mechanics (prebuilt append + open-index bookkeeping, swap at
  stop, truncation close, coexistence with `partial_arguments` streaming)
  were spike-verified against the real `_ChatCompletionAccumulator`.

**Tests** (`test_openai_responses/test_stream*.py` style, SSE-line
fixtures): apply_patch sequence added→deltas→done asserts the public event
sequence (`tool_call_start` name `apply_patch`, no `tool_call_delta`,
`tool_call_end` with full operation); shell sequence; truncation mid-call →
empty-arguments close + terminal-derived finish; mixed
reasoning/text/function_call/native item ordering keeps indices dense.
Real captured SSE for all four native tools lives in
`specs/0009-provider-native-tools/captures/` (captured live 2026-08-06; the
Anthropic ones verified to replay through the current parser + accumulator):
base the fixtures on these, don't hand-invent event shapes. The shell
capture shows `response.shell_call_command.added/.delta/.done` raw-text
events — assert they fall through ignored.

## Step 9 — Anthropic

**Files**: `luca/client/transports/anthropic/transport.py`,
`luca/client/transports/anthropic/native_tools.py` (new),
`luca/client/transports/anthropic/__init__.py`

- `AnthropicToolProjector`: declaration, `tool_use` build, `tool_use` replay
  block, `tool_result` block (transport keeps the `{"role": "user"}`
  envelope), forcing `{"type": "tool", "name": …}`. Transport gets the same
  resolver/walk/choice structure as step 6; the parse `tool_use` branch calls
  the default projector (native calls arrive through it — no registry on
  this wire).
- Walk omits an assistant message that projects to zero wire blocks (foreign
  thinking + foreign natives — the typical post-switch OpenAI turn); this
  wire rejects empty content. Closes a pre-existing latent hole
  (thinking-only foreign turns already project `content: []`).
- `native_tools.py`: `TextEditorTool` / `TextEditorProjector`
  (declaration-only override, `max_characters` conditional), `BashTool` /
  `BashProjector`.

**Tests** (`test_anthropic/`): existing pass untouched; declarations with and
without `max_characters`; native `tool_use` round trips as base `ToolCall`
(wire name preserved); a `ShellToolCall` + result in history are dropped at
projection (provider-switch case); an OpenAI-minted `[thinking, ShellToolCall]`
turn + its result project to NOTHING (message omitted, not `content: []`);
`tool_choice={"name": "bash"}` → `{"type": "tool", "name": "bash"}`.

## Step 10 — chat completions + OpenRouter

**Files**: `luca/client/transports/openai/transport.py`
(`luca/client/transports/openrouter/` verify-only)

- `OpenAIToolProjector`: declaration (`function` envelope), build from a
  `tool_calls` array entry, replay entry, result → the whole
  `{"role": "tool"}` message (top-level there), forcing
  `{"type": "function", "function": {"name": …}}`.
- Transport: same resolver/walk/choice structure. The walk assembles
  projector-returned entries into the assistant message's `tool_calls`
  array; drop policy removes a foreign call's entry and its result message.
  An assistant message left with no content AND no `tool_calls` is omitted
  (invalid on this wire). Same omission rule in Bedrock (step 11).
- OpenRouter: must NOT redefine `TOOL_PROJECTOR_BASE` — inheriting the
  acceptance set is the point of the class-based check.

**Tests** (`test_openai/`, `test_openrouter/`): existing pass untouched;
`BashTool()` on OpenRouter → `BadRequestError` before HTTP;
`ShellToolCall` + result in history dropped; standard flows byte-identical.

## Step 11 — Bedrock

**Files**: `luca/client/transports/bedrock/transport.py`

- `BedrockToolProjector`: `toolSpec` declaration, `toolUse` build, `toolUse`
  replay block, `toolResult` block, forcing `{"tool": {"name": …}}`
  (`_project_tool_choice` becomes an instance method taking tools; the
  `auto`/omit-on-none mapping stays transport). Same walk + drop policy.

**Tests** (`test_bedrock/`): existing pass untouched; foreign-drop case.

## Step 12 — re-exports, registration, faux

**Files**: `luca/client/providers/openai.py`,
`luca/client/providers/anthropic.py`

- Re-export the native surface per `examples.md` (tools + call/message/result
  types from `providers/openai`; `BashTool`, `TextEditorTool` from
  `providers/anthropic`).
- Registration test: importing `luca.client` alone fills both registries with
  every first-party native type (assert exact registry contents — this pins
  the eager import chain).
- Faux: test that `tools=[ApplyPatchTool()]` passes through `FauxTransport`
  unharmed (faux never projects tools). Expect zero faux code changes.

## Step 13 — docs

Read `docs/llm.txt` first. New `docs/client/` page: native tools +
projectors (declaring, typed calls/results, transport compatibility,
`tool_choice` forcing, provider-switch drop policy, streaming semantics —
Anthropic full deltas, OpenAI start+end). Update the tools page, the
providers-and-transports page, and `13-roadmap.md`.

---

## Live validation checklist — CLOSED 2026-08-06

Nothing remains. Validated live, streamed, both providers:

- `tool_choice` forcing: `{"type": "apply_patch"}` and `{"type": "shell"}`
  both accepted (Anthropic forcing is the ordinary named-tool shape).
- Streamed shapes for all four natives (captures in `captures/*.sse`):
  apply_patch (diff via raw-text deltas), shell (command via
  `response.shell_call_command.*` raw-text events — unhandled fall-through,
  skeleton action EMPTY at added), bash and text editor (ordinary tool_use
  JSON deltas through the CURRENT parser + accumulator, unmodified).
- Two-turn loops streamed end to end: OpenAI shell with the
  PROJECTOR-shaped replay (no `environment` key) + structured output
  echoing `max_output_length`; Anthropic text editor with tool_use replay +
  tool_result.

Only re-check if a provider ships API changes before implementation lands.

## Mocking

Nothing new. Transport tests call `_build_chat_completion_payload` /
`_parse_chat_completion_response` (via `httpx.Response` fixtures) and stream
parsers with SSE-line lists — existing style. Full-object asserts per
AGENTS.md. Faux stays as-is.

## Risks

- **Pydantic mechanics**: spike-verified on the installed stack (pydantic
  2.13.4 / py3.14): wrap-validator dispatch, duplicate-registration guard,
  `SerializeAsAny` union round trip (and the warning WITHOUT it — the canary
  is real), subclass passthrough on `list[BaseTool]`, `ToolMessage` dispatch
  with legacy payloads untouched. One gotcha found by the spike:
  `projector_class` must be `ClassVar[Any]`, not `ClassVar[type | None]` —
  the sibling `type` field shadows the builtin during annotation evaluation
  and the class fails to construct. The full-suite `filterwarnings=error`
  run from step 3 onward remains the coverage backstop.
- **Import order**: `native_tools` imports its transport module's projector
  base; package `__init__` must import `transport` before `native_tools`.
  Registration-at-import is pinned by the step-12 test.
- **Byte-for-byte drift** while moving standard logic: mitigated by gate 1 —
  the existing full-payload tests are the spec.
- **Registry state across tests**: registration is import-time and global;
  tests defining throwaway `ToolCall` subclasses must use unique `type`
  literals (or clean up) to avoid duplicate-key `TypeError` across the suite.
- **`tool_choice` forcing**: both OpenAI variants validated live (see
  checklist) — no remaining wire-shape risk.

## Out of scope (recorded)

- Agent-layer persistence of native calls (`SerializeAsAny` on the agent's
  `Message` containers, permission-layer rendering of `operation`/`action`):
  step two, separate spec.
- Incremental diff rendering for apply_patch streaming (raw-text deltas):
  deferred until a consumer needs it.
- Policing tool-name collisions between a custom `Tool` and a native tool's
  synthesized name: documented, not policed.
