# 2026-08-06

The user's initial intent was: make tool *declarations* extensible — a
`BaseTool` + `ToolProjector` that lets provider-native tools own their wire
declaration, with everything else (parsing, results) assumed to keep working.

By inspecting the codebase and validating live against both providers
(Anthropic text editor + bash, OpenAI Responses apply_patch + shell, mixed
with custom function tools, full multi-turn loops), we established that the
declaration was only a quarter of the problem on OpenAI: the Responses
transport silently drops native call items (dead conversation reported as
`finish_reason="stop"`), the native call must be replayed verbatim in `input`
on every later turn under `store: false`, results need native output item
types (shell's with structured per-command payloads that cannot ride a text
`ToolMessage`), and apply_patch diffs stream as raw-text deltas that break
the JSON `partial_arguments` contract. Anthropic needed the declaration only
— validated working end to end, streaming included, with today's client.

The PRD was rewritten around the user's revised design: `ToolProjector` owns
all four wire touchpoints (declaration, call parse, call replay, result), each
transport ships a default projector implementing standard function-tool
behavior, and native tools carry typed `ToolCall` / `ToolMessage` subclasses.
Details settled on top of the sketch: a `project_tool_call_to_llm` replay
method; a wire-type registry (replacing the per-projector matching hook)
driving parse, replay, and result resolution by call lineage; `arguments` as
single storage with `operation` / `action` as typed properties; no
`ApplyPatchToolMessage` (`is_error` already expresses `status`); `ShellToolMessage`
for structured shell results; serialization via wrap-validator dispatch +
`SerializeAsAny`; class-based projector/transport compatibility checked
before HTTP; V1 native streaming as start+end with no argument deltas.
`plan.md` and `examples.md` were written alongside.

# 2026-08-06 (refinement round, fresh plan)

The user reframed the objective: this is not a native-tools feature but a
client-design improvement — transports keep wire-protocol knowledge only,
every tool-specific shape moves to tools/projectors. A grounded review of the
PRD against the code confirmed the design already matched, and folded in:

- An explicit "what stays in the transport" boundary section (routing branch,
  `tool_choice` strings/raw dicts, envelopes, the walk). The standard-call
  wire string stays an explicit branch — a lookup table would be equivalent
  denormalization (user's call).
- A fifth projector touchpoint, `project_tool_choice_to_llm`: the user chose
  to support forcing native tools now rather than defer. Transport resolves
  `{"name": …}` against declared tools and delegates; OpenAI natives force by
  type (`{"type": "apply_patch"}` — flagged as the one shape not yet
  validated live); Anthropic natives inherit the default.
- A parse-side projector compatibility check (registry hit must pass the
  `TOOL_PROJECTOR_BASE` isinstance check, else ignored like a miss).
- Registry-key decision recorded: key = canonical serialization
  discriminator, today equal to the wire type; a colliding future family adds
  its own wire→class parse map.
- `SerializeAsAny` extended to `ToolCallEndEvent.tool_call` and
  `FinishEvent.tool_calls`; agent-side containers noted as step-two work.
- `_coerce_tools` wrapping into `BadRequestError` recorded as a deliberate
  behavior fix (today leaks raw `ValidationError`); Faux passthrough and the
  all-four-families scope (chat completions, Bedrock get the walk + drop
  policy despite shipping no natives) made explicit.

A fresh `plan.md` was written from the refined PRD (old plan superseded as
`old_plan.md`).

# 2026-08-06 (implemented)

All 13 plan steps landed in one pass, suite green after each step. Both
gates held: the pre-existing 409 client tests pass byte-for-byte untouched
(two private-method signatures grew optional `lineage` params so
direct-call tests keep working; `ChatCompletionRequest` gained a
before-validator so dict-shaped tools still coerce at the DTO), and the
full repo suite (2247 tests, `filterwarnings=error`) is green with 67 new
tests — stream tests replay the live SSE captures (now also under
`tests/client/test_transports/*/captures/`). Shared projector mechanics
(`TOOL_PROJECTOR_BASE`, `_resolve_projector`, `_projector_for_call`,
`_native_projector_for_item`, `_declared_tool_named`) landed on
`ChatCompletionTransportMixin` rather than per-transport copies. Docs: new
"Provider-native tools" sections on `docs/client/06-tools.md`, a streaming
callout, roadmap update — snippets validated against live code. Final
verification: an end-to-end LIVE smoke through the implemented client —
multi-turn loops on both providers (OpenAI apply_patch forced + streamed +
shell with structured results; Anthropic text editor + bash), files
created on disk, correct final answers.
