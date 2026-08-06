# Extensible tool projection and provider-native tools

## Objectives

1. Remove every hardcoded tool assumption from the transports. Today each
   transport hardcodes five things: the declaration shape (`_project_tools`),
   the call parse (`function_call` / `tool_use` branches), the call replay
   (every `ToolCall` re-projects as a `function_call` item), the result
   shape (`_project_tool_message`), and the forced-tool shape
   (`_project_tool_choice`'s `{"name": …}` branch).
2. Make all five dynamic through one mechanism — a per-tool **projector** —
   so tool types we don't know today can plug in without transport changes.
3. Preserve the existing `Tool` / `ToolCall` / `ToolMessage` API for standard
   function tools, byte-for-byte on the wire.
4. Ship four provider-native tools on that mechanism: OpenAI Apply Patch,
   OpenAI Local Shell, Anthropic Text Editor, Anthropic Bash.
5. Keep declaration and execution separate: the client describes tools, parses
   calls, and projects results; the caller executes.

## Validated wire facts (live, 2026-08-06)

The design below was validated end to end against both live APIs with
prototype harnesses (see `examples.md` for the caller-visible version):

- **Anthropic**: native tools ride the existing `tool_use` / `tool_result`
  blocks. Parsing, replay, results, and streaming all work through today's
  canonical types unchanged; only the declaration shape is new.
- **OpenAI Responses**: native tools return new item types. Observed:

  ```python
  {"type": "apply_patch_call", "id": "apc_…", "call_id": "call_…",
   "status": "completed",
   "operation": {"type": "create_file", "path": "hello.txt", "diff": "+hi luca\n"}}

  {"type": "shell_call", "id": "sh_…", "call_id": "call_…", "status": "…",
   "action": {"commands": ["wc -c hello.txt"], "timeout_ms": 10000,
              "max_output_length": 10240}}
  ```

  Under `store: false` the native call item must be **replayed verbatim in
  `input` on every later turn** (verified accepted, item `id` included), and
  the result must be a matching native output item
  (`apply_patch_call_output` with `status: "completed" | "failed"`;
  `shell_call_output` with a structured per-command array, echoing
  `max_output_length` when the call carried it).
- **Streaming (Responses)**: the item skeleton arrives on
  `response.output_item.added` (status `in_progress`), the apply_patch diff
  streams as **raw text** deltas (`response.apply_patch_call_operation_diff.delta`
  — not JSON fragments), and the complete item arrives on
  `response.output_item.done`. Re-confirmed 2026-08-06 with forced streamed
  calls for BOTH OpenAI natives:
  - apply_patch: skeleton at `added` carries `id`, `call_id`, `status` AND a
    partial `operation` (`type` + `path` known, `diff: ""`) — the prebuilt
    call is buildable (and useful) at added-time.
  - shell: skeleton carries `call_id` with an EMPTY action
    (`{"commands": [], …nulls}`, `environment: null`); the command streams
    through its own raw-text events (`response.shell_call_command.added` /
    `.delta` / `.done`) — unhandled fall-through, same V1 rule as the
    apply_patch diff; the complete action (commands, model-chosen
    `timeout_ms`, `max_output_length`) arrives at `output_item.done`.
  - Full two-turn shell loop validated streamed: the PROJECTOR-shaped replay
    (`type`/`id`/`call_id`/`status`/`action`, no `environment` key) is
    accepted, with the structured `shell_call_output` echoing
    `max_output_length`.
  - `tool_choice` forcing accepted for both: `{"type": "apply_patch"}` and
    `{"type": "shell"}`.
  Raw captures for all four native tools live in `captures/*.sse` and double
  as stream-test fixtures. Both live Anthropic captures (bash AND text
  editor, the latter through a full call→result second turn) were replayed
  through the CURRENT client parser + accumulator unmodified: full JSON
  argument deltas, parsed arguments, `tool_use` terminal — "Anthropic needs
  zero streaming changes" is demonstrated, not assumed.
- Today's transport silently drops these items: empty content, no tool calls,
  `finish_reason="stop"` — a dead conversation reported as success.

## Architecture

### What stays in the transport

The goal is not "the transport knows nothing about tools" — it is that the
transport keeps only wire-protocol knowledge and zero per-tool shape:

- **Routing**: the standard-call wire string (the `function_call` /
  `tool_use` branch) hands off to the default projector; every other tool
  item type goes through the registry. Kept as an explicit branch — a
  per-transport lookup table would hold the same one string; pure
  denormalization, either works.
- **`tool_choice` strings and raw dicts** (`"auto"` mapping, passthrough).
  Only the `{"name": …}` forcing shape is per-tool and moves to projectors.
- **Envelopes**: Anthropic's `{"role": "user"}` wrapper around result
  blocks, chat completions' `tool_calls` array assembly.
- **The message walk**: lineage, the foreign-drop policy, attestation rules.

All four transport families adopt the full structure — default projector,
lineage walk, drop policy — including chat completions (inherited by
OpenRouter) and Bedrock, which ship no native tools: the walk is what
protects a provider switch, and moving the standard logic verbatim is where
the byte-for-byte guarantee is enforced by the existing payload tests.

### `BaseTool`

The common base for everything accepted through `tools=`.

```python
class BaseTool(BaseModel):
    def get_projector(self) -> ToolProjector | None:
        return None          # None -> use the transport's default projector
```

`Tool` becomes `Tool(BaseTool)` with its existing `name` / `description` /
`parameters` contract, unchanged. Raw dicts at the client boundary continue to
mean *standard* tools and coerce into `Tool` (validation failures wrap into
`BadRequestError` — today they leak the raw pydantic `ValidationError`;
wrapping is a deliberate fix). Native tools are always passed as instances; they carry
their configuration (`max_characters`, …) and a `name: ClassVar[str]` naming
the `ToolCall.name` their calls will surface with.

`ChatCompletionRequest.tools` becomes `list[BaseTool] | None`. (Verified:
pydantic passes subclass instances through the DTO untouched.)

### `ToolProjector`

One projector owns every wire touchpoint of one tool kind:

```python
class ToolProjector:
    def project_tool_to_llm(self, tool: BaseTool) -> dict: ...          # declaration
    def build_tool_call(self, item: dict) -> ToolCall: ...              # wire -> canonical call
    def project_tool_call_to_llm(self, tc: ToolCall) -> dict: ...       # replay call in input
    def project_tool_message_to_llm(
        self, msg: ToolMessage, call: ToolCall | None
    ) -> dict: ...                                                      # result -> wire
    def project_tool_choice_to_llm(self, tool: BaseTool) -> dict: ...   # forced tool_choice
```

`project_tool_message_to_llm` receives the originating `ToolCall` when the
transport found it in history (`None` otherwise): shell's result must echo
`max_output_length` from the *call's* action. On Anthropic the method returns
the `tool_result` block; the transport owns the `{"role": "user"}` envelope
around it.

`project_tool_choice_to_llm` is the shape that FORCES this one tool. The
transport resolves `tool_choice={"name": N}` against the declared tools
(first match on the tool's `name`; the collision case is already documented)
and delegates to that tool's projector. A name matching no declared tool
keeps today's blind standard shape — the API rejects it either way. Family
defaults are today's shapes; Anthropic natives inherit
(`{"type": "tool", "name": …}` already forces them), OpenAI Apply Patch /
Shell override to `{"type": "apply_patch"}` / `{"type": "shell"}`
(`apply_patch` forcing validated live on a streamed request, 2026-08-06).
`tool_choice` strings and raw dicts stay transport-level, unchanged.

Each transport defines its **own projector base class implementing all five
methods with today's standard function-tool behavior** (that logic moves out
of the transport methods and into the default projector, same package):
`OpenAIResponsesToolProjector`, `OpenAIToolProjector` (chat completions,
inherited by OpenRouter), `AnthropicToolProjector`, `BedrockToolProjector`.

A native projector subclasses its transport's base and overrides only what
differs. Anthropic Text Editor / Bash override *only*
`project_tool_to_llm`. OpenAI Apply Patch / Shell override all five.

### Projector resolution and compatibility

```python
def _resolve_projector(self, tool: BaseTool) -> ToolProjector:
    projector = tool.get_projector() or self._default_tool_projector()
    if not isinstance(projector, self.TOOL_PROJECTOR_BASE):
        raise BadRequestError(...)   # before any HTTP
    return projector
```

`TOOL_PROJECTOR_BASE` is a ClassVar on each transport. The check is
class-based, not string-based, so it composes with transport subclassing:
OpenRouter inherits `OpenAITransport`'s base and accepts the same projectors;
an Anthropic-native tool on any OpenAI-family transport fails before the
request. (String ids would not compose — OpenRouter redefines
`transport_id`.)

### Native `ToolCall` subclasses

A native call is a `ToolCall` subclass that overrides the `type` literal and
carries the extra wire identity the replay needs:

```python
class ApplyPatchToolCall(ToolCall):
    type: Literal["apply_patch_call"] = "apply_patch_call"
    item_id: str | None = None      # the wire item "id" (apc_…); replayed
    status: str = "completed"       # wire item status (in_progress while streaming)

    @property
    def operation(self) -> dict:    # typed accessor; storage is `arguments`
        return self.arguments
```

Decisions inside this shape:

- **`arguments` is the storage; `operation` / `action` are properties.** The
  payload lives once, generic consumers (logging, rendering, a future agent
  permission layer) keep reading `arguments`, serialization has a single
  source, and `parse_arguments` keeps working.
- **`name` is synthesized** from the tool kind (`"apply_patch"`, `"shell"`)
  since these wire items carry none. It matches the tool class's
  `name` ClassVar, so caller registries dispatch uniformly. Anthropic native
  calls keep their wire names (`"bash"`, `"str_replace_based_edit_tool"`) —
  and need **no subclass at all**: base `ToolCall` round-trips them (validated).
- `id` remains the `call_id` (what results match on), as for function calls.

### Native `ToolMessage` subclasses — only when the wire needs structure

`apply_patch_call_output` maps onto the base `ToolMessage`: `is_error` →
`status` (`"failed"` / `"completed"`), text content → `output`. No subclass —
a `status` field would duplicate `is_error` and the two would drift.

`shell_call_output` cannot ride prose (per-command stdout/stderr/exit codes),
so it gets a typed message:

```python
class ShellCommandResult(BaseModel):
    stdout: str
    stderr: str
    outcome: ShellExitOutcome | ShellTimeoutOutcome   # discriminated on "type"

class ShellToolMessage(ToolMessage):
    type: Literal["shell_call_output"] = "shell_call_output"
    results: list[ShellCommandResult]
    content: str | list[TextBlock | ImageBlock] = ""   # optional human summary
```

The shell projector requires a `ShellToolMessage`; handing it a plain
`ToolMessage` raises `BadRequestError` naming the expected type.

### The wire-type registry (parse, replay, results — one lookup)

A module-level registry maps native wire item types to their call classes:

```python
NATIVE_TOOL_CALL_TYPES: dict[str, type[ToolCall]] = {}
```

Subclasses self-register via `__pydantic_init_subclass__` (key = the `type`
literal's default; duplicate keys from different classes are an error). The
key is the canonical serialization discriminator, which for every first-party
native equals the wire item type; if a future family's wire ever reuses a
taken string, that family adds its own wire→class map in its parse — the
global registry stays the serialization authority. Each
call class binds its projector (`projector_class` ClassVar) — tool, call
class, and projector live together in one native-tools module per transport
package.

This one registry replaces the sketch's `get_tool_message_matching_values`
hook and drives all three transport moments:

1. **Parse**: unknown output item type → `NATIVE_TOOL_CALL_TYPES.get(type)`
   → `projector.build_tool_call(item)`. The hit's projector must pass the
   same `TOOL_PROJECTOR_BASE` isinstance check as declarations; an
   incompatible hit is ignored like a miss — a foreign family's registration
   can never mint calls on this wire. Miss → ignore (hosted-tool items,
   unchanged behavior).
2. **Replay**: projecting an `AssistantMessage`, a `ToolCall` whose class is
   registered projects through its projector (`project_tool_call_to_llm`);
   plain `ToolCall`s keep projecting as `function_call` via the default.
3. **Results**: while walking the message list, the transport records
   `{call_id: (projector, call)}` from each assistant message it projects; a
   `ToolMessage` whose `tool_call_id` has native lineage projects through
   that projector, which also receives the call. Calls always precede their
   results in a valid conversation; an unmatched result falls back to the
   default (`function_call_output`, `call=None`). This works even though
   apply_patch results are plain `ToolMessage`s.

A native call whose projector does not pass the *current* transport's
compatibility check (a `ShellToolCall` in history after `/model` switched the
session to Anthropic) is **dropped at projection, together with its result**
— the same policy, and the same rationale, as a foreign thinking-block
attestation: one lost exchange beats a 400 that kills the conversation.

Dropping can empty a whole assistant turn: an OpenAI native turn is typically
`[reasoning, shell_call]`, and after a provider switch both blocks are
foreign. An assistant message that projects to **zero wire blocks is omitted
entirely** on envelope wires (Anthropic / chat completions / Bedrock reject
empty content); on Responses the items simply never appear. Today's code has
this hole latently — a thinking-only foreign turn already projects
`content: []` — the walk closes it.

Registration happens at import: the native-tools modules are imported by
their transport packages, which `luca.client` imports eagerly — so any
`luca.client` import registers all first-party native types. Third-party
native tools register when their module imports (same rule as any plugin
type: you cannot deserialize a class you never imported).

### Serialization / deserialization

Conversations containing native calls must survive a JSON round trip (the
agent will persist them in step two; client users already persist message
lists). Two mechanisms:

- **Validation dispatch**: `ToolCall` and `ToolMessage` each get a
  `model_validator(mode="wrap")` that, when validating against the *base*
  class, looks up the payload's `type` in the registry and dispatches to the
  subclass. `AssistantMessage.content`'s plain union and the `Message` union
  (discriminated by `role`) then accept native payloads with no change to
  their annotations. Base `ToolMessage` gains no `type` field — payloads
  without one validate as the base, so existing serialized data is untouched.
- **Duck-typed dump**: `AssistantMessage.content` annotates its `ToolCall`
  member as `SerializeAsAny[ToolCall]` so subclass fields survive
  `model_dump()`. (Without it pydantic dumps per the declared base and drops
  them — the warning it emits would fail the test suite's
  `filterwarnings = error`, which is the canary.) The same treatment applies
  to the other containers declared as base `ToolCall`:
  `ToolCallEndEvent.tool_call` and `FinishEvent.tool_calls`.

`ContentBlock` (the exported discriminated alias) keeps enumerating the
standard blocks only; it is not used by any model field. Native call types
live outside it.

Step two (the agent) applies the same `SerializeAsAny` treatment to its own
`Message`-annotated containers when sessions start persisting native calls —
noted here so it is not rediscovered.

## Streaming

Anthropic: unchanged — native `tool_use` streams through the existing
`tool_call_start` / `tool_call_delta` / `tool_call_end` events with JSON
argument deltas (validated live).

OpenAI Responses, V1 rule: **native calls stream as start + end, no public
argument deltas.** The diff deltas are raw text, which violates the
`partial_arguments`-accumulates-JSON contract; and the complete operation is
available on `output_item.done` anyway. Mechanics: the raw vocabulary grows
two optional fields — `RawBlockStart.prebuilt: ToolCall | None` (the
transport builds the typed subclass at `output_item.added`, status
`in_progress`) and `RawBlockStop.replacement: ToolCall | None` (the complete
call built at `output_item.done`; the accumulator swaps it in and emits
`ToolCallEnd` with final arguments, skipping the JSON parse). The accumulator
stays transport-agnostic — it never sees a projector. Incremental diff
rendering is deferred until a consumer needs it.

A stream that goes terminal mid-native-call closes the block with whatever
was accumulated (no `replacement`): empty arguments, same degenerate outcome
as a truncated function call today.

## Initial scope

| Tool | Declaration emitted | Call surfaced as | Result projected as |
|---|---|---|---|
| `ApplyPatchTool()` | `{"type": "apply_patch"}` | `ApplyPatchToolCall`, name `"apply_patch"`, arguments = operation | `apply_patch_call_output` from plain `ToolMessage` (`is_error` → status) |
| `LocalShellTool()` | `{"type": "shell", "environment": {"type": "local"}}` | `ShellToolCall`, name `"shell"`, arguments = action | `shell_call_output` from `ShellToolMessage.results` |
| `TextEditorTool(max_characters=None)` | `{"type": "text_editor_20250728", "name": "str_replace_based_edit_tool", "max_characters"?}` | base `ToolCall` (wire name) | base `tool_result` |
| `BashTool()` | `{"type": "bash_20250124", "name": "bash"}` | base `ToolCall` (wire name) | base `tool_result` |

Versions are pinned to the current provider versions; no version knob —
every older Anthropic tool version targets a retired model family. OpenAI
native tools exist only on the Responses transport. Implementations live in
the transport packages, re-exported from `luca.client.providers.openai` /
`luca.client.providers.anthropic`.

## Edge cases

| Case | Behavior |
|---|---|
| Native tool on an incompatible transport | `BadRequestError` before any HTTP (projector-base isinstance check) |
| `{"type": "apply_patch"}` as a raw dict in `tools=` | `BadRequestError` — dicts mean standard tools; native tools are instances |
| Plain `ToolMessage` answering a shell call | `BadRequestError` at projection, naming `ShellToolMessage` |
| `ToolMessage` whose `call_id` matches no native call in history | Default projection (`function_call_output`) |
| Custom `Tool(name="apply_patch")` alongside `ApplyPatchTool()` | Allowed; calls are distinguished by wire item type. Caller registries that key on name collide — documented, not policed |
| Foreign native call in history (provider switched mid-session) | Call and its result dropped at projection (foreign-attestation policy) |
| Assistant turn whose every block was dropped (foreign thinking + foreign natives) | Whole wire message omitted on envelope wires (empty content is a 400); on Responses the items just don't appear |
| Deserializing a native call without its module imported | `ValidationError`; importing `luca.client` registers all first-party types |
| Unknown output item type not in the registry | Ignored (hosted-tool behavior, unchanged) |
| Registry hit whose projector targets another family | Ignored like a miss (parse-side isinstance check) |
| `tool_choice={"name": "apply_patch"}` with `ApplyPatchTool()` declared | Forced through the tool's projector: `{"type": "apply_patch"}` on the wire |
| `tool_choice={"name": …}` matching no declared tool | Today's blind standard shape, unchanged (the API rejects it; not policed) |
| Native tools through `FauxTransport` | Pass through untouched; faux never projects tools |
| Stream truncated mid native call | Block closed with empty arguments, `finish` reflects the terminal |

## Responsibilities

- `BaseTool` — the `tools=` contract and the projector hook.
- `Tool` — the standard function-tool data model, unchanged.
- `ToolProjector` — all five wire touchpoints of one tool kind.
- Transport — resolves projectors, enforces compatibility, walks messages,
  resolves `tool_choice` names against the declared tools, owns its default
  projector and everything non-tool (headers, errors, finish classification,
  streams).
- Native modules — tool + call class + (when needed) message class +
  projector, registered at import.
- Caller — executes: applies diffs, owns the persistent shell session and
  restarts, timeouts, sandboxing, truncation, honest failure messages.
