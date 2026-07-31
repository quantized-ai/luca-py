# Tool structured output

## Objective

Let a tool declare the SHAPE of a machine-readable result it can produce, and
let one execution carry that result alongside its text.

Two optional fields, mirroring MCP's `outputSchema` / `structuredContent`:

- `ToolSpec.output_schema: dict | None` — the declaration.
- `ExecutionResult.structured_content: dict | None` — the payload.

Plus the contrib ergonomics to produce the first from a Pydantic model, exactly
as `Args` produces `input_schema` today.

Best effort throughout: nothing validates the payload against the schema,
nothing requires either field, and an execution with a schema and no payload is
normal.

## Why

Today a tool's only structured channel is `ExecutionResult.metadata` — free-form,
registry-owned, explicitly "never interpreted by the core", and undeclared, so
a consumer can only guess what a given tool puts there. `output_schema` gives
that channel a contract a consumer can read off the durable session.

The concrete driver is MCP. An MCP server declares `outputSchema` per tool and
returns `structuredContent` per call; without a landing place both are dropped
at the boundary. This PRD builds the landing place. **No MCP registry ships
here** — see Non-goals.

## Decisions

### D1 — `structured_content` never reaches the model

`ExecutionResult.content` stays the SOLE model-facing channel.
`ConversationProjector.project_tool_execution` is untouched: a COMPLETED
execution projects `result.content` and nothing else, whether or not
`structured_content` is set.

A tool that wants the model to see the payload serializes it into `content`
itself — which is what MCP tells servers to do, for the same reason.

Follows from this, and both are already true, so both are decisions rather than
changes: `ContextManager.calculate_context` keeps counting `result.content`
only (a non-model-facing payload must not inflate a context estimate), and no
transport changes (see F2).

### D2 — `structured_content` and `metadata` coexist; nothing migrates

Different jobs:

| | `metadata` | `structured_content` |
|---|---|---|
| Declared? | no | yes, via `output_schema` |
| Owner | registry / app bookkeeping | the tool's own result payload |
| Typical | `{"exit_code": 0}`, `{"preview": …}`, `{"diff": …}` | the tool's declared output model, dumped |

Neither supersedes the other. The seven shell tools keep every existing
`metadata=` payload untouched, and no contrib tool declares an output model in
this PRD.

### D3 — `output_schema` is a declared ClassVar, and the tool populates the payload itself

On `luca.agent.contrib.tools.Tool`:

```python
output_schema: ClassVar[type[BaseModel] | None] = None
```

Declared and defaulting to `None` — **not** a `hasattr` probe. `Args` is
declared, and a base-class convention that exists only when a subclass happens
to define the attribute is the implicit coupling this codebase avoids. (The
contrast is `get_approval_context`, which is duck-typed precisely because the
CORE must not know it exists; here the reader is the base class itself.)

**snake_case, not `OutputSchema`,** because the preferred way to declare one is
binding a module-level model:

```python
class WeatherReport(BaseModel): ...

class GetWeatherTool(Tool):
    output_schema = WeatherReport
```

`OutputSchema = WeatherReport` reads as a class alias or a re-export; a
snake_case attribute reads as what it is — a binding. `Args` keeps CapWords
because it is overwhelmingly written as a nested `class Args(BaseModel)`
(`contrib/shell/tools.py:315` and every sibling), where CapWords is correct.

> ⚠️ **Accepted collision.** `Tool.output_schema` is a Pydantic model CLASS;
> `ToolSpec.output_schema` is a JSON Schema DICT. Same name, two types, one
> `get_tool_spec()` line apart — and deliberately unlike `Args → input_schema`,
> which differ in name precisely because they differ in type. The trade was
> made for the declaration site, which is read far more often than
> `get_tool_spec()`. Both docstrings must state the type explicitly; this is
> not a defect to be tidied away later.

`get_tool_spec()` stamps it:

```python
output_schema=self.output_schema.model_json_schema() if self.output_schema is not None else None
```

Producing the payload is the tool's job: override `execute()` and set
`structured_content=` on the returned `ExecutionResult`. **No new override
point** — `_execute` stays the `-> str` path and never produces structured
content. A base-class helper that derives both the text and the payload from a
returned model would have to invent a text-rendering policy, and there is no
second case asking for one yet.

### D4 — `tool()` / `tool_class()` gain `output=`

```python
output: type[BaseModel] | dict[str, Any] | None = None
```

Same two forms and the same rule as `arguments`: a ready `BaseModel` class is
bound to `output_schema` verbatim; a dict is a `create_model` field spec
compiled into an `extra="forbid"` model named `f"{name}_output"`. **A dict is
ALWAYS a field spec, never a raw JSON Schema** — one rule for both parameters.
`None` (the default) leaves the ClassVar `None`, so the spec's `output_schema`
is `None` too.

`output` becomes a factory-managed name, so passing it together with
`class_attrs={"output_schema": …}` raises the existing collision `ValueError`.

> ⚠️ **Known limitation, accepted.** The factories only wire the simple text
> path (`execute` → `_execute` → `str`), so a factory-built tool can *declare*
> `output_schema` but cannot populate `structured_content`. Declaring alone is
> still useful (an app reads the spec; the schema is part of the tool's
> identity), and the escape hatch is unchanged: hand-write the class, or pass a
> `bases=` mixin that overrides `execute`.

### D5 — Types

`dict | None` for both, defaulting to `None`.

`dict` rather than "any JSON value": MCP restricts `structuredContent` to an
object, and an object-shaped payload is what a JSON Schema of `type: object`
declares. `None` (not `{}`) is the absent state — the two are different facts,
same as `CompactionEntry.parts`.

No validator on either field. `structured_content` is exactly as JSON-clean as
the tool makes it, which is the existing contract for `ExecutionResult.metadata`
and `ToolSpec.metadata`.

### D6 — Names

Keep `output_schema` and `structured_content`. They are MCP's `outputSchema`
and `structuredContent` in this codebase's casing, and matching an external
protocol we intend to bridge beats a locally prettier name.

## Codebase findings

### F1 — The client's structured output is an unrelated mechanism

`luca.client` structured output is `ChatCompletionRequest.response_format`: it
constrains the ASSISTANT's reply text to a JSON Schema, is projected per
transport (`text.format` on Responses, `output_config.format` on Anthropic,
nested `json_schema` on chat completions), and is read back through
`response.parse()`. It shares nothing with a tool declaring its result shape but
the word "structured". There is no compatibility question to answer and nothing
to reuse.

### F2 — No provider luca talks to accepts a tool output schema

Every transport projects a tool as name + description + input schema, and
nothing else:

| Transport | Emitted per tool |
|---|---|
| `anthropic` (`transport.py:325`) | `name`, `description`, `input_schema` |
| `openai_responses` (`transport.py:302`) | `type`, `name`, `description`, `parameters` |
| `openai` (chat, + `openrouter` subclass) | `function: {name, description, parameters}` |
| `bedrock` (Converse) | `toolSpec: {name, description, inputSchema}` |

OpenAI, Anthropic and Bedrock have no output-schema field on a function tool.
(Gemini's `FunctionDeclaration.response` is the only mainstream one that does;
luca has no Gemini transport.)

**So `output_schema` is advertised to the APPLICATION, not to the model.** The
draft's "the tool then advertises that it can produce structured output" reads
like a wire change; it is not one. `luca.client` needs zero changes, and
`adapter.tool_spec_to_luca_tool` stays as it is.

### F3 — The change is contained

Two `ToolSpec(` construction sites exist in library code: the class itself and
`Tool.get_tool_spec()`. `ToolExecuted` already carries the whole `ToolExecution`
deep snapshot, so `structured_content` reaches every event consumer for free.

## Scope

**Changes**

| File | Change |
|---|---|
| `luca/agent/core/models.py` | `ToolSpec.output_schema`; `ExecutionResult.structured_content` (+ docstrings) |
| `luca/agent/contrib/tools.py` | `Tool.output_schema` ClassVar; `get_tool_spec()` stamps `ToolSpec.output_schema`; `output=` on `tool()` / `tool_class()` |
| `tests/agent/test_models.py` | new cases + two mechanical fixtures (below) |
| `tests/agent/contrib/test_tools.py` | new cases |
| `docs/` | 4 pages (see implementation proposal), incl. the **required worked example** below |
| `AGENTS.agent.md` | "Tool identity" section |

### Required docs example

`docs/agent/contrib/tools/README.md` MUST carry a complete worked example, not
just field tables. It is the only place a reader learns that declaring a schema
and producing a payload are two separate acts. It must show, end to end:

1. A module-level output model (`class WeatherReport(BaseModel)`) bound with
   `output_schema = WeatherReport` — the preferred form, and the one that makes
   the model importable so a consumer can validate the payload back through it.
2. `execute()` (not `_execute`) returning BOTH channels: human/model text in
   `content`, `reading.model_dump()` in `structured_content`.
3. The resulting `ToolSpec` — `input_schema` and `output_schema` side by side.
4. The resulting `ExecutionResult`.
5. The two consumers contrasted: the model receives a `ToolMessage` built from
   `content` alone, byte-identical to the same tool without any of this; the app
   reads `execution.result.structured_content`, with
   `execution.tool_spec.output_schema` as the cross-process contract.

`GetWeatherTool` is the agreed example.

**Explicitly unchanged**: `luca/client/**` (all of it), `adapter.py`,
`projection.py`, `context_manager.py`, `events.py`, `runner.py`, `ledger.py`,
`tool_registry.py`, `contrib/simple_tool_registry/`, `contrib/shell/`,
`contrib/memory/`, `contrib/tui/`.

## Non-goals

- Any validation of `structured_content` against `output_schema` — modes and
  enforcement are deferred (`future.md`).
- Sending `output_schema` on any wire, or a `luca.client.Tool` field for it.
- An MCP registry, or any consumer of either field. **V1 ships two fields
  nothing in the repo reads.** That is the intent, not an oversight.
- A TUI change.
- Migrating existing `metadata` payloads.
- Strict-mode rewriting of `output_schema` (`strictify_json_schema` is a
  wire-projection helper for a schema being sent; nothing is being sent).

## Edge cases and consequences

1. **`spec_id()` changes for every spec.** `output_schema` joins the hashed
   payload, so existing session files mint new spec ids on next write. Harmless
   by design — a content hash's only failure mode is a redundant row, and the
   old row stays resolvable by the executions pointing at it. V1, unreleased: no
   migration.
2. **Two mechanical test updates.**
   `test_spec_id_is_the_sha256_hex_of_the_canonical_json` pins the canonical
   JSON byte-for-byte and must gain `"output_schema":null` (sorted between
   `"namespace"` and `"timeout_in_ms"`);
   `test_a_serialized_session_carries_no_inline_tool_spec` asserts a full dumped
   execution and must gain `"structured_content": null` inside `result`.
3. **Serialized shape grows two nulls** — `"output_schema": null` per
   `tool_specs` row, `"structured_content": null` per stored result. The JSON
   sample in `docs/agent/02-data-model.md` needs refreshing.
4. **A custom `process_tool_output` that REBUILDS the result drops the
   payload.** The shipped default is identity, so nothing in-tree is affected —
   but the worked example in `docs/agent/11-context-and-usage.md` constructs a
   fresh `ExecutionResult(...)` and would silently lose `structured_content`.
   Call it out where that hook is documented.
5. **Schema declared, payload absent** is legal and expected (`_execute`-only
   tools, error paths, a tool that only sometimes has structured data). Nothing
   warns.
6. **Payload set, no schema declared** is equally legal — the fields are
   independent.
7. **`is_error=True` with `structured_content`** is allowed; a tool may return a
   structured failure. No rule.
8. **A non-terminal or non-COMPLETED execution** has no `ExecutionResult` at
   all, so the question of a payload does not arise.

## Acceptance criteria

- `ToolSpec(output_schema={…})` round-trips through JSON and yields a
  `spec_id()` distinct from the same spec without it.
- `ExecutionResult(structured_content={…})` round-trips inside a full
  `AgentSession` dump/load.
- A `Tool` binding `output_schema` to a model stamps
  `spec.output_schema == TheModel.model_json_schema()`; one that does not stamps
  `None`.
- `tool(output=SomeModel)` and `tool(output={"x": (int, ...)})` both produce a
  spec with a non-null `output_schema`; `tool()` with no `output=` produces
  `None`.
- `tool(output=…, class_attrs={"output_schema": …})` raises `ValueError`.
- A COMPLETED execution carrying `structured_content` projects to a `ToolMessage`
  **byte-identical** to the same execution without it — the wire never sees it.
- `ContextManager.calculate_context` returns the same count with and without
  `structured_content` — context never counts it.
- `uv run py.test tests/` green (warnings are errors); `uv run ruff check --fix`
  and `uv run ruff format` clean.

## Open questions

None blocking.
