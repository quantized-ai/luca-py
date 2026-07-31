# Implementation proposal — tool structured output

Reads `prd.md`. Six steps, in order. Steps 1–2 are the whole feature; 3–6 are
tests and docs.

## 1. `luca/agent/core/models.py`

### 1a. `ToolSpec.output_schema`

Insert directly after `input_schema` (before `metadata`):

```python
    # The shape of the machine-readable result this tool CAN produce, as a
    # JSON Schema dict — the declaration `ExecutionResult.structured_content`
    # is the payload for. Optional and ADVISORY in both directions: nothing in
    # the framework validates a payload against it, and no provider luca talks
    # to accepts an output schema on a function tool, so it is never sent on
    # the wire. It advertises to the APPLICATION (a registry, a UI, an MCP
    # bridge mapping `outputSchema`), not to the model. `None` = the tool
    # declares nothing.
    output_schema: dict | None = None
```

Extend the class docstring's advertisement/snapshot paragraph: `output_schema`
is neither — it is an application-facing declaration. One sentence.

### 1b. `ExecutionResult.structured_content`

Insert directly after `content` (before `metadata`):

```python
    # The tool's machine-readable payload, ideally conforming to its
    # `ToolSpec.output_schema`. BEST EFFORT: never validated, never projected
    # to the model, never counted toward context. `content` remains the sole
    # model-facing channel — a tool that wants the model to see this
    # serializes it into `content` itself. Distinct from `metadata`, which
    # stays free-form bookkeeping no schema describes.
    structured_content: dict | None = None
```

Nothing else in `models.py` changes. `spec_id()` picks the new field up
automatically (it dumps the whole model).

## 2. `luca/agent/contrib/tools.py`

### 2a. `Tool.output_schema`

Declare it beside `Args`:

```python
    Args: ClassVar[type[BaseModel]]
    # Optional: the Pydantic MODEL CLASS describing the machine-readable result
    # this tool can produce — usually a module-level model bound here
    # (`output_schema = WeatherReport`), which keeps it importable so a
    # consumer can validate a payload back through it. `get_tool_spec()` turns
    # it into `ToolSpec.output_schema`, which is the JSON Schema DICT derived
    # from it — same name, different type, one level apart.
    #
    # Declaring it populates NOTHING: a tool that produces a payload overrides
    # `execute()` and sets `structured_content=` on the returned
    # `ExecutionResult`. Nothing validates one against the other.
    output_schema: ClassVar[type[BaseModel] | None] = None
```

In `get_tool_spec()`, add one argument:

```python
            input_schema=self.Args.model_json_schema(),
            output_schema=(self.output_schema.model_json_schema() if self.output_schema is not None else None),
```

Extend `get_tool_spec()`'s docstring: the spec's `output_schema` is derived from
the ClassVar of the same name the way `input_schema` is derived from `Args`, is
`None` when undeclared, and is a dict where the ClassVar is a class (PRD D3's
accepted collision — say the types out loud, here and on the ClassVar).

Extend the module docstring's "WHAT A TOOL RECEIVES" neighbourhood with a short
"WHAT A TOOL RETURNS" note: `_execute` is the text path; `execute` is where a
tool that declared an `output_schema` sets `structured_content`; the framework
never checks one against the other.

### 2b. `output=` on `tool_class()` / `tool()`

Add to both signatures, after `arguments`:

```python
    output: type[BaseModel] | dict[str, Any] | None = None,
```

In `tool_class`, mirror the existing `arguments` normalization:

```python
    output_model = None
    if output is not None:
        if isinstance(output, type) and issubclass(output, BaseModel):
            output_model = output
        else:
            output_model = create_model(
                f"{name}_output",
                __config__=ConfigDict(extra="forbid"),
                **output,
            )
```

Put `"output_schema": output_model` into `ns` **unconditionally** (value `None`
when not given) so it participates in the existing `collisions` check — that is
what makes `output=` + `class_attrs={"output_schema": …}` raise, and setting the
ClassVar to `None` explicitly is identical to inheriting the base default.

`tool()` just forwards `output=output`.

Docstring additions on `tool_class` (both are load-bearing):
- `output` takes the same two forms as `arguments`, and **a dict is always a
  `create_model` field spec, never a raw JSON Schema**.
- The factories wire only the simple text path, so a factory-built tool
  *declares* `output_schema` but cannot populate `structured_content`; hand-write
  the class or pass a `bases=` mixin overriding `execute` for that.

## 3. `tests/agent/test_models.py`

**Fix two existing tests** (mechanical, both are pinned literals):

- `test_spec_id_is_the_sha256_hex_of_the_canonical_json` (~L211): add
  `"output_schema":null,` to the canonical string, sorted between
  `"namespace":null,` and `"timeout_in_ms":null`.
- `test_a_serialized_session_carries_no_inline_tool_spec` (~L548): add
  `"structured_content": None,` to the `result` dict, after `"content"`.

**Add**, next to the existing spec/result cases:

| Test | Asserts |
|---|---|
| `test_tool_spec_defaults_to_a_null_output_schema` | full-object: a spec built without it equals one with `output_schema=None` |
| `test_a_spec_declaring_an_output_schema_round_trips` | `ToolSpec.model_validate_json(dump) == original` |
| `test_declaring_an_output_schema_mints_a_new_spec_id` | same spec ± `output_schema` → different `spec_id()` (sibling of `test_a_reworded_description_mints_a_new_spec_id`) |
| `test_execution_result_defaults` (extend the existing one) | `structured_content is None` in the full-object literal |
| `test_an_execution_result_carrying_structured_content_round_trips` | round-trips inside a full `AgentSession` dump/load |

## 4. `tests/agent/contrib/test_tools.py`

| Test | Asserts |
|---|---|
| `test_get_tool_spec_stamps_the_declared_output_schema` | full `ToolSpec` equality for a `Tool` binding `output_schema` to a module-level model (sibling of `test_get_tool_spec_stamps_the_declared_timeout`) |
| `test_get_tool_spec_leaves_output_schema_null_when_undeclared` | the existing `test_get_tool_spec_stamps_the_identity_classvars_and_the_args_schema` already covers this via full-object equality — extend rather than duplicate if so |
| `test_output_accepts_an_existing_model_as_is` | `tool(output=Model)` → `type(built).output_schema is Model` (sibling of `test_arguments_accepts_existing_model_as_is`) |
| `test_generated_output_schema` | `tool(output={"x": (int, ...)})` → compiled model, `extra="forbid"`, schema on the spec |
| `test_output_collides_with_class_attrs` | `ValueError` (sibling of `test_class_attrs_collision_raises`) |
| `test_a_tool_may_return_structured_content` | a `Tool` overriding `execute()` returns the full expected `ExecutionResult` incl. `structured_content` |

## 5. Cross-cutting guard tests

Two one-liners that pin D1. Both should PASS unchanged after steps 1–2 — they
exist to fail loudly if someone later "helpfully" wires structured content to
the wire.

- `tests/agent/test_projection.py` —
  `test_structured_content_never_reaches_the_tool_message`: a COMPLETED
  execution ± `structured_content` projects to the identical `ToolMessage`.
- `tests/agent/test_context_manager.py` —
  `test_structured_content_is_not_counted_toward_context`: `calculate_context`
  returns the same int ± `structured_content`.

## 6. Docs (`docs/llm.txt` first)

| File | Change |
|---|---|
| `docs/agent/03-tools.md` | `ToolSpec` field table gains `output_schema` (§1); alongside "the core never validates arguments against `input_schema`", add the twin: it never validates `structured_content` against `output_schema` either |
| `docs/agent/02-data-model.md` | `ToolSpec` section + field table; the `ExecutionResult` row/table; refresh the serialized JSON sample (~L540) with both nulls |
| `docs/agent/contrib/tools/README.md` | §5 "What lands in the `ToolSpec`"; the factory section documents `output=` **and its limitation**; **plus the required `GetWeatherTool` example below** — this page is the feature's primary doc |
| `docs/agent/11-context-and-usage.md` | the `process_tool_output` worked example rebuilds `ExecutionResult(...)` — note that a rebuild drops `structured_content` unless carried over |
| `AGENTS.agent.md` | "Tool identity" section: `ToolSpec.output_schema` is a third role (application-facing declaration, never on the wire); `Tool.output_schema` is the model class it derives from, mirroring `Args → input_schema` |

### The required example (PRD §"Required docs example")

`docs/agent/contrib/tools/README.md` gets a new section — after §4 "Rich
results", which it builds on — carrying `GetWeatherTool` end to end. Not
optional and not reducible to a field table: it is the only place a reader
learns that **declaring a schema and producing a payload are two separate
acts**. Five parts, in order:

1. A module-level `class WeatherReport(BaseModel)` bound with
   `output_schema = WeatherReport`. Say why module-level is preferred: the
   model stays importable, so a consumer validates the payload back through it
   (`WeatherReport.model_validate(...)`) instead of indexing raw dict keys.
2. `execute()` — **not** `_execute`, and call that out — returning both
   channels: the sentence in `content`, `reading.model_dump()` in
   `structured_content`.
3. The resulting `ToolSpec`, with `input_schema` and `output_schema` side by
   side so the `Args`/`output_schema` symmetry is visible.
4. The resulting `ExecutionResult`.
5. The two consumers contrasted, which is the whole point of D1: the model
   receives a `ToolMessage` built from `content` alone — byte-identical to the
   same tool without any of this — while the app reads
   `execution.result.structured_content`, with
   `execution.tool_spec.output_schema` as the contract that survives a process
   boundary (a serialized session, an MCP server, a web UI).

Two ⚠️ callouts belong in that section: nothing validates the payload against
the schema (a tool returning `{"degrees": "warm"}` still records `COMPLETED`
and stores it verbatim), and the ClassVar/field type collision — the class on
`Tool`, the dict on `ToolSpec`.

Schema output shown in the docs must be the real `model_json_schema()` output,
generated and pasted, not hand-written.

## Risks

Low. The only durable consequence is the `spec_id()` hash change (PRD §Edge
cases 1) — expected, absorbed by the content-hash design, no migration in V1.

The one thing to get wrong quietly: putting `"output_schema"` into `ns`
conditionally in step 2b, which would silently skip the collision check.

## Order

1 → 2 → 3 → 4 → 5 → `uv run py.test tests/` → 6 → `uv run ruff check --fix` +
`uv run ruff format`.

Steps 1 and 2 are independent of each other; 3–5 depend on both.
