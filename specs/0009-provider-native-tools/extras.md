# 0009b — The generic form: `extras` on `ToolCall` / `ToolMessage`

Second representation for provider-native tool calls and results. The typed
subclasses (`ApplyPatchToolCall`, `ShellToolMessage`, …) stay exactly as they
are; this adds a canonical-type-only form that carries the same information in
a free-form `extras` dict, and makes the two provably interchangeable.

---

## Why

Today the only way to express a native call or result is a provider-specific
Python class:

```python
ApplyPatchToolCall(id="call_1", name="apply_patch", arguments={...},
                   item_id="apc_1", status="completed")
ShellToolMessage(tool_call_id="call_2", results=[ShellCommandResult(...)])
```

That forces every consumer of the SDK to import provider classes to say
something the canonical types could carry. Three concrete costs:

1. **`luca.agent` must know each native class by name.** Spec 0010's
   `project_call` is a chain of `if spec.name == "openai_apply_patch": return
   ApplyPatchToolCall(...)`. With `extras` that whole per-native branch
   collapses to one generic construction.
2. **Persistence depends on Python identity.** A stored session only restores
   its typed subclasses if the defining module was imported; the same session
   is unreadable to anything that is not this Python process.
3. **Every new native tool is a new class** before anyone can even *replay* a
   call the provider already sent.

## Why not (the honest limits)

`extras` does **not** make an unseen native tool work end to end. Declaring a
tool still needs a `BaseTool` + `ToolProjector` pair, because only the
projector knows the wire shape. What `extras` removes is the need for
*consumers* to import the class — not the need for *someone* to define the
mapping. A fully generic `NativeTool(declaration=…, call_type=…)` is a
possible follow-up; it is out of scope here and nothing below precludes it.

---

## The contract

`ToolCall.extras: dict[str, Any]` and `ToolMessage.extras: dict[str, Any]`,
both defaulting to `{}`. Free-form and inert, with **one reserved key**:

| Key | Meaning |
|---|---|
| `custom_type` | The native `type` literal (`"apply_patch_call"`, `"shell_call_output"`, …). Names the registered class this call/result really is. |

Every other key is a **field of that class**, by name. Not a raw wire field:
`item_id`, not the wire's `id` (which the canonical `ToolCall.id` already
owns).

```python
ToolCall(id="call_1", name="apply_patch",
         arguments={"type": "create_file", "path": "hello.txt", "diff": "+hi\n"},
         extras={"custom_type": "apply_patch_call", "item_id": "apc_1", "status": "completed"})

ToolMessage(tool_call_id="call_2", content="",
            extras={"custom_type": "shell_call_output",
                    "results": [{"stdout": "5\n", "stderr": "",
                                 "outcome": {"type": "exit", "exit_code": 0}}]})
```

### Two methods, one invariant

```python
ToolCall.as_native()    # extras form  -> the registered subclass instance
ToolCall.as_generic()   # subclass     -> base ToolCall + extras
```

(identical pair on `ToolMessage`), with

```
call.as_generic().as_native() == call
```

`as_generic()` derives `extras` mechanically: `custom_type` is the subclass's
`type` default, and every field the subclass declares beyond the base becomes
one entry. `as_native()` is the exact inverse. Neither is provider-aware —
they are pure type-layer conversions.

Both return `self` when there is nothing to do: `as_native()` on a call with no
`custom_type` (or one that is already native — its own `extras` are then
inert), `as_generic()` on a call that is already the base class.

### Errors

| Case | Result |
|---|---|
| `custom_type` naming an unregistered type | `BadRequestError` — the defining module was never imported, or it is a typo |
| extras that do not validate against the class (missing `results`, unknown key) | `BadRequestError` wrapping the pydantic error |
| `custom_type` registered but belonging to another transport family | **dropped with its result** — identical to the existing foreign-native policy |

---

## Equivalence by construction, not by parallel code

The transports do not learn to read `extras`. They **normalize first**: a
`ToolCall` becomes `call.as_native()` before any projector sees it, and a
`ToolMessage` becomes `msg.as_native()` at the top of result projection. Every
projector, every native class, every wire shape is untouched — which is what
makes the two forms produce byte-identical payloads without a second code
path to keep in sync.

The **parse** direction is unchanged: the wire always builds the typed
subclass. `extras` is an accepted *input* and *storage* form, never something
the client hands back.

---

## Changes

### `luca/client/types/content.py`
- `CUSTOM_TYPE_KEY = "custom_type"`.
- `ToolCall.extras`, `ToolCall.as_native()`, `ToolCall.as_generic()`.

### `luca/client/types/messages.py`
- `ToolMessage.extras`, `ToolMessage.as_native()`, `ToolMessage.as_generic()`.

### `luca/client/transports/base.py`
- `_projector_for_call(call) -> ToolProjector | None` becomes
  `_resolve_call(call) -> tuple[ToolProjector, ToolCall] | None` — the pair the
  four transports already build for lineage, now with the normalized call in it.

### The four transports (`openai_responses`, `openai`, `anthropic`, `bedrock`)
- Assistant projection: use the `_resolve_call` pair (3 lines, same shape).
- Result projection: `msg = msg.as_native()` as the first statement.

### Tests
- The four existing `test_native_tools.py` modules: every projection test
  parametrized over the two input forms, one shared literal expected payload.
- `tests/client/test_types/test_native_extras.py`: the conversion pair, the
  round-trip invariant, and the error cases.
- `tests/client/test_transports/test_openai_responses/test_native_extras.py`:
  extras-specific transport semantics (foreign drop, unknown type, inertness).

### Docs
- `docs/client/06-tools.md`: a "generic form" subsection under provider-native
  tools.
- `AGENTS.client.md`: one key-fact entry.
