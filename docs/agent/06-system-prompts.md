# System prompts

The runner takes no `system_prompt` string. Instead it takes
**`system_prompt_parts`** — a list of parts resolved **fresh before every model
call** — and an optional **assembler** that flattens the priority-sorted parts
into the final string.

Each item in the list can be:

| Form | Becomes |
|---|---|
| `str` | `SystemPromptPart(text=...)` |
| `dict` — `text` + optional `priority`, `source` | `SystemPromptPart(**d)` (validated strictly) |
| `SystemPromptPart` | itself |
| callable `(session_config, runtime_status) -> ` any of the above | invoked per model call, its return coerced the same way |

Static parts are validated at construction; callables run per call, so their
part can reflect the current runtime status.

```python
from luca.agent.core import SystemPromptAssembler, SystemPromptPart
```

## 1. Default — no system message

Omit `system_prompt_parts` and the runner sends **no** system message at all.
That's a valid agent; many tool-only agents need nothing more.

```python
runner = AgentSessionRunner(session, tool_registry=registry)          # no system prompt
```

## 2. The simplest prompt — one string

```python
runner = AgentSessionRunner(
    session, tool_registry=registry,
    system_prompt_parts=["You are a helpful coding assistant."],
)
```

## 3. Multiple parts — `source` and `priority`

Real prompts are assembled from fragments (base persona, project rules, tool
guidance). Each part carries provenance and an ordering key:

```python
runner = AgentSessionRunner(
    session, tool_registry=registry,
    system_prompt_parts=[
        SystemPromptPart(text=PERSONA, source="model", priority=0),           # sorts first
        {"text": read_project_rules(), "source": "agents.md", "priority": 10},
        "Prefer the provided tools over answering from memory.",              # priority -1
    ],
)
```

The runner sorts by `priority` ascending (`-1`, the default, is "unranked" and
sorts first) before assembling. `source` is just a label for your own
bookkeeping (default `"model"`).

## 4. Dynamic — a callable part

A callable receives the live `SessionConfig` and a freshly computed
`SessionRuntimeStatus` (`step_count`, `turn_count`, `status`) on every model
call and returns a part in any static form — so the prompt can adapt as the
turn progresses, e.g. nudge the model to wrap up as steps pile up:

```python
def wrap_up_nudge(session_config, runtime_status):
    if runtime_status.step_count > 8:
        return {"text": "You've taken many steps — converge to a final answer.", "priority": 99}
    return ""   # an empty part costs only a blank line in the joined prompt

runner = AgentSessionRunner(
    session, tool_registry=registry,
    system_prompt_parts=[SystemPromptPart(text=BASE, priority=0), wrap_up_nudge],
)
```

## 5. A custom assembler

The assembler is a duck-typed concrete base (no ABC) — subclass and override the
one hook. Override it only when newline-join isn't enough (e.g. XML-tag the
sources, or dedupe). A blank result means "no system message" — the runner drops
it.

```python
class TaggedAssembler(SystemPromptAssembler):
    def assemble_system_prompt(self, parts):
        return "\n\n".join(f"<{p.source}>\n{p.text}\n</{p.source}>" for p in parts)

runner = AgentSessionRunner(
    session, tool_registry=registry,
    system_prompt_parts=[SystemPromptPart(text=PERSONA, source="model")],
    system_prompt_assembler=TaggedAssembler(),
)
```

Next: [`07-middleware.md`](07-middleware.md).
