# Context and usage

Two measurements the framework keeps strictly apart:

- **`Entry.context_tokens`** — the estimated size of that entry's model-facing
  content. Intrinsic to the entry: every conversation that references it sees
  the same number.
- **`AgentSession.usages`** — what the provider *reported* for an entry in one
  conversation. Accessory data on the conversation↔entry relationship: the
  same assistant message shared by two conversations (a fork) can report
  different usage, because reported input covers the whole request context.

> ⚠️ **Never conflate them.** Provider usage is consumption accounting;
> `context_tokens` is content size. Neither is ever derived from the other.

## 1. Reading them

Both are plain session data — no API, no methods:

```python
conv = session.active_conversation

# context: sum the entries on the path
total_context = sum(session.entries[i].context_tokens for i in conv.nodes)

# usage: one record per (conversation, entry) pair, keyed conversation-first
for record in session.usages.get(conv.id, {}).values():
    print(record.entry_id, record.input, record.output, record.total_tokens)
```

Aggregates (per turn, per entry type) are deliberately **not** in the core —
they're two-line loops over data you already hold, and which totals matter is
your application's call.

## 2. `ContextManager` — an architectural helper

Everything above is *written* by one runtime collaborator, the
`ContextManager` — a strategy seam like the projector: lives on the runner,
never serialized, one object.

```python
from luca.agent.core import AgentSessionRunner, ContextManager

runner = AgentSessionRunner(
    session, tool_registry=registry,
    context_manager=MyContextManager(),   # None → the minimal default
)
```

Three hooks; the runner calls them at fixed points:

| Hook | Called | Default behavior |
|---|---|---|
| `calculate_context(entry) -> int` | on every **new** entry, before `before_entry_written`; again when a `ToolExecution` turns terminal, before `after_tool_execution` | `len(model-facing text) // 4`, plus `IMAGE_TOKENS` (1000) per image |
| `process_tool_output(result) -> ExecutionResult` | on a returned `ExecutionResult`, before the terminal execution is built (so session, `ToolExecuted` event, and wire all see the processed output) | identity pass-through |
| `prune_entry(entry) -> PrunedEntry` | never by the runner itself — you compose it with the ledger (§5) | terminal tool executions only → a fixed marker |

> ⚠️ **The default is a placeholder, not a policy.** Four-characters-per-token
> estimation, a flat 1000 tokens per image, no truncation, marker-only
> pruning — enough to make the seam real and the numbers non-zero. It exists
> so the *architecture* is in place; if context accounting matters to your
> application, improving this class is **your** job: bring a real tokenizer,
> a real truncation budget, a real pruning strategy.

The image constant is deliberately dimension-blind: a URL source has no local
bytes to measure, reading real dimensions would need an image decoder, and the
provider formulas disagree by an order of magnitude.

Per-type content ownership in the default estimate: a user message owns its
content; an assistant message its text + thinking + tool-call requests (name
and JSON arguments — counted once, never again on the execution); a tool
execution only its outcome (result content, else the structured error
message; `0` while nonterminal); a compaction its summary; a pruned entry its
replacement content; markers own nothing.

> ⚠️ **Middleware has the final say.** Context is calculated *before*
> `before_entry_written` / `after_tool_execution`; whatever middleware returns
> is persisted. The framework never recalculates, validates, or repairs
> `context_tokens` afterwards.

## 3. Improving it: estimation and truncation

Swap the estimate without touching ownership — the ratio is a class var, the
text→count step one method:

```python
import tiktoken   # your dependency, not luca's

class TiktokenContext(ContextManager):
    def __init__(self) -> None:
        self._encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    def _estimate_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))
```

Images are counted by a separate method, so a text tokenizer and an image
formula are independent overrides:

```python
class AnthropicImages(ContextManager):
    def _media_tokens(self, entry) -> int:
        return sum(w * h // 750 for w, h in dimensions_of(entry))
```

Truncate tool outputs before they become durable — preserve the original
under your own policy (`metadata` is yours):

```python
from luca.agent.core import ContextManager, ExecutionResult, TextContent

class TruncatingContext(ContextManager):
    MAX_CHARS = 4_000

    def process_tool_output(self, execution_result: ExecutionResult) -> ExecutionResult:
        text = "".join(p.text for p in execution_result.content)
        if len(text) <= self.MAX_CHARS:
            return execution_result
        return ExecutionResult(
            content=[TextContent(text=text[: self.MAX_CHARS] + " …[truncated]")],
            metadata={**execution_result.metadata, "original_chars": len(text)},
            is_error=execution_result.is_error,
        )
```

The processed result is what persists, what the `ToolExecuted` event renders,
and what every future LLM request projects — the three can never disagree.

## 4. Usage records

When an assistant message is recorded, the runner writes one `Usage` record to
`session.usages[conversation_id][entry_id]` — a self-describing association
(`conversation_id` and `entry_id` are required fields):

```python
session.usages == {
    "c1": {
        "a1": Usage(conversation_id="c1", entry_id="a1",
                    input=100, output=20, total_tokens=120),
    },
}
```

Entries carry **no** usage field; `TurnFinish` carries no rollup; `RunResult`
carries no usage. `SessionLedger.record_usage()` is the single write door, so
the keys always agree with the record and always reference an entry on the
conversation's path.

## 5. Pruning

Pruning replaces an entry's *contribution to the path* without mutating or
deleting the original. A `PrunedEntry` records what it replaced
(`pruned_entry_id`, `pruned_entry_type`) and the replacement `content`; the
path swaps the node id in place:

```
before:  nodes = A → B → C → D → E        D  tool_execution → completed (huge output)
after:   nodes = A → B → C → P → E        P  pruned(pruned_entry_id=D)
                                              └─ "[tool output has been pruned to reduce context]"
```

Only the machinery ships for now — the runner exposes **no** `prune()` method
and nothing triggers pruning automatically. You compose the pieces yourself:

```python
manager = runner.context_manager
template = manager.prune_entry(runner.session.entries["te1"])  # terminal executions only

def build(entry_id, parent_id, ts):
    pruned = template.model_copy(
        update={"id": entry_id, "parent_id": parent_id, "created_at": ts},
    )
    pruned.context_tokens = manager.calculate_context(pruned)
    return pruned

runner.ledger.prune("te1", build)   # verifies referent/type/terminality, swaps in place
```

On the next LLM call the projector resolves the referent and emits the
replacement under the original's role and `tool_call_id`
([10](10-projection.md)) — ordering and correlation survive. The original
entry stays in `session.entries` untouched.

> ⚠️ **Minimal on purpose, again.** The default prunes only terminal tool
> executions, with one fixed marker, and *when* to prune is entirely
> undecided. A real strategy — thresholds, which entries, budgets — is
> application policy you build on this seam.

That's the full agent surface. Back to the [index](README.md).
