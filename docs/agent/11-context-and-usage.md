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
conv = session.conversations[session.main_conversation_id]

# context: sum the entries on the path
total_context = sum(session.entries[i].context_tokens for i in conv.nodes)

# usage: one record per (conversation, entry) pair, keyed conversation-first
for record in session.usages.get(conv.id, {}).values():
    print(record.entry_id, record.input, record.output, record.total_tokens)
```

Both are **per conversation**. A session holds a catalog of them — archived
compaction predecessors, and one per subagent ([13](13-subagents.md)) — each
with its own window to fill, so "the context used" is only a question you can
ask about one of them.

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

Five hooks; the runner calls them at fixed points:

| Hook | Called | Default behavior |
|---|---|---|
| `calculate_context(session, entry) -> int` | on every **new** entry, before `before_entry_written`; again when a `ToolExecution` turns terminal, before `after_tool_execution` | `len(model-facing text) // 4`, plus `IMAGE_TOKENS` (1000) per image |
| `process_tool_output(session, execution, result) -> ExecutionResult` | on a returned `ExecutionResult`, before the terminal execution is built (so session, `ToolExecuted` event, and wire all see the processed output). A returned `ExecutionDeferred` skips it — there is no output to process | identity pass-through |
| `prune_entry(session, entry) -> PrunedEntry` | **never** — no framework call site; you compose it with the ledger (§5) | terminal tool executions only → a fixed marker |
| `should_compact(session, conversation_id) -> bool` | at the top of every drive, and at `start()` (hence sync) | `False` — never compacts |
| `compact(session, conversation_id, nodes, entry)` | once `should_compact` says yes, or after `schedule_compaction()` | raises `NotImplementedError` |

Only the compaction pair names a conversation. The other three are given the
entry, the execution, or the result they are measuring — objects that already
carry everything they need, and whose answer must be the same for every
conversation that references them (`context_tokens` is intrinsic; a tool output
does not change meaning because a subagent produced it).

The last two are compaction, covered on its own page —
[12](12-compaction.md). They live here because the collaborator that measures
context is the one with the standing to decide there is too much of it.

The live `AgentSession` comes first on all of them, so every policy sees the
same state — the active model included. It is **read-only** to the manager: the
runner owns every write.

> ⚠️ **`calculate_context` runs on every new entry.** Scanning
> `session.entries` inside it makes a turn quadratic — cross-entry work belongs
> in `process_tool_output` / `prune_entry`, which run rarely. On an append it
> also runs *before* the entry joins the session: the entry has its `id`, but
> `session.entries[entry.id]` raises `KeyError`.

> ⚠️ **The default is a placeholder, not a policy.** Four-characters-per-token
> estimation, a flat 1000 tokens per image, no truncation, marker-only
> pruning, no compaction — enough to make the seam real and the numbers
> non-zero. It exists so the *architecture* is in place; if context accounting
> matters to your application, improving this class is **your** job: bring a
> real tokenizer, a real truncation budget, a real pruning strategy. A
> compacting one ships in
> [contrib/simple_context_manager/](contrib/simple_context_manager/README.md).

The image constant is deliberately dimension-blind: a URL source has no local
bytes to measure, reading real dimensions would need an image decoder, and the
provider formulas disagree by an order of magnitude.

Per-type content ownership in the default estimate: a user message owns its
content; an assistant message its text + thinking + tool-call requests (name
and JSON arguments — counted once, never again on the execution); a tool
execution only its outcome (result content, else the structured error
message; `0` while nonterminal); a compaction its summary; a pruned entry its
replacement content; markers own nothing.

> ⚠️ **A parked tool call counts `0`.** An execution deferred at
> `AWAITING_RESULT` ([03](03-tools.md) §7) is nonterminal, so it contributes
> nothing however long it stays parked and however many times it is
> re-dispatched — `calculate_context` runs again only when it finally turns
> terminal, and a deferral is not a terminal transition. The placeholder it
> puts on the wire ([10](10-projection.md) §2) is therefore unbudgeted; it is
> one short line.

> ⚠️ **Middleware has the final say on every write.** Context is calculated
> *before* `before_entry_written` / `after_tool_execution`; whatever middleware
> returns is persisted. The framework never recalculates, validates, or repairs
> `context_tokens` afterwards. The one door that is not a write —
> `recalculate_context_tokens()` — runs no middleware at all (§4).

## 3. Improving it: estimation and truncation

Swap the estimate without touching ownership — the ratio is a class var, the
text→count step one method, and the session carries the model to count against:

```python
import tiktoken   # your dependency, not luca's

from luca.agent.core import AgentSession, ContextManager, Entry

class TiktokenContext(ContextManager):
    def calculate_context(self, session: AgentSession, entry: Entry) -> int:
        model = session.session_config.llm_config.model   # "openai/gpt-4o-mini"
        self._encoding = tiktoken.encoding_for_model(model.split("/")[-1])
        return super().calculate_context(session, entry)

    def _estimate_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))
```

`context_tokens` is *stored* on the entry, so a model-sensitive count goes
stale the moment the session switches models.
`AgentSessionRunner.recalculate_context_tokens()` re-derives every entry in
`session.entries` — not just the active path, since the count is intrinsic and
shared by every conversation:

```python
session.session_config.llm_config = LLMConfig(model="openai/gpt-4o-mini", provider="openrouter")
runner.recalculate_context_tokens()
```

> ⚠️ **Nothing in the framework calls it.** No constructor keyword, no CLI
> flag, no automatic invocation on a model switch. It exists for the
> application that swapped in a real tokenizer, and that application calls it.

> **It runs no middleware.** `before_entry_written` is scoped to the
> conversation whose operation caused a write, and this rewrites every entry
> across every conversation at once — no single id would be honest. It is an
> operational refresh of a derived estimate, not a write ([07](07-middleware.md)).

Images are counted by a separate method, so a text tokenizer and an image
formula are independent overrides:

```python
class AnthropicImages(ContextManager):
    def _media_tokens(self, entry) -> int:
        return sum(w * h // 750 for w, h in dimensions_of(entry))
```

Truncate tool outputs before they become durable — preserve the original
under your own policy (`metadata` is yours). The execution is handed in too,
so the policy can vary per tool — truncate `bash`, never `read`:

```python
from luca.agent.core import (
    AgentSession, ContextManager, ExecutionResult, TextContent, ToolExecution,
)

class TruncatingContext(ContextManager):
    LIMITS = {"bash": 4_000, "grep": 2_000}   # `read` is absent → never truncated

    def process_tool_output(
        self,
        session: AgentSession,
        execution: ToolExecution,
        result: ExecutionResult,
    ) -> ExecutionResult:
        limit = self.LIMITS.get(execution.raw_tool_call.name)
        if limit is None:
            return result
        text = "".join(p.text for p in result.content if isinstance(p, TextContent))
        if len(text) <= limit:
            return result
        return ExecutionResult(
            content=[TextContent(text=text[:limit] + " …[truncated]")],
            structured_content=result.structured_content,   # carry it over
            metadata={**result.metadata, "original_chars": len(text)},
            is_error=result.is_error,
        )
```

> ⚠️ **Rebuilding the result drops what you don't copy.** `structured_content`
> ([`02-data-model.md`](02-data-model.md) §4) is the easy one to lose — it is
> not model-facing, so truncating text is never a reason to discard it. Copy
> every field you are not deliberately changing.

> ⚠️ **The execution is mid-transition.** `status` is still `RUNNING` and
> `result` is not attached yet — read it for *identity* (`raw_tool_call.name`,
> `tool_spec`), never for outcome. `tool_spec` is `ToolSpec | None` (a registry
> may dispatch a call it never snapshotted), so branch on `raw_tool_call.name`
> or guard the `None`.

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

A compaction's own summarization call is recorded the same way, under the
**pre-compaction** conversation — where the request was actually made, with
that conversation's context as its input ([12](12-compaction.md)). A subagent's
calls land under its own conversation. So a session's total cost means summing
the whole catalog:

```python
spent = sum(
    usage.total_tokens
    for records in session.usages.values()
    for usage in records.values()
)
```

> ⚠️ **Atomic session writes are the application's job.** The core owns no
> persistence. Write to a temporary file and `os.replace` it into place — a
> crash mid-write otherwise leaves truncated JSON and an unloadable session,
> which is the one way to lose a whole conversation.
> `luca.agent.contrib.tui.sessions.save_session` does exactly that.

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
manager, session = runner.context_manager, runner.session
template = manager.prune_entry(session, session.entries["te1"])  # terminal executions only

def build(entry_id, parent_id, ts):
    pruned = template.model_copy(
        update={"id": entry_id, "parent_id": parent_id, "created_at": ts},
    )
    pruned.context_tokens = manager.calculate_context(session, pruned)
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

A `ChildConversation` is sized by its result only when the link itself renders
it — a resolution written without a result execution (a cancel wind-down).
Ordinarily the result execution carries the content, its own `context_tokens`
cover it, and the link contributes `0` — as does an unresolved link, exactly
like a nonterminal execution.

> ⚠️ **Never prune a subagent's result execution mid-orchestration.** A
> pruned PRIVATE execution projects as nothing (no `ToolMessage` can carry a
> runner-minted correlation id), so pruning one erases the child's entire
> answer from the wire with no marker — and if it is pruned before the parent
> woke for it, the wake signal goes with it. Prune old updates only once the
> turn is over.

Next: [`12-compaction.md`](12-compaction.md) — the other half of the context
story: when pruning single entries is not enough, replace the whole older span
with a summary.
