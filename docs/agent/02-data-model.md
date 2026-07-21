# Data model

A conversation in luca is an ordered log of **entries**. An entry is one typed
record of something that happened: a user message, an assistant message, a tool
execution, a turn boundary. New entries are always appended at the end, and
reading the log top to bottom replays the conversation exactly. Every concept
in the framework — tool calls, approvals, cancellation, compaction — is just
another entry type in this log, so this page introduces them one example at a
time. The container that holds the log (`AgentSession`) comes at the end; the
log itself is the model.

The examples use a compact notation: one entry per line, `type` on the left,
payload indented underneath. Bookkeeping fields (ids, timestamps) are omitted
until the section that explains them.

## 1. The simplest conversation

```
User: reply with 'hello world'
Assistant: hello world
```

As a log, this is four entries — the two messages, bracketed by a pair of
markers delimiting the **turn** (one user request and everything the agent did
to answer it):

```
user
 └─ text        "reply with 'hello world'"
turn_start
assistant
 └─ text        "hello world"
turn_finish     outcome=completed
```

Two things to notice already:

- Messages carry their content as a list of **parts**. Here each message has a
  single `text` part; the next section shows richer messages.
- `turn_start` comes *after* the user message: posting a message and running
  the agent are separate acts — a message can sit queued in the log before any
  turn begins answering it.

## 2. Messages are made of parts

A message's `parts` is an ordered list of content — in the order it was
produced. A user message can mix text and images; an assistant message can
carry its reasoning alongside its answer:

```
User: [receipt.jpg] how much did I tip here?
Assistant: You tipped $12.40 — about 18%.
```

```
user
 ├─ image       receipt.jpg
 └─ text        "how much did I tip here?"
turn_start
assistant
 ├─ thinking    "Total $81.40, subtotal $69.00 → tip $12.40 ≈ 18%."
 └─ text        "You tipped $12.40 — about 18%."
turn_finish     outcome=completed
```

| Part `type` | Meaning |
|---|---|
| `text` | prose |
| `image` | an image the user attached |
| `thinking` | the model's reasoning, when it emits any |

An `image` part carries a `source` — one of `ImageURL`, `ImageBase64` or
`ImageFileId` — plus free-form `metadata`:

```python
from luca.agent.core import ImageBase64, ImageContent, TextContent

runner.post_message([
    ImageContent(
        source=ImageBase64(data=b64_bytes, media_type="image/png"),
        metadata={"name": "receipt.jpg"},
    ),
    TextContent(text="how much did I tip here?"),
])
```

`metadata` is yours and is never sent to the provider. It stays in the saved
session, so a replayed transcript can still describe an image whose file has
since been deleted.

| Source | Support |
|---|---|
| `ImageBase64` | everywhere |
| `ImageURL` | everywhere (the provider fetches it, so it must be publicly reachable) |
| `ImageFileId` | Anthropic only — the OpenAI chat-completions API has no file-id shape for images and raises |

The same union backs `ExecutionResult.content`, so a tool can return an image
too — the shell `read` tool returns one for a png or jpeg. An assistant message
is the exception: it carries text, thinking and tool calls, not images.

> ⚠️ **The conversation is the source of truth.** What a given provider can
> actually receive is the adapter layer's problem, not the data model's. An
> image in a tool result is stored either way; today it reaches Anthropic and
> raises on the OpenAI chat-completions API.

Beyond `parts`, an assistant entry records its provenance: the `llm_config`
that produced it and a `stop_reason` — `"stop"` here (the model finished its
answer), `"tool_use"` when it asks for a tool instead. Provider token usage is
deliberately *not* on the entry — it lives on the session, keyed by
conversation ([11](11-context-and-usage.md)).

There is one more part type — `tool_call`, the model asking to run a tool —
and it changes the shape of the log enough to deserve its own section.

## 3. Tool executions — the entry that changes

```
User: what's in notes.txt?
```

The model doesn't answer directly — it asks for a tool, with a `tool_call`
part:

```
user
 └─ text        "what's in notes.txt?"
turn_start
assistant
 ├─ text        "Let me read that file."
 └─ tool_call   call_1 · read_file(path="notes.txt")
```

A `tool_call` part carries the request and nothing else: `name`, `arguments`,
and a model-assigned id (`call_1`). Like every part it is immutable — it's
part of what the model said. What *happened* to the request is recorded by a
new entry type.

When the runner picks up `call_1` it appends a **`tool_execution`** entry: the
durable record of everything that happens to that request, correlated back by
the tool call's id (`tool_call_id`). Unlike every other entry it is **updated
in place** as the call moves through its lifecycle — the log gains no new
entries:

```
tool_execution  call_1 · read_file → pending      # appended; body not started
tool_execution  call_1 · read_file → running      # same entry, updated in place
tool_execution  call_1 · read_file → completed    # same entry, now terminal
 └─ result      "milk, eggs, bread"
```

The result feeds the next model call, and the turn closes:

```
user
 └─ text        "what's in notes.txt?"
turn_start
assistant
 ├─ text        "Let me read that file."
 └─ tool_call   call_1 · read_file(path="notes.txt")
tool_execution  call_1 · read_file → completed
 └─ result      "milk, eggs, bread"
assistant
 └─ text        "Your notes say: milk, eggs, bread."
turn_finish     outcome=completed
```

So a tool call is **two things**: the immutable `tool_call` part (what the
model asked) and the mutable `tool_execution` entry (what the framework did
about it). The execution is self-contained — it carries its own copy of the
request:

| Field | What it holds |
|---|---|
| `tool_call_id` | the correlation key back to the `tool_call` part |
| `raw_tool_call` | the request being executed — starts as the model's, middleware may swap it ([07](07-middleware.md)) |
| `tool_spec` | identity snapshot of the resolved tool (name, kind, version, declared `timeout_in_ms`); `None` if it never resolved |
| `status` | the lifecycle state (table below) |
| `result` | what the tool returned — set iff `status=completed` |
| `error` | structured failure (`error_type`, `error_message`, `details`) for `failed` / `not_found` / `invalid` |
| `started_at` / `ended_at` | body dispatch / terminal transition (unix ms) |
| `cancel_signalled_at` | when a run cancellation reached this execution |
| `is_doom_loop_flagged` | set by doom-loop detection ([08](08-runtime-config.md)) |

The lifecycle (`ExecutionStatus`):

| `status` | Meaning |
|---|---|
| `pending` | body not started, no terminal outcome |
| `running` | body started, no terminal outcome |
| `completed` | the body returned a result |
| `failed` | tool code raised |
| `not_found` | no such tool |
| `invalid` | arguments failed validation |
| `rejected` | the registry's decide() denied it |
| `cancelled` | cancellation prevented the body from starting |
| `interrupted` | a started body didn't finish (crash, orphan recovery) |
| `timed_out` | the framework-enforced deadline expired |

> ⚠️ **`completed` ≠ "the tool succeeded."** It means the framework received a
> result. The tool's own verdict is `result.is_error`: a file tool returning
> "file does not exist" with `is_error=True` is still `completed` — `failed`
> is reserved for tool code that *raised*.

Every tool call yields **exactly one** tool output for the model — even a
denied, cancelled, or malformed one (error text is derived from `status` +
`error` at projection time, never stored; see [10](10-projection.md)).

## 4. Approval — an orthogonal fact

The execution also records whether the call was *allowed to run* —
independently of the lifecycle. Suppose `delete_file` requires approval:

```
user
 └─ text        "clean up the old export"
turn_start
assistant
 ├─ text        "I'll delete export.csv."
 └─ tool_call   call_2 · delete_file(path="export.csv")
tool_execution  call_2 · delete_file → pending    approval=pending   # ⏸ run pauses
```

No `turn_finish` — the turn is left **open** and the session is awaiting
approval. The application resolves the decision out of band and calls `run()`
again ([05](05-permissions.md)); the same execution then advances in place:

```
tool_execution  call_2 · delete_file → completed  approval=allowed
 └─ result      "export.csv deleted"
```

Denied instead — a terminal status of its own, still producing a tool output
for the model:

```
tool_execution  call_2 · delete_file → rejected   approval=rejected
```

Three fields carry this:

| Field | What it holds |
|---|---|
| `approval_status` | the CURRENT state: `None` (never processed), `pending`, `allowed`, `rejected` |
| `approval_decisions` | append-only audit trail of every decide() verdict — never read state from it |
| `extras` | free-form dict written by registries/middleware, never interpreted by the core — `SimpleToolRegistry` stores the tool's approval context under `extras["approval_context"]` |

Three facts about one execution are deliberately orthogonal:

| Fact | Field |
|---|---|
| Did the framework run it, and how did that end? | `status` |
| Was it allowed to run? | `approval_status` |
| Does the tool consider its own result an error? | `result.is_error` |

## 5. Turns

A turn is the `turn_start … turn_finish` bracket around one user request —
however many model calls and tool round-trips it took. There is no `Turn`
object, just the two markers. One assistant message = one **step**; the
notes.txt conversation above is a two-step turn:

```
turn_start                          ┐
assistant        (step 1)           │  one turn:
tool_execution                      │  one user request,
assistant        (step 2)           │  as many steps as it takes
turn_finish      outcome=completed  ┘
```

`turn_finish` is a boundary and **outcome** record (`TurnOutcome`) — nothing
else, no usage rollup:

| `outcome` | Meaning |
|---|---|
| `completed` | the loop finished on its own |
| `cancelled` | the user ended the turn |
| `timed_out` | an LLM timeout ended the attempt |
| `errored` | any other failure |

A `turn_start` with no matching `turn_finish` means the turn is **open** —
"resume this turn," not "start a new one." That's how a turn survives an
approval pause, several `run()` calls, even a process restart.

## 6. Cancellation

Cancellation is durable too. `cancel()` appends a `cancel_requested` entry
inside the open turn; the wind-down consumes it and closes the bracket with
the requested outcome:

```
user
 └─ text        "summarize every file in the repo"
turn_start
assistant
 ├─ text        "Starting with src/…"
 └─ tool_call   call_3 · read_file(path="src/app.py")
tool_execution  call_3 · read_file → completed
 └─ result      "…"
cancel_requested  outcome=cancelled     # appended by cancel()
turn_finish       outcome=cancelled     # written at the next step boundary
```

Because the request is an entry, a session reloaded mid-cancel still knows it
is cancelling; consumed requests accumulate across turns as an audit trail. An
execution in flight when the cancel lands turns `cancelled` (never started) or
`interrupted` (started, didn't finish). The runner mechanics — grace periods,
what happens to an in-flight LLM call — live in [04](04-runner.md).

## 7. Where the log lives: `AgentSession`

Everything so far is one log. The container is `AgentSession` — a single
Pydantic object, JSON-serializable, lossless round-trip. It does **not** store
the log as a list. Two fields, and keeping them distinct is the core idea:

```python
from luca.agent.core import AgentSession

session.entries                     # dict[str, AnyEntry] — flat store, keyed by id
session.active_conversation.nodes   # list[str] — ordered entry ids: THE conversation
```

`entries` is an append-only **bag** of everything that ever happened. `nodes`
is the ordered id list that forms the current **path** through it. Labelling
the notes.txt entries A–F:

```
entries:
  A  user            "what's in notes.txt?"
  B  turn_start
  C  assistant
     ├─ text         "Let me read that file."
     └─ tool_call    call_1 · read_file(path="notes.txt")
  D  tool_execution  call_1 · read_file → completed
     └─ result       "milk, eggs, bread"
  E  assistant       "Your notes say: milk, eggs, bread."
  F  turn_finish     outcome=completed

nodes = A → B → C → D → E → F
```

Redundant for a straight-line conversation — deliberately. To read the
conversation you walk `nodes` and look each id up in `entries`; the payoff of
the split is that a *different* path over the same bag is a different
conversation (forking, compaction — next two sections).

Bookkeeping now: every entry carries `id` (its key in the bag), `parent_id`
(the entry appended before it — a recovery backstop only, **never
traversed**; `nodes` is the sole ordering authority), `created_at` (unix ms),
`context_tokens` (the entry's estimated content size —
[11](11-context-and-usage.md)), and the `type` discriminator that deserializes
each bag value straight to its concrete class (the `AnyEntry` union). Real ids
are opaque 8-char hex; these docs use letters for readability.

The full container:

| Field | What it holds |
|---|---|
| `id` | the session id |
| `entries` | the append-only bag |
| `tool_executions` | denormalized index `tool_call_id → [execution ids]` — never scan the bag |
| `usages` | provider-usage records, `conversation_id → entry_id → Usage` ([11](11-context-and-usage.md)) |
| `active_conversation` | `Conversation`: `id`, `nodes`, `status`, timestamps |
| `conversation_history` | prior `Conversation` paths kept alongside the active one |
| `session_config` | `LLMConfig` + `RuntimeConfig` (§10) |

## 8. Forking

A fork is a deep copy under a new id:

```python
forked = session.model_copy(deep=True)
forked.id = "a-new-id"
```

Same bag, same path — but by value, nothing shared by reference — so future
writes diverge:

```
original: nodes = A → B → C → D → E → F
fork:     nodes = A → B → C → D → G → H     # diverged after D
```

## 9. Compaction

A `compaction` entry is a summary standing in for a span of older entries. The
bag keeps everything (append-only, nothing is deleted); the *path* stops
visiting the summarized span:

```
before:  nodes = A → B → C → D → E → F
after:   nodes = A → S → E → F

S  compaction
   ├─ summary      "Read notes.txt; it lists milk, eggs, bread."
   └─ summarized   [B, C, D]      # the span this entry replaced
```

`summarized` makes the entry self-describing — you can always recover what it
replaced. The model sees the summary as a user message
([10](10-projection.md)).

A `pruned` entry works the same way for a *single* entry: replacement content
standing in for one original (typically a bulky tool output), swapped into the
path **in place** while the original stays in the bag:

```
before:  nodes = A → B → C → D → E → F
after:   nodes = A → B → C → P → E → F

P  pruned
   ├─ pruned_entry_id    D           # the original, untouched in the bag
   └─ content            "[tool output has been pruned to reduce context]"
```

Who produces pruned entries and when is a strategy concern —
[11](11-context-and-usage.md).

## 10. Status and config

```python
session.status                   # ConversationStatus — a persisted cache
session.session_runtime_status   # recomputed from the entries on every access
```

| `status` | Meaning |
|---|---|
| `idle` | nothing queued; awaiting a user message |
| `pending` | work queued — call `run()` |
| `running` | a run is actively driving (crash-recovery marker) |
| `awaiting_approval` | paused at a tool-approval gate (§4) |
| `cancelling` | an unconsumed `cancel_requested` exists (§6) |

> ⚠️ **Never trust a persisted status.** `status` is a cache the runner
> maintains; the entries are the truth. A session that crashed mid-`running`
> self-heals when a runner takes ownership and re-derives it.

`session_config` holds the `LLMConfig` for the *next* turn plus the
`RuntimeConfig` knobs ([08](08-runtime-config.md)). What is **not** on the
session: the tool registry, the projector, system-prompt parts, the live
cancellation token. Those are runtime collaborators you pass to the
runner — which is exactly what keeps the session a pure, portable record.

## 11. Serialize and resume

```python
text = session.model_dump_json(indent=2)              # lossless round-trip
session = AgentSession.model_validate_json(text)

runner = AgentSessionRunner(session, tool_registry=registry)
```

Loading is just deserializing; resuming is constructing a runner around the
loaded session and supplying the collaborators again. An open turn resumes
(§5), a stale status self-heals (§10), a pending approval is still pending
(§4).

## 12. Entry types, recapped

| Entry `type` | Carries | Mutable? |
|---|---|---|
| `user` | `parts` | no |
| `assistant` | `parts`, `llm_config`, `stop_reason` | no |
| `tool_execution` | one tool call's whole lifecycle (§3–§4) | **yes — the only one** |
| `turn_start` | — | no |
| `turn_finish` | `outcome`, `error` | no |
| `cancel_requested` | requested `outcome`, `error` | no |
| `compaction` | `summary`, `summarized` ids | no |
| `pruned` | replacement `content` for one original entry (§9) | no |

(Every entry also carries the shared base fields — `id`, `parent_id`,
`created_at`, `context_tokens`.)

Next: [`03-tools.md`](03-tools.md).
