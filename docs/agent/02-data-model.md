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
| `thinking` | the model's reasoning, when it emits any — plus the provider's `signature` over it, its `id` for the reasoning item (OpenAI's `rs_…`, needed to replay it), and `redacted` when the body was withheld |

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

There are two part unions, and they stay separate so a `tool_call` can never
land in a user message:

| Union | Parts | Used by |
|---|---|---|
| `ContentPart` | `text`, `image` | user messages, `ExecutionResult.content`, `PrunedEntry.content` |
| `AssistantContentPart` | `text`, `thinking`, `tool_call` | assistant messages |

So a tool can return an image too — the shell `read` tool returns one for a png
or jpeg. An assistant message is the exception: it carries text, thinking and
tool calls, not images.

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
| `conversation_id` | provenance: the conversation this call was born in (§7). It is on this entry and no other, because an execution is the only entry a consumer receives **detached from a path** — handed to a registry, to middleware, carried inside the tool events |
| `raw_tool_call` | the request being executed — starts as the model's, middleware may swap it ([07](07-middleware.md)) |
| `tool_spec_id` | the durable reference to the resolved tool — a key into `session.tool_specs` (§7); `None` if it never resolved |
| `tool_spec` | the resolved tool itself (`name`, `description`, `input_schema`, optional `output_schema`, kind, version, declared `timeout_in_ms`) — a cache restored from `tool_spec_id`, never the durable truth (§7) |
| `status` | the lifecycle state (table below) |
| `result` | what the tool returned — set iff `status=completed` |
| `error` | structured failure (`error_type`, `error_message`, `details`) for `failed` / `not_found` / `invalid`; the runner records the failing `phase` under `details` |
| `started_at` / `ended_at` | when the body was dispatched / when the execution turned terminal (unix ms) |
| `cancel_signalled_at` | when a run cancellation reached this execution |
| `is_doom_loop_flagged` | set by doom-loop detection ([08](08-runtime-config.md)) |

The lifecycle (`ExecutionStatus`):

| `status` | Meaning |
|---|---|
| `received` | the model asked for the call and the entry exists; the registry has not been consulted yet |
| `pending` | born, body not started, no terminal outcome |
| `running` | body started, no terminal outcome |
| `completed` | the body returned a result |
| `failed` | tool- or registry-owned code raised — while resolving the call, or inside the body |
| `not_found` | no such tool |
| `invalid` | arguments failed validation |
| `rejected` | the registry's decide() denied it |
| `refused` | a framework runtime limit refused the call before dispatch (e.g. the spawn budget, [13](13-subagents.md)) |
| `cancelled` | cancellation prevented the body from starting |
| `interrupted` | a started body didn't finish (crash, orphan recovery) |
| `timed_out` | the framework-enforced deadline on the body expired |

> ⚠️ **`completed` ≠ "the tool succeeded."** It means the framework received a
> result. The tool's own verdict is `result.is_error`: a file tool returning
> "file does not exist" with `is_error=True` is still `completed` — `failed`
> is reserved for tool code that *raised*.

**An assistant message and its executions are written together.** The moment a
response with N tool calls is recorded, N `received` executions are appended
with it, in one synchronous step — before the registry is asked anything.
Asking the registry (`create_execution`) is a separate, resumable step that
folds its answer into those entries, moving each to `pending` or straight to a
terminal status.

That ordering is not an implementation detail. It is what guarantees a path can
never hold a `tool_call` without the execution node that answers it, so nothing
appended concurrently — a `post_message` arriving mid-turn most of all
([04](04-runner.md)) — can wedge itself between a tool call and its result and
produce a request every provider rejects. `received` is the durable
proof that the promise was recorded before the work began; a session that
crashes there reloads and births on the next drive.

`started_at` is stamped **iff the body was dispatched**, and `execution.dispatched`
is exactly `started_at is not None`. Everything the framework settles before
dispatch leaves it `None`:

| `dispatched` | Statuses |
|---|---|
| `True` | `running`, `completed`, `timed_out`, `interrupted`, and a `failed` raised by the tool body |
| `False` | `pending`, `rejected`, `refused`, `cancelled`, `not_found`, `invalid`, and a `failed` raised while resolving or validating the call |

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

No `turn_finish` — the turn is left **open** and the conversation derives
`blocked` (§10). The application resolves the decision out of band and calls
`run()` again ([05](05-permissions.md)); the same execution then advances in
place:

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

An `ExecutionResult` carries two output channels, and only one is model-facing:

| Field | Who reads it |
|---|---|
| `content` | the **model** — projected as the tool output ([`10`](10-projection.md)) |
| `structured_content` | your **application** — a machine-readable payload, shaped by the tool's `output_schema` ([`03`](03-tools.md)) |
| `metadata` | your application — free-form bookkeeping no schema describes |

```python
ExecutionResult(
    content=[TextContent(text="25°C, wind from the south.")],
    structured_content={"degrees_in_celsius": 25, "wind_direction": "south"},
)
```

> ⚠️ **`structured_content` never reaches the model** and never counts toward
> context ([`11`](11-context-and-usage.md)). A tool that wants the model to see
> the payload serializes it into `content` itself. Nothing validates it against
> `output_schema`.

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

session.entries                                     # dict[str, AnyEntry] — flat store, keyed by id
session.conversations[session.main_conversation_id].nodes   # list[str] — ordered ids: THE conversation
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
| `tool_specs` | normalized spec store `spec_id → ToolSpec`, append-only: one row per distinct tool definition ever *called* |
| `usages` | provider-usage records, `conversation_id → entry_id → Usage` ([11](11-context-and-usage.md)) |
| `conversations` | the CATALOG: `dict[str, Conversation]` — every path over the bag, live or archived |
| `main_conversation_id` | which one the user is talking to |
| `session_config` | `LLMConfig` + `RuntimeConfig` (§10) |
| `extras` | free-form application state, stored verbatim and never interpreted — the session-level twin of `ToolExecution.extras` |

`extras` exists so a tool, a registry or a plugin can keep state that outlives
the process without the application inventing a second file for it. Whoever
composes the runner hands the state in; it rides along on every save:

```python
plugin = MemoryPlugin(todo_store=session.extras.setdefault("todos", {}))
```

Namespace your key and keep the value JSON-serializable — this is dumped with
the session. The core never reads it ([09](09-plugins.md)).

A `Conversation` is a path and its bookkeeping — `id`, `nodes`, `created_at`,
`updated_at`, `previous_conversation_id` (the one a compaction replaced, §9) and
`depth` (0 for the main conversation, 1 for a subagent's). **It stores no
status**: status is derived from the entries on every read (§10).

The catalog is flat, not a tree, and there is no parent pointer — parent → child
is the only direction anything traverses, through the entry below. That is
exactly why `depth` is stored rather than computed.

### Subagents: more than one conversation at a time

A subagent is a second conversation in the same session, advancing at the same
time as the main one and linked into its parent's path by a
**`child_conversation`** entry:

```
main conversation c1                     subagent conversation c2 (depth 1)
──────────────────────────────           ───────────────────────────────────
user   "summarize alpha and beta"
turn_start
assistant
 └─ tool_call  call_9 · spawn_subagent
tool_execution     call_9 → completed
child_conversation  → c2  ─────────────▶ user   "Read alpha.txt and report back"
                                         turn_start
                                         assistant … tool_execution …
                                         assistant "alpha is a shopping list"
                                         turn_finish  outcome=completed
tool_execution  call_10 · <result tool>
child_conversation  → c2  ◀────────────  execution_result "alpha is a shopping list"
assistant  "alpha is a shopping list…"
turn_finish     outcome=completed
```

`child_conversation` is the **third mutable entry type**: it is appended
unresolved when the child is spawned and gains its `execution_result` (plus
`result_execution_id`) when the child's turn closes.

| Field | What it holds |
|---|---|
| `conversation_id` | the child conversation in `session.conversations` |
| `tool_execution_id` | the spawning execution — the durable record that this spawn was handled, so a reload never spawns it twice |
| `execution_result` | the child's answer, `None` until its turn closes |
| `result_execution_id` | the runtime-minted result execution that produced it — its path position is where the answer projects, so history stays append-only while this entry mutates; `None` with a result set means the link was resolved without the tool (a cancel wind-down) and renders in place |

The parent's turn cannot CLOSE while a link is unresolved — but the model is
re-engaged per resolution, and an unresolved link inside the open turn simply
projects as nothing (outside one it raises — no close may leave one behind:
[10](10-projection.md)). What spawns children, who drives them, how the model
steers and stops them, and how they are cancelled is [13](13-subagents.md).

An execution names its tool by `tool_spec_id`; the spec is stored once, under an
id derived from its own content:

```python
from luca.agent.core import ToolKind, ToolSpec

spec = ToolSpec(
    name="read_file",
    description="Read a UTF-8 text file.",
    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
    tool_kind=ToolKind.READ,
)
spec.spec_id()      # '243938d2…' — 64 hex chars

session.tool_specs[execution.tool_spec_id] is execution.tool_spec   # True
```

`description` and `input_schema` are required: a tool that takes no arguments
declares the empty object schema `{"type": "object", "properties": {}}`, never
`None` — an absent schema and an empty one mean different things to a provider.
`output_schema` is optional and, like every other field, part of the hash: a
tool that gains one becomes a second row, and the executions that ran before it
keep resolving to the first.
`spec_id()` is SHA-256 over the spec's JSON with recursively sorted keys and no
whitespace, hex-encoded in full, so the same definition yields the same 64
characters in every process and every language: two calls to one tool write one
row and share one `ToolSpec` object. A tool whose definition changes writes a
second row, and the executions that ran the old definition keep resolving to the
old one. Only the runner writes this store — a registry returns a spec and never
computes an id.

> ⚠️ **A `ToolSpec` must be a pure function of the tool definition.** A registry
> that puts anything volatile in `metadata` — a timestamp, a request id — mints
> a new `tool_specs` row on every call and silently defeats the normalization,
> with no error and no warning.

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
bag keeps everything (append-only, nothing is deleted); compaction **archives
the conversation and opens a new one** over the path that survives:

```
before:  main c1 = A → B → C → D → E → F

after:   c1 → [A, B, C, D, E, F, ts, S, tf]   ← archived, intact
         c2 → [S, E, F]                       ← the new main conversation

S  compaction
   ├─ source           "user" | "policy"    # who asked
   ├─ parts            [TextContent("Read notes.txt; it lists milk, eggs…")]
   ├─ compacted_nodes  [A, B, C, D]         # the span this entry replaced
   ├─ llm_config       the model that WROTE the summary
   └─ started_at / ended_at
```

Both conversations stay in the catalog; `main_conversation_id` moves to `c2` and
`c2.previous_conversation_id` names `c1`, so the chain back through every
compaction is walkable one hop at a time.

`compacted_nodes` makes the entry self-describing — you can always recover what
it replaced, because those entries are still in the bag and the whole
pre-compaction path is still a conversation in the catalog. The model sees `parts` as a
user message ([10](10-projection.md)); who triggers it and what it says is a
`ContextManager`'s job ([12](12-compaction.md)).

`compaction` is a **mutable entry type** (the second of three, with
`tool_execution` and `child_conversation`): it is written the moment a
compaction is intended, mutated as it progresses, and left in its terminal
state whether it succeeded or not. It carries no `status` field — the turn
bracket around it owns how the attempt ended, and these fields own what it
produced, so nothing can disagree:

| Log state | Means |
|---|---|
| open bracket, no `started_at` | scheduled, not yet started |
| open bracket, `started_at` set | running — or crashed mid-run |
| closed `COMPLETED`, `parts` set | succeeded |
| closed `COMPLETED`, `parts` None | nothing to compact |
| closed `ERRORED` / `TIMED_OUT` / `CANCELLED` | failed; never retried |

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

Status is **never stored**. There is one door, and it recomputes from the nodes
on every call:

```python
from luca.agent.core import ConversationStatus

state = session.get_conversation_status(session.main_conversation_id)
state.status        # ConversationStatus
state.turn_count    # conversational turns in this conversation
state.step_count    # assistant messages in the OPEN turn
```

| `status` | Derived when |
|---|---|
| `cancelling` | the open turn holds an unconsumed `cancel_requested` (§6) |
| `busy` | the open turn has something runnable — or a trailing `UserMessage` is queued, or a subagent result / mid-turn post awaits the model (a post into a gated turn included: the next drive answers it past the gate, [04](04-runner.md)) |
| `blocked` | the open turn has nothing runnable: every execution is waiting on an approval, or every subagent is and nothing new awaits the model |
| `idle` | anything else, INCLUDING a closed `turn_finish` whatever its outcome |

Two consequences, both deliberate. A **failed** turn derives `idle`, so
recovering from one means posting a new message rather than re-driving the same
request. And a trailing user message derives `busy` (queued work) — more
messages may still be posted behind it, and posting into an open turn is legal
too: the status says what the next `run()` will do, never whether input is
accepted (that is `post_message`'s own acceptance matrix — [04](04-runner.md)).

`blocked` is **subtree-aware**: a parent whose subagents are still working is
`busy`, and flips to `blocked` only when every one of them is AND nothing in
the open turn awaits the model (`open_turn_unseen_material` — a posted
message, a resolved child's result; with
`wake_parent_on_subagent_completion=False` a resolved child's result no
longer counts, [08](08-runtime-config.md)). A gate on the parent's own
execution outranks that material term — the next `run()` can only re-park at
the gate, so the honest answer is `blocked` — with one exception: an unseen
user POST lets the gate term yield (`open_turn_unseen_post`, deliberately
narrower than the material predicate), because the gated call projects a
placeholder and the next drive can answer the post
([10](10-projection.md) §2). The busy/blocked transition can be
triggered by a *sibling* finishing, with nothing in the parent's own entries
changing at all — which is exactly why nothing can cache it.

> ⚠️ **A crashed session self-heals for free.** Nothing persists a "running"
> marker to go stale, because the entries were always the truth.

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
(§5), status re-derives itself (§10), a pending approval is still pending (§4),
and an unresolved subagent is picked up where it stopped (§7).

Tool specs go out normalized and come back restored. Two calls to the same tool
are one stored spec and two references — the inline `tool_spec` is not written
at all:

```jsonc
{
  "tool_specs": {
    "243938d2…64 hex chars…": {
      "name": "read_file",
      "description": "Read a UTF-8 text file.",
      "input_schema": { "type": "object", "properties": { "path": { "type": "string" } } }
    }
  },
  "entries": {
    "e1": { "type": "tool_execution", "tool_spec_id": "243938d2…" },
    "e2": { "type": "tool_execution", "tool_spec_id": "243938d2…" }
  }
}
```

Constructing the session puts `tool_spec` back on every execution, all of them
sharing the one object. Serializing a `ToolExecution` **on its own** keeps its
spec inline instead — that is what the tool lifecycle events carry to consumers
holding no session ([04](04-runner.md)).

Two shapes refuse to construct — a loaded file or a hand-built literal alike —
rather than degrade quietly:

| Serialized shape | Why it raises |
|---|---|
| `tool_spec_id` naming a spec absent from `tool_specs` | the session would run with the tool's declared `timeout_in_ms` and `tool_kind` gone — an unbounded body and a different approval path, both untraceable |
| an inline `tool_spec` with no `tool_spec_id` | tolerating it would load and run fine, then lose every spec on the first save — the serializer strips inline specs and would have no id to write instead |

> ⚠️ **No migration.** Session files written before tool-spec normalization are
> exactly the second shape and do not load. Regenerate them.

## 12. Entry types, recapped

| Entry `type` | Carries | Mutable? |
|---|---|---|
| `user` | `parts` | no |
| `assistant` | `parts`, `llm_config`, `stop_reason` | no |
| `tool_execution` | one tool call's whole lifecycle (§3–§4) | **yes** |
| `turn_start` | — | no |
| `turn_finish` | `outcome`, `error` | no |
| `cancel_requested` | requested `outcome`, `error` | no |
| `child_conversation` | the link to one subagent and its result (§7) | **yes** |
| `compaction` | `source`, `parts`, `compacted_nodes`, `llm_config`, timestamps (§9) | **yes** |
| `pruned` | replacement `content` for one original entry (§9) | no |

(Every entry also carries the shared base fields — `id`, `parent_id`,
`created_at`, `context_tokens`.)

`id` and `created_at` are `None` until an entry is **committed**. A template a
strategy builds — a registry's birth draft, a `ContextManager` pruned
replacement, a compaction plan's new entry — carries no identity; the
persisting door stamps both. Every entry in `session.entries` has them. A
registry's draft likewise carries its `tool_spec` with no `tool_spec_id` — the
same door files the spec and stamps the id (§7).

## 13. Read a saved session

`pretty_print` renders one conversation as a plain-text transcript — the way to
inspect a `<session-id>.json` without loading it into an app. No argument means
the main one; pass an id for a subagent's.

```python
from luca.agent.core import pretty_print

session = AgentSession.model_validate_json(text)
print(pretty_print(session))                     # the main conversation
print(pretty_print(session, child_id))           # one subagent's, header + depth
```

```
LUCA SESSION fb89c986
Conversation 1a277f0c · idle · 2 turns
Default: openrouter/openai/gpt-5.4 · reasoning medium
────────────────────────────────────────────────────────────────

TURN 2 · 2026-07-24 19:32:13
User
  what's in my local filesystem?

Assistant · step 1 · openrouter/openai/gpt-5.4
  [thinking hidden]

  Tools
  ├─ read(file_path="/", offset=1, limit=200)
  │  └─ DENIED · approval denied by user
  │
  └─ read(
       file_path="/Users/santiagobasulto/code/python/python-py",
       offset=1,
       limit=200
     )
     ├─ ALLOWED · permission rule
     └─ OK · 2 ms
        <path>/Users/santiagobasulto/code/python/python-py</path>
        … (+245 more characters)

✓ completed · stop · 7,782 tokens

────────────────────────────────────────────────────────────────
TOTAL · 2 turns · 3 model calls · 3 tool calls · 13,247 tokens
```

It reads the durable session, not the wire view ([10](10-projection.md)): a
tool node shows the stored `ExecutionResult` or `ToolExecutionError`, the
approval line comes from `approval_status` plus the last decision's
provenance, and the token counts come from `usages` for **this** conversation
(§7). The status line is derived like everywhere else (§10), and a subagent
link renders as a `Subagent · <id>` node with its result — print that id to see
the child's own transcript. Reasoning renders as a marker, tool output clips, and a node id missing
from the store prints as `[missing entry <id>]` instead of raising — a
debugging view of a broken session still has to print.

The TUI exposes it as [`--pretty-print`](contrib/tui/README.md#1-run-it).

Next: [`03-tools.md`](03-tools.md).
