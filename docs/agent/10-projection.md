# Conversation projection

The message list the model sees is **derived state** — recomputed from the
durable session on every call, never stored. The strategy that derives it is
public: `ConversationProjector`. Subclass it to own history policy end to end:
what projects, how tool outcomes read, what gets dropped or injected.

```python
from luca.agent.core import ConversationProjector

runner = AgentSessionRunner(
    session, tool_registry=registry,
    conversation_projector=MyProjector(),     # None → the default projector
)
```

The projector is a runtime collaborator like the permission policy: it lives on
the runner, is never serialized, and is one object — not a stack (plugins don't
contribute projectors). It is **not middleware**; `before_llm_call`
([07](07-middleware.md)) stays downstream for last-mile request edits.

## 1. What the default does

`project(nodes, entries)` walks an ordered path of entry ids and maps each entry
to a canonical `luca.client` message (provider wire formats are the client's job
— the projector never builds OpenAI dicts or Anthropic blocks). It takes the
`nodes` list, not a `Conversation`: a projection is a pure function of a path
and a bag, and passing the container would suggest it could read status or ids
it has no business reading.

| Entry | Projects to |
|---|---|
| `UserMessage` | client `UserMessage`, content in order — text and image blocks |
| `AssistantMessage` | client `AssistantMessage` — text / thinking / tool-call blocks in order, plus the producing model as `provider` / `model` |
| `ToolExecution` (terminal) | one correlated client `ToolMessage` (below) |
| `ToolExecution` gated or parked at `AWAITING_RESULT` | a placeholder `ToolMessage` — the two nonterminal carve-outs (§2) |
| `ToolExecution` whose spec is **private** | nothing — `project_private_execution` (§8) — unless it is a subagent's RESULT execution, which renders that link's task update at its own position (§8) |
| `ChildConversation` | nothing — its result renders at the result execution's position; a link resolved *without* one (a cancel wind-down, a hard-limit settle) renders its task update in place, and an unresolved link renders nothing inside the open turn (§8) |
| `CompactionEntry` | a synthetic user message carrying the summary |
| `PrunedEntry` | its replacement content, under the *original* entry's role and correlation ([11](11-context-and-usage.md)) |
| `TurnFinish(CANCELLED)` | a synthetic user message: `[Request interrupted by user]` |
| `TurnFinish` (other), `TurnStart`, `CancelRequested` | nothing — bookkeeping |

Every per-entry method takes `(entry, entries)` — the resolved entry plus the
read-only entry mapping, so a projection can resolve cross-entry references
(that's how `project_pruned` finds its original). No merging, trimming, or
token counting happens by default — that's yours to add by overriding
`project`.

### Why an assistant message keeps its provenance

Usage, stop reason and the rest of the durable record are NOT copied —
projection reconstructs conversation content, not response objects. `provider`
and `model` are the exception, and they are load-bearing.

A `ThinkingContent` carries opaque provider data: Anthropic's `signature`,
OpenAI's `rs_…` id plus its encrypted payload. Each is minted by one
(provider, model) pair and rejected by every other with a 400 that makes the
conversation permanently unusable. Whether a given attestation may go back on
the wire is therefore per-vendor knowledge, and it lives on the **transport**
(`_attestation_is_replayable`), which compares this provenance against the
model it is about to call and drops the block when they disagree.

That matters because switching model mid-session is one keystroke — the TUI's
`/model` rewrites `session_config.llm_config` and the runner re-reads it on
every call, so the next request replays a history produced by something else.
A projector subclass that rebuilds the message itself must carry the two
fields, or reasoning replay breaks silently.

## 2. Tool outcomes on the wire

`project_tool_execution(entry, entries)` is the single customization point
for every tool status. `COMPLETED` projects the tool's own `result.content` and preserves
`result.is_error`. Every other terminal status derives error text (always
`is_error=True`) from `status` + the structured `error`:

| Status | Default output |
|---|---|
| `NOT_FOUND` | `Unknown tool: 'read_database'.` |
| `INVALID` | `Arguments for tool 'add' are invalid.` + the validation errors as JSON |
| `FAILED` | `Tool execution failed: ConnectionError: …` |
| `REFUSED` | the refusing limit's own message, e.g. `Spawn limit reached (3/3 subagents this turn). …` |
| `REJECTED` / `CANCELLED` / `INTERRUPTED` / `TIMED_OUT` | `[tool execution rejected]` etc. |

All of this wording lives **on the class** — swap a placeholder without
touching the method, or replace the method wholesale:

```python
class MyProjector(ConversationProjector):
    CANCELLED_TURN_MARKER = "[User stopped the previous request]"
    AWAITING_APPROVAL_OUTPUT = "[waiting for your approval — the call has not run]"
    AWAITING_RESULT_OUTPUT = "[still working — the result will follow; do not call it again]"
    STATUS_ONLY_OUTPUTS = {
        **ConversationProjector.STATUS_ONLY_OUTPUTS,
        ExecutionStatus.REJECTED: "The user declined this tool call.",
    }
```

### The two projectable nonterminal states

A **gated** execution — `PENDING` with `approval_status=PENDING`, the policy
explicitly deferred to a human — projects a placeholder `ToolMessage` carrying
`AWAITING_APPROVAL_OUTPUT`:

```text
[tool execution is awaiting approval — it has not run, and this is not its result]
```

with `is_error=False`, deliberately: the call has not failed, it has not run,
and an error result is exactly what makes a model retry a call. It is its own
ClassVar rather than a `STATUS_ONLY_OUTPUTS` row because that table is keyed
by TERMINAL status and the nonterminal carve-outs are named individually.

A **parked** execution — `AWAITING_RESULT`, the tool answered *not yet*
([03](03-tools.md) §7) — is the second, and the same kind of state: a durable
resting point only the application can move. It projects
`AWAITING_RESULT_OUTPUT`, also with `is_error=False`:

```text
[tool execution has not finished — this is not its result; do not call it again, the result will follow when it completes]
```

Heavier wording than the gate's, on purpose. A gate can afford to be terse
because the model has no available action — it cannot approve itself. Here it
does: a bare "still executing" invites a re-call, and a re-call mints a NEW
`ToolExecution` under a new `tool_call_id`, leaving the tool holding two open
calls for one job. That sentence is the framework's one chance to discourage it.

The placeholder is what lets a message posted into a `BLOCKED` conversation
reach the model while the gate — or the parked call — is still open: the path
becomes well-formed, so the drive can run one round, answer the user, and
re-park in the same place ([04](04-runner.md) §9).

**Only a post ever forces either projection.** Nothing else in the framework
calls the model with a nonterminal execution on the path: compaction skips any
conversation with an open turn, pruning refuses a nonterminal execution, and the
context manager leaves it at zero tokens ([11](11-context-and-usage.md)).
`post_message()` into a `BLOCKED` turn is the one caller — plus
`AgentSessionRunner.build_messages()`, which is public, so a debug view or a
token counter projecting a parked conversation reaches it too.

Two properties follow, and they hold for both placeholders:

- **A projected tool message is no longer always final.** Every other
  fabricated tool message comes from a terminal state and never changes; these
  two are replaced by the real result at the same path position once the
  approval is answered or the tool finally returns one — the model's history is
  rewritten underneath them. Consistent with a design that re-derives the whole
  payload on every call and caches nothing, but worth knowing.
- **The next request after it resolves ends with an assistant
  message.** The real result replaces the placeholder in place, above the
  post and the model's answer to it, so the request trails
  `…, tool (real result), user (the post), assistant (the answer)`. A `tool`
  message must answer the assistant turn that issued the call on every
  provider, so the shape cannot be fixed by reordering. Anthropic treats a
  trailing assistant message as a *prefill* — the model continues it rather
  than starting fresh, and with extended thinking enabled the request is
  rejected outright. The framework accepts and documents this; an application
  that needs it normalized overrides `project()` (at the cost of discarding
  the model's own last reply).

## 3. One projection, two consumers

The same `project_tool_execution` output feeds the correlated `ToolMessage` in
the next LLM request **and** the `ToolExecuted` event's `result_text` /
`is_error` — so what your UI renders is exactly what the model is told:

```python
class Redacting(ConversationProjector):
    def project_tool_execution(self, entry, entries):
        message = super().project_tool_execution(entry, entries)
        return message.model_copy(update={"content": redact(message.content)})
```

That is why the projector must be **deterministic** for the same durable
execution: no wall clock, no live registry, no transient state. (The event may
fire now and the request re-project after a reload — they must agree.) Read
only the execution — `status`, `error`, `result`, `raw_tool_call`, and
`tool_spec`, which a session restores from its `tool_specs` store on load
([02](02-data-model.md)).

## 4. History policy — override `project`

Trimming, synthetic context, translations — anything that used to be a
"message middleware" belongs here:

```python
class KeepRecent(ConversationProjector):
    def project(self, nodes, entries):
        return super().project(nodes, entries)[-40:]
```

## 5. Rewriting image media

`_image_block(part)` maps an `ImageContent` to the client's `ImageBlock`.
Override it to rewrite media without touching the rest of the projection —
uploading base64 bytes once and sending an id instead, for example:

```python
class Uploading(ConversationProjector):
    def _image_block(self, part):
        return ImageBlock(source=MediaFileId(file_id=upload(part.source)))
```

`part.metadata` is application-owned and is dropped on the way to the wire.

## 6. Fail-loud rules

Projection never papers over broken state — errors raise `ProjectionError`
instead of producing invented content:

- a conversation node missing from the entry store;
- an entry type the projector doesn't know;
- a nonterminal (`RECEIVED`, `PENDING`, `RUNNING` or `AWAITING_RESULT`) tool
  execution (the runtime never calls the model mid-execution) — with exactly
  two carve-outs, both durable resting points only the application can move
  rather than runtimes in flight: a GATED execution (`PENDING` with
  `approval_status=PENDING`) projects `AWAITING_APPROVAL_OUTPUT`, and a PARKED
  one (`AWAITING_RESULT`) projects `AWAITING_RESULT_OUTPUT` (§2). `PENDING`
  with approval `None` or `ALLOWED`, and `RUNNING`, still raise;
- an UNRESOLVED `ChildConversation` **outside the open turn** — no close may
  leave an unresolved subagent behind, so that state is corruption; inside
  the open turn it is legal and renders nothing (the orchestration is simply
  still running);
- a `COMPLETED` execution without a result;
- a `PrunedEntry` whose referent is missing, whose `pruned_entry_type`
  disagrees with the referent, or whose referent has no pruned projection.

> ⚠️ **Correlation is sacred.** Every projected `ToolMessage` must keep
> `execution.tool_call_id` as its `tool_call_id`, and every model tool call
> must end up with exactly one correlated output. Rewrite content freely;
> never drop or re-key a tool message.

## 7. The one path-level rule

Every rule above is per-entry. One is not, because it cannot be decided from a
single entry: a **compaction bracket** — the whole span
`turn_start → compaction → [cancel_requested] → turn_finish`, whatever its
outcome — projects as **nothing**, while a `compaction` entry reached *outside*
a bracket projects its `parts` as a synthetic user message.

```python
def project_compaction(self, entry, entries):        # override point
    if not entry.parts:                              # scheduled, failed, no-op
        return None
    return UserMessage(content=[self._content_block(p) for p in entry.parts])
```

A summary only means something on the path where the history it replaces is
gone — the new conversation, where the entry sits bare. Inside its bracket it
is the *record of the operation*, on a path that still holds the originals, so
an archived conversation projects its originals and no summary.

> ⚠️ **The rule is required, not tidiness.** A cancelled compaction never
> transitions, so its `turn_finish(CANCELLED)` stays on the active path —
> without the rule, every later request would carry
> `[Request interrupted by user]` about a question the model was never shown.

The rule lives on `project()`, which already owns path-level policy; every
per-entry method keeps its signature. See [`12-compaction.md`](12-compaction.md).

## 8. Subagents on the wire

A parent conversation holds two extra kinds of entry once it spawns
([13](13-subagents.md)), and their rendering has its own override points:

```python
class MyProjector(ConversationProjector):
    CHILD_UPDATE_PREAMBLE = "Subagent task update:\n"                              # the defaults
    CHILD_UPDATE_TEMPLATE = "<task id={task_id} status={status}{completed_at}>\n{content}\n</task>"
    CHILD_COMPLETED_AT_TEMPLATE = ' completed_at="{iso}"'

    def project_child_update(self, link, execution, entries): ...   # the resolution, at its position
    def project_child_conversation(self, entry, entries): ...       # the link itself → None, almost always
    def project_private_execution(self, entry, entries): ...        # → None
```

A subagent's result renders as a **synthetic user message at the RESULT
EXECUTION's path position** — the private execution
`ChildConversation.result_execution_id` names. Position is the point: the
link is appended at spawn time and mutated in place at resolution, so
rendering there would rewrite mid-history on every resolution; the result
execution is appended WHEN the child resolved, so the update always lands
below the parent's last reply and the projected history stays append-only —
exactly what a re-awakened model needs to see ([13](13-subagents.md) §2). The
tag's `id` is the spawn payload's `task_id` (the identifier `stop_subagent`
takes), `status` is `completed` / `failed` from the result's own verdict, and
`completed_at` is absolute UTC — deterministic, unlike relative wording. A
synthetic user message is the only legal shape: the spawn tool already got
its own `ToolMessage`, and a second one correlating to the same
`tool_call_id` would be a protocol violation.

`project_child_conversation` — the link at its OWN position — renders nothing,
with one exception: a link resolved *without* a result execution
(`result_execution_id is None` — the cancel wind-down and the hard-limit
settle write the result directly) renders its tag in place, timestamp-less,
inside what is by construction a failing bracket.

`project_private_execution` returns `None` — a private tool was never
advertised, so the model never made that call, and a `ToolMessage`
correlating to a call the model did not make is malformed. (The same is true
of a PRUNED private execution: `project_pruned` renders it as nothing.)
`project_tool_execution` is still called directly for the `ToolExecuted`
event, so a private execution's event stays self-describing even though its
wire projection is nothing.

Next: [`11-context-and-usage.md`](11-context-and-usage.md).
