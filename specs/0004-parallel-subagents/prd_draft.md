# Parallel Subagents — V0

Status: design agreed. This document describes **intent and behavior**, not
implementation. Sections marked **Settled** record decisions that have already
been argued and closed — do not re-litigate them; extend around them.

---

## 1. Intent

The LLM can spawn subagents that work in parallel on independent tasks and
report back. This is a basic capability of every AI coding agent.

The bet of this design is that `luca`'s data model already supports it: the
session separates **stores** (a flat entry space, a tool-spec catalog, a usage
store) from **traversal** (a conversation is an ordered list of entry ids over
that store). A subagent is therefore not a new session and not a new runtime —
it is one more conversation over the same stores, linked into its parent's path
by one new entry type.

V0 is deliberately minimal. The goal is a data model and an architecture that
will not have to change when the feature grows.

## 2. Scope

**In V0:**

- A tool that spawns a subagent, one conversation per subagent.
- Subagents run in parallel with each other.
- The parent turn stays open until every subagent it spawned has finished.
- Per-subagent approvals and per-subagent cancellation.
- `subagents_max_depth = 1` — a subagent cannot spawn subagents.
- **Private tools** (§3.6) — a tangential capability this feature needs and is
  simply the first consumer of: a `ToolSpec` the runtime can resolve and
  dispatch but that is never advertised to the model. Nothing about it is
  subagent-specific.

**Not in V0, but the data model must not preclude them:**

- **Background subagents** — a subagent whose result arrives after the parent's
  turn has already closed. Section 5.1 explains why the spawn tool call
  completes immediately, which is what keeps this possible later.
- Depth > 1. The data model supports arbitrary nesting; only the runtime limit
  is fixed at 1.
- Named subagent types with their own prompts, tools or models.

## 3. Data model

One new entry type, one new field on `ToolExecution`, one new field on
`ToolSpec` (§3.6), and a reshaped session container. Nothing else changes.

```python
class ChildConversation(Entry):
    type: Literal["child_conversation"] = "child_conversation"
    conversation_id: str
    tool_execution_id: str        # which tool call created this child conversation
    execution_result: ExecutionResult | None = None
    # possibly more later, e.g. timeout


class ToolExecution(Entry):
    ...                                   # every existing field unchanged
    conversation_id: str | None = None    # PROVENANCE, not traversal — see §3.2


class Conversation(BaseModel):
    id: str                                          # usage records key on it
    nodes: list[str] = Field(default_factory=list)   # ordered entry ids = THE path
    created_at: int
    updated_at: int
    previous_conversation_id: str | None = None      # what compaction archived
    depth: int = 0                                   # 0 = main; a child is its parent + 1
    # NO status field — see §3.1


class ConversationRuntimeStatus(BaseModel):
    """What `SessionRuntimeStatus` was, scoped to the conversation it was always
    really about. Derived, never stored."""

    status: ConversationStatus = ConversationStatus.IDLE   # the four values of §9.6
    turn_count: int = 0                                    # conversational turns on this path
    step_count: int = 0                                    # AssistantMessages in the open turn


class AgentSession(BaseModel):
    main_conversation_id: str
    conversations: dict[str, Conversation] = Field(default_factory=dict)  # the catalog of ALL

    def get_conversation_status(self, conversation_id: str) -> ConversationRuntimeStatus:
        """Recomputed from the nodes on every call. The only door."""
```

`conversations` is the store/catalog, keyed by id like every other store
(`entries`, `tool_specs`, `usages`), and `main_conversation_id` is a pointer into
it. This mirrors the existing entries/nodes split one level up: conversations are
stored in one place and referenced from several. `active_conversation` is gone —
with several conversations advancing at once there is no single active one.

**A conversation owns its own history, and owns it by reference.**
`conversation_history` leaves `AgentSession`; it was never a session-level fact.
Compaction archives one conversation and installs another over a rewritten path,
so the predecessor belongs to its successor — `previous_conversation_id`. A
subagent that compacts extends *its own* chain and never touches the main
conversation's, which is exactly what one flat session-level list could not
express once children exist.

The link is deliberately **not** a nested `list[Conversation]`. An archived
conversation stays a first-class row in `conversations` — it has to, because
usage records key on conversation id and §8's session total sums across the
catalog, archived conversations included. Nesting the objects would either put
the same conversation in two places (drift, and a doubled serialization) or
leave `usages[cid]` with no row to resolve against. One home per conversation;
pointers everywhere else.

**A transition re-points whoever names the conversation.** Compaction installs a
new conversation, so the name has to move with it: `main_conversation_id` for the
main conversation, and the `ChildConversation` entry's `conversation_id` for a
subagent's. The archived row stays reachable backwards through
`previous_conversation_id`, and nothing needs a forward link. (This only arises
if subagent conversations compact at all — §8 leaves that open.)

**No parent link, but depth IS stored.** A conversation carries no
`parent_conversation_id`: the parent mints the child's handle, and after a reload
it re-mints from its own unresolved `ChildConversation` entries (§9.5), so
parent → child is the only direction anything traverses. That is precisely why
`depth` has to be a field — with no link upwards it is not derivable from the
conversation alone, and the registry reads it on every `get_tools` call (§3.1,
§7). It is stamped at creation and copied when a compaction transition installs a
successor, alongside `previous_conversation_id`.

`ChildConversation` is a node in the **parent's** path. That position is what
makes the subagent's outcome appear in the parent's projected history at the
right point. The child conversation itself lives in the catalog like any other.

Everything else is reused as-is: entries, tool specs, usage records, turn
markers. A subagent does not duplicate a `ToolSpec` that is already cached in
the session, and does not get its own session file.

### 3.1 Conversation becomes an argument

A great deal of the framework is implicitly scoped to "the one active
conversation" — not by passing it, but by reading `session.active_conversation`
behind the caller's back. With several conversations advancing at once that
referent stops existing, so the scope becomes an explicit argument. These are
corrections the multi-conversation model forces on its own; they would be right
even if subagents were never built.

**1. `SessionRuntimeStatus` → `ConversationRuntimeStatus`, and it is never
stored.** It was always a conversation-scoped object wearing a session name —
`status`, `turn_count`, `step_count` are all facts about one path. It becomes
`ConversationRuntimeStatus`, and the only door is
`session.get_conversation_status(conversation_id)`, recomputed from the nodes on
every call.

`Conversation.status` is **deleted**, not upgraded to hold the new model. The
justification for keeping it was always "we never trust it, we recompute from the
nodes anyway" — and a field nobody trusts is a rendering, not state. Concretely:

- Storing the full object widens a documented exception. "Nothing on the session
  is transient" has exactly two exceptions today; a persisted status is a third,
  and a persisted `ConversationRuntimeStatus` lets a stale file disagree with the
  nodes in three ways instead of one.
- Staleness gains a blast radius. A stale `status` self-heals today because the
  runner re-derives *the one* conversation on load. With N conversations that
  needs a rule for which ones get refreshed and when — and archived conversations
  must be left frozen, exactly as the compaction transition already insists.
  Keeping N cached statuses fresh means writing to conversations nobody is
  driving.
- Nothing needs it stored. A registry holding `(session, conversation_id)` has
  the session; handles have the runner; `RunResult` carries a value at a point
  in time, which is a different thing.

`_set_status` and the transition's `outgoing.status = IDLE` disappear with it —
the whole denormalized-status machinery goes, rather than being generalized to N
conversations. If the status is wanted in the serialized JSON for a human reading
the file, that is `pretty_print`'s job; a field that serializes but must not
validate is a round-trip trap under `extra="forbid"`, so it needs verifying
rather than assuming.

The method lives on the **session**, not on `Conversation`, and has to: §9.6's
derivation is subtree-aware (a parent is `BUSY` while its children work), and a
`Conversation` cannot see its children's entries.

**2. `ToolRegistry` takes the conversation.**

```python
async def get_tools(self, session: AgentSession, conversation_id: str) -> list[ToolSpec]: ...
```

`get_tools(session)` silently meant "the active conversation"; there is no active
conversation any more. This is what lets a registry answer differently for a
subagent — omitting the spawn tool at the depth cap (§7, **Gating**) — without
the core making tool-list policy of its own.

All four methods take it, not just `get_tools`. A half-scoped contract invites
the same question again, and `decide()` is the obvious next case: "approvals
behave differently inside a subagent" is a policy most applications eventually
want. `ContextManager.should_compact` / `compact` carry the identical defect and
get the identical fix — §8 records their new signatures, and why the three
per-entry `ContextManager` methods do NOT change.

**3. `system_prompt_parts` callables take `(session, conversation_id)`**,
replacing `(session_config, runtime_status)`. `session.session_config` covers the
first argument, and the second was a session-wide `SessionRuntimeStatus` that no
longer means anything — a callable that wants `step_count` asks
`session.get_conversation_status(conversation_id)` for the conversation it is
actually building a prompt for.

**4. The scope is always the ID, never the `Conversation` object — Settled.**
Every signature in this refactor passes `conversation_id: str`. Nothing in a
public contract passes a `Conversation`. A collaborator that needs the object
resolves it — `session.conversations[conversation_id]` — which is one lookup,
because the session is always the first argument.

Three reasons, and the second is the load-bearing one:

- **Uniformity.** Every cross-reference in the data model is already an id:
  `usages` keys, `main_conversation_id`, `previous_conversation_id`,
  `ChildConversation.conversation_id`, `ToolExecution.conversation_id` (§3.2),
  and `session.get_conversation_status(conversation_id)`. A `Conversation`
  parameter would be the single exception.
- **`Conversation` is LIVE and MUTABLE.** Its `nodes` list is appended to by the
  runner while application code holds it. Handing the object to a registry, a
  context manager or a tool invites exactly the stale-reference class of bug the
  "`ToolExecution` is a snapshot, the `AgentSession` is live" rule already exists
  to prevent — and here it would be worse, because the object looks immutable
  and is not. An id cannot go stale; resolving it always yields current state.
- **One door.** `AgentSession` stays the only way to reach conversation state,
  which is what keeps `get_conversation_status()` "the only door" (item 1) true
  in practice rather than just on paper.

Concretely, §7's gating predicate is
`session.conversations[conversation_id].depth >= subagents_max_depth`.

**Scope.** This is a repo-wide refactor: every registry, every plugin, every
prompt callable and every tool in the codebase is updated in the same change
(§3.3 covers the tool-authoring surfaces). `luca` is V1 and unreleased — no
shims, no deprecation path, no dual signatures.

### 3.2 `ToolExecution` carries its conversation — **Settled**

`ToolExecution` gains exactly one field:

```python
conversation_id: str | None = None
```

**Why this entry and no other.** `ToolExecution` is the only entry a consumer
ever receives DETACHED from a path. It is handed to `ToolRegistry.decide()`, to
the four tool middleware hooks, and carried as a deep snapshot inside
`ToolCallReceived` / `ToolExecutionStarted` / `ToolExecuted` /
`ApprovalRequired` and `RunResult.pending_approvals` — none of which carry a
session. Every other entry reaches a consumer by walking `conversation.nodes`,
where the conversation is already in hand. Do NOT generalize the field onto
`Entry`: it goes on the one type that structurally needs it.

**It is provenance, never traversal.** It records the conversation the execution
was BORN in, exactly as `AssistantMessage.llm_config` records the config that
produced that message, and with the same standing as `parent_id` — a pointer
nothing resolves a path through. Nothing may read
`session.conversations[execution.conversation_id].nodes` and assume membership.

There is one real case behind that distinction. An entry CAN be referenced by
two conversations — a compaction carries node ids from the archived conversation
into its successor — but only ever within one LINEAGE; nothing copies node ids
between sibling conversations. And compaction never runs while a conversational
turn is open, so for every LIVE use (a gated execution, an execution in
transition, `notify()`) the birth conversation IS the current conversation. For
a historical execution carried across a compaction the field names the
predecessor — under-reporting, never wrong.

**The runner stamps it.** It joins `id` / `parent_id` / `created_at` in the
identity set the runner owns: a registry's `create_execution` birth draft leaves
it `None` and a registry must NOT set it, which is why the type is `str | None`.

**What it closes.** Every approval surface keeps its current shape and gains
attribution for free — no wrapper type, no new vocabulary:

| surface | type | change |
|---|---|---|
| `runner.pending_approvals()` | `list[ToolExecution]` | subtree-scoped (§9.2); TYPE UNCHANGED |
| `run.approvals` | stream of `ToolExecution` | new door (§9.7); same element type |
| `RunResult.pending_approvals` | `list[ToolExecution]` | unchanged |
| `ApprovalRequired.executions` | `list[ToolExecution]` | unchanged |
| `run.notify(execution)` | — | reads `execution.conversation_id` (§9.8) |

There is deliberately **no approval-request wrapper object**. An earlier draft of
§9.7 / §9.9 showed `request.execution`; that is superseded and those loops now
iterate `ToolExecution` directly. A wrapper would also collide with
`contrib.resource_permissions.PermissionRequest`, which is a different concept
(one resource step inside one execution's approval).

Contrib is untouched by this: `PermissionStrategy.pending_requests(execution)`,
`apply_answer(execution, answers)` and `build_approval_prompts(execution,
strategy)` keep their signatures, and an interactive app can now label which
subagent is asking without any of them changing.

### 3.3 Writing a tool is a scoped contract too — **Settled**

`Tool.execute` is the single most-written extension point in the framework, and
it has the §3.1 defect in its purest form: nothing it receives identifies a
conversation. `args` is the model's payload; `session` is the whole session.
A tool body cannot tell whether it is running for the main agent or for
subagent B.

This is not hypothetical. `contrib/memory`'s `MemoryPlugin` holds ONE `store`
dict shared by its scratchpad and todo tools, so with subagents the parent and
every child write over each other's todo list. That is shipping code today.

**The core does not change.** `PreparedTool` stays exactly as it is:

```python
async def run(*, cancellation_token: CancellationToken) -> ExecutionResult
```

The conversation reaches the body through the CLOSURE, not through a new core
parameter: `prepare(session, conversation_id, execution)` has the id in hand and
binds it, exactly as it already binds the validated arguments and the session.
So the core still depends on no Python tool class, and a registry fronting a
remote tool server is unaffected.

**Five contrib surfaces move together.** Doing only `execute` would leave the
same hole one call earlier, in the approval path:

```python
Tool.execute(args, session, conversation_id, *, cancellation_token)  -> ExecutionResult
Tool._execute(args, session, conversation_id, *, cancellation_token) -> str
tool() / tool_class()                       # the same contract for factory-built tools
get_approval_context(args, session, conversation_id) -> dict           # duck-typed convention
ResourcePermissionToolMixin.build_permission_requests(args, session, conversation_id)
```

`get_approval_context` matters as much as `execute`. It runs inside
`create_execution` — which §3.1 already scopes — and "describe this call for the
approval prompt" is precisely where "a subagent is asking for this" belongs.

Per §3.1 item 4 all five take the **id**, never a `Conversation`.

### 3.4 Every seam is concurrency-safe by contract — **Settled**

Today the runner drives one conversation at a time, so a `ToolRegistry`, a
`ContextManager`, a `ConversationProjector` and a tool instance are effectively
called serially. With N conversations advancing at once that stops being true,
and **the obligation moves to the implementor**: the framework does NOT
serialize these calls.

That is the decision. The rejected alternative was a runner-side lock around the
seams, which would keep every existing implementation correct — and would also
serialize the exact work parallel subagents exist to overlap. A registry doing a
200ms remote resolution would gate every sibling subagent behind it. Not worth
it.

**What can be in flight at once.** The rule is not new for everything — two of
these are already concurrent today, across sibling tool calls in one assistant
response:

| call | concurrent today | concurrent after |
|---|---|---|
| `create_execution` | yes — siblings, via `asyncio.gather` | yes, **and across conversations** |
| `decide` (and the `PermissionPolicy` behind it) | yes — siblings, via `asyncio.gather` | yes, **and across conversations** |
| `get_tools` | no | yes, across conversations |
| `prepare` | no | yes, across conversations |
| the prepared callable / `Tool.execute` | no | across conversations; still SEQUENTIAL within one |
| `get_approval_context`, `build_permission_requests` | inside `create_execution` | yes, across conversations |
| `ContextManager.calculate_context` / `process_tool_output` / `prune_entry` | no | yes, across conversations |
| `ConversationProjector.project` | no | yes, across conversations |
| prompt-part callables + `SystemPromptAssembler` | no | yes, across conversations |
| every middleware hook | no | yes, across conversations |
| `should_compact` / `compact` | no | main conversation only (§8) — but runs while children drive |

**The rules an implementor owes us.** These extend the registry-author rules in
`tool_registry.py`, they do not replace them:

1. **No per-call state on `self`.** A registry, a context manager and a tool are
   single instances shared by every conversation. `self` is for immutable
   configuration. Anything stashed on it and read back after an `await` may
   belong to another conversation by then — this is the failure mode, and it is
   silent.
2. **State keyed by `conversation_id` needs no lock.** Tool dispatch within one
   conversation is sequential, so exactly one body ever touches a given
   conversation's slot at a time. This is why §3.3 is the fix for
   `MemoryPlugin`, not a lock. (Dispatch being sequential is a runner CHOICE,
   documented as such; if it ever becomes parallel this line has to be revisited.)
3. **State deliberately shared ACROSS conversations needs an `asyncio.Lock`
   around the mutation, never around the I/O.** A counter, a rate limiter, a
   cache, a shared connection. Holding a lock across a slow await re-serializes
   the tree and gives back everything this feature buys. Do the I/O unlocked,
   take the lock for the read-modify-write.
4. **`asyncio.to_thread` is real parallelism.** The existing rule sends blocking
   work there, which means two conversations' bodies genuinely run on two OS
   threads. Nothing the single-threaded argument buys you survives that boundary:
   shared state touched inside needs a `threading.Lock`, on both sides.
5. **Cancellation can kill a body at any await, and the lock is not what is at
   risk.** `async with` releases correctly (rules 6 and 7 there), but the DATA
   can be left half-written and a sibling may read it. Make the mutation the last
   await-free step, or make it idempotent.
6. **Process-global and external resources are not a locking problem.** The
   current working directory, relative paths, temp files, subprocesses, a git
   worktree, a connection that cannot have two queries in flight. Scope them per
   conversation instead — a mutex around a shared cwd is the wrong shape.

Rule 1 is the one to lead with in the docs. The others have visible symptoms;
rule 1 produces a tool that returns another subagent's answer, intermittently,
with nothing in the log.

### 3.5 What this breaks in contrib — audited and **Settled**

Every contrib package that implements tools or approvals was reviewed against
§3.1–§3.4. The results are recorded here so the audit is not repeated: what
follows is the complete list, both the things that change and the things that
deliberately do not.

**Fixed by keying an internal data structure on `conversation_id`.** All four
are the same shape — runner-scoped state that was always logically
conversation-scoped, and only looked correct because there was one conversation:

1. **`ProxyToolRegistry._route`.** The proxy keeps one `{tool name → child
   registry}` dict, rebuilt wholesale by `get_tools` and read by
   `create_execution` / `decide` / `prepare`. That was safe only while
   `get_tools` returned the same list every time — and §3.1 plus §7's gating
   make it return a DIFFERENT list per conversation, which is the entire point.
   Both directions fail, and neither is a data race (the dict swap is atomic) —
   both are wrong routing:

   - Main warms the route WITH `spawn_subagent`; a child at the depth cap then
     calls `get_tools`, the registry withholds the tool, and the route is
     overwritten WITHOUT it. Main's next spawn resolves to `None` and records a
     silent NOT_FOUND.
   - A child resolves `decide` / `prepare` through a route warmed by the MAIN
     conversation, so a tool deliberately withheld from that child routes and
     dispatches anyway — the permission bypass `_resolve`'s docstring already
     exists to prevent.

   **Decision: the route is keyed by conversation.** Same for the `_warmed` flag.

2. **`FileReadTracker` (contrib/shell).** One tracker is built by
   `ShellAccessPlugin` and shared by `ReadTool`, `EditTool` and `WriteTool`, so
   the read-before-edit guard is runner-scoped: subagent A reads `main.py`, and
   subagent B — which never read it — passes `was_read` and edits blind. A
   safety guard that silently weakens the moment a second conversation exists.
   **Decision: the tracker is keyed by conversation.**

3. **`MemoryPlugin`'s `store`.** Covered by §3.3, same fix, same reason.

4. **`calculate_context_used`.** An exported part of
   `simple_context_manager`'s public surface (the gauge, consumed by the TUI's
   context bar) that sums `session.active_conversation.nodes`.
   **Decision: `calculate_context_used(session, conversation_id)`.**
   `SummarizingContextManager.should_compact` / `compact` take §8's signatures.

**Evaluated and deliberately UNCHANGED.** Recorded so a later reader does not
"fix" them:

- **`PermissionStrategy.add_rule` stays global.** An ALWAYS-scoped answer given
  inside a subagent installs a rule that also covers the main conversation and
  every sibling. This is accepted for V0. It is a POLICY question, not a defect:
  the strategy is concurrency-safe as written (`decide`'s body has no awaits, and
  `_verdicts` is keyed by execution id, which is unique per call). Per-scope
  rules are a later version's problem if anyone wants them.
- **`SimpleToolRegistry`** is already concurrency-clean: `tools_by_name` is
  immutable construction config and `prepare` binds locals into the closure it
  returns. Only its four signatures change.
- **Shell tool state** — `workdir`, `rg_path`, `shell`, `output_dir` — is all
  immutable construction config. There is no `os.chdir` anywhere in contrib and
  subprocesses are given an explicit `cwd=`, so nothing mutates process-global
  state. Nothing to do.
- **Plugin hooks keep taking `(agent_session)` only.**
  `get_tool_registry` / `get_system_prompt_parts` / `get_middleware` are
  construction-time and produce one object per runner; the conversation arrives
  per call on the objects they return. Adding a conversation to a plugin hook
  would be the wrong fix.
- **One `workdir` shared by every conversation** is inherent to V0, not a
  defect: subagents working the same repo is the point. Two children writing the
  same file is §3.4 rule 6 territory and the application's call.

**Scope.** This list is what the AUDIT found, not the full work item. The entire
codebase will need to be adapted to these changes — the TUI most visibly
(history replay walks the active conversation, the approval modal must name
which conversation is asking now that `pending_approvals()` is subtree-scoped,
and events from several conversations arrive interleaved on one stream). Those
fall out of implementation planning; they are not decisions this document owes.

### 3.6 Private tools — a new, tangential capability

This is **not a subagent concept**. It is a standalone capability that subagents
happen to be the first consumer of, and it resolves an item §7 previously left
open ("the concept of private tools comes later" — it is now here).

**The field, and the ClassVar.**

```python
class ToolSpec(BaseModel):
    ...
    is_private: bool = False


class CreateConversationResult(Tool):
    is_private = True        # stamped onto the spec by get_tool_spec()
```

`Tool` declares it as a `ClassVar` and `get_tool_spec()` stamps it exactly like
`tool_kind` / `namespace` / `version`; `tool()` / `tool_class()` accept it too.

**It changes exactly one thing: the tool is never advertised to the model.** The
runner omits private specs when it builds the wire tool list, and
`tool_spec_to_luca_tool` drops the field like every other non-wire field.
Everything else is untouched:

- **`get_tools()` still returns it.** That is the whole point. The RUNTIME must
  still see the tool, because that is how it resolves, prepares and dispatches
  it. A registry that simply omitted the tool from `get_tools` — the alternative
  §7 used to float — would hide it from the runtime that has to invoke it.
- The spec is filed in `session.tool_specs` under `spec_id()` as normal.
- A `ToolExecution` entry is created and recorded as normal: approvals,
  middleware, timeouts, cancellation, events, `context_tokens`, usage. A private
  tool is written exactly like any other tool.

**Two rules it forces, both about the wire.**

1. **A private tool's execution never projects a `ToolMessage`.** This one is
   FORCED, not a policy choice: a private tool is invoked by the runtime, so no
   `ToolCall` for it exists in any `AssistantMessage` on the path, and a tool
   result carrying a `tool_call_id` the provider never issued is a protocol
   violation every provider rejects. There is no projector setting that makes
   this legal.

   **Whether it projects as anything ELSE is projector policy, and V0's answer
   is "no".** The default projector ignores a private execution entirely, and
   that is safe here for one specific reason: its output already reaches the
   model through the entry that owns it — for `CreateConversationResult`,
   `ChildConversation.execution_result` (§3, §4.2). Projecting it a second time
   would duplicate the content, not add it.

   A future projector MAY render private executions as synthetic **user**
   messages. That is a well-established shape in this framework, not a
   workaround: `project_compaction` renders a `CompactionEntry` as a synthetic
   user message today, and a CANCELLED `TurnFinish` becomes
   `[Request interrupted by user]` the same way. Nothing here precludes it and
   it is deliberately out of scope for V0 — the point is only that the ToolMessage
   channel is closed, not that private work is structurally invisible.
2. **A model tool call naming a private spec is refused.** The tool was never
   offered, but a model can still emit a name it was never given. The runner
   records `NOT_FOUND` — from the model's point of view that tool does not exist
   — rather than resolving and dispatching it. Without this rule "private" is
   advisory, and any model that guesses the name gets to call it.

**Why the flag lives on `ToolSpec`.** The same argument §7 makes for
`output_schema`: the thing that builds the wire tool list is the CORE, so a
marker only a registry understands cannot express this. `is_private` is the
second field the core reads off a spec, and it is there for the same reason as
the first. `ToolSpec.metadata` stays exactly what it is documented to be —
free-form, registry-owned, never interpreted by the core.

Like every other field it is **definition-scoped** and participates in
`spec_id()`, so a tool that gains `is_private` is a new row. Being a `ClassVar`,
that holds automatically.

**Deliberately not in scope.** Other uses exist — runtime-only utility tools,
tools gated behind application state, tools a middleware injects — and none are
designed here. V0 adds the field, the ClassVar, the wire-list filter and the two
rules above. Nothing more.

**Documentation impact.** `ToolSpec` is documented as "the advertisement sent to
the model" in `docs/agent/02-data-model.md` and `docs/agent/03-tools.md`. That
stays true with exactly one exception, and the docs have to name it.

## 4. Behavior

### 4.1 Timeline

**Stage 1 — the model asks for two subagents.** The turn is open; nothing has
been executed yet.

```
main_conversation_id = #C1

#C1: u1 > t1 > A1(tc1, tc2)                          | Running

u1: UserMessage("Research A and B in parallel, then compare")
t1: TurnStart()
A1: tc1: spawn_subagent("Research A")   — ToolSpec #spec1
    tc2: spawn_subagent("Research B")   — ToolSpec #spec1
    finish_reason: tool_call

ToolSpecs:
#spec1: spawn_subagent(prompt, description, task_id)
        output_schema declares `is_subagent_spawn` — what the gate reads (§7)
```

**Stage 2 — the spawns execute.** Both tool calls complete immediately: their
job was to create a conversation, and they did. Each result carries the spawn
payload in `structured_content`; the runner reads it, and two
`ChildConversation` entries join the parent's path while two conversations are
created and seeded.

```
#C1: u1 > t1 > A1(tc1, tc2) > TE1 > TE2 > CC2 > CC3  | Running
#C2: u2 > t2                                         | Running
#C3: u3 > t3                                         | Running

TE1: ToolExecution(spawn_subagent("Research A"), status=COMPLETED)
     result.structured_content = {is_subagent_spawn: true, task_id: …, prompt: …}
TE2: ToolExecution(spawn_subagent("Research B"), status=COMPLETED)
     result.structured_content = {is_subagent_spawn: true, task_id: …, prompt: …}
CC2: ChildConversation(conversation_id=#C2, tool_execution_id=TE1)
CC3: ChildConversation(conversation_id=#C3, tool_execution_id=TE2)

u2: UserMessage("You are a subagent, you're tasked with researching A")
u3: UserMessage("You are a subagent, you're tasked with researching B")
t2: TurnStart()
t3: TurnStart()
```

The parent turn stays open. The parent conversation cannot call the model again
until every `ChildConversation` in the open turn has resolved.

**Stage 3 — the children work in parallel.** Each child drives its own
conversation with the full ordinary machinery: assistant messages, tool calls,
tool executions, approvals.

```
#C1: u1 > t1 > A1(tc1, tc2) > TE1 > TE2 > CC2 > CC3  | Running
#C2: u2 > t2 > A2(tc3, tc4) > TE3 > TE4              | Running
#C3: u3 > t3 > A3(tc5) > TE5                         | Running

A2: tc3: read_file(main.py)        — ToolSpec #read_file
    tc4: read_file(pyproject.toml) — ToolSpec #read_file
A3: tc5: read_file(uv.lock)        — ToolSpec #read_file

TE3: ToolExecution(read_file(main.py), status=PENDING)
TE4: ToolExecution(read_file(pyproject.toml), status=PENDING)
TE5: ToolExecution(read_file(uv.lock), status=PENDING)

ToolSpecs:
#spec1: spawn_subagent(prompt, description, task_id)
#read_file: read_file(path, ...)
```

**Stage 4 — the children finish and the parent resumes.** A child is finished
when its turn bracket closes. A child conversation never receives user messages,
so the turn closing is the whole signal. The child's outcome is derived into
`ChildConversation.execution_result`, and the parent can call the model again.

"Its turn bracket closes" means **whatever the outcome** — a child that failed
or timed out on its LLM call is a finished child, not an exception travelling
upward. See §5.6, which settles that.

```
#C1: u1 > t1 > A1(tc1, tc2) > TE1 > TE2 > CC2 > CC3 > A4   | Running
#C2: u2 > t2 > A2(tc3, tc4) > TE3 > TE4 > AC2 > TF_C2      | finished
#C3: u3 > t3 > A3(tc5) > TE5 > AC3 > TF_C3                 | finished

TE3/TE4/TE5: status=COMPLETED, output=…

AC2: AssistantMessage("I found the issue reading main.py and pyproject.toml. …")
AC3: AssistantMessage("I read uv.lock but couldn't find anything")

TF_C2: TurnFinish(outcome=COMPLETED)
TF_C3: TurnFinish(outcome=COMPLETED)

A4:  Reasoning: "The subagents found something interesting… I better investigate more"
```

### 4.2 Projection

The parent's wire history is derived, as always. The spawn calls project as
ordinary tool outputs carrying a short status line; the subagent results project
separately from their `ChildConversation` entries. One possible rendering of
Stage 4's parent conversation:

```
User: Research A and B in parallel, then compare
Assistant:
    tool_call_1: spawn_subagent("Research A")
    tool_call_2: spawn_subagent("Research B")
User:
    tool_call_1: "COMPLETED"     # wording can reflect the child's outcome —
    tool_call_2: "COMPLETED"     #   cancelled / interrupted / failed / completed
User:
    """The result of the subagents was:
<task id=tool_call_1>
    <tool_call tc3 read_file(main.py) contents=…>
    <tool_call tc4 read_file(pyproject.toml) contents=…>
    <text>I found the issue reading main.py and pyproject.toml. …</text>
</task>
<task id=tool_call_2>
    <tool_call tc5 read_file(uv.lock) contents=…>
    <text>I read uv.lock but couldn't find anything</text>
</task>"""
```

This is one option of many. The projector and adapter are the customization
point: how much of a child's transcript to include, and how to render a child
that failed or produced nothing, are projector policy. If the child has a result,
render it; if not, say so or omit it. Richer strategies come later — the point of
V0 is that the architecture already puts this decision in the right place.

## 5. Settled decisions

These have been argued and closed. They are recorded here so the reasoning is
not rediscovered and so the trade-offs are not mistaken for oversights.

Further decisions are equally settled but live where they belong topically
rather than in this section, and carry the same weight:

- **§3.1 item 4** — every scoped signature passes `conversation_id: str`; no
  public contract passes a `Conversation` object.
- **§3.2** — `ToolExecution.conversation_id`, and the fact that no
  approval-request wrapper type exists.
- **§3.3** — the five contrib tool-authoring signatures take `conversation_id`,
  and the core's `PreparedTool` does not change.
- **§3.4** — every strategy seam and tool body must be concurrency-safe; the
  framework does not serialize them.
- **§3.5** — the contrib audit: what is keyed by `conversation_id`, and what is
  deliberately left alone.
- **§3.6** — private tools (`ToolSpec.is_private`), a standalone capability this
  feature is the first consumer of.
- **§7.1** — the tools ship as `SubagentsPlugin` in `contrib/subagents/`,
  auto-approved in their own registry, with a callable system-prompt part driven
  by the same predicate as the gate.
- **§8** — `ContextManager`'s new signatures, and compaction is
  main-conversation only in V0.
- **§9.6** — `post_message` and `schedule_compaction` are main-conversation
  only, `post_message` requires `IDLE`, and message queueing is removed.

### 5.1 The spawn tool call completes immediately — **Settled**

`ToolExecution` for `spawn_subagent` reaches `COMPLETED` as soon as the child
conversation is created. It does **not** stay open until the child finishes.

The scope of the tool is to spawn, and it did. More importantly: **background
subagents require it.** A subagent whose result arrives after the parent's turn
has closed cannot be represented by a tool call that is still open — a
non-terminal execution blocks the parent's projection by design. Closing the
execution at spawn time is what keeps background mode available later without
changing this foundation.

The consequence, accepted: the spawn call's tool output carries a status line
rather than the subagent's answer, and the answer reaches the model as separate
content derived from `ChildConversation`.

### 5.2 The order results land in is nondeterministic — **Settled, and fine**

Subagents are spawned in a fixed order and finish in whatever order they finish,
so the parent's path — and therefore its projected history — can differ between
two runs of identical work. This is accepted. It is not worth constraining
subagent scheduling or delaying results to make the path reproducible.

### 5.3 The signal that creates a child is a declared schema — **Settled**

*Revised.* This was settled before tools had structured output, and the earlier
version — an untyped signal in `ExecutionResult.metadata` — is superseded.

`spawn_subagent` declares an output schema containing an `is_subagent_spawn`
field and returns an `ExecutionResult` whose `structured_content` carries the
facts needed to create the child (task id, prompt, description, and the name of
the tool that derives the child's result). The runner acts on that payload.

**What stands from the earlier decision.** The core recognizes a spawn by a
*convention* — the field name `is_subagent_spawn` — and not by a core type the
tool has to import or subclass. That is what keeps "what a spawn looks like"
owned by the plugin and customizable without touching the core, and it is the
whole reason the flag exists: a generic reader needs something generic to read.

**What changed.** The convention now rides on a *published schema* instead of a
free-form dict. The plugin declares the shape once, the spec carries it before
any call is made, and the same declaration serves both the gate and the
handshake (§7). The signal is no longer untyped on the plugin's side —
`SubagentSpawn` is a real model — even though the core still matches one field
name.

### 5.4 Depth is capped at 1 in V0 — **Settled**

The data model supports arbitrary nesting. The runtime allows exactly one level:
a subagent cannot spawn subagents. The implementation may rely on this.

### 5.5 Background subagents are out of scope for V0 — **Settled**

Not implemented, not designed here. The only requirement V0 carries is that
nothing in the data model forecloses them (see 5.1).

### 5.6 A child's failure never escapes the child — **Settled**

Today a turn that times out or fails on the LLM call closes its bracket
(`TurnFinish(TIMED_OUT | ERRORED)`) **and re-raises** through `await run` /
iteration, with `run.result` left `None`. That contract is unchanged for the
main conversation. It does **not** extend upward from a child.

**The rule.** When a child conversation's drive fails — provider error,
timeout, anything that would re-raise today — the child's turn closes `ERRORED`
(or `TIMED_OUT`) exactly as it would anywhere else, its `ChildConversation`
**resolves** through the ordinary §7 handshake, and the resulting
`ExecutionResult` describes the failure. The parent then continues: to the
parent, a failed child is just a finished child whose result says it failed, and
the model reads that like any other tool output. The exception is never
propagated into the parent's run, and never ends the parent's turn.

**Why it cannot be otherwise.** The parent's turn is blocked on its children
(§4.1). An exception that escapes a child has exactly two other destinations,
and both are broken: propagating it to the parent's `await run` lets one
subagent's transient provider error kill work its siblings already completed;
leaving it in an unawaited task means the `ChildConversation` never resolves and
the parent blocks forever. Resolving with an error result is the only outcome
where the tree always makes progress.

**Scope of the rule, precisely.**

- The DURABLE outcome is mode-independent: under `autostart_subagents` `True`
  *or* `False`, the child's bracket closes and its `ChildConversation` resolves.
  A session must not depend on who drove the child.
- The child's OWN handle keeps today's contract: if an application is consuming
  `run.child(cid)` — which is the whole point of `autostart_subagents=False` —
  the failure re-raises there, as it would for any run. Under `True` nobody is
  consuming that handle, so the failure is recorded and nothing raises.
- Nothing here concerns tool failures inside a child. Those were always ordinary
  execution outcomes and are unaffected.
- A child that trips `subagent_hard_max_steps` already lands on this path: the
  step limit closes the turn `ERRORED` without raising, and the child resolves
  with whatever result that produces.

The parent's projection is where a failed child becomes visible to the model;
§4.2 already anticipates that ("cancelled / interrupted / failed / completed").

## 6. Runtime configuration

New `RuntimeConfig` fields, all prefixed `subagents_`:

- `subagents_enabled: bool = False` — default off, so existing sessions and
  tests are unaffected.
- `subagents_max_depth: int = 1` — the only supported value in V0.
- `subagent_soft_max_steps`, `subagent_hard_max_steps` — the step limits applied
  to subagent conversations; when absent they fall back to the main runtime
  config's values.

When subagents are disabled, or the depth limit has been reached for the
conversation being driven, the spawn tool is **not offered to the model**. The
mechanism is in §7 (**Gating**): the registry omits it — it receives
`conversation_id` and reads `session.conversations[conversation_id].depth`
(§3.1) — and the runner verifies, raising if a spec declaring
`is_subagent_spawn` comes back for a conversation at or past the cap.

## 7. Tool implementation — THE CONTRACT

This is the main contract of the feature. The tool is decoupled from the runner
work it triggers, and the tool's **structured output** is what carries the
handshake: declared on the spec as `output_schema`, populated on the result as
`structured_content`.

A new `subagents` plugin provides `get_system_prompt_parts` (telling the model
the capability exists and how to use it) and a tool registry with two tools:

```python
class SubagentSpawn(BaseModel):
    """The shape of a spawn tool's result payload. Declared on the spec via
    `output_schema`, returned on the result via `structured_content`."""

    model_config = ConfigDict(extra="forbid")

    is_subagent_spawn: bool = True
    task_id: str
    prompt: str
    description: str
    process_subagent_result_tool_name: str


class SpawnSubagent(Tool):
    namespace = "contrib.subagents"
    name = "spawn_subagent"
    description = "Spawn a subagent"
    output_schema = SubagentSpawn   # → ToolSpec.output_schema; see **Gating** below

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        task_id: str | None = Field(
            description="Make up a unique id for the task or we'll make it up for you",
            default=None,
        )
        prompt: str = Field(description="The prompt for the subagent")
        description: str = Field(description="The description of the task")

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,          # the SPAWNING conversation — §3.3
        *,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        task_id = args.get("task_id") or random_id()
        prompt = args["prompt"]
        description = args["description"]
        return ExecutionResult(
            content=[TextContent(text=f"Spawned subagent: {description}")],
            structured_content=SubagentSpawn(
                is_subagent_spawn=True,
                task_id=task_id,
                prompt=prompt,
                description=description,
                process_subagent_result_tool_name="create_conversation_result",
            ).model_dump(),
        )


class CreateConversationResult(Tool):
    namespace = "contrib.subagents"
    name = "create_conversation_result"
    description = "Derive the result of a finished subagent conversation"
    is_private = True          # never advertised to the model — §3.6

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        task_id: str = Field(description="Make up a unique id for the task or we'll make it up for you")
        prompt: str = Field(description="The prompt for the subagent")
        description: str = Field(description="The description of the task")
        conversation_id: str

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,          # the PARENT — where this tool runs (§3.3)
        *,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        task_id = args.get("task_id", random_id())
        prompt = args["prompt"]
        description = args["description"]
        # NOTE the two ids. The PARAMETER is the parent, per §3.3. The ARGUMENT
        # is the child being summarized, per the §7 handshake. They are never
        # the same conversation; do not collapse them.
        child_id = args["conversation_id"]
        conversation = session.conversations[child_id]
        nodes = figure_out_nodes_of_conversation()

        if type(nodes[-1]) is AssistantMessage:
            # in the future we can add Image, etc
            parts = [part for part in nodes[-1].parts if type(part) in {TextContent}]
            return ExecutionResult(content=parts)
        else:
            return ExecutionResult(
                content=[
                    TextContent(
                        text=f"""The subagent finished successfully, here's a summary:
{pretty_print_conversation(conversation)}
"""
                    )
                ]
            )  # obviously here it should be just a summary
```

`CreateConversationResult` declares no `output_schema`: what it produces is
model-facing prose, not a machine payload. That is also what keeps it out of the
gate — a tool that declares nothing is not a spawn tool.

**The handshake.** When the LLM requests a subagent it is just a tool call, so
the whole tool execution flow happens naturally — that part is already covered by
the framework and needs nothing new.

1. The model calls `spawn_subagent`. Ordinary birth, decide, dispatch.
2. The execution **completes**. The runner reads the result's
   `structured_content`, sees `is_subagent_spawn: true`, and validates the
   payload. It creates the `ChildConversation`, creates and seeds the child
   conversation with `prompt` — stamping `depth = parent.depth + 1` — and
   schedules its drive.
3. The runner also knows that whenever that `ChildConversation` finishes, it must
   invoke the tool named in the payload's `process_subagent_result_tool_name`,
   passing `task_id` / `prompt` / `description` / `conversation_id`.
4. That tool's `ExecutionResult` becomes `ChildConversation.execution_result`.

**Why this shape.** There is a contract between the outside world (contrib) and
the runner. The runner knows how to *execute* things that are controlled from
outside — by reading the spec's declared schema (which spec can spawn) and the
returned payload (that this call did spawn, and which tool derives the result).
That lets a developer customize their own "spawn subagent" and "process subagent
result" tools without touching the core. The core's responsibility is only to
schedule and run.

**One declaration, read at two moments.** The runner's gate and step 2 of the
handshake both have to recognize a spawn, and it MUST be the same recognition.
Structured output makes that structural rather than a rule to remember: there is
one declaration, `SubagentSpawn`, and two moments at which it is read.

| | what is read | when | what it means |
|---|---|---|---|
| **Gate** | `ToolSpec.output_schema` declares `is_subagent_spawn` | before the model call, from the spec alone | this tool **can** spawn → it is subject to the depth cap |
| **Handshake** | `structured_content["is_subagent_spawn"] is True` | after the execution completes | this call **did** spawn → create the child |

The asymmetry is deliberate and it is what the flag buys: `is_subagent_spawn` is
a `bool`, not a constant. A spawn tool that decides at runtime *not* to spawn — a
rejected task, a failed validation, a request it chose to answer inline —
returns the payload with `is_subagent_spawn=False` and no child is created. It is
still gated, because the gate reads the declaration, not the outcome.

Concretely, the gate's predicate is a schema-shape check —
`"is_subagent_spawn" in spec.output_schema["properties"]` — not a value check;
there is no value until the tool has run.

**Gating — how the spawn tool is withheld.** The **registry** decides. It
receives the `conversation_id` it is answering for (§3.1), so it omits any tool
whose `output_schema` declares `is_subagent_spawn` when `subagents_enabled` is
False or when `session.conversations[conversation_id].depth >=
subagents_max_depth`. The core makes no tool-list policy of its own.

**The runner verifies, and fails loud.** Three violations, all raising:

- **A gate that leaked.** A registry that returns a spawn-declaring spec for a
  conversation at or past the cap has violated the contract, and the runner
  raises rather than quietly filtering the spec out. That keeps
  `subagents_max_depth` a hard guarantee — §5.4 lets the implementation *rely* on
  depth 1 — and it surfaces at `build_tool_list`, before the model call, naming
  the registry that misbehaved. The alternative failure is invisible: the spec is
  offered, the model calls it, the handshake fires, and a grandchild appears with
  nothing in the log explaining why.
- **A spawn that was never declared.** A completed execution whose
  `structured_content` claims `is_subagent_spawn: true` from a spec whose
  `output_schema` never declared the field has gone around the gate entirely —
  the child is spawned, at any depth, and the cap silently does not exist. The
  runner raises instead of spawning. This is the same invisible failure a name
  match would have had, closed on the other side.
- **A spawn from a conversation at or past the cap.** Checked at handshake step
  2, immediately before the child is created, on the spawning conversation's
  `depth`. The first violation catches a spec that was OFFERED wrongly; this one
  catches a spawn that EXECUTED wrongly, and they are not the same event. The
  gate lives in `get_tools`, but `SimpleToolRegistry.prepare()` resolves by name
  out of `tools_by_name`, which holds the spawn tool at every depth — so
  anything that reaches dispatch with that name runs, cap or no cap. A stale
  `ProxyToolRegistry` route was exactly such a path before §3.5 keyed it by
  conversation. That specific hole is closed; the invariant should not depend on
  a single enforcement point when §5.4 lets the implementation *rely* on
  depth 1. This check puts it where the cap actually means something — child
  creation — rather than only where the tool is advertised.

**The runner validates the payload, because nothing else does.** The core never
checks `structured_content` against `output_schema` — a tool that contradicts its
own declaration still records `COMPLETED` and the payload is stored verbatim. So
the runner checks the fields it is about to act on, at the moment it acts on
them: a result flagged `is_subagent_spawn: true` but missing `prompt` or
`process_subagent_result_tool_name` raises. That is a contract violation, not a
subagent with empty fields.

It checks *required keys*, not contrib's model. `SubagentSpawn` is the plugin's
guarantee about the payload it emits; the core cannot import it (core never
imports from contrib) and does not need to. It reads the same five names §5.3
settled as the convention — the declaration is what makes them discoverable
before the call, not what makes them typed for the core.

**Why the declaration is safe on a `ToolSpec`.** Two properties, both
load-bearing:

- **It is definition-scoped.** A spec must be a pure function of the tool
  definition — anything call-scoped mints a stored row per call and defeats
  normalization. `output_schema` is derived from a ClassVar and is exactly that;
  it is part of `spec_id()`'s hash like every other field, so two calls to one
  spawn tool still write one row.
- **`Tool.get_tool_spec()` already stamps it.** The previous version of this
  section noted that contrib's `Tool` would need a new `metadata` ClassVar, or a
  `get_tool_spec()` override, to carry the marker onto the spec. That work is
  done: `output_schema` is a `Tool` ClassVar today and `get_tool_spec()` derives
  the JSON Schema dict from it. This feature needs no change to contrib's `Tool`
  at all.

**Why not a marker in `ToolSpec.metadata`.** `metadata` is documented as
free-form, registry-owned and **never interpreted by the core**. A marker there
would have made the core read one specific key out of the one dict it promises
not to read. `output_schema` is the opposite: a field whose entire purpose is to
declare what a result will contain, which is precisely what the runner needs to
know before and after the call. The convention moves from a dict nobody
documents to a schema the tool publishes.

**A custom spawn tool still works, and still gates.** The promise is unchanged: a
developer ships their own `delegate_work` tool by declaring `is_subagent_spawn`
in its own output schema and returning the payload. It gates correctly, because
the gate reads the declaration and not the name. A name match would not have
survived that — `delegate_work` would spawn correctly through the payload
handshake and never be filtered, so a subagent would spawn subagents and the
depth cap would quietly not exist, with no error and no warning.

**The payload is free.** `structured_content` never reaches the model and never
counts toward context, so the handshake — prompt, description, task id, the
result tool's name — costs the parent conversation nothing. The model sees only
the `content` status line (§4.2). This is a real improvement over routing the
same facts through `content`, and it is why the spawn tool's model-facing output
can stay one short sentence.

**Documentation impact.** `output_schema` is currently documented as advisory in
both directions and read by *the application* — `ToolSpec`'s docstring,
`docs/agent/03-tools.md` and `docs/agent/02-data-model.md` all say so. It gains a
second reader here: the runner, for the gate. That does not change the field's
meaning, but "the framework never reads it" stops being true and the wording has
to say so. What stays true, and should stay stated: no provider ever sees it, and
the core still never validates a payload against it.

**The result tool is private — settled by §3.6.** Whether the tool named by
`process_subagent_result_tool_name` should appear in `get_tools` at all used to
be open here, with a description begging the model not to call it as the V0
stopgap. `is_private = True` replaces that: the spec IS returned by `get_tools`,
so the runtime resolves and dispatches it normally, and it is simply never put
on the wire. The description no longer has to argue with the model, and §3.6's
second rule means a model that guesses the name gets `NOT_FOUND` rather than an
invocation.

The alternative this section used to float — the subagents plugin omitting its
own result tool from `get_tools`, on nothing more than knowledge of its own
tools — is now rejected outright. It would hide the tool from the runtime that
has to invoke it. `ToolSpec.metadata` stays exactly what it is documented to be
— free-form, registry-owned, never interpreted by the core — and the core now
reads two declarative fields off a spec, `output_schema` and `is_private`, for
the same reason in both cases.

**Step 3 materializes as a `ToolExecution`.** This was the last unresolved
detail here and §3.6 settles it: a private tool records an ordinary execution
entry — approvals, middleware, timeouts, events, usage, all of it. It never
projects as a `ToolMessage` (there is no `tool_call_id` for one), and in V0 the
default projector ignores it altogether, because its result already reaches the
model via `ChildConversation.execution_result` — which is what the parent's
projection renders. The entry is the durable record of the invocation, not a
second channel to the model. A projector that later chooses to surface private
executions some other way (§3.6) is free to; V0 does not.

(One more item here used to be "how does step 2 identify a spawn". It is settled
under **One declaration, read at two moments** above, because the gate and the
handshake are not allowed to disagree about what a spawn is — and now they
cannot, since they read the same declaration. §5.3's acceptance of a *convention*
rather than a core type stands; what changed is that the convention now rides on
a published schema instead of a free-form dict.)

### 7.1 The `contrib.subagents` package — **Settled**

Both tools ship in a new contrib package, `luca/agent/contrib/subagents/`,
delivered as a **plugin**: `SubagentsPlugin`. It is contrib like every other
capability bundle, it consumes only core's public surface, and the core knows
nothing about it.

**Two hooks, no middleware.**

```python
class SubagentsPlugin:
    def get_tool_registry(self, agent_session: AgentSession) -> SimpleToolRegistry:
        return SimpleToolRegistry(
            tools=[SpawnSubagent(), CreateConversationResult()],
            permission_policy=YoloPermissionPolicy(),
        )

    def get_system_prompt_parts(self, agent_session: AgentSession) -> list:
        return [spawning_prompt_part]     # a CALLABLE — see below
```

**Auto-approved, and that is the right call — not a shortcut.** The tools ship in
their own always-allowing registry, which is exactly the pattern `MemoryPlugin`
already ships and documents ("the tools ship in their own auto-allowing registry
— an application that wants them gated composes its own registry over
`get_tools()`'s output"). Spawning is cheap to approve because the dangerous work
is not in the spawn: it is in the tools the CHILD calls, and those are gated
inside the child by their own registries (§9.2). Prompting on the spawn would ask
about the wrapper and never the payload. `CreateConversationResult` is private
and runtime-invoked, so a prompt for it would be meaningless — but it still goes
through `decide()` (§3.6), so it needs a policy that answers ALLOW.

**Coexistence with `shell`, `memory` and the rest is free**, because there is no
global permission gate anywhere: each registry answers `decide()` for its own
tools, and `PluginAgentSessionRunner` makes every plugin registry a child of one
`ProxyToolRegistry`. Shell keeps its `PermissionStrategy`, memory keeps Yolo,
subagents keeps Yolo. Nothing is shared and nothing negotiates.

**Tool names are globally unique — required, and enforced.** `ProxyToolRegistry`
routes by `spec.name`; `namespace = "contrib.subagents"` does NOT disambiguate.
So `spawn_subagent` and `create_conversation_result` must not collide with any
tool from any other installed plugin. The failure is loud rather than silent —
the proxy raises on duplicate names at `get_tools` — so this is a naming rule,
not a risk to design around.

**The system-prompt part is a CALLABLE, and it has to be.** Per §3.1 item 3 a
prompt callable takes `(session, conversation_id)`, and V0's behavior is
deliberately trivial:

- Spawning possible → `"You can spawn multiple subagents"` (final wording is the
  implementation's; the behavior is what is fixed here).
- Spawning not possible → contribute nothing.

"Possible" is **the same predicate the gate uses** (§7, Gating):
`subagents_enabled` is True and
`session.conversations[conversation_id].depth < subagents_max_depth`. That
identity is the whole reason this is a callable rather than a static string: a
static part would tell a subagent at the depth cap that it can spawn subagents
while the tool list withholds the tool, and the model would try. **The prompt and
the tool list must never disagree**, and the cheapest way to guarantee that is to
derive both from one predicate.

One small thing implementation has to settle, flagged so it is not discovered
late: today a prompt callable's return value is coerced by
`coerce_system_prompt_part`, which accepts a `SystemPromptPart`, a `str` or a
dict and **raises `TypeError` on `None`** — so "contribute nothing" is not
currently expressible. An empty string is accepted but assembles into a stray
blank line, because `DefaultSystemPromptAssembler` newline-joins part texts and
only the WHOLE prompt is dropped when blank. Recommendation: let a callable
return `None`, meaning "no part", and skip it during resolution. This is the
first real case for it, so it is not a speculative hook.

## 8. Context, compaction and usage

**Context.** `Entry.context_tokens` for a `ChildConversation` is the size of what
it contributes to its parent — its result, when it has one. A child's own
conversation does not count against its parent's window; that separation is the
main reason subagents are useful.

**Compaction — signatures, and who gets compacted. Settled.** The
`ContextManager` owns compaction, as it does today. Its two compaction methods
take the conversation, closing the §3.1 defect:

```python
def should_compact(self, session: AgentSession, conversation_id: str) -> bool: ...

async def compact(
    self,
    session: AgentSession,
    conversation_id: str,
    nodes: tuple[str, ...],
    entry: CompactionEntry,
) -> CompactionPlan | None: ...
```

`compact` barely needs it — `nodes` already carries the work — but it takes it
anyway. The signature IS the contract, and adding a parameter later breaks every
implementor, while adding it now costs nothing.

The other three `ContextManager` methods do **not** change, and this is a
decision, not an oversight:

- `calculate_context(session, entry)` and `prune_entry(session, entry)` stay as
  they are because `context_tokens` is intrinsic to the entry and shared by
  every conversation referencing it. A `ChildConversation`'s size is its result,
  which is on the entry.
- `process_tool_output(session, execution, result)` stays as it is because §3.2
  puts `conversation_id` on the execution it already receives. A manager that
  wants to truncate subagent output harder than the main conversation's reads it
  from there.

**In V0, `should_compact` is consulted for the MAIN conversation only.** Subagent
conversations are never compaction-checked and never compacted. What makes that
safe is not "children are short" — it is that a child is bounded by
`subagent_hard_max_steps` (§6), so an unchecked child cannot grow without limit.
Lifting this is a later version's work; the requirement on V0 is that lifting it
needs no data-model change, which the signatures above already guarantee.

How a compacting MAIN conversation treats its children is the manager's call: it
may use a `ChildConversation`'s result when present, or ignore child
conversations entirely. Expected to improve later.

**Usage.** Usage records are already conversation-scoped, so a subagent's
consumption is recorded against the subagent's conversation. A session total is
the sum across the catalog. This must be tested explicitly: a deterministic
session driven to spawn subagents, with scripted subagent results and usages,
asserting that every record lands under the right conversation and that the
total is correct.

## 9. Run handles and control

### 9.1 One `AgentRun` per conversation

Driving a subagent produces its own `AgentRun`, exactly like the main
conversation's. `AgentRun` is already a per-drive object with its own
cancellation token, so one handle per conversation is the natural shape rather
than an addition.

```python
run = runner.start(streaming=True, on_event=render)   # the main conversation's handle

run.children                 # dict[conversation_id, AgentRun] — grows as spawns land
run.child(conversation_id)   # -> AgentRun | None

child = run.child(cid)
await child                  # RunResult for THAT subagent
async with child:            # that subagent's event stream
    async for event in child: ...
child.pending_approvals()    # that subagent's gate only
child.cancel()               # cancel only this subagent
run.cancel()                 # cancel the main turn and cascade to every live child
```

`await run` on the main handle joins the whole tree: the main turn cannot close
until every child it spawned has resolved.

Handles are obtained from the parent run, never from an event — a live handle is
not serializable, and events are snapshots meant to be forwarded.

**The main handle's stream is the tree's stream.** Every event it yields —
subagents' included — carries a `conversation_id`, so the ordinary consumption
pattern keeps working unchanged and a consumer routes by that field. A child
handle yields only its own conversation's events; consume one or the other for a
given conversation, never both.

Handles are therefore a **control** surface, not a display one: cancelling one
subagent, joining one, querying one's approvals. Attribution is carried by the
events, not by which handle you hold.

### 9.2 Approvals

Approvals are scoped to the conversation that raised them.

- A gate in a **subagent** parks only that subagent. Its siblings keep running
  uninterrupted — two subagents whose tool calls are auto-approved carry on
  while a third waits. The main conversation is transitively blocked, because it
  cannot resume until every child resolves, but nothing else stops.
- A gate in the **main conversation** is the only full block: the main turn
  cannot proceed.

Consequently `ApprovalRequired` is **not terminal on the main handle's stream**:
a subagent gating does not end the run while a sibling is still working. It stays
terminal on that subagent's own handle. The degenerate case is unchanged — the
main conversation gating with nothing else running means nothing can advance, and
the run returns exactly as today.

`pending_approvals()` is **subtree-scoped**: asking a conversation returns every
gated execution beneath it. That is how subagent C's request reaches the main
conversation's caller while the parent is still `BUSY` and its siblings are
working. Asking a child handle returns that subagent's subtree only. Resolution
stays out-of-band on the registry/permission strategy, as it is today.

The element type does **not** change: it stays `list[ToolExecution]`. A flat
list spanning several conversations is still attributable because §3.2 puts
`conversation_id` on the execution itself — which is what lets an interactive
app say "subagent B is asking" without a wrapper type. The same holds for
`RunResult.pending_approvals`, `ApprovalRequired.executions`, and the elements
of the `run.approvals` stream.

### 9.3 Cancellation

Cancellation becomes conversation-scoped with a cascade, replacing today's
session-scoped meaning.

- `child.cancel()` cancels that subagent only. The main conversation proceeds:
  the `ChildConversation` still resolves, with a result that reflects the
  cancellation.
- `run.cancel()` on the main handle cancels the main turn and cascades to every
  live child. The cascade must tolerate a child that is already cancelling, so
  that cancelling one subagent and then the whole tree is not an error.

### 9.4 Who drives the subagents — `autostart_subagents`

A run takes `autostart_subagents: bool = True`. Both behaviors are supported;
the default is the one almost every application wants.

**`True` — the framework drives them.** Every spawned subagent begins
immediately and advances on its own, whatever mode the parent is in. This is why
the pattern in 9.1 needs no extra code, and why `await run` or a bare `on_event`
consumer sees the whole tree without doing anything.

It has one consequence for a lazy parent: iteration stops gating all work. The
consumer's pull rate controls the main conversation's drive and the delivery of
events, but subagents progress regardless. Suspension still cascades — leaving a
lazy parent's context manager parks every child at its next step boundary.

**`False` — the application drives them.** Each subagent's handle is an ordinary
lazy `AgentRun` that has not started. The application consumes it however it
likes: one task each for parallelism, sequentially, or selectively.

One assistant message can spawn several subagents at once, so the spawn is
announced as ONE batch signal rather than one per child — otherwise the
application would be forced to drive them sequentially.

```python
async with runner.run(autostart_subagents=False) as run:
    async for event in run:
        match event:
            case SubagentsSpawned(conversation_ids=cids):
                children = [run.child(cid) for cid in cids]      # nothing running yet
                await asyncio.gather(*(drive(c) for c in children))
            case _:
                render(event)
```

This restores "iteration is the engine" for the whole tree, and it hands the
application a real obligation: the main turn is blocked on its children, so a
subagent that is never driven blocks it forever. Every spawn must be driven or
cancelled — `run.child(cid).cancel()` resolves the child and lets the main turn
continue. The `await gather` is also not optional politeness: the parent's next
pull cannot advance until the children resolve, so skipping it deadlocks.

Ownership of a subagent's lifecycle follows the flag; the handle is access
either way.

### 9.5 Handles are a view

Handles are runtime objects; the durable graph is the truth. After a reload, the
main conversation's open turn and its unresolved `ChildConversation` entries
describe exactly what is outstanding, and the next run re-mints the child
handles from them.

### 9.6 Conversation status

Four values, replacing today's five. `RUNNING`, `PENDING` and
`AWAITING_APPROVAL` all disappear.

These are the `status` field of `ConversationRuntimeStatus`, and they are
computed by `AgentSession.get_conversation_status()` — never stored on the
conversation (§3.1). That method is also where the subtree term this section
depends on has to live: a parent is `BUSY` while its children work, and a
`Conversation` on its own cannot see its children's entries.

The status answers exactly one question: **what will the next `run()` do?**

| | the next `run()` | post a message? |
|---|---|---|
| `IDLE` | nothing — there is no work | yes |
| `BUSY` | work — the run can still be exhausted | no |
| `BLOCKED` | stop again immediately; you must act first | no |
| `CANCELLING` | flush the turn, not answer it | no |

Derivation, computable from the entries alone:

- trailing `TurnFinish`, **whatever the outcome** → `IDLE`
- trailing `UserMessage`, or other queued work → `BUSY`
- open turn with something runnable → `BUSY`
- open turn with nothing runnable → `BLOCKED`
- open turn with an unconsumed cancel → `CANCELLING`

**The status says nothing about approvals.** This is the important part, and it
is what `AWAITING_APPROVAL` could never express once subagents exist.

> `BUSY` means this run can still be exhausted — if you used `start()`, it will
> keep going and you can watch the events if you want. `BLOCKED` means the run
> cannot be advanced or exhausted; you MUST take action.
>
> But neither says anything about approvals, because an approval can belong to a
> subagent. Say a main conversation has three subagents, A, B and C. A and B are
> running normally; C has a pending approval. The conversation's status is
> `BUSY` — A and B can keep running — while `pending_approvals()` already
> returns C's gated `ToolExecution`, tagged with C's `conversation_id`.
>
> If A and B finish and C's approval still has not been answered, then and only
> then does the parent switch to `BLOCKED`: there is nothing else I can do here,
> nothing else I can run.
>
> And if the subagents had been spawned with `background=True`, the parent would
> switch to `IDLE` instead — its own turn is closed and it will take input.

(That last case depends on the rule in §5.5/§10: blocking children count toward
the parent's derivation, background children do not. Background is not in V0.)

Two consequences worth stating:

**Postable ⇔ `IDLE`.** Because a closed turn derives `IDLE` whatever its
outcome, the status alone answers whether a message can be posted — no separate
bracket check. `post_message` is legal when the MAIN conversation is `IDLE`, and
in no other state. Two of today's behaviors are removed by that, both
deliberately:

- **A failed turn no longer auto-retries.** `run()` on an `IDLE` conversation
  has nothing to do, so recovering from a failure means posting a new message
  rather than re-driving the same context. A bare retry silently re-sends the
  identical request, which is worth losing.
- **Message queueing is gone. Settled.** Today `post_message` also accepts
  `PENDING`, so a second user message can be appended behind an already-queued
  one — documented as "consecutive user messages are an established shape" and
  covered by a test. Under this model a trailing `UserMessage` derives `BUSY`,
  and `BUSY` does not accept input, so the second post raises. The docstring and
  the test go with it. Nothing shipping relies on it (the TUI already gates its
  input on `idle()`), and "let the user type while the agent is working" is an
  application-level input buffer that posts on the next `IDLE` — not a fact the
  session needs to represent. Keeping it would require `IDLE` to distinguish
  "nothing queued" from "a message is queued", which is precisely the fifth
  state §9.6 exists to remove.

**`post_message` and `schedule_compaction` are MAIN-conversation only.
Settled.** Neither takes a conversation and neither is ever valid against a
child. `post_message` follows §4.1 — a child conversation never receives user
messages, so its seed prompt is written by the spawn handshake and nothing else
ever appends one. `schedule_compaction` follows §8 — subagent conversations are
not compaction-checked in V0, so there is nothing to arm. Both therefore read
and write the main conversation's status, and since a parent is `BUSY` while its
children work, it falls out that neither is legal while subagents are running.

**`BUSY` is still true after a crash.** The old `RUNNING` was definitionally a
lie on disk — the process died, nothing was driving — which is why it needed a
self-healing rule. "Can be advanced" survives the crash intact: the next `run()`
recovers the orphaned execution and continues.

### 9.7 The consumer loop

**Approvals reach the application through two doors — one per shape, never two
ways to do the same thing.**

- **`runner.pending_approvals()`** — the durable list. Read from the entries, so
  it works with no run in existence (a cold load), and subtree-scoped, so the
  main runner returns every subagent's gate too.
- **`run.approvals`** — the stream. It yields `ToolExecution`s — the same
  objects `pending_approvals()` returns, attributed by
  `execution.conversation_id` (§3.2); there is no request wrapper. It lives on
  the `AgentRun`, **never on the runner**, and it closes when the run ends. The
  top-level run notifies its own gates plus every subagent's;
  `run.child(cid).approvals` narrows to that subtree. It exists for one
  situation: a gate raised *during* a drive, when a subagent stalls while its
  siblings keep working, so the run does not return. That is the only moment the
  main loop cannot reach on its own. (Still under exploration — see §10.2.)

If you want the hard answer you ask the runner; if you want a stream you use a
run. Starting and stopping the watcher per run is the point — an explicit,
unambiguous lifetime rather than a session-long object with two entry points.

**Ordering rule: the drive comes before the prompt.** `strategy.apply_answer`
writes to the permission strategy, not to the execution, so `approval_status`
stays `PENDING` until a drive re-asks `decide()` — the engine's single decide
call site. Status derivation is therefore fresh with respect to the entry log and
stale with respect to the strategy.

Closing that gap by having the status ask the strategy would turn a pure sync
read into an async call into application code — a TUI repainting a status bar
would poll the permission strategy every frame — and would add a second
`decide()` call site whose answer nothing records. Ordering gets the same result
for free: drive first, prompt second. "Still `BLOCKED` after a drive" is a
genuinely unanswered gate; prompting before the drive re-asks for whatever the
watcher just answered.

That is the *between-drives* half of the contract. The mid-drive half — a gate
answered while the run is still going, which is the only reason the stream
exists — needs `run.notify()`; see §9.8.

**`run()` with `autostart_subagents=True` (the default).** Subagent events arrive
on the same stream, tagged by `conversation_id`; nothing extra to do.

```python
async def watch_approvals(run, strategy):
    async for execution in run.approvals:    # gates raised DURING this drive
        # execution.conversation_id says WHICH conversation is asking (§3.2)
        strategy.apply_answer(execution, await ui.ask(execution))
        run.notify(execution)                # "look again" — see §9.8
    # returns on its own when the run ends and the stream closes


while True:
    if runner.idle():
        prompt = input("> ").strip()
        if not prompt:
            continue
        if prompt in {"q", "quit"}:
            return
        runner.post_message(prompt)
        continue

    run = runner.run(autostart_subagents=True)
    watcher = asyncio.create_task(watch_approvals(run, strategy))
    try:
        async with run:
            async for event in run:
                render(event)
    finally:
        watcher.cancel()          # no-op if the closing stream already ended it
        await watcher

    if runner.blocked():          # still stuck AFTER the drive → genuinely unanswered
        for execution in runner.pending_approvals():
            strategy.apply_answer(execution, await ui.ask(execution))
```

A cold load lands on this naturally: a session that reloads `BLOCKED` is not
`IDLE`, so it drives; the drive re-asks `decide()`, re-publishes the gate to the
stream, and the watcher handles it; the next pass advances. The `blocked()`
branch is the fallback for an application with no watcher at all.

**`run()` with `autostart_subagents=False`.** Same loop; the drive also drives
the children, and must exhaust them before its next pull can advance the main
turn.

```python
async def drive(child):
    async with child:
        async for event in child:
            render(event)

# … same watcher, same outer loop, drive replaced:

    run = runner.run(autostart_subagents=False)
    async with run:
        async for event in run:
            match event:
                case SubagentsSpawned(conversation_ids=cids):
                    children = [run.child(cid) for cid in cids]
                    await asyncio.gather(*(drive(c) for c in children))
                case _:
                    render(event)
```

`gather` is what makes them parallel; a sequential `for c in children: await
drive(c)` runs them one at a time. The loop shape decides the concurrency, not
the runner. In this mode `BLOCKED` can also mean "you spawned subagents and never
drove them", not only "a gate is unanswered".

**`start()`**, which implies `autostart_subagents=True`, is the first loop with
one line changed:

```python
    run = runner.start()
```

Three things hold across all three:

- **The watcher's lifetime is the run's.** It is created with the run and ends by
  itself when the run finishes and the stream closes. The `cancel()` in the
  `finally` is for the path where your own event loop raised or broke out — a
  pending task left behind is a `filterwarnings = ["error"]` failure, not a log
  line.
- **The drive is unconditional after the `IDLE` branch.** It is legal for `BUSY`,
  `BLOCKED` and `CANCELLING`; only `IDLE` has nothing to do, and that branch
  always posts a message or exits.
- **Nothing waits on a signal.** The stream covers the mid-drive case and the
  durable list covers everything else, so there is no state the loop can be stuck
  in with no way forward.

### 9.8 `run.notify()` — waking a gated conversation

**Intention.** Answering is out-of-band by design: `strategy.apply_answer` writes
to the permission strategy, `approval_status` stays `PENDING`, and *a drive* is
what re-asks `decide()`. §9.7's ordering rule is what makes that work — drive
first, prompt second — and it holds for as long as **the application is the thing
that calls `run()`**.

A gated subagent under `autostart_subagents=True` is the first place in `luca`
where it isn't. That child's drive parked and returned (§9.2 — `ApprovalRequired`
is terminal on the child's own handle), the framework owns its lifecycle, and the
parent's drive is still open waiting for it. Nobody is left to re-ask:

```
t=0   #C1  spawns #C2 and #C3.  BUSY, waiting on both.
t=1   #C2  dispatches a 90-second tool call.  BUSY.
t=2   #C3  decide() → PENDING.  Its drive parks and RETURNS.  #C3 = BLOCKED.
           run.approvals yields the ToolExecution (conversation_id = #C3).
t=4   app  strategy.apply_answer(...) — the verdict is now in the strategy.
           #C3's approval_status is still PENDING, #C3 has no live drive,
           and decide() is the only thing that reads that verdict.
t=4→91     #C3 sits idle for 87s holding an answered gate.
           The app is still inside `async for event in run`, so it never
           reaches the blocked() branch either.
t=91  #C2  finishes → #C1 has nothing runnable → BLOCKED → the run returns.
t=92  app  blocked() → pending_approvals() returns #C3 AGAIN → the user is
           asked the identical question a second time.
```

Deleting the watcher entirely finishes at the same instant. Answering mid-drive
achieved nothing — which is the one job §9.7 gives the stream, so without a way
to say "look again *now*", `run.approvals` is decoration.

`notify()` is that link, and only that: **no decision travels through it.** The
answer still reaches the runner through `decide()`, the engine's single call
site. It says one thing — *something changed out of band; check again.*

**Shape.**

```python
run.notify(execution)              # sync; returns immediately; legal in any state
run.child(cid).notify(execution)   # subtree-scoped, like every handle surface
```

**It is sync, like `cancel()`, and for the same reason: it is a signal, not
work.** The temptation is to make it async because `decide()` is async and may do
I/O — but that would put the re-decision on the *caller's* task, creating a second
`decide()` call site outside the engine. Three things break there: it escapes the
cancellation race that stops a hung registry call from making `cancel()` a no-op;
it can run concurrently with the engine's own decide step on the same execution,
giving two writers to `approval_status` and the append-only `approval_decisions`
log, both threading through `before_entry_written`; and a DENY is terminal on the
spot, so it must run the outcome middleware pair and emit `ToolExecuted` — and
every event in the framework is yielded by the drive generator, which a watcher
task has no access to. The work goes back onto the drive's task either way. Once
it does, `notify()` performs no I/O, so nothing blocks the caller's loop.

**There is no `runner.notify()`.** Outside a run, the next `run()` already
re-asks every undecided execution, so a notify with no live drive has nothing to
add. Same rule that keeps the stream off the runner (§10.2).

**Internals.** Two pieces. The door is on the handle, the state is on the
long-lived object — the same split `cancel()` already uses.

```python
class AgentSessionRunner:
    self._recheck: set[str] = set()   # conversation ids with an unconsumed "look again";
                                      # runtime-only, never serialized — the same class
                                      # of state as CancellationToken

class AgentRun:
    self._wake = asyncio.Event()      # already exists, for the eager buffer


def notify(self, execution) -> None:
    cid = execution.conversation_id
    self._runner._recheck.add(cid)
    self._runner._ensure_driven(cid)  # live drive → set its _wake
                                      # no live drive → start one
```

The set holds **conversation ids, not execution ids.** Once a drive is woken its
decide step already re-asks every undecided execution in the open turn, and
`decide()` is contractually an idempotent query of the registry's own state, so
gated siblings are re-asked for free. The execution you pass in is an *address* —
it resolves to `execution.conversation_id` (§3.2), which is exactly why that
field exists and why `notify()` needs no conversation argument. Do not build a
per-execution dirty set: that is a second, narrower decide path running alongside
the one the loop already has.

It lives on the **runner**, not the parent handle, because the window it exists
to cover is a *child* drive that has already ended while the parent's run is
still open — the state has to outlive individual child drives. It is never
serialized: on a cold load the next `run()` re-asks everything undecided anyway,
so there is nothing to persist. A stale id left behind after a run is harmless —
waking a conversation with nothing undecided is a no-op.

**Ordering: consume before asking, never after.**

```python
# at the top of the drive loop, immediately before the existing decide step
self._recheck.discard(cid)
undecided = self.ledger.open_turn_undecided_executions()
```

Clear first, then read. A `notify()` that lands *while* `decide()` is in flight
has to survive it and cause another pass; consuming after the fact swallows it
and puts you straight back in the timeline above. This is the same discipline
`AgentRun._next_buffered` already uses — clear the event, re-check, then wait.

**The teardown window.** A `notify()` can land on a drive that is already
returning, where the wake goes to something that will never loop again. That is
why the *set* is the source of truth and the event is only a wake: a drive
re-checks `_recheck` for its own conversation immediately before parking or
returning, and loops again instead of ending if its id is back in it.

**Caveats.**

- **For a parked child it is a restart, not a wake.** That child's drive is gone;
  there is nothing to signal, so `notify()` starts a new one. That is legitimate
  precisely because `autostart_subagents=True` handed the framework that child's
  lifecycle — and it is why `False` needs no equivalent: there a gated child's
  `drive(c)` returns and the application already has control.
- **The approvals stream becomes at-least-once.** Notifying while a gate is still
  unanswered — three requests, one answered — re-parks that conversation and
  re-publishes `ApprovalRequired` for the same execution. `approval_status` stays
  `PENDING` until `decide()` says otherwise, so consumers must dedup. §9.9
  already requires this of the polling shapes; `notify()` extends it to the
  stream.
- **Ids, not object identity.** `ApprovalRequired` and the `run.approvals`
  stream carry deep snapshots, never live ledger references, so
  `notify(execution)` means `execution.id` + `execution.conversation_id` and the
  runner re-reads the live entry.
- **Not a subagent-only door.** If the *main* conversation gates while a child is
  still running, its drive does not return either and the answer is equally
  inert. Same fix, same door.

**Documentation impact.** `docs/agent/05-permissions.md` describes the
out-of-band loop as four steps ending in "you call `run()` again". That stays
true and is now only half the contract. The doc must state both paths and the
rule that separates them: **outside a run, the next `run()` re-asks; inside a
run, `notify()` is what re-asks.** The existing "`decide()` must be an idempotent
query of your own state" rule gets stronger rather than weaker — it is now
re-invoked on a signal the application controls, not only once per `run()`.

### 9.9 The stream is optional — `pending_approvals()` alone is sufficient

A gate becomes durable the moment `decide()` returns PENDING, so
`pending_approvals()` sees it *while a drive is still going*. An application can
therefore poll the list from inside the event loop and never touch the stream at
all. Two shapes work:

```python
# inline — the answer is awaited in the loop
async with run:
    async for event in run:
        render(event)
        for execution in runner.pending_approvals():
            strategy.apply_answer(execution, await ui.ask(execution))
            run.notify(execution)              # §9.8 — mid-drive, so it needs the nudge

# task per execution — the loop keeps pulling events
async with run:
    async for event in run:
        render(event)
        for execution in runner.pending_approvals():
            tasks.add(asyncio.create_task(handle(strategy, execution)))
            # handle() calls apply_answer then run.notify(), same as above
```

Both are legitimate. Three things the application then owns:

- **Dedup.** `apply_answer` writes to the permission strategy, not to the
  execution — `approval_status` stays `PENDING` until that conversation's drive
  re-asks `decide()`. So the same request comes back on the next iteration and
  you must track what you have already asked. The task version turns one gate
  into N modals without it.
- **Task references.** `asyncio.create_task` without holding the result lets the
  task be garbage-collected mid-flight and swallows its exceptions.
- Nothing else. The inline version blocks less than it appears to: with
  `autostart_subagents=True` the subagents are eager background tasks and keep
  working while `ui.ask` is awaited. Only the parent's drive and event delivery
  pause.

**So the stream is syntactic sugar.** It exposes no fact the durable list does
not already expose; what it adds is once-only delivery, no polling, and a
consumer that can live somewhere other than the event loop. That is an
ergonomics argument, and a good one — but it is not a capability argument, and
the design does not depend on it.

That claim survives §9.8 only because `notify()` is a **handle** door. Both
shapes above are polling from inside a live drive, so both hit the same
inertness a subagent's gate does, and both need the same nudge — and both have
`run` in scope. Had the nudge been a method on a streamed request object instead
(the `request.resolved()` sketch §10.2 records and drops), the shapes sanctioned
here would have no way to call it, and the stream would stop being optional.
That is the second reason there is no request wrapper type; §3.2 is the first.

## 10. Under exploration — nothing here is decided

Two ideas raised while designing §9. They are recorded because they change the
shape of the approval surface, not because they have been chosen. Neither is a
prerequisite for V0: the design in §9 stands on its own without them.

### 10.1 A recursive `AWAITING_APPROVAL` — resolved, superseded by §9.6

Recorded for the trail: the first attempt was to make `AWAITING_APPROVAL`
propagate up the conversation tree. That is no longer needed — the status was
removed entirely, and §9.6's four values cover the same ground without recursion,
because an open turn derives `BUSY`/`BLOCKED` regardless of what its children are
doing.

Two things from that discussion survive and are required:

- **`pending_approvals()` is subtree-scoped.** Asking a conversation returns
  every gated execution beneath it, which is how subagent C's request reaches the
  main conversation's caller while the parent is still `BUSY`. Everything you ask
  a conversation is about its subtree.
- **A parent's status depends on its descendants**, so the ancestor chain
  re-derives whenever a descendant transitions. The `BUSY` → `BLOCKED` flip is
  triggered by C's *siblings* finishing — nothing in the parent's own entries
  changes — and the whole caller contract rides on that landing at the right
  instant.

### 10.2 An approvals stream instead of an approval event

§9.7 gives the stream a narrow, specific job: gates raised *during* a drive, when
a subagent stalls while its siblings keep working and the run therefore does not
return. Everything else is covered by the durable `pending_approvals()` list —
and per §9.9 that list covers the mid-drive case too, if the application is
willing to poll and dedup. **The stream is syntactic sugar over a fact already
exposed**; the open questions are whether it earns its place, and whether the
gate should **block** rather than park-and-return:

```python
async def watch_approvals(run, strategy):
    async for execution in run.approvals:                     # suspends until one arrives
        strategy.apply_answer(execution, await ui.ask(execution))
        run.notify(execution)                                 # "ask me again" — §9.8
```

The decision still reaches the runner only through `decide()`, exactly as today —
`notify()` is a notification, an in-process equivalent of re-entering `run()`.

**Two things that WERE open here are now settled.** The stream's element type is
`ToolExecution` (§3.2) — there is no request wrapper object, so an earlier
sketch of a `request.resolved()` method is dropped entirely: the nudge is
`run.notify(execution)`, which is required whether or not the stream survives,
because the polling shapes in §9.9 need it too.

What remains open here is only the two questions this section started with:
whether the stream earns its place at all, and whether a gate should **block**
rather than park-and-return.

The stream lives on the `AgentRun` only, subtree-scoped like every other handle
surface (`run.approvals`, `run.child(cid).approvals`), and closes with the run.
There is deliberately no `runner.approvals`: the runner's door is the durable
`pending_approvals()` list, and having both would be two ways to ask the same
question with different freshness guarantees.

The cost is a failure mode today's design cannot have: if the gate blocks, a
request nobody answers blocks that conversation indefinitely, and an application
that forgets to wire up the consumer deadlocks rather than getting a `RunResult`
back. Bounding that — a timeout on the request, or falling back to
park-and-return when no consumer is registered — is part of what would have to be
decided.

If the stream is adopted, `ApprovalRequired` stops being load-bearing and goes
back to being informational.

## 11. Tests

All subagent tests live in `tests/agent/subagents/`. `subagents_enabled` defaults
to `False`, so existing tests are unaffected and this functionality is exercised
entirely from its own submodule.

## 12. Not in this document

Which events the feature provides, and whether any new middleware hooks are
needed, are deliberately unresolved. They are worth designing only once the data
model, behavior and control surface here are validated.

Two event-shaped facts are fixed here anyway, because §9 depends on them: every
event carries a `conversation_id`, and a spawn is announced on the stream so an
application running with `autostart_subagents=False` knows a subagent exists. How
the rest of the vocabulary looks is still open.

The event-level `conversation_id` is NOT made redundant by §3.2 and does not
replace it. The three tool-lifecycle events end up carrying the conversation
twice — once on the event, once inside the `ToolExecution` snapshot — and that
is fine and required: the event-level field also has to serve text, reasoning
and finish events, which carry no entry at all, while the field on the execution
has to survive being handed to `decide()`, to middleware and to
`pending_approvals()`, where there is no event. The two must always agree.

**Known and deliberately deferred: middleware scoping. Validated as safe to
skip.** §3.2 incidentally fixes the four tool middleware hooks
(`before_permission_check`, `after_permission_decision`, `before_tool_execution`,
`after_tool_execution`), which now read `execution.conversation_id`, and
`before_post_message` is moot once posting is main-only (§9.6). The remaining
per-LLM-call hooks — `build_model_string`, `build_tool_list`, `before_llm_call`,
`after_llm_response` and `before_entry_written` — still receive no conversation,
so per-subagent model routing and prompt injection are not expressible.

Middleware is **not touched by this feature at all**; the whole hook surface gets
its own refactor after subagents is merged and working. That was audited rather
than assumed, and the audit is recorded here so it is not repeated:

- **There are zero middleware implementations in `luca/`.** All ten hook names
  across the package resolve to `middleware.py` (the mixin definition) and four
  call sites (`runner.py`, plus docstring references in `ledger.py`,
  `context_manager.py` and `events.py`). No contrib package implements a hook,
  the only `get_middleware` is `BasePlugin`'s stub returning `[]`, the TUI wires
  none, and `main.py` wires none. Middleware exists only as test doubles.
- **No signature change is forced.** Every hook takes objects — `entry`,
  `execution`, `messages`, `model_string`, `tools`, `parts` — and keeps working
  unchanged. Conversation-blind is a missing capability, never a failure.
- **Existing tests cannot observe it.** `subagents_enabled` defaults to `False`
  (§6) and subagent tests are confined to `tests/agent/subagents/` (§11), so no
  existing middleware test ever sees a second conversation.
- **The one real capability gap is already out of scope.**
  `build_model_string` is what routing subagents to a cheaper model would need,
  and §2 already lists named subagent types with their own models as not-in-V0.

Two caveats that survive the deferral. First, this is safe for the LIBRARY, not
as a general claim: an application that ships middleware gets conversation-blind
hooks with no error, just wrong behavior if it assumed one conversation —
`docs/agent/07-middleware.md` must say so. Second, what is NOT deferred is
middleware's **concurrency** obligation: §3.4 covers every hook whether or not it
ever learns which conversation it is serving. The two are independent — a hook
can be conversation-blind and still must not keep per-call state on `self`.

One doc task falls out regardless: `before_entry_written`'s docstring enumerates
the entry types it fires for, and it gains `ChildConversation` — a third mutable
entry type whose later update fires the hook a second time (§3).
