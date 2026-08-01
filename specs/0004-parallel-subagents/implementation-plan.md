# Parallel Subagents — Implementation Plan

Companion to `prd_draft.md`. The PRD owns **what** and **why**; this owns
**how** and **in what order**. Where the PRD settles something, this plan
implements it without re-arguing. Where the PRD left an implementation detail
open, this plan decides it and says why — those are marked **[D-n]** and are
the only things here that are up for discussion.

Written against the code as of `4fea1c5`. Every line reference is to that
state.

---

## 0. Shape of the work

### 0.1 The ordering problem

Three changes are load-bearing and every other change depends on at least one
of them:

| | change | depends on |
|---|---|---|
| **S** | strategy seams gain `conversation_id` | nothing |
| **C** | the session container becomes a catalog; status is derived | nothing |
| **P** | the runner drives N conversations at once | S, C |

S and C are independent of each other. S is purely mechanical and touches the
widest area (every registry, every tool, every context manager, ~180 test call
sites); C is narrow but deep (every session literal, every status assertion,
the whole ledger). Doing them together produces one unreviewable diff in which
a mechanical rename and a semantic change to status derivation are
indistinguishable. **They are separate stages, S first**, because S makes every
seam able to *say* which conversation it means while there is still only one —
so when C makes several conversations real, nothing has to be re-plumbed.

Everything else (private tools, `ChildConversation`, the spawn handshake, the
contrib package, the TUI, the docs) sits downstream.

### 0.2 Stages and their green checkpoints

`uv run py.test tests/` is green at the end of every stage. It is **not** green
in the middle of one — each of A, B, D is a single atomic breaking change and
there is no useful intermediate commit.

| Stage | Title | Green at end | Rough size |
|---|---|---|---|
| **A** | Conversation as an argument | ✅ **done** — 1404 pass | ~40 files, ~250 sites |
| **B** | The container, and status becomes derived | ✅ **done** — 1404 pass | ~35 files, ~700 sites |
| **C** | Private tools, `ChildConversation`, subagent config | ✅ **done** | ~15 files |
| **D** | The concurrent drive engine (no subagents yet) | ✅ **done** | runner + tests |
| **E** | The spawn handshake and subagent scheduling | ✅ **done** | runner + new tests |
| **F** | `contrib/subagents/` | ✅ **done** | new package + new tests |
| **G** | TUI and demo | ✅ **done** | `contrib/tui/`, `main.py` |
| **H** | Docs and `AGENTS*.md` | ✅ **done** — 1514 pass | `docs/`, 2 root files |

Three gaps surfaced after the stages and were closed on top of them, each with
its own test: a subagent parked at a gate had **no way back** (a run handle is
single-use and the parent's drive returns while its only child is BLOCKED), a
cancelled subagent with no drive left had **nobody to consume its cancel**, and
`cancel()` on a spawned-but-never-driven child was a **silent no-op** that hung
its parent. See `AGENTS.agent.md`'s Subagents section for the resulting rules.

`AGENTS.agent.md` is corrected **at the end of each stage** for the parts that
stage changed — it is what the next session reads to work in this repo, and a
stale one is worse than a stale `docs/` page. Full `docs/` rewrite is Stage H.

### 0.3 What does not change, verified

- **`luca/client/` and `tests/client/`: zero edits.** `rg "luca\.agent|AgentSession|Conversation" luca/client/ tests/client/` → no hits. The only conversation-shaped field client-side is the inert `ChatCompletionRequest.session_id` (`luca/client/types/completion.py:87`), which no transport reads and the agent never sets. The dependency is strictly one-way.
- **`luca/agent/core/context.py`** (`CancellationToken`) — untouched. It is already per-run and conversation-blind by design.
- **`luca/agent/core/middleware.py`** — untouched. §12 of the PRD audits this and defers the whole hook surface; the four tool hooks gain attribution for free through `ToolExecution.conversation_id`. Only two docstrings change (`before_entry_written`'s entry-type list, and a concurrency note).
- **`luca/agent/core/adapter.py`** — one docstring line. `tool_spec_to_luca_tool` already drops every non-wire field, so `is_private` needs no code there (verified: `luca/agent/core/adapter.py:47-64`).
- **The 7 shell tool test files** (2524 lines, 131 tests) contain no `session` / `AgentSession` / `Conversation` reference at all. They are absorbed entirely by two lines in `tests/agent/contrib/shell/conftest.py` (`:35`, `:48`).

---

## Stage A — Conversation as an argument

**Goal.** Every strategy seam can name the conversation it is answering for.
No behavior changes; the runner passes `session.active_conversation.id`
everywhere. The container is untouched.

### A.1 Core contracts

`luca/agent/core/tool_registry.py` — all four methods:

```python
async def get_tools(self, session, conversation_id: str) -> list[ToolSpec]
async def create_execution(self, session, conversation_id: str, call: ToolCall) -> ToolExecution
async def decide(self, session, conversation_id: str, tool_execution: ToolExecution) -> ApprovalDecision
async def prepare(self, session, conversation_id: str, tool_execution: ToolExecution) -> PreparedTool
```

Position is fixed: `session` first (the existing rule), `conversation_id`
second, the call/execution last. Module docstring gains a **rule 13** —
concurrency, transcribed from PRD §3.4 (no per-call state on `self`; state
keyed by `conversation_id` needs no lock; deliberately-shared state needs an
`asyncio.Lock` around the mutation and never around the I/O; `asyncio.to_thread`
is real parallelism; a cancelled body can leave data half-written; process-global
resources are scoped per conversation, not mutexed).

`luca/agent/core/context_manager.py`:

```python
def should_compact(self, session, conversation_id: str) -> bool
async def compact(self, session, conversation_id: str, nodes, entry) -> CompactionPlan | None
```

`calculate_context` / `prune_entry` / `process_tool_output` are unchanged —
PRD §8 settles this, and the reason goes in the docstring.

`luca/agent/core/system_prompt.py`:

```python
SystemPromptPartInput = SystemPromptPartLike | Callable[[AgentSession, str], SystemPromptPartLike | None]

def coerce_system_prompt_part(value) -> SystemPromptPart | None   # None → no part
```

`coerce_system_prompt_part(None)` returns `None` instead of raising `TypeError`
(PRD §7.1's flagged item — this is its first real consumer). `build_system_message`
drops `None` parts during resolution.

> **[D-1] `ConversationProjector.project` takes `nodes`, not a `Conversation`.**
> ```python
> def project(self, nodes: Sequence[str], entries: Mapping[str, AnyEntry]) -> list[Message]
> ```
> PRD §3.1 item 4 forbids a `Conversation` in a public contract, and `project`
> is the only place one is passed. `(session, conversation_id)` would satisfy
> the letter but breaks `SummarizingContextManager.summarize`, which projects a
> synthetic `Conversation(id="_compaction_head", nodes=folded, …)`
> (`simple_context_manager/manager.py:150-151`) — that trick has no session to
> register against. Taking the path directly satisfies the rule, keeps the
> trick as `project(folded, session.entries)`, and is a smaller change than
> either alternative. `_skip_compaction_bracket` already takes `nodes`.

### A.2 Contrib tool-authoring surfaces (PRD §3.3)

All five, `conversation_id` after `session`:

```python
Tool.execute(args, session, conversation_id, *, cancellation_token) -> ExecutionResult
Tool._execute(args, session, conversation_id, *, cancellation_token) -> str
tool() / tool_class()          # the wrapped `execute=` callable becomes (args, session, conversation_id)
get_approval_context(args, session, conversation_id) -> dict
ResourcePermissionToolMixin.build_permission_requests(args, session, conversation_id)
```

`PreparedTool` is unchanged — the conversation reaches the body through the
closure `SimpleToolRegistry.prepare` already builds (PRD §3.3).

> **[D-2] `PermissionPolicy.decide` keeps `(session, tool_execution)`.**
> It is not in §3.3's list of five, and PRD §3.2 names `ToolRegistry.decide()`
> as the first surface that "gains attribution for free" from
> `execution.conversation_id`. Verified: `PermissionStrategy` never touches the
> session at all — `session` appears only in the signature
> (`resource_permissions/strategy.py:189-197`), so it is not an instance of the
> §3.1 defect (nothing is read "behind the caller's back"). Keeping it saves 47
> of 51 test call sites and three signatures. The docstring gains one line
> pointing at `tool_execution.conversation_id`. **Overruling this is one
> forwarded argument in `SimpleToolRegistry.decide` plus 3 signatures + 51 test
> sites — say so and I will.**

### A.3 Contrib state keyed by conversation (PRD §3.5)

| what | file | change |
|---|---|---|
| `ProxyToolRegistry._route` / `_warmed` | `simple_tool_registry/registry.py:213-214` | `dict[str, dict[str, ToolRegistry]]` / `set[str]`, keyed by conversation id. `_resolve(session, conversation_id, name)`; `create_execution` reads the per-conversation route. |
| `FileReadTracker` | `shell/plugin.py:79`, `shell/tools.py` | One tracker per conversation. `ShellAccessPlugin` holds `dict[str, FileReadTracker]`; read/edit/write resolve `self._tracker_for(conversation_id)`. |
| `MemoryPlugin.scratchpad_store` / `todo_store` | `memory/plugin.py:165-166` | `dict[conversation_id, dict]`; each tool's body indexes by the id it now receives. |
| `calculate_context_used` | `simple_context_manager/manager.py:63` | `calculate_context_used(session, conversation_id)`. |

`SimpleToolRegistry` needs only its four signatures (already
concurrency-clean); `PermissionStrategy.add_rule` stays global (§3.5, accepted
for V0).

### A.4 Runner (mechanical)

Every runner call into a seam gains `self.session.active_conversation.id`.
The runner's own conversation threading is **Stage B** — here it is a single
expression at ~12 call sites. `build_system_message` resolves callables with
`(self.session, self.session.active_conversation.id)` instead of
`(session_config, runtime_status)`.

### A.5 Tests in Stage A

| file | sites | note |
|---|---|---|
| `tests/agent/scenarios.py` | `FakeToolRegistry` ×4, `FakeContextManager` ×2, `FakeTool.execute/_execute` ×2 + 6 doubles | the pivot — nothing else compiles until it lands |
| `tests/agent/test_runner_system_prompt.py` | callable-part args | behavior change: callables now get `(session, conversation_id)` |
| `tests/agent/test_context_manager.py` | `should_compact` / `compact` | |
| `tests/agent/test_projection.py` | `project(conversation, …)` → `project(nodes, …)` | ~15 sites |
| `tests/agent/contrib/shell/conftest.py:35,48` | 2 lines | absorbs 126 call sites across 8 files |
| `tests/agent/contrib/test_memory.py:50` | `run_kwargs()` | absorbs 16 sites |
| `tests/agent/contrib/test_simple_tool_registry.py` | 29 registry call sites + `RecordingPolicy` | |
| `tests/agent/contrib/test_tools.py` | 21 factory calls + 8 bodies | |
| `tests/agent/contrib/test_resource_permissions.py` | 4 `build_permission_requests` + 10 `get_approval_context` | `.decide()` ×40 untouched under **[D-2]** |
| `tests/agent/contrib/test_plugins.py:106` | `RemoteToolRegistry.get_tools` | the only test-defined `ToolRegistry` |

**Stage A exit:** `uv run py.test tests/` green, `uv run ruff check --fix && uv run ruff format` clean.

---

## Stage B — The container, and status becomes derived

**Goal.** `AgentSession` holds a catalog of conversations; status is never
stored; the runner drives one conversation *named by an id*. Still one
conversation at a time.

### B.1 `models.py`

```python
class Conversation(BaseModel):
    id: str
    nodes: list[str] = Field(default_factory=list)
    created_at: int
    updated_at: int
    previous_conversation_id: str | None = None
    depth: int = 0
    # NO status


class ConversationStatus(str, Enum):
    IDLE = "idle"        # nothing to do; the only postable state
    BUSY = "busy"        # the run can still be exhausted
    BLOCKED = "blocked"  # you must act first
    CANCELLING = "cancelling"


class ConversationRuntimeStatus(BaseModel):   # was SessionRuntimeStatus
    status: ConversationStatus = ConversationStatus.IDLE
    turn_count: int = 0
    step_count: int = 0


class AgentSession(BaseModel):
    id: str
    entries: dict[str, AnyEntry]
    tool_specs: dict[str, ToolSpec]
    usages: dict[str, dict[str, Usage]]
    conversations: dict[str, Conversation]
    main_conversation_id: str
    session_config: SessionConfig

    def get_conversation_status(self, conversation_id: str) -> ConversationRuntimeStatus: ...
```

Deleted: `active_conversation`, `conversation_history`, `Conversation.status`,
`AgentSession.status` (zero call sites — the only reference anywhere is
`docs/agent/02-data-model.md:516`), `session_runtime_status`,
`SessionRuntimeStatus.get_runtime_status_from_agent_session`.

A new `@model_validator(mode="after")` refuses a session whose
`main_conversation_id` is absent from `conversations` — the same class of load
guard `_restore_tool_specs` already is.

> **[D-3] The pure path derivations move into `models.py`.**
> `get_conversation_status` lives on the session (PRD §3.1 item 1 — it has to,
> because §9.6's derivation is subtree-aware). It needs the same entry-walking
> the ledger's read methods do (`open_turn_index`, the four execution subsets,
> the trailing-compaction skip). `models.py` cannot import `ledger.py` (ledger
> imports models), and a third module cannot host them because they need
> `isinstance` against the entry classes. So the walks become **module-level
> free functions in `models.py`**, taking `(nodes, entries)` — exactly the shape
> `is_compaction_bracket` already has, and for exactly the same stated reason
> ("one definition, four consumers … upstream of all of them"). `SessionLedger`'s
> read methods become one-line delegations bound to a conversation id. One
> implementation, two doors.

Derivation, per PRD §9.6, extended for the subtree term that Stage E will
populate (in Stage B `unresolved_children` is always empty):

```
get_conversation_status(cid):
    open turn?
        unconsumed CancelRequested            → CANCELLING
        runnable(cid)                         → BUSY
        otherwise                             → BLOCKED
    no open turn (trailing closed compaction brackets skipped):
        empty path                            → IDLE
        leaf is a UserMessage                 → BUSY      (queued work)
        otherwise (any TurnFinish outcome)    → IDLE

runnable(cid):
    any RUNNING execution in the open turn                       → True  (orphan recovery)
    any PENDING execution with approval_status in (None, ALLOWED) → True
    unresolved ChildConversations in the open turn:
        → any child whose status is BUSY / IDLE / CANCELLING     (IDLE = resolvable now)
    no nonterminal execution and no unresolved child             → True  (the model can be called)
    otherwise (only gated executions remain)                     → False
```

Two behavior changes fall out and are deliberate (PRD §9.6):
a closed `TurnFinish(TIMED_OUT|ERRORED)` derives **IDLE**, not retry-ready
PENDING — a failed turn no longer auto-retries; and a trailing `UserMessage`
derives **BUSY**, so message queueing is gone.

### B.2 `ledger.py`

Every method takes `conversation_id: str` first:

```python
append(conversation_id, build)          put_entry(conversation_id, entry)
refresh_entry(entry)                    record_usage(conversation_id, entry_id, **counters)
prune(conversation_id, original_id, build)
transition_conversation(conversation_id, *, updates, created, closing, nodes, ts)
open_turn_index(conversation_id)        …and the six other reads
```

`refresh_entry` keeps no conversation — it is a store-only write
(`recalculate_context_tokens` spans every conversation).

`transition_conversation` rewrite:
- resolve `outgoing = session.conversations[conversation_id]`;
- preconditions unchanged (the fallible block stays before the commit point);
- **delete** both status writes (`ledger.py:259`, `:271`) and their comments — there is nothing to freeze;
- **delete** the `conversation_history.append` — the archived row stays in `conversations`;
- install `Conversation(id=new_id, nodes=…, created_at=ts, updated_at=ts, previous_conversation_id=outgoing.id, depth=outgoing.depth)` into `conversations`;
- **re-point the namer**: if `conversation_id == session.main_conversation_id`, update it; else find the single `ChildConversation` entry whose `conversation_id` names it and update that; else raise `AgentError`. (V0 never compacts a child — PRD §8 — but a silently orphaned successor is the worst failure mode here, so the door refuses rather than assumes.)

`_store_tool_spec` and the four tool-spec write doors are unchanged.

### B.3 `compaction.py`

```python
def check_snapshot(*, session, conversation_id: str, snapshot) -> None
def validate_plan(plan, *, entry_id, session, conversation_id: str, snapshot) -> None
```

`check_snapshot` compares against `session.conversations[conversation_id]`
instead of `active_conversation` (`compaction.py:116`). The runner re-resolves
`conversation_id` from `session.main_conversation_id` before the check, so the
original "did the active conversation move under the plan?" semantics survive
a transition.

### B.4 `events.py`

Every one of the 15 event classes gains `conversation_id: str` (required — PRD
§12: "every event carries a `conversation_id`"). Introduce a shared
`AgentEventBase(BaseModel)` carrying the field and `model_config`, so the
declaration lives once.

This is the single largest mechanical cost in the plan: ~200 event literals in
`tests/agent/`. It is unavoidable — a defaulted field would let the runner ship
a wrong attribution silently, which is the exact failure §9.1's routing rule
depends on not happening. Migration approach in B.7.

### B.5 `utils.py`

`pretty_print(session, conversation_id: str | None = None)` — defaults to the
main conversation. The header line (`utils.py:109`) drops
`conversation.status.value` and gains `depth` when non-zero;
`session.get_conversation_status(cid).status.value` is the replacement value
(PRD §3.1: "if the status is wanted in the serialized JSON for a human reading
the file, that is `pretty_print`'s job").

### B.6 `runner.py`

Threading. `_drive(conversation_id, streaming, token)` and **every**
conversation-scoped method gains `conversation_id: str` as its first parameter
— public override points included (`build_tool_list`, `build_messages`,
`build_system_message`, `prepare_llm_call`). `post_message` /
`schedule_compaction` / `pending_approvals` / `cancel` take NO conversation:
they are main-conversation-only by PRD §9.6, and `runner.status` stays a
property answering for the main conversation (a subagent's is
`session.get_conversation_status(id)`, the one door). No private scope object:
PRD §3.1 item 4's rule is "the scope is always the id", and one convention
inside the engine and another at its edges is how the two drift.

`AgentRun` gains `conversation_id` here — the drive needs one, and Stage D
builds `children` on it.

One G2 subtlety surfaced in Stage B: the compaction step must re-resolve
`self.main_conversation_id` when it calls `check_snapshot` / `validate_plan`,
not reuse the id the drive started with. G2 asks *"is the conversation I
compacted still the one that is named?"*, so a manager that installed a
successor under the runner is only caught by a fresh read.

35 `self.ledger.*` call sites and 6 direct `active_conversation` touches
(`runner.py:562, 617, 1158, 1232, 1514, 2391`) resolve to
`session.main_conversation_id` in this stage.

Deleted: `_set_status`, `_refresh_status`, the `active_conversation.status`
write in `__init__` (`:562`), `AgentSessionRunner.status`'s backing field. The
predicates become:

```python
def status(self, conversation_id: str | None = None) -> ConversationStatus   # None → main
def idle()  def busy()  def blocked()  def cancelling()      # main conversation
```

`pending()` / `running()` / `awaiting_approval()` are deleted.
`RunResult.status` is the four-valued enum; `_build_run_result` reads
`get_conversation_status(main).status` and no longer special-cases
AWAITING_APPROVAL — it reports `pending_approvals()` whenever that list is
non-empty.

`post_message` requires **IDLE only** (the closed-bracket check is now
implied — an open bracket derives BUSY/BLOCKED/CANCELLING, and an open
compaction bracket derives BUSY). `_begin_run` raising on IDLE now also means
a failed turn is recovered by posting, not by re-running.

`new_session(...)` builds `conversations={cid: Conversation(id=cid, …,
depth=0)}` and `main_conversation_id=cid`.

### B.7 Tests in Stage B

The three mechanical sweeps, in order:

1. **`tests/agent/scenarios.py`** first — 8 session literals, ~30
   `active_conversation` / `ConversationStatus` sites, the only
   `conversation_history=[…]` literal (`:1067`) and the two-conversation
   `POST_COMPACTION_SESSION` build (`:1188-1251`). Six contrib test files
   import from it, so nothing downstream compiles until it lands. Add a
   `conversation(...)` helper next to `spec()` / `make_session()` so a literal
   reads `conversations={"c1": conversation("c1", ["u1", "ts", …])}` and the
   `depth` / `previous_conversation_id` defaults stay invisible.
2. **The event field.** Add `conversation_id="c1"` (or the file's session
   conversation id) to every event literal. Mostly `test_runner.py` (82),
   `test_runner_approvals.py` (51), `test_runner_tool_output.py` (14),
   `test_runner_cancellation.py` (14). Codemod-able: the constructors are
   unambiguous and each file has one conversation.
3. **Status assertions.** `RUNNING` → `BUSY`, `AWAITING_APPROVAL` → `BLOCKED`,
   `PENDING` → `BUSY` **except** where PENDING meant "closed failed turn,
   retry-ready" — those become **IDLE** and their `run()`-after-failure
   assertions become `post_message()`-after-failure. Contrib is exposed at
   exactly one line (`tests/agent/contrib/tui/test_app.py:201`).

Tests that change *meaning*, not just shape — each needs a reviewed rewrite,
not a rename:

| test | was | becomes |
|---|---|---|
| `test_runner_failures.py` — `post_message` matrix | PENDING accepted, queueing allowed | IDLE only; the consecutive-user-message test is **deleted** (PRD §9.6, "the docstring and the test go with it") |
| `test_runner_failures.py` — retry after failure | `run()` re-answers | `run()` raises "nothing to run"; `post_message()` then drives |
| `test_ledger.py` — `derive_status` skip matrix (147 sites) | five values | four; the closed-failed-bracket row flips to IDLE |
| `test_ledger.py` — `transition_conversation` | asserts `conversation_history` | asserts `conversations` + `previous_conversation_id` + the re-pointed `main_conversation_id` |
| `test_models.py` — `session_runtime_status` (`:1356,1383,1421`) | property | `get_conversation_status(cid)` |
| `tests/agent/contrib/tui/test_app_compaction.py:46` | `conversation_history` predicate inside `wait_until(timeout=8.0)` | a wrong rewrite **hangs 8s instead of failing** — rewrite this one by hand |

**Stage B exit:** green; `pretty_print` snapshots in `test_utils.py` regenerated
and reviewed; ruff clean.

---

## Stage C — Private tools, `ChildConversation`, subagent config

Small, additive, independently reviewable. Nothing produces a
`ChildConversation` yet.

### C.1 `ToolSpec.is_private` (PRD §3.6)

- `ToolSpec.is_private: bool = False` — participates in `spec_id()` like every other field, so every stored spec hash changes. No migration (the PRD's existing rule: regenerate pre-refactor session files).
- `Tool.is_private: ClassVar[bool] = False`; `get_tool_spec()` stamps it; `tool()` / `tool_class()` take `is_private: bool = False`.
- **Wire filter** — exactly one point, `runner.py:1148`: private specs are dropped after `get_tools()` and before `tool_spec_to_luca_tool`.
- **Refusal rule** — a model tool call naming a private spec records `NOT_FOUND` without reaching `create_execution`. The runner needs the private names *offered this iteration*; see **[D-4]**.
- `ToolSpec`'s docstring gains the one exception to "the advertisement sent to the model".

> **[D-4] The tool step returns specs and the wire list; `build_tool_list` takes the specs.**
> ```python
> async def resolve_tool_specs(self, conversation_id) -> list[ToolSpec]   # get_tools + the §7 gate check
> def build_tool_list(self, conversation_id, specs) -> list[LucaTool]     # filter private, adapt, middleware
> ```
> The drive races one coroutine that calls both and keeps `specs` as a **local**
> — never on `self`, which under PRD §3.4 rule 1 is exactly the state that
> silently belongs to another conversation after an await. `build_tool_list`
> stays the public override point and the `build_tool_list` middleware hook
> still sees the post-adapter wire list, unchanged.

### C.2 `ChildConversation` (PRD §3)

```python
class ChildConversation(Entry):
    type: Literal["child_conversation"] = "child_conversation"
    conversation_id: str
    tool_execution_id: str
    execution_result: ExecutionResult | None = None
```

Following `AGENTS.agent.md`'s "Add an entry type" checklist:

| step | change |
|---|---|
| union | added to `AnyEntry` |
| export | `luca/agent/core/__init__.py` |
| projection | `project_child_conversation(entry, entries)` — **one synthetic `ClientUserMessage` per resolved child**, rendering `execution_result.content` inside a `<task id="{tool_execution_id}">…</task>` wrapper. `execution_result is None` **raises `ProjectionError`** — design principle 6, identical to a nonterminal `ToolExecution` (the runtime must never call the model with an unresolved child). Wording lives on the class as `CHILD_TASK_TEMPLATE`. |
| context | `calculate_context` counts `execution_result.content`; `_media_parts` returns it too. 0 until it resolves, so the runner recalculates on resolution — same shape as a `ToolExecution`'s terminal transition. |
| `pretty_print` | a `Subagent · <task_id> · <status>` section with the clipped result under it |
| mutability | **the third mutable entry type.** `SessionLedger.put_entry`'s docstring, `before_entry_written`'s docstring (PRD §12's flagged doc task), and `AGENTS.agent.md`'s "two mutable entry types" all change to three. |

### C.3 `ToolExecution.conversation_id`

`conversation_id: str | None = None` — provenance, never traversal (PRD §3.2).
The runner stamps it in `_create_executions`'s build closure alongside
`id` / `parent_id` / `created_at`, and on every synthesized draft in
`_birth_draft`. A registry must not set it; the runner overwrites
unconditionally.

The 21 contrib + ~120 core `ToolExecution` literals only break where a test
asserts full-object equality against something the runner produced — 7 in
`test_simple_tool_registry.py` plus the runner-test assertions.

### C.4 `RuntimeConfig`

```python
subagents_enabled: bool = False
subagents_max_depth: int = 1
subagent_soft_max_steps: int | None = None   # None → fall back to soft_max_steps
subagent_hard_max_steps: int | None = None   # None → fall back to hard_max_steps
```

The two `int | None` fields get their own validator (the existing
`_inf_or_natural` runs over non-optional ints). The drive resolves the pair
from `session.conversations[cid].depth`: 0 → the main fields, > 0 → the
subagent fields with fallback.

**Stage C exit:** green. New tests: `is_private` on the spec / in `spec_id()` /
off the wire list / `NOT_FOUND` on a model call (`test_models.py`,
`test_runner.py`, `tests/agent/contrib/test_tools.py`); `ChildConversation`
projection + context + `pretty_print` (`test_projection.py`,
`test_context_manager.py`, `test_utils.py`).

---

## Stage D — The concurrent drive engine

**Goal.** The runner can drive N conversations at once and one `AgentRun` can
own children. No subagents exist yet, so every new path is exercised in its
degenerate (single-conversation) form and the observable behavior is unchanged.

### D.1 Runner state

```python
self._runs: dict[str, AgentRun]          # one live run per conversation  (was _active_run)
self._wakes: dict[str, asyncio.Event]    # a live drive's wake, per conversation
self._recheck: set[str]                  # unconsumed "look again"  (PRD §9.8)
```

All three are runtime-only and never serialized — the same class of state as
`CancellationToken`. `_begin_run` / `_end_run` become per conversation, so
"one engine at a time" becomes "one engine per conversation".

```python
def _ensure_driven(self, conversation_id: str) -> None:
    if conversation_id in self._wakes:          # live drive → wake it
        self._wakes[conversation_id].set()
    elif (run := self._runs.get(conversation_id)) is not None and run._framework_owned:
        run._redrive()                          # parked child → restart (PRD §9.8 caveat 1)
    # app-owned or absent → nothing; the recheck flag is enough
```

### D.2 `AgentRun`

```python
AgentRun(runner, *, conversation_id, streaming, on_event, eager,
         autostart_subagents=True, parent=None)

run.conversation_id
run.children -> dict[str, AgentRun]        # grows as spawns land
run.child(cid) -> AgentRun | None          # searches this run's subtree
run.approvals                              # async iterator of ToolExecution
run.notify(execution) -> None              # sync; legal in any state
run.cancel(...)                            # this conversation + cascade
```

**Event fan-in.** Each conversation's drive is its own async generator.
A framework-driven child is an eager `AgentRun` whose `_consume` appends every
event to its own buffer **and** puts it on its root's `_inbox` queue.

- **Lazy root:** `_pump` races `ensure_future(own_gen.__anext__())` against `inbox.get()`, `FIRST_COMPLETED`; the pending own-step task is kept for the next pull. This is what PRD §9.4 means by "the consumer's pull rate controls the main conversation's drive and the delivery of events, but subagents progress regardless".
- **Eager root:** `_consume` drains both into the buffer.
- **Termination:** `StopAsyncIteration` only when the own generator is exhausted **and** every child task is done **and** the inbox is drained. The run's lifetime is the tree's, not the parent drive's — which is what makes PRD §9.2's "`ApprovalRequired` is not terminal on the main handle's stream" true.

> **[D-5] Forwarding follows ownership (confirmed with you).**
> `autostart_subagents=True` → child events forward to the root's stream (and
> the child handle buffers them too — hence §9.1's "consume one or the other,
> never both"). `autostart_subagents=False` → **no forwarding**; the
> application consumes the child handles. Without this, §9.7's `False` loop
> renders every child event twice.

**`run.approvals`.** A per-run `asyncio.Queue[ToolExecution]` plus an async
generator over it, closed by a sentinel when the run finishes. The drive
publishes an execution the moment `decide()` returns PENDING for it, walking
`run → parent → …` so a root's stream carries the whole subtree and
`run.child(cid).approvals` narrows. Element type is `ToolExecution` — no
wrapper (PRD §3.2). At-least-once; consumers dedup (PRD §9.8).

**`run.notify(execution)`.** Sync, no I/O, legal in any state:
```python
def notify(self, execution) -> None:
    cid = execution.conversation_id
    self._runner._recheck.add(cid)
    self._runner._ensure_driven(cid)
```
There is no `runner.notify()` (PRD §9.8).

### D.3 The drive loop

```
_drive(conversation_id, streaming, token):
    compaction step            # only when conversation_id == main_conversation_id (PRD §8)
    _ensure_open_turn(cid)
    _recover_orphans(cid)
    loop:
      0  unconsumed cancel?           → await _wind_down(cid, token); return
      0b self._recheck.discard(cid)   # CLEAR BEFORE READING — PRD §9.8's ordering rule
      1  undecided executions         → gather decide; publish each PENDING to the approvals streams
      2  ready executions             → dispatch batch
      3  spawn handshake              → (Stage E)
      4  resolve finished children    → (Stage E)
      5  did 1-4 do anything?         → continue
      6  gated executions present?    → emit ApprovalRequired (once per park/wait entry)
      7  can the subtree still advance?
             any unresolved child whose status is BUSY / IDLE / CANCELLING,
             or cid back in self._recheck
         → await wake, raced against the token; then continue
      8  otherwise                    → return  (parked / nothing runnable)
      9  step limits (depth-resolved) → resolve_tool_specs + gate check → build_tool_list
         → LLM call → record assistant → create executions → continue
```

Step 7 is the whole asymmetry PRD §9.8's timeline describes, and it is **one
rule**, not two: *a drive returns when nothing in its subtree can advance*. A
gated child (no subtree) returns; a gated parent whose children are still
working waits. The degenerate single-conversation case is byte-identical to
today.

**Teardown window (PRD §9.8).** Steps 7 and 8 re-check `self._recheck` for
`cid` immediately before waiting or returning, and loop instead if it is back.
The set is the source of truth; the event is only a wake.

### D.4 Cancellation

```python
runner.cancel(conversation_id=None, outcome=CANCELLED, error=None)   # None → main
```
Appends `CancelRequested` to that conversation's open turn, trips its live
run's token, then **cascades**: for every live descendant conversation, the
same, swallowing `AlreadyCancellingError` (PRD §9.3 — "the cascade must
tolerate a child that is already cancelling"). `child.cancel()` cancels that
subtree only.

> **[D-6] The wind-down resolves unresolved `ChildConversation` entries. (PRD gap.)**
> `_wind_down` becomes `async` and, before terminalizing PENDING executions:
> awaits the (already-cancelled) live child drives of this conversation's open
> turn to settle, then writes each unresolved `ChildConversation.execution_result`
> = `ExecutionResult(content=[TextContent("[subagent cancelled]")], is_error=True)`
> **without running the result tool**. Same treatment PENDING executions already
> get. The PRD does not say what happens to an unresolved child when its parent
> is cancelled; leaving it unresolved would make the closed turn unprojectable
> (C.2's fail-loud rule) and permanently wedge the conversation. This keeps the
> invariant *every `ChildConversation` on a closed turn is resolved* total.
> It also covers the `autostart_subagents=False` child that was never driven.

**Stage D exit:** green with every existing test unchanged in behavior. New
tests in `tests/agent/test_runner_lifecycle.py` for `children`/`child()`
(empty), `approvals` (own gates, closes with the run), and `notify()`
(re-asks `decide()` mid-drive without a second decide call site).

---

## Stage E — The spawn handshake and subagent scheduling

**Goal.** Steps 3 and 4 of D.3, plus the child lifecycle.

### E.1 The gate (before the model call)

```python
def _declares_spawn(spec: ToolSpec) -> bool:
    schema = spec.output_schema
    return isinstance(schema, dict) and "is_subagent_spawn" in (schema.get("properties") or {})
```

In `resolve_tool_specs`: if any returned spec declares it while
`not subagents_enabled` or `conversations[cid].depth >= subagents_max_depth`,
**raise** — violation 1 of PRD §7, surfacing before the model call and naming
the spec and the conversation.

### E.2 Step 3 — the handshake

For each `COMPLETED` execution in the open turn whose `result.structured_content`
has `is_subagent_spawn is True` and for which no `ChildConversation` in the open
turn has `tool_execution_id == execution.id`:

1. `_declares_spawn(execution.tool_spec)` false → **raise** (violation 2).
2. `conversations[cid].depth >= subagents_max_depth` or not enabled → **raise** (violation 3).
3. Required keys present and non-empty (`task_id`, `prompt`, `description`, `process_subagent_result_tool_name`) → else **raise**. The core reads the five names §5.3 settled; it never imports contrib's `SubagentSpawn`.
4. Create `Conversation(id=generate_id(), nodes=[], depth=parent.depth + 1)` in `conversations`.
5. Append `UserMessage(parts=[TextContent(text=prompt)])` to the child (the `TurnStart` is opened by the child's own drive — §4.1 stage 2).
6. Append `ChildConversation(conversation_id=child_id, tool_execution_id=execution.id)` to the **parent's** path.
7. Register an `AgentRun` for the child in `run.children` — eager under `autostart_subagents=True`, lazy (undriven) under `False`.

Executions are processed in path order, so the CC entries land in call order:
`TE1 > TE2 > CC2 > CC3` (PRD §4.1). Then one `SubagentsSpawned(conversation_ids=[…])`
per batch (PRD §9.4 — "announced as ONE batch signal rather than one per child").
It is yielded **before** the children are started, so an
`autostart_subagents=False` consumer can call `run.child(cid)` inside the
event's handler.

Idempotency across a reload is structural: the `ChildConversation` is the
durable record that a spawn was handled.

### E.3 Step 4 — resolving a finished child

For each `ChildConversation` in the open turn with `execution_result is None`
whose child conversation derives `IDLE` (its turn bracket closed, **whatever
the outcome** — PRD §4.1 stage 4, §5.6):

1. Mint `ToolCall(id=generate_id(), name=payload["process_subagent_result_tool_name"], arguments={task_id, prompt, description, conversation_id: child_id})`. The payload comes from the spawn execution named by `tool_execution_id`.
2. Run it through the **full ordinary lifecycle in the parent conversation**: `create_execution` → append (stamping `conversation_id` = parent) → `ToolCallReceived` → `decide` → `prepare` → body → `ToolExecuted`. Approvals, middleware, timeouts, cancellation, `context_tokens` and usage all apply (PRD §3.6, §7).
3. Set `ChildConversation.execution_result` through `put_entry` (so `before_entry_written` fires a second time for that entry), recalculating `context_tokens`.

Two derived rules:

> **[D-7] A runner-originated invocation skips the doom-loop check and the private-name refusal.**
> `_is_doom_loop` is a heuristic about *model* behavior; the refusal rule (§3.6
> rule 2) is about a *model* naming a private tool. Neither applies to a call
> the runtime minted.

> **[D-8] A non-COMPLETED result execution still resolves the child. (PRD gap.)**
> §7 step 4 says "that tool's `ExecutionResult` becomes
> `ChildConversation.execution_result`" and assumes one exists. If the result
> execution terminalizes without a result (FAILED / NOT_FOUND / TIMED_OUT /
> REJECTED / CANCELLED), the runner writes
> `ExecutionResult(content=[TextBlock(derived text)], is_error=True)` using
> `ConversationProjector.project_tool_execution` — the same derivation the wire
> and the `ToolExecuted` event already share, so the three never disagree. The
> alternative (leaving the child unresolved) blocks the parent forever.

### E.4 Child drives

- A child never runs the compaction step and never receives `post_message` / `schedule_compaction` (PRD §8, §9.6).
- Step limits resolve from `depth` (C.4).
- A child's failure **never escapes** (PRD §5.6): under `autostart_subagents=True` the child's `_consume` catches the drive's exception, its bracket has already closed `ERRORED`/`TIMED_OUT`, and step 4 resolves its `ChildConversation` with the derived error text. Under `False` the failure re-raises on **that child's own handle** — the app is consuming it — and the durable outcome is identical.
- `get_conversation_status` becomes genuinely subtree-aware here (the recursion added in B.1 starts returning non-empty `unresolved_children`).

### E.5 Tests

New directory `tests/agent/subagents/` (PRD §11), with a
`FakeSpawnTool` / `FakeResultTool` pair built on `FakeTool` (core tests must
not import contrib):

| file | covers |
|---|---|
| `test_spawn_handshake.py` | the §4.1 four-stage timeline as one full-session assertion; `is_subagent_spawn=False` creates no child; the three violations each raise |
| `test_parallel.py` | two children advancing concurrently; the parent blocked until both resolve; nondeterministic landing order is tolerated (assert as a set) |
| `test_approvals.py` | §9.6's A/B/C story — parent `BUSY` while C is gated, flipping to `BLOCKED` when A and B finish; `pending_approvals()` subtree-scoped; `notify()` unsticking a parked child |
| `test_cancellation.py` | `child.cancel()` resolves that child and the parent continues; `run.cancel()` cascades; **[D-6]**'s wind-down resolution |
| `test_failures.py` | §5.6 — a child's provider error resolves rather than propagating; `subagent_hard_max_steps` |
| `test_autostart.py` | `False`: `SubagentsSpawned` → `run.child(cid)` → `gather(drive(...))`; a never-driven child blocks; no double delivery (**[D-5]**) |
| `test_usage.py` | PRD §8's explicit requirement — every usage record under the right conversation, session total correct across the catalog |
| `test_resume.py` | reload mid-tree: unresolved `ChildConversation` entries re-mint child handles (§9.5) |
| `test_depth.py` | the gate withholds the spawn tool at the cap; the prompt part goes silent with it |

---

## Stage F — `luca/agent/contrib/subagents/`

```
subagents/
├── __init__.py    # SubagentsPlugin, SpawnSubagent, CreateConversationResult, SubagentSpawn
├── tools.py       # the two tools + the SubagentSpawn payload model
└── plugin.py      # SubagentsPlugin + the gating registry + the prompt part
```

Exactly PRD §7 / §7.1. Two implementation notes:

> **[D-9] Gating lives in a `SimpleToolRegistry` subclass.**
> §7.1 sketches `get_tool_registry` returning a plain `SimpleToolRegistry`, but
> the gate has to live somewhere and §7 puts it on the registry ("The
> **registry** decides"). `SubagentToolRegistry(SimpleToolRegistry)` overrides
> `get_tools` only, filtering spawn-declaring specs by the same predicate the
> prompt part uses. Everything else is inherited.

The prompt part is a **callable** `(session, conversation_id) -> str | None`,
returning `None` (Stage A's new `coerce_system_prompt_part` behavior) when
spawning is not possible — driven by the identical predicate, which is the
whole reason it is a callable (PRD §7.1).

Tool names `spawn_subagent` / `create_conversation_result` must be globally
unique; `ProxyToolRegistry` raises loudly on a collision, so this is a naming
rule and not a risk to design around.

Tests: `tests/agent/contrib/test_subagents.py` — the plugin surface, the spec
shapes (incl. `is_private` and the declared `output_schema`), the gate at each
depth, the prompt part's two states, and `CreateConversationResult`'s two
branches (trailing `AssistantMessage` → its text parts; otherwise a summary).

---

## Stage G — TUI and demo

PRD §3.5's Scope paragraph names three things, and they are the work:

1. **History replay** (`tui/app.py:361`) walks `session.conversations[session.main_conversation_id].nodes`, and renders a `ChildConversation` as a collapsed subagent cell.
2. **The approval modal must name which conversation is asking.** `pending_approvals()` is now subtree-scoped, so `_resolve_approvals` (`tui/app.py:230-249`) groups by `execution.conversation_id` and labels non-main ones. `build_approval_prompts` / `apply_answer` / `pending_requests` keep their signatures (PRD §3.2).
3. **Interleaved events.** `_on_agent_event` keys its live-cell state (`_live_reasoning`, `_live_text`, `_tool_cells`) by `event.conversation_id` — today they are single slots, and two conversations streaming text would splice into one cell.

Plus the mechanical ones: `_drive`'s loop inverts to PRD §9.7's ordering
(**drive first, prompt second**) and switches `awaiting_approval()` → `blocked()`;
`_refresh_status` uses `runner.status().value`; the context bar passes the main
conversation id to `calculate_context_used`; `test_wiring.py:50`'s
set-equality-over-tool-names assertion stays valid precisely because
`is_private` keeps the result tool off the wire list.

> **[D-10] The demo wires `SubagentsPlugin` behind a flag.**
> `build_runner(..., subagents: bool = False)` adds the plugin; `main.py` gains
> `--subagents`, which also sets `subagents_enabled=True` on a new session's
> `RuntimeConfig`. Default off, so `--faux` and every existing TUI test are
> untouched. Wiring it is what makes the feature exercisable by hand; the flag
> is what keeps it out of everyone's way.

---

## Stage H — Docs

Per `docs/llm.txt`: prefer a new `## N.` section on an existing page over a new
file; update the folder `README.md` table and the `Next:` chain when a page is
added; validate every snippet with `uv run python - <<'PY'`.

| page | change |
|---|---|
| `agent/02-data-model.md` (665 L) | biggest. The container section (`conversations` + `main_conversation_id` + `previous_conversation_id` + `depth`), the status table (5→4), `ConversationRuntimeStatus`, `ChildConversation` in the entry recap, `ToolExecution.conversation_id`, `ToolSpec.is_private`, three mutable entry types. Delete the `AgentSession.status` line (`:516`). |
| `agent/03-tools.md` | `ToolSpec` "advertisement" gains its one exception; `output_schema` gains its second reader (the runner's gate) — "the framework never reads it" stops being true and must say so. |
| `agent/04-runner.md` | the status machine, `run(autostart_subagents=)`, `run.children` / `child()` / `approvals` / `notify()`, `RunResult`, cancellation cascade, "a drive returns when nothing in its subtree can advance". |
| `agent/05-permissions.md` | the four registry signatures; the out-of-band loop becomes two paths — **outside a run, the next `run()` re-asks; inside a run, `notify()` is what re-asks** (PRD §9.8's named doc task); `pending_approvals()` subtree-scoped. |
| `agent/06-system-prompts.md` | callables take `(session, conversation_id)` and may return `None`. |
| `agent/07-middleware.md` | `before_entry_written`'s entry types gain `ChildConversation`; a §3.4 concurrency section; the explicit caveat that hooks stay conversation-blind and an application that assumed one conversation gets wrong behavior with no error. |
| `agent/08-runtime-config.md` | the four `subagents_*` fields. |
| `agent/10-projection.md` | `project(nodes, entries)`; `project_child_conversation`; a private execution projects nothing. |
| `agent/11-context-and-usage.md` | `calculate_context_used(session, conversation_id)`; a `ChildConversation`'s size is its result; usage across the catalog. |
| `agent/12-compaction.md` | the two new signatures; main-conversation-only in V0; `previous_conversation_id` replaces `conversation_history`. |
| `agent/13-subagents.md` | **new** — the feature page: spawn, parallelism, approvals, cancellation, `autostart_subagents`, the two loops, depth, config. Renumber nothing; it appends. |
| `contrib/tools/README.md` | the five moved signatures; `is_private`. |
| `contrib/subagents/README.md` | **new** — the package page. |
| `contrib/simple_tool_registry/`, `resource_permissions/`, `shell/`, `simple_context_manager/`, `plugins/` READMEs | signatures + the per-conversation keying. |
| `agent/README.md`, `contrib/README.md` | page/package tables. |
| `AGENTS.agent.md` | the file layout, the two-mutable-entry-types rule, the status machine, the registry/context-manager contracts, the engine order, the test-file table (+ `tests/agent/subagents/`). |
| `AGENTS.md` | one line for `contrib/subagents`. |

Private tools get a `## N.` section in `agent/03-tools.md`, not a page —
`docs/llm.txt`'s explicit preference.

---

## Risks, ranked

1. **The lazy-root fan-in (D.2).** Racing a held generator-step task against a
   queue is the one genuinely novel piece of concurrency. Failure mode: a
   dropped event or a hung pull. Mitigation: it is the *only* new primitive —
   child drives reuse the existing eager `_consume` path verbatim — and it is
   exercised in its degenerate form by the whole existing suite at the end of
   Stage D, before a single subagent exists.
2. **The `_recheck` ordering (D.3 step 0b, PRD §9.8).** Clear-then-read, and the
   pre-return re-check. Getting it backwards reproduces §9.8's exact timeline —
   an answered gate sitting inert for the length of a sibling's tool call.
   Mitigation: a dedicated test that fires `notify()` while `decide()` is in
   flight.
3. **Status derivation recursion (B.1 + E.4).** Called on every TUI repaint and
   every drive boundary. Failure modes: a parent that never flips `BUSY → BLOCKED`
   (the run hangs) or flips too early (the run returns with work outstanding).
   Mitigation: §9.6's A/B/C story is a test, not a comment.
4. **The event `conversation_id` sweep (B.4).** ~200 literals. Purely mechanical,
   but a wrong id in a fixture is invisible until something routes by it.
   Mitigation: each test file has exactly one conversation in Stages B–D, so the
   value is constant per file; the subagent tests in Stage E are where a wrong
   id would actually fail.
5. **`filterwarnings = ["error"]`.** Any un-awaited task or unclosed resource in
   the new child-task machinery fails the build rather than warning. This is a
   feature — it is how a leaked child drive gets caught — but it will bite during
   Stage D.

---

## Open, and deliberately not decided here

- **Middleware scoping.** PRD §12 defers the whole hook surface; this plan does not touch it beyond docstrings.
- **Background subagents, depth > 1, named subagent types.** Out of scope; nothing here forecloses them (the spawn execution completes immediately, the data model nests, and the gate reads a declaration rather than a name).
- **Per-scope permission rules.** `PermissionStrategy.add_rule` stays global (PRD §3.5).
- **Richer child projection.** V0 ships one synthetic user message per child; §4.2's batched rendering remains a projector subclass away.
