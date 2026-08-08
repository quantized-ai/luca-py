# Subagents

A subagent is a **second conversation in the same session**, working on its own
task at the same time as the main one. The parent spawns one per independent
piece of work, they run in parallel, and each reports back a single result.

Nothing about the data model changes: a subagent's conversation is an ordinary
path over the same entry bag, and it is linked into its parent by one entry
([02](02-data-model.md) §7).

```
main c1                                subagent c2 (depth 1)
─────────────────────────              ───────────────────────────
assistant  → spawn_subagent ×2
tool_execution ×2
child_conversation → c2  ────────────▶ user "Read alpha.txt and report back"
child_conversation → c3  ─┐            … its own turn, its own tools …
                          │            turn_finish
                          └──────────▶ (c3, in parallel)
tool_execution  (private result tool — c2's answer, projected as a task update)
assistant  "alpha is a shopping list; still waiting on beta."   ← a wake round
tool_execution  (private result tool — c3's answer)
assistant  "alpha is a shopping list; beta is a changelog."
turn_finish
```

## 1. Turning them on

Two things, deliberately separate — the capability is **configuration**, not
installation:

```python
from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.subagents import SubagentsPlugin
from luca.agent.core import AgentSessionRunner, LLMConfig, RuntimeConfig

session = AgentSessionRunner.new_session(
    LLMConfig(model="openai/gpt-4o-mini", provider="openrouter"),
    runtime_config=RuntimeConfig(subagents_enabled=True, subagents_max_workers=20),
)
runner = PluginAgentSessionRunner(session, plugins=[SubagentsPlugin(), *your_plugins])
```

Installing the plugin without `subagents_enabled=True` changes nothing: the
spawn tool is withheld and the prompt that teaches it stays silent. See
[contrib/subagents/](contrib/subagents/README.md) for the package itself.

| Config | Effect |
|---|---|
| `subagents_enabled` | the gate. `False` by default |
| `subagents_max_depth` | how deep the tree may go: `N` lets depths `0..N-1` spawn, so the deepest subagent sits at depth `N` — main plus `N` levels. Default `1`: the main conversation spawns, a subagent does not. Nesting multiplies cost; step limits are per conversation, not per tree |
| `subagents_max_per_turn` | how many subagents one conversation may spawn in one turn (§2). `Inf` by default; each conversation has its own budget, reset when its turn closes |
| `subagents_max_workers` | how many subagents may be doing work at once, session-wide (§4). `Inf` by default |
| `subagent_soft_max_steps` / `subagent_hard_max_steps` | step ceilings for a SUBAGENT's turn; `None` falls back to the main ones |

## 2. What happens when the model spawns

1. The model calls the spawn tool like any other tool. It completes
   immediately — spawning is its whole job — and returns a short status line.
2. Its result carries a **structured payload** the model never sees, and that
   payload is the handshake: the runner creates a child conversation, seeds it
   with the prompt, and appends a `ChildConversation` entry linking the two.
3. `SubagentsSpawned` is emitted with the new conversation ids, then one
   `SubagentStarted` per child that began (all of them, unless
   `subagents_max_workers` queued some — §4).
4. The children run; the parent parks. A pure spawn round does not call the
   model again — the confirmations travel with the first real update.
5. As **each** child's turn closes, the runner runs a **private** tool
   ([03](03-tools.md) §6) that turns the finished conversation into one
   `ExecutionResult`, stores it on the link, and **re-engages the model**: the
   parent takes a *wake round* with the new result while its siblings keep
   working, and can act on it — answer, spawn more, stop a task (§6). The
   wake condition is durable (`open_turn_unseen_material`: anything after the
   parent's last assistant message that is a posted user message or a
   terminal non-spawn tool result), so a reload mid-orchestration resumes the
   same behavior. `wake_parent_on_subagent_completion=False`
   ([08](08-runtime-config.md)) narrows the condition to everything BUT
   resolutions: each child still resolves the moment it finishes, but the
   model is re-engaged once, after the last one — a post or an ordinary tool
   result still wakes it either way, and the projection of the updates is
   unchanged, so the model sees every result it has not seen whenever it IS
   called.
6. The turn **cannot close** until every link has a result: a text-only
   answer with children still out is recorded and the parent parks again;
   the answer that lands after the last resolution closes the turn.

A child that failed, timed out or ran out of steps is a *finished* child: its
result says so, and the parent carries on. **A subagent's failure never
propagates as an exception.**

### Task updates

A resolution reaches the model as a synthetic user message rendered at the
**result execution's** path position — always below the parent's last reply,
never a rewrite of an earlier message ([10](10-projection.md) §8):

```
Subagent task update:
<task id=t1 status=completed completed_at="2026-08-01T18:03:12Z">
alpha is a shopping list
</task>
```

The `id` is the spawn payload's `task_id` — the same identifier the spawn
confirmation, `list_subagents` and `stop_subagent` speak — `status` is
`completed` or `failed` from the result's own verdict, and `completed_at` is
absolute UTC (projection stays deterministic; relative wording would need a
wall clock).

> ⚠️ **Wake rounds are real steps.** A turn orchestrating N subagents now
> records ~N extra assistant messages where the parked design recorded one,
> so `hard_max_steps` budgets tuned for the old shape trip earlier — and a
> hard-limit trip with children still running CANCELS them, settles every
> link, and closes ERRORED: a cost stop wins over work in flight. The
> doom-loop flag is per-turn and sticky too; a model repeating an identical
> call across wake rounds can pin `tool_choice="none"` for a long turn.
> `wake_parent_on_subagent_completion=False` is the lever back: it restores
> the one-round-at-the-end shape, trading mid-flight reactivity for steps.

### The spawn budget

`subagents_max_per_turn` caps how many subagents one conversation may create in
one open turn, in two layers:

- **The gate withholds.** Once the budget is spent, `get_tools` stops offering
  the spawn tool and the prompt part that teaches it goes silent — the
  capability simply leaves the table, and comes back when the turn closes.
- **The runner refuses at birth.** The tool list is fixed before the model
  call, so one response can still carry more spawn calls than the budget has
  room for. The overflow execution is born `REFUSED` with a
  `SpawnLimitReached` error whose text the model reads verbatim
  ("Spawn limit reached (3/3 subagents this turn). Complete the remaining
  work yourself; do not retry."). Its body never runs and no child is created.

Only calls that actually commit a subagent count: a spawn tool that declines
(`is_subagent_spawn: False`), a denied call, or one that raised consumes
nothing. Each subagent has its own turn and therefore its own budget — the caps
never aggregate across the tree.

## 3. The two loops

`autostart_subagents=` (default `True`) decides who drives the children. Both
shapes are supported end to end; pick by who owns the pacing.

**The framework drives them** — the ordinary case. Every child begins
immediately on its own task, advances whether or not you pull, and its events
arrive on the parent's stream tagged with its conversation:

```python
async with runner.run() as run:
    async for event in run:
        cell = ui.get(event.conversation_id)      # the stream is the whole tree's
        cell.render(event)
```

**You drive them** — when the application wants a handle per child (a pane per
subagent, its own back-pressure):

```python
from luca.agent.core.events import SubagentsSpawned

async with runner.run(autostart_subagents=False) as run:
    async for event in run:
        if isinstance(event, SubagentsSpawned):
            for cid in event.conversation_ids:    # handles exist by announcement time
                async with run.child(cid) as child:
                    async for child_event in child:
                        render(cid, child_event)
```

> ⚠️ **`False` is an obligation.** The parent's turn cannot CLOSE until every
> child resolves, so every spawn must be driven or cancelled — otherwise the
> conversation never finishes. Forwarding follows ownership: under `False` a
> child's events reach you on the child handle only, never twice, and no
> lifecycle events exist. Drive the children serially inside the
> `SubagentsSpawned` branch (the pattern above) — that is the shape the mode
> is designed around.

`start()` always implies `True`: an eager run completes regardless of
observation, and a subagent nobody drives would stop it doing so.

## 4. How many run at once

`subagents_max_workers` caps how many subagents are **doing work** at the same
instant, session-wide. Spawning always succeeds and always creates the child;
a spawn past the cap simply waits, holding its one seed message, until a slot
frees. The main conversation never competes for a slot.

```python
RuntimeConfig(subagents_enabled=True, subagents_max_workers=20)
```

One rule drives the scheduling: **a slot is held only while a subagent does
its own productive work**. A conversation waiting on its own children, waiting
at an approval gate, or winding a cancelled turn down holds none — a nested
parent steps out of the pool the moment it starts waiting, which is why a
nested tree cannot deadlock at any cap (`subagents_max_workers=1` is legal at
any depth: it serializes subagents completely and never wedges). Admission is
FIFO, in spawn order. Size the cap by **fan-out** — how many siblings should
work at once — never by `subagents_max_depth`: a nested parent holds a slot
only for its own rounds (a spawn round, a wake round per resolution), so depth
costs mostly sequence, not sustained concurrency. 20–30 suits a real
workload; the pool is runtime state, rebuilt
from the session on the next `run()`, and changing the knob mid-session never
interrupts work in progress — it only decides what starts next.

Three lifecycle events make the pool observable without inspecting the
session; together with `SubagentsSpawned` they drive a one-assignment-per-event
table. They fire whether or not a cap is set, for framework-driven subagents
only, and each names the SUBAGENT in `conversation_id`:

| Event | Meaning |
|---|---|
| `SubagentStarted` | its drive began (or restarted after an answered gate / a resume). Announced but not started = queued |
| `SubagentPaused` | its drive ended with the turn still open — an approval gate, a suspension. It will run again |
| `SubagentFinished` | its turn closed; `.outcome` carries the closing `TurnFinish`'s outcome. Its link resolves next. A `Finished(cancelled)` with no `Started` = cancelled before admission — the parent's flush announces the close |

```python
from luca.agent.core.events import (
    SubagentsSpawned, SubagentStarted, SubagentPaused, SubagentFinished,
)

states: dict[str, str] = {}
async with runner.run() as run:
    async for event in run:
        match event:
            case SubagentsSpawned(conversation_ids=ids):
                states |= dict.fromkeys(ids, "waiting")
            case SubagentStarted(conversation_id=cid):
                states[cid] = "running"
            case SubagentPaused(conversation_id=cid):
                states[cid] = "waiting"
            case SubagentFinished(conversation_id=cid, outcome=outcome):
                states[cid] = f"finished ({outcome.value})"
```

> ⚠️ **The cap is framework-owned.** It works by withholding a subagent's
> start, which under `autostart_subagents=False` is the application's to
> withhold — `run()` therefore raises when a cap is set with `False`. A gated
> subagent releases its slot, so the cap bounds how many *work*, not how many
> sit at a gate; `subagents_max_per_turn` bounds how many *exist*. The model
> is never told about the cap: the spawn tool succeeds either way and results
> simply arrive later. After a reload, rebuild any state table from the
> session (`IDLE` → finished, everything else → waiting) — a paused subagent
> leaves no durable trace, and the next `run()` re-announces what it admits.

## 5. Approvals and parked tool calls

A subagent's tools are gated exactly like the main agent's — same registries,
same policy. What changes is only *where* the gate comes from:

```python
for execution in runner.pending_approvals():      # SUBTREE-scoped
    print(execution.conversation_id)              # who is asking — no wrapper type
```

`pending_approvals(conversation_id=None)` returns every gated execution beneath
that conversation, so a subagent's request reaches the main runner's loop
untouched. Two ways to answer, differing only in when the re-ask happens
([05](05-permissions.md) §3):

| | When |
|---|---|
| the next `runner.run()` | the run has ended — the ordinary drive-then-prompt loop |
| `run.notify(execution)` | the run is still going: a subagent gated while its siblings keep working, so there is no between-drives moment to poll in |

```python
async for execution in run.approvals:      # gates as they are raised, at-least-once
    policy.record(execution.id, ask_user(execution))
    run.notify(execution)
```

A gate does **not** end the parent's run while anything else in the tree can
still advance. One rule explains that and everything like it: **a drive returns
only when nothing in its subtree can advance.**

A subagent's tool can also answer *not yet* ([03](03-tools.md) §7), and a
parked call surfaces through the twin door, subtree-scoped in the same way:

```python
for execution in runner.pending_deferred_tool_executions():   # SUBTREE-scoped
    await handlers[execution.tool_spec.name](execution)       # blocks until resolved
    run.notify(execution)                                     # inside a live run
```

`run.notify(execution)` is what makes that usable mid-run, for the identical
reason it exists for gates: a subagent parked on a deferred tool has no live
drive of its own — its drive returned, and the parent is what is still running —
so there is no between-drives moment to poll in. Its meaning is unchanged
(*something changed out of band for that execution's conversation, look again
NOW*; no decision and no result travels through it), and the restarted child
re-dispatches the call from scratch ([04](04-runner.md) §9).
`pending_deferred_tool_executions()` is a plain read over the session, so
calling it from another task while the run is live is safe — which is the whole
point, since this feature ships no stream: the application polls rather than
being pushed.

> **A parked subagent starves nobody.** Its drive returns, so its worker slot
> goes back exactly like at a gate (§4), and a nested parent waiting on it holds
> none either. And the parent's *own* path holds no unfinished tool call, so it
> can still be posted to, still call the model, and still `stop_subagent` the
> waiting child (§6).

## 6. Steering and stopping mid-flight

The wake rounds are what make a running orchestration steerable, from both
sides of the conversation:

- **The user posts.** `post_message` accepts a mid-orchestration parent — the
  children never see the message, but the parent does: the post is material,
  a parked drive is woken, and the model answers promptly (a message that
  lands while a wake round's own LLM call is in flight is caught by the
  drive's close-site fingerprint and answered in an immediate extra round).
- **The model manages its tasks.** Once the open turn has subagents, the
  plugin offers two more tools (withheld before the first spawn, taught by
  their own prompt part):

| Tool | Does |
|---|---|
| `list_subagents` | the turn's roster: each task's id, status (`pending` / `cancelling` / `completed` / `failed` — the same completed/failed split the task updates use), description and prompt |
| `stop_subagent(task_id, reason=None)` | cancels the named DIRECT child, best effort. The child winds down through the ordinary cancel machinery and reports back one final time as stopped |

A stop is the spawn contract's mirror, value-side only: the tool's completed
result carries `is_subagent_stop=True` in `structured_content`, and the
runner's drive consumes it — matching the task id among children spawned
**before** the stop was issued, so a later task reusing the id can never be
killed by an old signal. Task ids are model-authored and nothing enforces
uniqueness; with duplicates the target is *the first match still unresolved
when the stop was issued* (read off path positions), and a match that
resolved after the stop consumes the signal — a replayed stop can never fall
through to a same-id sibling. A target that already finished, is already stopping, or never
existed makes the signal a no-op (the shipped tool also tells the model so,
returning the payload with the flag down). In-flight tool bodies in the
stopped child follow `tool_cancellation_grace_period` — with the default `0`
they are hard-interrupted.

Two limits worth knowing when steering: `subagents_max_per_turn` counts every
spawn the OPEN TURN committed — a stopped task never releases its slot, so
stop-and-respawn spends two of the budget; and once `soft_max_steps` forces
`tool_choice="none"`, wake rounds can only answer in text — the control tools
are on the list but uncallable until the turn closes.

## 7. Cancellation

`cancel()` is conversation-scoped and cascades downward:

```python
run.cancel()                      # this conversation and every live subagent under it
run.child(cid).cancel()           # one subagent; its siblings and its parent keep going
```

Two rules make that safe:

- **Cancelling a conversation always ENDS it.** A subagent that was spawned but
  never driven has no bracket, so the cancellation opens one to close — nothing
  in the catalog is left mid-flight, and `run.child(cid).cancel()` is never a
  silent no-op.
- **Cancelling a subagent still RESOLVES its link**, with a result that says it
  was cancelled. That is not tidiness: an unresolved child on a closed turn is
  unprojectable, so the parent would wedge.

`cancel()` is a signal — it writes the request and returns; consuming one is a
drive's job. A subagent's own drive is usually gone by the time a cancel lands
(it parked at a gate, or nobody started it), so the **parent's** drive does that
flush. It is the only conversation that knows the child exists.

## 8. What a subagent sees

| | |
|---|---|
| its conversation | starts with ONE user message: the spawn prompt. Nothing of the parent's history — make the prompt self-contained |
| its tools | the same registries the parent has, minus the spawn tool once it is at the depth cap or past its spawn budget; `stop_subagent` / `list_subagents` appear only once its OWN open turn has children |
| its system prompt | assembled for **its** conversation, so a callable part can say something different to it ([06](06-system-prompts.md)) |
| its context | its own window, never compaction-checked in V0 — bound it with `subagent_hard_max_steps` |
| its usage | its own row in `session.usages` ([11](11-context-and-usage.md)) |
| its result | its last assistant message, turned into one `ExecutionResult` |

The seed is the child's **first** user message, not its only one:
`post_message("…", conversation_id=child_id)` appends into a live child's open
turn exactly as it does for the main conversation ([04](04-runner.md)). A
**finished** child rejects posts — its result is already resolved into the
parent. A mid-orchestration parent ACCEPTS them: the post wakes it, and
steering (including `stop_subagent`) is exactly what the wake rounds are for
(§6). (The shipped TUI still routes input to the main conversation only —
application policy, not a framework rule.)

## 9. Shared state is yours to key

One registry, one plugin, one permission strategy serve the whole tree — so any
state they hold must be keyed by conversation or the conversations overwrite
each other. The shipped contrib packages already are: the memory plugin's
scratchpad and todo list, and the shell plugin's read-before-write tracker.

| Rule | Why |
|---|---|
| No per-call state on `self` | two conversations call the same method concurrently |
| State keyed by `conversation_id` needs no lock | dispatch within one conversation is sequential |
| Deliberately shared state locks the **mutation**, never the I/O | locking the I/O serializes the parallelism you asked for |
| Process-global resources are scoped per conversation, not mutexed | a chdir or an env var set by one subagent is seen by all of them |

> **Middleware is conversation-aware.** Every hook receives
> `(session, conversation_id)` ([07](07-middleware.md) §5), so one instance
> safely serves the whole tree — route a subagent to a cheaper model, withhold
> tools by depth, attribute cost per conversation. The same keying rules above
> apply to a middleware's own state: it is shared exactly like the registry.

## 10. Bringing your own spawn tool

The runner matches a **declaration**, never a tool name. A tool whose
`output_schema` declares `is_subagent_spawn` is a spawn tool: it is subject to
the depth cap and the spawn budget, and a completed call whose `structured_content` says
`is_subagent_spawn: True` creates a child.

```python
class DelegateWork(Tool):
    name = "delegate_work"
    output_schema = SubagentSpawn          # the declaration IS the contract

    async def execute(
        self, args, session, conversation_id,
        *, tool_name, tool_call_id, cancellation_token,
    ):
        if not self.worth_delegating(args):
            return ExecutionResult(                       # gated, but spawns nothing
                content=[TextContent(text="Handled inline.")],
                structured_content={"is_subagent_spawn": False},
            )
        ...
```

That is why `is_subagent_spawn` is a `bool` rather than a marker: the gate reads
the declaration (before the call), the handshake reads the value (after it), so
a spawn tool that decides *not* to spawn is expressible and still gated. A name
match would not survive that — `delegate_work` would spawn through the handshake
and never be filtered, and the depth cap would quietly stop existing.

The payload also names which tool derives the child's result, so an application
can pair its own spawn tool with its own summarizer without the core knowing
either name. Three shapes are refused loudly (`AgentError`) rather than
half-honored: a payload from a tool that never declared one, a spawn from a
conversation past the cap, and a payload missing a required field. A registry
that ships spawn tools must build its tool list from `spawn_gate_open` (or
reimplement all three clauses — enabled, depth, budget): one that keeps
offering the spawn tool past the cap or the budget trips the runner's
`_verify_gate` check loudly. The budget overflow itself is not a violation —
it is an ordinary `REFUSED` execution the model reads and moves on from.

Next: [`14-logging.md`](14-logging.md).
