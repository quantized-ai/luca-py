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
tool_execution  (private result tool)
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
    runtime_config=RuntimeConfig(subagents_enabled=True, subagents_max_depth=1),
)
runner = PluginAgentSessionRunner(session, plugins=[SubagentsPlugin(), *your_plugins])
```

Installing the plugin without `subagents_enabled=True` changes nothing: the
spawn tool is withheld and the prompt that teaches it stays silent. See
[contrib/subagents/](contrib/subagents/README.md) for the package itself.

| Config | Effect |
|---|---|
| `subagents_enabled` | the gate. `False` by default |
| `subagents_max_depth` | how deep the tree may go. `1` — the only supported value in V0 — means the main conversation spawns and a subagent does not |
| `subagent_soft_max_steps` / `subagent_hard_max_steps` | step ceilings for a SUBAGENT's turn; `None` falls back to the main ones |

## 2. What happens when the model spawns

1. The model calls the spawn tool like any other tool. It completes
   immediately — spawning is its whole job — and returns a short status line.
2. Its result carries a **structured payload** the model never sees, and that
   payload is the handshake: the runner creates a child conversation, seeds it
   with the prompt, and appends a `ChildConversation` entry linking the two.
3. `SubagentsSpawned` is emitted with the new conversation ids.
4. The children run. The parent's turn **cannot end** until every link has a
   result — projecting an unresolved one raises ([10](10-projection.md)).
5. As each child's turn closes, the runner runs a **private** tool
   ([03](03-tools.md) §6) that turns the finished conversation into one
   `ExecutionResult`, and stores it on the link.
6. With every child resolved, the parent calls the model again — now with each
   subagent's answer on its path as a synthetic user message.

A child that failed, timed out or ran out of steps is a *finished* child: its
result says so, and the parent carries on. **A subagent's failure never
propagates as an exception.**

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

> ⚠️ **`False` is an obligation.** The parent's turn is blocked on its
> children, so every spawn must be driven or cancelled — otherwise the
> conversation never finishes. Forwarding follows ownership: under `False` a
> child's events reach you on the child handle only, never twice.

`start()` always implies `True`: an eager run completes regardless of
observation, and a subagent nobody drives would stop it doing so.

## 4. Approvals

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

## 5. Cancellation

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

## 6. What a subagent sees

| | |
|---|---|
| its conversation | starts with ONE user message: the spawn prompt. Nothing of the parent's history — make the prompt self-contained |
| its tools | the same registries the parent has, minus the spawn tool once it is at the depth cap |
| its system prompt | assembled for **its** conversation, so a callable part can say something different to it ([06](06-system-prompts.md)) |
| its context | its own window, never compaction-checked in V0 — bound it with `subagent_hard_max_steps` |
| its usage | its own row in `session.usages` ([11](11-context-and-usage.md)) |
| its result | its last assistant message, turned into one `ExecutionResult` |

A subagent never receives a user message after the seed: `post_message` is
main-conversation only.

## 7. Shared state is yours to key

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

> ⚠️ **Middleware is conversation-blind.** No hook receives a
> `conversation_id`, so an application middleware written for a single
> conversation gets wrong behavior — silently — once subagents are on
> ([07](07-middleware.md) §5).

## 8. Bringing your own spawn tool

The runner matches a **declaration**, never a tool name. A tool whose
`output_schema` declares `is_subagent_spawn` is a spawn tool: it is subject to
the depth cap, and a completed call whose `structured_content` says
`is_subagent_spawn: True` creates a child.

```python
class DelegateWork(Tool):
    name = "delegate_work"
    output_schema = SubagentSpawn          # the declaration IS the contract

    async def execute(self, args, session, conversation_id, *, cancellation_token):
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
either name. Three shapes are refused loudly rather than half-honored: a payload
from a tool that never declared one, a spawn from a conversation past the cap,
and a payload missing a required field.

Next: back to the [index](README.md).
