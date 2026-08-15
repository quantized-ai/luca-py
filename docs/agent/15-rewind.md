# 15. Rewind

`AgentSessionRunner.rewind_to()` makes a conversation be one of its own earlier
prefixes again. It is the primitive behind undo, retry-from-here and
branch-from-here — and it is the *whole* of what core knows about them.

```python
from luca.agent.core import AgentSessionRunner

# ... a finished turn: [u1, ts1, a1, tf1, u2, ts2, a2, tf2]
runner.rewind_to("tf1")          # the conversation is now [u1, ts1, a1, tf1]
```

Nothing is deleted. The old conversation is ARCHIVED and the new one points
back at it, exactly as [compaction](12-compaction.md) already does — through the
same `SessionLedger.transition_conversation` door.

```python
session.conversations[old_id].nodes        # still the full path
runner.session.main_conversation_id        # now the successor
session.conversations[new_id].previous_conversation_id == old_id
```

## 1. What it takes

| Argument | Meaning |
|---|---|
| `entry_id: str` | The node that must remain the path's LAST one |
| `entry_id=None` | Rewind to an empty path |

Rewinding to the current leaf is a no-op and returns the conversation unchanged.

## 2. When it refuses

All four checks run before anything is written, so a refused rewind leaves the
session exactly as it was.

| Refuses | Why |
|---|---|
| A live run on the main conversation | A transition re-points the conversation a drive is holding in a local |
| An open turn | Work is in flight or parked; cancel and flush first |
| An `entry_id` off the main path | Nothing to cut at |
| A cut inside a turn bracket | The prefix would hold an assistant message whose tool executions were dropped |

```python
runner.cancel()
async with runner.run() as run:      # flush the cancelled turn
    async for _ in run:
        pass
runner.rewind_to("tf1")              # now it is allowed
```

> ⚠️ **Only the main conversation.** A subagent's conversation is never
> transitioned. A `ChildConversation` inside the dropped span leaves its child
> in `session.conversations` as inert history — nothing drives it again, because
> the link is no longer on any live path.

## 3. What survives

| Survives | Note |
|---|---|
| Every entry | `session.entries` is untouched; only the path changed |
| The archived conversation | A first-class row, reachable via `previous_conversation_id` |
| Usage records | `session.usages` keys on the conversation, so a catalog-wide total is unchanged |

Undoing a turn does not refund it. That is deliberate and it is the same answer
compaction gives: the tokens were really spent.

## 4. The model never sees it

A rewound span is simply absent from the next projection. No marker is written
and no synthetic message is added, so the model's history reads as though the
turn never happened.

## 5. Checkpoints build on it

Core stops here. Binding a rewind to a workspace snapshot — so undoing a turn
also puts the files back — is application policy, and
[`contrib/checkpoints`](contrib/checkpoints/README.md) is the shipped one.

```python
from luca.agent.contrib.checkpoints import CheckpointService, ShadowGitStore

service = CheckpointService(ShadowGitStore(workspace, store_dir / "checkpoints.git"))
await service.take(session, label="add the parser")   # before post_message
...
await service.restore(runner, service.checkpoints(session)[0])   # files + conversation
```

Next: back to the [index](README.md).
