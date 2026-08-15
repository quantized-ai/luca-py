# `contrib/checkpoints` — undo that puts the files back too

One checkpoint per user turn. Restoring one reverts the workspace with git AND
rewinds the conversation with [`rewind_to`](../../15-rewind.md), so the model's
history and the files agree again.

```python
from luca.agent.contrib.checkpoints import CheckpointService, ShadowGitStore

service = CheckpointService(ShadowGitStore(".", "~/.luca/projects/-me-proj/checkpoints.git"))

await service.take(session, label="add the parser")   # BEFORE post_message
runner.post_message("add the parser")
# ... the agent edits files ...

await service.restore(runner, service.checkpoints(session)[0])
# the files are back, and the turn is gone from the conversation
```

In the TUI this is `/undo` (the most recent checkpoint) and `/rewind` (a picker),
on by default and switched off with `--no-checkpoints` or `"checkpoints": false`.

## 1. The three pieces

| Piece | Job |
|---|---|
| `ShadowGitStore` | Snapshots the workspace into a private git repo the user never sees |
| `CheckpointIndex` | Maps a snapshot to the conversation position it belongs to, on `AgentSession.extras` |
| `CheckpointService` | Joins the two: `take()` and `restore()` |

## 2. Why git, and not per-edit undo records

Asking each mutating tool to record what it replaced covers `edit`, `write` and
`apply_patch` — and misses `bash` completely. One `sed -i` and the inverse record
does not exist. A snapshot captures the workspace however it got that way, and
gets renames and deletions for free.

The shadow repo names its git dir and work-tree explicitly, so nothing is ever
written into the workspace and the user's own `.git` is excluded rather than
traversed:

```
git --git-dir=<store>/checkpoints.git --work-tree=<workspace> add -A
```

## 3. Anchoring

`take()` records the current path LEAF, then the caller posts the message. So
restoring truncates to just before the turn, message included.

```python
await service.take(session)      # anchor = the previous turn's TurnFinish
runner.post_message("...")       # opens the turn this checkpoint undoes
```

`checkpoints(session)` returns only the ones whose anchor is still on the main
path, newest first — after a rewind, checkpoints from the discarded future are
kept in the index but never offered.

## 4. Restoring is two-sided and rolls back

Files first, then the conversation: if git fails the session is untouched. If
`rewind_to` then REFUSES (a live run, an open turn), the workspace is rolled back
to a safety snapshot taken on the way in, so a refused restore leaves nothing
half-done.

```python
try:
    ok = await service.restore(runner, checkpoint)
except AgentError:      # open turn / live run — cancel and retry
    ...
```

> ⚠️ **`.gitignore`d files are not snapshotted, so they are not restored.**
> Capturing them would mean swallowing `node_modules` and every virtualenv on
> the first snapshot. Edits the agent makes to ignored paths are not undoable.

## 5. Limits worth knowing

| Limit | Detail |
|---|---|
| No git binary | The feature reports `available is False` and every call no-ops |
| Untracked files | A restore removes non-ignored files created since the snapshot, including ones a human made |
| One repo per project | Two sessions on one workspace can restore over each other, as they already race on the files |
| The model cannot restore | No tool is registered; this is a person's action |

## 6. Not a plugin

`CheckpointService` contributes no tools, no prompt parts and no middleware, so
it is a plain object the application wires rather than a `BasePlugin`. A model
able to undo its own last turn could erase a mistake and retry it silently.

Snapshotting also stays off the middleware chain: every hook is synchronous, and
a git commit over a large workspace on the event loop once per entry write is
not a trade worth making. `take()` is awaited by the application instead.

Next: back to the [contrib index](../README.md).
