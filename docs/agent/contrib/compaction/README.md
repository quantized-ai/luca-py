# Compaction policy

Core defines the `CompactionPolicy` contract and owns the transition (archive
the old conversation, swap in the new one) — see
[12-compaction.md](../../12-compaction.md). It ships no concrete policy. This
package is one: `SummarizingCompactionPolicy`, which decides when to compact and
produces the summary.

```python
from luca.agent.contrib.compaction import SummarizingCompactionPolicy

policy = SummarizingCompactionPolicy(
    keep_turns=2,  # 0 = full summary
    threshold=0.8,
)
runner = AgentSessionRunner(session, compaction_policy=policy)
```

## What it does

- `should_compact(session)` is the **context gauge**: sum the active path's
  `context_tokens`, divide by the model's window (`luca.client.catalog`, falling
  back to a default), and compare to `threshold`. Sync, as the contract
  requires. `enabled=False` turns auto-compaction off; `/compact` still works.
- `compact(session, nodes, entry)` folds the older span into a summary. It
  projects the folded nodes, calls the session's own model for a summary, fills
  the entry's `parts`, and returns the new path `[entry.id, *kept]`.

## `keep_turns` — what survives verbatim

One knob decides the split:

| `keep_turns` | Keeps |
|---|---|
| `0` (default) | nothing — summarize everything |
| `N` | the last N exchanges (each user message plus its turn); the cut is always a turn boundary |

## Extending it

Subclass `SummarizingCompactionPolicy` and override one of the public seams:

| Override | Changes |
|---|---|
| `should_compact(session)` | when compaction fires |
| `select_keep(candidates, session)` | what survives verbatim (e.g. by tokens instead of turns) |
| `summarize(session, folded)` | how the summary is produced — the prompt, the model, the request. The `text_of` / `usage_of` static helpers are there for a custom implementation |

`compact` just orchestrates the three, so it rarely needs overriding.

## In the TUI

`cli.py` builds a policy from flags and passes it to the app; the context bar
under the transcript shows utilization, colored toward red as it nears the
threshold.

| Flag | Effect |
|---|---|
| `--no-autocompact` | disable auto-compaction (keep `/compact`) |
| `--compact-threshold F` | auto-compact at this utilization fraction (default 0.8) |
| `--compact-keep-turns N` | keep the last N exchanges verbatim (0 = summary only) |
