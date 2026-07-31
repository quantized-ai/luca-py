# Simple context manager

Core's `ContextManager` accounts for context and declares the compaction pair,
but its default never compacts — see
[11-context-and-usage.md](../../11-context-and-usage.md) and
[12-compaction.md](../../12-compaction.md). This package ships one that does:
`SummarizingContextManager`, which decides when to compact and produces the
summary. Core still owns the transition (archive the old conversation, swap in
the new one).

```python
from luca.agent.contrib.simple_context_manager import SummarizingContextManager

manager = SummarizingContextManager(
    keep_turns=2,  # 0 = full summary
    threshold=0.8,
)
runner = AgentSessionRunner(session, context_manager=manager)
```

## 1. What it does

Core's per-entry accounting is inherited untouched; two methods are added:

- `should_compact(session, conversation_id)` is the **context gauge**: sum that
  conversation's `context_tokens`, divide by the model's window
  (`luca.client.catalog`, falling back to a default), and compare to
  `threshold`. Sync, as the contract requires. `enabled=False` turns
  auto-compaction off; `/compact` still works.
- `compact(session, conversation_id, nodes, entry)` folds the older span into a
  summary. It projects the folded nodes, calls the session's own model for a
  summary, fills the entry's `parts`, and returns the new path
  `[entry.id, *kept]`.

The gauge is three exported functions, usable on their own — the TUI context bar
calls them directly:

| Function | Returns |
|---|---|
| `calculate_context_used(session, conversation_id)` | sum of `context_tokens` over that conversation's path |
| `get_context_window_size(session, default=200_000)` | the model's `context_window` from `luca.client.catalog`, else `default` |
| `calculate_utilization_ratio(session, conversation_id, *, default_window=200_000)` | `used / window`, clamped to `[0, 1]` |

The two that measure a path name a conversation; the window does not, because it
is a fact about the model and every conversation in a session uses the same one.
A session holds several paths ([`13-subagents.md`](../../13-subagents.md)), so a
gauge that summed "the" conversation would report a number with no single basis.

> ⚠️ **Only the MAIN conversation is compaction-checked in V0.** A subagent is
> bounded by `subagent_hard_max_steps` instead
> ([`08-runtime-config.md`](../../08-runtime-config.md)).

> ⚠️ **The gauge is only as good as the counts.** `context_tokens` is stored per
> entry by [`calculate_context`](../../11-context-and-usage.md) — inherited here
> as the character estimate. Override it with a model-aware tokenizer and the
> stored counts go stale on a model switch; `runner.recalculate_context_tokens()`
> re-derives them, and nothing in the framework calls it for you.

## 2. `keep_turns` — what survives verbatim

One knob decides the split:

| `keep_turns` | Keeps |
|---|---|
| `0` (default) | nothing — summarize everything |
| `N` | the last N exchanges (each user message plus its turn); the cut is always a turn boundary |

## 3. Extending it

Subclass `SummarizingContextManager` and override one of the public seams:

| Override | Changes |
|---|---|
| `should_compact(session, conversation_id)` | when compaction fires |
| `select_keep(candidates, session)` | what survives verbatim (e.g. by tokens instead of turns) |
| `summarize(session, folded)` | how the summary is produced — the prompt, the model, the request. The `text_of` / `usage_of` static helpers are there for a custom implementation |

`compact` just orchestrates the three, so it rarely needs overriding. Changing
only the prompt text needs no subclass at all: pass `summary_prompt=` (the
default is exported as `DEFAULT_SUMMARY_PROMPT`).

## 4. In the TUI

`cli.py` builds a manager from `luca.json` and the CLI flags below (a flag wins)
and passes it to the app; the context bar under the transcript shows
utilization, colored toward red as it nears the threshold.

| Flag | Effect |
|---|---|
| `--no-autocompact` | disable auto-compaction (keep `/compact`) |
| `--compact-threshold F` | auto-compact at this utilization fraction (default 0.8) |
| `--compact-keep-turns N` | keep the last N exchanges verbatim (0 = summary only) |

Next: back to the [contrib index](../README.md).
