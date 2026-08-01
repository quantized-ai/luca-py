# Runtime config

`RuntimeConfig` holds the runner's behavioral knobs — timeouts, step limits,
doom-loop detection. It rides on the session (`session_config.runtime_config`),
persists with it, and is read **live** on every use (there are no constructor
kwargs for these). The defaults reproduce the unconfigured behavior exactly:
nothing is limited.

```python
from luca.agent.core import (
    AgentSessionRunner, LLMConfig, RuntimeConfig, Seconds, MilliSeconds, Inf,
)

session = AgentSessionRunner.new_session(
    LLMConfig(model="openai/gpt-4o-mini", provider="openrouter"),
    runtime_config=RuntimeConfig(
        tool_execution_timeout_in_ms=Seconds(30),
        hard_max_steps=50,
        doom_loop_threshold=3,
    ),
)
```

Durations are **integer milliseconds**. `Seconds(30)` → `30000`;
`MilliSeconds(500)` is an explicit-unit identity. `Inf` (`-1`) **disables** any
knob; `0` disables a step/doom-loop count, but on a duration it means zero — an
immediate deadline, no grace.

## Timeouts

| Field | Effect |
|---|---|
| `tool_execution_timeout_in_ms` | Deadline for the dispatched tool **body**. Expiry hard-cancels it → `TIMED_OUT` (resultless). The birth `ToolSpec.timeout_in_ms` beats it — `Inf` there means that tool's body is unbounded whatever this says. |
| `client_completion_timeout_in_ms` | Wall-clock (`total_timeout`) for a model call; also bounds a compaction policy's `compact()`. |
| `builtin_client_completion_timeout_in_ms` | Per-phase HTTP timeout. **Inert** when the runner is built with a `provider=` instance (the caller owns that lifecycle). |
| `tool_cancellation_grace_period` | On cancel, how long a dispatched body may keep running before a hard kill. `0` = immediate. A tool returning within grace records its real result. |
| `llm_completion_cancellation_grace_period` | Same grace window for an in-flight model call (and for `compact()`). |

> ⚠️ **The deadline bounds the body, not the call.** It starts when the prepared
> callable is invoked; the registry's `get_tools`, `create_execution`, `decide`
> and `prepare` have **no deadline at all**. A tool configured with
> `timeout_in_ms=5000` is therefore not bounded end to end — the 5s bounds its
> body, and a registry that hangs in `prepare()` hangs the run indefinitely.

What ends a hang in those four phases is `cancel()`, not a deadline — and only
if the code there is cancellable. Each of them is raced against the run's token
with zero grace, and the runner awaits the killed task's unwinding so a
`finally` / `async with` cleanup completes. A phase that blocks the event loop
(sync I/O, a CPU spin) cannot be interrupted and will ignore the cancel until it
returns; push blocking work into `asyncio.to_thread`.

A model call that times out closes the turn `TIMED_OUT` and re-raises; the
conversation is then `IDLE` — a closed turn — so recovering means posting again. See [`contrib/tools/`](contrib/tools/README.md) §7 for
the cooperative side of tool cancellation.

## Step limits

A "step" is one `AssistantMessage` in the current turn — i.e. one model response.
Two ceilings guard runaway loops:

| Field | Behavior when reached |
|---|---|
| `soft_max_steps` | The next model call gets `tool_choice="none"` (if `limit_tool_choice_on_soft_max_steps_reached`, default `True`) — the model must answer in text, ending the turn gracefully. |
| `hard_max_steps` | The turn closes immediately with `TurnOutcome.ERRORED`. A hard stop, not a graceful one. |

Set `soft` below `hard` so the agent gets a chance to wrap up before the hard cut.
Setting them **equal** (both > 0) emits a `UserWarning` — hard prevails, so the
soft stop never happens.

```python
RuntimeConfig(soft_max_steps=20, hard_max_steps=30)   # nudge at 20, force-stop at 30
```

## Subagents

Off by default. Switching them on is **configuration, not installation** —
installing the spawn tool changes nothing until this says yes
([13](13-subagents.md)).

| Field | Effect |
|---|---|
| `subagents_enabled` | `False` by default. The gate every spawn is checked against |
| `subagents_max_depth` | how deep the tree may go: `N` allows spawning from depths `0..N-1`, so the deepest subagent sits at depth `N` — the main conversation plus `N` levels. Default `1`: the main conversation spawns, a subagent does not |
| `subagents_max_per_turn` | how many subagents one conversation may spawn in one turn; `Inf` (default) = no limit. Spent budget withholds the spawn tool; an overflow call in one response is born `REFUSED` ([13](13-subagents.md)) |
| `subagents_max_workers` | how many subagents may be **doing work** at once, session-wide; `Inf` (default) = no limit. Spawning always succeeds — the rest queue for a slot ([13](13-subagents.md)). Size it by fan-out, never by depth; `0` is invalid (that is `subagents_enabled=False` spelled incorrectly) |
| `subagent_soft_max_steps` | soft step ceiling for a SUBAGENT's turn; `None` falls back to `soft_max_steps` |
| `subagent_hard_max_steps` | the same for the hard ceiling |

```python
RuntimeConfig(subagents_enabled=True, subagents_max_depth=2, subagents_max_workers=20, subagent_hard_max_steps=20)
```

> ⚠️ **A subagent is never compaction-checked in V0**, so its own step ceiling
> is what stops one growing without limit. Set it if a subagent can loop.

## Doom-loop detection

When the model repeats the **same tool call** (same name + arguments) several
times in a row, that's a doom loop. Set the threshold to flag it:

```python
RuntimeConfig(doom_loop_threshold=3)   # flag the 3rd identical consecutive call
```

On the Nth identical call the runner sets `ToolExecution.is_doom_loop_flagged =
True`. If `limit_tool_choice_on_doom_loop_flagged` (default `True`), subsequent
model calls in that turn get `tool_choice="none"`, breaking the loop by forcing a
text answer. `Inf` / `0` disables detection.

## Reading it back

The config is on the session, so it serializes and reloads with everything else:

```python
session.session_config.runtime_config.hard_max_steps   # 50
```

Next: [`09-plugins.md`](09-plugins.md).
