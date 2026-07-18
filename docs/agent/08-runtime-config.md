# Runtime config

`RuntimeConfig` holds the runner's behavioral knobs — timeouts, step limits,
doom-loop detection. It rides on the session (`session_config.runtime_config`),
persists with it, and is read **live** on every use (there are no constructor
kwargs for these). The defaults reproduce the unconfigured behavior exactly:
nothing is limited.

```python
from luca.agent.core import RuntimeConfig, Seconds, MilliSeconds, Inf

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
`MilliSeconds(500)` is an explicit-unit identity. Any int knob accepts `Inf`
(`-1`) or `0` to **disable** the limit.

## Timeouts

| Field | Effect |
|---|---|
| `tool_execution_timeout_in_ms` | Deadline for a tool call. Expiry hard-cancels it → `TIMED_OUT` (resultless). A tool's own `timeout_in_ms` class var (snapshotted into the birth `ToolSpec`) beats this. |
| `client_completion_timeout_in_ms` | Wall-clock (`total_timeout`) for a model call. |
| `builtin_client_completion_timeout_in_ms` | Per-phase HTTP timeout. **Inert** when the runner is built with a `provider=` instance (the caller owns that lifecycle). |
| `tool_cancellation_grace_period` | On cancel, how long a tool may keep running before a hard kill. `0` = immediate. A tool returning within grace records its real result. |
| `llm_completion_cancellation_grace_period` | Same grace window for an in-flight model call. |

A model call that times out closes the turn `TIMED_OUT` and re-raises (status →
`PENDING`, retry-ready). See [`03-tools.md`](03-tools.md) §6 for the cooperative
side of tool cancellation.

## Step limits

A "step" is one `AssistantMessage` in the current turn — i.e. one model response.
Two ceilings guard runaway loops:

| Field | Behavior when reached |
|---|---|
| `soft_max_steps` | The next model call gets `tool_choice="none"` (if `limit_tool_choice_on_soft_max_steps_reached`, default `True`) — the model must answer in text, ending the turn gracefully. |
| `hard_max_steps` | The turn closes immediately with `TurnOutcome.ERRORED` (status → `PENDING`, retry-ready). A hard stop, not a graceful one. |

Set `soft` below `hard` so the agent gets a chance to wrap up before the hard cut.
Setting them **equal** (both > 0) emits a `UserWarning` — hard prevails, so the
soft stop never happens.

```python
RuntimeConfig(soft_max_steps=20, hard_max_steps=30)   # nudge at 20, force-stop at 30
```

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
session.session_config.runtime_config.hard_max_steps   # 30
```

Next: [`09-plugins.md`](09-plugins.md).
