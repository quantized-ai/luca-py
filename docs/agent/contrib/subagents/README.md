# Sub-agents

Delegate read-only research to background sub-agents. The `task` tool lets the
agent spawn a focused, read-only sub-agent (map a subsystem, grep for a pattern,
answer a self-contained question); each runs as its own `AgentSessionRunner`
over its own `AgentSession`, in the background, and its findings are delivered
back into the conversation when it finishes. Spawning several in one turn runs
them in parallel.

The isolation is not a style choice: one runner drives one run at a time, and an
`AgentSession` is mutated in place with no locks, so a sub-agent *must* have its
own runner and session. This package is application code on the core's public
surface — the core has no sub-agent concept.

```python
from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.simple_tool_registry import YoloPermissionPolicy
from luca.agent.contrib.subagents import (
    BUILTIN_AGENT_TYPES,
    SubAgentManager,
    SubAgentPlugin,
    build_readonly_registry,
)
from luca.agent.core import AgentSessionRunner


def factory(agent_type, child_session):
    profile = BUILTIN_AGENT_TYPES[agent_type]
    return AgentSessionRunner(
        child_session,
        tool_registry=build_readonly_registry("."),
        system_prompt_parts=[profile.system_prompt],
        provider=None,  # real runs resolve the model from the client cache
    )


manager = SubAgentManager(
    runner_factory=factory,
    child_model=session.session_config.llm_config,
)
runner = PluginAgentSessionRunner(
    session,
    plugins=[SubAgentPlugin(manager, YoloPermissionPolicy())],
)
```

The TUI wires all of this for you (`build_runner` builds the manager and returns
it); see §4.

## 1. The flow: spawn → run → notify

1. The model calls `task(agent_type, prompt)`. The tool body calls
   `manager.spawn(...)` and returns **immediately** with a task id — the parent
   turn is never blocked.
2. `spawn` builds a child runner over a fresh child session, posts the prompt,
   and launches a background `asyncio.Task` that drives the child under a
   concurrency cap (`max_concurrency`, default 2) and a per-task timeout
   (`per_task_timeout_s`, default 300s).
3. Each transition is published as an immutable `SubAgentTask` snapshot on
   `manager.updates()`. The caller renders progress and, on completion, injects
   the result back into the parent conversation (the TUI does both).

`SubAgentTask` fields:

| Field | Meaning |
|---|---|
| `id` | task id (from `generate_id`) |
| `agent_type` | which built-in type was spawned |
| `title` / `prompt` | UI label and the task text |
| `status` | `queued` → `running` → `done` / `failed` / `cancelled` |
| `steps` | assistant steps observed while running (coarse progress) |
| `result` | the sub-agent's final text (on `done`) |
| `error` | the failure reason (on `failed` / `cancelled`) |

## 2. The read-only toolset

A sub-agent gets exactly the three read-only shell tools, under a YOLO policy:

```python
from luca.agent.contrib.subagents import build_readonly_registry

registry = build_readonly_registry(workspace)   # read, glob, grep — nothing else
```

`build_readonly_registry` never includes `edit` / `write` / `apply_patch` /
`bash`, and never the `task` tool, so a sub-agent cannot mutate anything and
cannot spawn another sub-agent (recursion is prevented by construction).

> ⚠️ **YOLO is deliberate, and it is a gate, not a sandbox.** A sub-agent is
> headless, so an ASK-mode strategy would stall on the first prompt with no human
> to answer it. YOLO is safe here only because the toolset cannot mutate; a YOLO
> read can still read outside the workspace.

## 3. Built-in agent types

`BUILTIN_AGENT_TYPES` maps a name to a `SubAgentType` (system prompt + step
budget). Both types share the read-only toolset and inherit the parent's model.

| Type | For | Budget (soft / hard steps) |
|---|---|---|
| `explore` | locate and map code, ending with `file:line` findings | 12 / 16 |
| `general` | answer one focused research question | 20 / 25 |

The budget rides the child session's `RuntimeConfig`: `soft_max_steps` drops the
model's tool calls so it must finalize, `hard_max_steps` is the backstop.

## 4. In the TUI

`build_runner` builds the `SubAgentManager` (returned as its third value) and
installs `SubAgentPlugin` sharing the one `PermissionStrategy`. The app runs a
dedicated worker that consumes `manager.updates()`, renders a compact
`SubAgentCell` per task (type, running step count, then the final result), and on
completion injects the result as a user message and re-drives — but **only when
the parent is idle**, so a live turn is never cancelled. Child token streams stay
inside the manager; the transcript shows coarse status only.

> ⚠️ **Background results wait for an idle parent.** A finished sub-agent's result
> is injected the moment the parent turn closes, not mid-turn. Cancelling a turn
> (Escape) leaves background sub-agents running; quitting the app closes them.

## 5. Extending it

| Override / argument | Changes |
|---|---|
| `SubAgentManager.extract_result(session)` | how a child's answer is read (default: the last assistant message's text) |
| `runner_factory` | how a child runner is built — the seam for a custom toolset, prompt, or (in tests) a per-child provider |
| `agent_types=` | supply your own `{name: SubAgentType}` map instead of `BUILTIN_AGENT_TYPES` |
| `max_concurrency=` / `per_task_timeout_s=` | the concurrency cap and per-task deadline |

Next: back to the [contrib index](../README.md).
