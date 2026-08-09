# Quickstart

A working agent is four things: an **`AgentSession`** (the state), a **`Tool`**
or two, a **tool registry** that owns them (resolution + approval), and the
**`AgentSessionRunner`** that drives it all. This page builds one, then shows
the loop that powers a real app.

Needs a provider key: either in the environment (`OPENROUTER_API_KEY` by
default — see [`../client/01-installation.md`](../client/01-installation.md)),
or passed explicitly as `AgentSessionRunner(..., api_key=...)`. It is a runner
argument and not an `LLMConfig` field, because the config is persisted with the
session ([02](02-data-model.md)).

## 1. One tool, one turn

```python
import asyncio
from pydantic import BaseModel, Field
from luca.agent.core import AgentSession, AgentSessionRunner, CancellationToken, LLMConfig
from luca.agent.core.events import TextBlock, ToolCallReceived, ToolExecuted
from luca.agent.contrib.tools import Tool
from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry, YoloPermissionPolicy

class WeatherArgs(BaseModel):
    city: str = Field(description="City to look up.")

class WeatherTool(Tool):
    name = "get_weather"
    description = "Return the current weather for a city."
    Args = WeatherArgs
    async def _execute(
        self, args: dict, session: AgentSession, conversation_id: str,
        *, tool_name: str, tool_call_id: str, cancellation_token: CancellationToken,
    ) -> str:
        return f"It's 22°C and sunny in {args['city']}."

async def main() -> None:
    session = AgentSessionRunner.new_session(
        LLMConfig(model="openai/gpt-4o-mini", provider="openrouter"),
    )
    runner = AgentSessionRunner(
        session,
        tool_registry=SimpleToolRegistry(
            tools=[WeatherTool()],
            permission_policy=YoloPermissionPolicy(),   # allow every tool call
        ),
    )

    runner.post_message("What's the weather in Madrid?")
    async with runner.run() as run:
        async for event in run:
            match event:
                case ToolCallReceived(execution=ex):
                    call = ex.raw_tool_call
                    print(f"  → {call.name}({call.arguments})")
                case ToolExecuted(result_text=text): print(f"  ← {text}")
                case TextBlock(text=text): print(text)

asyncio.run(main())
```

`SimpleToolRegistry` owns the whole tool lifecycle (resolution, validation,
approval, dispatch) — the runner only talks to it. `YoloPermissionPolicy`
auto-approves, so a single `run()` drives the whole turn:
model → `get_weather` call → tool result → model → final answer. The turn ends
with status `IDLE`.

`Tool` is a contrib convenience for Python tool authors: it turns `Args` into
the JSON Schema the model is shown. The core's only tool type is the plain
`ToolSpec`, so a registry fronting a remote tool server never imports `Tool`.
Bodies receive the live session — read `session.id` and
`session.session_config.llm_config`; see
[`contrib/tools/`](contrib/tools/README.md) for rich results and cancellation.

> ⚠️ **The session is read-only inside a tool.** The runner owns every write to
> it. Per-run application state belongs on your own object or a
> `contextvars.ContextVar`.

## 2. The drive loop

A real app polls the runner's status and reacts. This is the canonical shape —
it handles new input, approval gates, and cancellation resumption uniformly:

```python
while True:
    if runner.idle():
        runner.post_message(input("> "))          # nothing running → take input
    async with runner.run() as run:               # BUSY / BLOCKED / CANCELLING → advance
        async for event in run:
            render(event)
    if runner.blocked():
        resolve(runner.pending_approvals())       # a gate → answer it (see 05-permissions)
    save(runner.session)                          # persist after every turn
```

Each `run()` advances as far as it can, then stops at the next point that needs
you: the turn finished (`IDLE`) or nothing can advance until you answer a gate
(`BLOCKED`). The sketch polls, but posting is not idle-only — a message posted
*while* the agent works lands inside the open turn and is answered before the
turn completes ([04](04-runner.md) has the acceptance rules). Drive first, prompt second — answering writes to your policy, and
only a drive re-asks it. See [`04-runner.md`](04-runner.md) for the full status
machine and [`../../main.py`](../../main.py) for a complete TUI.

## 3. Persist and resume

The session *is* the state — save it as JSON, reload it later, keep going:

```python
# save
open(f"{session.id}.json", "w").write(session.model_dump_json(indent=2))

# resume — reload into a fresh runner; status is derived from the entries
session = AgentSession.model_validate_json(open("abc123.json").read())
runner = AgentSessionRunner(
    session,
    tool_registry=SimpleToolRegistry(
        tools=[WeatherTool()], permission_policy=YoloPermissionPolicy(),
    ),
)
```

The tool registry is **not** saved (it's a runtime collaborator); you supply it
again when you reconstruct the runner.

> ⚠️ **Tool specs are stored once per session.** Each `ToolExecution` keeps a
> `tool_spec_id` into `session.tool_specs`; loading restores `tool_spec` from it
> and raises on a file whose executions carry no id. Session files written
> before tool-spec normalization do not load — regenerate them.

Next: [`02-data-model.md`](02-data-model.md).
