# Quickstart

A working agent is four things: an **`AgentSession`** (the state), a **`Tool`**
or two, a **tool registry** that owns them (resolution + approval), and the
**`AgentSessionRunner`** that drives it all. This page builds one, then shows
the loop that powers a real app.

Needs a provider key in the environment (`OPENROUTER_API_KEY` by default) — see
[`../client/01-installation.md`](../client/01-installation.md).

## 1. One tool, one turn

```python
import asyncio
from pydantic import BaseModel, Field
from luca.agent.core import AgentSessionRunner, CancellationToken, Tool, LLMConfig, ToolContext
from luca.agent.core.events import TextBlock, ToolCallReceived, ToolExecuted
from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry, YoloPermissionPolicy

class WeatherArgs(BaseModel):
    city: str = Field(description="City to look up.")

class WeatherTool(Tool):
    name = "get_weather"
    description = "Return the current weather for a city."
    Args = WeatherArgs
    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
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
approval, execution) — the runner only talks to it. `YoloPermissionPolicy`
auto-approves, so a single `run()` drives the whole turn:
model → `get_weather` call → tool result → model → final answer. The turn ends
with status `IDLE`.

## 2. The drive loop

A real app polls the runner's status and reacts. This is the canonical shape —
it handles new input, approval gates, and cancellation resumption uniformly:

```python
while True:
    if runner.idle():
        runner.post_message(input("> "))         # nothing running → take input
    elif runner.awaiting_approval():
        resolve(runner.pending_approvals())       # a gate → answer it (see 05-permissions)
    # PENDING / CANCELLING (and the fall-through from the branches above) → make progress
    async with runner.run() as run:
        async for event in run:
            render(event)
    save(runner.session)                          # persist after every turn
```

Each `run()` advances as far as it can, then stops at the next point that needs
you: the turn finished (`IDLE`) or a tool call needs approval
(`AWAITING_APPROVAL`). See [`04-runner.md`](04-runner.md) for the full status
machine and [`../../main.py`](../../main.py) for a complete REPL.

## 3. Persist and resume

The session *is* the state — save it as JSON, reload it later, keep going:

```python
# save
open(f"{session.id}.json", "w").write(session.model_dump_json(indent=2))

# resume — reload into a fresh runner; it self-heals the status from the entries
from luca.agent.core import AgentSession
session = AgentSession.model_validate_json(open("abc123.json").read())
runner = AgentSessionRunner(
    session,
    tool_registry=SimpleToolRegistry(
        tools=[WeatherTool()], permission_policy=YoloPermissionPolicy(),
    ),
)
```

The tool registry is **not** saved (it's a runtime collaborator); you supply it
again when you reconstruct the runner. Next:
[`02-data-model.md`](02-data-model.md).
