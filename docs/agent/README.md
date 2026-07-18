# `luca.agent` — the agent framework

A durable, resumable AI agent built around one serializable object: the
**`AgentSession`**. It records the whole conversation — messages, tool calls,
tool results, reasoning, turn boundaries — and reloads to resume exactly where
it stopped. An async **runner** drives it. The tool registry and the system
prompt are **strategies you hand to the runner** — never stored in the session.

> **session = data. runner = behavior.** The session is pure Pydantic and
> round-trips through JSON losslessly. Everything transient (the live model
> call, the tool registry, cancellation) lives on the runner.

Everything the core knows lives in `luca.agent.core`; the batteries-included
registry lives in contrib:

```python
from luca.agent.core import AgentSessionRunner, Tool, LLMConfig
from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry, YoloPermissionPolicy
```

## 60-second agent

```python
import asyncio
from pydantic import BaseModel
from luca.agent.core import AgentSessionRunner, CancellationToken, Tool, LLMConfig, ToolContext
from luca.agent.core.events import TextBlock, ToolExecuted
from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry, YoloPermissionPolicy

class AddArgs(BaseModel):
    a: float
    b: float

class AddTool(Tool):
    name = "add"
    description = "Add two numbers and return the sum."
    Args = AddArgs
    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        return str(args["a"] + args["b"])

async def main() -> None:
    session = AgentSessionRunner.new_session(
        LLMConfig(model="openai/gpt-4o-mini", provider="openrouter"),
    )
    registry = SimpleToolRegistry(tools=[AddTool()], permission_policy=YoloPermissionPolicy())
    runner = AgentSessionRunner(session, tool_registry=registry)
    runner.post_message("What is 21 + 21?")
    async with runner.run() as run:            # drive the turn, render events
        async for event in run:
            match event:
                case ToolExecuted(execution=ex, result_text=text):
                    print(f"[{ex.raw_tool_call.name}] -> {text}")
                case TextBlock(text=text): print(text)

asyncio.run(main())
```

## Pages

Read top to bottom; each page starts simple and deepens.

| Page | Topic |
|---|---|
| [`01-quickstart.md`](01-quickstart.md) | A full runnable agent + the drive loop |
| [`02-data-model.md`](02-data-model.md) | `AgentSession`, entries, turns, serialize / resume / fork |
| [`03-tools.md`](03-tools.md) | Define a tool → context → rich results → cancellation → the registry |
| [`04-runner.md`](04-runner.md) | `run()`/`start()`, events, streaming, async, cancel, the status machine |
| [`05-permissions.md`](05-permissions.md) | The `ToolRegistry` contract — approval as a registry concern |
| [`06-system-prompts.md`](06-system-prompts.md) | System-prompt parts (static or callable) + the assembler |
| [`07-middleware.md`](07-middleware.md) | Intercept the pipeline at 10 hook points |
| [`08-runtime-config.md`](08-runtime-config.md) | Timeouts, step limits, doom-loop detection |
| [`09-plugins.md`](09-plugins.md) | Plugins — bundle a registry + prompt parts + middleware in one object (contrib) |
| [`10-projection.md`](10-projection.md) | `ConversationProjector` — own the LLM message history and tool-output wording |
| [`11-context-and-usage.md`](11-context-and-usage.md) | `context_tokens`, the usage store, pruning — and the `ContextManager` seam to improve them |
| [`contrib/`](contrib/README.md) | Optional packages built on the core — registries, plugins, rule-based resource permissions |

The agent talks to models through [`luca.client`](../client/README.md); you never
call it directly. Install is the single `luca-ai` package — see
[`../client/01-installation.md`](../client/01-installation.md).
