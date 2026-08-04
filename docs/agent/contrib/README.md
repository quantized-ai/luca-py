# `luca.agent.contrib` — packages beyond the core

The boundary is sharp: [`luca.client`](../../client/README.md) is the LLM SDK,
`luca.agent.core` is the agent core (data model, runner, strategy contracts) —
**everything else ships here**. A contrib package consumes only the public
`luca.agent.core` surface, exactly like your application code would; nothing in
the core imports from contrib (contrib→contrib dependencies are allowed —
`plugins` builds on `simple_tool_registry`, which builds on `tools`). Each
package is optional: ignore it and write your own, or import it and go.

```python
from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry
from luca.agent.contrib.tools import Tool
```

## Packages

| Package | Topic |
|---|---|
| [`tools/`](tools/README.md) | `Tool` + the `tool()` / `tool_class()` factories — the ergonomic way to write a Python tool; the core itself only knows `ToolSpec` |
| [`simple_tool_registry/`](simple_tool_registry/README.md) | The batteries-included `ToolRegistry`: a static tool list + `PermissionPolicy`, and `ProxyToolRegistry` for composition |
| [`plugins/`](plugins/README.md) | `BasePlugin` + `PluginAgentSessionRunner` — install a capability (registry + prompt parts + middleware) in one move |
| [`resource_permissions/`](resource_permissions/README.md) | Rule-based tool approval — modes, resource globs, answer-decoupled grants, and a typed tool mixin |
| [`shell/`](shell/README.md) | The seven shell tools (read/glob/grep/edit/write/apply_patch/bash) + `ShellAccessPlugin` — workspace-scoped, two-step directory permissions |
| [`tui/`](tui/README.md) | The Textual terminal UI, built as a design system — declarative screen states, a fixture gallery, snapshot tests; the runnable demo behind `main.py` |
| [`subagents/`](subagents/README.md) | The two subagent tools + `SubagentsPlugin` — parallel subagents, gated by depth and budget ([`13-subagents.md`](../13-subagents.md)) |
| [`skills/`](skills/README.md) | `SKILL.md` instruction sets discovered from the Claude/Agent locations + `SkillsPlugin` — name and description in the prompt, body loaded on demand |
| [`prompts/`](prompts/README.md) | `SystemPromptPlugin` (a base prompt picked for the model family, plus an environment block) and `InstructionsPlugin` (`LUCA.md` / `AGENTS.md` / `CLAUDE.md`) |
| [`simple_context_manager/`](simple_context_manager/README.md) | `SummarizingContextManager` — a ready-made compacting `ContextManager` (the context gauge + an LLM summary), with a `keep_turns` knob |
| `memory` | An in-memory scratchpad + todo list packaged as `MemoryPlugin` — documented in [`09-plugins.md`](../09-plugins.md) |

Next: [`tools/README.md`](tools/README.md).
