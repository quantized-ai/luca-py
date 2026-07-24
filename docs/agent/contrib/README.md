# `luca.agent.contrib` — packages beyond the core

The boundary is sharp: [`luca.client`](../../client/README.md) is the LLM SDK,
`luca.agent.core` is the agent core (data model, runner, strategy contracts) —
**everything else ships here**. A contrib package consumes only the public
`luca.agent.core` surface, exactly like your application code would; nothing in
the core imports from contrib (contrib→contrib dependencies are allowed —
`plugins` builds on `simple_tool_registry`). Each package is optional: ignore
it and write your own, or import it and go.

```python
from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry
```

## Packages

| Package | Topic |
|---|---|
| [`simple_tool_registry/`](simple_tool_registry/README.md) | The batteries-included `ToolRegistry`: a static tool list + `PermissionPolicy`, and `ProxyToolRegistry` for composition |
| [`plugins/`](plugins/README.md) | `BasePlugin` + `PluginAgentSessionRunner` — install a capability (registry + prompt parts + middleware) in one move |
| [`resource_permissions/`](resource_permissions/README.md) | Rule-based tool approval — modes, resource globs, answer-decoupled grants, and a typed tool mixin |
| [`shell/`](shell/README.md) | The seven shell tools (read/glob/grep/edit/write/apply_patch/bash) + `ShellAccessPlugin` — workspace-scoped, two-step directory permissions |
| [`tui/`](tui/README.md) | The Textual terminal UI — transcript, streaming, modal approvals, cancellation; the runnable demo behind `main.py` |
| [`compaction/`](compaction/README.md) | Summarize a long session into a fresh one — pluggable `CompactionStrategy`, the `Compactor` gauge + operation, wired into the TUI as auto-compact + `/compact` |
| `memory` | An in-memory scratchpad + todo list packaged as `MemoryPlugin` — documented in [`09-plugins.md`](../09-plugins.md) |

Next: [`simple_tool_registry/README.md`](simple_tool_registry/README.md).
