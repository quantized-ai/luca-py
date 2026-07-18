# Simple tool registry

The batteries-included `ToolRegistry`. The core only knows the four-method
contract ([`05-permissions.md`](../../05-permissions.md)); this package
supplies the implementations that cover most applications:
**`SimpleToolRegistry`** (a static tool list gated by one `PermissionPolicy`)
and **`ProxyToolRegistry`** (composition + routing over child registries),
plus the `PermissionPolicy` strategy contract and `YoloPermissionPolicy`.

```python
from luca.agent.contrib.simple_tool_registry import (
    SimpleToolRegistry, ProxyToolRegistry,            # the registries
    PermissionPolicy, YoloPermissionPolicy,           # the approval strategy
)
```

## 1. `SimpleToolRegistry` in 30 seconds

Tools + one policy → a registry the runner drives:

```python
registry = SimpleToolRegistry(tools=TOOLS, permission_policy=YoloPermissionPolicy())
runner = AgentSessionRunner(session, tool_registry=registry)
```

What it does per contract method:

| Method | Behavior |
|---|---|
| `get_tools` | returns the static list |
| `create_execution` | the classic preflight: resolve the name (miss → `NOT_FOUND` birth), validate `Args` (failure → `INVALID` birth), collect the duck-typed approval context (a raise → `FAILED` birth; success → stored under `extras["approval_context"]`), else a `PENDING` birth carrying the tool's `ToolSpec` (incl. `timeout_in_ms`) |
| `decide` | delegates to `permission_policy.decide(execution)` |
| `execute` | re-resolves by the *effective* call's name, re-validates, invokes `tool.execute(args, context, cancellation_token=...)` |

## 2. The policy seam

`PermissionPolicy` is one async hook — see
[`05-permissions.md`](../../05-permissions.md) for the contract, the PENDING
gate, and idempotency; see
[`resource_permissions`](../resource_permissions/README.md) for the
full-featured rule-based implementation:

```python
strategy = PermissionStrategy(mode=PermissionMode.ASK)      # from resource_permissions
registry = SimpleToolRegistry(tools=TOOLS, permission_policy=strategy)
```

## 3. `ProxyToolRegistry` — composition

Concatenate registries; each child keeps its own tools *and* its own approval
policy. `add_registry()` appends a child after construction:

```python
proxy = ProxyToolRegistry(app_registry, plugin_registry)
proxy.add_registry(another)
runner = AgentSessionRunner(session, tool_registry=proxy)
```

Routing: `get_tools` concatenates the children's tools in child order —
duplicate tool names raise `ValueError` — and rebuilds an internal
`{name → child}` route; the other three methods route through it. Nesting
proxies needs nothing special.

On a route **miss** (a name no child claimed), the proxy degrades instead of
guessing: `create_execution` authors a `NOT_FOUND` birth, `decide` allows (so
`execute` produces the honest `NOT_FOUND` terminal rather than a false
`REJECTED`), and `execute` raises `ToolNotFound`.

> ⚠️ **Cold-resume degradation.** The route is warmed by `get_tools` (an LLM
> call). Resuming a gated session in a fresh process and driving it *without*
> any LLM call first means pending calls terminalize as `NOT_FOUND` instead of
> re-asking. Predictable, documented — for now.

Next: [`plugins/README.md`](../plugins/README.md).
