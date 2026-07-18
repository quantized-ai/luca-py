# Resource permissions

A full-featured, rule-based `PermissionPolicy` for
[`SimpleToolRegistry`](../simple_tool_registry/README.md). The core only knows
the `ToolRegistry` contract ([`05-permissions.md`](../../05-permissions.md));
this package supplies everything richer: permission **modes**, an ordered
**rule list** over `(permission, resource)` pairs / tool kinds, interactive
**answers** the user records back onto the strategy, and a typed **tool
mixin** so tools emit the approval-context shape the strategy reads.

```python
from luca.agent.contrib.resource_permissions import (
    PermissionMode, PermissionMatchMode, PermissionStrategy,  # the strategy
    ToolRule, ToolKindRule,                                   # seedable rules
    ApprovalAnswer, AnswerDecision, AnswerScope,              # interactive answers
    ResourcePermissionToolMixin,                              # the tool side
    ResourcePermission, PermissionRequest, AnswerOption,
)
```

## 1. The strategy in 30 seconds

Construct it, gate a registry with it, hand the registry to the runner — and
**keep your own reference**: answers are recorded on the strategy, never
posted through the runner.

```python
from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry

strategy = PermissionStrategy(mode=PermissionMode.ASK)
registry = SimpleToolRegistry(tools=TOOLS, permission_policy=strategy)
runner = AgentSessionRunner(session, tool_registry=registry)
```

Every unresolved call now flows through `strategy.decide()`. Uncovered calls
come back `PENDING` in ASK mode, so the run pauses (`AWAITING_APPROVAL`), you
prompt your user, record the answers (§5), and `run()` again.

## 2. The vocabulary — `ResourcePermission`

Everything speaks one pair: **what** may be done (`permission`) to **what**
(`resource`). Tool requirements, answer options, user grants, and rules all
use the same type — that unification is the model.

```python
ResourcePermission(permission="read", resource="/etc/hosts")
ResourcePermission(permission="add")            # resource=None: resource-less
```

By convention a single-action tool's permission is its name (`"bash"`,
`"edit"`); a tool with distinguishable actions uses verbs (`"read"`,
`"write"`) — that is what lets one grant cover the same action across tools.
A call that declares no pairs carries the implicit resource-less pair
`(permission=<tool_name>, resource=None)`.

## 3. Modes

Two independent knobs on the strategy:

`mode` decides what happens to a pair **nothing resolved**:

| Mode | Uncovered pair becomes |
|---|---|
| `ASK` | `PENDING` — pause and ask the user |
| `YOLO` | `ALLOW` — but an explicit DENY (rule or verdict) still blocks |
| `AUTO` | same as YOLO (reserved for divergence later) |

`permission_mode` decides whether a rule's `tool_name` participates in
matching:

| Match mode | `ToolRule.tool_name` is |
|---|---|
| `RELAXED` (default) | ignored — a `"read"` grant is a read, whichever tool emitted it |
| `STRICT` | required to match when set; `None` still matches any tool |

## 4. Rules

One unified, ordered list; the **last** matching rule wins. Seed it at
construction, grow it with `add_rule()`, or let ALWAYS-scoped answers grow it
(§5). `ToolRule` embeds the pair; its decision is `ALLOW` or `DENY` (never
`PENDING`).

```python
strategy = PermissionStrategy(
    mode=PermissionMode.ASK,
    rules=[
        ToolKindRule(tool_kind=ToolKind.READ, decision=ApprovalOption.ALLOW),  # all reads
        ToolRule(
            resource_permission=ResourcePermission(permission="read", resource="/etc/*"),
            decision=ApprovalOption.DENY,
        ),
    ],
)
```

| Rule | Matches |
|---|---|
| `ToolKindRule(tool_kind, decision)` | every call whose `ToolSpec.tool_kind` matches — resource-agnostic |
| `ToolRule(tool_name, resource_permission, decision)` | a required pair with that exact `permission` touching a matching resource (`tool_name` per the match mode, §3) |

Resource matching is strict about shape: a glob (`"*"`, `"/etc/*"`) matches
only a resourceful pair that fnmatches it; `None` matches only the
resource-less invocation; mixed never matches — so "always allow `add`" is
`ToolRule(resource_permission=ResourcePermission(permission="add"), decision=...)`,
and it won't cover an `add` that later reports a resource.

Whether a call runs is **emergent from coverage**: every required pair must
resolve ALLOW; the per-pair results aggregate with precedence
**DENY > PENDING > ALLOW** — one denied pair denies the call, one uncovered
pair keeps it pending.

## 5. Answering a pending call

Answers are **decoupled from requests**: no ids, no addressing, no replies.
Each `ApprovalAnswer` is a free-standing verdict — a decision × scope over an
`AnswerOption`'s pairs. The tool's suggested `answer_options` are just that,
suggestions; custom-built options are legal and nothing validates membership.

| `decision` \ `scope` | `ONCE` (this execution) | `ALWAYS` (strategy lifetime) |
|---|---|---|
| `APPROVE` | ephemeral ALLOW verdict | ALLOW `ToolRule` per pair |
| `DENY` | ephemeral DENY verdict | DENY `ToolRule` per pair |

```python
if runner.awaiting_approval():
    for execution in runner.pending_approvals():
        requests = strategy.pending_requests(execution)     # only uncovered steps
        answers = ask_user(execution, requests)             # your UI, any verdicts
        strategy.apply_answer(execution, answers)
async with runner.run() as run:                             # re-asks decide()
    async for event in run:
        render(event)
```

Two hydration queries, one difference: `permission_requests(execution)`
returns every stored request; `pending_requests(execution)` filters them to
the pairs `decide()` would leave PENDING (a fully covered request drops out,
a partially covered one keeps only its unresolved pairs). Prompt with
`pending_requests` so rule-covered steps — e.g. an auto-granted workspace
`access_directory` — stay silent.

Answers accumulate across `apply_answer()` calls (all at once or one per
call — identical result). `add_rule(tool_name, resource_permission, decision)`
appends a rule directly, deduping identical ones; ALWAYS-scoped answers go
through it, always recording the calling tool's name so the rule survives a
later switch to STRICT.

> ⚠️ **Verdicts precede rules.** In `decide()` an execution's own ONCE
> verdicts win over rules (and DENY beats ALLOW among them) — so a call the
> user explicitly denied stays denied even when a sibling's ALWAYS answer
> wrote a rule that covers it in the same pause. Unanswered siblings ARE
> cleared by new rules; only answered calls are pinned.

> ⚠️ **The approval loop.** Recording answers does not advance the runner —
> the session stays `AWAITING_APPROVAL` until the next `run()` asks
> `decide()` again. An answer that doesn't cover every required pair (e.g. a
> glob that doesn't match) leaves the call `PENDING` and you'll be asked
> again — the failure mode is a re-ask, never a false approval.

## 6. The tool side — `ResourcePermissionToolMixin`

Tools declare *what they need* and *what they suggest*: mix in the mixin and
return an ordered list of `PermissionRequest`s from the one override point,
`build_permission_requests(args, context)` (it receives the validated args).
Most tools return one request:

```python
class ReadFileTool(ResourcePermissionToolMixin, Tool):
    name = "read_file"
    description = "Read a file from disk and return its contents."
    Args = ReadFileArgs
    tool_kind = ToolKind.READ

    def build_permission_requests(self, args, context):
        return [PermissionRequest(
            resources=[ResourcePermission(permission="read", resource=args["path"])],
            answer_options=[
                AnswerOption(
                    resource_permissions=[
                        ResourcePermission(permission="read", resource="/etc/*"),
                    ],
                    metadata={"preview": "Approve all reads in /etc/*"},
                ),
            ],
            metadata={"preview": f"Read {args['path']}"},
        )]
```

A tool that performs several distinguishable actions returns one request per
action, in presentation order — but requests are presentation grouping only:
`decide()` flattens all their pairs into one required set.

`metadata` (on requests and options) is **UX-only** — previews, labels — and
is never read by the strategy. `ResourcePermission` itself carries no
metadata, so rules stay free of UX baggage and pair equality is pure.

The mixin's `get_approval_context()` — the duck-typed convention
`SimpleToolRegistry` reads (`Tool` itself declares no such method) —
serializes the requests to the wire dict stored under
`extras["approval_context"]`:

```json
{"requests": [
  {"resources": [{"permission": "read", "resource": "/src/main.py"}],
   "answer_options": [
     {"resource_permissions": [{"permission": "read", "resource": "/src/*"}],
      "metadata": {"preview": "Approve read src/*"}}],
   "metadata": {"preview": "Read /src/main.py"}}
]}
```

Applications never touch the raw dicts — `strategy.permission_requests()`
hydrates them back into typed models for your approval UI. It is the tool
author's responsibility to emit options that cover their own requirements.

## 7. How `decide()` resolves a call

For each required pair (every request's pairs flattened; a request with
empty `resources` — or a call with no requests — contributes the implicit
resource-less pair):

1. **Ephemeral verdicts** recorded for this execution id (`via: "user"`);
   DENY beats ALLOW among matches. Verdict matching uses the rule-resource
   algorithm (exact permission, fnmatch resource, None↔None) but ignores
   `tool_name` — verdicts are execution-scoped; the tool is implied.
2. **Rules**, last match wins (`via: "rule"` / `"kind_default"`), honoring
   `permission_mode`.
3. **Mode**: ASK leaves PENDING; YOLO / AUTO promote a residual PENDING to
   ALLOW (`via: "mode"`); explicit DENY always blocks.

The per-pair results aggregate DENY > PENDING > ALLOW.

The full interactive flow — approval prompt, suggested-grant menu, rule
creation — is runnable in [`main.py`](../../../../main.py).

Next: [`shell/README.md`](../shell/README.md).
