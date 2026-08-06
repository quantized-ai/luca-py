# Shell access

`luca.agent.contrib.shell` is the filesystem/process tool suite — seven tools
modeled on Claude Code behavior, some of which a provider may define itself
(§6) — plus `ShellAccessPlugin`, which bundles them behind one workspace
directory with a seeded, resource-aware permission strategy (built on
[`resource_permissions`](../resource_permissions/README.md)).

## 1. The plugin in 30 seconds

```python
from pathlib import Path

from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.shell import ShellAccessPlugin
from luca.agent.core import AgentSessionRunner, LLMConfig

shell = ShellAccessPlugin(workspace=Path("."))
session = AgentSessionRunner.new_session(
    LLMConfig(model="openai/gpt-4o-mini", provider="openrouter"),
)
runner = PluginAgentSessionRunner(session, plugins=[shell])
```

One plugin instance owns what the tools can't wire individually: the
workspace every tool resolves relative paths against, ONE shared
`FileReadTracker` (the read-first contract below), and one
`PermissionStrategy` exposed as `shell.permission_strategy` — feed your
approval prompt from it, or hand it to your own registries so a single
strategy gates everything (see [`main.py`](../../../../main.py)).

`workspace` and `additional_directories` are stored **absolute** at
construction (cwd-anchored, no symlink resolution — the same convention the
tools use), so grants keep their meaning across resumed sessions.

## 2. The tools

| Tool | Kind | Does |
|---|---|---|
| `read` | READ | Numbered text pages, directory listings, images as real `ImageContent` (PDFs still a stub); caps at 2000 lines / 50 KiB |
| `glob` | SEARCH | ripgrep-backed file finding under a root |
| `grep` | SEARCH | ripgrep-backed content search, grouped per file |
| `edit` | EDIT | Unique exact replacement with fuzzy-correction strategies; unified diff in metadata |
| `write` | EDIT | Full-content write, creates parents |
| `apply_patch` | EDIT | `*** Begin/End Patch` envelope; verify-everything-then-commit |
| `bash` | EXECUTE | Fresh shell per call, streamed, timeout + cancellation kill the process group |

Domain failures (missing file, ambiguous edit, non-zero exit) come back as
`ExecutionResult(is_error=True)` — never exceptions to the runner.

## 3. The two-step permission model

Every call declares two ordered approval steps through
`build_permission_requests(args, session, conversation_id)`:

```
access_directory <directory the call touches>   # step 1
<verb> <resource>                               # step 2: read /ws/tests.py, bash "git status", …
```

The plugin seeds ALLOW rules over the workspace and each additional
directory (the root itself plus `<root>/*` — fnmatch `*` crosses `/`, so
one glob covers every depth). In ASK mode the seeded rules cover the whole
**read tier** — `access_directory`, `read`, `glob`, `grep` — so:

| Call | Prompts for |
|---|---|
| `read tests.py` (inside the workspace) | nothing — fully covered |
| `edit tests.py` (inside) | the `edit` step only |
| `read ../../secrets.txt` (outside) | both steps |

Prompt with `pending_requests()` so covered steps stay silent:

```python
strategy = shell.permission_strategy
for execution in runner.pending_approvals():
    steps = strategy.pending_requests(execution)  # only the uncovered steps
    strategy.apply_answer(execution, ask_user(execution, steps))
```

Each `access_directory` step suggests one answer option per directory —
"Always allow access to `<dir>`", granting `[<dir>, <dir>/*]`.

> ⚠️ **A gate, not a sandbox.** Approval is the only containment: an
> approved `bash` command can touch any path regardless of its workdir, and
> YOLO mode is full-disk for all seven tools. A `bash` call also runs with the
> process's own cwd and environment, which every conversation shares — with
> subagents running in parallel, one `cd` is not a private one.

## 4. Modes

`mode="ask"` (default) — anything the rules don't cover comes back PENDING and
parks the call for your approval prompt. `mode="yolo"` — everything is allowed
(explicit DENY rules added to the strategy still block).

## 5. The read-first contract

`read` records every text file it returns on the plugin's `FileReadTracker`;
`edit` and `write` refuse to mutate an existing file that was never recorded
(and record their own writes). This is why the tracker must be shared — the
plugin owns that wiring; if you construct tools standalone, pass one
`tracker=` instance to read/edit/write yourself:

```python
from luca.agent.contrib.shell import EditTool, FileReadTracker, ReadTool

tracker = FileReadTracker()
tools = [ReadTool(workdir="/ws", tracker=tracker), EditTool(workdir="/ws", tracker=tracker)]
```

The tracker is keyed **by conversation**, and that is a safety property rather
than tidiness: one plugin instance serves the main agent and every subagent
([`13-subagents.md`](../../13-subagents.md)), so unkeyed, subagent A reading
`main.py` would satisfy the guard for subagent B, which never read it.

```python
tracker.was_read("c_main", "/ws/main.py")     # True  — the main agent read it
tracker.was_read("c_child", "/ws/main.py")    # False — this subagent did not
```

> ⚠️ **Two subagents can write the same file at the same time.** The tools take
> a per-path lock around the write itself, so neither sees a half-written file —
> but "who wins" is still whoever went last. The permission gate is per pair,
> not per conversation; if a task must not be done twice, do not spawn it twice.

## 6. Provider-defined tools

Anthropic and OpenAI define some of these tools themselves and train the model
on the schema. When the session's model routes to one of them, the plugin
installs the provider's version instead of luca's:

| Route | Swapped in | Replaces |
|---|---|---|
| `anthropic` + a Claude | `str_replace_based_edit_tool` | `read`, `edit`, `write` |
| `anthropic` + a Claude | `bash` (one persistent shell) | `bash` |
| `openai` + a GPT-5.x | `apply_patch` | `apply_patch` |
| `openai` + a GPT-5.x | `shell` (one persistent shell) | `bash` |

`glob` and `grep` have no provider equivalent and always stay. The editor
REPLACES the three file tools rather than joining them: offering both asks the
model to choose between two tools that do one job.

Turn it off per project:

```json
{"tools": {"native": false}}
```

Selection keys on the TRANSPORT, never the model family. Bedrock and OpenRouter
both serve Claude and neither speaks the Messages API, so a "is this a Claude?"
check would send a tool the endpoint rejects.

### The persistent shell

Both providers specify their run tool as ONE session: the working directory,
environment variables and background processes survive between calls, and a
`restart` starts clean. luca's own `bash` is a fresh subprocess per call, so
this is a different tool rather than a rename. One shell per conversation, or
a subagent's `cd` would relocate the main agent's next command.

> ⚠️ **A cancel or a timeout resets the session.** The command runs IN the
> shell, so there is nothing to kill that leaves the shell standing. Both
> return whatever the command printed and say the working directory and
> environment are back to their defaults — the silent version is what makes the
> NEXT command land somewhere the model did not choose.

### Extending it

`ShellAccessPlugin.native_key_for` decides which provider tools a route gets;
`install_tools` composes the list. Both are public, and both are re-consulted
per request, so `/model` mid-session rebuilds the set:

```python
class MyPlugin(ShellAccessPlugin):
    def native_key_for(self, session):
        # (editor type, bash type, OpenAI types)
        return ("text_editor_20250728", None, ())
```

See [`docs/client/06-tools.md`](../../../client/06-tools.md#provider-defined-tools)
for what `provider_type` does on the wire.

Next: [`tui/README.md`](../tui/README.md).
