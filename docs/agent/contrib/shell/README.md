# Shell access

`luca.agent.contrib.shell` is the filesystem/process tool suite — seven tools
modeled on Claude Code behavior — plus `ShellAccessPlugin`, which bundles
them behind one workspace directory with a seeded, resource-aware permission
strategy (built on
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
| `read` | READ | Numbered text pages, directory listings, image/PDF attachments; caps at 2000 lines / 50 KiB |
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
`build_permission_requests`:

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
> YOLO mode is full-disk for all seven tools.

## 4. Modes

`mode="ask"` (default) — anything the rules don't cover comes back PENDING
and pauses the runner for your approval prompt. `mode="yolo"` — everything
is allowed (explicit DENY rules added to the strategy still block).

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

Next: [`tui/README.md`](../tui/README.md).
