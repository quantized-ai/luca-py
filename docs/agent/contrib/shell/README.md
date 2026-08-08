# Shell access

`luca.agent.contrib.shell` is the filesystem/process tool suite — eight tools
modeled on Claude Code behavior, plus four
[provider-native ones](#6-provider-native-tools) — and `ShellAccessPlugin`,
which bundles them behind one workspace directory with a seeded,
resource-aware permission strategy (built on
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
| `delete_file` | DELETE | Removes one file (a symlink, not its target); refuses directories |
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
| `delete_file tests.py` (inside) | the `delete_file` step only |
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
> YOLO mode is full-disk for every tool. A `bash` call also runs with the
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

`delete_file` is deliberately outside the contract: `read` refuses binaries,
so a read-first guard would make a stray `.pyc`, archive or screenshot
undeletable — and having read 2000 lines of a file says nothing about all of
it being destroyed. Its containment is the approval step, which (unlike
`read`/`glob`/`grep`) the plugin never auto-allows.

> ⚠️ **Two subagents can write the same file at the same time.** The tools take
> a per-path lock around the write itself, so neither sees a half-written file —
> but "who wins" is still whoever went last. The permission gate is per pair,
> not per conversation; if a task must not be done twice, do not spawn it twice.

## 6. Provider-native tools

OpenAI and Anthropic ship their own file-editing and shell tools, declared by
TYPE rather than as a function the model has to be taught. The plugin owns four
of them, and offers each one only to a model that actually supports it:

| Internal name | On the wire | Replaces |
|---|---|---|
| `openai_apply_patch` | `{"type": "apply_patch"}` | `edit` `write` `apply_patch` `delete_file` |
| `openai_shell` | `{"type": "shell", "environment": {"type": "local"}}` | `bash` |
| `anthropic_text_editor_20250728` | `{"type": "text_editor_20250728", …}` | `edit` `write` `apply_patch` |
| `anthropic_bash_20250124` | `{"type": "bash_20250124", …}` | `bash` |

So one workspace, three tool sets:

```
openai + native      apply_patch  shell                         + read glob grep
anthropic + native   str_replace_based_edit_tool  bash          + read glob grep delete_file
anything else        read glob grep edit write apply_patch delete_file bash
```

`read`, `glob` and `grep` survive everywhere — nothing native returns an image
or searches a repo — and `delete_file` survives on Anthropic, whose text editor
cannot delete.

It is **off by default in the library** and on in the TUI:

```python
session.session_config.use_native_tools = True   # or --no-use-native in the TUI
```

Support is per MODEL, not per provider. `gpt-5.5-pro` has the native shell and
no native `apply_patch`, so it keeps every generic editor; `gpt-5.4-pro` is the
mirror image. An OpenAI model reached through OpenRouter gets none of them —
the native items only exist on OpenAI's own Responses wire. An unrecognised
model simply gets the generic tools, never an error.

Three things this costs you nothing to know but are worth knowing:

- **Switching providers mid-session is safe.** Every call is stored under its
  internal name and replays as an ordinary function call on any provider —
  both accept a name they were not offered. Switch back and the original
  native payload is rebuilt byte for byte from storage. A call left waiting
  for approval under one provider still executes under another.
- **Approvals carry over.** A native tool asks for the generic tool's verb
  (`apply_patch`, `bash`, and `read`/`write`/`edit` for the text editor's four
  commands), so a rule or an "always allow" answer keeps meaning across a
  switch. The text editor's `view` lands in the auto-allowed read tier, so
  reading inside the workspace stays silent.
- **Uninstalling the plugin breaks nothing.** The provider-blind form is the
  DEFAULT, not a fallback: with no plugin and no middleware, a session that ran
  native still projects and still runs.

The native tools keep the generic ones' guarantees, deliberately: a file edited
through `apply_patch` or `text_editor` keeps its BOM and its CRLF line endings
exactly as `edit` would leave them; overwriting a whole existing file still
requires having read it first (`text_editor view` counts as the read); and
shell output is still capped at 2000 lines / 50 KiB per command, with the rest
spilled to a file the output names.

Next: [`tui/README.md`](../tui/README.md).
