# `luca.agent.contrib.skills` — `SKILL.md` instruction sets

A **skill** is a directory holding a `SKILL.md`: YAML frontmatter with a
`name` and a `description`, then a body of instructions. Drop one on disk and
the agent can load it when the work matches.

```
.claude/skills/
└── greeting/
    ├── SKILL.md
    └── references/
        └── tone.md          # bundled files, read on demand
```

```markdown
---
name: greeting
description: How to greet someone. Use when opening a conversation.
---

Always say hello twice. See references/tone.md for register.
```

## Progressive disclosure

The model does **not** see skill bodies. The system prompt carries one line per
skill — name and description — and a `skill` tool returns the body only when the
model decides it is relevant:

```
### Skills
Skills are instruction sets for specific kinds of work. Load one with the
`skill` tool when its description matches what you are doing, and follow it.
Available:
- greeting: How to greet someone. Use when opening a conversation.
```

That matters because the alternative scales badly: two real skills in this repo
are ~8k tokens, paid on every request of every conversation including each
subagent's, for material that is usually irrelevant to the turn.

The tool's answer names the skill's directory, so a body that says "see
references/tone.md" is actionable.

## Where skills are read from

In precedence order, first match winning a name collision:

| Location | Scope |
|---|---|
| `<workspace>/.claude/skills/` | project, Claude-compatible |
| `<workspace>/.agents/skills/` | project, Agent-compatible |
| `extra_skill_locations` from `luca.json` | configured |
| `~/.claude/skills/` | global, Claude-compatible |
| `~/.agents/skills/` | global, Agent-compatible |

A repo's skill therefore overrides a personal one of the same name. Extras sit
between the two tiers: more deliberate than a machine-wide default, less
specific than a file committed in the project.

Project roots are anchored at the **workspace**, the same directory the shell
tools scope to, not the process cwd.

```jsonc
// luca.json
{
  "extra_skill_locations": [".opencode/skills/", "~/.config/opencode/skills/"]
}
```

## Bundled files are readable without a prompt

`build_runner` hands the roots that actually held a skill to `ShellAccessPlugin`
as additional directories, so `read` / `glob` / `grep` over them are pre-allowed
and a global skill's `references/*.md` opens with no approval modal. Only the
read tier: `write`, `edit` and `bash` over the same paths still gate normally.

## Nothing here is fatal

A malformed `SKILL.md` is skipped, never raised. A missing `description` (the
only thing the model sees before choosing), a header that is not a mapping, an
unterminated `---` fence, invalid YAML, an unreadable file: each drops that one
directory and leaves the rest loading. A forgotten directory must not stop the
agent starting.

`name` falls back to the directory name, which is how skills are addressed
anyway.

## Using it

```python
from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.skills import SkillsPlugin

runner = PluginAgentSessionRunner(
    session,
    plugins=[SkillsPlugin(workspace=".", extra_locations=["./vendor/skills"])],
)
```

In the TUI it is on by default; `--no-skills` withholds the tool and the prompt.

Needs the `skills` dependency group (PyYAML). It is a default group, so
`uv sync` installs it — real skills use YAML block scalars (`description: >`,
`description: |`) that a hand-rolled parser would silently mangle.

## Surface

| Name | What |
|---|---|
| `SkillsPlugin` | The plugin: `get_tool_registry` + `get_system_prompt_parts`, plus `skills` and `skill_directories` |
| `SkillTool` | The `skill` tool — one `name` argument, returns body + directory |
| `Skill` | A frozen record: `name`, `description`, `path`, `directory`, `body` |
| `discover_skills` / `load_skill` | Scan roots / read one directory |
| `resolve_locations` / `default_locations` | The precedence list above |
| `parse_frontmatter` | `---` fenced YAML → `(mapping, body)` |

## Not implemented

`allowed-tools` in frontmatter is parsed and ignored — restricting a skill's
tool access is a separate capability. Skills are also not injected into subagent
prompts, and there is no slash-command invocation.
