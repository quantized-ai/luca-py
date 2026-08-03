# `luca.agent.contrib.prompts` — the base prompt and the project's rules

Two plugins, both contributing [system prompt parts](../../06-system-prompts.md)
and nothing else. No extra dependencies.

| Plugin | Contributes |
|---|---|
| `SystemPromptPlugin` | the base coding-agent prompt, an addendum for the model's family, and an environment block |
| `InstructionsPlugin` | the project's `LUCA.md` / `AGENTS.md` / `CLAUDE.md` |

They are separate because the reasons to want them are separate: an application
with its own persona still wants a project's rules.

```python
from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.prompts import InstructionsPlugin, SystemPromptPlugin

runner = PluginAgentSessionRunner(
    session,
    tool_registry=registry,
    plugins=[SystemPromptPlugin(workspace="."), InstructionsPlugin(workspace=".")],
)
```

## 1. The base prompt

One shared prompt plus a short per-family addendum, rather than a complete
prompt per family — the families differ in a handful of known behaviours, and
near-identical copies drift apart.

The family comes from the **model id**, lowercased, first match wins. Never the
provider: openrouter serves every family, so `provider="openrouter"` says
nothing useful.

| Family | Matches | Addendum covers |
|---|---|---|
| `anthropic` | `claude` | brevity, no preamble, parallel tool calls |
| `gpt` | `gpt-`, `o1-`, `o3-`, `codex` | persistence, read rather than guess, minimal diffs |
| `gemini` | `gemini` | no whole-file dumps, no plan restatement, no drive-by refactors |
| `generic` | everything else | a conservative middle ground |

```python
from luca.agent.contrib.prompts import load_prompt, select_family

select_family("anthropic/claude-sonnet-5")   # "anthropic"
select_family("moonshotai/kimi-k2.7-code")   # "generic"
load_prompt("anthropic")                     # the addendum text
```

Adding a family is one row in `FAMILIES` and one `text/<family>.md`.

Both parts are **callables**, so a `/model` switch mid-session moves the prompt
with it rather than leaving the session tuned for the model it started on.

Every part builder is a public method, so the extension model is "subclass and
override one":

```python
class HousePrompt(SystemPromptPlugin):
    def family_part(self, session, conversation_id):
        return SystemPromptPart(text=OUR_OWN_TUNING, source="house")
```

## 2. The environment block

```
### Environment
You are powered by the model claude-sonnet-5 on the anthropic provider.
Working directory: /Users/me/project
Is a git repository: yes
Platform: Darwin
Today's date: 2026-08-03
```

`format_environment()` is pure — the date, platform and git verdict are all
arguments — so it is testable against a literal. The plugin resolves the git
question once at construction, not once per model call. Pass
`environment=False` to withhold the block.

## 3. Instruction files

Three tiers, concatenated **least specific first**, so the file nearest the
workspace is read last and wins by recency.

| Tier | Where | Rule |
|---|---|---|
| global | `~/.config/luca/` (or `$XDG_CONFIG_HOME/luca`) | one file |
| project | the git root **down to** the workspace | one file per directory |
| extra | the `instructions` list from `luca.json` | in the order listed |

Within any one directory the name precedence is `LUCA.md` → `AGENTS.md` →
`CLAUDE.md`, and **only the winner is read**. Precedence is applied per
directory, not once for the whole tree, so a repo root's `AGENTS.md` and a
subpackage's `CLAUDE.md` both contribute.

That per-directory rule also handles the common interop file. Claude Code reads
`CLAUDE.md`, not `AGENTS.md`, so repos supporting both often ship a `CLAUDE.md`
containing nothing but `@AGENTS.md`. Here `AGENTS.md` is checked first in that
same directory, so the one-line pointer is never what gets read, and no import
expansion is needed.

**Without a git repository the walk is the workspace directory alone.** An
unbounded walk upward reaches in from outside the project entirely.

A **discovered** file is read leniently: unreadable, undecodable or empty, it
is skipped and the rest still load. A file the config **names** is not. A path
in `instructions` that does not resolve to a readable file raises
`InstructionsError`, so a typo fails loudly instead of quietly contributing
nothing. The TUI turns that into `luca: <path>: not a readable instruction
file` and exit 1, the same as any other bad config value.

### The budget

The combined text rides on every request of every conversation in the session,
subagents included, so it is capped at `MAX_INSTRUCTION_BYTES` (32 KiB).

The budget is filled from the **most specific end**: when it runs out, the
global file is dropped and the project's own rules survive. The most specific
file is always kept, however large. Codex fills its equivalent budget in lookup
order instead, which lets a bloated personal file starve the repo's rules.

```python
from luca.agent.contrib.prompts import find_instructions

find_instructions(".", ["docs/conventions.md"])
# [InstructionFile(path=PosixPath('/repo/AGENTS.md'), text='...'), ...]
```

## 4. Where the parts land

Every other plugin leaves `priority` at its `-1` default, so the two positive
priorities here pin the tail of the assembled prompt regardless of plugin order.
Install `SystemPromptPlugin` first and the result reads persona → model tuning →
capabilities → environment → project rules:

| # | Part | `source` | `priority` |
|---|---|---|---|
| 1 | base persona | `prompt` | -1 |
| 2 | family addendum | `prompt.<family>` | -1 |
| 3-6 | the tool plugins' blurbs | `model` / `skills` / … | -1 |
| 7 | environment | `env` | 90 |
| 8 | project instructions | `agents.md` | 100 |

`instructions_part` is public too, so a subclass can reframe the files without
reimplementing discovery.

## 5. In the TUI

`build_runner` installs both. `--no-instructions` withholds the project rules;
`instructions` in [`luca.json`](../tui/config.md) names extra files.

Next: [`simple_context_manager/README.md`](../simple_context_manager/README.md).
