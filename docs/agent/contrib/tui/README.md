# TUI

A [Textual](https://textual.textualize.io/) terminal UI for the agent loop,
built as a **design system**: a full-screen app with one scrolling
conversation column, an inline approval prompt, an inline question set (the
agent's `ask_user`), overlay lists (command palette, context picker), and modal
screens for sessions, settings and cost.
Every visual state is expressible as declarative data (`state.ScreenState`),
renderable without a live agent (the gallery), and snapshot-tested. It is
also the reference for wiring a real interactive app on top of the runner:
one drive worker, one shared
[`PermissionStrategy`](../resource_permissions/README.md), sessions saved as
`~/.luca/projects/<encoded-project-path>/<session-id>.json` — with the TUI's
own state (outstanding questions) in a `<session-id>.tui.json` sidecar beside
it. Requires the
`tui` dependency group (installed by default with `uv sync`).

```python
from luca.agent.contrib.tui import AgentApp, GalleryApp, LucaApp, build_runner, main
```

## 1. Run it

```bash
uv run python main.py                     # fresh session (needs auth.json or OPENROUTER_API_KEY)
uv run python main.py --faux              # offline scripted demo — no key, no network
uv run python main.py --resume            # pick a past session for this project
uv run python main.py --resume <id>       # resume that session (--fork to branch)
uv run python main.py --gallery           # the design-system gallery — no agent at all
uv run python main.py --no-streaming      # block-level events instead of deltas
uv run python main.py --no-subagents      # stop it spawning subagents that work in parallel
uv run python main.py --model moonshotai/kimi-k2.7-code --reasoning high
uv run python main.py --config ./ci.json  # use THIS config file, skip discovery
uv run python main.py --resume <id> --pretty-print  # transcript, then exit
```

`--model` / `--provider` / `--reasoning` update the session's `LLMConfig`
(`--reasoning` lands in its `model_options`); they persist with the session and
override the stored values on a resume. The provider's api key does not — it is
read from `auth.json` at every boot and handed to the runner
([config.md](config.md#credentials)).
Subagent flags (`--subagents-max-depth`, `--subagents-max-per-turn`,
`--subagents-max-workers`) and `--pretty-print` behave as before — see
[`08-runtime-config.md`](../../08-runtime-config.md) and
[`config.md`](config.md) for the `luca.json` equivalents.

> ⚠️ **Old session files do not load.** A `<id>.json` written before tool-spec
> normalization fails validation on `--resume` and on `--pretty-print`.
> There is no migration — start a fresh session.

## 2. What's on screen

One `Vertical`: a custom status bar (`◧ luca · cwd · model · branch* ·
tokens · $cost`), the single rule, the scrolling transcript, the dock
(composer OR approval prompt OR question set OR overlay list), and the hint
legend — which is
always visible and always reflects the focused context. No stock
Header/Footer, no panel borders; blocks are separated by blank rows.

Transcript block vocabulary (`blocks.py`, rendered from `state.py` models):

| Block | Rendering |
|---|---|
| user | `›` in accent, then the message (images as `[image: name]` lines) |
| thinking | `∴ thought for 3s` collapsed; `∴ <activity>` while in progress |
| text | assistant prose at an 84-column measure; `` `code spans` `` in accent; streaming ends with a block cursor |
| tool | `▸ name arg` header (6-cell name column), then the result in a `1+│+2` gutter: a faint summary (`84 lines`, `exit 0 · removed 12 paths`), a diff stat (`+7 −1`), or verbatim output with `^o expand`. Denied calls turn `▸` and `denied` to `$error`; **every automatic permission decision is stated** (`approved by rule`, `denied · by rule`) |
| list | active skills / context set / consumers, the answered question set (§2.2), and the docked plan panel (§2.3): a faint label row + glyph rows (`☑ ◉ ☐ ✓`), settled rows struck through |
| diff | unified diff in the gutter: `[5 line-no][3 sign][code]`, added/removed row backgrounds spanning the block |
| notice | one faint (or `$error`) row: cancellations, turn failures |
| task | a subagent: `task · <description> · <status>` label, the child's own blocks nested in a gutter (see §2.1) |

Interactions: the composer stays enabled while the agent works (mid-turn
posts queue into the open turn; the placeholder flips to `working…` and the
legend to `enter queue`) — with one exception: a submit while the main
conversation is `BLOCKED` is refused with a notice, keeping the draft — naming
whichever of the two causes is in front of the user (`answer the approval
prompt first`, or `answer the questions first` when a deferred `ask_user` is
what parked the turn). That is the TUI
opting out of a framework capability, not a framework limit — the framework
would answer the post past the gate or past the parked call
([10-projection §2](../../10-projection.md)),
but in this UI the answer the user owes is the prompt two lines below. A
SUBAGENT's gate with siblings still working leaves the conversation `BUSY`,
so mid-orchestration steering posts keep working. `esc` interrupts a run,
backs out of a modal, and
selects Cancel turn at an approval prompt — it never quits (`ctrl+q` does,
saving first). `↑`/`↓` at the top/bottom of the composer walk the messages
already sent — read straight off the session's main conversation (the
compaction lineage included), so there is no history file and `/clear` or a
resume simply changes which history there is; what was half-typed is stashed
and handed back on the way down. Inside a multiline draft the arrows still
move the cursor. `^p` palette, `^s` skills, `^o` expand
the last clipped output, `ctrl+v` attach a clipboard image. A drive failure
renders as a `▸ model` error block plus a recovery prompt (retry / switch
model / cancel turn; `^r` retries).

### 2.1 How subagents are drawn

Several subagents' events arrive interleaved on one stream, so attribution
cannot be positional: every event carries `conversation_id`, and both the
live-widget state and the mount target are keyed by it. Each child renders as
a `task` block — its label carries the spawn's description and its status
(`waiting / running / done / failed`), and everything the child produces
mounts inside its gutter. The spawn tool renders as the task block itself and
the private result tool renders as nothing (both matched by DECLARATION —
`is_private`, `is_subagent_spawn` — never by name). A task settles off the
resolved `ChildConversation`, not the result tool's event, so a cancelled
subagent stops saying `running`. Resume replays the same tree.

### 2.2 The question set — the fourth dock

When the model calls `ask_user`
([questions](../questions/README.md)) the tool DEFERS: the runner parks the open
turn, the drive returns, and the question set takes the dock — same border, same
inner rules, same `▎❯` selection as the approval prompt. Up to four questions,
one tab each (`☑ storage  ☑ reader  ▎☐ backfill`, right-aligned `3 of 4`), the
question title, its long body at the full 99-column measure, then the options.

Every question ends with the same two rows, always last and never authored by
the model: **Custom answer** — an inline field, for when none of the offered
options is the answer — and **Chat about this**, which has no tick box at all
because it is the way out rather than an answer. There is no skip: those two
rows are what make every question answerable.

| key | single select | multi select |
|---|---|---|
| `↑` `↓` | move the caret | move the caret |
| `1`–`9` | pick that option and advance | toggle that option's tick |
| `space` | pick the caret's option and advance | toggle the caret's tick |
| `enter` | confirm, then advance — or open the confirmation | **commit** the ticks (never toggles), then advance — or open the confirmation |
| `⇥` `⇧⇥` | next / previous question | next / previous question |

**The custom row holds a real editor.** Picking it hands the keyboard to a
nested `PromptInput` (the same `TextArea` the composer uses), so everything you
expect from a text field works: cursor movement, `home`/`end`, word-jump,
selection, paste, undo. Digits and `space` become literal text there, which is
why the legend changes while it has focus. The row stays exactly one line —
`soft_wrap` is off, so an answer longer than the panel scrolls sideways under
the cursor instead of growing the dock. `enter` commits the answer, `esc`
hands the keyboard back to the option rows with the text intact, and `⇥` still
moves between questions from inside the field. `↑`/`↓` belong to the editor
while it has focus; leave the field to move the caret again.

> ⚠️ **`esc` never leaves a question unanswered.** Its one job in this panel is
> leaving the custom-answer field; everywhere else it is swallowed. It does not
> skip, does not clear what you typed, does not close the set, and does not
> reach the composer or the cancel binding.

Nothing reaches the agent until a **confirmation** panel: one read-only line per
question plus one optional free-text field. `enter` submits, `esc` goes back
with every answer intact, so submission always takes two deliberate presses.
Then the set collapses into an ordinary tool block with one `☑ tab → answer` row
per question.

`ScreenState.questions` carries the whole thing (`state.QuestionSetState`), and
`QuestionSetState.settled` is the ONE predicate behind both `enter`'s two
meanings and the legend that states which — so the panel and the hint row can
never disagree.

To see all four states without a model, browse them in the gallery (§4.1):
`--gallery dock/questions`, `dock/questions-answered`,
`dock/questions-confirming`, and `chat/questions-answered` for the collapsed
block the set leaves behind.

### 2.3 The docked plan panel

The todo list is not a transcript block. It sits in its own region between the
transcript and the dock (`#plan-dock`), which means it outlives the turn that
wrote it and stays put while an approval prompt or an overlay takes the dock.
`ScreenState.plan` carries it, so the gallery and the snapshot suite see
exactly what the live app draws.

Neither `read_todo` nor `update_todos` renders a transcript row — both are
matched by DECLARATION (`ToolSpec.namespace == "contrib.memory"` plus the tool
name, so an application's own `update_todos` is untouched) and swallowed. The
panel is the only thing either one produces.

What it shows:

```
12 tasks (7 done, 5 open)      ← zero terms are omitted; `Done (N tasks completed)` when finished
☐ #8 - update every import site
☐ #9 - rewrite the store tests
☐ #10 - benchmark the new path
☐ #11 - document the format
☐ #12 - cut the release note
… +7 completed                 ← names a status only when the hidden rows share one, else `+N more`
```

Five rows, because every row it takes is a row the transcript never gets back.
Which five depends on the turn: **while a run is live** it leads with the items
the last write moved (`structured_content.changed`), struck through, so
completions scroll past as they happen; **once the run settles** it goes back
to open-first. Settled rows are struck (Rich `Style(strike=True)`, which
survives SVG export, so snapshots catch it).

**The app owns the list.** `wiring.build_runner` hands `MemoryPlugin` two dicts
that live on `session.extras` (`"todos"` and `"scratchpad"`), so the tools write
straight into the session the app was already saving — which is the whole
reason a plan and a scratchpad survive `--resume`. The panel reads the same
dicts back (`session_todos`), so it cannot disagree with what the model sees,
and `plan_from_session` draws a stored session's panel with no runner at all.
Compaction installs a new conversation id, and `CompactionFinished` is where the
app moves both slots over. Only the main conversation's list is docked: one
panel cannot speak for a parent and three subagents at once.

**When the panel gives the rows back.** A plan with nothing open is dismissed
once the user speaks past it (`plan_dismissed`) — the agent keeps the list,
because its numbering is built on it and the model may reopen an item, but a
finished plan stops costing the transcript five rows. It comes back the moment
the next `update_todos` lands. The rule reads the session and changes nothing,
so the live panel and a resumed one dismiss at exactly the same point.

## 3. Slash commands

Type `/` in the empty composer (or `^p`) for the palette; typing filters, and
a full `/name arg` line submits directly while idle. Unknown commands are
sent to the agent as a normal message, so `/etc/hosts` is never swallowed.

| Command | Effect |
|---|---|
| `/skill [name]` | Sends `/skill <name>` to the agent (the model loads it with the skill tool); the palette inserts `/skill ` for the argument |
| `/session`, `/resume` | The sessions screen: `↑↓` move, `enter` resume, `f` fork, `d` (twice) delete, with a last-turn preview per row |
| `/context` | The `@` context picker (also opened by typing `@` inline) — `space` checks rows, `enter` writes the picked paths into the composer as `@path` mentions (comma-joined); nothing checked takes the highlighted row |
| `/cost` | The cost screen: estimated spend per category with cost-proportional meters, the context window bar, biggest consumers; `^k` compacts |
| `/settings` | The settings screen: `← →` adjusts model / reasoning / streaming / theme / counter live; `esc` saves and closes |
| `/clear`, `/new` | Save, then start a fresh session carrying the runtime config |
| `/model [provider:model]` | No arg: provider then model, in the overlay menu. Takes effect next turn |
| `/reasoning [level]` | No arg opens the level menu |
| `/theme` | Pick any registered theme (`luca-dark` ships; see `theme.py`) |
| `/compact` | Summarize the history and continue ([12](../../12-compaction.md)) |
| `/help` | Every command, as a list block |
| `/quit` | Save and exit |

Dollar figures anywhere (status counter, sessions, cost screen) are estimates
from the small `usage.PRICING` table and are omitted for unlisted models.

## 4. The design system

The design source of truth is the handoff in `design_handoff_luca_tui/` (the
eleven screens `1a`–`1k`). The system that implements it:

- **`theme.py`** — the palette as a registered `Theme`; the only hex source.
  The contrast floor is `$text-faint` (#7E7E7E) — nothing functional dimmer.
- **`app.tcss`** — every layout, spacing, border and color assignment, tokens
  only, commented by the screen ids each section serves.
- **`state.py`** — the view-model: `ScreenState` (status, transcript blocks,
  dock, modal, hints). Fixtures, the live app and the replay all produce it.
- **`catalog.py` + `gallery.py` + `fixtures/`** — the catalog, below.

### 4.1 The catalog

```bash
uv run python main.py --gallery                    # browse every state
uv run python main.py --gallery chat/subagents     # one catalog entry
uv run python main.py --gallery 1c_diff_approval   # one handoff screen
uv run python main.py --gallery my_state.yaml      # your own fixture
uv run python main.py --gallery ~/.luca/projects/<project>/<id>.json
```

`←`/`→` cycle, `g` opens the index, `ctrl+q` quits. No key or provider is
involved — nothing here talks to a model.

Two tiers, both browsable and both snapshot-tested.

**Derived (`fixtures/catalog.yaml`).** `screen × world`. A **screen** is a pure
projection in `catalog.SCREENS` (`chat`, `approval`, `questions`, `palette`,
`picker`, `cost`, `settings`, `sessions`); a **world** is a committed `AgentSession`
under `fixtures/sessions/` plus a `catalog.World` of the ambient state no
session holds (branch, theme, approval mode, a typed query, ticked boxes).
Every `World` field has the ordinary default, so an entry names only what makes
it different:

```yaml
- {name: chat/subagents, screen: chat, session: subagents}
- {name: modal/settings-yolo, screen: settings, session: conversation, world: {mode: yolo}}
```

The projections call the same functions the live app calls — `transcript_blocks`,
`cost_state`, `build_settings_state`, `build_sessions_state`,
`build_approval_prompts`, `question_set_state`. That is the point: a derived
state cannot depict a feature that no longer exists, because deleting it
changes the screen or breaks the build. `questions` goes further and rebuilds
the dock through `QuestionsTool.pending()` itself, seeded from the parked
call's own arguments — the same three steps the live app takes.

| To add… | Do this |
|---|---|
| a state | one line in `fixtures/catalog.yaml`, then `pytest --snapshot-update` |
| a world | a builder in `tests/agent/contrib/tui/session_library.py`, then `uv run python -m tests.agent.contrib.tui.session_library` |
| a screen | a `Scene → ScreenState` function in `catalog.py`, a name in `SCREENS`, and at least one entry (a test enforces the last part) |

Worlds are **authored, never captured** — a recorded session is a 100KB diff
nobody reviews. Ids, timestamps, token counts and `catalog.NOW` are all fixed,
so screens are byte-stable; `test_catalog.py` fails if the committed JSON and
the builders disagree.

**Hand-authored (`fixtures/*.yaml`).** YAML validated as `ScreenState`, for
states no producer exists for yet — so "what does the approval prompt look like
over a 200-line diff?" is a file, not a live-model hunt:

```yaml
name: my_state
status: {cwd: ~/quantized/luca, model: sonnet-4.5, branch: main, tokens: 12.4k}
transcript:
  - {kind: user, text: run the tests}
  - {kind: tool, tool: bash, arg: pytest -q, result: {summary: exit 0 · 1.2s}}
composer: {placeholder: "ask, or / for commands"}
hints: [enter send, ⇥ complete, ^p palette]
```

These are specifications, not records: a fixture pins its own output, so it
cannot notice drift. Prefer a catalog entry whenever the agent can actually
produce the state.

Every state in both tiers has a committed SVG snapshot
(`tests/agent/contrib/tui/test_snapshots.py`); `pytest --snapshot-update`
regenerates them after an intentional visual change — review the SVG diff like
code.

A stored `AgentSession` path also works: the gallery recognises one by shape and
derives its screen through the same `render.transcript_blocks` the live app
replays on resume, so what you see is what a resume would show.

> ⚠️ **YAML gotchas.** Quote strings containing commas inside `{…}` flow
> mappings, and diff line numbers use the key `num` — bare `no` is a YAML
> boolean.

## 5. Structure

| Module | Role |
|---|---|
| `theme.py` / `app.tcss` / `state.py` / `format.py` | The design system's contract: palette, geometry, view-model, pure text helpers (token spans, humanized figures, hint legends) |
| `render.py` | Pure derivations: entries/events → view-models (`tool_block`, `plan_block`, `preview_rows`, `subagent_task`, …) plus a whole stored session → transcript blocks (`entry_blocks`, `transcript_blocks`) — live, replay and the catalog share them, so they cannot drift |
| `blocks.py` / `chrome.py` / `shells.py` / `modals.py` | The widgets: transcript blocks, status bar + legend + composer, the shared selection treatment + approval prompt + question set + overlay list + modal base, the three modal screens |
| `frame.py` | `LucaApp` — the frame; `apply_state(ScreenState)` renders any state |
| `catalog.py` | The derived catalog: `screen × world → ScreenState`, the `SCREENS` registry and the `World` / `Entry` models |
| `gallery.py` | `GalleryApp` (`--gallery`) over both tiers; fixture and session loading |
| `app.py` | `AgentApp(LucaApp)` — the drive worker and one event handler for both streaming and block tiers |
| `wiring.py` | `build_runner(...)` — shell + memory + questions plugins, demo math tools, one shared strategy; returns `(runner, strategy, questions_plugin)`; `build_faux_provider()` scripts the `--faux` conversation |
| `approvals.py` | Pending permission steps → the fixed 4-option prompt model (`Approve once / Approve always — <scope> / Deny / Cancel turn`), no UI |
| `usage.py` / `gitinfo.py` / `files.py` | Status counter + cost-screen state; branch/dirty; `@`-picker file listing |
| `sessions.py` / `commands.py` / `config.py` / `cli.py` / `prompt.py` / `clipboard.py` | The store (atomic save, summaries with previews, and the `.tui.json` sidecar), the command registry + live modal-state builders, `luca.json`, argparse, the composer's TextArea, clipboard image read |

## 6. Test with the faux client

```python
provider = FauxProvider()
provider.set_responses([faux_assistant_message([faux_text("Hello!")])])
app = AgentApp(session, provider=provider, workspace=tmp_path,
               session_dir=tmp_path, skills=False, instructions=False)

async with app.run_test(size=(105, 35)) as pilot:
    app.query_one(PromptInput).load_text("hi")
    await pilot.press("enter")
    ...
    assert [t.text for t in app.query(AssistantText)] == ["Hello!"]
```

Widgets keep their view-models on `.model`, so tests assert on data, not on
rendered output. Approval flows are inline (`ApprovalPromptView` in the dock
— press a digit), cancellation is `faux_hang()` + `esc`, and subagent tests
assert on the transcript TREE (blocks inside `TaskBlockView.body`), never a
flat query. See `tests/agent/contrib/tui/` for the full patterns and
`test_snapshots.py` for the fixture snapshots.

> ⚠️ **The app owns the wiring.** `AgentApp` builds its runner via
> `build_runner` — inject behavior through `provider=`, `workspace=`,
> `mode=` ("ask" / "yolo" / "auto") and `context_manager=`, not by passing a
> runner.

Next: [`subagents/README.md`](../subagents/README.md).
