Guidance for the Textual TUI. Read `AGENTS.agent.md` first for the rules shared by the whole `luca.agent` layer.

## What this package is

`luca/agent/contrib/tui/` is the Textual terminal UI, built as a design system: every visual state is expressible as declarative data, renderable without a live agent, and snapshot-tested. The visual design is specified by the handoff in `design_handoff_luca_tui/` at the repo root (eleven screens, `1a`–`1k`); treat that README as the design source of truth.

## The design-system layers

```
theme.py      the palette as a registered Textual Theme — THE ONLY HEX SOURCE
app.tcss      every layout, spacing, border and color assignment ($tokens only)
state.py      the view-model: ScreenState + the block vocabulary (pure Pydantic)
format.py     pure text helpers: token spans, humanized figures, hint legends
render.py     pure derivations: agent entries/events → view-models, and a whole
              stored session → transcript blocks (`transcript_blocks`)
blocks.py     the transcript block widgets (user, thinking, text, tool, list, diff, notice, task)
chrome.py     StatusBar, HintLegend, Composer
shells.py     SelectRow (THE selection treatment, defined once), ApprovalPromptView,
              QuestionSetView + QuestionConfirmView (the ask_user dock),
              OverlayListView (palette/picker/menu), LucaModalScreen
modals.py     SessionsScreen, SettingsScreen, CostScreen
frame.py      LucaApp — the global frame; apply_state(ScreenState) renders any state
catalog.py    the DERIVED catalog: screen × world → ScreenState (see below)
gallery.py    GalleryApp (`--gallery`) over both tiers; fixture loading
fixtures/     catalog.yaml (the grid) + sessions/ (the worlds)
              + 1a–1k and components/ (hand-authored screen states)
app.py        AgentApp(LucaApp) — the live agent wiring (drive worker, events)
```

Supporting live modules: `usage.py` (token totals, estimated cost, the 1k cost state), `gitinfo.py` (branch/dirty), `files.py` (@-picker file listing), `prompt_files/` (`@`-mention expansion: `parse_prompt` → content parts, via a first-match-wins handler chain — add a format by adding a handler above `BinaryHandler`, nothing else changes), `sessions.py` (session store + summaries, and the `<session-id>.tui.json` sidecar), `commands.py` (the 14 slash commands + live modal-state builders), `approvals.py` (the 4-option prompt model), `config.py` (`luca.json`), `cli.py`, `prompt.py` (the composer's TextArea).

## Hard rules (from the design handoff)

- **No hex literal outside `theme.py`. No geometry outside `app.tcss`.** No `styles.x = ...` in Python for anything a stylesheet can express. Widgets that need concrete colors resolve them through `theme.resolve_tokens(app)`.
- **Contrast floor `$text-faint` (#7E7E7E).** Nothing functional dimmer; if something seems to need a dimmer grey, remove the element.
- **No animation** — no spinners, progress bars, or transitions. The only moving thing is the terminal's cursor.
- **Selection never reflows** — `SelectRow` always allocates the bar and caret columns; the treatment exists once (`shells.SelectRow` + `.select-row` in TCSS).
- **Plain keys** — arrows, tab, enter, esc, digits, documented `^` pairs. The hint legend always reflects the focused context.
- **Every automatic permission decision is stated in the transcript** (`approved by rule`, `denied · by rule`).
- Don't invent UI the handoff doesn't specify. The two sanctioned extensions are `NoticeBlock` (turn-level notices) and `TaskBlock` (subagent conversations, built from the handoff's gutter idiom). The QUESTION SET has a handoff of its own — `specs/0007-deferred-tool-calling/design/README.md`, whose ASCII mockups are normative.
- `@` file mentions render as ordinary `ToolBlock`s (`tool: read`) under the user turn — same idiom, no new block kind. They are NOT tool calls: no `ToolExecution` backs them, so they never carry an approval note. A declined mention (too long, binary, a directory) is `status: error` with a `[error]×[/]` summary; a path that does not resolve gets no row at all and stays prose, which is what keeps `@property` and `@types/node` from being read as files. See `fixtures/components/mentions.yaml`, the requirements sheet for the feature.

## The catalog

```bash
uv run python -m luca.agent.contrib.tui --gallery                 # browse everything
uv run python -m luca.agent.contrib.tui --gallery chat/subagents  # one catalog entry
uv run python -m luca.agent.contrib.tui --gallery 1a_agent_loop   # one fixture
uv run python -m luca.agent.contrib.tui --gallery ~/.luca/projects/<project>/<id>.json
```

Two tiers, both browsable and both snapshot-tested (`tests/agent/contrib/tui/test_snapshots.py`; `pytest --snapshot-update` regenerates the committed SVGs).

**Tier 1 — the catalog (`fixtures/catalog.yaml`).** Two axes:

- **Screens** are projections, one per screen of the app: `catalog.SCREENS` maps a name to a pure `Scene → ScreenState` function. They call the SAME code the live app calls (`transcript_blocks`, `cost_state`, `build_settings_state`, `build_sessions_state`, `build_approval_prompts`), which is the whole point — a catalogued screen cannot depict a feature that no longer exists, because deleting it changes the screen or breaks the build.
- **Worlds** are data: a committed `AgentSession` under `fixtures/sessions/` plus a `catalog.World` of the ambient state no session holds (branch, theme, approval mode, a typed query, the boxes ticked in a picker). Every `World` field has the ordinary default, so an entry names only what makes it different.

```yaml
- {name: chat/subagents, screen: chat, session: subagents}
- {name: modal/settings-yolo, screen: settings, session: conversation, world: {mode: yolo}}
```

| To add… | Do this |
|---|---|
| a state | one line in `fixtures/catalog.yaml`, then `pytest --snapshot-update` |
| a world | a builder in `tests/agent/contrib/tui/session_library.py`, then `uv run python -m tests.agent.contrib.tui.session_library` |
| a screen | a `Scene → ScreenState` function in `catalog.py` + a name in `SCREENS` + at least one entry (a test enforces the last part) |

Worlds are **authored, never captured**. A recorded real session is a 100KB diff nobody reviews, full of absolute paths; the builders are a dozen lines each. Everything is fixed — ids, timestamps, token counts, and `catalog.NOW` — so the built screens are byte-stable. `test_catalog.py` fails if the committed JSON and the builders disagree.

**Tier 2 — fixtures (`fixtures/*.yaml`).** Hand-authored `ScreenState` documents: the eleven handoff screens `1a`–`1k` plus `components/` sheets. These are for states with no producer yet ("what does 90% context look like?") — specifications, not records. A fixture cannot notice drift, so prefer a catalog entry whenever the agent can actually produce the state.

YAML gotchas: quote strings containing commas inside `{...}` flow mappings, and diff line numbers use the key `num` (`no` is a YAML boolean).

A stored `AgentSession` path also works: the gallery recognises one by shape and derives its screen through `render.transcript_blocks` — the same derivation `app._replay_path` mounts on resume, so what you see is what a resume would show.

## The fourth dock — the agent's questions

`ask_user` (`luca.agent.contrib.questions`) is the first DEFERRED tool: it returns `ExecutionDeferred()`, the runner parks the open turn at `AWAITING_RESULT`, and the drive returns so the application can answer. The TUI answers it in the dock, mutually exclusive with composer / approval / overlay — `ScreenState.questions` plus a fourth `dock()` branch.

- **The widget decides nothing.** `QuestionSetView` reports what the user did (`Answered` / `Toggled` / `CustomChanged` / `ChatRequested` / `Moved`); `AgentApp` owns advancing the active tab, flipping `phase` and rebuilding the state. Same split `ApprovalPromptView.Decided` already has.
- **`QuestionSetState.settled` is the one predicate** behind both `enter`'s two meanings and the legend that states which (`question_set_is_last` / `question_hints_for`), so the panel and the hint row cannot disagree.
- **`enter` commits and never toggles.** In multi select only digits and `space` tick a row; the keystroke that finishes a question must not also change the answer it is finishing.
- **`esc` is swallowed in this panel** (it only clears the custom-answer field while that field is being edited) and `action_cancel_run` is a no-op while a set is out — otherwise the drive worker is left blocked on a future nothing will resolve. The design's only exits are the custom-answer row and Chat about this.
- **The store is the app's, not the session's.** Outstanding questions are the INTERFACE's state, so they live in `<session-id>.tui.json` beside the session rather than on `AgentSession.extras`. Losing that file is survivable — the tool re-seeds from `raw_tool_call.arguments` and defers again — which is why every read and write of it is best-effort and never raises.
- Fixtures: `1l_questions.yaml` plus the five `components/questions_*.yaml` sheets. They are Tier 2 (hand-authored) because there is no `Scene → ScreenState` producer for a live question set yet.

## Agent integration details

- The core default for `subagents_max_depth` is 1; the TUI opts into 3.
- `/model` can switch models mid-session; the whole history replays to the new model (reasoning-attestation provenance rules apply).
- Sessions live in `~/.luca/projects/<encoded-project-path>/<id>.json`, keyed on the WORKSPACE. `resolve_session_directory` in `sessions.py` computes it.
- `_reset_session` is the "switch to this session" primitive behind `/clear`, `/session` resume and fork: rebuild the runner, wipe the transcript, replay.
- The sessions screen parses every stored session to build its rows (~3ms per 500KB session); no index file to keep in sync.
- The `@` picker commits by writing `@path` mentions into the composer (`format.inline_paths`) — it does not attach file contents, so the paths reach the model as text and it reads them with its own tools.
- Mock-only surfaces (until the agent grows the feature): the settings screen's `approval mode` row is read-only.
- Model facts come from `luca.client.catalog`, generated from [models.dev](https://models.dev) and refreshed with `--refresh-models`. `commands.pickable_models()` is the one door for "what may `/model` offer": the catalog intersected with `luca.client.providers.PROVIDERS`, unioned with the `models` key from `luca.json`. Offering a host with no transport would be a dead end, and `ollama` and custom providers are not in models.dev, so config is their only route. `commands.model_context_note()` puts the context window in each row's SECONDARY column — a menu row draws `primary` and `secondary`, never `annotation`, which is the `@` picker's field. `commands.recent_models()` is the short list for the settings row's `← →` and the retry-on-failure offer: newest first, one per `family`, because a host publishes the same model many times.
- Dollar figures are estimates, but from the catalog now rather than a hand table, keyed on `(provider, model)`. Omitted for models the catalog does not price. A few models price by context tier and `ModelCost` carries one flat rate, so a very long session under-reports.
- **The catalog is metadata, never a gate.** `/model provider:model` still switches to anything; a lookup miss means "no metadata", not "unusable".
- The overlay list SCROLLS (`.overlay-options` is a `VerticalScroll` bounded by `max-height` in `app.tcss`, with `#dock` capped so the transcript survives). Every matching row is in the state; the stylesheet decides how many are visible, and `_move` scrolls the caret into view. A cell count, not a percentage — a percentage there resolves against the content, which is the thing being bounded. The `@` picker still trims its matches at `files.MAX_ROWS` because ranking makes row 200 meaningless; a model list has no ranking, so nothing is dropped.

## Testing

Tests live in `tests/agent/contrib/tui/`. Pure-module coverage: `test_state.py`, `test_format.py`, `test_render.py`, `test_approvals.py`, `test_sessions.py`, `test_config.py`, `test_wiring.py`, `test_cli.py`, `test_commands.py`, `test_gallery.py`. Headless Pilot tests drive `AgentApp` with a scripted `FauxProvider` in `test_app*.py` and `test_subagents.py`. `test_snapshots.py` renders every bundled fixture at 105×35 and compares committed SVGs. The directory skips itself when Textual is missing.
