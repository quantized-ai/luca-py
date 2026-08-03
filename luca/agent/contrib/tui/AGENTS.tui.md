Guidance for the Textual TUI. Read `AGENTS.agent.md` first for the rules shared by the whole `luca.agent` layer.

## What this package is

`luca/agent/contrib/tui/` contains the Textual terminal UI: `AgentApp`, its wiring, and the approval modal. The logic in `approvals.py`, `render.py`, `sessions.py`, and `wiring.py` stays Textual-free. The package needs the `tui` dependency group, which is a uv default group.

`main.py` is the runnable agent demo and launches this contrib TUI. It drives the resource-permission modes, rules, resource globs, answer-decoupled interactive approvals, and `ResourcePermissionToolMixin` integration from `luca/agent/contrib/resource_permissions/`.

## Agent integration details

- The core default for `subagents_max_depth` is 1; the TUI opts into 3.
- The `/model` picker's providers come from the model catalog intersected with `luca.client.providers.PROVIDERS`, unioned with the `models` key from `luca.json`. Offering a provider luca has no transport for would be a dead end, and `ollama` / custom hosts are not in models.dev, so config is their only route. `pickable_models` in `commands.py` is that one door.
- `PickerScreen` filters when `filterable=True`. The list reports the index of the row you picked, so `_visible` remaps it back through the full option list — without that, filtering silently returns the wrong value.
- The `/model` command can switch models in the middle of a session. The entire history is then replayed to the new model, exercising the reasoning-attestation provenance rules described in `AGENTS.agent.md`.
- Sessions live in a global store, `~/.luca/projects/<encoded-project-path>/<id>.json`, keyed on the WORKSPACE rather than the process cwd — the same anchor the shell root, skills discovery, and instruction files use. `resolve_session_directory` in `sessions.py` computes it and `cli.main` resolves it once, because `build_session` loads a session before `AgentApp` exists.
- `_reset_session` is the "switch to this session" primitive behind both `/new` and `/resume`: it rebuilds the runner, wipes the transcript, and replays the new session's history. `/new`'s session is empty, so the replay contributes nothing there and one path serves both.
- The `/resume` picker parses every session in the project to title its rows. That is ~3ms each for a 500KB session, which is why there is no index file to keep in sync.

## Testing

Tests live in `tests/agent/contrib/tui/`. Pure-module coverage includes `test_approvals.py`, `test_render.py`, `test_sessions.py`, `test_wiring.py`, `test_cli.py`, `test_config.py`, and `test_context_bar.py`. Headless Pilot tests drive `AgentApp` with a scripted `FauxProvider` in `test_app*.py`. The directory skips itself when Textual is missing.
