Guidance for the Textual TUI. Read `AGENTS.agent.md` first for the rules shared by the whole `luca.agent` layer.

## What this package is

`luca/agent/contrib/tui/` contains the Textual terminal UI: `AgentApp`, its wiring, and the approval modal. The logic in `approvals.py`, `render.py`, `sessions.py`, and `wiring.py` stays Textual-free. The package needs the `tui` dependency group, which is a uv default group.

`main.py` is the runnable agent demo and launches this contrib TUI. It drives the resource-permission modes, rules, resource globs, answer-decoupled interactive approvals, and `ResourcePermissionToolMixin` integration from `luca/agent/contrib/resource_permissions/`.

## Agent integration details

- The core default for `subagents_max_depth` is 1; the TUI opts into 3.
- The `/model` command can switch models in the middle of a session. The entire history is then replayed to the new model, exercising the reasoning-attestation provenance rules described in `AGENTS.agent.md`.

## Testing

Tests live in `tests/agent/contrib/tui/`. Pure-module coverage includes `test_approvals.py`, `test_render.py`, `test_sessions.py`, `test_wiring.py`, `test_cli.py`, `test_config.py`, and `test_context_bar.py`. Headless Pilot tests drive `AgentApp` with a scripted `FauxProvider` in `test_app*.py`. The directory skips itself when Textual is missing.
