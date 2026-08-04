The TUI ships its own design system: every screen state is a declarative
fixture under `luca/agent/contrib/tui/fixtures/`, rendered by the gallery —
this replaced the old Storybook-style previews catalog.

```bash
uv run python -m luca.agent.contrib.tui --gallery                # browse all states
uv run python -m luca.agent.contrib.tui --gallery 1c_diff_approval
uv run python -m luca.agent.contrib.tui --gallery my_state.yaml  # your own fixture
```

`←`/`→` cycle fixtures, `g` opens the index, `ctrl+q` quits.

Rules of the road:

- Any new component or visual state gets a fixture (screens at the top level,
  per-component sheets under `components/`).
- Every fixture is snapshot-tested: `uv run py.test tests/agent/contrib/tui/test_snapshots.py`
  compares committed SVGs; `--snapshot-update` regenerates them after an
  intentional visual change — review the diff like code.
- Visual changes go in `app.tcss` (geometry, spacing, color assignment) or
  `theme.py` (the palette — the only hex source). If a change seems to need a
  literal color or a `styles.x =` assignment in Python, it is a gap in the
  design system: stop and fix the system instead.
