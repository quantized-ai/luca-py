"""Skips the whole directory when textual (the `tui` dependency group) is not
installed, and keeps every TUI test off the developer's real configuration."""

import pytest

pytest.importorskip("textual")

from luca.agent.contrib.tui.config import ENV_CONFIG_PATH
from luca.client.providers import PROVIDERS


@pytest.fixture(autouse=True)
def _isolated_config_environment(monkeypatch, tmp_path):
    """Point config discovery at nothing.

    `main()` discovers for real: `LUCA_CONFIG_PATH` first, then
    `$XDG_CONFIG_HOME/luca/luca.json`, falling back to `~/.config`. Tests that
    drive the CLI already chdir away from `./luca.json`, but nothing stopped
    them reading the CONTRIBUTOR'S personal config — so an invalid one failed
    `test_main_prints_the_resume_hint_after_the_app_exits` outright. CI has no
    home config, so that only ever reproduced on one machine."""
    monkeypatch.delenv(ENV_CONFIG_PATH, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


@pytest.fixture(autouse=True)
def _restore_providers():
    """Snapshot the process-global provider registry.

    `register_config_providers` writes straight into `PROVIDERS`, so any config
    carrying a `providers` block leaks a host into every test that runs after
    it. Autouse here rather than per-file: `test_cli.py` is the one that calls
    `main()`, and it is the one that had no protection."""
    saved = dict(PROVIDERS)
    yield
    PROVIDERS.clear()
    PROVIDERS.update(saved)
