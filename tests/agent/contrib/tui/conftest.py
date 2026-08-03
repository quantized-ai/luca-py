"""Skips the whole directory when textual (the `tui` dependency group) is not
installed, and keeps every TUI test off the developer's real configuration."""

import pytest

pytest.importorskip("textual")

from luca.agent.contrib.tui.config import ENV_CONFIG_PATH
from luca.client.providers import PROVIDERS


@pytest.fixture(autouse=True)
def _isolated_config_environment(monkeypatch, tmp_path):
    """Keep discovery off the contributor's real home: config reads
    `~/.config/luca/luca.json`, skills read `~/.claude/skills`, and
    instructions read `~/.config/luca/LUCA.md`."""
    monkeypatch.delenv(ENV_CONFIG_PATH, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@pytest.fixture(autouse=True)
def _restore_providers():
    """`register_config_providers` writes into the global `PROVIDERS`, so a
    config with a `providers` block would leak a host into later tests."""
    saved = dict(PROVIDERS)
    yield
    PROVIDERS.clear()
    PROVIDERS.update(saved)
