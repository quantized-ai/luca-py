"""Skips the whole directory when textual (the `tui` dependency group) is not
installed, and keeps every TUI test off the developer's real configuration."""

import logging
import os

import pytest

pytest.importorskip("textual")

from luca.agent.contrib.tui.cli import _remove_log_handlers
from luca.agent.contrib.tui.config import ENV_CONFIG_PATH
from luca.client.catalog import _store
from luca.client.providers import PROVIDERS


@pytest.fixture(autouse=True)
def _restore_environ():
    """`apply_env_file` writes straight to `os.environ`, which monkeypatch has
    no record of and so never rolls back. Declared FIRST so it tears down last
    and restores what the other env fixtures set up over."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def _isolated_config_environment(monkeypatch, tmp_path):
    """Keep discovery off the contributor's real home: config reads
    `~/.config/luca/luca.json`, skills read `~/.claude/skills`, slash commands
    read `~/.claude/commands`, and instructions read `~/.config/luca/LUCA.md`."""
    monkeypatch.delenv(ENV_CONFIG_PATH, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    # The model catalog layers `$XDG_CACHE_HOME/luca/models.json` over the
    # vendored records, so without this a contributor who has run
    # `--refresh-models` tests against different models than CI does.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    # the store loads once per process; drop it so the patched env is what it reads
    _store._clear_for_tests()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@pytest.fixture(autouse=True)
def _restore_luca_logger():
    """`main()` points the process-wide `luca` logger at a session file. Left
    attached, it holds a deleted tmp_path open and its `propagate=False` blinds
    `caplog` for every test that runs after it."""
    log = logging.getLogger("luca")
    level = log.level
    yield
    _remove_log_handlers()
    log.setLevel(level)


@pytest.fixture(autouse=True)
def _restore_providers():
    """Nothing in the TUI registers a host any more — provider settings are
    carried per call. The guard stays because `PROVIDERS` is a module-level
    dict `luca.client` exposes for registration, and a test that reaches for
    `register_provider` must not leak a host into every test after it."""
    saved = dict(PROVIDERS)
    yield
    PROVIDERS.clear()
    PROVIDERS.update(saved)
