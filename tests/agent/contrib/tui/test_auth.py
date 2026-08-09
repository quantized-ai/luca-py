"""`auth.json`: where it is read from, what shape it takes, and what a missing
or malformed one does."""

import json
from pathlib import Path

import pytest

from luca.agent.contrib.tui.auth import (
    ENV_AUTH_PATH,
    AuthEntry,
    api_key_for,
    auth_home,
    load_auth,
    resolve_auth_path,
)
from luca.agent.contrib.tui.config import LucaConfigError


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


# ── location ─────────────────────────────────────────────────────────────────


def test_the_default_location_is_the_xdg_data_directory(monkeypatch):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: Path("/home/someone"))

    assert auth_home() == Path("/home/someone/.local/share/luca")


def test_xdg_data_home_moves_it(monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", "/elsewhere/data")

    assert auth_home() == Path("/elsewhere/data/luca")


def test_the_env_var_names_a_file_outright(monkeypatch):
    monkeypatch.setenv(ENV_AUTH_PATH, "~/keys.json")

    assert resolve_auth_path() == Path.home() / "keys.json"


def test_an_explicit_path_beats_the_env_var(monkeypatch):
    monkeypatch.setenv(ENV_AUTH_PATH, "/from/env.json")

    assert resolve_auth_path("/from/the/caller.json") == Path("/from/the/caller.json")


# ── loading ──────────────────────────────────────────────────────────────────


def test_a_file_of_credentials_loads(tmp_path):
    path = _write(
        tmp_path / "auth.json",
        {
            "openrouter": {"type": "api", "key": "sk-or-1"},
            "my_custom_provider": {"type": "api", "key": "sk-2"},
        },
    )

    assert load_auth(path) == {
        "openrouter": AuthEntry(type="api", key="sk-or-1"),
        "my_custom_provider": AuthEntry(type="api", key="sk-2"),
    }


def test_a_missing_file_is_simply_no_credentials(tmp_path):
    # Running entirely off environment variables is the default experience,
    # not a degraded one.
    assert load_auth(tmp_path / "nope.json") == {}


def test_a_malformed_entry_is_an_error_rather_than_a_silent_skip(tmp_path):
    # Skipping it would send the request unauthenticated and report a 401.
    path = _write(tmp_path / "auth.json", {"openrouter": {"type": "api"}})

    with pytest.raises(LucaConfigError, match="'openrouter' is invalid"):
        load_auth(path)


def test_an_unknown_credential_type_is_rejected(tmp_path):
    path = _write(tmp_path / "auth.json", {"openrouter": {"type": "oauth", "key": "x"}})

    with pytest.raises(LucaConfigError, match="'openrouter' is invalid"):
        load_auth(path)


def test_an_unknown_key_in_an_entry_is_rejected(tmp_path):
    path = _write(tmp_path / "auth.json", {"openrouter": {"type": "api", "key": "x", "keyy": "y"}})

    with pytest.raises(LucaConfigError, match="'openrouter' is invalid"):
        load_auth(path)


def test_invalid_json_names_the_file(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text("{not json")

    with pytest.raises(LucaConfigError, match="not valid JSON"):
        load_auth(path)


def test_a_top_level_list_is_rejected(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text("[]")

    with pytest.raises(LucaConfigError, match="must be a JSON object"):
        load_auth(path)


# ── lookup ───────────────────────────────────────────────────────────────────


def test_a_provider_with_no_entry_has_no_key():
    # None, not "", so the client falls back to its own environment variable.
    assert api_key_for({"openrouter": AuthEntry(type="api", key="sk-1")}, "anthropic") is None


def test_a_provider_with_an_entry_gets_its_key():
    assert api_key_for({"openrouter": AuthEntry(type="api", key="sk-1")}, "openrouter") == "sk-1"
