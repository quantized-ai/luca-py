"""The gallery's fixture machinery: listing the bundled set, resolving a
reference to a path, and loading + validating a fixture into a `ScreenState`."""

import pytest

from luca.agent.contrib.tui import state as vm
from luca.agent.contrib.tui.gallery import (
    FIXTURES_DIR,
    FixtureError,
    fixture_name,
    list_fixtures,
    load_fixture,
    resolve_fixture,
)

BUNDLED_NAMES = [
    "1a_agent_loop",
    "1b_plan",
    "1c_diff_approval",
    "1d_command_execution",
    "1e_palette",
    "1f_context_picker",
    "1g_skills",
    "1h_denial_error",
    "1i_sessions",
    "1j_settings",
    "1k_cost",
    "components/blocks",
    "components/empty",
    "components/mentions",
    "components/menu",
    "components/model_menu",
    "components/tasks",
    "components/tool_states",
]


def test_list_fixtures_returns_every_bundled_fixture():
    assert [fixture_name(path) for path in list_fixtures()] == BUNDLED_NAMES


def test_list_fixtures_on_a_missing_directory_is_empty(tmp_path):
    assert list_fixtures(tmp_path / "missing") == []


def test_resolve_fixture_by_bundled_name():
    assert resolve_fixture("1a_agent_loop") == FIXTURES_DIR / "1a_agent_loop.yaml"


def test_resolve_fixture_by_path(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("name: custom\n")

    assert resolve_fixture(str(path)) == path


def test_resolve_fixture_unknown_names_every_bundled_fixture():
    with pytest.raises(FixtureError) as excinfo:
        resolve_fixture("9z_nope")

    assert str(excinfo.value) == f"no fixture '9z_nope'; bundled fixtures: {', '.join(BUNDLED_NAMES)}"


def test_load_fixture_rejects_unparseable_yaml(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("{unclosed: [")

    with pytest.raises(FixtureError, match="not parseable"):
        load_fixture(path)


def test_load_fixture_rejects_a_non_mapping(tmp_path):
    path = tmp_path / "listy.yaml"
    path.write_text("- 1\n- 2\n")

    with pytest.raises(FixtureError, match="top level must be a mapping"):
        load_fixture(path)


def test_load_fixture_rejects_a_bad_mapping(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("transcript:\n  - kind: no_such_block\n")

    with pytest.raises(FixtureError, match="not a valid screen state"):
        load_fixture(path)


@pytest.mark.parametrize("path", list_fixtures(), ids=fixture_name)
def test_every_bundled_fixture_is_a_valid_screen_state(path):
    state = load_fixture(path)

    assert isinstance(state, vm.ScreenState)
    assert state.name == fixture_name(path)
