"""The gallery's fixture machinery: listing the bundled set, resolving a
reference to a path, and loading + validating a fixture into a `ScreenState`."""

import json

import pytest

from luca.agent.contrib.tui import state as vm
from luca.agent.contrib.tui.format import HINTS
from luca.agent.contrib.tui.gallery import (
    FIXTURES_DIR,
    FixtureError,
    _resolve,
    fixture_name,
    is_session_document,
    list_fixtures,
    load_fixture,
    resolve_fixture,
    session_state,
)
from luca.agent.core.models import AssistantMessage, SessionConfig, TextContent, UserMessage
from tests.agent.scenarios import MODEL, conversation, make_session

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


# ── a stored session as a screen ──────────────────────────────────────────────

SESSION = make_session(
    id="s1",
    entries={
        "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="migrate the store")]),
        "a1": AssistantMessage(
            id="a1",
            created_at=500,
            parts=[TextContent(text="On it.")],
            llm_config=MODEL,
            stop_reason="end_turn",
        ),
    },
    conversations={"c1": conversation("c1", ["u1", "a1"])},
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)


def test_a_session_document_is_recognised_by_its_shape():
    assert (
        is_session_document(json.loads(SESSION.model_dump_json())),
        is_session_document({"name": "1a", "transcript": []}),
    ) == (True, False)


def test_a_session_renders_as_the_screen_it_would_resume_into():
    assert session_state(SESSION) == vm.ScreenState(
        name="s1",
        status=vm.StatusState(cwd="~", model="test-model"),
        transcript=[vm.UserBlock(text="migrate the store"), vm.TextBlock(text="On it.")],
        composer=vm.ComposerState(),
        hints=HINTS["idle"],
    )


def test_load_fixture_derives_a_screen_from_a_session_file(tmp_path):
    path = tmp_path / "s1.json"
    path.write_text(SESSION.model_dump_json())
    assert load_fixture(path).transcript == [
        vm.UserBlock(text="migrate the store"),
        vm.TextBlock(text="On it."),
    ]


def test_an_unreadable_session_file_names_itself(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"entries": {"u1": {"type": "nonsense"}}, "conversations": {}}')
    with pytest.raises(FixtureError, match="not a readable session"):
        load_fixture(path)


def test_a_file_given_by_path_is_never_swapped_for_a_bundled_one(tmp_path):
    """`./scratch/1a_agent_loop.yaml` must render YOUR file, not the bundled
    fixture that happens to share its stem."""
    decoy = tmp_path / "1a_agent_loop.yaml"
    decoy.write_text("name: mine\ntranscript: [{kind: user, text: mine}]\n")
    item, bundled = _resolve(str(decoy))
    assert (bundled, item.build().transcript) == (False, [vm.UserBlock(text="mine")])


def test_a_catalog_name_wins_over_a_file_at_that_path():
    item, bundled = _resolve("chat/empty")
    assert (bundled, item.name) == (True, "chat/empty")
