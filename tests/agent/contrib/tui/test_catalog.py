"""The declarative catalog: `screen × world → ScreenState`.

The contract under test is that a catalog entry is a PURE function of its
inputs — same entry plus same library, same screen, on any machine — and that
the committed worlds match the builders that authored them.
"""

import pytest

from luca.agent.contrib.tui import state as vm
from luca.agent.contrib.tui.catalog import (
    NOW,
    SCREENS,
    CatalogError,
    Entry,
    FileEntry,
    Scene,
    World,
    build_catalog,
    build_screen,
    catalog_names,
    load_catalog,
    load_session_library,
)
from tests.agent.contrib.tui.session_library import build_all

LIBRARY = load_session_library()


# ── the worlds ────────────────────────────────────────────────────────────────


def test_every_authored_world_is_committed():
    """The JSON under `fixtures/sessions/` IS what `session_library.py` builds.
    Regenerate with `uv run python -m tests.agent.contrib.tui.session_library`."""
    assert {name: session.model_dump() for name, session in build_all().items()} == {
        name: session.model_dump() for name, session in LIBRARY.items()
    }


def test_the_library_covers_the_states_that_matter():
    assert sorted(LIBRARY) == ["approval", "conversation", "empty", "failures", "planning", "subagents"]


def test_an_unreadable_world_names_itself(tmp_path):
    (tmp_path / "broken.json").write_text("{not json")
    with pytest.raises(CatalogError, match="not a readable session"):
        load_session_library(tmp_path)


# ── the grid ──────────────────────────────────────────────────────────────────


def test_every_catalogued_screen_builds():
    assert [entry.name for entry, _state in build_catalog()] == catalog_names()


def test_every_screen_in_the_registry_is_catalogued():
    """A projection nobody catalogues is a projection nobody reviews."""
    assert {entry.screen for entry in load_catalog()} == set(SCREENS)


def test_building_an_entry_twice_gives_the_same_screen():
    entry = Entry(name="chat/conversation", screen="chat", session="conversation")
    assert build_screen(entry, LIBRARY) == build_screen(entry, LIBRARY)


def test_an_entry_naming_an_unknown_world_says_which_exist():
    entry = Entry(name="chat/nope", screen="chat", session="nope")
    with pytest.raises(CatalogError, match="no session 'nope'"):
        build_screen(entry, LIBRARY)


def test_the_entry_name_reaches_the_built_screen():
    entry = Entry(name="chat/conversation", screen="chat", session="conversation")
    assert build_screen(entry, LIBRARY).name == "chat/conversation"


# ── the screens ───────────────────────────────────────────────────────────────


def scene(session: str = "conversation", **world) -> Scene:
    return Scene(session=LIBRARY[session], world=World(**world), library=list(LIBRARY.values()))


def test_chat_docks_the_composer_and_nothing_else():
    state = SCREENS["chat"](scene())
    assert (state.dock(), state.modal, state.composer.placeholder) == ("composer", None, "ask, or / for commands")


def test_a_busy_world_queues_instead_of_sending():
    state = SCREENS["chat"](scene(busy=True))
    assert (state.composer.placeholder, state.hints[0]) == ("working…", "enter queue")


def test_an_empty_world_renders_an_empty_column():
    assert SCREENS["chat"](scene("empty")).transcript == []


def test_the_status_bar_reads_the_world_and_the_session():
    assert SCREENS["chat"](scene(branch="topic", dirty=False)).status == vm.StatusState(
        cwd="~/quantized/luca",
        model="claude-sonnet-5",
        branch="topic",
        dirty=False,
        tokens="3.2k",
        cost="$0.23",
    )


def test_hiding_the_counter_drops_both_figures():
    status = SCREENS["chat"](scene(show_counter=False)).status
    assert (status.tokens, status.cost) == (None, None)


def test_the_approval_prompt_says_exactly_what_it_widens():
    state = SCREENS["approval"](scene("approval"))
    assert state.approval == vm.ApprovalState(
        question="Apply this edit to [accent]luca/events.py[/]?",
        options=[
            vm.ApprovalOption(label="Approve once"),
            vm.ApprovalOption(label="Approve always — edits under luca/"),
            vm.ApprovalOption(label="Deny — tell Luca what to do instead"),
            vm.ApprovalOption(label="Cancel turn", key_hint="esc"),
        ],
        selected=0,
    )


def test_the_approval_screen_needs_a_gated_call():
    with pytest.raises(CatalogError, match="gated, unresolved tool call"):
        SCREENS["approval"](scene("conversation"))


def test_the_palette_filters_and_counts():
    overlay = SCREENS["palette"](scene(query="se")).overlay
    assert (overlay.query, overlay.counter, [row.primary for row in overlay.rows]) == (
        "se",
        "8 of 14",
        ["/skill", "/session", "/context", "/settings", "/clear", "/reasoning", "/theme", "/resume"],
    )


def test_the_picker_marks_matches_and_keeps_the_ticks():
    overlay = SCREENS["picker"](scene(query="writ", checked=["luca/store/writer.py"])).overlay
    assert overlay.rows == [
        vm.OverlayRow(primary="luca/store/[accent]writ[/]er.py", annotation="2.2k", checked=True),
        vm.OverlayRow(primary="docs/[accent]writ[/]ing-skills.md", annotation="3.4k", checked=False),
        vm.OverlayRow(primary="tests/store/test_[accent]writ[/]er.py", annotation="1.1k", checked=False),
    ]


def test_a_picker_world_can_replace_the_file_list():
    overlay = SCREENS["picker"](scene(files=[FileEntry(path="only/one.py", tokens=500)])).overlay
    assert (overlay.counter, overlay.rows) == (
        "1 files",
        [vm.OverlayRow(primary="only/one.py", annotation="500", checked=False)],
    )


def test_the_cost_screen_sums_its_meters_to_the_headline():
    cost = SCREENS["cost"](scene()).modal.cost
    assert (cost.headline, [item.cost for item in cost.items]) == (
        "$0.23",
        ["$0.128", "$0.035", "$0.010", "$0.055"],
    )


def test_a_world_with_no_usage_still_prices_its_zero():
    """The model IS in the price table, so the breakdown exists and is zero —
    a session with no turns reads `$0.00`, not `— tokens`."""
    cost = SCREENS["cost"](scene("empty")).modal.cost
    assert (cost.headline, {item.cost for item in cost.items}) == ("$0.00", {"$0.000"})


def test_the_settings_world_drives_the_rows():
    groups = SCREENS["settings"](scene(mode="yolo", streaming=False)).modal.settings.groups
    assert [row for group in groups for row in group.rows] == [
        vm.SettingRow(name="model", value="claude-sonnet-5"),
        vm.SettingRow(name="provider", value="anthropic"),
        vm.SettingRow(name="reasoning", value="provider-default"),
        vm.SettingRow(name="streaming", value="off"),
        vm.SettingRow(name="approval mode", value="yolo", color="accent"),
        vm.SettingRow(name="theme", value="luca-dark"),
        vm.SettingRow(name="show token counter", value="on"),
    ]


def test_the_sessions_screen_lists_the_whole_library_against_a_fixed_clock():
    sessions = SCREENS["sessions"](scene()).modal.sessions
    assert [(row.when, row.first_message) for row in sessions.rows] == [
        ("18m ago", "migrate the event store to sqlite"),
        ("42m ago", "split @luca/store/reader.py into a reader and a writer"),
        ("2h ago", "add a --json flag to the events command"),
        ("3h ago", "clean the build directory and re-run the suite"),
        ("5h ago", "audit the docs and the tests for stale references"),
        ("yesterday", "(empty)"),
    ]


def test_the_fixed_clock_is_what_makes_that_reproducible():
    assert NOW.isoformat() == "2026-03-14T09:30:00"


# ── the modal screens still describe what sits underneath ─────────────────────


def test_a_modal_keeps_the_transcript_behind_it():
    state = SCREENS["cost"](scene())
    assert (bool(state.transcript), state.status.label) == (True, "cost · this session")


def test_a_world_never_shares_its_file_list_with_the_next_one():
    """The default files are a module constant; a world that could mutate them
    would silently change every other catalogued picker."""
    first = World()
    first.files[0].tokens = 999
    assert World().files[0].tokens == 1400


def test_the_approval_worlds_selection_moves_the_caret():
    assert SCREENS["approval"](scene("approval", selected=2)).approval.selected == 2


# ── the plan fan-out (chat/planning) ──────────────────────────────────────────


def test_a_mention_renders_as_its_own_read_row_under_the_user_turn():
    blocks = SCREENS["chat"](scene("planning")).transcript
    assert blocks[:2] == [
        vm.UserBlock(text="split @luca/store/reader.py into a reader and a writer"),
        vm.ToolBlock(
            tool="read",
            arg="/quantized/luca/luca/store/reader.py",
            status="ok",
            result=vm.ToolResult(summary="84 lines"),
        ),
    ]


def test_the_mention_path_does_not_depend_on_where_you_run_from(monkeypatch, tmp_path):
    """`home_path` resolves relative paths against the cwd and contracts
    anything under `~`; the world stores an absolute path outside both."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    row = SCREENS["chat"](scene("planning")).transcript[1]
    assert row.arg == "/quantized/luca/luca/store/reader.py"


def test_the_plan_shows_its_first_step_active():
    plan = next(b for b in SCREENS["chat"](scene("planning")).transcript if b.kind == "list")
    assert plan == vm.ListBlock(
        label="plan · 1 of 4",
        rows=[
            vm.ListRow(glyph="active", text="audit the reader and draft the split"),
            vm.ListRow(glyph="pending", text="move the write path into writer.py"),
            vm.ListRow(glyph="pending", text="update every import site"),
            vm.ListRow(glyph="pending", text="backfill the tests"),
        ],
    )


def test_the_first_plan_step_fans_out_into_two_tasks_one_reading_one_editing():
    tasks = [b for b in SCREENS["chat"](scene("planning")).transcript if b.kind == "task"]
    assert [(t.description, t.status, [c.tool for c in t.blocks if c.kind == "tool"]) for t in tasks] == [
        ("audit the reader", "done", ["read", "read", "grep"]),
        ("draft the writer", "done", ["write", "edit"]),
    ]
