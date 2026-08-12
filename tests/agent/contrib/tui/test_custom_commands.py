"""User-defined slash commands: reading `.md` files off disk, turning them
into palette entries, and running one.

The discovery half is pure and takes an explicit `home`, so it is asserted
directly. The app half goes through the real `dispatch` / palette / `/help`
path — a custom command has to be indistinguishable from a built-in once it is
loaded, and the only way to show that is to run it.
"""

from pathlib import Path

from luca.agent.contrib.tui import state as vm
from luca.agent.contrib.tui.app import AgentApp
from luca.agent.contrib.tui.blocks import ListBlockView, UserTurn
from luca.agent.contrib.tui.commands import (
    BUILTIN_NAMES,
    COMMANDS,
    dispatch,
    load_custom_commands,
    palette_rows,
    to_slash_commands,
)
from luca.agent.contrib.tui.custom_commands import (
    CustomCommand,
    default_locations,
    discover_commands,
    expand,
    load_command,
    resolve_locations,
)
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text

from .helpers import fresh_session, submit

REVIEW = """---
description: review the working tree
argument-hint: "[path]"
---
Review $ARGUMENTS and report only real defects.
"""


def write(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def scripted(*responses) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return provider


def agent_app(tmp_path, provider=None) -> AgentApp:
    return AgentApp(
        fresh_session(),
        provider=provider,
        workspace=tmp_path,
        session_dir=tmp_path,
        skills=False,
        instructions=False,
    )


# ── reading one file ─────────────────────────────────────────────────────────


def test_a_command_file_is_read_whole(tmp_path):
    path = write(tmp_path, "review.md", REVIEW)

    assert load_command(path) == CustomCommand(
        name="review",
        description="review the working tree",
        argument_hint="[path]",
        body="Review $ARGUMENTS and report only real defects.",
        path=path,
    )


def test_a_file_without_frontmatter_falls_back_to_the_stem_and_first_line(tmp_path):
    path = write(tmp_path, "standup.md", "# Daily standup\n\nSummarize what changed since yesterday.\n")

    assert load_command(path) == CustomCommand(
        name="standup",
        description="Daily standup",
        argument_hint="",
        body="# Daily standup\n\nSummarize what changed since yesterday.",
        path=path,
    )


def test_frontmatter_name_overrides_the_stem(tmp_path):
    path = write(tmp_path, "not-this.md", "---\nname: ship\n---\nCut a release.\n")

    assert load_command(path) == CustomCommand(
        name="ship",
        description="Cut a release.",
        argument_hint="",
        body="Cut a release.",
        path=path,
    )


def test_an_empty_body_is_not_a_command(tmp_path):
    assert load_command(write(tmp_path, "blank.md", "---\ndescription: nothing\n---\n\n")) is None


def test_a_name_that_cannot_be_typed_after_a_slash_is_rejected(tmp_path):
    assert load_command(write(tmp_path, "my command.md", "Do the thing.")) is None


def test_an_unreadable_file_is_skipped(tmp_path):
    assert load_command(tmp_path / "missing.md") is None


# ── where they are read from ─────────────────────────────────────────────────


def test_the_four_default_locations_are_project_then_home(tmp_path):
    assert default_locations(tmp_path, home=Path("/home/u")) == [
        tmp_path / ".claude" / "commands",
        tmp_path / ".agents" / "commands",
        Path("/home/u/.claude/commands"),
        Path("/home/u/.agents/commands"),
    ]


def test_extra_locations_sit_between_project_and_home(tmp_path):
    assert resolve_locations(tmp_path, ["~/elsewhere/commands"], home=Path("/home/u")) == [
        tmp_path / ".claude" / "commands",
        tmp_path / ".agents" / "commands",
        Path.home() / "elsewhere/commands",
        Path("/home/u/.claude/commands"),
        Path("/home/u/.agents/commands"),
    ]


def test_the_first_location_wins_a_name_collision(tmp_path):
    write(tmp_path / ".claude" / "commands", "dup.md", "from claude")
    write(tmp_path / ".agents" / "commands", "dup.md", "from agents")

    found = discover_commands(default_locations(tmp_path, home=tmp_path / "nohome"))

    assert [(name, command.body) for name, command in found.items()] == [("dup", "from claude")]


def test_a_reserved_name_never_shadows_a_builtin(tmp_path):
    write(tmp_path / ".claude" / "commands", "quit.md", "rm -rf /")
    write(tmp_path / ".claude" / "commands", "mine.md", "ok")

    found = discover_commands(default_locations(tmp_path, home=tmp_path / "nohome"), reserved=BUILTIN_NAMES)

    assert sorted(found) == ["mine"]


def test_only_markdown_files_are_read(tmp_path):
    write(tmp_path / ".claude" / "commands", "notes.txt", "not a command")
    (tmp_path / ".claude" / "commands" / "sub").mkdir()

    assert discover_commands(default_locations(tmp_path, home=tmp_path / "nohome")) == {}


# ── filling in the argument ──────────────────────────────────────────────────


def test_arguments_token_takes_the_whole_argument():
    assert expand("Review $ARGUMENTS now.", "src/ and tests/") == "Review src/ and tests/ now."


def test_positionals_take_whitespace_split_words():
    assert expand("Move $1 to $2.", "alpha beta") == "Move alpha to beta."


def test_a_missing_positional_resolves_to_nothing():
    assert expand("Move $1 to $2.", "alpha") == "Move alpha to ."


def test_a_body_with_no_placeholder_still_gets_the_argument():
    assert expand("Review the diff.", "src/") == "Review the diff.\n\nsrc/"


def test_a_body_with_no_placeholder_and_no_argument_is_unchanged():
    assert expand("Review the diff.", "") == "Review the diff."


# ── becoming palette entries ─────────────────────────────────────────────────


def test_discovered_files_become_sorted_palette_entries(tmp_path):
    write(tmp_path / ".claude" / "commands", "review.md", REVIEW)
    write(tmp_path / ".claude" / "commands", "audit.md", "Audit the dependencies.")

    entries = to_slash_commands(discover_commands(default_locations(tmp_path, home=tmp_path / "nohome")))

    # a declared argument-hint is what makes a pick insert rather than run
    assert [(c.name, c.usage, c.summary, c.insert) for c in entries] == [
        ("audit", "", "Audit the dependencies.", False),
        ("review", "[path]", "review the working tree", True),
    ]


def test_a_broken_commands_directory_costs_the_user_nothing_else(tmp_path):
    write(tmp_path / ".claude" / "commands", "review.md", "---\n: not: yaml:\n---\nstill a body")

    assert [c.name for c in load_custom_commands(tmp_path)] == ["review"]


# ── running one ──────────────────────────────────────────────────────────────


async def test_a_custom_command_is_listed_by_help_and_the_palette(tmp_path):
    write(tmp_path / ".claude" / "commands", "review.md", REVIEW)
    app = agent_app(tmp_path)
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/help")
        await pilot.pause()

        assert palette_rows(app) == [
            *(vm.OverlayRow(primary=f"/{c.name}", secondary=c.summary) for c in COMMANDS),
            vm.OverlayRow(primary="/review", secondary="review the working tree"),
        ]
        listed = app.transcript.query_one(ListBlockView).model
        assert (listed.label, listed.rows[-1]) == (
            f"commands · {len(COMMANDS) + 1}",
            vm.ListRow(glyph="none", text="/review [path]", annotation="review the working tree"),
        )


async def test_running_a_custom_command_sends_its_expanded_body(tmp_path):
    write(tmp_path / ".claude" / "commands", "review.md", REVIEW)
    app = agent_app(tmp_path, provider=scripted(faux_assistant_message([faux_text("Looks fine.")])))
    async with app.run_test(size=(105, 35)) as pilot:
        await submit(pilot, "/review src/")
        await pilot.pause()

        assert [turn.text for turn in app.query(UserTurn)] == ["Review src/ and report only real defects."]


async def test_no_commands_loads_none_of_them(tmp_path):
    write(tmp_path / ".claude" / "commands", "review.md", REVIEW)
    app = AgentApp(
        fresh_session(),
        workspace=tmp_path,
        session_dir=tmp_path,
        skills=False,
        instructions=False,
        commands=False,
    )
    async with app.run_test(size=(105, 35)) as pilot:
        handled = await dispatch(app, "/review")
        await pilot.pause()

        assert (app.custom_commands, handled) == ((), False)
