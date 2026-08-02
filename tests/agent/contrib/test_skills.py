"""Self-scoped contrib tests: frontmatter parsing, discovery and precedence,
the `skill` tool, and the `SkillsPlugin` surface. No runner."""

from pathlib import Path

import pytest

from luca.agent.contrib.skills import (
    Skill,
    SkillsPlugin,
    SkillTool,
    default_locations,
    discover_skills,
    format_skill_listing,
    load_skill,
    parse_frontmatter,
    resolve_locations,
)
from luca.agent.core import CancellationToken
from luca.agent.core.models import ToolKind

BLOCK_SCALAR_SKILL = """---
name: railway
description: >
  Operate Railway infrastructure: create projects, provision
  services, deploy code.
---

# Railway

Body text.
"""

LITERAL_SCALAR_SKILL = """---
name: humanizer
version: 2.5.1
license: MIT
allowed-tools: [read, write]
description: |
  Remove signs of AI writing.
  Use when editing text.
---
Body.
"""


def write_skill(root, name, text):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(text)
    return directory


def simple(name, description="does a thing", body="Body."):
    return f"---\nname: {name}\ndescription: {description}\n---\n{body}"


# ── frontmatter ──────────────────────────────────────────────────────────────


def test_a_plain_scalar_header_parses_and_the_body_is_returned():
    assert parse_frontmatter("---\nname: a\ndescription: b\n---\nBody.") == (
        {"name": "a", "description": "b"},
        "Body.",
    )


def test_a_folded_block_scalar_is_joined_into_one_line(tmp_path):
    write_skill(tmp_path, "railway", BLOCK_SCALAR_SKILL)

    skill = load_skill(tmp_path / "railway")

    assert skill.description == "Operate Railway infrastructure: create projects, provision services, deploy code."


def test_a_literal_block_scalar_and_unknown_keys_are_handled(tmp_path):
    write_skill(tmp_path, "humanizer", LITERAL_SCALAR_SKILL)

    skill = load_skill(tmp_path / "humanizer")

    assert (skill.name, skill.description, skill.body) == (
        "humanizer",
        "Remove signs of AI writing. Use when editing text.",
        "Body.",
    )


def test_a_file_with_no_header_is_all_body():
    assert parse_frontmatter("# Just markdown\n") == ({}, "# Just markdown\n")


def test_an_unterminated_fence_is_not_a_header():
    assert parse_frontmatter("---\nname: a\nno closing fence") == ({}, "---\nname: a\nno closing fence")


def test_malformed_yaml_degrades_to_no_header():
    assert parse_frontmatter("---\n[unclosed\n---\nBody.") == ({}, "---\n[unclosed\n---\nBody.")


# ── loading one skill ────────────────────────────────────────────────────────


def test_the_directory_name_is_used_when_the_header_names_none(tmp_path):
    write_skill(tmp_path, "from-directory", "---\ndescription: d\n---\nBody.")

    assert load_skill(tmp_path / "from-directory").name == "from-directory"


@pytest.mark.parametrize(
    "text",
    [
        "no frontmatter at all",
        "---\n- a list, not a mapping\n---\nBody.",
        "---\nname: nameless\n---\nBody.",  # no description
        "---\nname: n\ndescription: '   '\n---\nBody.",  # blank description
    ],
    ids=["no-header", "not-a-mapping", "no-description", "blank-description"],
)
def test_a_skill_without_a_usable_description_is_skipped(tmp_path, text):
    write_skill(tmp_path, "broken", text)

    assert load_skill(tmp_path / "broken") is None


def test_a_directory_without_a_skill_file_is_skipped(tmp_path):
    (tmp_path / "empty").mkdir()

    assert load_skill(tmp_path / "empty") is None


# ── locations and discovery ──────────────────────────────────────────────────


def test_the_four_default_locations_are_project_then_global(tmp_path):
    assert default_locations(tmp_path, home=Path("/home/u")) == [
        tmp_path / ".claude" / "skills",
        tmp_path / ".agents" / "skills",
        Path("/home/u/.claude/skills"),
        Path("/home/u/.agents/skills"),
    ]


def test_extra_locations_sit_between_project_and_global(tmp_path):
    resolved = resolve_locations(tmp_path, ["~/elsewhere/skills"], home=Path("/home/u"))

    assert resolved == [
        tmp_path / ".claude" / "skills",
        tmp_path / ".agents" / "skills",
        Path.home() / "elsewhere" / "skills",
        Path("/home/u/.claude/skills"),
        Path("/home/u/.agents/skills"),
    ]


def test_every_default_location_is_scanned(tmp_path):
    home = tmp_path / "home"
    write_skill(tmp_path / ".claude" / "skills", "a", simple("a"))
    write_skill(tmp_path / ".agents" / "skills", "b", simple("b"))
    write_skill(home / ".claude" / "skills", "c", simple("c"))
    write_skill(home / ".agents" / "skills", "d", simple("d"))

    found = discover_skills(default_locations(tmp_path, home=home))

    assert sorted(found) == ["a", "b", "c", "d"]


def test_the_first_location_wins_a_name_collision(tmp_path):
    home = tmp_path / "home"
    write_skill(tmp_path / ".claude" / "skills", "shared", simple("shared", "the project one"))
    write_skill(home / ".claude" / "skills", "shared", simple("shared", "the personal one"))

    found = discover_skills(default_locations(tmp_path, home=home))

    assert found["shared"].description == "the project one"


def test_missing_roots_are_simply_empty(tmp_path):
    assert discover_skills(default_locations(tmp_path, home=tmp_path / "nope")) == {}


def test_one_broken_skill_does_not_hide_its_neighbours(tmp_path):
    root = tmp_path / ".claude" / "skills"
    write_skill(root, "good", simple("good"))
    write_skill(root, "broken", "no frontmatter")

    assert sorted(discover_skills([root])) == ["good"]


# ── the skill tool ───────────────────────────────────────────────────────────


async def test_the_tool_returns_the_body_and_the_bundled_file_directory(tmp_path):
    directory = write_skill(tmp_path, "writing", simple("writing", body="Rule one."))
    tool = SkillTool(discover_skills([tmp_path]))

    result = await tool._execute(
        {"name": "writing"},
        session=None,
        conversation_id="c1",
        cancellation_token=CancellationToken(),
    )

    assert result == f"Skill: writing\nBundled files are in: {directory}\n\nRule one."


async def test_an_unknown_name_reports_what_is_available(tmp_path):
    write_skill(tmp_path, "writing", simple("writing"))
    tool = SkillTool(discover_skills([tmp_path]))

    result = await tool._execute(
        {"name": "nope"},
        session=None,
        conversation_id="c1",
        cancellation_token=CancellationToken(),
    )

    assert result == "No skill named 'nope'. Available: writing."


def test_the_tool_spec_is_a_read_tool_taking_one_name():
    spec = SkillTool({}).get_tool_spec()

    assert (spec.name, spec.tool_kind, spec.namespace) == ("skill", ToolKind.READ, "contrib.skills")
    assert list(spec.input_schema["properties"]) == ["name"]


# ── the plugin ───────────────────────────────────────────────────────────────


def test_the_prompt_part_lists_name_and_description_but_not_bodies(tmp_path):
    write_skill(tmp_path / ".claude" / "skills", "writing", simple("writing", "Edit prose.", "SECRET BODY"))
    plugin = SkillsPlugin(workspace=tmp_path, home=tmp_path / "nohome")

    part = plugin._prompt_part(None, "c1")

    assert part.source == "skills"
    assert "- writing: Edit prose." in part.text
    assert "SECRET BODY" not in part.text


def test_the_prompt_part_is_absent_when_nothing_is_installed(tmp_path):
    plugin = SkillsPlugin(workspace=tmp_path, home=tmp_path / "nohome")

    assert plugin._prompt_part(None, "c1") is None


def test_only_roots_that_produced_a_skill_are_offered_for_read_access(tmp_path):
    write_skill(tmp_path / ".claude" / "skills", "writing", simple("writing"))
    (tmp_path / ".agents" / "skills").mkdir(parents=True)
    plugin = SkillsPlugin(workspace=tmp_path, home=tmp_path / "nohome")

    assert plugin.skill_directories == [tmp_path / ".claude" / "skills"]


def test_extra_locations_are_scanned(tmp_path):
    write_skill(tmp_path / "vendor", "vendored", simple("vendored"))
    plugin = SkillsPlugin(
        workspace=tmp_path,
        extra_locations=[str(tmp_path / "vendor")],
        home=tmp_path / "nohome",
    )

    assert sorted(plugin.skills) == ["vendored"]


def test_the_listing_is_sorted_so_the_prompt_is_stable():
    skills = {
        name: Skill(name=name, description=name, path=Path("p"), directory=Path("d"), body="b")
        for name in ("zebra", "apple")
    }

    assert format_skill_listing(skills).splitlines()[-2:] == ["- apple: apple", "- zebra: zebra"]
