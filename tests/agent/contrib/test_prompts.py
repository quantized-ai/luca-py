"""Self-scoped contrib tests: model-family selection, the environment block,
instruction discovery and precedence, and the two plugins. No runner."""

from datetime import date
from pathlib import Path

import pytest

from luca.agent.contrib.prompts import (
    FAMILIES,
    GENERIC,
    InstructionFile,
    InstructionsPlugin,
    SystemPromptPlugin,
    apply_budget,
    environment_text,
    find_instructions,
    first_in_directory,
    load_prompt,
    project_directories,
    select_family,
)
from luca.agent.core import AgentSessionRunner
from luca.agent.core.models import LLMConfig


def session(model="anthropic/claude-sonnet-5", provider="openrouter"):
    return AgentSessionRunner.new_session(LLMConfig(model=model, provider=provider))


def repo(root: Path) -> Path:
    """A directory that bounds the upward walk, the way a checkout does."""
    (root / ".git").mkdir(parents=True)
    return root


def resolved(*parts: InstructionFile) -> list[Path]:
    return [file.path for file in parts]


# ── family selection ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("anthropic/claude-sonnet-5", "anthropic"),
        ("claude-opus-4-8", "anthropic"),
        ("openai/gpt-5.4-mini", "gpt"),
        ("gpt-5.4", "gpt"),
        ("openai/o3-pro", "gpt"),
        ("gemini-2.5-pro", "gemini"),
        ("moonshotai/kimi-k2.7-code", GENERIC),
        ("meta-llama/llama-3.3-70b-instruct", GENERIC),
        ("fake-model", GENERIC),
    ],
)
def test_the_family_comes_from_the_model_id(model, family):
    assert select_family(model) == family


def test_the_provider_never_decides_the_family():
    # openrouter serves every family, so the provider says nothing useful
    assert select_family("anthropic/claude-sonnet-5") == select_family("claude-sonnet-5")


def test_every_family_in_the_table_has_a_prompt_to_load():
    for family, _ in [*FAMILIES, (GENERIC, ())]:
        assert load_prompt(family).strip()


# ── the environment block ────────────────────────────────────────────────────


def test_the_environment_block_renders_every_field():
    assert environment_text(
        workspace="/w",
        model="claude-sonnet-5",
        provider="anthropic",
        platform_name="Darwin",
        today=date(2026, 8, 3),
        is_git_repo=True,
    ) == (
        "### Environment\n"
        "You are powered by the model claude-sonnet-5 on the anthropic provider.\n"
        "Working directory: /w\n"
        "Is a git repository: yes\n"
        "Platform: Darwin\n"
        "Today's date: 2026-08-03"
    )


def test_a_workspace_outside_a_repo_says_so():
    text = environment_text(
        workspace="/w",
        model="m",
        provider="p",
        platform_name="Linux",
        today=date(2026, 8, 3),
        is_git_repo=False,
    )

    assert "Is a git repository: no" in text


# ── the bounded walk ─────────────────────────────────────────────────────────


def test_the_chain_runs_from_the_git_root_down_to_the_workspace(tmp_path):
    repo(tmp_path)
    workspace = tmp_path / "packages" / "api"
    workspace.mkdir(parents=True)

    assert project_directories(workspace) == [tmp_path, tmp_path / "packages", workspace]


def test_without_a_repository_the_walk_is_the_workspace_alone(tmp_path):
    # An unbounded walk would climb toward / and read files from outside the
    # project entirely — which is what the repo-less case used to do.
    workspace = tmp_path / "loose"
    workspace.mkdir()

    assert project_directories(workspace) == [workspace]


def test_a_worktree_git_file_bounds_the_walk_too(tmp_path):
    (tmp_path / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt")
    workspace = tmp_path / "src"
    workspace.mkdir()

    assert project_directories(workspace) == [tmp_path, workspace]


def test_a_symlinked_workspace_still_finds_its_repository(tmp_path):
    # The #21 regression: comparing an unresolved path against a resolved one
    # silently never matches when a symlink is in play (/tmp vs /private/tmp).
    repo(tmp_path)
    (tmp_path / "src").mkdir()
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "src")

    assert project_directories(link) == [tmp_path.resolve(), (tmp_path / "src").resolve()]


# ── name precedence and discovery ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("present", "winner"),
    [
        (("LUCA.md", "AGENTS.md", "CLAUDE.md"), "LUCA.md"),
        (("AGENTS.md", "CLAUDE.md"), "AGENTS.md"),
        (("CLAUDE.md",), "CLAUDE.md"),
    ],
    ids=["luca-wins", "agents-beats-claude", "claude-alone"],
)
def test_one_directory_contributes_one_file_by_name_precedence(tmp_path, present, winner):
    for name in present:
        (tmp_path / name).write_text(f"from {name}")

    assert first_in_directory(tmp_path) == InstructionFile(
        path=tmp_path / winner,
        text=f"from {winner}",
    )


def test_a_claude_md_that_only_imports_agents_md_is_never_the_one_read(tmp_path):
    # The Claude Code interop pattern: CLAUDE.md is a one-line pointer. Name
    # precedence within the directory means AGENTS.md is what actually loads.
    (tmp_path / "CLAUDE.md").write_text("@AGENTS.md")
    (tmp_path / "AGENTS.md").write_text("The real rules.")

    assert first_in_directory(tmp_path).text == "The real rules."


def test_a_directory_with_nothing_contributes_nothing(tmp_path):
    assert first_in_directory(tmp_path) is None


def test_an_empty_instruction_file_is_skipped(tmp_path):
    (tmp_path / "AGENTS.md").write_text("   \n\n")

    assert first_in_directory(tmp_path) is None


def test_each_directory_in_the_chain_contributes_its_own_file(tmp_path):
    # Different NAMES at different levels: precedence is per directory, so the
    # subpackage's CLAUDE.md is not shadowed by the root's AGENTS.md.
    repo(tmp_path)
    (tmp_path / "AGENTS.md").write_text("root rules")
    workspace = tmp_path / "packages" / "api"
    workspace.mkdir(parents=True)
    (workspace / "CLAUDE.md").write_text("api rules")

    found = find_instructions(workspace, config_dir=tmp_path / "nope")

    assert resolved(*found) == [tmp_path / "AGENTS.md", workspace / "CLAUDE.md"]


def test_the_global_file_comes_first_so_the_project_is_read_last(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "LUCA.md").write_text("personal")
    workspace = repo(tmp_path / "repo")
    (workspace / "AGENTS.md").write_text("project")

    found = find_instructions(workspace, config_dir=config_dir)

    assert [file.text for file in found] == ["personal", "project"]


def test_the_global_tier_is_one_file_by_the_same_precedence(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "AGENTS.md").write_text("global agents")
    (config_dir / "CLAUDE.md").write_text("global claude")
    workspace = repo(tmp_path / "repo")

    found = find_instructions(workspace, config_dir=config_dir)

    assert [file.text for file in found] == ["global agents"]


def test_extra_entries_are_read_last_and_resolve_against_the_workspace(tmp_path):
    workspace = repo(tmp_path)
    (workspace / "AGENTS.md").write_text("project")
    (workspace / "docs").mkdir()
    (workspace / "docs" / "conventions.md").write_text("conventions")

    found = find_instructions(workspace, ["docs/conventions.md"], config_dir=tmp_path / "nope")

    assert [file.text for file in found] == ["project", "conventions"]


def test_a_missing_extra_entry_is_simply_skipped(tmp_path):
    workspace = repo(tmp_path)

    assert find_instructions(workspace, ["nope.md"], config_dir=tmp_path / "nope") == []


def test_the_same_file_reached_twice_is_only_read_once(tmp_path):
    workspace = repo(tmp_path)
    (workspace / "AGENTS.md").write_text("project")

    found = find_instructions(workspace, ["AGENTS.md"], config_dir=tmp_path / "nope")

    assert resolved(*found) == [workspace / "AGENTS.md"]


def test_a_workspace_outside_any_repository_reads_nothing_from_above(tmp_path):
    parent = tmp_path / "outside"
    parent.mkdir()
    (parent / "AGENTS.md").write_text("not this project's")
    workspace = parent / "loose"
    workspace.mkdir()

    assert find_instructions(workspace, config_dir=tmp_path / "nope") == []


# ── the byte budget ──────────────────────────────────────────────────────────


def test_the_budget_drops_the_least_specific_file_first():
    # Inverted from Codex, which fills its budget in lookup order and lets a
    # bloated personal file starve the repo's own rules.
    personal = InstructionFile(path=Path("/global/LUCA.md"), text="g" * 900)
    project = InstructionFile(path=Path("/repo/AGENTS.md"), text="p" * 200)

    assert apply_budget([personal, project], max_bytes=1000) == [project]


def test_the_most_specific_file_survives_however_large():
    huge = InstructionFile(path=Path("/repo/AGENTS.md"), text="p" * 5000)

    assert apply_budget([huge], max_bytes=10) == [huge]


def test_everything_that_fits_is_kept():
    files = [InstructionFile(path=Path(f"/{n}"), text="x" * 10) for n in "ab"]

    assert apply_budget(files, max_bytes=1000) == files


# ── SystemPromptPlugin ───────────────────────────────────────────────────────


def test_the_prompt_parts_are_the_base_the_family_and_the_environment(tmp_path):
    plugin = SystemPromptPlugin(workspace=tmp_path, today=date(2026, 8, 3))

    parts = [part(session(), "c1") for part in plugin.get_system_prompt_parts(None)]

    assert [(part.source, part.priority) for part in parts] == [
        ("prompt", -1),
        ("prompt.anthropic", -1),
        ("env", 90),
    ]


def test_the_family_part_follows_the_session_model(tmp_path):
    # `/model` reassigns llm_config mid-session, which is why the parts are
    # callables rather than static text resolved once at construction.
    plugin = SystemPromptPlugin(workspace=tmp_path)
    _, family_part, _ = plugin.get_system_prompt_parts(None)

    anthropic = family_part(session("anthropic/claude-sonnet-5"), "c1")
    gpt = family_part(session("openai/gpt-5.4-mini"), "c1")

    assert (anthropic.source, gpt.source) == ("prompt.anthropic", "prompt.gpt")
    assert anthropic.text != gpt.text


def test_the_environment_can_be_withheld(tmp_path):
    plugin = SystemPromptPlugin(workspace=tmp_path, environment=False)

    parts = [part(session(), "c1") for part in plugin.get_system_prompt_parts(None)]

    assert [part.source for part in parts] == ["prompt", "prompt.anthropic"]


def test_the_environment_part_reports_the_repository(tmp_path):
    workspace = repo(tmp_path)
    plugin = SystemPromptPlugin(workspace=workspace, today=date(2026, 8, 3))

    part = plugin.get_system_prompt_parts(None)[2](session("m", "p"), "c1")

    assert part.text == environment_text(
        workspace=workspace.resolve(),
        model="m",
        provider="p",
        platform_name=__import__("platform").system(),
        today=date(2026, 8, 3),
        is_git_repo=True,
    )


# ── InstructionsPlugin ───────────────────────────────────────────────────────


def test_the_instructions_part_labels_every_file_it_carries(tmp_path):
    workspace = repo(tmp_path)
    (workspace / "AGENTS.md").write_text("Follow the house style.")
    plugin = InstructionsPlugin(workspace=workspace, config_dir=tmp_path / "nope")

    part = plugin._prompt_part(session(), "c1")

    assert part.source == "agents.md"
    assert part.priority == 100
    assert part.text.endswith(f"--- {workspace / 'AGENTS.md'} ---\nFollow the house style.")


def test_the_instructions_part_is_absent_when_nothing_is_on_disk(tmp_path):
    plugin = InstructionsPlugin(workspace=repo(tmp_path), config_dir=tmp_path / "nope")

    assert plugin._prompt_part(session(), "c1") is None
