"""Unit tests for the shell `glob` tool. The ripgrep subprocess is mocked at
the `_run_ripgrep` boundary: tests assert the argv/cwd the tool builds and
how it renders ripgrep's file list."""

import pytest
from pydantic import ValidationError

from luca.agent.contrib.resource_permissions import ResourcePermission
from luca.agent.contrib.shell import GlobTool

RG = "/fake/bin/rg"


def make_tool(tmp_path, stdout="", stderr="", code=0):
    """A GlobTool whose ripgrep invocation is canned; returns (tool, calls)."""
    tool = GlobTool(workdir=tmp_path, rg_path=RG)
    calls = []

    async def fake_run_ripgrep(argv, cwd):
        calls.append((argv, cwd))
        return stdout, stderr, code

    tool._run_ripgrep = fake_run_ripgrep
    return tool, calls


def body(result) -> str:
    return result.content[0].text


# ── scenario 1: match one extension ───────────────────────────────────────────


async def test_returns_absolute_paths_for_matches(tmp_path, run):
    tool, calls = make_tool(tmp_path, stdout="a.ts\n")

    result = await run(tool, {"pattern": "*.ts"})

    assert result.is_error is False
    assert body(result) == str(tmp_path / "a.ts")
    assert result.metadata == {"truncated": False, "count": 1}
    argv, cwd = calls[0]
    assert argv[0] == RG
    assert "--files" in argv
    assert argv[-2:] == ["--glob", "*.ts"]
    assert cwd == tmp_path


# ── scenario 2: recursive pattern ─────────────────────────────────────────────


async def test_nested_matches_resolve_below_the_search_root(tmp_path, run):
    tool, _ = make_tool(tmp_path, stdout="src/app/main.ts\n")

    result = await run(tool, {"pattern": "**/*.ts"})

    assert body(result) == str(tmp_path / "src/app/main.ts")


# ── scenario 3: explicit search directory ─────────────────────────────────────


async def test_relative_path_resolves_from_the_workdir(tmp_path, run):
    (tmp_path / "src").mkdir()
    tool, calls = make_tool(tmp_path, stdout="")

    await run(tool, {"pattern": "*.py", "path": "src"})

    assert calls[0][1] == tmp_path / "src"


# ── scenario 4: file path is rejected ─────────────────────────────────────────


async def test_a_regular_file_as_path_is_rejected(tmp_path, run):
    (tmp_path / "file.txt").write_text("")
    tool, calls = make_tool(tmp_path)

    result = await run(tool, {"pattern": "*.py", "path": "file.txt"})

    assert result.is_error is True
    assert body(result) == f"glob path must be a directory: {tmp_path / 'file.txt'}"
    assert calls == []


async def test_a_missing_path_is_rejected(tmp_path, run):
    tool, calls = make_tool(tmp_path)

    result = await run(tool, {"pattern": "*.py", "path": "nope"})

    assert result.is_error is True
    assert body(result) == f"glob path does not exist: {tmp_path / 'nope'}"
    assert calls == []


# ── scenario 5: no matches ────────────────────────────────────────────────────


async def test_no_matches_returns_exactly_no_files_found(tmp_path, run):
    tool, _ = make_tool(tmp_path, stdout="", code=1)

    result = await run(tool, {"pattern": "*.zig"})

    assert result.is_error is False
    assert body(result) == "No files found"
    assert result.metadata == {"truncated": False, "count": 0}


# ── scenario 6: ignore behavior ───────────────────────────────────────────────


async def test_argv_includes_hidden_files_but_excludes_git_metadata(tmp_path, run):
    tool, calls = make_tool(tmp_path, stdout="")

    await run(tool, {"pattern": "*"})

    argv, _ = calls[0]
    assert "--hidden" in argv
    git_exclusion = argv.index("!**/.git/**")
    assert argv[git_exclusion - 1] == "--glob"


# ── scenario 7: result limit ──────────────────────────────────────────────────


async def test_exactly_100_results_is_treated_as_truncated(tmp_path, run):
    listing = "".join(f"f{n}.py\n" for n in range(100))
    tool, _ = make_tool(tmp_path, stdout=listing)

    result = await run(tool, {"pattern": "*.py"})

    assert result.metadata == {"truncated": True, "count": 100}
    assert (
        "(Results are truncated: showing first 100 results."
        " Consider using a more specific path or pattern.)"
    ) in body(result)


async def test_more_than_100_results_are_capped_at_100(tmp_path, run):
    listing = "".join(f"f{n}.py\n" for n in range(150))
    tool, _ = make_tool(tmp_path, stdout=listing)

    result = await run(tool, {"pattern": "*.py"})

    assert result.metadata == {"truncated": True, "count": 100}
    assert str(tmp_path / "f99.py") in body(result)
    assert str(tmp_path / "f100.py") not in body(result)


async def test_99_results_are_not_truncated(tmp_path, run):
    listing = "".join(f"f{n}.py\n" for n in range(99))
    tool, _ = make_tool(tmp_path, stdout=listing)

    result = await run(tool, {"pattern": "*.py"})

    assert result.metadata == {"truncated": False, "count": 99}
    assert "(Results are truncated" not in body(result)


# ── failures ──────────────────────────────────────────────────────────────────


async def test_ripgrep_failure_surfaces_stderr(tmp_path, run):
    tool, _ = make_tool(tmp_path, stderr="rg exploded\n", code=2)

    result = await run(tool, {"pattern": "*"})

    assert result.is_error is True
    assert body(result) == "glob failed: rg exploded"


async def test_missing_rg_binary_is_an_error(tmp_path, run, monkeypatch):
    monkeypatch.setattr("luca.agent.contrib.shell.tools.shutil.which", lambda _: None)
    tool = GlobTool(workdir=tmp_path)

    result = await run(tool, {"pattern": "*"})

    assert result.is_error is True
    assert body(result) == "ripgrep (rg) was not found on PATH"


# ── arguments ─────────────────────────────────────────────────────────────────


def test_args_require_a_pattern_and_forbid_extras():
    with pytest.raises(ValidationError):
        GlobTool.Args.model_validate({})
    with pytest.raises(ValidationError):
        GlobTool.Args.model_validate({"pattern": ""})
    with pytest.raises(ValidationError):
        GlobTool.Args.model_validate({"pattern": "*", "surprise": 1})


# ── permission resource ───────────────────────────────────────────────────────


def test_permission_resource_defaults_to_the_workdir(tmp_path, perm):
    tool, _ = make_tool(tmp_path)

    [access, request] = perm(tool, {"pattern": "*.ts"})

    assert access.resources == [
        ResourcePermission(permission="access_directory", resource=str(tmp_path)),
    ]
    assert access.metadata["preview"] == f"Access directory {tmp_path}"
    assert [
        (o.resource_permissions, o.metadata["preview"])
        for o in access.answer_options
    ] == [
        (
            [
                ResourcePermission(
                    permission="access_directory", resource=str(tmp_path),
                ),
                ResourcePermission(
                    permission="access_directory", resource=f"{tmp_path}/*",
                ),
            ],
            f"Always allow access to {tmp_path}",
        ),
    ]
    assert request.resources == [
        ResourcePermission(permission="glob", resource=str(tmp_path)),
    ]
    assert request.metadata["preview"] == f'Find files matching "*.ts" in {tmp_path}'
    assert [
        (o.resource_permissions, o.metadata["preview"])
        for o in request.answer_options
    ] == [
        (
            [ResourcePermission(permission="glob", resource=f"{tmp_path}/*")],
            f"Search files under {tmp_path}",
        ),
    ]


def test_permission_resource_uses_the_explicit_search_directory(tmp_path, perm):
    tool, _ = make_tool(tmp_path)

    [access, request] = perm(tool, {"pattern": "*.py", "path": "src"})

    assert access.resources == [
        ResourcePermission(
            permission="access_directory", resource=str(tmp_path / "src"),
        ),
    ]
    assert request.resources == [
        ResourcePermission(permission="glob", resource=str(tmp_path / "src")),
    ]
    assert request.metadata["preview"] == (
        f'Find files matching "*.py" in {tmp_path / "src"}'
    )
