"""Unit tests for the shell `grep` tool. The ripgrep subprocess is mocked at
the `_run_ripgrep` boundary with canned `--json` events; tests assert the
argv the tool builds and how it groups, caps, and renders matches."""

import json

import pytest
from pydantic import ValidationError

from luca.agent.contrib.resource_permissions import ResourcePermission
from luca.agent.contrib.shell import GrepTool

RG = "/fake/bin/rg"


def match_event(path: str, line_number: int, text: str) -> str:
    return json.dumps({
        "type": "match",
        "data": {
            "path": {"text": path},
            "lines": {"text": text + "\n"},
            "line_number": line_number,
            "absolute_offset": 0,
            "submatches": [],
        },
    })


def make_tool(tmp_path, stdout="", stderr="", code=0):
    tool = GrepTool(workdir=tmp_path, rg_path=RG)
    calls = []

    async def fake_run_ripgrep(argv, cwd):
        calls.append((argv, cwd))
        return stdout, stderr, code

    tool._run_ripgrep = fake_run_ripgrep
    return tool, calls


def body(result) -> str:
    return result.content[0].text


# ── scenario 1: basic regex search ────────────────────────────────────────────


async def test_matches_are_grouped_by_file_with_line_numbers(tmp_path, run):
    stdout = "\n".join([
        match_event("/workspace/a.py", 3, "first matching line"),
        match_event("/workspace/b.py", 8, "second matching line"),
    ])
    tool, calls = make_tool(tmp_path, stdout=stdout)

    result = await run(tool, {"pattern": "export"})

    assert result.is_error is False
    assert body(result) == (
        "Found 2 matches\n"
        "/workspace/a.py:\n"
        "  Line 3: first matching line\n"
        "\n"
        "/workspace/b.py:\n"
        "  Line 8: second matching line"
    )
    argv, _ = calls[0]
    assert argv[0] == RG
    assert "--json" in argv
    assert argv[-4:] == ["--regexp", "export", "--", str(tmp_path)]


async def test_two_matches_in_the_same_file_share_one_block(tmp_path, run):
    stdout = "\n".join([
        match_event("/w/a.py", 1, "one"),
        match_event("/w/a.py", 5, "two"),
    ])
    tool, _ = make_tool(tmp_path, stdout=stdout)

    result = await run(tool, {"pattern": "o"})

    assert body(result) == (
        "Found 2 matches\n/w/a.py:\n  Line 1: one\n  Line 5: two"
    )


# ── scenario 2: no matches ────────────────────────────────────────────────────


async def test_no_matches_returns_exactly_no_files_found(tmp_path, run):
    tool, _ = make_tool(tmp_path, stdout="", code=1)

    result = await run(tool, {"pattern": "absent"})

    assert result.is_error is False
    assert body(result) == "No files found"
    assert result.metadata == {"truncated": False, "count": 0}


# ── scenario 3: include filter ────────────────────────────────────────────────


async def test_include_pattern_is_passed_as_a_glob(tmp_path, run):
    tool, calls = make_tool(tmp_path, stdout="", code=1)

    await run(tool, {"pattern": "needle", "include": "*.ts"})

    argv, _ = calls[0]
    assert argv[argv.index("*.ts") - 1] == "--glob"


# ── scenario 4: exact file path ───────────────────────────────────────────────


async def test_an_exact_file_path_is_searched_directly(tmp_path, run):
    target = tmp_path / "test.txt"
    target.write_text("line1\nline2\n")
    tool, calls = make_tool(tmp_path, stdout=match_event(str(target), 2, "line2"))

    result = await run(tool, {"pattern": "line2", "path": "test.txt"})

    assert calls[0][0][-1] == str(target)
    assert body(result) == f"Found 1 match\n{target}:\n  Line 2: line2"


async def test_missing_search_path_is_an_error(tmp_path, run):
    tool, calls = make_tool(tmp_path)

    result = await run(tool, {"pattern": "x", "path": "nope"})

    assert result.is_error is True
    assert body(result) == f"grep path does not exist: {tmp_path / 'nope'}"
    assert calls == []


# ── scenario 5: invalid regex ─────────────────────────────────────────────────


async def test_invalid_regex_surfaces_the_parse_error(tmp_path, run):
    stderr = "regex parse error:\n    [unclosed\nerror: unclosed character class"
    tool, _ = make_tool(tmp_path, stderr=stderr, code=2)

    result = await run(tool, {"pattern": "[unclosed"})

    assert result.is_error is True
    assert "regex parse error" in body(result)


# ── scenario 6: result limit ──────────────────────────────────────────────────


async def test_101_matches_cap_at_100_without_inventing_a_total(tmp_path, run):
    stdout = "\n".join(
        match_event("/w/f.py", n, f"match {n}") for n in range(1, 102)
    )
    tool, _ = make_tool(tmp_path, stdout=stdout)

    result = await run(tool, {"pattern": "match"})

    assert body(result).startswith("Found 100 matches (more matches available)")
    assert "101" not in body(result).split("\n")[0]
    assert body(result).count("  Line ") == 100
    assert (
        "(Results truncated. Consider using a more specific path or pattern.)"
    ) in body(result)
    assert result.metadata == {"truncated": True, "count": 100}


async def test_exactly_100_matches_is_a_complete_result(tmp_path, run):
    stdout = "\n".join(
        match_event("/w/f.py", n, f"match {n}") for n in range(1, 101)
    )
    tool, _ = make_tool(tmp_path, stdout=stdout)

    result = await run(tool, {"pattern": "match"})

    assert body(result).startswith("Found 100 matches\n")
    assert "(more matches available)" not in body(result)
    assert result.metadata == {"truncated": False, "count": 100}


# ── scenario 7: hidden files yes, git metadata no ─────────────────────────────


async def test_argv_includes_hidden_but_excludes_git_metadata(tmp_path, run):
    tool, calls = make_tool(tmp_path, stdout="", code=1)

    await run(tool, {"pattern": "x"})

    argv, _ = calls[0]
    assert "--hidden" in argv
    git_exclusion = argv.index("!**/.git/**")
    assert argv[git_exclusion - 1] == "--glob"


# ── scenario 8: long matching line ────────────────────────────────────────────


async def test_long_matching_line_is_shortened_to_2000_chars(tmp_path, run):
    tool, _ = make_tool(tmp_path, stdout=match_event("/w/a.py", 1, "y" * 2_500))

    result = await run(tool, {"pattern": "y"})

    assert "y" * 2_000 + "..." in body(result)
    assert "y" * 2_001 not in body(result)


# ── arguments ─────────────────────────────────────────────────────────────────


def test_args_require_a_pattern_and_forbid_extras():
    with pytest.raises(ValidationError):
        GrepTool.Args.model_validate({"pattern": ""})
    with pytest.raises(ValidationError):
        GrepTool.Args.model_validate({"pattern": "x", "surprise": 1})


# ── permission resource ───────────────────────────────────────────────────────


def test_permission_resource_defaults_to_the_workdir(tmp_path, perm):
    tool, _ = make_tool(tmp_path)

    [access, request] = perm(tool, {"pattern": "needle"})

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
        ResourcePermission(permission="grep", resource=str(tmp_path)),
    ]
    assert request.metadata["preview"] == f'Search for "needle" in {tmp_path}'
    assert [
        (o.resource_permissions, o.metadata["preview"])
        for o in request.answer_options
    ] == [
        (
            [ResourcePermission(permission="grep", resource=f"{tmp_path}/*")],
            f"Search files under {tmp_path}",
        ),
    ]


def test_permission_resource_for_a_file_target_remembers_its_parent(tmp_path, perm):
    target = tmp_path / "test.txt"
    target.write_text("")
    tool, _ = make_tool(tmp_path)

    [access, request] = perm(tool, {"pattern": "x", "path": "test.txt"})

    assert access.resources == [
        ResourcePermission(permission="access_directory", resource=str(tmp_path)),
    ]
    assert request.resources == [
        ResourcePermission(permission="grep", resource=str(target)),
    ]
    assert request.answer_options[0].resource_permissions == [
        ResourcePermission(permission="grep", resource=f"{tmp_path}/*"),
    ]
