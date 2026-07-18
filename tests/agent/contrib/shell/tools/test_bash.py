"""Unit tests for the shell `bash` tool: real short-lived subprocesses cover
exit codes, merged output, workdir selection, timeout/cancellation kills,
output truncation to a saved file, and its permission resource."""

import asyncio
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from luca.agent.core import CancellationToken

from luca.agent.contrib.resource_permissions import ResourcePermission
from luca.agent.contrib.shell import BashTool
from luca.agent.contrib.shell.tools import BASH_DESCRIPTION_TEMPLATE


def make_tool(tmp_path, **kwargs) -> BashTool:
    kwargs.setdefault("shell", "/bin/bash")
    kwargs.setdefault("output_dir", tmp_path)
    return BashTool(workdir=tmp_path, **kwargs)


def body(result) -> str:
    return result.content[0].text


# ── scenario 1: successful command ────────────────────────────────────────────


async def test_successful_command_returns_output_and_exit_0(tmp_path, run):
    result = await run(make_tool(tmp_path), {"command": "echo test"})

    assert result.is_error is False
    assert "test" in body(result)
    assert result.metadata == {"exit": 0, "truncated": False, "output_path": None}


# ── scenario 2: workdir ───────────────────────────────────────────────────────


async def test_workdir_selects_the_process_current_directory(tmp_path, run):
    (tmp_path / "sub").mkdir()

    result = await run(make_tool(tmp_path), {"command": "pwd", "workdir": "sub"})

    assert os.path.realpath(body(result).strip()) == os.path.realpath(
        tmp_path / "sub",
    )


# ── scenario 3: invalid workdir ───────────────────────────────────────────────


async def test_missing_workdir_fails_before_execution(tmp_path, run):
    result = await run(make_tool(tmp_path), {"command": "echo hi", "workdir": "nope"})

    assert result.is_error is True
    assert body(result) == f"workdir does not exist: {tmp_path / 'nope'}"


async def test_a_regular_file_as_workdir_fails_before_execution(tmp_path, run):
    (tmp_path / "file.txt").write_text("")

    result = await run(make_tool(tmp_path), {
        "command": "echo hi", "workdir": "file.txt",
    })

    assert result.is_error is True
    assert body(result) == f"workdir is not a directory: {tmp_path / 'file.txt'}"


# ── scenario 4: combined stdout and stderr ────────────────────────────────────


async def test_stdout_and_stderr_are_combined_in_observed_order(tmp_path, run):
    result = await run(make_tool(tmp_path), {
        "command": "echo stdout_msg; echo stderr_msg 1>&2; echo last",
    })

    assert result.is_error is False
    assert result.metadata["exit"] == 0
    assert "stdout_msg\nstderr_msg\nlast" in body(result)


# ── scenario 5: non-zero exit ─────────────────────────────────────────────────


async def test_non_zero_exit_is_a_result_not_an_exception(tmp_path, run):
    result = await run(make_tool(tmp_path), {"command": "exit 42"})

    assert result.metadata["exit"] == 42
    assert result.is_error is True


# ── scenario 6: timeout ───────────────────────────────────────────────────────


async def test_timeout_terminates_the_command_and_explains_itself(tmp_path, run):
    result = await run(make_tool(tmp_path), {
        "command": "echo before; sleep 5", "timeout": 500,
    })

    assert result.is_error is True
    assert "before" in body(result)
    assert "<shell_metadata>" in body(result)
    assert (
        "shell tool terminated command after exceeding timeout 500 ms"
    ) in body(result)
    assert result.metadata["exit"] is None


# ── scenario 7: cancellation ──────────────────────────────────────────────────


async def test_cancellation_kills_the_process_and_keeps_partial_output(tmp_path, run):
    token = CancellationToken()
    task = asyncio.create_task(run(
        make_tool(tmp_path),
        {"command": "echo before; sleep 5"},
        cancellation_token=token,
    ))
    await asyncio.sleep(0.3)

    token.cancel()
    result = await asyncio.wait_for(task, timeout=2)

    assert result.is_error is True
    assert "before" in body(result)
    assert "shell tool cancelled the command" in body(result)
    assert result.metadata["exit"] is None


# ── scenario 8 + 11: line truncation retains the full output ──────────────────


async def test_line_truncation_saves_the_complete_output(tmp_path, run):
    result = await run(make_tool(tmp_path), {"command": "seq 1 2100"})

    assert result.metadata["truncated"] is True
    saved = result.metadata["output_path"]
    assert saved is not None
    lines = Path(saved).read_text().splitlines()
    assert lines == [str(n) for n in range(1, 2101)]
    assert body(result).startswith("...output truncated...")
    assert f"Full output saved to: {saved}" in body(result)
    preview = body(result).split("\n\n", 2)[2]
    assert len(preview.splitlines()) <= 2_000
    assert preview.splitlines()[-1] == "2100"


# ── scenario 9: byte truncation ───────────────────────────────────────────────


async def test_byte_truncation_saves_the_complete_output(tmp_path, run):
    result = await run(make_tool(tmp_path), {
        "command": "head -c 60000 /dev/zero | tr '\\0' a",
    })

    assert result.metadata["truncated"] is True
    saved = result.metadata["output_path"]
    assert len(Path(saved).read_text()) == 60_000
    assert body(result).startswith("...output truncated...")
    assert len(body(result).encode()) < 60_000


# ── scenario 10: small output ─────────────────────────────────────────────────


async def test_small_output_is_returned_unchanged(tmp_path, run):
    result = await run(make_tool(tmp_path), {"command": "printf 'one\\ntwo'"})

    assert body(result) == "one\ntwo"
    assert result.metadata == {"exit": 0, "truncated": False, "output_path": None}


# ── scenario 12: no output ────────────────────────────────────────────────────


async def test_silent_success_returns_the_no_output_marker(tmp_path, run):
    result = await run(make_tool(tmp_path), {"command": "true"})

    assert body(result) == "(no output)"
    assert result.metadata["exit"] == 0
    assert result.is_error is False


# ── scenario 13: fresh process per call ───────────────────────────────────────


async def test_shell_state_does_not_persist_across_calls(tmp_path, run):
    tool = make_tool(tmp_path)
    await run(tool, {"command": "cd /; export LUCA_TEST_VAR=leaked"})

    result = await run(tool, {"command": "pwd; echo ${LUCA_TEST_VAR:-unset}"})

    pwd, var = body(result).strip().splitlines()
    assert os.path.realpath(pwd) == os.path.realpath(tmp_path)
    assert var == "unset"


# ── arguments ─────────────────────────────────────────────────────────────────


def test_args_reject_blank_commands_and_bad_timeouts():
    with pytest.raises(ValidationError):
        BashTool.Args.model_validate({"command": ""})
    with pytest.raises(ValidationError):
        BashTool.Args.model_validate({"command": "   "})
    with pytest.raises(ValidationError):
        BashTool.Args.model_validate({"command": "ls", "timeout": 0})
    with pytest.raises(ValidationError):
        BashTool.Args.model_validate({"command": "ls", "timeout": -5})


# ── description rendering ─────────────────────────────────────────────────────


def test_description_is_rendered_with_the_actual_environment(tmp_path):
    tool = make_tool(tmp_path)

    assert "Shell: /bin/bash" in tool.description
    assert "time out after 120000ms" in tool.description
    for placeholder in ("{os}", "{shell}", "{tmp}", "{default_timeout_ms}"):
        assert placeholder not in tool.description
    assert type(tool).description == BASH_DESCRIPTION_TEMPLATE


# ── permission resource ───────────────────────────────────────────────────────


def test_permission_resource_exposes_the_command(tmp_path, perm):
    [access, request] = perm(make_tool(tmp_path), {"command": "echo test"})

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
        ResourcePermission(permission="bash", resource="echo test"),
    ]
    assert request.metadata["preview"] == "Run command: echo test"
    assert [
        (o.resource_permissions, o.metadata["preview"])
        for o in request.answer_options
    ] == [
        (
            [ResourcePermission(permission="bash", resource="echo *")],
            "Run any 'echo' command",
        ),
    ]


def test_permission_resource_strips_surrounding_whitespace(tmp_path, perm):
    [access, request] = perm(make_tool(tmp_path), {"command": "  git status  "})

    assert access.resources == [
        ResourcePermission(permission="access_directory", resource=str(tmp_path)),
    ]
    assert request.resources == [
        ResourcePermission(permission="bash", resource="git status"),
    ]
    assert request.answer_options[0].resource_permissions == [
        ResourcePermission(permission="bash", resource="git *"),
    ]


def test_permission_resource_uses_the_workdir_argument(tmp_path, perm):
    [access, request] = perm(make_tool(tmp_path), {
        "command": "ls", "workdir": "sub",
    })

    assert access.resources == [
        ResourcePermission(
            permission="access_directory", resource=str(tmp_path / "sub"),
        ),
    ]
    assert request.resources == [
        ResourcePermission(permission="bash", resource="ls"),
    ]
