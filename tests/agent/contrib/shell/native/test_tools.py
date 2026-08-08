"""Unit tests for the four provider-native shell tools: the operations they
perform on disk, the native result shape `openai_shell` captures at birth,
the read-first guard the whole-file writes keep, and the permission steps each
one declares. No runner, no middleware — projection is the battery's job."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from luca.agent.contrib.resource_permissions import ResourcePermission
from luca.agent.contrib.shell import (
    AnthropicBashTool,
    AnthropicTextEditorTool,
    FileReadTracker,
    OpenAIApplyPatchTool,
    OpenAIShellTool,
)
from luca.agent.contrib.shell.native.anthropic import RESTART_OUTPUT
from luca.agent.contrib.shell.native.openai import SHELL_DESCRIPTION
from luca.agent.contrib.shell.tools import BASH_MAX_OUTPUT_BYTES
from luca.agent.core import ToolKind, ToolSpec
from tests.agent.contrib.shell.conftest import CONVERSATION

BOM = b"\xef\xbb\xbf"

NATIVE_TOOLS = [
    OpenAIApplyPatchTool(workdir="."),
    OpenAIShellTool(workdir="."),
    AnthropicTextEditorTool(workdir="."),
    AnthropicBashTool(workdir="."),
]


def patcher(tmp_path, *read_paths) -> OpenAIApplyPatchTool:
    """An `openai_apply_patch` whose tracker has already read the given paths."""
    tracker = FileReadTracker()
    for path in read_paths:
        tracker.record(CONVERSATION, path)
    return OpenAIApplyPatchTool(workdir=tmp_path, tracker=tracker)


def editor(tmp_path, *read_paths) -> AnthropicTextEditorTool:
    tracker = FileReadTracker()
    for path in read_paths:
        tracker.record(CONVERSATION, path)
    return AnthropicTextEditorTool(workdir=tmp_path, tracker=tracker)


def body(result) -> str:
    return result.content[0].text


# ── the specs the registry advertises ─────────────────────────────────────────


def test_every_native_tool_carries_a_human_title_beside_its_internal_name():
    # `name` is the identity; `title` is what a UI shows. A native tool is the
    # reason the field exists, so a spec that lost it would render as
    # `openai_apply_patch` in the transcript.
    assert [(tool.get_tool_spec().name, tool.get_tool_spec().title) for tool in NATIVE_TOOLS] == [
        ("openai_apply_patch", "Apply patch"),
        ("openai_shell", "Shell"),
        ("anthropic_text_editor_20250728", "Edit file"),
        ("anthropic_bash_20250124", "Bash"),
    ]


def test_the_shell_specs_full_shape():
    assert OpenAIShellTool(workdir=".").get_tool_spec() == ToolSpec(
        name="openai_shell",
        title="Shell",
        description=SHELL_DESCRIPTION,
        input_schema=OpenAIShellTool.Args.model_json_schema(),
        tool_kind=ToolKind.EXECUTE,
        namespace="contrib.shell",
    )


# ── openai_apply_patch ────────────────────────────────────────────────────────


async def test_apply_patch_creates_a_file_from_a_create_diff(tmp_path, run):
    # The final `+` line is the file's trailing newline: V4A carries one line
    # per `+`, so a diff that does not end in a blank one produces a file that
    # does not end in a newline. Faithful to OpenAI's own applier.
    result = await run(
        patcher(tmp_path),
        {"type": "create_file", "path": "hello.txt", "diff": "+Hello\n+world\n+\n"},
    )

    assert result.is_error is False
    assert body(result) == "Created hello.txt"
    assert result.metadata["created"] is True
    assert (tmp_path / "hello.txt").read_text() == "Hello\nworld\n"


async def test_apply_patch_refuses_to_overwrite_a_file_that_was_never_read(tmp_path, run):
    target = tmp_path / "notes.txt"
    target.write_text("old\n")

    result = await run(
        patcher(tmp_path),
        {"type": "create_file", "path": "notes.txt", "diff": "+new\n"},
    )

    assert result.is_error is True
    assert body(result).startswith(f"File has not been read yet: read {target}")
    assert target.read_text() == "old\n"


async def test_apply_patch_overwrites_a_file_that_was_read(tmp_path, run):
    target = tmp_path / "notes.txt"
    target.write_text("old\n")

    result = await run(
        patcher(tmp_path, target),
        {"type": "create_file", "path": "notes.txt", "diff": "+new\n+\n"},
    )

    assert result.is_error is False
    assert result.metadata["created"] is False
    assert target.read_text() == "new\n"


async def test_apply_patch_updates_a_file_in_place(tmp_path, run):
    target = tmp_path / "app.py"
    target.write_text("def greet():\n    print('Hi')\n")

    result = await run(
        patcher(tmp_path),
        {
            "type": "update_file",
            "path": "app.py",
            "diff": "@@ def greet():\n-    print('Hi')\n+    print('Hello')\n",
        },
    )

    assert result.is_error is False
    assert body(result) == "Updated app.py"
    assert target.read_text() == "def greet():\n    print('Hello')\n"
    assert "+    print('Hello')" in result.metadata["diff"]


async def test_apply_patch_update_preserves_a_bom_and_crlf_endings(tmp_path, run):
    target = tmp_path / "win.txt"
    target.write_bytes(BOM + b"alpha\r\nbeta\r\n")

    result = await run(
        patcher(tmp_path),
        {"type": "update_file", "path": "win.txt", "diff": "@@\n-beta\n+gamma\n"},
    )

    assert result.is_error is False
    assert target.read_bytes() == BOM + b"alpha\r\ngamma\r\n"


async def test_apply_patch_update_with_move_to_renames_the_file(tmp_path, run):
    source = tmp_path / "app.py"
    source.write_text("one\ntwo\n")

    result = await run(
        patcher(tmp_path),
        {
            "type": "update_file",
            "path": "app.py",
            "diff": "@@\n-two\n+TWO\n",
            "move_to": "main.py",
        },
    )

    assert result.is_error is False
    assert body(result) == "Updated app.py\nMoved app.py to main.py"
    assert source.exists() is False
    assert (tmp_path / "main.py").read_text() == "one\nTWO\n"


async def test_apply_patch_update_of_a_missing_file_is_an_error_result(tmp_path, run):
    result = await run(
        patcher(tmp_path),
        {"type": "update_file", "path": "gone.txt", "diff": "@@\n-a\n+b\n"},
    )

    assert result.is_error is True
    assert body(result) == f"File not found: {tmp_path / 'gone.txt'}"


async def test_apply_patch_reports_a_diff_that_does_not_apply(tmp_path, run):
    (tmp_path / "app.py").write_text("one\n")

    result = await run(
        patcher(tmp_path),
        {"type": "update_file", "path": "app.py", "diff": "@@\n-nowhere\n+x\n"},
    )

    assert result.is_error is True
    assert "nowhere" in body(result)


async def test_apply_patch_deletes_a_file(tmp_path, run):
    target = tmp_path / "obsolete.txt"
    target.write_text("bye\n")

    result = await run(patcher(tmp_path), {"type": "delete_file", "path": "obsolete.txt"})

    assert result.is_error is False
    assert body(result) == "Deleted obsolete.txt"
    assert target.exists() is False


async def test_apply_patch_deletes_a_file_that_is_not_utf_8(tmp_path, run):
    # Under OpenAI natives the generic `delete_file` is not advertised, so this
    # is the only way to remove a `.pyc` or a screenshot — reading it for the
    # metadata diff must not be what stops it.
    target = tmp_path / "icon.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe\x00binary")

    result = await run(patcher(tmp_path), {"type": "delete_file", "path": "icon.png"})

    assert result.is_error is False
    assert target.exists() is False


async def test_apply_patch_needs_a_diff_for_an_update(tmp_path, run):
    (tmp_path / "app.py").write_text("one\n")

    result = await run(patcher(tmp_path), {"type": "update_file", "path": "app.py"})

    assert result.is_error is True
    assert body(result) == "diff is required for update_file."


def test_apply_patch_rejects_an_unknown_operation_type(tmp_path):
    with pytest.raises(ValidationError):
        OpenAIApplyPatchTool.Args.model_validate({"type": "rename_file", "path": "a.txt"})


def test_apply_patch_declares_the_directory_step_then_its_own_verb(tmp_path, perm):
    [access, patch] = perm(
        patcher(tmp_path),
        {"type": "update_file", "path": "app.py", "diff": "@@\n-a\n+b\n", "move_to": "sub/main.py"},
    )

    assert access.resources == [
        ResourcePermission(permission="access_directory", resource=str(tmp_path)),
        ResourcePermission(permission="access_directory", resource=str(tmp_path / "sub")),
    ]
    assert patch.resources == [
        ResourcePermission(permission="apply_patch", resource=str(tmp_path / "app.py")),
        ResourcePermission(permission="apply_patch", resource=str(tmp_path / "sub/main.py")),
    ]
    assert patch.metadata == {"preview": "Apply patch to app.py, sub/main.py"}


# ── openai_shell ──────────────────────────────────────────────────────────────


async def test_shell_runs_every_command_in_order_and_captures_each_result(tmp_path, run):
    result = await run(
        OpenAIShellTool(workdir=tmp_path),
        {"commands": ["printf one", "printf two"]},
    )

    assert result.is_error is False
    assert result.extras == {
        "custom_type": "shell_call_output",
        "results": [
            {"stdout": "one", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}},
            {"stdout": "two", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}},
        ],
    }
    assert body(result) == "one\ntwo"


async def test_shell_keeps_stdout_and_stderr_apart(tmp_path, run):
    result = await run(
        OpenAIShellTool(workdir=tmp_path),
        {"commands": ["printf out; printf err >&2"]},
    )

    assert result.extras["results"] == [
        {"stdout": "out", "stderr": "err", "outcome": {"type": "exit", "exit_code": 0}},
    ]


async def test_shell_runs_the_commands_after_a_failing_one(tmp_path, run):
    result = await run(
        OpenAIShellTool(workdir=tmp_path),
        {"commands": ["exit 3", "printf after"]},
    )

    assert result.is_error is True
    assert result.extras["results"] == [
        {"stdout": "", "stderr": "", "outcome": {"type": "exit", "exit_code": 3}},
        {"stdout": "after", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}},
    ]


async def test_shell_records_a_timeout_as_the_native_timeout_outcome(tmp_path, run):
    result = await run(
        OpenAIShellTool(workdir=tmp_path),
        {"commands": ["sleep 5"], "timeout_ms": 500},
    )

    assert result.is_error is True
    assert result.extras["results"] == [
        {"stdout": "", "stderr": "", "outcome": {"type": "timeout"}},
    ]


async def test_shell_clips_each_command_to_max_output_length(tmp_path, run):
    result = await run(
        OpenAIShellTool(workdir=tmp_path),
        {"commands": ["printf abcdefghij"], "max_output_length": 4},
    )

    spill = result.metadata["output_path"]
    assert result.extras["results"] == [
        {
            "stdout": f"...output truncated...\nFull output saved to: {spill}\n\nghij",
            "stderr": "",
            "outcome": {"type": "exit", "exit_code": 0},
        },
    ]
    assert "abcdefghij" in Path(spill).read_text()


async def test_shell_caps_the_captured_output_even_when_the_model_asks_for_no_limit(tmp_path, run):
    # The cap has to land on `extras`, not on the prose: the projector builds
    # the wire item from the per-command results and never sends `content`, so
    # a cap applied only to the text would be inert on the wire.
    result = await run(
        OpenAIShellTool(workdir=tmp_path),
        {"commands": [f"printf 'x%.0s' $(seq 1 {BASH_MAX_OUTPUT_BYTES * 2})"]},
    )

    stdout = result.extras["results"][0]["stdout"]
    assert result.metadata["truncated"] is True
    assert stdout.startswith("...output truncated...\nFull output saved to: ")
    assert len(stdout) <= BASH_MAX_OUTPUT_BYTES + len(result.metadata["output_path"]) + 64
    assert len(Path(result.metadata["output_path"]).read_text()) > BASH_MAX_OUTPUT_BYTES * 2
    # the prose the transcript shows is the SAME capped text
    assert body(result) == stdout


def test_shell_needs_at_least_one_command():
    with pytest.raises(ValidationError):
        OpenAIShellTool.Args.model_validate({"commands": []})


def test_shell_asks_for_every_command_and_offers_one_grant_per_head(tmp_path, perm):
    [access, run_step] = perm(
        OpenAIShellTool(workdir=tmp_path),
        {"commands": ["git status", "git diff", "pytest -q"]},
    )

    assert access.resources == [
        ResourcePermission(permission="access_directory", resource=str(tmp_path)),
    ]
    assert run_step.resources == [
        ResourcePermission(permission="bash", resource="git status"),
        ResourcePermission(permission="bash", resource="git diff"),
        ResourcePermission(permission="bash", resource="pytest -q"),
    ]
    assert [option.resource_permissions for option in run_step.answer_options] == [
        [ResourcePermission(permission="bash", resource="git *")],
        [ResourcePermission(permission="bash", resource="pytest *")],
    ]


# ── anthropic_text_editor_20250728 ────────────────────────────────────────────


async def test_text_editor_view_numbers_the_lines(tmp_path, run):
    (tmp_path / "notes.txt").write_text("alpha\nbeta\n")

    result = await run(editor(tmp_path), {"command": "view", "path": "notes.txt"})

    assert result.is_error is False
    assert body(result) == "     1\talpha\n     2\tbeta"


async def test_text_editor_view_honours_a_range(tmp_path, run):
    (tmp_path / "notes.txt").write_text("a\nb\nc\nd\n")

    result = await run(
        editor(tmp_path),
        {"command": "view", "path": "notes.txt", "view_range": [2, 3]},
    )

    assert body(result) == "     2\tb\n     3\tc"


async def test_text_editor_view_lists_a_directory(tmp_path, run):
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("hi")
    (tmp_path / ".hidden").write_text("x")

    result = await run(editor(tmp_path), {"command": "view", "path": "."})

    assert body(result) == "README.md\nsrc/"


async def test_text_editor_view_satisfies_the_read_first_guard(tmp_path, run):
    target = tmp_path / "notes.txt"
    target.write_text("old\n")
    tool = editor(tmp_path)

    await run(tool, {"command": "view", "path": "notes.txt"})
    result = await run(tool, {"command": "create", "path": "notes.txt", "file_text": "new\n"})

    assert result.is_error is False
    assert target.read_text() == "new\n"


async def test_text_editor_create_refuses_an_unviewed_existing_file(tmp_path, run):
    target = tmp_path / "notes.txt"
    target.write_text("old\n")

    result = await run(
        editor(tmp_path),
        {"command": "create", "path": "notes.txt", "file_text": "new\n"},
    )

    assert result.is_error is True
    assert body(result) == f"File has not been read yet: view {target} before replacing it with create."
    assert target.read_text() == "old\n"


async def test_text_editor_create_writes_a_new_file(tmp_path, run):
    result = await run(
        editor(tmp_path),
        {"command": "create", "path": "nested/new.txt", "file_text": "hello\n"},
    )

    assert result.is_error is False
    assert body(result) == "Created nested/new.txt"
    assert (tmp_path / "nested/new.txt").read_text() == "hello\n"


async def test_text_editor_str_replace_swaps_the_unique_match(tmp_path, run):
    target = tmp_path / "app.py"
    target.write_text("value = 1\nother = 2\n")

    result = await run(
        editor(tmp_path),
        {"command": "str_replace", "path": "app.py", "old_str": "value = 1", "new_str": "value = 9"},
    )

    assert result.is_error is False
    assert body(result) == "Updated app.py"
    assert target.read_text() == "value = 9\nother = 2\n"


async def test_text_editor_str_replace_preserves_a_bom_and_crlf_endings(tmp_path, run):
    target = tmp_path / "win.txt"
    target.write_bytes(BOM + b"alpha\r\nbeta\r\n")

    result = await run(
        editor(tmp_path),
        {"command": "str_replace", "path": "win.txt", "old_str": "beta", "new_str": "gamma"},
    )

    assert result.is_error is False
    assert target.read_bytes() == BOM + b"alpha\r\ngamma\r\n"


async def test_text_editor_str_replace_matches_and_writes_crlf_without_doubling_it(tmp_path, run):
    # The model may put `\r\n` in either string. The FILE's convention wins:
    # `old_str` still has to match, and the result must not become `\r\r\n`.
    target = tmp_path / "win.txt"
    target.write_bytes(b"alpha\r\nbeta\r\n")

    result = await run(
        editor(tmp_path),
        {"command": "str_replace", "path": "win.txt", "old_str": "alpha\r\nbeta", "new_str": "one\r\ntwo"},
    )

    assert result.is_error is False
    assert target.read_bytes() == b"one\r\ntwo\r\n"


async def test_text_editor_insert_of_crlf_text_into_an_lf_file_stays_lf(tmp_path, run):
    target = tmp_path / "unix.txt"
    target.write_bytes(b"first\n")

    await run(
        editor(tmp_path),
        {"command": "insert", "path": "unix.txt", "insert_line": 1, "insert_text": "second\r\nthird"},
    )

    assert target.read_bytes() == b"first\nsecond\nthird\n"


async def test_text_editor_str_replace_refuses_when_nothing_matches(tmp_path, run):
    (tmp_path / "app.py").write_text("value = 1\n")

    result = await run(
        editor(tmp_path),
        {"command": "str_replace", "path": "app.py", "old_str": "missing", "new_str": "x"},
    )

    assert result.is_error is True
    assert body(result).startswith("No match found for old_str.")


async def test_text_editor_str_replace_refuses_an_ambiguous_match(tmp_path, run):
    (tmp_path / "app.py").write_text("x = 1\nx = 1\n")

    result = await run(
        editor(tmp_path),
        {"command": "str_replace", "path": "app.py", "old_str": "x = 1", "new_str": "x = 2"},
    )

    assert result.is_error is True
    assert body(result).startswith("Found 2 matches for old_str.")


async def test_text_editor_insert_puts_text_after_the_named_line(tmp_path, run):
    target = tmp_path / "app.py"
    target.write_text("first\nsecond\n")

    result = await run(
        editor(tmp_path),
        {"command": "insert", "path": "app.py", "insert_line": 1, "insert_text": "middle"},
    )

    assert result.is_error is False
    assert target.read_text() == "first\nmiddle\nsecond\n"


async def test_text_editor_insert_at_line_zero_prepends(tmp_path, run):
    target = tmp_path / "app.py"
    target.write_text("first\n")

    await run(
        editor(tmp_path),
        {"command": "insert", "path": "app.py", "insert_line": 0, "insert_text": "header"},
    )

    assert target.read_text() == "header\nfirst\n"


async def test_text_editor_insert_past_the_end_is_an_error_result(tmp_path, run):
    (tmp_path / "app.py").write_text("first\n")

    result = await run(
        editor(tmp_path),
        {"command": "insert", "path": "app.py", "insert_line": 9, "insert_text": "x"},
    )

    assert result.is_error is True
    assert body(result) == "insert_line 9 is out of range; expected 0..1."


async def test_text_editor_edit_of_a_missing_file_is_an_error_result(tmp_path, run):
    result = await run(
        editor(tmp_path),
        {"command": "str_replace", "path": "gone.txt", "old_str": "a", "new_str": "b"},
    )

    assert result.is_error is True
    assert body(result) == f"File not found: {tmp_path / 'gone.txt'}"


@pytest.mark.parametrize(
    ("command", "arguments", "permission"),
    [
        ("view", {}, "read"),
        ("create", {"file_text": "x"}, "write"),
        ("str_replace", {"old_str": "a", "new_str": "b"}, "edit"),
        ("insert", {"insert_line": 0, "insert_text": "x"}, "edit"),
    ],
)
def test_text_editor_asks_for_the_verb_its_command_means(tmp_path, perm, command, arguments, permission):
    [access, verb] = perm(
        editor(tmp_path),
        {"command": command, "path": "app.py", **arguments},
    )

    assert access.resources == [
        ResourcePermission(permission="access_directory", resource=str(tmp_path)),
    ]
    assert verb.resources == [
        ResourcePermission(permission=permission, resource=str(tmp_path / "app.py")),
    ]


# ── anthropic_bash_20250124 ───────────────────────────────────────────────────


async def test_anthropic_bash_runs_a_command(tmp_path, run):
    result = await run(AnthropicBashTool(workdir=tmp_path), {"command": "printf hi"})

    assert result.is_error is False
    assert body(result) == "hi"
    assert result.metadata["exit"] == 0


async def test_anthropic_bash_reports_a_non_zero_exit(tmp_path, run):
    result = await run(AnthropicBashTool(workdir=tmp_path), {"command": "exit 2"})

    assert result.is_error is True
    assert result.metadata["exit"] == 2


async def test_anthropic_bash_answers_a_restart_without_running_anything(tmp_path, run):
    result = await run(AnthropicBashTool(workdir=tmp_path), {"restart": True})

    assert result.is_error is False
    assert body(result) == RESTART_OUTPUT


async def test_anthropic_bash_needs_a_command_or_a_restart(tmp_path, run):
    result = await run(AnthropicBashTool(workdir=tmp_path), {})

    assert result.is_error is True
    assert body(result) == "Provide a command, or restart: true."


def test_anthropic_bash_rejects_a_blank_command():
    # the same validator the generic `bash` has: without it a whitespace-only
    # command reaches `build_permission_requests` and raises IndexError there,
    # recording FAILED instead of INVALID
    with pytest.raises(ValidationError):
        AnthropicBashTool.Args.model_validate({"command": "   "})


def test_anthropic_bash_restart_declares_only_the_directory_step(tmp_path, perm):
    assert perm(AnthropicBashTool(workdir=tmp_path), {"restart": True}) == [
        AnthropicBashTool(workdir=tmp_path)._access_request(tmp_path),
    ]


def test_anthropic_bash_declares_the_same_verb_as_the_generic_bash(tmp_path, perm):
    [_access, run_step] = perm(AnthropicBashTool(workdir=tmp_path), {"command": "git status"})

    assert run_step.resources == [ResourcePermission(permission="bash", resource="git status")]
