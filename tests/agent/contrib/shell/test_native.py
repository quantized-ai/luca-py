"""Anthropic's `str_replace_based_edit_tool`, executed by luca.

The provider owns the schema, so the interesting questions are the ones the
schema does not answer: does the approval gate still fire with the same rules a
`read` or an `edit` would produce, does the read-first guard still hold, and
does each command reach the tool that already implements it. Real files under
`tmp_path`; no runner.
"""

import asyncio
from pathlib import Path

import pytest

from luca.agent.contrib.shell.native import (
    BASH_NAME,
    BASH_TYPE,
    TEXT_EDITOR_NAME,
    TEXT_EDITOR_TYPE_CURRENT,
    TEXT_EDITOR_TYPE_LEGACY,
    NativeApplyPatchTool,
    NativeBashTool,
    NativeShellTool,
    NativeTextEditorTool,
    native_editor_type,
    native_openai_tool_types,
    to_patch_envelope,
)
from luca.agent.contrib.shell.tools import EditTool, FileReadTracker, ReadTool, ShellToolError, WriteTool
from luca.agent.core import CancellationToken
from luca.agent.core.models import LLMConfig, ToolKind
from luca.agent.core.runner import AgentSessionRunner
from tests.agent.scenarios import conversation

SESSION = AgentSessionRunner.new_session(LLMConfig(model="claude-opus-5", provider="anthropic"))
CONVERSATION = SESSION.main_conversation_id


@pytest.fixture
def editor(tmp_path):
    return NativeTextEditorTool(tmp_path, FileReadTracker())


async def run(tool, **args):
    return await tool.execute(
        tool.Args.model_validate(args).model_dump(),
        SESSION,
        CONVERSATION,
        cancellation_token=CancellationToken(),
    )


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("one\ntwo\nthree\n")
    return path


# ── the spec ─────────────────────────────────────────────────────────────────


def test_the_spec_carries_the_provider_type_instead_of_advertising_a_schema(editor):
    spec = editor.get_tool_spec()

    assert (spec.name, spec.provider_type, spec.tool_kind) == (
        TEXT_EDITOR_NAME,
        TEXT_EDITOR_TYPE_CURRENT,
        ToolKind.EDIT,
    )
    # still populated: it is what validates the call and what still describes
    # the tool in a session resumed years from now
    assert list(spec.input_schema["properties"]) == [
        "command",
        "path",
        "file_text",
        "insert_line",
        "new_str",
        "old_str",
        "view_range",
    ]


def test_the_editor_version_is_per_instance_not_per_class(tmp_path):
    # two sessions in one process can target different model generations
    current = NativeTextEditorTool(tmp_path)
    legacy = NativeTextEditorTool(tmp_path, provider_type=TEXT_EDITOR_TYPE_LEGACY)

    assert current.get_tool_spec().provider_type == TEXT_EDITOR_TYPE_CURRENT
    assert legacy.get_tool_spec().provider_type == TEXT_EDITOR_TYPE_LEGACY


# ── the commands ─────────────────────────────────────────────────────────────


async def test_view_reads_the_file(editor, sample):
    result = await run(editor, command="view", path=str(sample))

    assert not result.is_error
    assert "two" in result.content[0].text


async def test_view_range_windows_the_read(editor, sample):
    result = await run(editor, command="view", path=str(sample), view_range=[2, 2])

    assert "two" in result.content[0].text
    assert "three" not in result.content[0].text


async def test_view_range_to_the_end_is_minus_one(editor, sample):
    result = await run(editor, command="view", path=str(sample), view_range=[2, -1])

    assert "two" in result.content[0].text
    assert "three" in result.content[0].text


async def test_create_writes_a_new_file(editor, tmp_path):
    target = tmp_path / "fresh.txt"

    result = await run(editor, command="create", path=str(target), file_text="hello\n")

    assert not result.is_error
    assert target.read_text() == "hello\n"


async def test_str_replace_edits_after_a_view(editor, sample):
    await run(editor, command="view", path=str(sample))

    result = await run(editor, command="str_replace", path=str(sample), old_str="two", new_str="TWO")

    assert not result.is_error
    assert sample.read_text() == "one\nTWO\nthree\n"


async def test_str_replace_with_no_new_str_deletes_the_match(editor, sample):
    await run(editor, command="view", path=str(sample))

    await run(editor, command="str_replace", path=str(sample), old_str="two\n")

    assert sample.read_text() == "one\nthree\n"


async def test_insert_places_text_after_the_given_line(editor, sample):
    await run(editor, command="view", path=str(sample))

    result = await run(editor, command="insert", path=str(sample), insert_line=1, new_str="INSERTED")

    assert not result.is_error
    assert sample.read_text() == "one\nINSERTED\ntwo\nthree\n"


async def test_insert_at_line_zero_goes_to_the_top(editor, sample):
    await run(editor, command="view", path=str(sample))

    await run(editor, command="insert", path=str(sample), insert_line=0, new_str="FIRST")

    assert sample.read_text() == "FIRST\none\ntwo\nthree\n"


# ── the read-first guard ─────────────────────────────────────────────────────


async def test_str_replace_on_a_file_never_viewed_is_refused(editor, sample):
    # the guard has to survive the provider's argument names, or the native
    # editor is a way around a safety check the other tools enforce
    result = await run(editor, command="str_replace", path=str(sample), old_str="two", new_str="TWO")

    assert result.is_error
    assert "has not been read yet" in result.content[0].text
    assert sample.read_text() == "one\ntwo\nthree\n"


async def test_insert_keeps_a_files_crlf_endings(editor, tmp_path):
    """`read_text` folds CRLF to LF, so a one-line insert used to rewrite the
    whole file. `edit` guards against this; `insert` has to as well or the two
    disagree on the same user action."""
    path = tmp_path / "windows.txt"
    path.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    await run(editor, command="view", path=str(path))

    await run(editor, command="insert", path=str(path), insert_line=1, new_str="MID")

    assert path.read_bytes() == b"one\r\nMID\r\ntwo\r\nthree\r\n"


async def test_insert_at_the_top_of_a_bom_file_does_not_duplicate_the_bom(editor, tmp_path):
    """Decoding as plain utf-8 leaves the BOM as the first character of line 1,
    so splicing in front of it strands the old one mid-file while the write
    prepends a fresh one."""
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfone\ntwo\n")
    await run(editor, command="view", path=str(path))

    await run(editor, command="insert", path=str(path), insert_line=0, new_str="TOP")

    assert path.read_bytes() == b"\xef\xbb\xbfTOP\none\ntwo\n"


async def test_insert_counts_lines_the_way_view_numbers_them(editor, tmp_path):
    """A form feed is a line break to `str.splitlines` and not to the read
    tool, so the model's line numbers and the splice used to disagree."""
    path = tmp_path / "pagebreak.py"
    path.write_bytes(b"a = 1\n\x0c\nb = 2\nc = 3\n")
    await run(editor, command="view", path=str(path))

    await run(editor, command="insert", path=str(path), insert_line=3, new_str="INSERTED")

    assert path.read_bytes() == b"a = 1\n\x0c\nb = 2\nINSERTED\nc = 3\n"


async def test_insert_after_the_last_line_of_a_file_with_no_final_newline(editor, tmp_path):
    path = tmp_path / "bare.txt"
    path.write_bytes(b"one\ntwo")
    await run(editor, command="view", path=str(path))

    await run(editor, command="insert", path=str(path), insert_line=2, new_str="three")

    assert path.read_bytes() == b"one\ntwo\nthree\n"


async def test_insert_does_not_overwrite_a_concurrent_edit(editor, tmp_path):
    """The read and the write have to be one locked region. Reading outside it
    lets another conversation's `edit` commit in between, and the stale
    snapshot lands on top of it — both calls reporting success."""
    path = tmp_path / "shared.txt"
    path.write_text("alpha\nbeta\ngamma\n")
    await run(editor, command="view", path=str(path))
    editor.tracker.record("other", path)
    other = EditTool(tmp_path, editor.tracker)

    await asyncio.gather(
        other.execute(
            {"file_path": str(path), "old_string": "beta", "new_string": "BETA", "replace_all": False},
            SESSION,
            "other",
            cancellation_token=CancellationToken(),
        ),
        run(editor, command="insert", path=str(path), insert_line=3, new_str="delta"),
    )

    assert path.read_text() == "alpha\nBETA\ngamma\ndelta\n"


async def test_insert_into_a_file_never_viewed_is_refused(editor, sample):
    result = await run(editor, command="insert", path=str(sample), insert_line=1, new_str="X")

    assert result.is_error
    assert "has not been read yet" in result.content[0].text


async def test_view_marks_the_file_read_for_the_delegates(tmp_path, sample):
    # one tracker across the native editor and luca's own tools: viewing
    # through the native editor must satisfy a later plain `edit`
    tracker = FileReadTracker()
    editor = NativeTextEditorTool(tmp_path, tracker)
    plain = EditTool(tmp_path, tracker)

    await run(editor, command="view", path=str(sample))
    result = await plain.execute(
        {"file_path": str(sample), "old_string": "two", "new_string": "2", "replace_all": False},
        SESSION,
        CONVERSATION,
        cancellation_token=CancellationToken(),
    )

    assert not result.is_error


async def test_the_guard_is_per_conversation(tmp_path, sample):
    editor = NativeTextEditorTool(tmp_path, FileReadTracker())
    await run(editor, command="view", path=str(sample))

    other = await editor.execute(
        editor.Args.model_validate(
            {"command": "str_replace", "path": str(sample), "old_str": "two", "new_str": "TWO"}
        ).model_dump(),
        SESSION,
        "a-different-conversation",
        cancellation_token=CancellationToken(),
    )

    assert other.is_error


# ── permissions ──────────────────────────────────────────────────────────────


def _resources(requests):
    return [(r.permission, r.resource) for request in requests for r in request.resources]


def test_view_asks_for_exactly_what_read_asks_for(tmp_path, sample):
    tracker = FileReadTracker()
    editor = NativeTextEditorTool(tmp_path, tracker)
    plain = ReadTool(tmp_path, tracker)

    native = editor.build_permission_requests({"command": "view", "path": str(sample)}, SESSION, CONVERSATION)
    luca = plain.build_permission_requests({"file_path": str(sample)}, SESSION, CONVERSATION)

    assert _resources(native) == _resources(luca)


def test_str_replace_asks_for_exactly_what_edit_asks_for(tmp_path, sample):
    tracker = FileReadTracker()
    editor = NativeTextEditorTool(tmp_path, tracker)
    plain = EditTool(tmp_path, tracker)

    native = editor.build_permission_requests(
        {"command": "str_replace", "path": str(sample), "old_str": "a", "new_str": "b"},
        SESSION,
        CONVERSATION,
    )
    luca = plain.build_permission_requests(
        {"file_path": str(sample), "old_string": "a", "new_string": "b"}, SESSION, CONVERSATION
    )

    assert _resources(native) == _resources(luca)


def test_create_asks_for_exactly_what_write_asks_for(tmp_path):
    target = tmp_path / "new.txt"
    tracker = FileReadTracker()
    editor = NativeTextEditorTool(tmp_path, tracker)
    plain = WriteTool(tmp_path, tracker)

    native = editor.build_permission_requests(
        {"command": "create", "path": str(target), "file_text": "x"}, SESSION, CONVERSATION
    )
    luca = plain.build_permission_requests({"file_path": str(target), "content": "x"}, SESSION, CONVERSATION)

    assert _resources(native) == _resources(luca)


def test_insert_is_gated_as_an_edit(tmp_path, sample):
    editor = NativeTextEditorTool(tmp_path, FileReadTracker())

    requests = editor.build_permission_requests(
        {"command": "insert", "path": str(sample), "insert_line": 1, "new_str": "x"}, SESSION, CONVERSATION
    )

    assert ("edit", str(sample)) in _resources(requests)


# ── bad arguments ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"command": "create"}, "requires file_text"),
        ({"command": "str_replace"}, "requires old_str"),
        ({"command": "insert", "new_str": "x"}, "requires insert_line"),
        ({"command": "insert", "insert_line": 1}, "requires new_str"),
    ],
    ids=["create-no-text", "replace-no-old", "insert-no-line", "insert-no-str"],
)
async def test_a_missing_argument_is_an_error_result_not_a_crash(editor, sample, args, message):
    result = await run(editor, path=str(sample), **args)

    assert result.is_error
    assert message in result.content[0].text


async def test_insert_past_the_end_of_the_file_is_refused(editor, sample):
    await run(editor, command="view", path=str(sample))

    result = await run(editor, command="insert", path=str(sample), insert_line=99, new_str="x")

    assert result.is_error
    assert "out of range" in result.content[0].text


async def test_insert_into_a_missing_file_is_refused(editor, tmp_path):
    result = await run(editor, command="insert", path=str(tmp_path / "nope.txt"), insert_line=0, new_str="x")

    assert result.is_error
    assert "File not found" in result.content[0].text


# ── which routes get a native editor ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("anthropic", "claude-opus-5", TEXT_EDITOR_TYPE_CURRENT),
        ("anthropic", "claude-opus-4-8", TEXT_EDITOR_TYPE_CURRENT),
        ("anthropic", "claude-fable-5", TEXT_EDITOR_TYPE_CURRENT),
        ("anthropic", "claude-3-haiku", TEXT_EDITOR_TYPE_LEGACY),
        ("anthropic", "claude-3-5-sonnet", TEXT_EDITOR_TYPE_LEGACY),
        # serves Claude, but Converse does not accept a tool declared by type
        ("bedrock", "anthropic.claude-opus-5", None),
        # also serves Claude, also not the Messages API
        ("openrouter", "anthropic/claude-sonnet-5", None),
        ("anthropic", "gpt-5.4", None),
        ("openai", "gpt-5.4", None),
        ("faux", "fake-model", None),
        ("not-registered", "claude-opus-5", None),
    ],
)
def test_only_the_anthropic_transport_gets_a_native_editor(provider, model, expected):
    assert native_editor_type(provider, model) == expected


# ── through the real runner ──────────────────────────────────────────────────


async def test_a_native_tool_call_drives_end_to_end(tmp_path):
    """A scripted `tool_use` naming the provider's tool: resolved, approved,
    executed, and recorded like any other call."""
    from luca.agent.contrib.tui.wiring import build_runner
    from luca.agent.core.models import ExecutionStatus, ToolExecution
    from luca.client.testing import FauxProvider, faux_assistant_message, faux_text, faux_tool_call

    target = tmp_path / "hello.py"
    target.write_text("print('old')\n")
    session = AgentSessionRunner.new_session(LLMConfig(model="claude-opus-5", provider="anthropic"))
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call(TEXT_EDITOR_NAME, {"command": "view", "path": str(target)}, id="t1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [
                    faux_tool_call(
                        TEXT_EDITOR_NAME,
                        {"command": "str_replace", "path": str(target), "old_str": "old", "new_str": "new"},
                        id="t2",
                    )
                ],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    runner, _ = build_runner(session, workspace=tmp_path, provider=faux, mode="yolo")

    runner.post_message("edit the file")
    await runner.run()

    executions = [e for e in session.entries.values() if isinstance(e, ToolExecution)]
    assert [e.raw_tool_call.name for e in executions] == [TEXT_EDITOR_NAME, TEXT_EDITOR_NAME]
    assert [e.status for e in executions] == [ExecutionStatus.COMPLETED, ExecutionStatus.COMPLETED]
    assert target.read_text() == "print('new')\n"


async def test_the_wire_list_swaps_the_editor_in_and_lucas_own_out(tmp_path):
    from luca.agent.contrib.tui.wiring import build_runner

    session = AgentSessionRunner.new_session(LLMConfig(model="claude-opus-5", provider="anthropic"))
    runner, _ = build_runner(session, workspace=tmp_path)

    specs = await runner.resolve_tool_specs(runner.main_conversation_id)
    tools = {tool.name: tool.provider_type for tool in runner.build_tool_list(runner.main_conversation_id, specs)}

    assert tools[TEXT_EDITOR_NAME] == TEXT_EDITOR_TYPE_CURRENT
    assert tools[BASH_NAME] == BASH_TYPE
    assert {"read", "edit", "write"}.isdisjoint(tools)
    # the tools with no native counterpart still travel as ordinary ones
    assert tools["glob"] is None
    assert tools["apply_patch"] is None


# ── Anthropic's bash ─────────────────────────────────────────────────────────


async def bash_run(tool, conversation=CONVERSATION, **args):
    return await tool.execute(
        tool.Args.model_validate(args).model_dump(),
        SESSION,
        conversation,
        cancellation_token=CancellationToken(),
    )


async def test_the_bash_spec_carries_the_provider_type(tmp_path):
    tool = NativeBashTool(tmp_path)
    try:
        spec = tool.get_tool_spec()
        assert (spec.name, spec.provider_type, spec.tool_kind) == (BASH_NAME, BASH_TYPE, ToolKind.EXECUTE)
        assert list(spec.input_schema["properties"]) == ["command", "restart"]
    finally:
        await tool.close()


async def test_bash_state_persists_between_calls(tmp_path):
    (tmp_path / "sub").mkdir()
    tool = NativeBashTool(tmp_path)
    try:
        await bash_run(tool, command="cd sub")
        result = await bash_run(tool, command="pwd")

        assert result.content[0].text.strip().endswith("/sub")
    finally:
        await tool.close()


async def test_bash_restart_clears_the_session(tmp_path):
    (tmp_path / "sub").mkdir()
    tool = NativeBashTool(tmp_path)
    try:
        await bash_run(tool, command="cd sub")
        restarted = await bash_run(tool, restart=True)
        result = await bash_run(tool, command="pwd")

        assert "restarted" in restarted.content[0].text
        assert not result.content[0].text.strip().endswith("/sub")
    finally:
        await tool.close()


async def test_each_conversation_gets_its_own_shell(tmp_path):
    # a tool instance is shared by the main agent and every subagent; one
    # session would mean a subagent's `cd` relocating the main agent
    (tmp_path / "sub").mkdir()
    tool = NativeBashTool(tmp_path)
    try:
        await bash_run(tool, conversation="subagent", command="cd sub")
        main = await bash_run(tool, conversation=CONVERSATION, command="pwd")

        assert not main.content[0].text.strip().endswith("/sub")
        assert tool.shell_for("subagent") is not tool.shell_for(CONVERSATION)
    finally:
        await tool.close()


async def test_a_failing_command_is_an_error_result_carrying_the_code(tmp_path):
    tool = NativeBashTool(tmp_path)
    try:
        result = await bash_run(tool, command="sh -c 'exit 3'")

        assert result.is_error
        # "exit", the key luca's own bash writes and the TUI renders from
        assert result.metadata == {"exit": 3, "outcome": "completed", "truncated": False, "output_path": None}
    finally:
        await tool.close()


async def test_bash_with_neither_command_nor_restart_is_an_error(tmp_path):
    tool = NativeBashTool(tmp_path)
    try:
        result = await bash_run(tool)

        assert result.is_error
        assert "requires a command" in result.content[0].text
    finally:
        await tool.close()


def test_bash_asks_for_the_same_permissions_lucas_own_does(tmp_path):
    from luca.agent.contrib.shell.tools import BashTool

    native = NativeBashTool(tmp_path)
    plain = BashTool(workdir=tmp_path)

    assert _resources(native.build_permission_requests({"command": "ls -la"}, SESSION, CONVERSATION)) == _resources(
        plain.build_permission_requests({"command": "ls -la"}, SESSION, CONVERSATION)
    )


def test_restart_asks_for_nothing(tmp_path):
    # it runs no command and touches no directory
    tool = NativeBashTool(tmp_path)

    assert tool.build_permission_requests({"restart": True}, SESSION, CONVERSATION) == []


async def test_closing_the_tool_ends_every_shell_it_opened(tmp_path):
    tool = NativeBashTool(tmp_path)
    await bash_run(tool, conversation="a", command="true")
    await bash_run(tool, conversation="b", command="true")
    shells = [tool.shell_for("a"), tool.shell_for("b")]

    await tool.close()

    assert not any(shell.running for shell in shells)


# ── OpenAI's apply_patch ─────────────────────────────────────────────────────


def test_an_update_operation_becomes_a_patch_envelope():
    envelope = to_patch_envelope({"type": "update_file", "path": "lib/fib.py", "diff": "@@\n-a\n+b"})

    assert envelope == "*** Begin Patch\n*** Update File: lib/fib.py\n@@\n-a\n+b\n*** End Patch"


def test_a_create_operation_uses_the_add_header():
    envelope = to_patch_envelope({"type": "create_file", "path": "new.py", "diff": "+hello"})

    assert envelope == "*** Begin Patch\n*** Add File: new.py\n+hello\n*** End Patch"


def test_a_delete_operation_carries_no_diff():
    envelope = to_patch_envelope({"type": "delete_file", "path": "gone.py"})

    assert envelope == "*** Begin Patch\n*** Delete File: gone.py\n*** End Patch"


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        ({"type": "rename_file", "path": "a"}, "unknown apply_patch operation"),
        ({"type": "update_file"}, "requires a path"),
        ({"type": "update_file", "path": "a"}, "requires a diff"),
    ],
    ids=["unknown-type", "no-path", "no-diff"],
)
def test_a_malformed_operation_is_refused(operation, message):
    with pytest.raises(ShellToolError, match=message):
        to_patch_envelope(operation)


async def test_apply_patch_creates_a_file(tmp_path):
    tool = NativeApplyPatchTool(tmp_path)

    result = await tool.execute(
        {"type": "create_file", "path": "greet.py", "diff": "+print('hi')"},
        SESSION,
        CONVERSATION,
        cancellation_token=CancellationToken(),
    )

    assert not result.is_error
    assert (tmp_path / "greet.py").read_text() == "print('hi')\n"


async def test_apply_patch_deletes_a_file(tmp_path):
    doomed = tmp_path / "gone.py"
    doomed.write_text("x\n")
    tool = NativeApplyPatchTool(tmp_path)

    result = await tool.execute(
        {"type": "delete_file", "path": "gone.py"},
        SESSION,
        CONVERSATION,
        cancellation_token=CancellationToken(),
    )

    assert not result.is_error
    assert not doomed.exists()


def test_apply_patch_is_gated_like_lucas_own(tmp_path):
    from luca.agent.contrib.shell.tools import ApplyPatchTool

    native = NativeApplyPatchTool(tmp_path)
    plain = ApplyPatchTool(workdir=tmp_path)
    operation = {"type": "create_file", "path": "a.py", "diff": "+x"}

    assert _resources(native.build_permission_requests(operation, SESSION, CONVERSATION)) == _resources(
        plain.build_permission_requests({"patch_text": to_patch_envelope(operation)}, SESSION, CONVERSATION)
    )


# ── OpenAI's shell ───────────────────────────────────────────────────────────


async def shell_run(tool, conversation=CONVERSATION, **args):
    return await tool.execute(
        tool.Args.model_validate(args).model_dump(),
        SESSION,
        conversation,
        cancellation_token=CancellationToken(),
    )


async def test_shell_runs_every_command_in_order(tmp_path):
    tool = NativeShellTool(tmp_path)
    try:
        result = await tool.execute(
            {"commands": ["echo one", "echo two"]},
            SESSION,
            CONVERSATION,
            cancellation_token=CancellationToken(),
        )

        assert not result.is_error
        assert "one" in result.content[0].text
        assert "two" in result.content[0].text
    finally:
        await tool.close()


async def test_shell_commands_share_one_session(tmp_path):
    (tmp_path / "sub").mkdir()
    tool = NativeShellTool(tmp_path)
    try:
        result = await tool.execute(
            {"commands": ["cd sub", "pwd"]},
            SESSION,
            CONVERSATION,
            cancellation_token=CancellationToken(),
        )

        assert "/sub" in result.content[0].text
    finally:
        await tool.close()


async def test_shell_stops_at_the_first_failure(tmp_path):
    # they run in order, so a later command usually assumes the earlier worked
    tool = NativeShellTool(tmp_path)
    try:
        result = await tool.execute(
            {"commands": ["false", "echo unreachable"]},
            SESSION,
            CONVERSATION,
            cancellation_token=CancellationToken(),
        )

        assert result.is_error
        assert "unreachable" not in result.content[0].text
    finally:
        await tool.close()


def test_shell_asks_about_every_command_not_just_the_first(tmp_path):
    # approving a batch by its first line is how the rest get in unseen
    tool = NativeShellTool(tmp_path)

    requests = tool.build_permission_requests({"commands": ["ls", "rm -rf /"]}, SESSION, CONVERSATION)

    assert ("bash", "ls") in _resources(requests)
    assert ("bash", "rm -rf /") in _resources(requests)


async def test_shell_with_no_commands_is_an_error(tmp_path):
    tool = NativeShellTool(tmp_path)
    try:
        result = await tool.execute({"commands": []}, SESSION, CONVERSATION, cancellation_token=CancellationToken())

        assert result.is_error
        assert "at least one command" in result.content[0].text
    finally:
        await tool.close()


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("openai", "gpt-5.4", ("apply_patch", "shell")),
        ("openai", "gpt-5.6", ("apply_patch", "shell")),
        ("openai", "gpt-4o", ()),  # predates the tools
        ("groq", "gpt-5.4", ()),  # chat completions: no shell item types
        ("anthropic", "claude-opus-5", ()),
        ("faux", "fake-model", ()),
    ],
)
def test_only_the_responses_transport_gets_openais_tools(provider, model, expected):
    assert native_openai_tool_types(provider, model) == expected


# ── output caps, cancellation and the paging hint ────────────────────────────


async def test_native_bash_truncates_like_lucas_own_bash(tmp_path):
    """`bash` caps at 2000 lines / 50 KiB and spills the rest to a file. The
    native replacement covers the same ground, so one `cat` of a build log
    would otherwise go into the request, the response, and every save of the
    session file."""
    tool = NativeBashTool(tmp_path, output_dir=str(tmp_path))
    try:
        result = await bash_run(tool, command="for i in $(seq 1 5000); do echo line-$i; done")

        text = result.content[0].text
        assert text.startswith("...output truncated...")
        assert result.metadata["truncated"] is True
        assert Path(result.metadata["output_path"]).read_text().count("\n") == 5000
    finally:
        await tool.close()


async def test_native_shell_honours_max_output_length(tmp_path):
    # declared in the provider's schema, so a model that asks for a bound gets one
    tool = NativeShellTool(tmp_path, output_dir=str(tmp_path))
    try:
        result = await shell_run(tool, commands=["python3 -c \"print('x' * 20000)\""], max_output_length=500)

        assert len(result.content[0].text.encode()) < 5_000
        assert result.metadata["truncated"] is True
    finally:
        await tool.close()


async def test_native_shell_leaves_small_output_alone(tmp_path):
    tool = NativeShellTool(tmp_path)
    try:
        result = await shell_run(tool, commands=["echo hi"], max_output_length=500)

        assert result.content[0].text == "$ echo hi\nhi\n"
        assert result.metadata == {"exit": 0, "outcome": "completed", "truncated": False, "output_path": None}
    finally:
        await tool.close()


async def test_cancelling_native_bash_returns_partial_output_and_says_the_session_reset(tmp_path):
    """luca's own `bash` hands back what the command printed before the ESC.
    The native one dropped the token entirely, so the runner hard-cancelled and
    the model got a bare `[tool execution interrupted]` — and the killed shell
    took the working directory and environment with it, silently."""
    tool = NativeBashTool(tmp_path)
    token = CancellationToken()
    asyncio.get_running_loop().call_later(0.5, token.cancel)
    try:
        result = await tool.execute(
            tool.Args.model_validate({"command": "echo early; sleep 30"}).model_dump(),
            SESSION,
            CONVERSATION,
            cancellation_token=token,
        )

        assert result.is_error
        assert "early" in result.content[0].text
        assert "shell session was reset" in result.content[0].text
        assert result.metadata["outcome"] == "cancelled"
    finally:
        await tool.close()


async def test_native_bash_sets_no_outer_deadline(tmp_path):
    """The runner's deadline starts before the spawn, so one equal to the
    tool's own always wins: the tool's timeout branch would be dead code and
    the model would get `[tool execution timed_out]` with no output."""
    tool = NativeBashTool(tmp_path)
    try:
        assert tool.get_tool_spec().timeout_in_ms is None
    finally:
        await tool.close()


async def test_a_long_view_points_at_view_range_not_offset(editor, tmp_path):
    # `offset` is not in the native schema, so a model that follows the hint
    # gets a validation error and burns a round trip
    path = tmp_path / "long.txt"
    path.write_text("".join(f"line {n}\n" for n in range(1, 3001)))

    result = await run(editor, command="view", path=str(path))

    assert "Use view_range=[2001, -1] to continue." in result.content[0].text
    assert "offset=" not in result.content[0].text


def test_the_legacy_editor_uses_the_name_that_goes_with_its_type(tmp_path):
    # Anthropic pairs the type and the name; changing one alone is a 400
    legacy = NativeTextEditorTool(tmp_path, provider_type="text_editor_20250124")
    current = NativeTextEditorTool(tmp_path, provider_type="text_editor_20250728")

    assert (legacy.name, current.name) == ("str_replace_editor", "str_replace_based_edit_tool")
    assert legacy.get_tool_spec().name == "str_replace_editor"


async def test_a_finished_subagents_shell_is_released(tmp_path):
    """One shell per conversation, and nothing resumes a subagent — so without
    a release a run that spawns forty of them ends holding forty-one idle
    shells plus whatever each left in the background."""
    session = AgentSessionRunner.new_session(LLMConfig(model="claude-opus-5", provider="anthropic"))
    child = conversation("child", depth=1)
    session.conversations[child.id] = child
    tool = NativeBashTool(tmp_path)
    try:
        await tool.execute(
            tool.Args.model_validate({"command": "echo hi"}).model_dump(),
            session,
            child.id,
            cancellation_token=CancellationToken(),
        )
        assert len(tool._shells) == 1

        await tool.execute(
            tool.Args.model_validate({"command": "echo hi"}).model_dump(),
            session,
            session.main_conversation_id,
            cancellation_token=CancellationToken(),
        )

        assert len(tool._shells) == 1
        assert session.main_conversation_id in tool._shells._shells
    finally:
        await tool.close()
