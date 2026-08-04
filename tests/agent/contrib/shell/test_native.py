"""Anthropic's `str_replace_based_edit_tool`, executed by luca.

The provider owns the schema, so the interesting questions are the ones the
schema does not answer: does the approval gate still fire with the same rules a
`read` or an `edit` would produce, does the read-first guard still hold, and
does each command reach the tool that already implements it. Real files under
`tmp_path`; no runner.
"""

import pytest

from luca.agent.contrib.shell.native import (
    TEXT_EDITOR_NAME,
    TEXT_EDITOR_TYPE_CURRENT,
    TEXT_EDITOR_TYPE_LEGACY,
    NativeTextEditorTool,
    native_editor_type,
)
from luca.agent.contrib.shell.tools import EditTool, FileReadTracker, ReadTool, WriteTool
from luca.agent.core import CancellationToken
from luca.agent.core.models import LLMConfig, ToolKind
from luca.agent.core.runner import AgentSessionRunner

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
    assert {"read", "edit", "write"}.isdisjoint(tools)
    # everything else still travels as an ordinary tool
    assert tools["bash"] is None
