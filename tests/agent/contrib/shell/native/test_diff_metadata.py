"""The provider-native editors record the same two sides the generic ones do.

A model gets `apply_patch` / `text_editor` instead of `edit` / `write` when its
provider ships them, and which set it got is not something a client should be
able to feel. So these tools carry the same `path` / `old_text` / `new_text`
that `tools.py` writes, alongside the unified-diff string the TUI reads.

See `tests/agent/contrib/shell/tools/test_edit_diff_metadata.py` for the
generic half and for why the assertions are on the whole dict.
"""

from luca.agent.contrib.shell import (
    AnthropicTextEditorTool,
    FileReadTracker,
    OpenAIApplyPatchTool,
)
from tests.agent.contrib.shell.conftest import CONVERSATION


def patcher(tmp_path, *read_paths) -> OpenAIApplyPatchTool:
    tracker = FileReadTracker()
    for path in read_paths:
        tracker.record(CONVERSATION, path)
    return OpenAIApplyPatchTool(workdir=tmp_path, tracker=tracker)


def editor(tmp_path, *read_paths) -> AnthropicTextEditorTool:
    tracker = FileReadTracker()
    for path in read_paths:
        tracker.record(CONVERSATION, path)
    return AnthropicTextEditorTool(workdir=tmp_path, tracker=tracker)


# ── anthropic text_editor ────────────────────────────────────────────────────


async def test_text_editor_str_replace_records_both_sides(tmp_path, run):
    target = tmp_path / "notes.txt"
    target.write_text("alpha\nbeta\n")

    result = await run(
        editor(tmp_path, target),
        {"command": "str_replace", "path": "notes.txt", "old_str": "beta", "new_str": "gamma"},
    )

    assert result.metadata == {
        "diff": result.metadata["diff"],
        "created": False,
        "path": str(target),
        "old_text": "alpha\nbeta\n",
        "new_text": "alpha\ngamma\n",
    }


async def test_text_editor_create_on_a_new_file_has_no_old_text(tmp_path, run):
    result = await run(
        editor(tmp_path),
        {"command": "create", "path": "fresh.txt", "file_text": "hello\n"},
    )

    assert result.metadata == {
        "diff": result.metadata["diff"],
        "created": True,
        "path": str(tmp_path / "fresh.txt"),
        "old_text": None,
        "new_text": "hello\n",
    }


async def test_text_editor_create_over_an_existing_file_records_what_it_replaced(tmp_path, run):
    target = tmp_path / "notes.txt"
    target.write_text("old\n")

    result = await run(
        editor(tmp_path, target),
        {"command": "create", "path": "notes.txt", "file_text": "new\n"},
    )

    assert (result.metadata["created"], result.metadata["old_text"], result.metadata["new_text"]) == (
        False,
        "old\n",
        "new\n",
    )


# ── openai apply_patch ───────────────────────────────────────────────────────


async def test_apply_patch_update_records_both_sides(tmp_path, run):
    target = tmp_path / "notes.txt"
    target.write_text("alpha\nbeta\n")

    result = await run(
        patcher(tmp_path, target),
        {"type": "update_file", "path": "notes.txt", "diff": "@@\n-beta\n+gamma\n"},
    )

    assert result.metadata == {
        "diff": result.metadata["diff"],
        "created": False,
        "move_to": None,
        "path": str(target),
        "old_text": "alpha\nbeta\n",
        "new_text": "alpha\ngamma\n",
    }


async def test_apply_patch_create_has_no_old_text(tmp_path, run):
    result = await run(
        patcher(tmp_path),
        {"type": "create_file", "path": "hello.txt", "diff": "+Hello\n+\n"},
    )

    assert (result.metadata["path"], result.metadata["old_text"], result.metadata["new_text"]) == (
        str(tmp_path / "hello.txt"),
        None,
        "Hello\n",
    )


async def test_apply_patch_delete_records_an_empty_new_text(tmp_path, run):
    """A delete is a change to nothing, not an absent change: a client renders
    the whole file as removed."""
    target = tmp_path / "gone.txt"
    target.write_text("bye\n")

    result = await run(patcher(tmp_path, target), {"type": "delete_file", "path": "gone.txt"})

    assert (result.metadata["path"], result.metadata["old_text"], result.metadata["new_text"]) == (
        str(target),
        "bye\n",
        "",
    )
    assert not target.exists()
