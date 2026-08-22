"""What every editing tool records for a client that renders the change.

`metadata["diff"]` is a unified-diff STRING, which is all the TUI ever needed.
An editor speaking ACP needs the two sides instead: its diff view takes
`{path, oldText, newText}` and renders the change itself. So each of these
tools also records `path`, `old_text` and `new_text`, and `old_text` is None
exactly when the file did not exist before the call.

The assertions here are on the WHOLE metadata dict on purpose. A key added
later shows up as a diff rather than passing unnoticed, and the expected value
doubles as the record of what the contract is.
"""

import pytest

from luca.agent.contrib.shell import ApplyPatchTool, EditTool, FileReadTracker, WriteTool
from tests.agent.contrib.shell.conftest import CONVERSATION


def tracked(cls, tmp_path, *read_paths):
    tracker = FileReadTracker()
    for path in read_paths:
        tracker.record(CONVERSATION, path)
    return cls(workdir=tmp_path, tracker=tracker)


def patcher(tmp_path):
    """`apply_patch` carries the whole change in its own text, so it has no
    tracker and no read-first contract."""
    return ApplyPatchTool(workdir=tmp_path)


# ── edit ─────────────────────────────────────────────────────────────────────


async def test_edit_records_both_sides_of_the_change(tmp_path, run):
    target = tmp_path / "greeting.txt"
    target.write_text("Hello World\n")

    result = await run(
        tracked(EditTool, tmp_path, target),
        {"file_path": "greeting.txt", "old_string": "World", "new_string": "Moon"},
    )

    assert result.metadata == {
        "diff": result.metadata["diff"],  # the string form, asserted elsewhere
        "created": False,
        "replacements": 1,
        "path": str(target),
        "old_text": "Hello World\n",
        "new_text": "Hello Moon\n",
    }


async def test_edit_creating_a_file_has_no_old_text(tmp_path, run):
    target = tmp_path / "new.txt"

    result = await run(
        tracked(EditTool, tmp_path),
        {"file_path": "new.txt", "old_string": "", "new_string": "fresh\n"},
    )

    assert result.metadata == {
        "diff": result.metadata["diff"],
        "created": True,
        "path": str(target),
        "old_text": None,
        "new_text": "fresh\n",
    }


async def test_edit_records_the_normalised_text_not_the_bytes_written(tmp_path, run):
    """A CRLF file is written back as CRLF, but both recorded sides are LF.

    The tool works in LF internally and re-applies the file's own endings on
    the way out. Recording the bytes would hand a client two strings that
    differ on every line, and it would render as a whole-file rewrite."""
    target = tmp_path / "dos.txt"
    target.write_bytes(b"alpha\r\nbeta\r\n")

    result = await run(
        tracked(EditTool, tmp_path, target),
        {"file_path": "dos.txt", "old_string": "beta", "new_string": "gamma"},
    )

    assert (result.metadata["old_text"], result.metadata["new_text"]) == (
        "alpha\nbeta\n",
        "alpha\ngamma\n",
    )
    assert target.read_bytes() == b"alpha\r\ngamma\r\n"


# ── write ────────────────────────────────────────────────────────────────────


async def test_write_over_an_existing_file_records_what_it_replaced(tmp_path, run):
    target = tmp_path / "notes.txt"
    target.write_text("old body\n")

    result = await run(
        tracked(WriteTool, tmp_path, target),
        {"file_path": "notes.txt", "content": "new body\n"},
    )

    assert result.metadata == {
        "existed": True,
        "path": str(target),
        "old_text": "old body\n",
        "new_text": "new body\n",
    }


async def test_write_to_a_new_file_has_no_old_text(tmp_path, run):
    result = await run(
        tracked(WriteTool, tmp_path),
        {"file_path": "fresh.txt", "content": "hello\n"},
    )

    assert result.metadata == {
        "existed": False,
        "path": str(tmp_path / "fresh.txt"),
        "old_text": None,
        "new_text": "hello\n",
    }


# ── apply_patch ──────────────────────────────────────────────────────────────


async def test_apply_patch_records_both_sides_per_file(tmp_path, run):
    target = tmp_path / "app.py"
    target.write_text("import os\nprint(os)\n")
    patch = "*** Begin Patch\n*** Update File: app.py\n@@\n-print(os)\n+print(os.getcwd())\n*** End Patch"

    result = await run(patcher(tmp_path), {"patch_text": patch})

    [entry] = result.metadata["files"]
    assert entry == {
        "path": "app.py",  # what the model wrote, which the format lets be relative
        "type": "update",
        "patch": entry["patch"],
        "additions": 1,
        "deletions": 1,
        "move_to": None,
        "absolute_path": str(target),
        "old_text": "import os\nprint(os)\n",
        "new_text": "import os\nprint(os.getcwd())\n",
    }


async def test_apply_patch_adding_a_file_has_no_old_text(tmp_path, run):
    patch = "*** Begin Patch\n*** Add File: made.py\n+print(1)\n*** End Patch"

    result = await run(patcher(tmp_path), {"patch_text": patch})

    [entry] = result.metadata["files"]
    assert (entry["absolute_path"], entry["old_text"], entry["new_text"]) == (
        str(tmp_path / "made.py"),
        None,
        "print(1)\n",
    )


async def test_apply_patch_move_records_the_destination(tmp_path, run):
    """`absolute_path` is where the file IS after the call, not where it was.
    A client opens that path to show the change."""
    source = tmp_path / "before.py"
    source.write_text("value = 1\n")
    patch = (
        "*** Begin Patch\n*** Update File: before.py\n*** Move to: after.py\n@@\n-value = 1\n+value = 2\n*** End Patch"
    )

    result = await run(patcher(tmp_path), {"patch_text": patch})

    [entry] = result.metadata["files"]
    assert (entry["absolute_path"], entry["move_to"]) == (str(tmp_path / "after.py"), "after.py")
    assert not source.exists()


@pytest.mark.parametrize("tool", [EditTool, WriteTool])
async def test_a_failed_call_records_nothing(tmp_path, run, tool):
    """The read-first contract refuses the call, so there is no change to
    describe and no half-filled metadata to mislead a client."""
    target = tmp_path / "unread.txt"
    target.write_text("untouched\n")

    result = await run(
        tool(workdir=tmp_path, tracker=FileReadTracker()),
        {"file_path": "unread.txt", "old_string": "untouched", "new_string": "x"}
        if tool is EditTool
        else {"file_path": "unread.txt", "content": "x"},
    )

    assert result.is_error
    assert "old_text" not in result.metadata
