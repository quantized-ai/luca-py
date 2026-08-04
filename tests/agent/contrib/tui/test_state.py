"""The `ScreenState` fixture schema — the view-model as declarative data.

A gallery fixture is exactly a dict like `FIXTURE` below; these tests pin
that the discriminated block union types it, that `extra="forbid"` rejects a
misspelled fixture instead of silently dropping the key, and that the dock
precedence the frame renders by is a property of the data.
"""

import pytest
from pydantic import ValidationError

from luca.agent.contrib.tui import state as vm

# One dict shaped exactly like a YAML fixture, covering every block kind.
FIXTURE = {
    "name": "components/sample",
    "status": {
        "cwd": "~/quantized/luca",
        "model": "sonnet-4.5",
        "branch": "main",
        "dirty": True,
        "tokens": "12.4k",
        "cost": "$0.08",
    },
    "transcript": [
        {"kind": "user", "text": "add a --json flag"},
        {"kind": "thinking", "duration": 3},
        {"kind": "text", "text": "On it.", "streaming": True},
        {"kind": "tool", "tool": "edit", "arg": "luca/cli.py", "result": {"stat": {"add": 7, "del": 1}}},
        {
            "kind": "tool",
            "tool": "bash",
            "arg": "pytest -q",
            "status": "error",
            "output": {
                "lines": ["FAILED test_x"],
                "summary": "exit 1 · 1.9s",
                "expand_hint": True,
                "hidden_lines": ["boom"],
            },
        },
        {
            "kind": "list",
            "label": "plan · 1 of 2",
            "rows": [{"glyph": "done", "text": "read"}, {"glyph": "active", "text": "edit"}],
        },
        {
            "kind": "diff",
            "lines": [{"num": 41, "sign": "-", "text": "old"}, {"num": 41, "sign": "+", "text": "new"}],
        },
        {"kind": "notice", "text": "turn cancelled"},
        {
            "kind": "task",
            "description": "check arithmetic",
            "status": "done",
            "blocks": [{"kind": "tool", "tool": "multiply", "arg": "a=6, b=7", "result": {"summary": "42"}}],
        },
    ],
    "composer": {"placeholder": "ask, or / for commands"},
    "hints": ["enter send", "^p palette"],
}


def test_fixture_dict_validates_to_the_typed_block_union():
    assert vm.ScreenState.model_validate(FIXTURE) == vm.ScreenState(
        name="components/sample",
        status=vm.StatusState(
            cwd="~/quantized/luca",
            model="sonnet-4.5",
            branch="main",
            dirty=True,
            tokens="12.4k",
            cost="$0.08",
        ),
        transcript=[
            vm.UserBlock(text="add a --json flag"),
            vm.ThinkingBlock(duration=3),
            vm.TextBlock(text="On it.", streaming=True),
            vm.ToolBlock(tool="edit", arg="luca/cli.py", result=vm.ToolResult(stat=vm.DiffStat(add=7, remove=1))),
            vm.ToolBlock(
                tool="bash",
                arg="pytest -q",
                status="error",
                output=vm.ToolOutput(
                    lines=["FAILED test_x"],
                    summary="exit 1 · 1.9s",
                    expand_hint=True,
                    hidden_lines=["boom"],
                ),
            ),
            vm.ListBlock(
                label="plan · 1 of 2",
                rows=[vm.ListRow(glyph="done", text="read"), vm.ListRow(glyph="active", text="edit")],
            ),
            vm.DiffBlock(lines=[vm.DiffLine(num=41, sign="-", text="old"), vm.DiffLine(num=41, sign="+", text="new")]),
            vm.NoticeBlock(text="turn cancelled"),
            vm.TaskBlock(
                description="check arithmetic",
                status="done",
                blocks=[vm.ToolBlock(tool="multiply", arg="a=6, b=7", result=vm.ToolResult(summary="42"))],
            ),
        ],
        composer=vm.ComposerState(placeholder="ask, or / for commands"),
        hints=["enter send", "^p palette"],
    )


def test_screen_state_round_trips_through_a_plain_dict():
    state = vm.ScreenState.model_validate(FIXTURE)
    assert vm.ScreenState.model_validate(state.model_dump()) == state


def test_an_unknown_top_level_key_is_rejected():
    with pytest.raises(ValidationError):
        vm.ScreenState.model_validate({"unknown": 1})


def test_an_unknown_block_key_is_rejected():
    with pytest.raises(ValidationError):
        vm.ScreenState.model_validate({"transcript": [{"kind": "user", "text": "hi", "color": "red"}]})


def test_an_unknown_block_kind_is_rejected():
    with pytest.raises(ValidationError):
        vm.ScreenState.model_validate({"transcript": [{"kind": "banner", "text": "x"}]})


def test_diff_stat_reads_the_del_alias_and_the_field_name():
    assert vm.DiffStat.model_validate({"add": 3, "del": 1}) == vm.DiffStat(add=3, remove=1)


def test_diff_stat_serializes_back_to_the_alias():
    assert vm.DiffStat(add=3, remove=1).model_dump(by_alias=True) == {"add": 3, "del": 1}


def test_dock_prefers_approval_then_overlay_then_composer():
    approval = vm.ApprovalState(question="Run bash?", options=[vm.ApprovalOption(label="Approve once")])
    overlay = vm.OverlayState()
    assert vm.ScreenState(approval=approval, overlay=overlay).dock() == "approval"
    assert vm.ScreenState(overlay=overlay).dock() == "overlay"
    assert vm.ScreenState().dock() == "composer"
