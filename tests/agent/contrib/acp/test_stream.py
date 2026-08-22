"""The event → update mapping, one event at a time.

`test_agent.py` proves the mapping works over a whole turn. This proves the
individual translations are the ones intended, including the ones the faux
conversation never reaches: a failed call, a diff, a plan, a redacted thought.
"""

from __future__ import annotations

import pytest

from luca.agent.contrib.acp.stream import (
    TOOL_KINDS,
    TOOL_STATUSES,
    Translator,
    tool_diffs,
    tool_locations,
    tool_title,
)
from luca.agent.core.events import (
    ApprovalRequired,
    FinishReason,
    ReasoningBlock,
    ReasoningDelta,
    SubagentsSpawned,
    TextBlock,
    TextDelta,
    ToolCallReceived,
    ToolExecuted,
    ToolExecutionStarted,
)
from luca.agent.core.models import (
    ExecutionResult,
    ExecutionStatus,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolKind,
    ToolSpec,
)

MAIN = "conv_main"
CHILD = "conv_child"


def execution(
    *,
    name: str = "edit",
    arguments: dict | None = None,
    kind: ToolKind = ToolKind.EDIT,
    status: ExecutionStatus = ExecutionStatus.COMPLETED,
    metadata: dict | None = None,
    conversation_id: str = MAIN,
    call_id: str = "tc_1",
) -> ToolExecution:
    return ToolExecution(
        id="ex_1",
        conversation_id=conversation_id,
        tool_call_id=call_id,
        raw_tool_call=ToolCall(id=call_id, name=name, arguments=arguments or {}),
        tool_spec=ToolSpec(name=name, description="", input_schema={}, tool_kind=kind),
        status=status,
        result=ExecutionResult(content=[TextContent(text="ok")], metadata=metadata or {}) if metadata else None,
    )


# ── the two vocabularies ─────────────────────────────────────────────────────


def test_every_tool_kind_we_have_maps_to_one_acp_kind():
    """A kind added to core without a mapping here would silently render as
    the generic icon for every call."""
    assert set(TOOL_KINDS) == set(ToolKind)


def test_every_execution_status_maps_to_one_acp_status():
    assert set(TOOL_STATUSES) == set(ExecutionStatus)
    assert set(TOOL_STATUSES.values()) == {"pending", "in_progress", "completed", "failed"}


def test_web_fetch_is_the_one_name_that_differs():
    assert TOOL_KINDS[ToolKind.WEB_FETCH] == "fetch"


# ── titles ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"file_path": "/tmp/a.py"}, "edit /tmp/a.py"),
        ({"command": "ls -la"}, "edit ls -la"),
        ({}, "edit"),
        ({"unrecognised": "x"}, "edit"),
        ({"command": "first\nsecond"}, "edit first"),
    ],
)
def test_the_title_names_the_most_telling_argument(arguments, expected):
    assert tool_title(execution(arguments=arguments)) == expected


def test_a_long_argument_is_elided_rather_than_wrapped():
    title = tool_title(execution(arguments={"command": "x" * 200}))

    assert len(title) < 80
    assert title.endswith("…")


# ── diffs and locations ──────────────────────────────────────────────────────


def test_an_edits_metadata_becomes_one_diff_block():
    call = execution(metadata={"path": "/tmp/a.py", "old_text": "a\n", "new_text": "b\n"})

    [diff] = tool_diffs(call)

    assert (diff.type, diff.path, diff.old_text, diff.new_text) == ("diff", "/tmp/a.py", "a\n", "b\n")


def test_an_apply_patch_becomes_one_diff_block_per_file():
    call = execution(
        name="apply_patch",
        metadata={
            "files": [
                {"absolute_path": "/tmp/a.py", "old_text": "a\n", "new_text": "b\n"},
                {"absolute_path": "/tmp/b.py", "old_text": None, "new_text": "new\n"},
            ]
        },
    )

    diffs = tool_diffs(call)

    assert [(d.path, d.old_text) for d in diffs] == [("/tmp/a.py", "a\n"), ("/tmp/b.py", None)]


def test_a_tool_that_changed_nothing_has_no_diffs_and_no_locations():
    call = execution(name="bash", kind=ToolKind.EXECUTE, metadata={"exit_code": 0})

    assert tool_diffs(call) == []
    assert tool_locations(call) == []


def test_locations_come_from_the_result_not_the_arguments():
    """Only the tool knows what a relative path resolved to."""
    call = execution(arguments={"file_path": "a.py"}, metadata={"path": "/abs/a.py", "new_text": "x"})

    assert tool_locations(call) == [{"path": "/abs/a.py"}]


# ── the translator ───────────────────────────────────────────────────────────


def test_text_deltas_share_a_message_id_until_a_tool_call_breaks_them_up():
    translator = Translator(MAIN)

    [first] = translator.translate(TextDelta(conversation_id=MAIN, text="Hello "))
    [second] = translator.translate(TextDelta(conversation_id=MAIN, text="world"))
    translator.translate(
        ToolCallReceived(conversation_id=MAIN, tool_call_id="tc_1", execution=execution(status=ExecutionStatus.PENDING))
    )
    [third] = translator.translate(TextDelta(conversation_id=MAIN, text="Done"))

    assert first.message_id == second.message_id
    assert third.message_id != first.message_id


def test_an_empty_delta_produces_nothing():
    """Providers emit empty chunks. Forwarding them means a client redraws for
    no reason, once per chunk."""
    assert Translator(MAIN).translate(TextDelta(conversation_id=MAIN, text="")) == []


def test_a_redacted_thought_is_not_forwarded():
    """The provider withheld the body and sent only an encrypted attestation;
    there is nothing to show and an empty bubble is worse than none."""
    translator = Translator(MAIN)

    assert translator.translate(ReasoningBlock(conversation_id=MAIN, text="", redacted=True)) == []


def test_a_failed_call_reports_failed_whatever_the_status_says():
    """`is_error` is the projection's verdict: a tool can return COMPLETED with
    an error result, and the client should show the error."""
    translator = Translator(MAIN)
    call = execution(status=ExecutionStatus.COMPLETED)

    [update] = translator.translate(
        ToolExecuted(
            conversation_id=MAIN,
            tool_call_id="tc_1",
            execution=call,
            result_text="no such file",
            is_error=True,
        )
    )

    assert update.status == "failed"


def test_the_started_event_becomes_an_in_progress_update():
    translator = Translator(MAIN)

    [update] = translator.translate(
        ToolExecutionStarted(conversation_id=MAIN, tool_call_id="tc_1", execution=execution())
    )

    assert (update.session_update, update.tool_call_id, update.status) == ("tool_call_update", "tc_1", "in_progress")


@pytest.mark.parametrize(
    "event",
    [
        FinishReason(conversation_id=MAIN, finish_reason="stop"),
        ApprovalRequired(conversation_id=MAIN, executions=[]),
    ],
)
def test_events_with_nothing_to_render_translate_to_nothing(event):
    assert Translator(MAIN).translate(event) == []


# ── subagents ────────────────────────────────────────────────────────────────


def test_a_child_with_no_known_spawn_call_is_dropped():
    """Reachable on a resumed session: the spawn happened in a previous
    process and this translator never saw it."""
    translator = Translator(MAIN)

    assert translator.translate(TextDelta(conversation_id=CHILD, text="orphan")) == []


def test_a_childs_text_arrives_as_content_on_its_spawn_call():
    translator = Translator(MAIN)
    translator.translate(
        ToolCallReceived(
            conversation_id=MAIN,
            tool_call_id="tc_spawn",
            execution=execution(name="spawn_subagent", call_id="tc_spawn", status=ExecutionStatus.PENDING),
        )
    )
    translator.translate(SubagentsSpawned(conversation_id=MAIN, conversation_ids=[CHILD]))

    [update] = translator.translate(TextDelta(conversation_id=CHILD, text="child says hi"))

    assert update.session_update == "tool_call_update"
    assert update.tool_call_id == "tc_spawn"
    assert update.content[0].content.text == "child says hi"


def test_a_childs_thinking_is_not_forwarded_at_all():
    """One nested voice is enough. A child's reasoning folded onto the spawn
    call would bury its actual output."""
    translator = Translator(MAIN)
    translator.translate(
        ToolCallReceived(
            conversation_id=MAIN,
            tool_call_id="tc_spawn",
            execution=execution(name="spawn_subagent", call_id="tc_spawn", status=ExecutionStatus.PENDING),
        )
    )
    translator.translate(SubagentsSpawned(conversation_id=MAIN, conversation_ids=[CHILD]))

    assert translator.translate(ReasoningBlock(conversation_id=CHILD, text="hmm")) == []


# ── the two event tiers ──────────────────────────────────────────────────────


def test_a_streaming_run_forwards_deltas_and_drops_the_completed_block():
    """Under `streaming=True` text arrives TWICE — once as deltas while tokens
    land, once more as the finished block. Forwarding both sends every
    sentence to the client twice, which is what a live run showed."""
    translator = Translator(MAIN, streaming=True)

    delta = translator.translate(TextDelta(conversation_id=MAIN, text="Hello"))
    block = translator.translate(TextBlock(conversation_id=MAIN, text="Hello"))

    assert len(delta) == 1
    assert block == []


def test_a_non_streaming_run_forwards_the_block_and_never_sees_a_delta():
    """Blocks are the only tier a non-streaming run emits, so dropping them
    there would send the client nothing at all."""
    translator = Translator(MAIN, streaming=False)

    block = translator.translate(TextBlock(conversation_id=MAIN, text="Hello"))
    delta = translator.translate(TextDelta(conversation_id=MAIN, text="Hello"))

    assert len(block) == 1
    assert delta == []


def test_reasoning_follows_the_same_rule_as_text():
    streaming = Translator(MAIN, streaming=True)

    assert len(streaming.translate(ReasoningDelta(conversation_id=MAIN, text="hmm"))) == 1
    assert streaming.translate(ReasoningBlock(conversation_id=MAIN, text="hmm")) == []
