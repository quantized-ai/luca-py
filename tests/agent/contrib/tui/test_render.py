"""Pure derivations in `render.py`: entries → transcript view-models.

Each test: a known execution/session literal, one call, a full-object assert
on the resulting view-model. `execution()` is the module-level literal factory
(the `spec()` idiom): identity fields are never what a test is about.
"""

from luca.agent.contrib.questions import QUESTIONS_NAMESPACE, OptionsType, Question
from luca.agent.contrib.tui import state as vm
from luca.agent.contrib.tui.render import (
    QUESTION_MEASURE,
    answer_payload,
    child_links,
    diff_stat,
    entry_blocks,
    is_questions_tool,
    is_runtime_plumbing,
    plan_block,
    plan_dismissed,
    plan_from_session,
    preview_rows,
    question_blocks,
    question_set_state,
    question_tab,
    subagent_task,
    task_status,
    tool_arg,
    tool_block,
    transcript_blocks,
    user_prompts,
    user_transcript_text,
    was_auto_approved,
)
from luca.agent.core.models import (
    ApprovalStatus,
    AssistantMessage,
    ChildConversation,
    ExecutionAttempt,
    ExecutionAttemptOutcome,
    ExecutionResult,
    ExecutionStatus,
    ImageBase64,
    ImageContent,
    SessionConfig,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    UserMessage,
)
from tests.agent.scenarios import MODEL, conversation, make_session, spec

# ── literal factory ───────────────────────────────────────────────────────────

_ATTEMPT_OUTCOMES: dict[ExecutionStatus, ExecutionAttemptOutcome] = {
    ExecutionStatus.COMPLETED: ExecutionAttemptOutcome.COMPLETED,
    ExecutionStatus.FAILED: ExecutionAttemptOutcome.FAILED,
    ExecutionStatus.TIMED_OUT: ExecutionAttemptOutcome.TIMED_OUT,
    ExecutionStatus.INTERRUPTED: ExecutionAttemptOutcome.INTERRUPTED,
}


def execution(name: str, status: ExecutionStatus, arguments: dict | None = None, **over) -> ToolExecution:
    """A standalone `ToolExecution` literal; identity fields filled, any field
    overridable (`attempts`, `result`, `tool_spec`, …).

    `body=(started, ended)` is the shorthand for "one dispatched attempt",
    which is what the transcript's duration note reads — body wall-clock, not
    `duration_ms` (total time outstanding)."""
    fields: dict = {
        "id": "te1",
        "created_at": 500,
        "tool_call_id": "tc1",
        "raw_tool_call": ToolCall(id="tc1", name=name, arguments=arguments or {}),
        "tool_spec": spec(name),
        "status": status,
    }
    body = fields.pop("body", None) or over.pop("body", None)
    fields.update(over)
    if body is not None:
        started, ended = body
        fields["attempts"] = [
            ExecutionAttempt(
                outcome=_ATTEMPT_OUTCOMES.get(status),
                started_at=started,
                ended_at=ended,
            )
        ]
        fields.setdefault("finished_at", ended)
    return ToolExecution(**fields)


# ── tool_arg ──────────────────────────────────────────────────────────────────


def test_tool_arg_prefers_the_primary_argument():
    assert tool_arg(execution("edit", ExecutionStatus.PENDING, {"content": "x = 1", "path": "luca/cli.py"})) == (
        "luca/cli.py"
    )


def test_tool_arg_skips_a_non_string_primary():
    assert tool_arg(execution("read", ExecutionStatus.PENDING, {"path": 7, "query": "events"})) == "events"


def test_tool_arg_collapses_whitespace():
    assert tool_arg(execution("bash", ExecutionStatus.PENDING, {"command": "ls   -la\n  /tmp"})) == "ls -la /tmp"


def test_tool_arg_bounds_the_length():
    assert tool_arg(execution("bash", ExecutionStatus.PENDING, {"command": "a" * 100})) == "a" * 80 + "…"


def test_tool_arg_falls_back_to_compact_key_values():
    assert tool_arg(execution("add", ExecutionStatus.PENDING, {"a": 1, "flag": True, "s": "hi"})) == (
        "a=1, flag=true, s=hi"
    )


def test_tool_arg_empty_arguments():
    assert tool_arg(execution("status", ExecutionStatus.PENDING)) == ""


def test_tool_arg_joins_a_command_list():
    # the native shell's primary argument is a list, not a string
    assert tool_arg(execution("openai_shell", ExecutionStatus.PENDING, {"commands": ["ruff check", "pytest -q"]})) == (
        "ruff check; pytest -q"
    )


def test_tool_arg_falls_back_when_the_command_list_is_empty():
    assert tool_arg(execution("openai_shell", ExecutionStatus.PENDING, {"commands": []})) == "commands=[]"


# ── tool_block: the header name ───────────────────────────────────────────────


def test_the_header_shows_the_specs_title_when_it_has_one():
    # a provider-native tool is `openai_apply_patch` to the framework and
    # "Apply patch" to a person
    titled = execution(
        "openai_apply_patch",
        ExecutionStatus.PENDING,
        {"path": "app.py"},
        tool_spec=spec("openai_apply_patch", title="Apply patch"),
    )

    assert tool_block(titled) == vm.ToolBlock(tool="Apply patch", arg="app.py", status="pending")


def test_the_header_falls_back_to_the_raw_name_when_the_call_never_resolved():
    assert tool_block(execution("mystery", ExecutionStatus.NOT_FOUND, tool_spec=None)).tool == "mystery"


# ── tool_block: non-terminal and denied ───────────────────────────────────────


def test_nonterminal_statuses_render_the_header_only():
    assert [
        tool_block(execution("bash", ExecutionStatus.RECEIVED, {"command": "sleep 2"})),
        tool_block(execution("bash", ExecutionStatus.PENDING, {"command": "sleep 2"})),
        tool_block(execution("bash", ExecutionStatus.RUNNING, {"command": "sleep 2"})),
    ] == [vm.ToolBlock(tool="bash", arg="sleep 2", status="pending")] * 3


def test_rejected_by_rule_states_the_decision():
    assert tool_block(execution("edit", ExecutionStatus.REJECTED, {"path": "luca/cli.py"})) == vm.ToolBlock(
        tool="edit",
        arg="luca/cli.py",
        status="denied",
        result=vm.ToolResult(summary="denied", note="by rule"),
    )


def test_rejected_by_the_user_states_the_decision():
    block = tool_block(execution("edit", ExecutionStatus.REJECTED, {"path": "luca/cli.py"}), denied_by_user=True)
    assert block == vm.ToolBlock(
        tool="edit",
        arg="luca/cli.py",
        status="denied",
        result=vm.ToolResult(summary="denied", note="by you"),
    )


# ── tool_block: failure statuses ──────────────────────────────────────────────


def test_every_failure_status_names_itself():
    assert [
        tool_block(execution("bash", ExecutionStatus.FAILED, {"command": "make"})),
        tool_block(execution("bash", ExecutionStatus.NOT_FOUND, {"command": "make"})),
        tool_block(execution("bash", ExecutionStatus.INVALID, {"command": "make"})),
        tool_block(execution("bash", ExecutionStatus.REFUSED, {"command": "make"})),
        tool_block(execution("bash", ExecutionStatus.CANCELLED, {"command": "make"})),
        tool_block(execution("bash", ExecutionStatus.INTERRUPTED, {"command": "make"})),
        tool_block(execution("bash", ExecutionStatus.TIMED_OUT, {"command": "make"})),
    ] == [
        vm.ToolBlock(tool="bash", arg="make", status="error", output=vm.ToolOutput(lines=[], summary="failed")),
        vm.ToolBlock(tool="bash", arg="make", status="error", output=vm.ToolOutput(lines=[], summary="not found")),
        vm.ToolBlock(
            tool="bash", arg="make", status="error", output=vm.ToolOutput(lines=[], summary="invalid arguments")
        ),
        vm.ToolBlock(tool="bash", arg="make", status="error", output=vm.ToolOutput(lines=[], summary="refused")),
        vm.ToolBlock(tool="bash", arg="make", status="error", output=vm.ToolOutput(lines=[], summary="cancelled")),
        vm.ToolBlock(tool="bash", arg="make", status="error", output=vm.ToolOutput(lines=[], summary="interrupted")),
        vm.ToolBlock(tool="bash", arg="make", status="error", output=vm.ToolOutput(lines=[], summary="timed out")),
    ]


def test_failure_summary_carries_the_error_text_and_duration():
    block = tool_block(
        execution("bash", ExecutionStatus.FAILED, {"command": "make"}, body=(1000, 3000)),
        "make: *** [all] Error 2",
    )
    assert block == vm.ToolBlock(
        tool="bash",
        arg="make",
        status="error",
        output=vm.ToolOutput(lines=["make: *** [all] Error 2"], summary="failed · 2s"),
    )


def test_sub_second_duration_keeps_one_decimal():
    block = tool_block(execution("bash", ExecutionStatus.FAILED, {"command": "make"}, body=(1000, 1500)))
    assert block == vm.ToolBlock(
        tool="bash",
        arg="make",
        status="error",
        output=vm.ToolOutput(lines=[], summary="failed · 0.5s"),
    )


def test_negligible_duration_is_omitted():
    block = tool_block(execution("bash", ExecutionStatus.TIMED_OUT, {"command": "make"}, body=(1000, 1040)))
    assert block == vm.ToolBlock(
        tool="bash",
        arg="make",
        status="error",
        output=vm.ToolOutput(lines=[], summary="timed out"),
    )


# ── tool_block: completed ─────────────────────────────────────────────────────


def test_completed_single_line_becomes_the_result_summary():
    assert tool_block(execution("bash", ExecutionStatus.COMPLETED, {"command": "echo hi"}), "hi") == vm.ToolBlock(
        tool="bash",
        arg="echo hi",
        status="ok",
        result=vm.ToolResult(summary="hi"),
    )


def test_completed_single_line_is_collapsed_to_the_measure():
    block = tool_block(execution("bash", ExecutionStatus.COMPLETED, {"command": "cat id"}), "a" * 100)
    assert block == vm.ToolBlock(
        tool="bash",
        arg="cat id",
        status="ok",
        result=vm.ToolResult(summary="a" * 84 + "…"),
    )


def test_auto_approved_call_notes_the_rule_on_its_result_row():
    block = tool_block(execution("bash", ExecutionStatus.COMPLETED, {"command": "echo hi"}), "hi", auto_approved=True)
    assert block == vm.ToolBlock(
        tool="bash",
        arg="echo hi",
        status="ok",
        result=vm.ToolResult(summary="hi", note="approved by rule"),
    )


EDIT_DIFF = "--- a/luca/cli.py\n+++ b/luca/cli.py\n@@ -1,2 +1,3 @@\n unchanged\n-old\n+new\n+added\n"


def test_completed_edit_shows_a_diff_stat():
    block = tool_block(
        execution(
            "edit",
            ExecutionStatus.COMPLETED,
            {"path": "luca/cli.py"},
            result=ExecutionResult(content=[TextContent(text="ok")], metadata={"diff": EDIT_DIFF}),
        )
    )
    assert block == vm.ToolBlock(
        tool="edit",
        arg="luca/cli.py",
        status="ok",
        result=vm.ToolResult(stat=vm.DiffStat(add=2, remove=1)),
    )


def test_auto_approved_edit_carries_both_stat_and_note():
    block = tool_block(
        execution(
            "edit",
            ExecutionStatus.COMPLETED,
            {"path": "luca/cli.py"},
            result=ExecutionResult(content=[TextContent(text="ok")], metadata={"diff": EDIT_DIFF}),
        ),
        auto_approved=True,
    )
    assert block == vm.ToolBlock(
        tool="edit",
        arg="luca/cli.py",
        status="ok",
        result=vm.ToolResult(stat=vm.DiffStat(add=2, remove=1), note="approved by rule"),
    )


def test_completed_shell_output_shows_exit_and_duration():
    block = tool_block(
        execution(
            "bash",
            ExecutionStatus.COMPLETED,
            {"command": "ls"},
            result=ExecutionResult(content=[TextContent(text="a\nb")], metadata={"exit": 0}),
            body=(1000, 1900),
        ),
        "a\nb",
    )
    assert block == vm.ToolBlock(
        tool="bash",
        arg="ls",
        status="ok",
        output=vm.ToolOutput(lines=["a", "b"], summary="exit 0 · 0.9s"),
    )


def test_error_exit_is_marked_and_the_tail_hidden():
    block = tool_block(
        execution(
            "bash",
            ExecutionStatus.COMPLETED,
            {"command": "pytest -q"},
            result=ExecutionResult(content=[], is_error=True, metadata={"exit": 1}),
        ),
        "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\nl11",
        is_error=True,
    )
    assert block == vm.ToolBlock(
        tool="bash",
        arg="pytest -q",
        status="ok",
        output=vm.ToolOutput(
            lines=["l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8"],
            hidden_lines=["l9", "l10", "l11"],
            summary="[error]exit 1[/] · 3 lines hidden",
            expand_hint=True,
        ),
    )


def test_unknown_exit_code_renders_a_question_mark():
    block = tool_block(
        execution(
            "bash",
            ExecutionStatus.COMPLETED,
            {"command": "ls"},
            result=ExecutionResult(content=[], metadata={"exit": None}),
        ),
        "done",
    )
    assert block == vm.ToolBlock(
        tool="bash",
        arg="ls",
        status="ok",
        output=vm.ToolOutput(lines=["done"], summary="exit ?"),
    )


def test_read_result_compacts_to_a_line_count():
    block = tool_block(execution("read", ExecutionStatus.COMPLETED, {"path": "luca/cli.py"}), "l1\nl2\nl3")
    assert block == vm.ToolBlock(
        tool="read",
        arg="luca/cli.py",
        status="ok",
        result=vm.ToolResult(summary="3 lines"),
    )


def test_grep_result_compacts_to_matches_and_files():
    block = tool_block(
        execution("grep", ExecutionStatus.COMPLETED, {"pattern": "add_argument"}),
        "Found 6 matches\nluca/cli.py:\n12:foo\nluca/events.py:\n30:bar",
    )
    assert block == vm.ToolBlock(
        tool="grep",
        arg="add_argument",
        status="ok",
        result=vm.ToolResult(summary="6 matches · luca/cli.py, luca/events.py"),
    )


def test_grep_result_ellipsizes_past_three_files():
    block = tool_block(
        execution("grep", ExecutionStatus.COMPLETED, {"pattern": "x"}),
        "Found 9 matches\na.py:\nb.py:\nc.py:\nd.py:",
    )
    assert block == vm.ToolBlock(
        tool="grep",
        arg="x",
        status="ok",
        result=vm.ToolResult(summary="9 matches · a.py, b.py, c.py, …"),
    )


def test_glob_result_compacts_to_a_file_count():
    block = tool_block(execution("glob", ExecutionStatus.COMPLETED, {"pattern": "**/*.py"}), "a.py\nb.py")
    assert block == vm.ToolBlock(
        tool="glob",
        arg="**/*.py",
        status="ok",
        result=vm.ToolResult(summary="2 files"),
    )


def test_generic_multi_line_output_shows_the_head_and_hides_the_rest():
    block = tool_block(
        execution("bash", ExecutionStatus.COMPLETED, {"command": "cat f"}),
        "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10",
    )
    assert block == vm.ToolBlock(
        tool="bash",
        arg="cat f",
        status="ok",
        output=vm.ToolOutput(
            lines=["l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8"],
            hidden_lines=["l9", "l10"],
            summary="10 lines · 2 lines hidden",
            expand_hint=True,
        ),
    )


# ── diff_stat ─────────────────────────────────────────────────────────────────


def test_diff_stat_counts_signed_lines_not_headers():
    assert diff_stat(EDIT_DIFF) == vm.DiffStat(add=2, remove=1)


def test_diff_stat_omits_the_absent_side():
    assert diff_stat("+only\n") == vm.DiffStat(add=1, remove=None)


def test_diff_stat_is_none_without_changes():
    assert diff_stat("--- a/f\n+++ b/f\n context\n") is None


# ── the sticky plan panel ─────────────────────────────────────────────────────


def todo(todo_id: int, content: str, status: str) -> dict:
    return {"id": todo_id, "content": content, "status": status}


def todo_execution(entry_id: str, todos: list[dict], changed: list[int]) -> ToolExecution:
    """A completed `update_todos`, written the way the memory plugin writes
    one — the numbered list on `structured_content`, never on the arguments."""
    return execution(
        "update_todos",
        ExecutionStatus.COMPLETED,
        {"todos": [{"content": item["content"], "status": item["status"]} for item in todos]},
        id=entry_id,
        tool_call_id=f"tc_{entry_id}",
        tool_spec=spec("update_todos", namespace="contrib.memory"),
        result=ExecutionResult(
            content=[TextContent(text="Todo list updated successfully")],
            structured_content={"todos": todos, "changed": changed},
        ),
    )


def test_plan_block_numbers_the_rows_and_strikes_the_settled_ones():
    # Four items fit the window, so they stay in their own order — nothing is
    # at risk of being dropped, and the numbers are what the user tracks.
    assert plan_block(
        [
            todo(1, "read the file", "completed"),
            todo(2, "edit it", "in_progress"),
            todo(3, "run tests", "pending"),
            todo(4, "drop the spike", "cancelled"),
        ]
    ) == vm.ListBlock(
        label="4 tasks (2 done, 2 open)",
        rows=[
            vm.ListRow(glyph="done", text="[faint]#1 -[/] read the file", strike=True),
            vm.ListRow(glyph="active", text="[faint]#2 -[/] edit it"),
            vm.ListRow(glyph="pending", text="[faint]#3 -[/] run tests"),
            vm.ListRow(glyph="done", text="[faint]#4 -[/] drop the spike", strike=True),
        ],
    )


def test_a_list_that_overflows_the_window_leads_with_what_is_open():
    assert [
        row.text
        for row in plan_block([todo(n, f"step {n}", "completed" if n < 3 else "pending") for n in range(1, 7)]).rows
    ] == [
        "[faint]#3 -[/] step 3",
        "[faint]#4 -[/] step 4",
        "[faint]#5 -[/] step 5",
        "[faint]#6 -[/] step 6",
        "[faint]#1 -[/] step 1",
        "[faint]… +1 completed[/]",
    ]


def test_plan_block_omits_a_zero_done_count():
    assert plan_block([todo(1, "a", "pending")]) == vm.ListBlock(
        label="1 task (1 open)",
        rows=[vm.ListRow(glyph="pending", text="[faint]#1 -[/] a")],
    )


def test_plan_block_of_a_finished_list_says_done():
    assert plan_block([todo(1, "a", "completed"), todo(2, "b", "cancelled")]) == vm.ListBlock(
        label="Done (2 tasks completed)",
        rows=[
            vm.ListRow(glyph="done", text="[faint]#1 -[/] a", strike=True),
            vm.ListRow(glyph="done", text="[faint]#2 -[/] b", strike=True),
        ],
    )


def test_plan_block_of_an_empty_list_is_none():
    assert plan_block([]) is None


def test_plan_block_windows_to_five_rows_and_names_a_uniform_remainder():
    assert plan_block([todo(n, f"step {n}", "completed" if n <= 7 else "pending") for n in range(1, 13)]) == (
        vm.ListBlock(
            label="12 tasks (7 done, 5 open)",
            rows=[
                vm.ListRow(glyph="pending", text="[faint]#8 -[/] step 8"),
                vm.ListRow(glyph="pending", text="[faint]#9 -[/] step 9"),
                vm.ListRow(glyph="pending", text="[faint]#10 -[/] step 10"),
                vm.ListRow(glyph="pending", text="[faint]#11 -[/] step 11"),
                vm.ListRow(glyph="pending", text="[faint]#12 -[/] step 12"),
                vm.ListRow(glyph="none", text="[faint]… +7 completed[/]"),
            ],
        )
    )


def test_plan_block_calls_a_mixed_remainder_more():
    # Six open and two done: the window takes five of the open ones, so what
    # is left over is one open item AND two settled ones. Calling that
    # "completed" would claim work that is still outstanding.
    block = plan_block([todo(n, f"step {n}", "completed" if n > 6 else "pending") for n in range(1, 9)])
    assert block.rows[-1] == vm.ListRow(glyph="none", text="[faint]… +3 more[/]")


def test_running_plan_block_leads_with_what_the_last_write_moved():
    todos = [todo(n, f"step {n}", "completed" if n <= 3 else "pending") for n in range(1, 7)]
    assert plan_block(todos, changed=[2, 3], running=True) == vm.ListBlock(
        label="6 tasks (3 done, 3 open)",
        rows=[
            vm.ListRow(glyph="done", text="[faint]#2 -[/] step 2", strike=True),
            vm.ListRow(glyph="done", text="[faint]#3 -[/] step 3", strike=True),
            vm.ListRow(glyph="done", text="[faint]#1 -[/] step 1", strike=True),
            vm.ListRow(glyph="pending", text="[faint]#4 -[/] step 4"),
            vm.ListRow(glyph="pending", text="[faint]#5 -[/] step 5"),
            vm.ListRow(glyph="none", text="[faint]… +1 pending[/]"),
        ],
    )


def test_plan_from_session_reads_the_list_off_the_session():
    # The panel is a LOOKUP, not a reconstruction: the app hands the memory
    # plugin a dict that lives under `extras`, so a stored session carries the
    # list and the entries only say when it was written.
    session = make_session(
        id="s_plan",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="plan it")]),
            "te1": todo_execution("te1", [todo(1, "a", "pending"), todo(2, "b", "pending")], [1, 2]),
            "te2": todo_execution("te2", [todo(1, "a", "completed"), todo(2, "b", "pending")], [1]),
        },
        conversations={"c1": conversation("c1", ["u1", "te1", "te2"])},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
        extras={"todos": {"c1": {"todos": [todo(1, "a", "completed"), todo(2, "b", "pending")], "next_id": 3}}},
    )

    assert plan_from_session(session) == vm.ListBlock(
        label="2 tasks (1 done, 1 open)",
        rows=[
            vm.ListRow(glyph="done", text="[faint]#1 -[/] a", strike=True),
            vm.ListRow(glyph="pending", text="[faint]#2 -[/] b"),
        ],
    )


def test_plan_from_session_is_none_once_a_settled_list_was_dismissed():
    # The list finished, then the user spoke past it. The agent still holds
    # the list; the dock is what gives up the rows, and a resumed session has
    # to give them up at exactly the same point the live one did.
    session = make_session(
        id="s_plan_done",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="plan it")]),
            "te1": todo_execution("te1", [todo(1, "a", "completed")], [1]),
            "u2": UserMessage(id="u2", created_at=600, parts=[TextContent(text="thanks")]),
        },
        conversations={"c1": conversation("c1", ["u1", "te1", "u2"])},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
        extras={"todos": {"c1": {"todos": [todo(1, "a", "completed")], "next_id": 2}}},
    )

    assert plan_from_session(session) is None


def test_a_settled_list_is_not_dismissed_until_the_user_speaks():
    session = make_session(
        id="s_plan_settled",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="plan it")]),
            "te1": todo_execution("te1", [todo(1, "a", "completed")], [1]),
        },
        conversations={"c1": conversation("c1", ["u1", "te1"])},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
        extras={"todos": {"c1": {"todos": [todo(1, "a", "completed")], "next_id": 2}}},
    )

    assert plan_dismissed(session) is False


def test_a_dismissed_list_comes_back_when_the_agent_writes_again():
    # Dismissal is not a state anything stores — it is a question about the
    # LAST write, so reopening an item answers it differently on its own.
    session = make_session(
        id="s_plan_reopened",
        entries={
            "te1": todo_execution("te1", [todo(1, "a", "completed")], [1]),
            "u1": UserMessage(id="u1", created_at=600, parts=[TextContent(text="actually, redo it")]),
            "te2": todo_execution("te2", [todo(1, "a", "in_progress")], [1]),
        },
        conversations={"c1": conversation("c1", ["te1", "u1", "te2"])},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
        extras={"todos": {"c1": {"todos": [todo(1, "a", "in_progress")], "next_id": 2}}},
    )

    assert plan_dismissed(session) is False


# ── runtime plumbing ──────────────────────────────────────────────────────────

SPAWN_SCHEMA = {"type": "object", "properties": {"is_subagent_spawn": {"type": "boolean"}}}


def test_private_and_spawn_declaring_tools_are_plumbing():
    private = execution("secret", ExecutionStatus.PENDING, tool_spec=spec("secret", is_private=True))
    spawner = execution("task", ExecutionStatus.PENDING, tool_spec=spec("task", output_schema=SPAWN_SCHEMA))
    assert is_runtime_plumbing(private) is True
    assert is_runtime_plumbing(spawner) is True


def test_ordinary_and_unresolved_tools_are_not_plumbing():
    assert is_runtime_plumbing(execution("bash", ExecutionStatus.PENDING)) is False
    assert is_runtime_plumbing(execution("ghost", ExecutionStatus.NOT_FOUND, tool_spec=None)) is False


# ── subagents ─────────────────────────────────────────────────────────────────

SPAWN_ARGUMENTS = {"description": "arg description", "prompt": "arg prompt"}

SUBAGENT_SESSION = make_session(
    id="s_sub",
    entries={
        "te_spawn": ToolExecution(
            id="te_spawn",
            conversation_id="c1",
            created_at=500,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="task", arguments=SPAWN_ARGUMENTS),
            tool_spec=spec("task", output_schema=SPAWN_SCHEMA),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(
                content=[TextContent(text="task started")],
                structured_content={
                    "is_subagent_spawn": True,
                    "task_id": "t1",
                    "description": "audit the docs",
                    "prompt": "Read every doc page",
                    "process_subagent_result_tool_name": "task_result",
                },
            ),
        ),
        "cc1": ChildConversation(id="cc1", created_at=500, conversation_id="c2", tool_execution_id="te_spawn"),
        "cc2": ChildConversation(id="cc2", created_at=500, conversation_id="c3", tool_execution_id="te_missing"),
    },
    conversations={
        "c1": conversation("c1", ["te_spawn", "cc1"]),
        "c2": conversation("c2", ["cc2"], depth=1),
        "c3": conversation("c3", [], depth=2),
    },
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)

# The same spawn before its result landed — only the call arguments describe it.
UNRESOLVED_SPAWN_SESSION = make_session(
    id="s_sub_pending",
    entries={
        "te_spawn": ToolExecution(
            id="te_spawn",
            conversation_id="c1",
            created_at=500,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="task", arguments=SPAWN_ARGUMENTS),
            tool_spec=spec("task", output_schema=SPAWN_SCHEMA),
            status=ExecutionStatus.RUNNING,
        ),
        "cc1": ChildConversation(id="cc1", created_at=500, conversation_id="c2", tool_execution_id="te_spawn"),
    },
    conversations={
        "c1": conversation("c1", ["te_spawn", "cc1"]),
        "c2": conversation("c2", [], depth=1),
    },
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)


def test_subagent_task_prefers_the_structured_result():
    entry = SUBAGENT_SESSION.entries["cc1"]
    assert subagent_task(SUBAGENT_SESSION, entry) == ("audit the docs", "Read every doc page")


def test_subagent_task_falls_back_to_the_call_arguments():
    entry = UNRESOLVED_SPAWN_SESSION.entries["cc1"]
    assert subagent_task(UNRESOLVED_SPAWN_SESSION, entry) == ("arg description", "arg prompt")


def test_subagent_task_without_a_spawn_execution_names_the_conversation():
    entry = SUBAGENT_SESSION.entries["cc2"]
    assert subagent_task(SUBAGENT_SESSION, entry) == ("subagent c3", "")


def test_child_links_yields_every_link_with_its_parent():
    assert list(child_links(SUBAGENT_SESSION)) == [
        ("c1", SUBAGENT_SESSION.entries["cc1"]),
        ("c2", SUBAGENT_SESSION.entries["cc2"]),
    ]


# ── session previews ──────────────────────────────────────────────────────────

PREVIEW_SESSION = make_session(
    id="s_preview",
    entries={
        "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="fix the flaky test\nplease")]),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
        "a1": AssistantMessage(
            id="a1",
            parent_id="ts",
            created_at=500,
            parts=[TextContent(text="Looking now.\nSecond line")],
            llm_config=MODEL,
            stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1",
            conversation_id="c1",
            parent_id="a1",
            created_at=500,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="bash", arguments={"command": "ls"}),
            tool_spec=spec("bash"),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="ok")]),
        ),
        "te2": todo_execution(
            "te2",
            [todo(1, "read the file", "completed"), todo(2, "run tests", "in_progress")],
            [1, 2],
        ),
    },
    conversations={"c1": conversation("c1", ["u1", "ts", "a1", "te1", "te2"])},
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)


def test_preview_rows_show_the_last_transcript_rows_in_order():
    assert preview_rows(PREVIEW_SESSION) == [
        "Looking now.",
        "▸ bash   ls",
        "[accent]◉[/] run tests",
    ]


def test_preview_rows_include_the_user_line_when_asked_for_more():
    assert preview_rows(PREVIEW_SESSION, count=5) == [
        "[accent]›[/] fix the flaky test",
        "Looking now.",
        "▸ bash   ls",
        "[accent]◉[/] run tests",
    ]


# ── user content ──────────────────────────────────────────────────────────────


def test_user_transcript_text_renders_images_as_placeholders():
    assert (
        user_transcript_text(
            [
                ImageContent(
                    source=ImageBase64(data="aGk=", media_type="image/png"),
                    metadata={"name": "receipt.jpg"},
                ),
                TextContent(text="how much did I tip?"),
            ]
        )
        == "[image: receipt.jpg]\nhow much did I tip?"
    )


def test_user_transcript_text_falls_back_to_the_media_type():
    assert (
        user_transcript_text(
            [
                ImageContent(source=ImageBase64(data="aGk=", media_type="image/png")),
            ]
        )
        == "[image: image/png]"
    )


def test_user_prompts_are_what_was_typed_oldest_first():
    # The composer's history. The mention part is the harness's expansion of
    # the `@` in the text, the image is not something to retype, and the
    # repeat of "again" collapses the way a shell's history does.
    session = make_session(
        id="s_history",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="first")]),
            "a1": AssistantMessage(
                id="a1",
                created_at=510,
                parts=[TextContent(text="on it")],
                llm_config=MODEL,
                stop_reason="stop",
            ),
            "u2": UserMessage(
                id="u2",
                created_at=520,
                parts=[
                    TextContent(text="read @notes.md"),
                    TextContent(text="# notes", metadata={"mention": {"path": "/w/notes.md", "success": True}}),
                    ImageContent(source=ImageBase64(data="aGk=", media_type="image/png")),
                ],
            ),
            "u3": UserMessage(id="u3", created_at=530, parts=[TextContent(text="again")]),
            "u4": UserMessage(id="u4", created_at=540, parts=[TextContent(text="again")]),
        },
        conversations={"c1": conversation("c1", ["u1", "a1", "u2", "u3", "u4"])},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )

    assert user_prompts(session) == ["first", "read @notes.md", "again"]


def test_user_prompts_walk_back_past_a_compaction():
    # A compaction archives one conversation and installs another over a
    # rewritten path: "first" is no longer ON the main path, but the entry is
    # still in the store and the predecessor still lists it. The tail the
    # successor kept ("second") is listed by both and must appear once.
    session = make_session(
        id="s_history_compacted",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="first")]),
            "u2": UserMessage(id="u2", created_at=510, parts=[TextContent(text="second")]),
            "u3": UserMessage(id="u3", created_at=520, parts=[TextContent(text="third")]),
        },
        conversations={
            "c1": conversation("c1", ["u1", "u2"]),
            "c2": conversation("c2", ["u2", "u3"], previous_conversation_id="c1"),
        },
        main_conversation_id="c2",
        session_config=SessionConfig(llm_config=MODEL),
    )

    assert user_prompts(session) == ["first", "second", "third"]


# ── entry_blocks / transcript_blocks ──────────────────────────────────────────

TODOS_FIRST = [
    {"content": "read the schema", "status": "in_progress"},
    {"content": "write the migration", "status": "pending"},
]
TODOS_SECOND = [
    {"content": "read the schema", "status": "completed"},
    {"content": "write the migration", "status": "in_progress"},
]


def test_entry_blocks_renders_a_user_message_with_its_mention_rows():
    entry = UserMessage(
        id="u1",
        created_at=500,
        parts=[
            TextContent(text="explain @luca/cli.py"),
            TextContent(
                text="…file contents…",
                metadata={"mention": {"path": "/repo/luca/cli.py", "success": True, "lines": 84}},
            ),
        ],
    )
    assert entry_blocks(entry) == [
        vm.UserBlock(text="explain @luca/cli.py"),
        vm.ToolBlock(
            tool="read",
            arg="/repo/luca/cli.py",
            status="ok",
            result=vm.ToolResult(summary="84 lines"),
        ),
    ]


def test_entry_blocks_drops_a_subagents_seed_message():
    entry = UserMessage(id="u1", created_at=500, parts=[TextContent(text="audit the docs")])
    assert entry_blocks(entry, subagent=True) == []


def test_entry_blocks_renders_thinking_and_non_empty_text_parts():
    entry = AssistantMessage(
        id="a1",
        created_at=500,
        parts=[ThinkingContent(thinking="hmm"), TextContent(text="done"), TextContent(text="   ")],
        llm_config=MODEL,
        stop_reason="end_turn",
    )
    assert entry_blocks(entry) == [vm.ThinkingBlock(), vm.TextBlock(text="done")]


def test_entry_blocks_reports_a_cancelled_turn():
    entry = TurnFinish(id="tf1", created_at=500, outcome=TurnOutcome.CANCELLED)
    assert entry_blocks(entry) == [vm.NoticeBlock(text="turn cancelled")]


def test_entry_blocks_leaves_a_tool_row_resultless_without_a_resolver():
    """No resolver means no model-facing text to summarize — the row still
    renders, it just has nothing to say."""
    entry = execution("read", ExecutionStatus.COMPLETED, {"path": "luca/cli.py"})
    assert entry_blocks(entry) == [
        vm.ToolBlock(tool="read", arg="luca/cli.py", status="ok", result=vm.ToolResult(summary=""))
    ]


def test_entry_blocks_takes_the_result_text_from_the_resolver():
    entry = execution("read", ExecutionStatus.COMPLETED, {"path": "luca/cli.py"})
    assert entry_blocks(entry, resolve_result=lambda _e: ("one\ntwo\nthree", False)) == [
        vm.ToolBlock(
            tool="read",
            arg="luca/cli.py",
            status="ok",
            result=vm.ToolResult(summary="3 lines"),
        )
    ]


def test_was_auto_approved_needs_a_gate_and_an_approval_context():
    gated = execution(
        "bash",
        ExecutionStatus.COMPLETED,
        approval_status=ApprovalStatus.ALLOWED,
        extras={"approval_context": {"resources": []}},
    )
    ungated = execution("bash", ExecutionStatus.COMPLETED)
    assert (was_auto_approved(gated), was_auto_approved(ungated)) == (True, False)


def test_task_status_reads_the_link_not_the_tool():
    pending = ChildConversation(id="cc1", created_at=500, conversation_id="c2", tool_execution_id="te1")
    failed = ChildConversation(
        id="cc2",
        created_at=500,
        conversation_id="c3",
        tool_execution_id="te1",
        execution_result=ExecutionResult(content=[TextContent(text="boom")], is_error=True),
    )
    assert (task_status(pending), task_status(failed)) == ("waiting", "failed")


PLAN_SESSION = make_session(
    id="s_plan_transcript",
    entries={
        "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="migrate the store")]),
        "te1": todo_execution("te1", [todo(1, "read the schema", "completed")], [1]),
        "te2": execution(
            "read_todo",
            ExecutionStatus.COMPLETED,
            id="te2",
            tool_call_id="tc2",
            tool_spec=spec("read_todo", namespace="contrib.memory"),
            result=ExecutionResult(content=[TextContent(text="[]")]),
        ),
    },
    conversations={"c1": conversation("c1", ["u1", "te1", "te2"])},
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)


def test_transcript_blocks_omit_both_halves_of_the_todo_pair():
    # The list is the docked panel, not a transcript block — and hiding only
    # the write is what made a preceding `read_todo` look like it did the work.
    assert transcript_blocks(PLAN_SESSION) == [vm.UserBlock(text="migrate the store")]


def test_transcript_blocks_keep_a_same_named_tool_from_another_namespace():
    session = make_session(
        id="s_plan_foreign",
        entries={
            "te1": execution(
                "update_todos",
                ExecutionStatus.COMPLETED,
                {"todos": []},
                id="te1",
                result=ExecutionResult(content=[TextContent(text="ok")]),
            )
        },
        conversations={"c1": conversation("c1", ["te1"])},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )

    assert transcript_blocks(session) == [
        vm.ToolBlock(tool="update_todos", arg="todos=[]", status="ok", result=vm.ToolResult(summary=""))
    ]


NESTED_SESSION = make_session(
    id="s_nested",
    entries={
        "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="audit the docs")]),
        "te_spawn": ToolExecution(
            id="te_spawn",
            conversation_id="c1",
            created_at=500,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="task", arguments=SPAWN_ARGUMENTS),
            tool_spec=spec("task", output_schema=SPAWN_SCHEMA),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(
                content=[TextContent(text="task started")],
                structured_content={
                    "is_subagent_spawn": True,
                    "task_id": "t1",
                    "description": "audit the docs",
                    "prompt": "Read every doc page",
                    "process_subagent_result_tool_name": "task_result",
                },
            ),
        ),
        "cc1": ChildConversation(
            id="cc1",
            created_at=500,
            conversation_id="c2",
            tool_execution_id="te_spawn",
            execution_result=ExecutionResult(content=[TextContent(text="12 pages")]),
        ),
        "u2": UserMessage(id="u2", created_at=500, parts=[TextContent(text="Read every doc page")]),
        "a1": AssistantMessage(
            id="a1",
            created_at=500,
            parts=[TextContent(text="all pages read")],
            llm_config=MODEL,
            stop_reason="end_turn",
        ),
    },
    conversations={
        "c1": conversation("c1", ["u1", "te_spawn", "cc1"]),
        "c2": conversation("c2", ["u2", "a1"], depth=1),
    },
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)


def test_transcript_blocks_nest_a_subagent_into_its_task_block():
    """The spawn renders as the task, not as a tool row, and the child's seed
    message is dropped — it already reads on the task label."""
    assert transcript_blocks(NESTED_SESSION) == [
        vm.UserBlock(text="audit the docs"),
        vm.TaskBlock(
            description="audit the docs",
            status="done",
            blocks=[vm.TextBlock(text="all pages read")],
        ),
    ]


# ── the agent's questions (`ask_user`) ────────────────────────────────────────

ASK_SPEC = spec("ask_user", title="Ask the user", namespace=QUESTIONS_NAMESPACE)


def ask_user(payload: dict | None) -> ToolExecution:
    """A finished `ask_user` call carrying the tool's OWN record — the
    questions it asked and the payload it was answered with. `None` is the call
    whose result never landed."""
    return execution(
        "ask_user",
        ExecutionStatus.COMPLETED,
        tool_spec=ASK_SPEC,
        result=(
            ExecutionResult(content=[TextContent(text="answers")], structured_content=payload)
            if payload is not None
            else None
        ),
    )


def test_the_ask_user_call_is_matched_by_namespace_and_name():
    # An application's own `ask_user` is not this package's, so the dock never
    # opens for a tool the questions plugin did not install.
    ours = execution("ask_user", ExecutionStatus.AWAITING_RESULT, tool_spec=ASK_SPEC)
    theirs = execution("ask_user", ExecutionStatus.AWAITING_RESULT, tool_spec=spec("ask_user", namespace="acme.forms"))
    unresolved = execution("ask_user", ExecutionStatus.NOT_FOUND, tool_spec=None)

    assert [is_questions_tool(ours), is_questions_tool(theirs), is_questions_tool(unresolved)] == [True, False, False]


def test_a_tab_label_takes_whole_words_up_to_the_cell_limit():
    assert question_tab("How should existing `.jsonl` sessions be backfilled?", 2) == "How should"


def test_a_first_word_longer_than_the_limit_is_cut_to_it():
    assert question_tab("internationalization first?", 0) == "internationa"


def test_a_title_with_no_usable_words_falls_back_to_its_position():
    assert question_tab("??? !!!", 3) == "question 4"


def test_the_body_measure_is_the_panels_inner_width_at_the_design_geometry():
    # 105 design columns − 2×2 margin − 2 border − 2 padding
    assert QUESTION_MEASURE == 97


def test_a_single_select_question_pre_wraps_its_body_and_ends_with_the_custom_and_chat_rows():
    built = question_set_state(
        [
            Question(
                title="How should existing sessions be backfilled?",
                body=(
                    "~4,100 stored sessions across nine projects, 380MB on disk. Conversion runs at ~40ms per "
                    "session, single-threaded, and is idempotent.\n\n"
                    "Lazily is invisible, but two readers then live in the tree forever."
                ),
                options_type=OptionsType.SINGLE_SELECT,
                options=["On first launch, in-process", "Lazily, when a session resumes"],
            )
        ]
    )

    assert built == vm.QuestionSetState(
        questions=[
            vm.Question(
                tab="How should",
                title="How should existing sessions be backfilled?",
                body=[
                    # the first row is exactly the 97-column measure
                    "~4,100 stored sessions across nine projects, 380MB on disk. Conversion runs at ~40ms per session,",
                    "single-threaded, and is idempotent.",
                    "",
                    "Lazily is invisible, but two readers then live in the tree forever.",
                ],
                mode="single",
                options=[
                    vm.QuestionOption(label="On first launch, in-process"),
                    vm.QuestionOption(label="Lazily, when a session resumes"),
                    vm.QuestionOption(label="Custom answer:", kind="custom"),
                    vm.QuestionOption(label="Chat about this", kind="chat", key_hint="enter"),
                ],
            )
        ]
    )


def test_a_multiple_select_question_becomes_a_multi_mode_tab_with_no_body_rows():
    built = question_set_state(
        [
            Question(
                title="Read sqlite first where?",
                options_type=OptionsType.MULTIPLE_SELECT,
                options=["the sessions screen", "the cost screen"],
            )
        ]
    )

    assert built == vm.QuestionSetState(
        questions=[
            vm.Question(
                tab="Read sqlite",
                title="Read sqlite first where?",
                body=[],
                mode="multi",
                options=[
                    vm.QuestionOption(label="the sessions screen"),
                    vm.QuestionOption(label="the cost screen"),
                    vm.QuestionOption(label="Custom answer:", kind="custom"),
                    vm.QuestionOption(label="Chat about this", kind="chat", key_hint="enter"),
                ],
            )
        ]
    )


# One question, four rows, in the shape `question_set_state` builds.
def dock_question(**over) -> vm.Question:
    fields: dict = {
        "tab": "storage",
        "title": "Where should the sqlite file live?",
        "options": [
            vm.QuestionOption(label="~/.luca/projects/<project>/events.db"),
            vm.QuestionOption(label="Beside the jsonl files"),
            vm.QuestionOption(label="Custom answer:", kind="custom"),
            vm.QuestionOption(label="Chat about this", kind="chat", key_hint="enter"),
        ],
    }
    fields.update(over)
    return vm.Question(**fields)


def test_answer_payload_reports_the_option_a_single_select_question_settled_on():
    # `answer` is the PICK and `selected` is the caret: the payload reads the
    # pick, so arrowing over an answered question to re-read it cannot rewrite
    # what the model is told.
    state = vm.QuestionSetState(
        questions=[dock_question(selected=0, answer=1, answered=True)],
        extra="in-process is fine if it's scoped",
    )

    assert answer_payload(state) == {
        "answers": [
            {
                "question": "Where should the sqlite file live?",
                "chat_about_this": False,
                "answers": ["Beside the jsonl files"],
                "custom_answer": None,
            }
        ],
        "custom_notes": "in-process is fine if it's scoped",
    }


def test_answer_payload_reports_nothing_picked_while_the_question_is_unanswered():
    # The caret sits on a row, but a radio question has not chosen it until
    # `answered` says so — and an empty `extra` is no note at all.
    state = vm.QuestionSetState(questions=[dock_question(selected=1)], extra="")

    assert answer_payload(state) == {
        "answers": [
            {
                "question": "Where should the sqlite file live?",
                "chat_about_this": False,
                "answers": [],
                "custom_answer": None,
            }
        ],
        "custom_notes": None,
    }


def test_answer_payload_submits_every_tick_and_the_custom_text_together():
    state = vm.QuestionSetState(
        questions=[
            dock_question(
                mode="multi",
                answered=True,
                options=[
                    vm.QuestionOption(label="~/.luca/projects/<project>/events.db", checked=True),
                    vm.QuestionOption(label="Beside the jsonl files"),
                    vm.QuestionOption(label="Custom answer:", kind="custom", checked=True, text="one db per workspace"),
                    vm.QuestionOption(label="Chat about this", kind="chat", key_hint="enter"),
                ],
            )
        ]
    )

    assert answer_payload(state) == {
        "answers": [
            {
                "question": "Where should the sqlite file live?",
                "chat_about_this": False,
                "answers": ["~/.luca/projects/<project>/events.db"],
                "custom_answer": "one db per workspace",
            }
        ],
        "custom_notes": None,
    }


def test_answer_payload_flags_the_question_the_user_wants_to_chat_about():
    state = vm.QuestionSetState(
        questions=[
            dock_question(
                options=[
                    vm.QuestionOption(label="~/.luca/projects/<project>/events.db"),
                    vm.QuestionOption(label="Beside the jsonl files"),
                    vm.QuestionOption(label="Custom answer:", kind="custom"),
                    vm.QuestionOption(label="Chat about this", kind="chat", key_hint="enter", checked=True),
                ]
            )
        ]
    )

    assert answer_payload(state) == {
        "answers": [
            {
                "question": "Where should the sqlite file live?",
                "chat_about_this": True,
                "answers": [],
                "custom_answer": None,
            }
        ],
        "custom_notes": None,
    }


ANSWERED_PAYLOAD = {
    "questions": [
        {
            "title": "Where should the sqlite file live?",
            "options_type": "single_select",
            "options": ["~/.luca/projects/<project>/events.db", "Beside the jsonl files"],
        },
        {
            "title": "Read sqlite first where?",
            "options_type": "multiple_select",
            "options": ["the sessions screen", "the cost screen"],
        },
    ],
    "answer": {
        "answers": [
            {
                "question": "Where should the sqlite file live?",
                "chat_about_this": False,
                "answers": ["~/.luca/projects/<project>/events.db"],
                "custom_answer": None,
            },
            {
                "question": "Read sqlite first where?",
                "chat_about_this": False,
                "answers": ["the sessions screen"],
                "custom_answer": "the replay screen",
            },
        ],
        "custom_notes": "in-process is fine if it's scoped",
    },
}


def test_a_finished_question_set_collapses_into_a_header_and_one_row_per_answer():
    assert question_blocks(ask_user(ANSWERED_PAYLOAD)) == [
        vm.ToolBlock(tool="Ask the user", arg="2 questions", status="ok", note_right="2 answered"),
        vm.ListBlock(
            rows=[
                vm.ListRow(glyph="done", text="Where should", annotation="→  ~/.luca/projects/<project>/events.db"),
                vm.ListRow(
                    glyph="done",
                    text="Read sqlite",
                    annotation="→  the sessions screen, custom · the replay screen",
                ),
                vm.ListRow(glyph="none", text="", annotation="added · in-process is fine if it's scoped"),
            ],
            column=28,
        ),
    ]


def test_a_declined_question_names_itself_and_the_header_says_so():
    payload = {
        "questions": ANSWERED_PAYLOAD["questions"],
        "answer": {
            "answers": [
                ANSWERED_PAYLOAD["answer"]["answers"][0],
                {
                    "question": "Read sqlite first where?",
                    "chat_about_this": True,
                    "answers": [],
                    "custom_answer": None,
                },
            ],
        },
    }

    assert question_blocks(ask_user(payload)) == [
        vm.ToolBlock(tool="Ask the user", arg="2 questions", status="ok", note_right="1 answered · chat about this"),
        vm.ListBlock(
            rows=[
                vm.ListRow(glyph="done", text="Where should", annotation="→  ~/.luca/projects/<project>/events.db"),
                vm.ListRow(glyph="pending", text="Read sqlite", annotation="→  [accent]chat about this[/]"),
            ],
            column=28,
        ),
    ]


def test_a_question_the_payload_never_answered_reads_as_no_answer():
    payload = {
        "questions": ANSWERED_PAYLOAD["questions"],
        "answer": {"answers": [ANSWERED_PAYLOAD["answer"]["answers"][0]]},
    }

    assert question_blocks(ask_user(payload)) == [
        vm.ToolBlock(tool="Ask the user", arg="2 questions", status="ok", note_right="1 answered"),
        vm.ListBlock(
            rows=[
                vm.ListRow(glyph="done", text="Where should", annotation="→  ~/.luca/projects/<project>/events.db"),
                vm.ListRow(glyph="pending", text="Read sqlite", annotation="→  no answer"),
            ],
            column=28,
        ),
    ]


def test_a_call_whose_result_never_landed_renders_an_empty_set():
    assert question_blocks(ask_user(None)) == [
        vm.ToolBlock(tool="Ask the user", arg="0 questions", status="ok", note_right="0 answered"),
        vm.ListBlock(rows=[], column=28),
    ]


def test_a_malformed_payload_renders_rather_than_raising():
    # A hand-edited store, or one written by an older version: the collapsed
    # block is a transcript row, and no transcript row is worth a crash.
    assert question_blocks(ask_user({"questions": "not a list", "answer": "not a dict"})) == [
        vm.ToolBlock(tool="Ask the user", arg="0 questions", status="ok", note_right="0 answered"),
        vm.ListBlock(rows=[], column=28),
    ]
