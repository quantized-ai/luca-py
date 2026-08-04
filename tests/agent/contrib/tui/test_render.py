"""Pure derivations in `render.py`: entries → transcript view-models.

Each test: a known execution/session literal, one call, a full-object assert
on the resulting view-model. `execution()` is the module-level literal factory
(the `spec()` idiom): identity fields are never what a test is about.
"""

from luca.agent.contrib.tui import state as vm
from luca.agent.contrib.tui.render import (
    child_links,
    diff_stat,
    is_plan_update,
    is_runtime_plumbing,
    plan_block,
    plan_from_execution,
    preview_rows,
    subagent_task,
    tool_arg,
    tool_block,
    user_transcript_text,
)
from luca.agent.core.models import (
    AssistantMessage,
    ChildConversation,
    ExecutionResult,
    ExecutionStatus,
    ImageBase64,
    ImageContent,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    TurnStart,
    UserMessage,
)
from tests.agent.scenarios import MODEL, conversation, make_session, spec

# ── literal factory ───────────────────────────────────────────────────────────


def execution(name: str, status: ExecutionStatus, arguments: dict | None = None, **over) -> ToolExecution:
    """A standalone `ToolExecution` literal; identity fields filled, any field
    overridable (`started_at`, `result`, `tool_spec`, …)."""
    fields: dict = {
        "id": "te1",
        "created_at": 500,
        "tool_call_id": "tc1",
        "raw_tool_call": ToolCall(id="tc1", name=name, arguments=arguments or {}),
        "tool_spec": spec(name),
        "status": status,
    }
    fields.update(over)
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
        execution("bash", ExecutionStatus.FAILED, {"command": "make"}, started_at=1000, ended_at=3000),
        "make: *** [all] Error 2",
    )
    assert block == vm.ToolBlock(
        tool="bash",
        arg="make",
        status="error",
        output=vm.ToolOutput(lines=["make: *** [all] Error 2"], summary="failed · 2s"),
    )


def test_sub_second_duration_keeps_one_decimal():
    block = tool_block(execution("bash", ExecutionStatus.FAILED, {"command": "make"}, started_at=1000, ended_at=1500))
    assert block == vm.ToolBlock(
        tool="bash",
        arg="make",
        status="error",
        output=vm.ToolOutput(lines=[], summary="failed · 0.5s"),
    )


def test_negligible_duration_is_omitted():
    block = tool_block(
        execution("bash", ExecutionStatus.TIMED_OUT, {"command": "make"}, started_at=1000, ended_at=1040)
    )
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
            started_at=1000,
            ended_at=1900,
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


# ── plan blocks ───────────────────────────────────────────────────────────────


def test_plan_block_maps_statuses_to_glyphs_and_counts_progress():
    assert plan_block(
        [
            {"content": "read the file", "status": "completed"},
            {"content": "edit it", "status": "in_progress"},
            {"content": "run tests", "status": "pending"},
            {"content": "drop the spike", "status": "cancelled"},
            {"content": "unstated"},
        ]
    ) == vm.ListBlock(
        label="plan · 2 of 5",
        rows=[
            vm.ListRow(glyph="done", text="read the file"),
            vm.ListRow(glyph="active", text="edit it"),
            vm.ListRow(glyph="pending", text="run tests"),
            vm.ListRow(glyph="done", text="drop the spike"),
            vm.ListRow(glyph="pending", text="unstated"),
        ],
    )


def test_plan_block_clamps_the_active_step_to_the_list():
    assert plan_block(
        [
            {"content": "a", "status": "completed"},
            {"content": "b", "status": "completed"},
        ]
    ) == vm.ListBlock(
        label="plan · 2 of 2",
        rows=[
            vm.ListRow(glyph="done", text="a"),
            vm.ListRow(glyph="done", text="b"),
        ],
    )


def test_is_plan_update_matches_by_tool_name():
    assert is_plan_update(execution("update_todos", ExecutionStatus.PENDING)) is True
    assert is_plan_update(execution("bash", ExecutionStatus.PENDING)) is False


def test_plan_from_execution_reads_todos_and_drops_non_dict_items():
    block = plan_from_execution(
        execution("update_todos", ExecutionStatus.PENDING, {"todos": [{"content": "a", "status": "pending"}, "junk"]})
    )
    assert block == vm.ListBlock(label="plan · 1 of 1", rows=[vm.ListRow(glyph="pending", text="a")])


def test_plan_from_execution_without_a_list_is_none():
    assert plan_from_execution(execution("update_todos", ExecutionStatus.PENDING, {"todos": "nope"})) is None


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
        "te2": ToolExecution(
            id="te2",
            conversation_id="c1",
            parent_id="te1",
            created_at=500,
            tool_call_id="tc2",
            raw_tool_call=ToolCall(
                id="tc2",
                name="update_todos",
                arguments={
                    "todos": [
                        {"content": "read the file", "status": "completed"},
                        {"content": "run tests", "status": "in_progress"},
                    ]
                },
            ),
            tool_spec=spec("update_todos"),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="ok")]),
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
