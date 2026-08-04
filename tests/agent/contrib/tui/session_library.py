"""The catalogue's worlds: `AgentSession` literals, authored here, committed
as JSON under `luca/agent/contrib/tui/fixtures/sessions/`.

Authored, never captured. A recorded real session is a 100KB diff nobody can
review, full of absolute paths and whatever model happened to be configured;
these are a dozen lines each and say exactly one thing. Everything is fixed —
ids, timestamps, token counts — so the built screens are byte-stable and the
snapshot suite is meaningful.

Regenerate after editing:

    uv run python -m tests.agent.contrib.tui.session_library

`test_catalog.py` fails if the committed JSON and these builders disagree, so
the two cannot drift.
"""

from __future__ import annotations

from pathlib import Path

from luca.agent.contrib.resource_permissions import (
    AnswerOption,
    PermissionRequest,
    ResourcePermission,
)
from luca.agent.core.models import (
    AgentSession,
    ApprovalStatus,
    AssistantMessage,
    ChildConversation,
    ExecutionResult,
    ExecutionStatus,
    LLMConfig,
    SessionConfig,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    Usage,
    UserMessage,
)
from tests.agent.scenarios import conversation, make_session, spec

SESSIONS_DIR = Path(__file__).resolve().parents[4] / "luca/agent/contrib/tui/fixtures/sessions"

MODEL = LLMConfig(model="claude-sonnet-5", provider="anthropic")

# Fixed wall clock in ms. The session spans ~22 minutes, which is what the cost
# screen's subline reports.
T0 = 1_700_000_000_000
MINUTE = 60_000


def at(minutes: float) -> int:
    return T0 + int(minutes * MINUTE)


# ── literal factories ─────────────────────────────────────────────────────────


def user(entry_id: str, text: str, minutes: float) -> UserMessage:
    return UserMessage(id=entry_id, created_at=at(minutes), parts=[TextContent(text=text)], context_tokens=40)


def assistant(entry_id: str, minutes: float, *, thinking: str | None = None, text: str = "") -> AssistantMessage:
    parts: list = []
    if thinking:
        parts.append(ThinkingContent(thinking=thinking))
    if text:
        parts.append(TextContent(text=text))
    return AssistantMessage(
        id=entry_id,
        created_at=at(minutes),
        parts=parts,
        llm_config=MODEL,
        stop_reason="tool_use" if not text else "end_turn",
        context_tokens=len(text) // 4 + 20,
    )


def call(
    entry_id: str,
    name: str,
    arguments: dict,
    minutes: float,
    *,
    status: ExecutionStatus = ExecutionStatus.COMPLETED,
    approval_status: ApprovalStatus | None = None,
    result: ExecutionResult | None = None,
    error: ToolExecutionError | None = None,
    seconds: float = 0.4,
    context_tokens: int = 300,
    extras: dict | None = None,
) -> ToolExecution:
    return ToolExecution(
        id=entry_id,
        created_at=at(minutes),
        started_at=at(minutes),
        ended_at=at(minutes) + int(seconds * 1000),
        tool_call_id=f"tc_{entry_id}",
        raw_tool_call=ToolCall(id=f"tc_{entry_id}", name=name, arguments=arguments),
        tool_spec=spec(name),
        status=status,
        approval_status=approval_status,
        result=result,
        error=error,
        context_tokens=context_tokens,
        extras=extras or {},
    )


def text_result(*lines: str, **metadata) -> ExecutionResult:
    return ExecutionResult(content=[TextContent(text="\n".join(lines))], metadata=metadata)


def turn(index: int, minutes: float) -> TurnStart:
    return TurnStart(id=f"ts{index}", created_at=at(minutes))


def finish(index: int, minutes: float, outcome: TurnOutcome = TurnOutcome.COMPLETED) -> TurnFinish:
    return TurnFinish(id=f"tf{index}", created_at=at(minutes), outcome=outcome)


def usage(conversation_id: str, entry_id: str, **counts: int) -> Usage:
    return Usage(conversation_id=conversation_id, entry_id=entry_id, **counts)


# ── the worlds ────────────────────────────────────────────────────────────────


def empty_session() -> AgentSession:
    """A session that has never been used: the first thing anyone sees."""
    return make_session(
        id="empty",
        entries={},
        conversations={"c1": conversation("c1", [], created_at=T0, updated_at=T0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )


def conversation_session() -> AgentSession:
    """The comprehensive one: two turns, thinking, a plan, reads, a rule-
    approved edit with a diff stat, a failing test run, and the recovery."""
    entries = {
        "ts1": turn(1, 0),
        "u1": user("u1", "migrate the event store to sqlite", 0),
        "a1": assistant("a1", 0.2, thinking="weighing the schema options"),
        "te1": call(
            "te1",
            "read",
            {"path": "luca/store/reader.py"},
            0.4,
            result=text_result(*[f"line {n}" for n in range(1, 85)]),
            context_tokens=1900,
        ),
        "te2": call(
            "te2",
            "grep",
            {"pattern": "def emit", "path": "luca"},
            0.9,
            result=text_result("Found 6 matches", "luca/cli.py:", "luca/events.py:"),
            context_tokens=210,
        ),
        "a2": assistant("a2", 1.2, text="The reader is the only place that opens the jsonl file. Planning the move."),
        "te3": call(
            "te3",
            "update_todos",
            {
                "todos": [
                    {"content": "sqlite schema + writer", "status": "completed"},
                    {"content": "port the reader", "status": "completed"},
                    {"content": "backfill script + drop the jsonl writer", "status": "in_progress"},
                    {"content": "update the docs", "status": "pending"},
                ]
            },
            1.4,
            result=text_result("plan updated"),
            context_tokens=90,
        ),
        "te4": call(
            "te4",
            "edit",
            {"path": "luca/events.py"},
            2.1,
            result=text_result(
                "edited luca/events.py",
                diff="--- a\n+++ b\n-    for e in events:\n-        out.append(e)\n+    if fmt == 'json':\n"
                "+        return [json.dumps(asdict(e)) for e in events]\n+\n+    for e in events:\n"
                "+        out.append(e)\n+",
            ),
            seconds=0.2,
            context_tokens=140,
            # Gated and allowed BY A RULE — the transcript has to say so.
            approval_status=ApprovalStatus.ALLOWED,
            extras={"approval_context": {"requests": [_edit_request().model_dump()]}},
        ),
        "te5": call(
            "te5",
            "bash",
            {"command": "uv run pytest tests/store -q"},
            2.6,
            status=ExecutionStatus.COMPLETED,
            result=text_result(
                "F....",
                "=================================== FAILURES ===================================",
                "____________________________ test_reader_roundtrip _____________________________",
                "    def test_reader_roundtrip():",
                ">       assert list(read(path)) == events",
                "E       AssertionError: assert [] == [Event(ts=1, ...)]",
                "tests/store/test_reader.py:41: AssertionError",
                "1 failed, 4 passed in 1.9s",
                "",
                "the writer never flushed",
                exit=1,
            ),
            seconds=1.9,
            context_tokens=420,
        ),
        "a3": assistant("a3", 4.8, text="The writer never flushes before the reader opens it. Fixing that."),
        "tf1": finish(1, 5.0),
        "ts2": turn(2, 12.0),
        "u2": user("u2", "yes, and add a regression test", 12.0),
        "a4": assistant("a4", 12.3, thinking="short"),
        "te6": call(
            "te6",
            "edit",
            {"path": "luca/store/writer.py"},
            12.6,
            result=text_result("edited luca/store/writer.py", diff="--- a\n+++ b\n-    pass\n+    handle.flush()\n"),
            seconds=0.2,
            context_tokens=110,
        ),
        "te7": call(
            "te7",
            "bash",
            {"command": "uv run pytest tests/store -q"},
            13.0,
            result=text_result("5 passed in 1.7s", exit=0),
            seconds=1.7,
            context_tokens=60,
        ),
        "a5": assistant(
            "a5",
            21.8,
            text="Green. `luca/store/writer.py` now flushes on every append, and the reader round-trips.",
        ),
        "tf2": finish(2, 22.0),
    }
    session = make_session(
        id="conversation",
        entries=entries,
        conversations={
            "c1": conversation("c1", list(entries), created_at=T0, updated_at=at(22)),
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    session.usages = {
        "c1": {
            "a1": usage("c1", "a1", input=8_200, output=310, cache_read=41_000, cache_write=2_100),
            "a2": usage("c1", "a2", input=9_600, output=520, cache_read=44_000, cache_write=300),
            "a3": usage("c1", "a3", input=11_800, output=640, cache_read=48_000, cache_write=200),
            "a5": usage("c1", "a5", input=13_100, output=880, cache_read=51_000, cache_write=180),
        }
    }
    return session


def _edit_request() -> PermissionRequest:
    return PermissionRequest(
        resources=[ResourcePermission(permission="edit", resource="luca/events.py")],
        answer_options=[
            AnswerOption(
                resource_permissions=[ResourcePermission(permission="edit", resource="luca/**")],
                metadata={"preview": "edits under luca/"},
            )
        ],
        metadata={"preview": "Apply this edit to [accent]luca/events.py[/]"},
    )


def approval_session() -> AgentSession:
    """A turn parked at the gate: the edit is proposed, not applied."""
    entries = {
        "ts1": turn(1, 0),
        "u1": user("u1", "add a --json flag to the events command", 0),
        "a1": assistant("a1", 0.2, thinking="reading the formatter"),
        "te1": call(
            "te1",
            "read",
            {"path": "luca/events.py"},
            0.4,
            result=text_result(*[f"line {n}" for n in range(1, 62)]),
            context_tokens=1400,
        ),
        "a2": assistant("a2", 0.8, text="I can add the flag in `format()`. It needs an edit to `luca/events.py`."),
        "te2": call(
            "te2",
            "edit",
            {"path": "luca/events.py"},
            1.0,
            status=ExecutionStatus.PENDING,
            approval_status=ApprovalStatus.PENDING,
            seconds=0,
            context_tokens=0,
            extras={"approval_context": {"requests": [_edit_request().model_dump()]}},
        ),
    }
    session = make_session(
        id="approval",
        entries=entries,
        conversations={"c1": conversation("c1", list(entries), created_at=T0, updated_at=at(1))},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    session.usages = {"c1": {"a2": usage("c1", "a2", input=7_400, output=260, cache_read=18_000)}}
    return session


def failures_session() -> AgentSession:
    """Everything that can go wrong in one column: denied by rule, a tool that
    failed, and a turn the user cancelled."""
    entries = {
        "ts1": turn(1, 0),
        "u1": user("u1", "clean the build directory and re-run the suite", 0),
        "a1": assistant("a1", 0.2, thinking="checking what is safe to remove"),
        "te1": call(
            "te1",
            "bash",
            {"command": "rm -rf build/ dist/"},
            0.4,
            status=ExecutionStatus.REJECTED,
            approval_status=ApprovalStatus.REJECTED,
            seconds=0,
            context_tokens=30,
            extras={"approval_context": {"requests": [_edit_request().model_dump()]}},
        ),
        "a2": assistant("a2", 0.6, text="That is blocked by a rule. Using `git clean` instead."),
        "te2": call(
            "te2",
            "bash",
            {"command": "git clean -fdx build/"},
            0.9,
            result=text_result("removed 12 paths", exit=0),
            context_tokens=45,
        ),
        "te3": call(
            "te3",
            "bash",
            {"command": "uv run pytest -q"},
            1.4,
            status=ExecutionStatus.TIMED_OUT,
            error=ToolExecutionError(error_type="TimeoutError", error_message="the tool exceeded its 120s budget"),
            seconds=120,
            context_tokens=25,
        ),
        "a3": assistant("a3", 3.6, text="The suite is hanging. Narrowing it to the store tests."),
        "tf1": finish(1, 3.8, TurnOutcome.CANCELLED),
    }
    session = make_session(
        id="failures",
        entries=entries,
        conversations={"c1": conversation("c1", list(entries), created_at=T0, updated_at=at(4))},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    session.usages = {"c1": {"a3": usage("c1", "a3", input=6_100, output=210, cache_read=12_000)}}
    return session


SPAWN_SCHEMA = {
    "type": "object",
    "properties": {"is_subagent_spawn": {"type": "boolean"}},
    "required": ["is_subagent_spawn"],
}


def _spawn(entry_id: str, description: str, prompt: str, minutes: float) -> ToolExecution:
    execution = call(
        entry_id,
        "task",
        {"description": description, "prompt": prompt},
        minutes,
        result=ExecutionResult(
            content=[TextContent(text="task started")],
            structured_content={
                "is_subagent_spawn": True,
                "task_id": entry_id,
                "description": description,
                "prompt": prompt,
                "process_subagent_result_tool_name": "task_result",
            },
        ),
        context_tokens=60,
    )
    execution.tool_spec = spec("task", output_schema=SPAWN_SCHEMA)
    return execution


def subagents_session() -> AgentSession:
    """One turn that fans out: two subagents, one done, one that failed."""
    entries = {
        "ts1": turn(1, 0),
        "u1": user("u1", "audit the docs and the tests for stale references", 0),
        "a1": assistant("a1", 0.2, text="Spawning two subagents in parallel."),
        "sp1": _spawn("sp1", "audit the docs", "Read every page under docs/ and list stale references.", 0.3),
        "cc1": ChildConversation(
            id="cc1",
            created_at=at(0.3),
            conversation_id="c2",
            tool_execution_id="sp1",
            execution_result=text_result("3 stale references"),
            context_tokens=180,
        ),
        "sp2": _spawn("sp2", "audit the tests", "Find tests referencing the jsonl store.", 0.3),
        "cc2": ChildConversation(
            id="cc2",
            created_at=at(0.3),
            conversation_id="c3",
            tool_execution_id="sp2",
            execution_result=ExecutionResult(
                content=[TextContent(text="the grep tool timed out")],
                is_error=True,
            ),
            context_tokens=40,
        ),
        "a2": assistant(
            "a2",
            6.0,
            text="The docs audit found 3 stale references; the test audit timed out and needs a rerun.",
        ),
        "tf1": finish(1, 6.2),
        # c2 — the docs subagent
        "u_c2": user("u_c2", "Read every page under docs/ and list stale references.", 0.3),
        "a_c2a": assistant("a_c2a", 0.5, thinking="listing docs/"),
        "te_c2a": call(
            "te_c2a",
            "glob",
            {"pattern": "docs/**/*.md"},
            0.7,
            result=text_result("docs/agent/README.md", "docs/client/README.md", "docs/llm.txt"),
            context_tokens=70,
        ),
        "te_c2b": call(
            "te_c2b",
            "grep",
            {"pattern": "jsonl", "path": "docs"},
            1.4,
            result=text_result("Found 3 matches", "docs/agent/README.md:", "docs/llm.txt:"),
            context_tokens=95,
        ),
        "a_c2b": assistant("a_c2b", 5.4, text="Three pages still describe the jsonl store."),
        # c3 — the tests subagent
        "u_c3": user("u_c3", "Find tests referencing the jsonl store.", 0.3),
        "a_c3a": assistant("a_c3a", 0.5, thinking="grepping tests/"),
        "te_c3a": call(
            "te_c3a",
            "grep",
            {"pattern": "jsonl", "path": "tests"},
            0.8,
            status=ExecutionStatus.TIMED_OUT,
            error=ToolExecutionError(error_type="TimeoutError", error_message="the tool exceeded its 30s budget"),
            seconds=30,
            context_tokens=20,
        ),
    }
    main = ["ts1", "u1", "a1", "sp1", "cc1", "sp2", "cc2", "a2", "tf1"]
    session = make_session(
        id="subagents",
        entries=entries,
        conversations={
            "c1": conversation("c1", main, created_at=T0, updated_at=at(6.2)),
            "c2": conversation("c2", ["u_c2", "a_c2a", "te_c2a", "te_c2b", "a_c2b"], depth=1, created_at=at(0.3)),
            "c3": conversation("c3", ["u_c3", "a_c3a", "te_c3a"], depth=1, created_at=at(0.3)),
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    session.usages = {
        "c1": {"a2": usage("c1", "a2", input=9_100, output=430, cache_read=22_000, cache_write=900)},
        "c2": {"a_c2b": usage("c2", "a_c2b", input=4_200, output=180, cache_read=8_000)},
        "c3": {"a_c3a": usage("c3", "a_c3a", input=2_100, output=60)},
    }
    return session


# Mention paths are stored ABSOLUTE, exactly as the live `prompt_files` handler
# stores them — and rooted somewhere that can never be a home directory.
# `render.mention_blocks` passes the path through `format.home_path`, which
# resolves relative paths against the process cwd and contracts anything under
# `~`; either would make this world render differently on every machine.
WORKSPACE = "/quantized/luca"


def mention(entry_id: str, text: str, path: str, lines: int, minutes: float) -> UserMessage:
    """A user turn carrying an `@` mention: the typed words, plus one part per
    inlined file. The mention part is annotated exactly as `prompt_files`
    annotates it, so it renders as its own `▸ read` row and never as prose."""
    return UserMessage(
        id=entry_id,
        created_at=at(minutes),
        parts=[
            TextContent(text=text),
            TextContent(
                text=f"<contents of {path}>",
                metadata={"mention": {"path": path, "success": True, "lines": lines}},
            ),
        ],
        context_tokens=1400,
    )


def planning_session() -> AgentSession:
    """The fan-out shape: a mentioned file, a plan, and the plan's first step
    carried out by two subagents at once — one reading, one editing."""
    entries = {
        "ts1": turn(1, 0),
        "u1": mention(
            "u1",
            "split @luca/store/reader.py into a reader and a writer",
            f"{WORKSPACE}/luca/store/reader.py",
            84,
            0,
        ),
        "a1": assistant("a1", 0.2, thinking="sizing the split"),
        "te_plan": call(
            "te_plan",
            "update_todos",
            {
                "todos": [
                    {"content": "audit the reader and draft the split", "status": "in_progress"},
                    {"content": "move the write path into writer.py", "status": "pending"},
                    {"content": "update every import site", "status": "pending"},
                    {"content": "backfill the tests", "status": "pending"},
                ]
            },
            0.4,
            result=text_result("plan updated"),
            context_tokens=95,
        ),
        "a2": assistant(
            "a2", 0.6, text="Step one, in parallel: one subagent audits the reader, one drafts the writer."
        ),
        "sp1": _spawn(
            "sp1",
            "audit the reader",
            "Read the store module and report what belongs on the write path.",
            0.7,
        ),
        "cc1": ChildConversation(
            id="cc1",
            created_at=at(0.7),
            conversation_id="c2",
            tool_execution_id="sp1",
            execution_result=text_result("4 write-path methods identified"),
            context_tokens=210,
        ),
        "sp2": _spawn("sp2", "draft the writer", "Create writer.py and move the write path into it.", 0.7),
        "cc2": ChildConversation(
            id="cc2",
            created_at=at(0.7),
            conversation_id="c3",
            tool_execution_id="sp2",
            execution_result=text_result("writer.py created, 2 files touched"),
            context_tokens=180,
        ),
        "a3": assistant(
            "a3",
            8.4,
            text="`writer.py` now owns the write path. Next: the import sites.",
        ),
        "tf1": finish(1, 8.6),
        # c2 — the reading subagent
        "u_c2": user("u_c2", "Read the store module and report what belongs on the write path.", 0.7),
        "a_c2a": assistant("a_c2a", 0.9, thinking="reading the store"),
        "te_c2a": call(
            "te_c2a",
            "read",
            {"path": "luca/store/reader.py"},
            1.1,
            result=text_result(*[f"line {n}" for n in range(1, 85)]),
            context_tokens=1900,
        ),
        "te_c2b": call(
            "te_c2b",
            "read",
            {"path": "luca/store/schema.sql"},
            1.9,
            result=text_result(*[f"line {n}" for n in range(1, 32)]),
            context_tokens=430,
        ),
        "te_c2c": call(
            "te_c2c",
            "grep",
            {"pattern": "def append|def flush", "path": "luca/store"},
            2.6,
            result=text_result("Found 4 matches", "luca/store/reader.py:", "luca/store/__init__.py:"),
            context_tokens=180,
        ),
        "a_c2b": assistant(
            "a_c2b",
            6.9,
            text="`append`, `flush`, `_rotate` and `close` are the write path — all four live in `reader.py` today.",
        ),
        # c3 — the editing subagent
        "u_c3": user("u_c3", "Create writer.py and move the write path into it.", 0.7),
        "a_c3a": assistant("a_c3a", 0.9, thinking="drafting the module"),
        "te_c3a": call(
            "te_c3a",
            "write",
            {"path": "luca/store/writer.py"},
            1.4,
            result=text_result(
                "created luca/store/writer.py",
                diff="--- /dev/null\n+++ b\n+class Writer:\n+    def append(self, event):\n+        ...\n"
                "+    def flush(self):\n+        ...\n+    def close(self):\n+        ...\n",
            ),
            seconds=0.3,
            context_tokens=160,
            approval_status=ApprovalStatus.ALLOWED,
            extras={"approval_context": {"requests": [_edit_request().model_dump()]}},
        ),
        "te_c3b": call(
            "te_c3b",
            "edit",
            {"path": "luca/store/__init__.py"},
            2.2,
            result=text_result(
                "edited luca/store/__init__.py",
                diff="--- a\n+++ b\n-from .reader import Reader, append, flush\n+from .reader import Reader\n"
                "+from .writer import Writer\n",
            ),
            seconds=0.2,
            context_tokens=70,
            approval_status=ApprovalStatus.ALLOWED,
            extras={"approval_context": {"requests": [_edit_request().model_dump()]}},
        ),
        "a_c3b": assistant(
            "a_c3b", 7.8, text="`writer.py` holds the write path and `store/__init__.py` re-exports it."
        ),
    }
    main = ["ts1", "u1", "a1", "te_plan", "a2", "sp1", "cc1", "sp2", "cc2", "a3", "tf1"]
    session = make_session(
        id="planning",
        entries=entries,
        conversations={
            "c1": conversation("c1", main, created_at=T0, updated_at=at(8.6)),
            "c2": conversation(
                "c2", ["u_c2", "a_c2a", "te_c2a", "te_c2b", "te_c2c", "a_c2b"], depth=1, created_at=at(0.7)
            ),
            "c3": conversation("c3", ["u_c3", "a_c3a", "te_c3a", "te_c3b", "a_c3b"], depth=1, created_at=at(0.7)),
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    session.usages = {
        "c1": {"a3": usage("c1", "a3", input=10_400, output=390, cache_read=26_000, cache_write=1_200)},
        "c2": {"a_c2b": usage("c2", "a_c2b", input=5_800, output=240, cache_read=11_000)},
        "c3": {"a_c3b": usage("c3", "a_c3b", input=4_900, output=310, cache_read=9_000)},
    }
    return session


BUILDERS = {
    "empty": empty_session,
    "conversation": conversation_session,
    "approval": approval_session,
    "failures": failures_session,
    "subagents": subagents_session,
    "planning": planning_session,
}


def build_all() -> dict[str, AgentSession]:
    return {name: builder() for name, builder in BUILDERS.items()}


def write_all(directory: Path | None = None) -> list[Path]:
    root = directory or SESSIONS_DIR
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for name, session in build_all().items():
        path = root / f"{name}.json"
        path.write_text(session.model_dump_json(indent=2) + "\n")
        written.append(path)
    return written


if __name__ == "__main__":
    for path in write_all():
        print(f"wrote {path}")
