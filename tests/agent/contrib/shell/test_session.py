"""Self-scoped tests for the persistent bash session: what survives between
commands, what a restart/timeout/cancel destroys, and how the pool splits one
shell per conversation lineage.

Every test that starts a shell must close it — the suite runs with
`-W error::ResourceWarning`, so a leaked process fails it rather than
lingering. The `shell` and `pool` fixtures are the only doors.
"""

import asyncio

import pytest

from luca.agent.contrib.shell.session import (
    CommandResult,
    ShellSession,
    ShellSessionPool,
    called_tool,
    lineage,
    lineage_root,
)
from luca.agent.core import (
    AgentSession,
    CancellationToken,
    ExecutionStatus,
    LLMConfig,
    SessionConfig,
    ToolCall,
    ToolExecution,
)
from tests.agent.scenarios import conversation

TIMEOUT_MS = 10_000


@pytest.fixture
async def shell(tmp_path):
    session = ShellSession(tmp_path)
    yield session
    await session.aclose()


@pytest.fixture
async def pool(tmp_path):
    shells = ShellSessionPool(tmp_path)
    yield shells
    await shells.aclose()


async def run(shell: ShellSession, command: str, *, timeout_ms: int = TIMEOUT_MS) -> CommandResult:
    return await shell.run(command, timeout_ms=timeout_ms, cancellation_token=CancellationToken())


def session_with(*conversations) -> AgentSession:
    return AgentSession(
        id="s_shell",
        conversations={c.id: c for c in conversations},
        main_conversation_id=conversations[0].id,
        session_config=SessionConfig(llm_config=LLMConfig(model="test-model", provider="faux")),
    )


# ── what one command reports ──────────────────────────────────────────────────


async def test_a_command_reports_its_output_and_exit_code(shell):
    assert await run(shell, "echo hi") == CommandResult(output="hi\n", exit_code=0, outcome="completed")


async def test_output_without_a_trailing_newline_is_exact(shell):
    # The driver prints its sentinel on a line of its own, and that added
    # newline is dropped again — otherwise every command would appear to end
    # with one.
    assert await run(shell, "printf abc") == CommandResult(output="abc", exit_code=0, outcome="completed")


async def test_a_non_zero_exit_is_reported_not_raised(shell):
    assert await run(shell, "(exit 7)") == CommandResult(output="", exit_code=7, outcome="completed")


async def test_stderr_is_merged_into_stdout(shell):
    result = await run(shell, "echo out; echo err 1>&2")

    assert sorted(result.output.split()) == ["err", "out"]


async def test_a_heredoc_survives_verbatim(shell):
    # The command text goes in a FILE, so newlines and quoting reach bash
    # exactly as the model wrote them.
    result = await run(shell, "cat <<'EOF'\nline one\nline 'two'\nEOF")

    assert result.output == "line one\nline 'two'\n"


async def test_a_command_reading_stdin_neither_hangs_nor_eats_the_next_one(shell):
    # `cat` with no redirection would read the pipe we drive the shell with and
    # swallow whatever we send next; `< /dev/null` on the sourced script is
    # what stops it.
    assert await run(shell, "cat") == CommandResult(output="", exit_code=0, outcome="completed")
    assert await run(shell, "echo alive") == CommandResult(output="alive\n", exit_code=0, outcome="completed")


async def test_output_quoting_a_sentinel_does_not_end_the_read_early(shell):
    # The real sentinel is a fresh uuid per command, so nothing the model can
    # print will match it.
    result = await run(shell, "printf '__LUCA_DONE_deadbeef__ 0\\n/nowhere\\nreal output\\n'")

    assert result == CommandResult(
        output="__LUCA_DONE_deadbeef__ 0\n/nowhere\nreal output\n",
        exit_code=0,
        outcome="completed",
    )


# ── what persists ─────────────────────────────────────────────────────────────


async def test_the_working_directory_persists(tmp_path, shell):
    (tmp_path / "sub").mkdir()

    await run(shell, "cd sub")

    assert await run(shell, "pwd") == CommandResult(
        output=f"{tmp_path / 'sub'}\n",
        exit_code=0,
        outcome="completed",
    )


async def test_the_reported_cwd_follows_the_shell(tmp_path, shell):
    # What `build_permission_requests` reads: the approval prompt has to name
    # the directory the NEXT command runs in.
    (tmp_path / "sub").mkdir()
    assert shell.cwd == str(tmp_path)

    await run(shell, "cd sub")

    assert shell.cwd == str(tmp_path / "sub")


async def test_environment_variables_persist_exported_or_not(shell):
    await run(shell, "export EXPORTED=one; unexported=two")

    result = await run(shell, 'printf "%s %s" "$EXPORTED" "$unexported"')

    assert result.output == "one two"


async def test_shell_functions_persist(shell):
    # The big one in practice: `source .venv/bin/activate`, `nvm` and `conda`
    # all install themselves as shell functions.
    await run(shell, "greet() { printf 'hello %s' \"$1\"; }")

    assert (await run(shell, "greet world")).output == "hello world"


async def test_shell_options_persist(shell):
    await run(shell, "set -o pipefail")

    assert (await run(shell, "false | true")).exit_code == 1


# ── what ends a shell ─────────────────────────────────────────────────────────


async def test_a_restart_empties_the_session(tmp_path, shell):
    (tmp_path / "sub").mkdir()
    await run(shell, "cd sub; export GREETING=hi")

    await shell.restart()

    assert await run(shell, 'printf "%s[%s]" "$PWD" "$GREETING"') == CommandResult(
        output=f"{tmp_path}[]",
        exit_code=0,
        outcome="completed",
    )


async def test_a_timeout_reports_partial_output_and_leaves_a_fresh_shell(tmp_path, shell):
    (tmp_path / "sub").mkdir()
    await run(shell, "cd sub")

    result = await run(shell, "echo before; sleep 30", timeout_ms=500)

    assert result == CommandResult(output="before\n", exit_code=None, outcome="timed_out")
    assert await run(shell, "pwd") == CommandResult(
        output=f"{tmp_path}\n",
        exit_code=0,
        outcome="completed",
    )


async def test_a_timeout_takes_the_runaway_command_with_it(tmp_path, shell):
    # The command runs IN the shell's process, so it has no process group of
    # its own — killing the SHELL's group is what reaps it and its children.
    await run(shell, f"sleep 30 & echo $! > {tmp_path}/pid; wait", timeout_ms=500)
    pid = (tmp_path / "pid").read_text().strip()

    assert await run(shell, f"kill -0 {pid} 2>/dev/null; echo $?") == CommandResult(
        output="1\n",
        exit_code=0,
        outcome="completed",
    )


async def test_cancellation_returns_the_output_that_arrived_first(shell):
    token = CancellationToken()
    task = asyncio.create_task(shell.run("echo before; sleep 30", timeout_ms=TIMEOUT_MS, cancellation_token=token))
    await asyncio.sleep(0.3)

    token.cancel()

    assert await asyncio.wait_for(task, timeout=5) == CommandResult(
        output="before\n",
        exit_code=None,
        outcome="cancelled",
    )


async def test_aclose_does_not_wait_for_a_running_command(shell):
    # The user is quitting; a command with two minutes left on its timeout must
    # not hold the door. The in-flight read sees EOF and reports what it has.
    task = asyncio.create_task(
        shell.run("echo before; sleep 30", timeout_ms=TIMEOUT_MS, cancellation_token=CancellationToken())
    )
    await asyncio.sleep(0.3)

    await shell.aclose()

    assert await asyncio.wait_for(task, timeout=5) == CommandResult(
        output="before\n",
        exit_code=None,
        outcome="shell_died",
    )


async def test_a_command_that_exits_the_shell_says_so(shell):
    result = await run(shell, "echo bye; exit 3")

    assert result == CommandResult(output="bye\n", exit_code=None, outcome="shell_died")


async def test_the_next_command_after_a_death_gets_a_new_shell(shell):
    await run(shell, "exit 3")

    assert await run(shell, "echo alive") == CommandResult(
        output="alive\n",
        exit_code=0,
        outcome="completed",
    )


# ── freshness: the condition behind the plugin's notice ───────────────────────


async def test_a_shell_that_has_run_nothing_is_fresh(shell):
    assert shell.fresh is True


async def test_running_a_command_spends_the_freshness(shell):
    await run(shell, "true")

    assert shell.fresh is False


@pytest.mark.parametrize("ending", ["restart", "timeout", "death"])
async def test_every_way_a_shell_ends_re_arms_the_notice(shell, ending):
    await run(shell, "true")

    if ending == "restart":
        await shell.restart()
    elif ending == "timeout":
        await run(shell, "sleep 30", timeout_ms=200)
    else:
        await run(shell, "exit 1")

    assert shell.fresh is True


# ── the pool ──────────────────────────────────────────────────────────────────


def test_one_shell_per_conversation(pool):
    session = session_with(
        conversation("c1", []),
        conversation("c2", []),
    )

    assert pool.get(session, "c1") is pool.get(session, "c1")
    assert pool.get(session, "c1") is not pool.get(session, "c2")


def test_a_compacted_conversation_keeps_the_shell_it_had(pool):
    # Compaction mints a new id for the SAME agent, mid-task, while the process
    # is still alive. Nothing restarted, so nothing may change.
    session = session_with(
        conversation("c2", [], previous_conversation_id="c1"),
        conversation("c1", []),
    )

    assert pool.get(session, "c2") is pool.get(session, "c1")


def test_an_unknown_conversation_still_gets_a_shell(pool):
    # A tool must never fail because history is not where it expected it.
    assert pool.get(session_with(conversation("c1", [])), "c9") is not None


async def test_a_conversation_that_never_asked_for_a_shell_is_fresh(pool):
    session = session_with(conversation("c1", []))

    assert pool.is_fresh(session, "c1") is True

    await run(pool.get(session, "c1"), "true")

    assert pool.is_fresh(session, "c1") is False


async def test_aclose_kills_every_shell_and_forgets_it(pool, tmp_path):
    session = session_with(conversation("c1", []))
    shell = pool.get(session, "c1")
    await run(shell, "true")

    await pool.aclose()

    assert shell.fresh is True
    assert pool.get(session, "c1") is not shell


# ── lineage and history ───────────────────────────────────────────────────────


def test_lineage_walks_back_through_every_compaction():
    session = session_with(
        conversation("c3", [], previous_conversation_id="c2"),
        conversation("c2", [], previous_conversation_id="c1"),
        conversation("c1", []),
    )

    assert lineage(session, "c3") == ["c3", "c2", "c1"]
    assert lineage_root(session, "c3") == "c1"


def test_lineage_of_an_unknown_conversation_is_itself():
    assert lineage(session_with(conversation("c1", [])), "c9") == ["c9"]


def test_called_tool_sees_a_call_made_before_a_compaction():
    # The successor's path holds the summary, not the calls it replaced — so
    # the question has to be asked of the whole lineage.
    session = session_with(
        conversation("c2", [], previous_conversation_id="c1"),
        conversation("c1", ["x_1"]),
    )
    session.entries["x_1"] = ToolExecution(
        id="x_1",
        created_at=1,
        conversation_id="c1",
        tool_call_id="c_1",
        raw_tool_call=ToolCall(id="c_1", name="anthropic_bash_20250124", arguments={"command": "pwd"}),
        status=ExecutionStatus.COMPLETED,
    )

    assert called_tool(session, "c2", "anthropic_bash_20250124") is True
    assert called_tool(session, "c2", "bash") is False
