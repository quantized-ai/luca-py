"""`PersistentShell` — the contract Anthropic's bash tool is specified against.

Real shell processes against `tmp_path`. Every test here exists because the
naive version of this component passes a smoke test and then fails in one
specific way: state that does not persist, a session wedged by a command
reading stdin, a marker a command can forge, a timeout that leaves the shell
owned by the dead command's output.
"""

import asyncio
import os

import pytest

from luca.agent.contrib.shell.session_shell import PersistentShell
from luca.agent.core import CancellationToken


@pytest.fixture
async def shell(tmp_path):
    session = PersistentShell(tmp_path)
    yield session
    await session.close()


async def run(session, command, timeout_ms=10_000):
    return await session.run(command, timeout_ms=timeout_ms)


# ── the persistent part ──────────────────────────────────────────────────────


async def test_a_command_returns_its_output_and_exit_code(shell):
    result = await run(shell, "echo hello")

    assert (result.stdout, result.exit_code, result.outcome) == ("hello\n", 0, "completed")


async def test_the_working_directory_survives_between_commands(shell, tmp_path):
    (tmp_path / "sub").mkdir()

    await run(shell, "cd sub")
    result = await run(shell, "pwd")

    assert result.stdout.strip().endswith("/sub")


async def test_environment_variables_survive_between_commands(shell):
    await run(shell, "export GREETING=hi")

    assert (await run(shell, "echo $GREETING")).stdout == "hi\n"


async def test_restart_clears_the_session(shell, tmp_path):
    (tmp_path / "sub").mkdir()
    await run(shell, "cd sub; export GREETING=hi")

    await shell.restart()

    assert (await run(shell, "echo $GREETING")).stdout == "\n"
    assert not (await run(shell, "pwd")).stdout.strip().endswith("/sub")


async def test_a_nonzero_exit_code_is_reported(shell):
    assert (await run(shell, "false")).exit_code == 1
    assert (await run(shell, "sh -c 'exit 7'")).exit_code == 7


async def test_stdout_and_stderr_stay_separate(shell):
    result = await run(shell, "echo out; echo err >&2")

    assert (result.stdout, result.stderr) == ("out\n", "err\n")


# ── the traps ────────────────────────────────────────────────────────────────


async def test_a_command_that_reads_stdin_does_not_hang_the_session(shell):
    """The shell's stdin is the control channel, so a command that reads it
    could eat what comes next — piped into a bare shell, a `cat` swallows the
    following line rather than running it.

    This passes with and without the `/dev/null` redirect today, because each
    command is written together with its markers and the shell buffers the lot
    before running anything. It is a regression guard on the outcome, not proof
    that the redirect is what produces it."""
    result = await asyncio.wait_for(run(shell, "cat"), timeout=10)

    assert result.outcome == "completed"
    assert (await run(shell, "echo alive")).stdout == "alive\n"


async def test_a_command_cannot_forge_the_end_of_its_own_output(shell):
    # the marker is fresh per command, so output the model controls cannot
    # truncate the read
    forged = "00112233445566778899aabbccddeeff 0"

    result = await run(shell, f"echo '{forged}'; echo after")

    assert result.stdout == f"{forged}\nafter\n"
    assert result.exit_code == 0


async def test_a_timeout_kills_the_command_and_the_session_recovers(shell):
    result = await run(shell, "sleep 30", timeout_ms=500)

    assert result.outcome == "timed_out"
    assert (await run(shell, "echo alive")).stdout == "alive\n"


async def test_a_timeout_kills_the_whole_process_group(shell, tmp_path):
    # a bare kill leaves the children the command spawned running
    marker = tmp_path / "child.pid"
    await run(shell, f"(sleep 30 & echo $! > {marker}); sleep 30", timeout_ms=800)

    child = int(marker.read_text().strip())
    await asyncio.sleep(0.3)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


async def test_a_shell_that_exits_is_reported_and_the_next_command_starts_a_new_one(shell):
    result = await run(shell, "exit")

    assert result.outcome == "died"
    assert (await run(shell, "echo alive")).stdout == "alive\n"


async def test_heavy_output_on_both_streams_is_captured_in_full(shell):
    """Well past a pipe buffer on both streams at once, interleaved.

    asyncio drains both pipes into memory as the loop runs, so this does not
    deadlock however the two reads are ordered — but it is the case where a
    hand-rolled read loop would lose data or stall, so it is worth pinning."""
    filler = "x" * 200
    command = f'for i in $(seq 1 800); do echo "o${{i}}{filler}"; echo "e${{i}}{filler}" >&2; done'

    result = await asyncio.wait_for(run(shell, command, timeout_ms=30_000), timeout=25)

    assert result.exit_code == 0
    assert result.stdout.count("\n") == 800
    assert result.stderr.count("\n") == 800
    assert len(result.stderr) > 128 * 1024  # past any plausible pipe buffer


# ── output fidelity ──────────────────────────────────────────────────────────


async def test_a_command_with_no_trailing_newline_keeps_none(shell):
    assert (await run(shell, "printf 'bare'")).stdout == "bare"


async def test_a_command_with_a_trailing_newline_keeps_it(shell):
    assert (await run(shell, "printf 'a\\nb\\n'")).stdout == "a\nb\n"


async def test_no_output_is_empty_not_a_stray_newline(shell):
    assert (await run(shell, "true")).stdout == ""


async def test_a_compound_command_is_one_unit(shell):
    # the braces have to wrap the whole thing, or the redirect and `$?` cover
    # only the last part of `a && b`
    assert (await run(shell, "false && echo unreachable")).exit_code == 1
    assert (await run(shell, "cd / && pwd")).stdout == "/\n"


# ── lifecycle ────────────────────────────────────────────────────────────────


async def test_the_shell_starts_lazily(tmp_path):
    session = PersistentShell(tmp_path)

    assert not session.running

    await run(session, "true")
    assert session.running
    await session.close()


async def test_close_ends_the_process(tmp_path):
    session = PersistentShell(tmp_path)
    await run(session, "true")
    pid = session._process.pid

    await session.close()

    assert not session.running
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_closing_twice_is_harmless(tmp_path):
    session = PersistentShell(tmp_path)
    await run(session, "true")

    await session.close()
    await session.close()

    assert not session.running


async def test_commands_serialize_rather_than_interleave(shell):
    # two writes into one stdin produce a shell executing neither
    results = await asyncio.gather(
        run(shell, "echo first"),
        run(shell, "echo second"),
        run(shell, "echo third"),
    )

    assert sorted(r.stdout for r in results) == ["first\n", "second\n", "third\n"]


# ── the traps the first version fell into ────────────────────────────────────


async def test_a_single_line_over_the_stream_limit_is_captured_not_fatal(shell):
    """One `cat` of a minified bundle. `readline()` is capped at 64 KiB and
    raises past it, and that exception used to escape `run()` and leave the
    other reader attached to the pipe for the life of the session."""
    result = await run(shell, "python3 -c \"print('y' * 200000)\"", timeout_ms=20_000)

    assert (len(result.stdout), result.exit_code, result.outcome) == (200_001, 0, "completed")
    assert (await run(shell, "echo alive")).stdout == "alive\n"


async def test_a_long_line_on_stderr_is_captured_too(shell):
    result = await run(shell, "python3 -c \"import sys; sys.stderr.write('z' * 200000)\"", timeout_ms=20_000)

    assert (len(result.stderr), result.exit_code) == (200_000, 0)


async def test_output_from_a_background_job_is_not_credited_to_the_next_command(shell):
    """The marker ends the read; it does not stop a backgrounded writer, which
    still holds the shell's stdout. Without a START marker its output sits in
    the pipe and is read as the beginning of the next command's."""
    await run(shell, "(sleep 0.3; echo STRAGGLER) & echo started")
    await asyncio.sleep(1.0)

    assert (await run(shell, "echo hello")).stdout == "hello\n"


async def test_the_login_shell_is_not_used(monkeypatch, tmp_path):
    """The wrapper is Bourne-specific: under fish or tcsh no marker is ever
    printed and every command sits until its timeout."""
    monkeypatch.setenv("SHELL", "/bin/tcsh")
    session = PersistentShell(tmp_path)

    assert session.shell in ("/bin/bash", "/bin/sh")
    await session.close()


async def test_a_shell_binary_that_does_not_exist_is_reported(tmp_path):
    session = PersistentShell(tmp_path, shell=str(tmp_path / "no-such-shell"))

    result = await run(session, "echo hi")

    assert (result.exit_code, result.outcome) == (None, "died")
    assert "could not start" in result.stderr


async def test_a_cancelled_command_returns_what_it_printed(shell):
    token = CancellationToken()
    started = asyncio.get_running_loop().call_later(0.6, token.cancel)

    result = await shell.run("echo early; sleep 30", timeout_ms=20_000, cancellation_token=token)
    started.cancel()

    assert (result.stdout, result.exit_code, result.outcome) == ("early\n", None, "cancelled")


async def test_a_cancelled_command_leaves_a_session_the_next_command_can_use(shell):
    token = CancellationToken()
    asyncio.get_running_loop().call_later(0.4, token.cancel)

    await shell.run("sleep 30", timeout_ms=20_000, cancellation_token=token)

    assert (await run(shell, "echo alive")).stdout == "alive\n"


async def test_a_timed_out_command_returns_what_it_printed(shell):
    result = await run(shell, "echo partial; sleep 30", timeout_ms=800)

    assert (result.stdout, result.outcome) == ("partial\n", "timed_out")


def test_a_shell_reused_across_event_loops_starts_a_new_process(tmp_path):
    """An embedder that calls `asyncio.run` once per turn. The cached process
    still reports `returncode is None`, but its pipes belong to the closed
    loop, so reusing the handle raised `Future attached to a different loop`
    on every later command."""
    session = PersistentShell(tmp_path)

    first = asyncio.run(session.run("echo one", timeout_ms=10_000))
    second = asyncio.run(session.run("echo two", timeout_ms=10_000))
    asyncio.run(session.close())

    assert (first.stdout, second.stdout) == ("one\n", "two\n")
    assert (first.outcome, second.outcome) == ("completed", "completed")
