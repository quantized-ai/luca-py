"""The persistent bash session behind `anthropic_bash_20250124`.

Anthropic's `bash_20250124` is declared by TYPE — `{"type": "bash_20250124",
"name": "bash"}` and nothing else. There is no description field on a native
declaration, so the model arrives already knowing the tool, INCLUDING the half
of the contract we cannot restate on the wire: one bash process stays alive
across calls, and every command runs inside it. A tool that spawns
`bash -c <command>` per call satisfies neither half and has no way to say so —
the model `cd`s, writes a relative path, and the file lands in the workspace
while its confirming `ls` agrees with the mistake.

`ShellSession` is that process. `ShellSessionPool` hands out one per
conversation LINEAGE: a shell is a serial pipe, so two subagents sharing one
would interleave their `cd`s and each other's output, exactly the reason
`FileReadTracker` keys by conversation. Compaction is the one thing that
changes a conversation's id without restarting anything, so the pool keys on
the lineage ROOT and the same shell survives it.

Nothing here is serialized. The process lives as long as the runner; a restart
— a timeout, a cancel, a bare `exit`, a resumed session — means a new process,
which means an empty shell. Rebuilding one from a stored snapshot would mostly
restore the USER's environment rather than the agent's, and would write
whatever `source .env` put in scope into a long-lived session file. Instead the
shell starts empty and `ShellAccessPlugin` tells the model so.

## The mechanism

Each command is written to the live shell's stdin as:

    . '<script>' < /dev/null
    __luca_rc=$?
    printf '\\n%s %s\\n%s\\n' '<sentinel>' "$__luca_rc" "$PWD"

- The command text goes in a FILE and is SOURCED. `.` runs it in this process,
  so `cd` and `export` persist, while the file keeps heredocs, newlines and
  quoting exactly as the model wrote them.
- `< /dev/null` covers the whole sourced script, so a bare `cat` (or `python`,
  or an `ssh` password prompt) reads EOF instead of eating the next command off
  the same pipe we drive the shell with.
- The sentinel is a fresh uuid per command, so output that quotes one cannot
  end its own read early. It carries the exit code (reported as
  `metadata["exit"]`, and what `is_error` derives from) and `$PWD` (the
  approval prompt has to name the directory the next command will actually run
  in, not the workspace the model left three calls ago).

Cancellation and timeouts kill the shell's process group. A command that
persists `cd` necessarily runs IN the shell's own process, so it has no group
of its own to kill; the runaway command and its children go with the shell, and
the next call gets a new one.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from luca.agent.core import AgentSession, CancellationToken, ToolExecution

from .tools import ShellToolError, _kill_process_group

READ_SIZE = 65_536
BASH = "/bin/bash"


@dataclass(frozen=True)
class CommandResult:
    """One command as the shell reported it. `outcome` is the vocabulary
    `BashTool._render` renders — `completed`, `timed_out`, `cancelled` — plus
    `shell_died`, which only a persistent shell can produce."""

    output: str
    exit_code: int | None
    outcome: str


class _ShellDied(Exception):
    """The shell's stdout hit EOF: the process is gone (a bare `exit`, a
    signal, an `exec` that replaced it). Carries the output that arrived
    before it went."""

    def __init__(self, partial: str) -> None:
        super().__init__("the shell exited")
        self.partial = partial


class _SentinelReader:
    """Reads one command's output off the shell's stdout, stopping at the
    sentinel line the driver prints.

    Only a short tail stays searchable, so a large output is appended to a list
    rather than re-joined on every read. `partial` is what has arrived so far —
    what a timeout or a cancel reports."""

    def __init__(self, stream: asyncio.StreamReader, sentinel: str) -> None:
        self._stream = stream
        self._sentinel = sentinel
        self._chunks: list[str] = []
        self._buffer = ""

    @property
    def partial(self) -> str:
        return "".join(self._chunks) + self._buffer

    async def read(self) -> tuple[str, int, str]:
        """(output, exit code, cwd) for the command that just ran."""
        keep = len(self._sentinel) + 64
        while True:
            index = self._buffer.find(self._sentinel)
            if index >= 0:
                return await self._finish(index)
            if len(self._buffer) > keep:
                self._chunks.append(self._buffer[:-keep])
                self._buffer = self._buffer[-keep:]
            self._buffer += await self._next()

    async def _finish(self, index: int) -> tuple[str, int, str]:
        head, tail = self._buffer[:index], self._buffer[index:]
        while tail.count("\n") < 2:  # "<sentinel> <rc>\n<cwd>\n"
            tail += await self._next()
        status, _, rest = tail.partition("\n")
        cwd, _, _ = rest.partition("\n")
        # The driver's own leading newline guarantees the sentinel starts a
        # line; dropping it back off is what makes a command's output exact,
        # trailing newline or not.
        output = ("".join(self._chunks) + head).removesuffix("\n")
        return output, int(status.split()[-1]), cwd

    async def _next(self) -> str:
        data = await self._stream.read(READ_SIZE)
        if not data:
            raise _ShellDied(self.partial)
        return data.decode("utf-8", errors="replace")


class ShellSession:
    """One long-lived `bash`, spawned on first use.

    `cwd` is where the NEXT command will run: the shell reports `$PWD` with
    every command, and the approval prompt reads it (`build_permission_requests`
    is synchronous, so this has to be a plain attribute)."""

    def __init__(
        self,
        workdir: str | os.PathLike[str],
        shell: str = BASH,
    ) -> None:
        self.workdir = Path(workdir)
        self.shell = shell
        self.cwd = str(self.workdir)
        self._process: asyncio.subprocess.Process | None = None
        self._scripts: tempfile.TemporaryDirectory | None = None
        self._used = False
        self._lock = asyncio.Lock()

    @property
    def fresh(self) -> bool:
        """Whether the shell this session would use next has run nothing yet —
        including the case where no process has been started at all. The
        condition behind the plugin's "your bash session was restarted" notice,
        which is spent by USE rather than by being emitted: the system prompt is
        re-sent whole every call, so a notice the model read and never acted on
        must still be there on the call where it finally runs a command."""
        return not self._used

    async def run(
        self,
        command: str,
        *,
        timeout_ms: int,
        cancellation_token: CancellationToken,
    ) -> CommandResult:
        # The write-read cycle is one critical section. Conversation-keyed
        # state usually needs no lock because dispatch is sequential, but that
        # is a runner CHOICE, and a single pipe answers a parallel one with
        # interleaved writes and garbage output rather than a stale read.
        async with self._lock:
            process = await self._ensure()
            sentinel = f"__LUCA_DONE_{uuid.uuid4().hex}__"
            script = Path(self._scripts.name) / f"{sentinel}.sh"
            script.write_text(command, encoding="utf-8")
            try:
                return await self._drive(process, script, sentinel, timeout_ms, cancellation_token)
            finally:
                with contextlib.suppress(OSError):
                    script.unlink()

    async def restart(self) -> None:
        """The wire tool's `restart: true`. Teardown IS the restart — the next
        command spawns a new shell, so there is nothing to start eagerly. Takes
        the lock, because a restart is a tool call like any other and must not
        cut across one that is mid-flight."""
        async with self._lock:
            await self._kill()

    async def aclose(self) -> None:
        """Kill the shell and drop its scratch directory.

        A hard process kill needs nothing: the shell reads commands from a pipe
        only we hold, so it gets EOF and exits on its own. Nothing runs
        `finally` there either, which is why the graceful paths (quit, `/clear`,
        `/resume`, fork) have to call this.

        Deliberately does NOT take the lock, unlike `restart`: the user is
        leaving, and a command with two minutes left on its timeout must not
        hold the door. A read in flight sees EOF and reports `shell_died`,
        which is what happened."""
        await self._kill()

    # ── the process ─────────────────────────────────────────────────────────

    async def _ensure(self) -> asyncio.subprocess.Process:
        if self._process is not None and self._process.returncode is None:
            return self._process
        await self._kill()
        try:
            self._process = await asyncio.create_subprocess_exec(
                self.shell,
                "-s",  # read commands from stdin, non-interactive
                cwd=self.workdir,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            raise ShellToolError(f"Failed to start shell: {error}") from error
        self._scripts = tempfile.TemporaryDirectory(prefix="luca_shell_")
        return self._process

    async def _kill(self) -> None:
        process, self._process = self._process, None
        self._used = False
        self.cwd = str(self.workdir)
        if process is not None:
            _kill_process_group(process)
            await process.wait()
            if process.stdin is not None:
                process.stdin.close()
        if self._scripts is not None:
            self._scripts.cleanup()
            self._scripts = None

    # ── one command ─────────────────────────────────────────────────────────

    async def _drive(
        self,
        process: asyncio.subprocess.Process,
        script: Path,
        sentinel: str,
        timeout_ms: int,
        cancellation_token: CancellationToken,
    ) -> CommandResult:
        driver = (
            f". {shlex.quote(str(script))} < /dev/null\n"
            f"__luca_rc=$?\n"
            f'printf \'\\n%s %s\\n%s\\n\' {shlex.quote(sentinel)} "$__luca_rc" "$PWD"\n'
        )
        try:
            process.stdin.write(driver.encode())
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            # The shell died between the previous command and this one, so
            # nothing ran: an empty result, and the next call respawns.
            await self._kill()
            return CommandResult("", None, "shell_died")
        self._used = True
        reader = _SentinelReader(process.stdout, sentinel)
        read = asyncio.create_task(reader.read())
        cancelled = asyncio.create_task(cancellation_token.wait_cancelled())
        try:
            done, _ = await asyncio.wait(
                {read, cancelled},
                timeout=timeout_ms / 1_000,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if read in done:
                output, exit_code, cwd = read.result()
                self.cwd = cwd or self.cwd
                return CommandResult(output, exit_code, "completed")
            outcome = "cancelled" if cancelled in done else "timed_out"
        except _ShellDied as error:
            await self._kill()
            return CommandResult(error.partial, None, "shell_died")
        except asyncio.CancelledError:
            await self._kill()
            raise
        finally:
            for task in (cancelled, read):
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError, _ShellDied):
                    await task
        partial = reader.partial
        await self._kill()
        return CommandResult(partial, None, outcome)


class ShellSessionPool:
    """One `ShellSession` per conversation lineage, all under one workspace.

    Owned by `ShellAccessPlugin`, which is also the only thing that closes it.
    A session is created on demand and holds no OS resource until its first
    command, so asking for one costs nothing."""

    def __init__(
        self,
        workdir: str | os.PathLike[str],
        shell: str = BASH,
    ) -> None:
        self.workdir = Path(workdir)
        self.shell = shell
        self._sessions: dict[str, ShellSession] = {}

    def get(self, session: AgentSession, conversation_id: str) -> ShellSession:
        key = lineage_root(session, conversation_id)
        shell = self._sessions.get(key)
        if shell is None:
            shell = self._sessions[key] = ShellSession(self.workdir, self.shell)
        return shell

    def is_fresh(self, session: AgentSession, conversation_id: str) -> bool:
        """`ShellSession.fresh` for the shell this conversation would use — and
        True when it has never asked for one, which is the same fact."""
        shell = self._sessions.get(lineage_root(session, conversation_id))
        return shell is None or shell.fresh

    async def aclose(self) -> None:
        sessions, self._sessions = list(self._sessions.values()), {}
        for shell in sessions:
            await shell.aclose()


def lineage(session: AgentSession, conversation_id: str) -> list[str]:
    """`conversation_id` followed by every conversation a compaction archived
    behind it, newest first.

    Compaction is the one thing that changes a conversation's id mid-task
    without restarting anything: it mints a successor over a rewritten path and
    points `previous_conversation_id` at what it archived. A spawned subagent
    has none. Walking that pointer separates the two cases with no hook — the
    shell follows the AGENT, not the id."""
    chain = [conversation_id]
    seen = {conversation_id}
    while (conversation := session.conversations.get(chain[-1])) is not None:
        previous = conversation.previous_conversation_id
        if previous is None or previous in seen:
            break
        chain.append(previous)
        seen.add(previous)
    return chain


def lineage_root(session: AgentSession, conversation_id: str) -> str:
    return lineage(session, conversation_id)[-1]


def called_tool(session: AgentSession, conversation_id: str, tool_name: str) -> bool:
    """Whether this conversation — or a predecessor it compacted — has ever
    called `tool_name`. Newest entry first, because the answer is usually
    recent."""
    for identifier in lineage(session, conversation_id):
        conversation = session.conversations.get(identifier)
        if conversation is None:
            continue
        for node in reversed(conversation.nodes):
            entry = session.entries.get(node)
            if isinstance(entry, ToolExecution) and entry.raw_tool_call.name == tool_name:
                return True
    return False
