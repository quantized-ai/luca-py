"""A shell process that outlives the command.

Anthropic's bash tool is specified as a persistent session: the working
directory, environment variables and background processes survive between
calls, and a `restart` starts clean. luca's own `bash` spawns a fresh
subprocess per call, which is a different contract — a model told state
persists will `cd` and expect the next command to land there.

The protocol is a sentinel. Each command is written to the live shell's stdin
followed by a marker carrying `$?`; the reader stops at the marker and takes
the exit code from it. stdout and stderr each get their own marker so the two
stay distinguishable.

Two details are load-bearing:

- **The token is fresh per command.** A command that prints something
  marker-shaped would otherwise end the read early, and the command's own
  output is exactly where an attacker-controlled string arrives.
- **The command's stdin is `/dev/null`, not the shell's.** The shell's stdin is
  the control channel, and a command that reads it eats what comes next: pipe
  a script into a bare `zsh` or `bash` and a `cat` in it swallows the following
  line instead of running it. It does not reproduce through this protocol —
  each command is written together with its markers in one go, so the shell has
  buffered all of it before the command starts — but that is a property of the
  shell's input buffering, not a guarantee. The redirect makes it independent
  of both.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
from dataclasses import dataclass

SHELL_START_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class ShellResult:
    stdout: str
    stderr: str
    exit_code: int | None  # None when the command did not finish
    outcome: str  # "completed" | "timed_out" | "cancelled" | "died"


class PersistentShell:
    """One long-lived shell. Not safe for concurrent use by design: a lock
    serializes commands, because two interleaved writes to one stdin produce a
    shell that is executing neither of them."""

    def __init__(self, workdir: str | os.PathLike[str], shell: str | None = None) -> None:
        self.workdir = os.fspath(workdir)
        self.shell = shell or os.environ.get("SHELL") or "/bin/bash"
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def _ensure_started(self) -> asyncio.subprocess.Process:
        if self._process is not None and self._process.returncode is None:
            return self._process
        self._process = await asyncio.create_subprocess_exec(
            self.shell,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.workdir,
            # Its own process group, so a timeout kills the command AND
            # anything it spawned rather than orphaning a tree.
            start_new_session=True,
        )
        return self._process

    async def restart(self) -> None:
        """Kill the session and let the next command start a fresh one. The
        working directory, the environment and any background process are
        gone, which is what the provider's `restart` promises."""
        await self.close()

    async def close(self) -> None:
        process, self._process = self._process, None
        if process is None or process.returncode is not None:
            return
        _kill_group(process)
        with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
            await asyncio.wait_for(process.wait(), timeout=5)

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    # ── running a command ────────────────────────────────────────────────────

    async def run(self, command: str, *, timeout_ms: int) -> ShellResult:
        async with self._lock:
            return await self._run_locked(command, timeout_ms=timeout_ms)

    async def _run_locked(self, command: str, *, timeout_ms: int) -> ShellResult:
        process = await self._ensure_started()
        token = secrets.token_hex(16)
        script = _wrap(command, token)
        try:
            process.stdin.write(script.encode("utf-8"))
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            # The shell died between commands — `exit`, or something killed it.
            await self.close()
            return ShellResult("", "the shell session ended; it will restart on the next command", None, "died")

        try:
            stdout, stderr, code = await asyncio.wait_for(
                self._read_until(process, token),
                timeout=timeout_ms / 1_000,
            )
        except TimeoutError:
            # The session is unusable: the command still owns the shell's
            # stdout. Kill it and start clean rather than leave every later
            # command reading someone else's output.
            await self.close()
            return ShellResult("", "", None, "timed_out")
        except asyncio.CancelledError:
            await self.close()
            raise
        if code is None:
            await self.close()
            return ShellResult(stdout, stderr, None, "died")
        return ShellResult(stdout, stderr, code, "completed")

    async def _read_until(
        self,
        process: asyncio.subprocess.Process,
        token: str,
    ) -> tuple[str, str, int | None]:
        """Both streams, each to its own marker.

        Awaited together rather than one after the other. Order happens not to
        matter — asyncio's transport drains both pipes into memory as the loop
        runs, so neither fills and blocks the command — but expressing it as
        one wait says what is true: the two markers are independent and either
        can arrive first."""
        out = asyncio.create_task(_read_stream(process.stdout, token))
        err = asyncio.create_task(_read_stream(process.stderr, token))
        try:
            (stdout, code), (stderr, _) = await asyncio.gather(out, err)
        except BaseException:
            for task in (out, err):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            raise
        return stdout, stderr, code


def _wrap(command: str, token: str) -> str:
    """The command, its stdin detached, followed by the markers.

    The braces keep a compound command (`a && b`) one unit, so the redirect
    and the exit code cover all of it. `$?` after the closing brace is the
    group's status."""
    return f'{{\n{command}\n}} </dev/null\n__code=$?\nprintf "\\n{token} %s\\n" "$__code"\nprintf "\\n{token}\\n" >&2\n'


async def _read_stream(stream: asyncio.StreamReader, token: str) -> tuple[str, int | None]:
    """Everything up to the marker line, plus the exit code when the marker
    carries one. `None` means the stream ended first — the shell died."""
    chunks: list[bytes] = []
    marker = token.encode()
    while True:
        line = await stream.readline()
        if not line:
            return _decode(chunks), None
        if marker in line:
            rest = line.split(marker, 1)[1].strip()
            code = int(rest) if rest.isdigit() else None
            return _decode(chunks), code
        chunks.append(line)


def _decode(chunks: list[bytes]) -> str:
    # The final newline is the one the marker's `printf` added, not the
    # command's, so a command whose output had no trailing newline keeps that.
    return b"".join(chunks).decode("utf-8", errors="replace").removesuffix("\n")


def _kill_group(process: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(os.getpgid(process.pid), 9)
