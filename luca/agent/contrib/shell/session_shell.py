"""A shell process that outlives the command.

Anthropic's bash tool is specified as a persistent session: the working
directory, environment variables and background processes survive between
calls, and a `restart` starts clean. luca's own `bash` spawns a fresh
subprocess per call, which is a different contract — a model told state
persists will `cd` and expect the next command to land there.

The protocol is a pair of sentinels. Each command is written to the live
shell's stdin between a start marker and an end marker carrying `$?`; the
reader discards everything up to the start marker and stops at the end one.
stdout and stderr each get their own markers so the two stay distinguishable.

Four details are load-bearing:

- **The tokens are fresh per command.** A command that prints something
  marker-shaped would otherwise end the read early, and the command's own
  output is exactly where an attacker-controlled string arrives.
- **There is a START marker, not just an end one.** A backgrounded process
  keeps the shell's stdout after the foreground command returns, so anything
  it writes between calls is sitting in the pipe when the next command runs.
  Without a start marker that output is read as the next command's, complete
  with its exit code. It is dropped instead: it belongs to neither call, and
  a model that wants a background job's log can redirect it to a file.
- **The reader takes CHUNKS, not lines.** `readline()` is bounded by the
  stream limit (64 KiB) and raises once one line exceeds it, which `cat` of a
  minified bundle does immediately. The exception escaped `run()` and left the
  sibling reader attached to the pipe, wedging the session for good.
- **The shell is `/bin/bash`, never `$SHELL`.** The wrapper is Bourne-specific.
  Under fish or tcsh none of it parses, no marker is ever printed, and every
  command sits until its timeout.

The command's stdin is `/dev/null` rather than the shell's. The shell's stdin
is the control channel, and a command that reads it eats what comes next: pipe
a script into a bare `zsh` or `bash` and a `cat` in it swallows the following
line instead of running it. It does not reproduce through this protocol — each
command is written together with its markers in one go, so the shell has
buffered all of it before the command starts — but that is a property of the
shell's input buffering, not a guarantee. The redirect makes it independent of
both.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
from dataclasses import dataclass

from luca.agent.core import CancellationToken

_READ_CHUNK = 65536

# The protocol is Bourne-specific, so the user's login shell is not a
# candidate. `/bin/sh` is the fallback for images that ship no bash.
_SHELL_CANDIDATES = ("/bin/bash", "/bin/sh")


def default_shell() -> str:
    for candidate in _SHELL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return _SHELL_CANDIDATES[-1]


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
        self.shell = shell or default_shell()
        self._process: asyncio.subprocess.Process | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def _ensure_started(self) -> asyncio.subprocess.Process:
        if self._alive():
            return self._process
        self._discard()
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
        self._loop = asyncio.get_running_loop()
        return self._process

    def _alive(self) -> bool:
        """Live AND usable from here.

        A process from a closed event loop still reports `returncode is None`,
        but its pipes and futures belong to that loop: an embedder that calls
        `asyncio.run` once per turn would get `Future attached to a different
        loop` on the second command, forever, because the dead handle is
        cached. Treating it as ended sends it down the path that already
        exists for a shell that exited."""
        if self._process is None or self._process.returncode is not None:
            return False
        with contextlib.suppress(RuntimeError):
            return self._loop is asyncio.get_running_loop()
        return False

    def _discard(self) -> None:
        """Drop a handle this loop cannot drive, without awaiting it.

        A shell left over from a previous event loop is a real running process
        whose transport belongs to a loop that is closed. `close()` cannot
        help — it awaits — and asyncio offers nothing else, because from its
        point of view this is the embedder failing to clean up before the loop
        ended. So this is done by hand: signal the group, reap the zombie,
        close the pipes. All plain syscalls, none of which care which loop
        owns what.

        Reaching into `_transport` / `_proc` / `_pipe` is the price of there
        being no public door. Guarded end to end, so a Python release that
        moves them degrades to what happens without this — a warning and a
        leaked descriptor — rather than taking the tool down."""
        process, self._process = self._process, None
        self._loop = None
        if process is None:
            return
        if process.returncode is None:
            _kill_group(process)
        with contextlib.suppress(Exception):
            _release_handles(process)

    async def restart(self) -> None:
        """Kill the session and let the next command start a fresh one. The
        working directory, the environment and any background process are
        gone, which is what the provider's `restart` promises."""
        await self.close()

    async def close(self) -> None:
        if not self._alive():
            # Either already gone, or owned by a loop that has closed — which
            # cannot be awaited, so it is reaped by hand instead.
            self._discard()
            return
        process, self._process = self._process, None
        self._loop = None
        _kill_group(process)
        with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError, RuntimeError):
            await asyncio.wait_for(process.wait(), timeout=5)

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    # ── running a command ────────────────────────────────────────────────────

    async def run(
        self,
        command: str,
        *,
        timeout_ms: int,
        cancellation_token: CancellationToken | None = None,
    ) -> ShellResult:
        async with self._lock:
            return await self._run_locked(
                command,
                timeout_ms=timeout_ms,
                cancellation_token=cancellation_token,
            )

    async def _run_locked(
        self,
        command: str,
        *,
        timeout_ms: int,
        cancellation_token: CancellationToken | None,
    ) -> ShellResult:
        try:
            process = await self._ensure_started()
        except OSError as error:
            # A shell binary that is not there. Reported, not raised: every
            # other way a command fails to run is a result too.
            return ShellResult("", f"could not start {self.shell}: {error}", None, "died")

        start, end = secrets.token_hex(16), secrets.token_hex(16)
        script = _wrap(command, start, end)
        try:
            process.stdin.write(script.encode("utf-8"))
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            # The shell died between commands — `exit`, or something killed it.
            await self.close()
            return ShellResult("", "the shell session ended; it will restart on the next command", None, "died")

        # Owned by the caller so partial output survives a timeout or a cancel,
        # which is the difference between telling the model what the command
        # managed to print and telling it nothing.
        out_buffer, err_buffer = bytearray(), bytearray()
        try:
            code = await self._await_read(
                process,
                start,
                end,
                out_buffer,
                err_buffer,
                timeout_ms=timeout_ms,
                cancellation_token=cancellation_token,
            )
        except TimeoutError:
            # The session is unusable: the command still owns the shell's
            # stdout. Kill it and start clean rather than leave every later
            # command reading someone else's output.
            await self.close()
            return ShellResult(_decode(out_buffer), _decode(err_buffer), None, "timed_out")
        except _Cancelled:
            await self.close()
            return ShellResult(_decode(out_buffer), _decode(err_buffer), None, "cancelled")
        except asyncio.CancelledError:
            await self.close()
            raise
        except Exception as error:
            # A reader failing leaves the pipe in an unknown state. Closing is
            # what keeps it from poisoning every later command in the session.
            await self.close()
            return ShellResult(_decode(out_buffer), f"{_decode(err_buffer)}\n{error}".strip(), None, "died")
        if code is None:
            await self.close()
            return ShellResult(_decode(out_buffer), _decode(err_buffer), None, "died")
        return ShellResult(_decode(out_buffer), _decode(err_buffer), code, "completed")

    async def _await_read(
        self,
        process: asyncio.subprocess.Process,
        start: str,
        end: str,
        out_buffer: bytearray,
        err_buffer: bytearray,
        *,
        timeout_ms: int,
        cancellation_token: CancellationToken | None,
    ) -> int | None:
        """The read, bounded by the deadline and by the user's cancel.

        Cancelling closes the session rather than killing just the command:
        the command runs IN the shell, not in a child of its own, so there is
        nothing to signal that leaves the shell standing. The caller says so
        in its result, which is the part that matters — a silently reset `cd`
        is what makes the next command land somewhere else."""
        read = asyncio.ensure_future(self._read_until(process, start, end, out_buffer, err_buffer))
        waiters: list[asyncio.Future] = [read]
        cancel_wait = None
        if cancellation_token is not None:
            cancel_wait = asyncio.ensure_future(cancellation_token.wait_cancelled())
            waiters.append(cancel_wait)
        try:
            done, _ = await asyncio.wait(
                waiters,
                timeout=timeout_ms / 1_000,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            if cancel_wait is not None and not cancel_wait.done():
                cancel_wait.cancel()
        if read in done:
            return read.result()
        await _abandon(read)
        if cancel_wait is not None and cancel_wait in done:
            raise _Cancelled
        raise TimeoutError

    async def _read_until(
        self,
        process: asyncio.subprocess.Process,
        start: str,
        end: str,
        out_buffer: bytearray,
        err_buffer: bytearray,
    ) -> int | None:
        """Both streams, each between its own pair of markers.

        Awaited together rather than one after the other. Order happens not to
        matter — asyncio's transport drains both pipes into memory as the loop
        runs, so neither fills and blocks the command — but expressing it as
        one wait says what is true: the two markers are independent and either
        can arrive first."""
        out = asyncio.ensure_future(_read_stream(process.stdout, start, end, out_buffer))
        err = asyncio.ensure_future(_read_stream(process.stderr, start, end, err_buffer))
        try:
            code, _ = await asyncio.gather(out, err)
        except BaseException:
            # Cancel BOTH before awaiting either. Awaiting first re-raises the
            # failure that got us here and the sibling is never cancelled at
            # all, left pending on the pipe for the life of the session.
            for task in (out, err):
                task.cancel()
            await asyncio.gather(out, err, return_exceptions=True)
            raise
        return code


class _Cancelled(Exception):
    """The user cancelled; not an error, and never seen outside this module."""


async def _abandon(task: asyncio.Future) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


def _wrap(command: str, start: str, end: str) -> str:
    """The command, its stdin detached, between the markers.

    The braces keep a compound command (`a && b`) one unit, so the redirect
    and the exit code cover all of it. `$?` after the closing brace is the
    group's status."""
    return (
        f'printf "\\n{start}\\n"\n'
        f'printf "\\n{start}\\n" >&2\n'
        f"{{\n{command}\n}} </dev/null\n"
        f"__code=$?\n"
        f'printf "\\n{end} %s\\n" "$__code"\n'
        f'printf "\\n{end}\\n" >&2\n'
    )


async def _read_stream(
    stream: asyncio.StreamReader,
    start: str,
    end: str,
    buffer: bytearray,
) -> int | None:
    """Everything between the two markers, left in `buffer`, plus the exit code
    the end marker carries. `None` means the stream ended first — the shell
    died, or the end marker arrived without a code.

    Whatever arrives before the start marker is dropped: it was written
    between calls by a background job the previous command left running, and
    crediting it to this command is how a model ends up reading a dev
    server's log as the output of its `git status`."""
    opened = await _scan(stream, start.encode(), buffer)
    if not opened:
        return None
    # Everything through the marker line belongs to the previous command.
    del buffer[: opened[1]]
    closed = await _scan(stream, end.encode(), buffer)
    if not closed:
        return None
    trailer = bytes(buffer[closed[0] + len(start) : closed[1]]).strip()
    # Trim the marker line back off, then the newline its `printf` opened with
    # — that one is the marker's, not the command's, so a command whose output
    # ended without a newline keeps ending without one. Done HERE rather than
    # on decode so a partial read, where no marker ever arrived, keeps every
    # byte the command actually produced.
    del buffer[closed[0] :]
    if buffer.endswith(b"\n"):
        del buffer[-1:]
    return int(trailer) if trailer.isdigit() else None


async def _scan(
    stream: asyncio.StreamReader,
    marker: bytes,
    buffer: bytearray,
) -> tuple[int, int] | None:
    """Read into `buffer` until a whole marker LINE is there. Returns where the
    marker starts and where its line ends, or None if the stream ended first.

    Chunks, not lines: `readline()` is bounded by the stream limit and raises
    on the first line past it, which one `cat` of a bundled asset produces."""
    scanned = 0
    while True:
        found = buffer.find(marker, scanned)
        if found != -1:
            tail = buffer.find(b"\n", found + len(marker))
            if tail != -1:
                return found, tail + 1
            scanned = found
        else:
            # A marker can straddle two chunks, so rescan its own length back.
            scanned = max(0, len(buffer) - len(marker))
        chunk = await stream.read(_READ_CHUNK)
        if not chunk:
            return None
        buffer += chunk


def _decode(chunks: bytearray) -> str:
    return bytes(chunks).decode("utf-8", errors="replace")


def _release_handles(process: asyncio.subprocess.Process) -> None:
    """Reap the process and close its pipes, for a handle whose loop is gone.
    See `PersistentShell._discard` for why this is done by hand."""
    transport = process._transport  # type: ignore[attr-defined]
    popen = transport._proc
    with contextlib.suppress(ChildProcessError, OSError):
        os.waitpid(popen.pid, 0)
    popen.returncode = -9
    for pipe in (popen.stdin, popen.stdout, popen.stderr):
        if pipe is not None:
            with contextlib.suppress(OSError):
                pipe.close()
    for stream in (process.stdin, process.stdout, process.stderr):
        pipe_transport = getattr(stream, "_transport", None)
        if pipe_transport is None:
            continue
        pipe_transport._closing = True
        raw = getattr(pipe_transport, "_pipe", None)
        if raw is not None:
            with contextlib.suppress(OSError):
                raw.close()
            pipe_transport._pipe = None
    # Last, or the transport's own finalizer reopens the question by warning
    # about a subprocess that is by now reaped.
    transport._returncode = -9
    transport._closed = True


def _kill_group(process: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(os.getpgid(process.pid), 9)
