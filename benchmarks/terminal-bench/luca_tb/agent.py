"""`LucaAgent` — luca as a Harbor installed agent.

Runs on the HOST. Harbor gives it a container per task; it installs the driver
in there, invokes it once with the task instruction, and reads the trajectory
back afterwards.

    harbor run -d terminal-bench/terminal-bench-2-1 \\
        -a luca_tb.agent:LucaAgent \\
        -m openrouter/openai/gpt-5.4-mini

Because `luca-ai` is not published and the repo is private, installation goes
through a wheel built on the host and uploaded, rather than `pip install`.

Written against harbor 0.20. Note that harbor's `main` has since grown
`ensure_system_dependencies` and `ModelConnectionSpec`, neither of which exists
in 0.20 — the system-package block and the key resolution below are the
long-hand versions, and can be swapped for those helpers when the pin moves.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from harbor.agents.installed.base import BaseInstalledAgent, CliFlag, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from luca_tb.mapping import api_key_env_names, context_from_session, parse_model
from luca_tb.runner import EXIT_BLOCKED, EXIT_TIMEOUT

#: Where the driver and its virtualenv live inside the task container. Under
#: /opt rather than the agent's home, so it survives a task that resets HOME
#: and never collides with files the task itself owns.
INSTALL_DIR = PurePosixPath("/opt/luca")
VENV_PYTHON = INSTALL_DIR / "venv" / "bin" / "python"

#: Harbor mounts /logs/agent back to the trial directory on the host, so
#: anything written here is what `populate_context_post_run` can read.
SESSION_FILENAME = "session.json"
STDOUT_FILENAME = "luca.txt"

#: Pinned rather than floating: an installer that changes under us turns one
#: benchmark run into a different benchmark run.
UV_INSTALL_URL = "https://astral.sh/uv/0.7.13/install.sh"

#: The interpreter uv provisions for the venv. luca needs >= 3.11 and task
#: images routinely ship less, so uv fetches a managed CPython when the image's
#: own python is too old.
VENV_PYTHON_VERSION = "3.11"

#: Driver exit codes that mean "did not finish", as opposed to "broke". These
#: are recorded and allowed through so the verifier still scores the task; see
#: `_interpret_exit_code`.
#:
#: Imported from the driver rather than restated, so the two cannot drift. The
#: driver runs in the container and this runs on the host, but they ship
#: together and both sides import cleanly here.
INCOMPLETE_EXIT_CODES = {
    EXIT_BLOCKED: "blocked on an approval gate",
    EXIT_TIMEOUT: "wall-clock timeout",
}

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent.parent


class LucaAgent(BaseInstalledAgent):
    """luca driven headlessly inside the task container."""

    # The driver writes an AgentSession, not Harbor's trajectory format. Token
    # counts and cost still come back through populate_context_post_run.
    SUPPORTS_ATIF: bool = False

    CLI_FLAGS: ClassVar[list[CliFlag]] = [
        CliFlag("max_steps", cli="--max-steps", type="int", default=200),
        # The kwarg is named `reasoning_effort`, not `reasoning`, and that is
        # not cosmetic: the leaderboard groups trials by
        # `(agent, version, model, kwargs["reasoning_effort"])` and renders the
        # Effort column from it. Under any other name the effort silently
        # records as "none" and two runs at different efforts merge into one
        # row. See terminal-bench-2-1 leaderboard/ci/static_analysis.py.
        CliFlag("reasoning_effort", cli="--reasoning", type="str"),
        # 0 disables the driver's own clock, which is what a benchmark wants.
        # Every task declares its own `agent.timeout_sec` and Harbor enforces
        # it; a second ceiling here could only ever be the smaller of the two,
        # so it would hand back failures the task's own budget allowed. Set it
        # for local debugging, never for a scored run.
        CliFlag("timeout", cli="--timeout", type="int", default=0),
        CliFlag("subagents", cli="--subagents", type="bool", default=False),
        CliFlag("permission_mode", cli="--permission-mode", type="str", default="yolo"),
    ]

    def __init__(self, *args: Any, wheel: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._wheel = Path(wheel).expanduser() if wheel else None

    @staticmethod
    def name() -> str:
        return "luca"

    def version(self) -> str | None:
        if self._version:
            return self._version
        from luca import __version__

        return __version__

    # ── install ──────────────────────────────────────────────────────────────

    @property
    def wheel_path(self) -> Path:
        """The luca wheel to install, built on demand.

        `LUCA_WHEEL` wins when set (the Docker image bakes one in), otherwise
        the repo is built here. Built rather than resolved from an existing
        `dist/`: a stale wheel silently benchmarks last week's code, and that
        is the kind of mistake you only notice after the run."""
        if self._wheel is not None:
            return self._wheel
        configured = os.environ.get("LUCA_WHEEL")
        if configured:
            self._wheel = Path(configured).expanduser()
            return self._wheel
        self._wheel = self._build_wheel()
        return self._wheel

    def _build_wheel(self) -> Path:
        out_dir = REPO_ROOT / "dist"
        self.logger.debug(f"Building the luca wheel from {REPO_ROOT}")
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(out_dir)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        wheels = sorted(out_dir.glob("luca_ai-*.whl"), key=lambda path: path.stat().st_mtime)
        if not wheels:
            raise RuntimeError(f"uv build produced no wheel in {out_dir}")
        return wheels[-1]

    async def install(self, environment: BaseEnvironment) -> None:
        await self._install_system_packages(environment)
        await self._install_uv(environment)
        await self._upload_driver(environment)
        await self._create_venv(environment)

    async def _install_system_packages(self, environment: BaseEnvironment) -> None:
        """curl and git for the install itself, ripgrep for the agent.

        ripgrep is not optional: luca's `glob` and `grep` tools shell out to
        `rg` and raise `ripgrep (rg) was not found on PATH` without it, so an
        image missing it costs the agent two of its seven tools and looks like
        a model failure rather than a setup one."""
        await self.exec_as_root(
            environment,
            command=(
                "if command -v apt-get >/dev/null 2>&1; then"
                "  apt-get update && apt-get install -y curl ca-certificates git ripgrep procps;"
                " elif command -v apk >/dev/null 2>&1; then"
                "  apk add --no-cache curl ca-certificates bash git ripgrep procps;"
                " elif command -v dnf >/dev/null 2>&1; then"
                "  dnf install -y curl ca-certificates git ripgrep procps-ng;"
                " elif command -v yum >/dev/null 2>&1; then"
                "  yum install -y curl ca-certificates git ripgrep procps-ng;"
                " else"
                '  echo "no known package manager; assuming curl/git/ripgrep are present" >&2;'
                " fi"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )

    async def _install_uv(self, environment: BaseEnvironment) -> None:
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "if ! command -v uv >/dev/null 2>&1; then"
                f"  curl -LsSf {UV_INSTALL_URL} | sh;"
                " fi && "
                f"{self._uv_on_path()} uv --version"
            ),
        )

    @staticmethod
    def _uv_on_path() -> str:
        """uv's installer drops it in ~/.local/bin, which is not on PATH in a
        non-login shell. Prefix any command that needs uv with this."""
        return (
            'if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; '
            'else export PATH="$HOME/.local/bin:$PATH"; fi;'
        )

    async def _upload_driver(self, environment: BaseEnvironment) -> None:
        install_dir = shlex.quote(str(INSTALL_DIR))
        # Created as root because /opt is root-owned, then handed to the agent
        # user so the venv can be built without sudo.
        owner = environment.default_user
        chown = f" && chown -R {shlex.quote(str(owner))} {install_dir}" if owner is not None else ""
        await self.exec_as_root(environment, command=f"mkdir -p {install_dir}{chown}")

        wheel = self.wheel_path
        await environment.upload_file(wheel, str(INSTALL_DIR / wheel.name))
        # `runner.py` imports only luca and the standard library, so it runs as
        # a loose file — no need to ship the rest of `luca_tb` with it.
        await environment.upload_file(HERE / "runner.py", str(INSTALL_DIR / "runner.py"))
        await environment.upload_file(HERE / "prompt_template.md", str(INSTALL_DIR / "prompt_template.md"))

    async def _create_venv(self, environment: BaseEnvironment) -> None:
        wheel = shlex.quote(str(INSTALL_DIR / self.wheel_path.name))
        venv = shlex.quote(str(INSTALL_DIR / "venv"))
        python = shlex.quote(str(VENV_PYTHON))
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"{self._uv_on_path()} "
                f"uv venv --python {VENV_PYTHON_VERSION} {venv} && "
                f"uv pip install --python {python} {wheel} pyyaml && "
                # Smoke check: prove the interpreter can actually import the
                # agent before a task's worth of tokens is spent finding out.
                f'{python} -c "import luca; print(luca.__version__)"'
            ),
        )

    # ── run ──────────────────────────────────────────────────────────────────

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name:
            raise ValueError("LucaAgent needs a model: pass -m <provider>/<model>")
        provider, model = parse_model(self.model_name)
        command = self._runner_command(instruction, provider, model, await self._workspace(environment))

        # `environment.exec` rather than `exec_as_agent`, because the exit code
        # carries meaning here and `exec_as_agent` raises on anything non-zero.
        # `set -o pipefail` so the `| tee` below cannot mask it.
        result = await environment.exec(
            command=f"set -o pipefail; {command}",
            env=self._model_env(provider),
        )
        self._interpret_exit_code(result, command, context)

    def _runner_command(self, instruction: str, provider: str, model: str, workspace: str) -> str:
        session_out = EnvironmentPaths.agent_dir / SESSION_FILENAME
        stdout_log = EnvironmentPaths.agent_dir / STDOUT_FILENAME
        return " ".join(
            [
                str(VENV_PYTHON),
                str(INSTALL_DIR / "runner.py"),
                f"--model {shlex.quote(model)}",
                f"--provider {shlex.quote(provider)}",
                f"--workspace {shlex.quote(workspace)}",
                f"--session-out {shlex.quote(str(session_out))}",
                f"--append-system-prompt-file {shlex.quote(str(INSTALL_DIR / 'prompt_template.md'))}",
                self.build_cli_flags(),
                # `--` before the instruction: task instructions are arbitrary
                # text and one starting with a dash would otherwise be read as
                # a flag. Quoting alone does not prevent that; this does.
                "--",
                shlex.quote(instruction),
                # stdout is the human-readable trace, and /logs/agent is mounted
                # back to the host, so this is what triage reads first.
                f"2>&1 | tee {shlex.quote(str(stdout_log))}",
            ]
        )

    def _interpret_exit_code(self, result: Any, command: str, context: AgentContext) -> None:
        """Decide whether the driver's exit code is a failed task or a broken run.

        This distinction is the whole reason `run()` does not use
        `exec_as_agent`. An agent that ran out of time or parked on an approval
        gate did not *break* — it just did not finish, which is an ordinary
        benchmark outcome that the verifier should still get to score as a zero.
        Raising instead would skip verification and inflate `n_errored_trials`
        with tasks luca simply lost. A genuine crash does raise, so it lands in
        the error count where it belongs and `--retry-include` can act on it."""
        code = result.return_code
        if code == 0:
            return
        outcome = INCOMPLETE_EXIT_CODES.get(code)
        if outcome is None:
            raise self._classify_exec_error(command, result)
        self.logger.warning(f"luca did not finish: {outcome}")
        context.metadata = {**(context.metadata or {}), "luca_outcome": outcome}

    def _model_env(self, provider: str) -> dict[str, str]:
        """The provider's API key, forwarded into the container.

        Whichever variable holds it on the host, it goes in under the name
        luca's provider actually reads (`OPENROUTER_API_KEY` and so on), which
        is the last name in the list. That is what makes `LUCA_API_KEY` usable
        as a single override across providers."""
        candidates = api_key_env_names(provider)
        target = candidates[-1]
        env = dict(self.resolve_env_vars())
        for name in candidates:
            value = self._get_env(name)
            if value is not None:
                env[target] = value
                return env
        if target == "LUCA_API_KEY":
            # A provider luca knows but that needs no key — a local Ollama, say.
            return env
        raise ValueError(f"No API key for provider {provider!r}. Set one of: {', '.join(candidates)}")

    async def _workspace(self, environment: BaseEnvironment) -> str:
        """The task's own working directory, so the agent's tools are rooted
        where the verifier will look. Falls back to `/app`, harbor's
        convention, when the image declares no WORKDIR."""
        result = await environment.exec(command="pwd")
        found = (result.stdout or "").strip()
        return found or "/app"

    # ── after the run ────────────────────────────────────────────────────────

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Read the trajectory harbor synced back and fill in the numbers.

        Best-effort by design: a trial that produced a real reward must not be
        recorded as an error because its accounting could not be parsed."""
        path = self.logs_dir / SESSION_FILENAME
        if not path.exists():
            self.logger.debug(f"No luca session at {path}; leaving usage unset")
            return
        try:
            session = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            self.logger.debug(f"Could not read the luca session at {path}: {error}")
            return

        totals = context_from_session(session)
        context.n_input_tokens = totals.n_input_tokens
        context.n_output_tokens = totals.n_output_tokens
        context.n_cache_tokens = totals.n_cache_tokens
        context.cost_usd = totals.cost_usd
        context.metadata = {
            **(context.metadata or {}),
            "luca_session_id": session.get("id"),
            "luca_entries": len(session.get("entries") or {}),
        }
