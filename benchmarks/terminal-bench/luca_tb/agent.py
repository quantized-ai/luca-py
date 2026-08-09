"""`LucaAgent` — luca as a Harbor installed agent, written against harbor 0.20.

Runs on the host: installs the driver into the task container, invokes it once,
reads the trajectory back. `luca-ai` is unpublished, so install goes through a
wheel built here and uploaded.
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

INSTALL_DIR = PurePosixPath("/opt/luca")
VENV_PYTHON = INSTALL_DIR / "venv" / "bin" / "python"

# /logs/agent is mounted back to the trial directory on the host.
SESSION_FILENAME = "session.json"
STDOUT_FILENAME = "luca.txt"

UV_INSTALL_URL = "https://astral.sh/uv/0.7.13/install.sh"

# luca needs >= 3.11; uv fetches a managed CPython when the image ships less.
VENV_PYTHON_VERSION = "3.11"

# "Did not finish" rather than "broke" — see _interpret_exit_code.
INCOMPLETE_EXIT_CODES = {
    EXIT_BLOCKED: "blocked on an approval gate",
    EXIT_TIMEOUT: "wall-clock timeout",
}

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent.parent


class LucaAgent(BaseInstalledAgent):
    """luca driven headlessly inside the task container."""

    # We write an AgentSession, not Harbor's trajectory format.
    SUPPORTS_ATIF: bool = False

    CLI_FLAGS: ClassVar[list[CliFlag]] = [
        CliFlag("max_steps", cli="--max-steps", type="int", default=200),
        # Named `reasoning_effort` because the leaderboard keys rows on that
        # exact kwarg; any other name records the effort as "none".
        CliFlag("reasoning_effort", cli="--reasoning", type="str"),
        # 0 = no driver clock. Each task declares its own agent.timeout_sec and
        # Harbor enforces it; a second ceiling could only ever be smaller.
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
        """`LUCA_WHEEL` when set, else built now — a stale `dist/` would
        silently benchmark last week's code."""
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
        """ripgrep is not optional: luca's `glob` and `grep` shell out to `rg`."""
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
        """uv lands in ~/.local/bin, off PATH in a non-login shell."""
        return (
            'if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; '
            'else export PATH="$HOME/.local/bin:$PATH"; fi;'
        )

    async def _upload_driver(self, environment: BaseEnvironment) -> None:
        install_dir = shlex.quote(str(INSTALL_DIR))
        # root owns /opt; hand it over so the venv builds without sudo.
        owner = environment.default_user
        chown = f" && chown -R {shlex.quote(str(owner))} {install_dir}" if owner is not None else ""
        await self.exec_as_root(environment, command=f"mkdir -p {install_dir}{chown}")

        wheel = self.wheel_path
        await environment.upload_file(wheel, str(INSTALL_DIR / wheel.name))
        # runner.py imports only luca + stdlib, so it ships as a loose file.
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
                # Prefer an interpreter the image already has. Asking uv for a
                # bare version downloads a managed CPython, which blew the 360s
                # agent-setup budget on heavy ML images.
                "PY=; for c in python3.13 python3.12 python3.11; do "
                'if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi; done; '
                f'uv venv --python "${{PY:-{VENV_PYTHON_VERSION}}}" {venv} && '
                f"uv pip install --python {python} {wheel} pyyaml && "
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

        # Not exec_as_agent: the exit code carries meaning and that raises.
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
                "--",  # an instruction starting with a dash is not a flag
                shlex.quote(instruction),
                f"2>&1 | tee {shlex.quote(str(stdout_log))}",
            ]
        )

    def _interpret_exit_code(self, result: Any, command: str, context: AgentContext) -> None:
        """A failed task or a broken run?

        Not finishing is an ordinary outcome the verifier should still score as
        zero; raising would skip verification and inflate `n_errored_trials`.
        A genuine crash does raise, so `--retry-include` can act on it."""
        code = result.return_code
        if code == 0:
            return
        outcome = INCOMPLETE_EXIT_CODES.get(code)
        if outcome is None:
            raise self._classify_exec_error(command, result)
        self.logger.warning(f"luca did not finish: {outcome}")
        context.metadata = {**(context.metadata or {}), "luca_outcome": outcome}

    def _model_env(self, provider: str) -> dict[str, str]:
        """The key, forwarded under the name luca's provider reads (the last
        candidate), which is what makes `LUCA_API_KEY` a cross-provider override."""
        candidates = api_key_env_names(provider)
        target = candidates[-1]
        env = dict(self.resolve_env_vars())
        for name in candidates:
            value = self._get_env(name)
            if value is not None:
                env[target] = value
                return env
        if target == "LUCA_API_KEY":
            return env  # a provider needing no key, e.g. local Ollama
        raise ValueError(f"No API key for provider {provider!r}. Set one of: {', '.join(candidates)}")

    async def _workspace(self, environment: BaseEnvironment) -> str:
        """The task's WORKDIR, so tools are rooted where the verifier looks."""
        result = await environment.exec(command="pwd")
        found = (result.stdout or "").strip()
        return found or "/app"

    # ── after the run ────────────────────────────────────────────────────────

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Best-effort: a real reward must not be lost to unparseable accounting."""
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
