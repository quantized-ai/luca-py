"""`LucaAgent` — everything about the adapter that does not need a container.

The install and run paths need Docker and a real task, so they are covered by
the staged protocol in the README rather than here. What is here is the wiring
that could be silently wrong without anyone noticing until a whole job has
been paid for: the flags handed to the driver, the key that reaches the model,
and the accounting read back off the trajectory.
"""

import json

import pytest
from harbor.agents.installed.base import (
    ApiRateLimitError,
    BaseInstalledAgent,
    NonZeroAgentExitCodeError,
)
from harbor.models.agent.context import AgentContext

from luca_tb.agent import LucaAgent


@pytest.fixture
def agent(tmp_path):
    return LucaAgent(logs_dir=tmp_path, model_name="openrouter/openai/gpt-5.4-mini")


def test_it_is_a_harbor_installed_agent_with_nothing_left_abstract():
    assert issubclass(LucaAgent, BaseInstalledAgent)
    assert LucaAgent.__abstractmethods__ == frozenset()


def test_the_import_path_is_the_one_the_readme_tells_people_to_use():
    assert LucaAgent.import_path() == "luca_tb.agent:LucaAgent"


def test_the_version_reported_is_lucas_own(agent):
    from luca import __version__

    assert agent.version() == __version__


# ── the flags handed to the driver ───────────────────────────────────────────


def test_the_defaults_are_leaderboard_legal(agent):
    # `--timeout 0` disables the driver's own clock on purpose: every task
    # declares its own agent.timeout_sec and Harbor enforces it, so a second
    # ceiling here could only be the smaller of the two and would hand back
    # failures the task's budget allowed. A non-yolo mode would make the
    # driver exit 2 at the first tool call.
    assert agent.build_cli_flags() == "--max-steps 200 --timeout 0 --permission-mode yolo"


def test_every_knob_is_reachable_through_harbors_ak_kwargs(tmp_path):
    agent = LucaAgent(
        logs_dir=tmp_path,
        model_name="anthropic/claude-opus-4-5",
        max_steps=300,
        timeout=1200,
        reasoning_effort="high",
        subagents=True,
    )

    assert agent.build_cli_flags() == (
        "--max-steps 300 --reasoning high --timeout 1200 --subagents --permission-mode yolo"
    )


def test_the_effort_kwarg_is_named_the_way_the_leaderboard_reads_it(tmp_path):
    # The leaderboard keys rows on kwargs["reasoning_effort"] and renders its
    # Effort column from it. Under any other name the effort records as "none"
    # and two runs at different efforts collapse into one row.
    agent = LucaAgent(logs_dir=tmp_path, model_name="anthropic/claude-opus-4-5", reasoning_effort="high")

    assert "reasoning_effort" in {flag.kwarg for flag in LucaAgent.CLI_FLAGS}
    assert "--reasoning high" in agent.build_cli_flags()


def test_subagents_stay_off_unless_asked_for(agent):
    # Off by default so the first baseline measures one thing, not two.
    assert "--subagents" not in agent.build_cli_flags()


# ── the key that reaches the model ───────────────────────────────────────────


def test_the_providers_own_variable_is_forwarded(agent, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter")

    assert agent._model_env("openrouter") == {"OPENROUTER_API_KEY": "sk-openrouter"}


def test_luca_api_key_overrides_and_arrives_under_the_providers_name(agent, monkeypatch):
    monkeypatch.setenv("LUCA_API_KEY", "sk-override")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ignored")

    assert agent._model_env("anthropic") == {"ANTHROPIC_API_KEY": "sk-override"}


def test_a_missing_key_fails_before_the_container_is_touched(agent, monkeypatch):
    monkeypatch.delenv("LUCA_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="No API key for provider 'openrouter'"):
        agent._model_env("openrouter")


def test_a_provider_that_needs_no_key_is_not_forced_to_have_one(agent, monkeypatch):
    monkeypatch.delenv("LUCA_API_KEY", raising=False)

    assert agent._model_env("ollama") == {}


# ── the command handed to the container ──────────────────────────────────────


def test_the_instruction_is_separated_from_the_flags(agent):
    # Task instructions are arbitrary text. One starting with a dash would be
    # read as a flag without the `--`, and quoting alone does not prevent it.
    command = agent._runner_command("--verbose is not a flag here", "openrouter", "gpt-x", "/app")

    assert command.index(" -- ") > command.index("--max-steps")
    assert command.endswith("2>&1 | tee /logs/agent/luca.txt")


def test_the_trajectory_and_the_log_go_where_harbor_syncs_them_back(agent):
    command = agent._runner_command("do it", "openrouter", "gpt-x", "/app")

    assert "--session-out /logs/agent/session.json" in command
    assert "/logs/agent/luca.txt" in command


def test_the_workspace_and_model_reach_the_driver(agent):
    command = agent._runner_command("do it", "anthropic", "claude-opus-4-5", "/srv/app")

    assert "--model claude-opus-4-5" in command
    assert "--provider anthropic" in command
    assert "--workspace /srv/app" in command


# ── failed task against broken run ───────────────────────────────────────────


class ExecResult:
    def __init__(self, return_code, stdout="", stderr=""):
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


def test_a_clean_finish_records_no_outcome(agent):
    context = AgentContext()

    agent._interpret_exit_code(ExecResult(0), "cmd", context)

    assert context.metadata is None


@pytest.mark.parametrize(
    ("code", "outcome"),
    [(124, "wall-clock timeout"), (2, "blocked on an approval gate")],
)
def test_not_finishing_is_a_failed_task_not_an_errored_trial(agent, code, outcome):
    # The verifier still has to run and score this a zero. Raising would skip
    # verification and count a task luca merely lost as a harness error.
    context = AgentContext()

    agent._interpret_exit_code(ExecResult(code), "cmd", context)

    assert context.metadata == {"luca_outcome": outcome}


def test_a_crash_does_raise_so_it_lands_in_the_error_count(agent):
    with pytest.raises(NonZeroAgentExitCodeError, match="exit 1"):
        agent._interpret_exit_code(ExecResult(1, stderr="ToolExecutionError"), "cmd", AgentContext())


def test_a_provider_rate_limit_keeps_harbors_classification(agent):
    # so that `harbor run --retry-include ApiRateLimitError` can act on it
    with pytest.raises(ApiRateLimitError):
        agent._interpret_exit_code(ExecResult(1, stdout="API Error: rate limit exceeded"), "cmd", AgentContext())


# ── the accounting read back ─────────────────────────────────────────────────


def write_session(logs_dir, payload):
    (logs_dir / "session.json").write_text(json.dumps(payload))


def test_usage_and_metadata_land_on_the_context(agent, tmp_path):
    write_session(
        tmp_path,
        {
            "id": "abc123",
            "entries": {"u1": {}, "a1": {}},
            "usages": {"c1": {"a1": {"input": 1_000, "output": 200, "cache_read": 300}}},
            "session_config": {"llm_config": {"provider": "faux", "model": "unpriced"}},
        },
    )
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.n_input_tokens == 1_300  # harbor counts cache reads as input
    assert context.n_output_tokens == 200
    assert context.n_cache_tokens == 300
    assert context.cost_usd is None  # the catalog does not price this model
    assert context.metadata == {"luca_session_id": "abc123", "luca_entries": 2}


def test_a_missing_trajectory_leaves_the_context_untouched(agent):
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.is_empty()


def test_an_unparseable_trajectory_does_not_fail_the_trial(agent, tmp_path):
    # The task may well have passed; losing the reward over broken accounting
    # would be the worse outcome.
    (tmp_path / "session.json").write_text("{not json")
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.is_empty()
