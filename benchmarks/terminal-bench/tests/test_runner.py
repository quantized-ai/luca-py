"""The driver's contact points with luca's core.

These exist because the harness sits outside the `luca` package and consumes
its public models. A core change lands here as a runtime crash inside a task
container, where it costs a whole benchmark run to notice — `reasoning` moving
from an `LLMConfig` field into `model_options` did exactly that, erroring all
10 trials with a validation error before a single token was spent. Constructing
the real objects in a unit test turns that into a red test instead.
"""

from luca.agent.core.models import AgentSession, LLMConfig, RuntimeConfig
from luca.agent.core.runner import AgentSessionRunner
from luca_tb.runner import build_runner, llm_config


def test_a_plain_config_carries_no_model_options():
    assert llm_config("openai/gpt-5.6-luna", "openrouter") == LLMConfig(
        model="openai/gpt-5.6-luna",
        provider="openrouter",
        model_options={},
    )


def test_reasoning_goes_into_model_options_not_onto_the_config():
    # it is a field on neither LLMConfig nor its validator; passing it as a
    # keyword raises extra_forbidden and kills the run before the first call
    assert llm_config("m", "openrouter", "high").model_options == {"reasoning": "high"}


def test_a_blank_effort_is_omitted_rather_than_sent_as_none():
    assert llm_config("m", "openrouter", None).model_options == {}
    assert llm_config("m", "openrouter", "").model_options == {}


def test_a_session_builds_from_the_config_the_driver_makes():
    session = AgentSessionRunner.new_session(
        llm_config("openai/gpt-5.6-luna", "openrouter", "high"),
        runtime_config=RuntimeConfig(hard_max_steps=200, subagents_enabled=False),
    )

    assert isinstance(session, AgentSession)
    assert session.session_config.llm_config.model_options == {"reasoning": "high"}
    assert session.session_config.runtime_config.hard_max_steps == 200


def test_the_composition_still_assembles_against_the_current_core(tmp_path):
    # the shape the container runs: yolo, no skills, no subagents
    session = AgentSessionRunner.new_session(llm_config("m", "faux"))

    runner = build_runner(session, workspace=tmp_path, mode="yolo", instructions=False, compaction=False)

    assert runner.session is session
