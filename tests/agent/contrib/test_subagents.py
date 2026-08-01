"""Self-scoped tests for `luca.agent.contrib.subagents` — the plugin surface,
the two tools' specs, the gate and the prompt part. No runner.

The end-to-end handshake lives in `tests/agent/subagents/`; what is pinned here
is the package's own product: the advertisements it hands a registry, and the
one predicate the gate and the prompt both derive from.
"""

import pytest

from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.simple_tool_registry import YoloPermissionPolicy
from luca.agent.contrib.subagents import (
    RESULT_TOOL_NAME,
    SPAWN_TOOL_NAME,
    SPAWNING_PROMPT,
    CreateConversationResult,
    SpawnSubagent,
    SubagentSpawn,
    SubagentsPlugin,
    SubagentToolRegistry,
    spawn_gate_open,
    spawning_prompt_part,
)
from luca.agent.core import (
    AgentSession,
    AssistantMessage,
    CancellationToken,
    ExecutionResult,
    LLMConfig,
    RuntimeConfig,
    SessionConfig,
    SystemPromptPart,
    TextContent,
    ToolKind,
    TurnFinish,
    TurnStart,
    UserMessage,
)
from tests.agent.scenarios import conversation, make_session

MODEL = LLMConfig(model="test-model", provider="faux")
TOKEN = CancellationToken()


def session_with(*, enabled: bool = True, max_depth: int = 1, child_depth: int = 1) -> AgentSession:
    return make_session(
        id="s_subagents",
        conversations={
            "c1": conversation("c1", [], created_at=500, updated_at=500),
            "c2": conversation("c2", [], created_at=500, updated_at=500, depth=child_depth),
        },
        main_conversation_id="c1",
        session_config=SessionConfig(
            llm_config=MODEL,
            runtime_config=RuntimeConfig(subagents_enabled=enabled, subagents_max_depth=max_depth),
        ),
    )


# ── the specs the model is (and is not) shown ────────────────────────────────


def test_the_spawn_spec_declares_the_handshake():
    spec = SpawnSubagent().get_tool_spec()

    assert spec.name == SPAWN_TOOL_NAME
    assert spec.namespace == "contrib.subagents"
    assert spec.tool_kind is ToolKind.OTHER
    assert spec.is_private is False
    # THE declaration: the gate reads this, from the spec alone, before any
    # call has been made
    assert "is_subagent_spawn" in spec.output_schema["properties"]
    assert set(SubagentSpawn.model_fields) == {
        "is_subagent_spawn",
        "task_id",
        "prompt",
        "description",
        "process_subagent_result_tool_name",
    }


def test_the_result_spec_is_private_and_declares_nothing():
    spec = CreateConversationResult().get_tool_spec()

    assert spec.name == RESULT_TOOL_NAME
    assert spec.is_private is True
    # declaring nothing is what keeps it OUT of the gate: a tool that declares
    # nothing is not a spawn tool
    assert spec.output_schema is None


def test_the_two_specs_are_distinct_rows():
    assert SpawnSubagent().get_tool_spec().spec_id() != CreateConversationResult().get_tool_spec().spec_id()


# ── the spawn tool's payload ──────────────────────────────────────────────────


async def test_spawn_returns_a_status_line_and_the_payload():
    result = await SpawnSubagent().execute(
        {"prompt": "Research A", "description": "research A", "task_id": "t1"},
        session_with(),
        "c1",
        cancellation_token=TOKEN,
    )

    # the model sees ONE short line, carrying the task id it must use to
    # correlate the answer…
    assert result.content == [TextContent(text="Spawned subagent with id t1: research A")]
    # …and the handshake rides free, on a channel the model never sees
    assert result.structured_content == {
        "is_subagent_spawn": True,
        "task_id": "t1",
        "prompt": "Research A",
        "description": "research A",
        "process_subagent_result_tool_name": RESULT_TOOL_NAME,
    }


async def test_a_missing_task_id_is_made_up_and_told_to_the_model():
    result = await SpawnSubagent().execute(
        {"prompt": "p", "description": "d", "task_id": None},
        session_with(),
        "c1",
        cancellation_token=TOKEN,
    )

    made_up = result.structured_content["task_id"]
    assert made_up
    # the status line is the ONLY channel that reaches the model, so an id the
    # model did not choose has to travel on it — otherwise the model cannot
    # correlate the answer with the task it asked for
    assert result.content == [TextContent(text=f"Spawned subagent with id {made_up}: d")]


async def test_the_result_tool_name_travels_in_the_payload():
    # an application can pair its own spawn tool with its own result tool
    # without the core knowing either name
    result = await SpawnSubagent(result_tool_name="my_summarizer").execute(
        {"prompt": "p", "description": "d", "task_id": "t1"},
        session_with(),
        "c1",
        cancellation_token=TOKEN,
    )

    assert result.structured_content["process_subagent_result_tool_name"] == "my_summarizer"


# ── the result tool ───────────────────────────────────────────────────────────


async def test_the_result_is_the_childs_last_words():
    session = make_session(
        id="s_result",
        entries={
            "u2": UserMessage(id="u2", created_at=500, parts=[TextContent(text="Research A")]),
            "ts": TurnStart(id="ts", parent_id="u2", created_at=500),
            "a1": AssistantMessage(
                id="a1",
                parent_id="ts",
                created_at=500,
                parts=[TextContent(text="A is fine.")],
                llm_config=MODEL,
                stop_reason="stop",
            ),
            "tf": TurnFinish(id="tf", parent_id="a1", created_at=500),
        },
        conversations={
            "c1": conversation("c1", [], created_at=500, updated_at=500),
            "c2": conversation("c2", ["u2", "ts", "a1", "tf"], created_at=500, updated_at=500, depth=1),
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )

    result = await CreateConversationResult().execute(
        {"task_id": "t1", "prompt": "Research A", "description": "research A", "conversation_id": "c2"},
        session,
        "c1",  # the PARENT — where this tool runs
        cancellation_token=TOKEN,
    )

    # the last node is the closing marker, so this takes the last assistant
    # MESSAGE, not the last entry
    assert result == ExecutionResult(content=[TextContent(text="A is fine.")])


async def test_a_child_that_never_answered_yields_a_transcript():
    session = make_session(
        id="s_result_empty",
        entries={"u2": UserMessage(id="u2", created_at=500, parts=[TextContent(text="Research A")])},
        conversations={
            "c1": conversation("c1", [], created_at=500, updated_at=500),
            "c2": conversation("c2", ["u2"], created_at=500, updated_at=500, depth=1),
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )

    result = await CreateConversationResult().execute(
        {"task_id": "t1", "prompt": "p", "description": "research A", "conversation_id": "c2"},
        session,
        "c1",
        cancellation_token=TOKEN,
    )

    assert result.is_error is True
    assert "finished without a final message" in result.content[0].text
    assert "Conversation c2" in result.content[0].text


# ── the gate, and the prompt that must agree with it ─────────────────────────


@pytest.mark.parametrize(
    ("enabled", "max_depth", "conversation_id", "expected"),
    [
        (True, 1, "c1", True),  # main, under the cap
        (True, 1, "c2", False),  # a subagent, at the cap
        (False, 1, "c1", False),  # switched off entirely
        (True, 0, "c1", False),  # cap of zero: nobody spawns
        (True, 2, "c2", True),  # a deeper cap lets a subagent spawn
    ],
)
def test_the_gate_predicate(enabled, max_depth, conversation_id, expected):
    session = session_with(enabled=enabled, max_depth=max_depth)

    assert spawn_gate_open(session, conversation_id) is expected


@pytest.mark.parametrize(
    ("enabled", "conversation_id", "expected"),
    [(True, "c1", SystemPromptPart(text=SPAWNING_PROMPT, source="subagents")), (True, "c2", None), (False, "c1", None)],
)
def test_the_prompt_part_follows_the_same_predicate(enabled, conversation_id, expected):
    # THE identity that matters: a static part would tell a subagent at the cap
    # that it can spawn while the tool list withholds the tool, and the model
    # would try. The prompt and the tool list must never disagree.
    assert spawning_prompt_part(session_with(enabled=enabled), conversation_id) == expected


async def test_the_registry_withholds_the_spawn_tool_past_the_cap():
    registry = SubagentToolRegistry(
        tools=SubagentsPlugin().get_tools(),
        permission_policy=YoloPermissionPolicy(),
    )
    session = session_with()

    assert [s.name for s in await registry.get_tools(session, "c1")] == [SPAWN_TOOL_NAME, RESULT_TOOL_NAME]
    # withheld by DECLARATION, not by name — which is what makes a custom spawn
    # tool gate correctly too
    assert [s.name for s in await registry.get_tools(session, "c2")] == [RESULT_TOOL_NAME]


async def test_the_private_result_tool_is_never_withheld():
    # the RUNTIME must still see it: that is how it resolves and dispatches
    registry = SubagentToolRegistry(
        tools=SubagentsPlugin().get_tools(),
        permission_policy=YoloPermissionPolicy(),
    )

    for conversation_id in ("c1", "c2"):
        names = [s.name for s in await registry.get_tools(session_with(enabled=False), conversation_id)]
        assert RESULT_TOOL_NAME in names


# ── the plugin surface ────────────────────────────────────────────────────────


def test_the_plugin_ships_both_tools_in_its_own_allowing_registry():
    session = session_with()
    registry = SubagentsPlugin().get_tool_registry(session)

    assert isinstance(registry, SubagentToolRegistry)
    assert isinstance(registry.permission_policy, YoloPermissionPolicy)
    assert [t.name for t in registry.tools] == [SPAWN_TOOL_NAME, RESULT_TOOL_NAME]


def test_the_plugin_contributes_one_callable_prompt_part():
    [part] = SubagentsPlugin().get_system_prompt_parts(session_with())

    assert callable(part)  # it HAS to be — see the gate identity above


def test_the_plugin_composes_with_the_plugin_runner():
    session = session_with()

    runner = PluginAgentSessionRunner(session, plugins=[SubagentsPlugin()])

    assert runner.plugins[0].__class__ is SubagentsPlugin
    assert len(runner.system_prompt_parts) == 1


def test_installing_the_plugin_does_not_by_itself_enable_subagents():
    # the capability is CONFIGURATION, not installation
    session = make_session(
        id="s_off",
        conversations={"c1": conversation("c1", [], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )

    assert session.session_config.runtime_config.subagents_enabled is False
    assert spawn_gate_open(session, "c1") is False
