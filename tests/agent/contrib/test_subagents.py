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
    CONTROL_PROMPT,
    LIST_TOOL_NAME,
    RESULT_TOOL_NAME,
    SPAWN_TOOL_NAME,
    SPAWNING_PROMPT,
    STOP_TOOL_NAME,
    CreateConversationResult,
    ListSubagents,
    SpawnSubagent,
    StopSubagent,
    SubagentSpawn,
    SubagentsPlugin,
    SubagentToolRegistry,
    control_prompt_part,
    spawn_gate_open,
    spawning_prompt_part,
)
from luca.agent.core import (
    AgentSession,
    AssistantMessage,
    CancellationToken,
    CancelRequested,
    ChildConversation,
    ExecutionResult,
    ExecutionStatus,
    LLMConfig,
    RuntimeConfig,
    SessionConfig,
    SystemPromptPart,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolKind,
    TurnFinish,
    TurnOutcome,
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


def test_the_plugin_ships_all_four_tools_in_its_own_allowing_registry():
    session = session_with()
    registry = SubagentsPlugin().get_tool_registry(session)

    assert isinstance(registry, SubagentToolRegistry)
    assert isinstance(registry.permission_policy, YoloPermissionPolicy)
    assert [t.name for t in registry.tools] == [
        SPAWN_TOOL_NAME,
        RESULT_TOOL_NAME,
        STOP_TOOL_NAME,
        LIST_TOOL_NAME,
    ]


def test_the_plugin_contributes_two_callable_prompt_parts():
    # each rides the same predicate that decides its tools' availability, so
    # they HAVE to be callables — see the gate identity above
    [spawning, control] = SubagentsPlugin().get_system_prompt_parts(session_with())

    assert spawning is spawning_prompt_part
    assert control is control_prompt_part


def test_the_plugin_composes_with_the_plugin_runner():
    session = session_with()

    runner = PluginAgentSessionRunner(session, plugins=[SubagentsPlugin()])

    assert runner.plugins[0].__class__ is SubagentsPlugin
    assert len(runner.system_prompt_parts) == 2


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


# ── the stop and list tools ───────────────────────────────────────────────────


def orchestrating_session(*, resolved: bool = False, cancelling: bool = False) -> AgentSession:
    """`c1` mid-orchestration: one spawn execution (task `t1`) and its link in
    the open turn; the child `c2` running, finished, or cancelling."""
    call = ToolCall(
        id="tc1",
        name=SPAWN_TOOL_NAME,
        arguments={"prompt": "Research A", "description": "research A", "task_id": "t1"},
    )
    entries = {
        "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
        "te1": ToolExecution(
            id="te1",
            conversation_id="c1",
            parent_id="ts",
            created_at=500,
            tool_call_id="tc1",
            raw_tool_call=call,
            tool_spec=SpawnSubagent().get_tool_spec(),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(
                content=[TextContent(text="spawned")],
                structured_content={
                    "is_subagent_spawn": True,
                    "task_id": "t1",
                    "prompt": "Research A",
                    "description": "research A",
                    "process_subagent_result_tool_name": RESULT_TOOL_NAME,
                },
            ),
            started_at=500,
            ended_at=500,
        ),
        "ch1": ChildConversation(
            id="ch1",
            parent_id="te1",
            created_at=500,
            conversation_id="c2",
            tool_execution_id="te1",
            execution_result=(ExecutionResult(content=[TextContent(text="A done")]) if resolved else None),
        ),
        "u_seed": UserMessage(id="u_seed", created_at=500, parts=[TextContent(text="Research A")]),
        "ts_c": TurnStart(id="ts_c", parent_id="u_seed", created_at=500),
    }
    child_nodes = ["u_seed", "ts_c"]
    if cancelling:
        entries["cr"] = CancelRequested(id="cr", parent_id="ts_c", created_at=600)
        child_nodes.append("cr")
    if resolved:
        entries["tf_c"] = TurnFinish(id="tf_c", parent_id="ts_c", created_at=700)
        child_nodes.append("tf_c")
    return make_session(
        id="s_orchestrating",
        entries=entries,
        conversations={
            "c1": conversation("c1", ["u1", "ts", "te1", "ch1"], created_at=500, updated_at=500),
            "c2": conversation("c2", child_nodes, created_at=500, updated_at=500, depth=1),
        },
        main_conversation_id="c1",
        session_config=SessionConfig(
            llm_config=MODEL,
            runtime_config=RuntimeConfig(subagents_enabled=True),
        ),
    )


def test_the_stop_spec_declares_its_payload_and_the_list_spec_reads():
    stop_spec = StopSubagent().get_tool_spec()
    list_spec = ListSubagents().get_tool_spec()

    assert stop_spec.name == STOP_TOOL_NAME
    assert stop_spec.is_private is False
    assert "is_subagent_stop" in stop_spec.output_schema["properties"]
    # declaring the STOP payload keeps it out of the SPAWN gate
    assert "is_subagent_spawn" not in stop_spec.output_schema["properties"]
    assert list_spec.name == LIST_TOOL_NAME
    assert list_spec.tool_kind is ToolKind.READ
    assert list_spec.output_schema is None


async def test_stop_signals_a_live_task():
    result = await StopSubagent().execute(
        {"task_id": "t1", "reason": "no longer needed"},
        orchestrating_session(),
        "c1",
        cancellation_token=TOKEN,
    )

    assert result.is_error is False
    assert result.content == [TextContent(text="Stop signal received for task t1. Reason: no longer needed")]
    assert result.structured_content == {
        "is_subagent_stop": True,
        "task_id": "t1",
        "reason": "no longer needed",
    }


async def test_stop_refuses_an_unknown_task():
    result = await StopSubagent().execute(
        {"task_id": "ghost"},
        orchestrating_session(),
        "c1",
        cancellation_token=TOKEN,
    )

    # the declared schema is the plugin's guarantee: the payload is returned
    # with the flag DOWN, exactly like a declining spawn — the runner does
    # nothing with it
    assert result.is_error is True
    assert result.content == [TextContent(text="No task ghost in this turn. Known task ids: t1.")]
    assert result.structured_content == {"is_subagent_stop": False, "task_id": "ghost", "reason": None}


async def test_stop_refuses_a_finished_task():
    result = await StopSubagent().execute(
        {"task_id": "t1"},
        orchestrating_session(resolved=True),
        "c1",
        cancellation_token=TOKEN,
    )

    assert result.is_error is True
    assert result.content == [TextContent(text="Task t1 has already finished; there is nothing to stop.")]
    assert result.structured_content == {"is_subagent_stop": False, "task_id": "t1", "reason": None}


async def test_stop_notes_an_already_stopping_task():
    result = await StopSubagent().execute(
        {"task_id": "t1"},
        orchestrating_session(cancelling=True),
        "c1",
        cancellation_token=TOKEN,
    )

    assert result.is_error is True
    assert result.content == [TextContent(text="Task t1 is already being stopped.")]
    assert result.structured_content == {"is_subagent_stop": False, "task_id": "t1", "reason": None}


async def test_list_renders_the_turns_tasks():
    result = await ListSubagents().execute({}, orchestrating_session(), "c1", cancellation_token=TOKEN)

    assert result.content == [
        TextContent(
            text="<task id=t1 status=pending>\ndescription: research A\nprompt: Research A\n</task>",
        ),
    ]


async def test_list_reports_completed_and_cancelling_states():
    completed = await ListSubagents().execute({}, orchestrating_session(resolved=True), "c1", cancellation_token=TOKEN)
    stopping = await ListSubagents().execute({}, orchestrating_session(cancelling=True), "c1", cancellation_token=TOKEN)

    assert completed.content == [
        TextContent(text="<task id=t1 status=completed>\ndescription: research A\nprompt: Research A\n</task>"),
    ]
    assert stopping.content == [
        TextContent(text="<task id=t1 status=cancelling>\ndescription: research A\nprompt: Research A\n</task>"),
    ]


async def test_list_reports_a_failed_resolution_as_failed():
    # the SAME completed/failed split the projected task update uses — one
    # vocabulary across every surface the model reads
    session = orchestrating_session(resolved=True)
    session.entries["ch1"] = session.entries["ch1"].model_copy(
        update={"execution_result": ExecutionResult(content=[TextContent(text="[subagent cancelled]")], is_error=True)},
    )

    result = await ListSubagents().execute({}, session, "c1", cancellation_token=TOKEN)

    assert result.content == [
        TextContent(text="<task id=t1 status=failed>\ndescription: research A\nprompt: Research A\n</task>"),
    ]


async def test_list_with_no_tasks_says_so():
    result = await ListSubagents().execute({}, session_with(), "c1", cancellation_token=TOKEN)

    assert result.content == [TextContent(text="No subagents have been spawned in the current turn.")]


async def test_the_registry_withholds_the_control_tools_without_tasks():
    registry = SubagentsPlugin().get_tool_registry(session_with())

    bare = [s.name for s in await registry.get_tools(session_with(), "c1")]
    managing = [s.name for s in await registry.get_tools(orchestrating_session(), "c1")]

    assert SPAWN_TOOL_NAME in bare
    assert STOP_TOOL_NAME not in bare
    assert LIST_TOOL_NAME not in bare
    assert {STOP_TOOL_NAME, LIST_TOOL_NAME} <= set(managing)


def test_the_control_prompt_part_follows_the_same_predicate():
    # the second prompt/tool-list identity: teaching stop/list exactly when
    # the registry offers them — a spent spawn budget must not silence it,
    # which is why it cannot ride on `spawning_prompt_part`
    assert control_prompt_part(session_with(), "c1") is None
    part = control_prompt_part(orchestrating_session(), "c1")
    assert part == SystemPromptPart(text=CONTROL_PROMPT, source="subagents")


async def test_the_result_tool_reports_a_stopped_child_as_stopped():
    # outcome-aware: a cancelled child's last words are a progress report, not
    # an answer — presenting them as one would tell the parent a task finished
    # that did not
    session = orchestrating_session()
    session.entries["a_c"] = AssistantMessage(
        id="a_c",
        parent_id="ts_c",
        created_at=600,
        parts=[TextContent(text="Halfway through the research.")],
        llm_config=MODEL,
        stop_reason="stop",
    )
    session.entries["tf_c"] = TurnFinish(id="tf_c", parent_id="a_c", created_at=700, outcome=TurnOutcome.CANCELLED)
    session.conversations["c2"].nodes.extend(["a_c", "tf_c"])

    result = await CreateConversationResult().execute(
        {"task_id": "t1", "prompt": "Research A", "description": "research A", "conversation_id": "c2"},
        session,
        "c1",
        cancellation_token=TOKEN,
    )

    assert result.is_error is True
    assert result.content == [
        TextContent(
            text="The subagent was cancelled before finishing. Its last message:\n\nHalfway through the research.",
        ),
    ]
