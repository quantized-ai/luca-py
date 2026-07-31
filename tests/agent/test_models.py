"""Declarative data-model tests for the tool-execution + approval types and
the runner-lifecycle value objects (TurnOutcome / TurnFinish / CancelRequested
/ RuntimeConfig).

Pure value-object checks — full-object asserts and JSON round-trips, no logic,
no helpers. Locks the execution-lifecycle vocabulary (`ExecutionStatus`), the
orthogonal approval state (`ApprovalStatus` + the ApprovalDecision audit log),
the structured `ToolExecutionError`, and the `ToolExecution` record with its
`raw_tool_call` / `tool_spec_id` split and lifecycle timestamps. The core
carries NO permission vocabulary beyond these: no modes, no rules, no intents.
Cross-field combinations are framework conventions, not validators — the model
must accept what application middleware authors.

Also locks tool-spec NORMALIZATION, which is the data model's only piece of
behavior: `ToolSpec.spec_id()` (the pinned content hash), the shared
`AgentSession.tool_specs` store, and the construction-time restore that hands
every execution referencing an id the one stored instance — plus the two
shapes a session refuses to load.
"""

import hashlib

import pytest
from pydantic import TypeAdapter, ValidationError

from luca.agent.core.models import (
    AgentSession,
    AnyEntry,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    CancelRequested,
    CompactionEntry,
    CompactionSource,
    Conversation,
    ExecutionResult,
    ExecutionStatus,
    ImageBase64,
    ImageContent,
    ImageFileId,
    ImageURL,
    Inf,
    LLMConfig,
    MilliSeconds,
    PrunedEntry,
    RuntimeConfig,
    Seconds,
    SessionConfig,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    ToolKind,
    ToolSpec,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    Usage,
    UserMessage,
    is_compaction_bracket,
)
from tests.agent.scenarios import (
    ADD_SPEC,
    EMPTY_SCHEMA,
    MODEL,
    READ_FILE_SPEC,
    make_session,
)

# The same tool called twice in one session — the precondition for the
# normalization tests: two executions, one distinct spec between them.
REPEATED_CALL_SESSION = make_session(
    id="s_repeated",
    entries={
        "te1": ToolExecution(
            id="te1",
            created_at=500,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            tool_spec=ADD_SPEC,
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="3")]),
            started_at=500,
            ended_at=500,
            updated_at=500,
        ),
        "te2": ToolExecution(
            id="te2",
            parent_id="te1",
            created_at=600,
            tool_call_id="tc2",
            raw_tool_call=ToolCall(id="tc2", name="add", arguments={"a": 3, "b": 4}),
            tool_spec=ADD_SPEC,
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="7")]),
            started_at=600,
            ended_at=600,
            updated_at=600,
        ),
    },
    active_conversation=Conversation(
        id="c1",
        nodes=["te1", "te2"],
        created_at=500,
        updated_at=600,
    ),
    session_config=SessionConfig(llm_config=MODEL),
)

# The same tool one release later: only the wording of `description` changed.
REPHRASED_ADD_SPEC = ADD_SPEC.model_copy(update={"description": "Add two integers."})

# A spec carrying every optional identity field — the historical snapshot an
# archived conversation still renders itself from.
BASH_SPEC = ToolSpec(
    name="bash",
    description="Run a shell command.",
    input_schema={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
    tool_kind=ToolKind.EXECUTE,
    namespace="builtin.shell_tools",
    version="0.0.1",
    timeout_in_ms=30_000,
)


def test_tool_spec_defaults_to_other_kind_and_null_namespace_version():
    assert ToolSpec(
        name="bash",
        description="Run a shell command.",
        input_schema=EMPTY_SCHEMA,
    ) == ToolSpec(
        name="bash",
        description="Run a shell command.",
        input_schema=EMPTY_SCHEMA,
        output_schema=None,
        metadata=None,
        tool_kind=ToolKind.OTHER,
        namespace=None,
        version=None,
        timeout_in_ms=None,
    )


def test_a_spec_declaring_an_output_schema_round_trips():
    declared = ToolSpec(
        name="get_weather",
        description="Get the current weather for a city.",
        input_schema=EMPTY_SCHEMA,
        output_schema={
            "type": "object",
            "properties": {"degrees_in_celsius": {"type": "integer"}},
            "required": ["degrees_in_celsius"],
        },
    )

    assert ToolSpec.model_validate_json(declared.model_dump_json()) == declared


def test_tool_spec_requires_a_description_and_an_input_schema():
    # both are what the model is shown; the client's wire tool type rejects
    # null for either
    with pytest.raises(ValidationError):
        ToolSpec(name="bash")


def test_tool_spec_rejects_a_null_input_schema():
    # "takes no arguments" is the empty object schema, not the absence of one
    with pytest.raises(ValidationError):
        ToolSpec(name="bash", description="Run a shell command.", input_schema=None)


def test_tool_spec_carries_no_invocation_arguments():
    # arguments belong to ToolExecution.raw_tool_call, not the tool snapshot
    with pytest.raises(ValidationError):
        ToolSpec(
            name="bash",
            description="Run a shell command.",
            input_schema=EMPTY_SCHEMA,
            parameters={"command": "ls"},
        )


def test_tool_spec_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        ToolSpec(
            name="bash",
            description="Run a shell command.",
            input_schema=EMPTY_SCHEMA,
            bogus="nope",
        )


def test_a_tool_taking_no_arguments_round_trips_with_the_empty_object_schema():
    no_args = ToolSpec(
        name="ping",
        description="Ping the server.",
        input_schema={"type": "object", "properties": {}},
    )

    reloaded = ToolSpec.model_validate_json(no_args.model_dump_json())

    assert reloaded == no_args
    # and it still keys the same row in the store after the round trip
    assert reloaded.spec_id() == no_args.spec_id()


# ── spec_id: the pinned content-derived identity ──────────────────────────────


def test_spec_id_is_the_sha256_hex_of_the_canonical_json():
    executable = ToolSpec(
        name="bash",
        description="Run a shell command.",
        input_schema=EMPTY_SCHEMA,
        tool_kind=ToolKind.EXECUTE,
    )

    # the exact bytes the rule pins: recursively sorted keys, no whitespace,
    # enums as their string values, None-valued fields kept in the payload
    canonical = (
        '{"description":"Run a shell command.",'
        '"input_schema":{"properties":{},"type":"object"},'
        '"metadata":null,"name":"bash","namespace":null,'
        '"output_schema":null,"timeout_in_ms":null,'
        '"tool_kind":"execute","version":null}'
    )

    assert executable.spec_id() == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_two_separately_built_identical_specs_share_one_id():
    assert (
        ToolSpec(
            name="add",
            description="Add two numbers.",
            input_schema=EMPTY_SCHEMA,
        ).spec_id()
        == ToolSpec(
            name="add",
            description="Add two numbers.",
            input_schema=EMPTY_SCHEMA,
        ).spec_id()
    )


def test_input_schema_key_order_does_not_affect_the_spec_id():
    # a dict literal remembers its insertion order; the canonical rendering
    # sorts keys recursively, so a reordered schema is the same tool
    assert (
        ToolSpec(
            name="add",
            description="Add two numbers.",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            },
        ).spec_id()
        == ToolSpec(
            name="add",
            description="Add two numbers.",
            input_schema={
                "properties": {"b": {"type": "integer"}, "a": {"type": "integer"}},
                "type": "object",
            },
        ).spec_id()
    )


def test_a_reworded_description_mints_a_new_spec_id():
    assert ADD_SPEC.spec_id() != REPHRASED_ADD_SPEC.spec_id()


def test_declaring_an_output_schema_mints_a_new_spec_id():
    # the declaration is part of the tool definition, so a tool that gains one
    # is a different spec — one extra stored row, old executions unaffected
    assert ADD_SPEC.spec_id() != ADD_SPEC.model_copy(update={"output_schema": EMPTY_SCHEMA}).spec_id()


def test_tool_kind_members():
    assert {kind.name: kind.value for kind in ToolKind} == {
        "READ": "read",
        "SEARCH": "search",
        "WEB_FETCH": "web_fetch",
        "EDIT": "edit",
        "MOVE": "move",
        "DELETE": "delete",
        "EXECUTE": "execute",
        "SWITCH_MODE": "switch_mode",
        "OTHER": "other",
    }


def test_approval_option_members():
    assert {option.name: option.value for option in ApprovalOption} == {
        "ALLOW": "allow",
        "DENY": "deny",
        "PENDING": "pending",
    }


def test_approval_status_members():
    assert {status.name: status.value for status in ApprovalStatus} == {
        "PENDING": "pending",
        "ALLOWED": "allowed",
        "REJECTED": "rejected",
    }


def test_approval_decision_defaults_stamp_created_at():
    decision = ApprovalDecision(decision=ApprovalOption.ALLOW)

    assert decision.decision == ApprovalOption.ALLOW
    assert decision.metadata is None
    assert isinstance(decision.created_at, int)
    assert decision.created_at > 0


def test_approval_decision_round_trips_with_metadata():
    decision = ApprovalDecision(
        decision=ApprovalOption.DENY,
        metadata={"via": "rule", "reason": "blocked by policy"},
        created_at=1780495331220,
    )
    assert ApprovalDecision.model_validate_json(decision.model_dump_json()) == decision


def test_approval_decision_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        ApprovalDecision(decision=ApprovalOption.ALLOW, via="mode", created_at=1)


def test_execution_result_defaults():
    assert ExecutionResult(content=[TextContent(text="3")]) == ExecutionResult(
        content=[TextContent(text="3")],
        structured_content=None,
        metadata={},
        is_error=False,
    )


def test_an_execution_result_carrying_structured_content_round_trips():
    result = ExecutionResult(
        content=[TextContent(text="25°C, wind from the south.")],
        structured_content={"degrees_in_celsius": 25, "wind_direction": "south"},
    )

    assert ExecutionResult.model_validate_json(result.model_dump_json()) == result


def test_execution_result_carries_no_timing():
    # timing lives on ToolExecution.started_at / ended_at
    with pytest.raises(ValidationError):
        ExecutionResult(content=[TextContent(text="3")], executed_at=1000)


def test_tool_execution_error_defaults_and_round_trip():
    error = ToolExecutionError(
        error_type="ConnectionError",
        error_message="Connection to api.example.com was closed.",
    )
    assert error.details == {}

    rich = ToolExecutionError(
        error_type="RemoteServiceError",
        error_message="The billing service rejected the request.",
        details={
            "phase": "execution",
            "service": "billing",
            "code": "ACCOUNT_SUSPENDED",
            "http_status": 403,
            "retryable": False,
        },
    )
    assert ToolExecutionError.model_validate_json(rich.model_dump_json()) == rich


def test_tool_execution_error_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        ToolExecutionError(error_type="X", error_message="boom", traceback="...")


def test_tool_execution_defaults_to_birth_state():
    assert ToolExecution(
        id="te1",
        created_at=1,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1}),
    ) == ToolExecution(
        id="te1",
        parent_id=None,
        created_at=1,
        type="tool_execution",
        context_tokens=0,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1}),
        tool_spec=None,
        tool_spec_id=None,
        extras={},
        approval_status=None,
        approval_decisions=[],
        status=ExecutionStatus.PENDING,
        result=None,
        error=None,
        started_at=None,
        ended_at=None,
        cancel_signalled_at=None,
        updated_at=None,
        is_doom_loop_flagged=False,
    )


def test_an_uncommitted_registry_draft_may_carry_a_spec_with_no_id():
    draft = ToolExecution(
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
    )

    # refusing an unstamped spec is a SESSION rule: a birth draft has no
    # identity yet, and the ledger's write door is what files it and stamps
    # the id
    assert (draft.id, draft.tool_spec, draft.tool_spec_id) == (None, ADD_SPEC, None)


def test_tool_execution_requires_raw_tool_call():
    with pytest.raises(ValidationError):
        ToolExecution(id="te1", created_at=1, tool_call_id="tc1")


def test_tool_execution_dispatched_and_duration_are_derived():
    undispatched = ToolExecution(
        id="te1",
        created_at=1,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add"),
    )
    settled = ToolExecution(
        id="te2",
        created_at=1,
        tool_call_id="tc2",
        raw_tool_call=ToolCall(id="tc2", name="add"),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="3")]),
        started_at=1000,
        ended_at=1250,
    )

    assert undispatched.dispatched is False
    assert undispatched.duration_ms is None
    assert settled.dispatched is True
    assert settled.duration_ms == 250


def test_tool_execution_does_not_enforce_cross_field_invariants():
    # framework conventions, not schema validation: middleware-authored
    # combinations must construct (the application owns the consequences)
    unusual = ToolExecution(
        id="te1",
        created_at=1,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add"),
        status=ExecutionStatus.COMPLETED,  # no result, error present
        error=ToolExecutionError(error_type="X", error_message="authored"),
        approval_status=ApprovalStatus.REJECTED,
        ended_at=5,
    )
    assert unusual.status == ExecutionStatus.COMPLETED


def test_versioned_tool_execution_round_trips_through_json():
    execution = ToolExecution(
        id="caf0ab9ac",
        parent_id="d4e5f6a7",
        created_at=1780495331220,
        tool_call_id="toolu_01Tg",
        raw_tool_call=ToolCall(
            id="toolu_01Tg",
            name="bash",
            arguments={"command": "pytest -x -q"},
        ),
        tool_spec=BASH_SPEC,
        tool_spec_id=BASH_SPEC.spec_id(),
        extras={
            "approval_context": {
                "resources": ["pytest -x -q"],
                "preview": "Run the test suite",
                "remember_as": [
                    {"resource": "pytest *", "preview": "Always allow pytest"},
                ],
            },
        },
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[
            ApprovalDecision(
                decision=ApprovalOption.PENDING,
                created_at=1780495331220,
            ),
            ApprovalDecision(
                decision=ApprovalOption.ALLOW,
                metadata={"via": "user"},
                created_at=1780495332220,
            ),
        ],
        status=ExecutionStatus.INTERRUPTED,
        started_at=1780495332500,
        ended_at=1780495333500,
        cancel_signalled_at=1780495333000,
        updated_at=1780495333500,
    )
    assert ToolExecution.model_validate_json(execution.model_dump_json()) == execution


def test_failed_tool_execution_round_trips_with_structured_error():
    execution = ToolExecution(
        id="te1",
        created_at=1780495331220,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(
            id="tc1",
            name="read_file",
            arguments={"encoding": "utf-16"},
        ),
        tool_spec=READ_FILE_SPEC,
        tool_spec_id=READ_FILE_SPEC.spec_id(),
        status=ExecutionStatus.INVALID,
        error=ToolExecutionError(
            error_type="InvalidToolArguments",
            error_message="Arguments for tool 'read_file' are invalid.",
            details={
                "errors": [
                    {
                        "type": "missing",
                        "loc": ["path"],
                        "msg": "Field required",
                        "input": {"encoding": "utf-16"},
                    },
                ],
            },
        ),
        ended_at=1780495331220,
    )
    assert ToolExecution.model_validate_json(execution.model_dump_json()) == execution


# ── tool-spec normalization inside a session ─────────────────────────────────


def test_the_same_tool_called_twice_is_stored_once():
    assert REPEATED_CALL_SESSION.tool_specs == {ADD_SPEC.spec_id(): ADD_SPEC}


def test_a_serialized_session_carries_no_inline_tool_spec():
    dumped = REPEATED_CALL_SESSION.model_dump(mode="json")

    # the whole persisted shape of one execution — `tool_spec` is gone and the
    # id is what stays behind (`te2` is the same shape one call later)
    assert dumped["entries"]["te1"] == {
        "id": "te1",
        "parent_id": None,
        "created_at": 500,
        "type": "tool_execution",
        "context_tokens": 0,
        "tool_call_id": "tc1",
        "raw_tool_call": {
            "type": "tool_call",
            "id": "tc1",
            "name": "add",
            "arguments": {"a": 1, "b": 2},
        },
        "tool_spec_id": ADD_SPEC.spec_id(),
        "extras": {},
        "approval_status": None,
        "approval_decisions": [],
        "status": "completed",
        "result": {
            "content": [{"type": "text", "text": "3"}],
            "structured_content": None,
            "metadata": {},
            "is_error": False,
        },
        "error": None,
        "started_at": 500,
        "ended_at": 500,
        "cancel_signalled_at": None,
        "updated_at": 500,
        "is_doom_loop_flagged": False,
    }


def test_a_session_round_trip_restores_every_tool_spec_by_reference():
    reloaded = AgentSession.model_validate_json(REPEATED_CALL_SESSION.model_dump_json())

    assert reloaded == REPEATED_CALL_SESSION
    # both executions hold the ONE stored instance, not a copy each
    assert reloaded.entries["te1"].tool_spec is reloaded.tool_specs[ADD_SPEC.spec_id()]
    assert reloaded.entries["te2"].tool_spec is reloaded.tool_specs[ADD_SPEC.spec_id()]


def test_structured_content_survives_a_session_round_trip():
    # the session serializer rewrites every execution dict to strip its inline
    # spec — the payload rides through that untouched
    session = make_session(
        id="s_structured",
        entries={
            "te1": ToolExecution(
                id="te1",
                created_at=500,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="get_weather", arguments={"city": "Berlin"}),
                tool_spec=ADD_SPEC,
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(
                    content=[TextContent(text="25°C, wind from the south.")],
                    structured_content={"degrees_in_celsius": 25, "wind_direction": "south"},
                ),
                started_at=500,
                ended_at=500,
                updated_at=500,
            ),
        },
        active_conversation=Conversation(id="c1", nodes=["te1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )

    assert AgentSession.model_validate_json(session.model_dump_json()) == session


def test_a_standalone_tool_execution_still_serializes_its_spec_inline():
    execution = ToolExecution(
        id="te1",
        created_at=500,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="3")]),
        started_at=500,
        ended_at=500,
        updated_at=500,
    )

    # nothing restores an execution forwarded on its own (a lifecycle event
    # reaching a log sink holds no session), so an equal round trip is the
    # proof the spec travelled with it
    assert ToolExecution.model_validate_json(execution.model_dump_json()) == execution


def test_a_dangling_tool_spec_id_refuses_to_load():
    # entries copied between sessions without their specs, a truncated file
    with pytest.raises(ValidationError):
        AgentSession(
            id="s_dangling",
            entries={
                "te1": ToolExecution(
                    id="te1",
                    created_at=500,
                    tool_call_id="tc1",
                    raw_tool_call=ToolCall(id="tc1", name="add"),
                    tool_spec_id=ADD_SPEC.spec_id(),
                ),
            },
            tool_specs={},
            active_conversation=Conversation(
                id="c1",
                nodes=["te1"],
                created_at=500,
                updated_at=500,
            ),
            session_config=SessionConfig(llm_config=MODEL),
        )


def test_a_pre_normalization_session_refuses_to_load():
    # an inline spec with no id is a file written before normalization: it
    # would load and run, then lose every spec on the first save
    with pytest.raises(ValidationError):
        AgentSession(
            id="s_pre_normalization",
            entries={
                "te1": ToolExecution(
                    id="te1",
                    created_at=500,
                    tool_call_id="tc1",
                    raw_tool_call=ToolCall(id="tc1", name="add"),
                    tool_spec=ADD_SPEC,
                ),
            },
            active_conversation=Conversation(
                id="c1",
                nodes=["te1"],
                created_at=500,
                updated_at=500,
            ),
            session_config=SessionConfig(llm_config=MODEL),
        )


def test_a_rephrased_tool_keeps_the_older_executions_pointing_at_the_older_spec():
    session = make_session(
        id="s_two_versions",
        entries={
            "te1": ToolExecution(
                id="te1",
                created_at=500,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
                tool_spec=ADD_SPEC,
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(content=[TextContent(text="3")]),
                started_at=500,
                ended_at=500,
                updated_at=500,
            ),
            "te2": ToolExecution(
                id="te2",
                parent_id="te1",
                created_at=600,
                tool_call_id="tc2",
                raw_tool_call=ToolCall(id="tc2", name="add", arguments={"a": 3, "b": 4}),
                tool_spec=REPHRASED_ADD_SPEC,
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(content=[TextContent(text="7")]),
                started_at=600,
                ended_at=600,
                updated_at=600,
            ),
        },
        active_conversation=Conversation(
            id="c1",
            nodes=["te1", "te2"],
            created_at=500,
            updated_at=600,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )

    reloaded = AgentSession.model_validate_json(session.model_dump_json())

    # both rows survive — a stored spec is never rewritten in place
    assert reloaded.tool_specs == {
        ADD_SPEC.spec_id(): ADD_SPEC,
        REPHRASED_ADD_SPEC.spec_id(): REPHRASED_ADD_SPEC,
    }
    assert reloaded.entries["te1"].tool_spec == ADD_SPEC
    assert reloaded.entries["te2"].tool_spec == REPHRASED_ADD_SPEC


def test_session_config_has_no_permission_state():
    with pytest.raises(ValidationError):
        SessionConfig(llm_config=MODEL, permission_policy={"mode": "ask"})


# ── runner-lifecycle value objects ───────────────────────────────────────────


def test_execution_status_members():
    assert {status.name: status.value for status in ExecutionStatus} == {
        "PENDING": "pending",
        "RUNNING": "running",
        "COMPLETED": "completed",
        "FAILED": "failed",
        "NOT_FOUND": "not_found",
        "INVALID": "invalid",
        "REJECTED": "rejected",
        "CANCELLED": "cancelled",
        "INTERRUPTED": "interrupted",
        "TIMED_OUT": "timed_out",
    }


def test_turn_outcome_members():
    assert {outcome.name: outcome.value for outcome in TurnOutcome} == {
        "COMPLETED": "completed",
        "CANCELLED": "cancelled",
        "TIMED_OUT": "timed_out",
        "ERRORED": "errored",
    }


def test_turn_finish_defaults_keep_existing_literals_valid():
    assert TurnFinish(id="tf", created_at=1) == TurnFinish(
        id="tf",
        parent_id=None,
        created_at=1,
        outcome=TurnOutcome.COMPLETED,
        error=None,
    )


def test_turn_finish_round_trips_with_outcome_and_error():
    finish = TurnFinish(
        id="tf",
        parent_id="a1",
        created_at=1780495331220,
        outcome=TurnOutcome.TIMED_OUT,
        error="client timeout after 30s",
    )
    assert TurnFinish.model_validate_json(finish.model_dump_json()) == finish


def test_cancel_requested_defaults_to_cancelled_outcome():
    assert CancelRequested(id="cr", created_at=1) == CancelRequested(
        id="cr",
        parent_id=None,
        created_at=1,
        outcome=TurnOutcome.CANCELLED,
        error=None,
    )


def test_cancel_requested_rejects_completed_outcome():
    with pytest.raises(ValidationError):
        CancelRequested(id="cr", created_at=1, outcome=TurnOutcome.COMPLETED)


def test_cancel_requested_round_trips():
    entry = CancelRequested(
        id="cr",
        parent_id="te1",
        created_at=1780495331220,
        outcome=TurnOutcome.ERRORED,
        error="abandoned at the approval gate",
    )
    assert CancelRequested.model_validate_json(entry.model_dump_json()) == entry


def test_runtime_config_defaults_are_infinite_and_zero_grace():
    assert RuntimeConfig() == RuntimeConfig(
        builtin_client_completion_timeout_in_ms=Inf,
        client_completion_timeout_in_ms=Inf,
        tool_execution_timeout_in_ms=Inf,
        llm_completion_cancellation_grace_period=0,
        tool_cancellation_grace_period=0,
        soft_max_steps=Inf,
        hard_max_steps=Inf,
        doom_loop_threshold=Inf,
        limit_tool_choice_on_soft_max_steps_reached=True,
        limit_tool_choice_on_doom_loop_flagged=True,
        extras={},
    )


def test_runtime_config_rejects_below_inf():
    with pytest.raises(ValidationError):
        RuntimeConfig(tool_execution_timeout_in_ms=-2)


def test_runtime_config_round_trips_with_extras():
    config = RuntimeConfig(
        tool_execution_timeout_in_ms=Seconds(30),
        client_completion_timeout_in_ms=MilliSeconds(120_000),
        tool_cancellation_grace_period=Seconds(2),
        extras={"app": {"theme": "dark"}},
    )
    assert RuntimeConfig.model_validate_json(config.model_dump_json()) == config
    assert config.tool_execution_timeout_in_ms == 30_000
    assert config.client_completion_timeout_in_ms == 120_000


def test_session_config_carries_a_default_runtime_config():
    assert SessionConfig(llm_config=MODEL) == SessionConfig(
        llm_config=MODEL,
        runtime_config=RuntimeConfig(),
    )


# ── context tokens, usage records, pruned entries, conversation identity ──────


def test_context_tokens_defaults_to_zero_and_rejects_negatives():
    assert TextContent(text="x") is not None  # anchor import
    entry = TurnFinish(id="tf", created_at=1)
    assert entry.context_tokens == 0
    with pytest.raises(ValidationError):
        TurnFinish(id="tf", created_at=1, context_tokens=-1)


def test_assistant_message_carries_no_usage_field():
    # provider usage is conversation-scoped accessory data — see
    # AgentSession.usages — never embedded in the entry
    with pytest.raises(ValidationError):
        AssistantMessage(
            id="a1",
            created_at=1,
            parts=[TextContent(text="hi")],
            llm_config=MODEL,
            stop_reason="stop",
            usage=Usage(conversation_id="c1", entry_id="a1"),
        )


def test_turn_finish_carries_no_usage_field():
    with pytest.raises(ValidationError):
        TurnFinish(id="tf", created_at=1, usage=Usage(conversation_id="c1", entry_id="a1"))


def test_usage_is_a_self_describing_association_record():
    # the ids are required: usage only has meaning together with the
    # conversation-entry relationship it describes
    with pytest.raises(ValidationError):
        Usage(input=10, output=5)
    assert Usage(conversation_id="c1", entry_id="a1") == Usage(
        conversation_id="c1",
        entry_id="a1",
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
    )


def test_conversation_requires_a_stable_id():
    with pytest.raises(ValidationError):
        Conversation(nodes=[], created_at=0, updated_at=0)


def test_pruned_entry_round_trips_inside_a_session():
    session = AgentSession(
        id="s",
        entries={
            "te1": ToolExecution(
                id="te1",
                created_at=1,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add", arguments={}),
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(content=[TextContent(text="3")]),
                started_at=1,
                ended_at=1,
            ),
            "p1": PrunedEntry(
                id="p1",
                created_at=2,
                pruned_entry_type="tool_execution",
                pruned_entry_id="te1",
                content=[TextContent(text="[pruned]")],
                context_tokens=2,
            ),
        },
        usages={
            "c1": {
                "a1": Usage(
                    conversation_id="c1",
                    entry_id="a1",
                    input=10,
                    output=5,
                    total_tokens=15,
                ),
            },
        },
        active_conversation=Conversation(
            id="c1",
            nodes=["p1"],
            created_at=0,
            updated_at=2,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )

    reloaded = AgentSession.model_validate_json(session.model_dump_json())

    assert reloaded == session
    # the discriminated union deserializes the node to its concrete subclass
    assert type(reloaded.entries["p1"]) is PrunedEntry


# ── image content ──────────────────────────────────────────────────────────────


def test_image_content_defaults_to_empty_metadata():
    source = ImageBase64(data="aGk=", media_type="image/png")

    assert ImageContent(source=source) == ImageContent(source=source, metadata={})


def test_image_content_round_trips_each_source_kind():
    for source in (
        ImageURL(url="https://example.com/a.png", media_type="image/png"),
        ImageBase64(data="aGk=", media_type="image/png"),
        ImageFileId(file_id="file_123"),
    ):
        part = ImageContent(source=source, metadata={"name": "a.png"})

        assert ImageContent.model_validate_json(part.model_dump_json()) == part


def test_image_content_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ImageContent(
            source=ImageBase64(data="aGk=", media_type="image/png"),
            bogus="nope",
        )


def test_image_source_rejects_an_unknown_kind():
    with pytest.raises(ValidationError):
        ImageContent.model_validate(
            {"source": {"kind": "carrier-pigeon", "data": "aGk="}},
        )


def test_user_message_mixes_image_and_text_parts_in_order():
    message = UserMessage(
        id="u1",
        created_at=1000,
        parts=[
            ImageContent(
                source=ImageBase64(data="aGk=", media_type="image/png"),
                metadata={"name": "receipt.jpg"},
            ),
            TextContent(text="how much did I tip?"),
        ],
    )

    reloaded = UserMessage.model_validate_json(message.model_dump_json())

    assert reloaded == message
    assert [type(part) for part in reloaded.parts] == [ImageContent, TextContent]


def test_user_message_parts_require_the_type_discriminator():
    with pytest.raises(ValidationError):
        UserMessage.model_validate(
            {"id": "u1", "created_at": 1000, "parts": [{"text": "hi"}]},
        )


def test_execution_result_carries_the_same_content_union_as_a_message():
    # the conversation is the source of truth: a tool that returns an image
    # stores one, whatever a given provider can receive today
    image = ImageContent(source=ImageBase64(data="aGk=", media_type="image/png"))
    result = ExecutionResult(content=[image, TextContent(text="a screenshot")])

    assert ExecutionResult.model_validate_json(result.model_dump_json()) == result


# ── the two content unions are separate on purpose ─────────────────────────────


def test_a_user_message_cannot_carry_assistant_only_parts():
    # the reason ContentPart and AssistantContentPart stay separate
    for rejected in (
        ThinkingContent(thinking="reasoning is the assistant's"),
        ToolCall(id="tc1", name="add"),
    ):
        with pytest.raises(ValidationError):
            UserMessage(id="u1", created_at=1000, parts=[rejected])


def test_an_assistant_message_cannot_carry_an_image():
    # assistant images are a separate change: the client's AssistantMessage
    # has no ImageBlock, so nothing downstream could project one yet
    with pytest.raises(ValidationError):
        AssistantMessage(
            id="a1",
            created_at=1000,
            parts=[
                ImageContent(
                    source=ImageBase64(data="aGk=", media_type="image/png"),
                ),
            ],
            llm_config=MODEL,
            stop_reason="stop",
        )


def test_text_is_the_part_both_unions_share():
    text = TextContent(text="shared")

    assert UserMessage(id="u1", created_at=1, parts=[text]).parts == [text]
    assert AssistantMessage(
        id="a1",
        created_at=1,
        parts=[text],
        llm_config=MODEL,
        stop_reason="stop",
    ).parts == [text]


def test_an_assistant_message_round_trips_every_part_type():
    message = AssistantMessage(
        id="a1",
        created_at=1000,
        parts=[
            ThinkingContent(thinking="let me add"),
            TextContent(text="adding now"),
            ToolCall(id="tc1", name="add", arguments={"a": 1}),
        ],
        llm_config=MODEL,
        stop_reason="tool_use",
    )

    reloaded = AssistantMessage.model_validate_json(message.model_dump_json())

    assert reloaded == message
    assert [type(part) for part in reloaded.parts] == [
        ThinkingContent,
        TextContent,
        ToolCall,
    ]


# ── thinking signatures ────────────────────────────────────────────────────────


def test_thinking_content_defaults_to_unsigned_and_unredacted():
    assert ThinkingContent(thinking="hmm") == ThinkingContent(
        thinking="hmm",
        signature=None,
        redacted=False,
    )


def test_thinking_content_round_trips_its_signature():
    part = ThinkingContent(thinking="let me think", signature="sig-abc")

    assert ThinkingContent.model_validate_json(part.model_dump_json()) == part


def test_a_redacted_thinking_part_round_trips_with_an_empty_body():
    # the reasoning is encrypted into the signature; `thinking` stays empty
    part = ThinkingContent(thinking="", signature="encrypted", redacted=True)

    assert ThinkingContent.model_validate_json(part.model_dump_json()) == part


def test_a_signature_survives_a_whole_session_round_trip():
    session = AgentSession(
        id="s_sig",
        entries={
            "a1": AssistantMessage(
                id="a1",
                created_at=1000,
                parts=[
                    ThinkingContent(thinking="reasoning", signature="sig-abc"),
                    TextContent(text="the answer"),
                ],
                llm_config=MODEL,
                stop_reason="stop",
            ),
        },
        active_conversation=Conversation(
            id="c1",
            nodes=["a1"],
            created_at=1000,
            updated_at=1000,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )

    reloaded = AgentSession.model_validate_json(session.model_dump_json())

    assert reloaded.entries["a1"].parts[0].signature == "sig-abc"


# ── compaction ─────────────────────────────────────────────────────────────────


def test_compaction_entry_is_born_with_nothing_but_its_source():
    entry = CompactionEntry(
        id="cmp",
        parent_id="ts_c",
        created_at=1000,
        source=CompactionSource.USER,
    )

    assert entry == CompactionEntry(
        id="cmp",
        parent_id="ts_c",
        created_at=1000,
        type="compaction",
        context_tokens=0,
        source=CompactionSource.USER,
        parts=None,
        compacted_nodes=None,
        llm_config=None,
        started_at=None,
        ended_at=None,
        metadata={},
    )


def test_compaction_entry_requires_a_source():
    with pytest.raises(ValidationError):
        CompactionEntry(id="cmp", created_at=1000)


def test_compaction_entry_rejects_the_old_field_names():
    with pytest.raises(ValidationError):
        CompactionEntry(
            id="cmp",
            created_at=1000,
            source=CompactionSource.POLICY,
            summary="…",
            summarized=["u1"],
            details={},
        )


def test_compaction_source_members():
    assert [member.value for member in CompactionSource] == ["user", "policy"]


def test_a_committed_compaction_round_trips_inside_a_session():
    session = AgentSession(
        id="s_compacted",
        entries={
            "cmp": CompactionEntry(
                id="cmp",
                parent_id="ts_c",
                created_at=1000,
                source=CompactionSource.POLICY,
                parts=[
                    TextContent(text="the story so far"),
                    ImageContent(
                        source=ImageBase64(data="aGk=", media_type="image/png"),
                    ),
                ],
                compacted_nodes=["u0", "a0"],
                llm_config=LLMConfig(model="cheap", provider="faux"),
                started_at=999,
                ended_at=1000,
                metadata={"strategy": "turn-brackets"},
                context_tokens=1_004,
            ),
        },
        active_conversation=Conversation(
            id="c2",
            nodes=["cmp"],
            created_at=1000,
            updated_at=1000,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )

    reloaded = AgentSession.model_validate_json(session.model_dump_json())

    assert reloaded == session
    assert type(reloaded.entries["cmp"]) is CompactionEntry


# ── uncommitted templates ──────────────────────────────────────────────────────


def test_an_entry_template_carries_no_identity():
    # `None` means "not yet committed" — a registry's birth draft, a pruned
    # replacement, a compaction plan's new entry
    template = CompactionEntry(source=CompactionSource.POLICY)

    assert (template.id, template.parent_id, template.created_at) == (
        None,
        None,
        None,
    )


def test_a_template_still_discriminates_through_the_entry_union():
    template = PrunedEntry(
        pruned_entry_type="tool_execution",
        pruned_entry_id="te1",
        content=[TextContent(text="[pruned]")],
    )

    adapter = TypeAdapter(AnyEntry)

    assert (
        adapter.validate_python(
            template.model_dump(),
        )
        == template
    )


# ── the compaction-bracket predicate ───────────────────────────────────────────


def test_a_turn_start_followed_by_a_compaction_entry_opens_a_compaction_bracket():
    entries = {
        "ts_c": TurnStart(id="ts_c", created_at=1000),
        "cmp": CompactionEntry(
            id="cmp",
            created_at=1000,
            source=CompactionSource.POLICY,
        ),
        "tf_c": TurnFinish(id="tf_c", created_at=1000),
    }

    assert is_compaction_bracket(["ts_c", "cmp", "tf_c"], entries, 0) is True


def test_a_turn_start_followed_by_anything_else_is_conversational():
    entries = {
        "ts": TurnStart(id="ts", created_at=1000),
        "a1": AssistantMessage(
            id="a1",
            created_at=1000,
            parts=[TextContent(text="hi")],
            llm_config=MODEL,
            stop_reason="stop",
        ),
        "cmp": CompactionEntry(
            id="cmp",
            created_at=1000,
            source=CompactionSource.POLICY,
        ),
    }

    # a summary later in the span does NOT make the bracket a compaction one
    assert is_compaction_bracket(["ts", "a1", "cmp"], entries, 0) is False


def test_a_bare_turn_start_is_conversational():
    entries = {"ts": TurnStart(id="ts", created_at=1000)}

    assert is_compaction_bracket(["ts"], entries, 0) is False


# ── turn counting ──────────────────────────────────────────────────────────────


def test_turn_count_counts_conversational_brackets_including_the_open_one():
    session = AgentSession(
        id="s_turns",
        entries={
            "ts1": TurnStart(id="ts1", created_at=1000),
            "tf1": TurnFinish(id="tf1", parent_id="ts1", created_at=1000),
            "ts2": TurnStart(id="ts2", parent_id="tf1", created_at=1000),
            "tf2": TurnFinish(
                id="tf2",
                parent_id="ts2",
                created_at=1000,
                outcome=TurnOutcome.ERRORED,
                error="boom",
            ),
            "ts3": TurnStart(id="ts3", parent_id="tf2", created_at=1000),
        },
        active_conversation=Conversation(
            id="c1",
            nodes=["ts1", "tf1", "ts2", "tf2", "ts3"],
            created_at=1000,
            updated_at=1000,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )

    # a failed bracket counts, and so does the open one before its first
    # assistant message lands
    assert session.session_runtime_status.turn_count == 3


def test_turn_count_excludes_compaction_brackets():
    session = AgentSession(
        id="s_turns_compacted",
        entries={
            "ts1": TurnStart(id="ts1", created_at=1000),
            "tf1": TurnFinish(id="tf1", parent_id="ts1", created_at=1000),
            "ts_c": TurnStart(id="ts_c", parent_id="tf1", created_at=1000),
            "cmp": CompactionEntry(
                id="cmp",
                parent_id="ts_c",
                created_at=1000,
                source=CompactionSource.POLICY,
            ),
            "tf_c": TurnFinish(id="tf_c", parent_id="cmp", created_at=1000),
        },
        active_conversation=Conversation(
            id="c1",
            nodes=["ts1", "tf1", "ts_c", "cmp", "tf_c"],
            created_at=1000,
            updated_at=1000,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )

    assert session.session_runtime_status.turn_count == 1


def test_turn_count_is_scoped_to_the_active_conversation():
    # entries outlive their conversation, so counting the store would include
    # every archived turn
    session = AgentSession(
        id="s_turns_archived",
        entries={
            "ts0": TurnStart(id="ts0", created_at=900),
            "tf0": TurnFinish(id="tf0", parent_id="ts0", created_at=900),
            "cmp": CompactionEntry(
                id="cmp",
                created_at=1000,
                source=CompactionSource.POLICY,
                parts=[TextContent(text="earlier")],
                compacted_nodes=["ts0", "tf0"],
            ),
            "ts1": TurnStart(id="ts1", parent_id="cmp", created_at=1000),
            "tf1": TurnFinish(id="tf1", parent_id="ts1", created_at=1000),
        },
        active_conversation=Conversation(
            id="c2",
            nodes=["cmp", "ts1", "tf1"],
            created_at=1000,
            updated_at=1000,
        ),
        conversation_history=[
            Conversation(
                id="c1",
                nodes=["ts0", "tf0"],
                created_at=900,
                updated_at=900,
            ),
        ],
        session_config=SessionConfig(llm_config=MODEL),
    )

    assert session.session_runtime_status.turn_count == 1
