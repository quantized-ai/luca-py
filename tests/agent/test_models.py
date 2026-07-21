"""Declarative data-model tests for the tool-execution + approval types and
the runner-lifecycle value objects (TurnOutcome / TurnFinish / CancelRequested
/ RuntimeConfig).

Pure value-object checks — full-object asserts and JSON round-trips, no logic,
no helpers. Locks the execution-lifecycle vocabulary (`ExecutionStatus`), the
orthogonal approval state (`ApprovalStatus` + the ApprovalDecision audit log),
the structured `ToolExecutionError`, and the `ToolExecution` record with its
`raw_tool_call` / optional `tool_spec` split and lifecycle timestamps. The
core carries NO permission vocabulary beyond these: no modes, no rules, no
intents. Cross-field combinations are framework conventions, not validators —
the model must accept what application middleware authors.
"""

import pytest
from pydantic import ValidationError

from luca.agent.core.models import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    CancelRequested,
    Conversation,
    ExecutionResult,
    ExecutionStatus,
    ImageBase64,
    ImageContent,
    ImageFileId,
    ImageURL,
    Inf,
    MilliSeconds,
    PrunedEntry,
    RuntimeConfig,
    Seconds,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    ToolKind,
    ToolSpec,
    TurnFinish,
    TurnOutcome,
    Usage,
    UserMessage,
)

from tests.agent.scenarios import MODEL


def test_tool_spec_defaults_to_other_kind_and_null_namespace_version():
    assert ToolSpec(name="bash") == ToolSpec(
        name="bash",
        description=None,
        metadata=None,
        tool_kind=ToolKind.OTHER,
        namespace=None,
        version=None,
    )


def test_tool_spec_carries_no_invocation_arguments():
    # arguments belong to ToolExecution.raw_tool_call, not the tool snapshot
    with pytest.raises(ValidationError):
        ToolSpec(name="bash", parameters={"command": "ls"})


def test_tool_spec_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        ToolSpec(name="bash", bogus="nope")


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
    assert isinstance(decision.created_at, int) and decision.created_at > 0


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
    result = ExecutionResult(content=[TextContent(text="3")])

    assert result.metadata == {}
    assert result.is_error is False


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
    execution = ToolExecution(
        id="te1", created_at=1, tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1}),
    )

    assert execution.tool_spec is None
    assert execution.extras == {}
    assert execution.approval_status is None
    assert execution.approval_decisions == []
    assert execution.status == ExecutionStatus.PENDING
    assert execution.result is None
    assert execution.error is None
    assert execution.started_at is None
    assert execution.ended_at is None
    assert execution.cancel_signalled_at is None
    assert execution.updated_at is None
    assert execution.is_doom_loop_flagged is False


def test_tool_execution_requires_raw_tool_call():
    with pytest.raises(ValidationError):
        ToolExecution(id="te1", created_at=1, tool_call_id="tc1")


def test_tool_execution_dispatched_and_duration_are_derived():
    undispatched = ToolExecution(
        id="te1", created_at=1, tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add"),
    )
    settled = ToolExecution(
        id="te2", created_at=1, tool_call_id="tc2",
        raw_tool_call=ToolCall(id="tc2", name="add"),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="3")]),
        started_at=1000, ended_at=1250,
    )

    assert undispatched.dispatched is False
    assert undispatched.duration_ms is None
    assert settled.dispatched is True
    assert settled.duration_ms == 250


def test_tool_execution_does_not_enforce_cross_field_invariants():
    # framework conventions, not schema validation: middleware-authored
    # combinations must construct (the application owns the consequences)
    unusual = ToolExecution(
        id="te1", created_at=1, tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add"),
        status=ExecutionStatus.COMPLETED,  # no result, error present
        error=ToolExecutionError(error_type="X", error_message="authored"),
        approval_status=ApprovalStatus.REJECTED,
        ended_at=5,
    )
    assert unusual.status == ExecutionStatus.COMPLETED


def test_versioned_tool_execution_round_trips_through_json():
    execution = ToolExecution(
        id="caf0ab9ac", parent_id="d4e5f6a7", created_at=1780495331220,
        tool_call_id="toolu_01Tg",
        raw_tool_call=ToolCall(
            id="toolu_01Tg", name="bash", arguments={"command": "pytest -x -q"},
        ),
        tool_spec=ToolSpec(
            name="bash", tool_kind=ToolKind.EXECUTE,
            namespace="builtin.shell_tools", version="0.0.1",
        ),
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
                decision=ApprovalOption.PENDING, created_at=1780495331220,
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
        id="te1", created_at=1780495331220, tool_call_id="tc1",
        raw_tool_call=ToolCall(
            id="tc1", name="read_file", arguments={"encoding": "utf-16"},
        ),
        tool_spec=ToolSpec(name="read_file"),
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
        id="tf", parent_id=None, created_at=1,
        outcome=TurnOutcome.COMPLETED, error=None,
    )


def test_turn_finish_round_trips_with_outcome_and_error():
    finish = TurnFinish(
        id="tf", parent_id="a1", created_at=1780495331220,
        outcome=TurnOutcome.TIMED_OUT, error="client timeout after 30s",
    )
    assert TurnFinish.model_validate_json(finish.model_dump_json()) == finish


def test_cancel_requested_defaults_to_cancelled_outcome():
    assert CancelRequested(id="cr", created_at=1) == CancelRequested(
        id="cr", parent_id=None, created_at=1,
        outcome=TurnOutcome.CANCELLED, error=None,
    )


def test_cancel_requested_rejects_completed_outcome():
    with pytest.raises(ValidationError):
        CancelRequested(id="cr", created_at=1, outcome=TurnOutcome.COMPLETED)


def test_cancel_requested_round_trips():
    entry = CancelRequested(
        id="cr", parent_id="te1", created_at=1780495331220,
        outcome=TurnOutcome.ERRORED, error="abandoned at the approval gate",
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
        llm_config=MODEL, runtime_config=RuntimeConfig(),
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
            id="a1", created_at=1, parts=[TextContent(text="hi")],
            llm_config=MODEL, stop_reason="stop",
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
        conversation_id="c1", entry_id="a1",
        input=0, output=0, cache_read=0, cache_write=0, total_tokens=0,
    )


def test_conversation_requires_a_stable_id():
    with pytest.raises(ValidationError):
        Conversation(nodes=[], created_at=0, updated_at=0)


def test_pruned_entry_round_trips_inside_a_session():
    session = AgentSession(
        id="s",
        entries={
            "te1": ToolExecution(
                id="te1", created_at=1, tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add", arguments={}),
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(content=[TextContent(text="3")]),
                started_at=1, ended_at=1,
            ),
            "p1": PrunedEntry(
                id="p1", created_at=2,
                pruned_entry_type="tool_execution",
                pruned_entry_id="te1",
                content=[TextContent(text="[pruned]")],
                context_tokens=2,
            ),
        },
        tool_executions={"tc1": ["te1"]},
        usages={
            "c1": {
                "a1": Usage(
                    conversation_id="c1", entry_id="a1",
                    input=10, output=5, total_tokens=15,
                ),
            },
        },
        active_conversation=Conversation(
            id="c1", nodes=["p1"], created_at=0, updated_at=2,
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
        id="u1", created_at=1000,
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
