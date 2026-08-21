from __future__ import annotations

import pytest
from pydantic import ValidationError

from tend._common.errors import ErrorInfo, UnsupportedSchemaVersionError
from tend._common.types import StopReason
from tend.agent.persistence.events import (
    CompactionCompletedEvent,
    CompactionCompletedPayload,
    CompactionStartedEvent,
    CompactionStartedPayload,
    EventType,
    ModelRequestFailedEvent,
    ModelRequestFailedPayload,
    ModelRequestStartedEvent,
    ModelRequestStartedPayload,
    ModelResponseCompletedEvent,
    ModelResponseCompletedPayload,
    RetryScheduledEvent,
    RetryScheduledPayload,
    SessionEvent,
    SessionResumedEvent,
    SessionResumedPayload,
    SessionStartedEvent,
    SessionStartedPayload,
    ToolCallCompletedEvent,
    ToolCallCompletedPayload,
    ToolCallStartedEvent,
    ToolCallStartedPayload,
    TurnCompletedEvent,
    TurnCompletedPayload,
    TurnInterruptedEvent,
    TurnInterruptedPayload,
    TurnStartedEvent,
    TurnStartedPayload,
    dump_event_json,
    event_order_key,
    next_event_sequence,
    parse_event,
    parse_event_json,
)
from tend.llm.models import AssistantMessage, ModelRequest, ModelResponse, TextContent, ToolCall
from tend.llm.models.tools import ToolError, ToolResult
from tend.llm.usage import TokenUsage, Usage

_TIMESTAMP = "2026-05-04T12:00:00Z"


def _sample_model_request() -> ModelRequest:
    return ModelRequest(request_id="model_req_1", model_name="scripted")


def _sample_model_response() -> ModelResponse:
    return ModelResponse(
        response_id="model_resp_1",
        request_id="model_req_1",
        assistant_message=AssistantMessage(
            message_id="msg_assistant_1",
            content=[TextContent(text="done")],
        ),
        stop_reason=StopReason.FINAL_RESPONSE,
        usage=Usage(tokens=TokenUsage(input_tokens=3, output_tokens=2), model_requests=1),
    )


def _sample_tool_call() -> ToolCall:
    return ToolCall(
        call_id="call_1",
        tool_name="ls",
        arguments={"path": "."},
        order=0,
        provider_call_id="provider_call_1",
    )


def _sample_tool_result() -> ToolResult:
    return ToolResult(
        tool_call_id="call_1",
        tool_name="ls",
        arguments={"path": "."},
        success=True,
        output="README.md",
        order=0,
        provider_call_id="provider_call_1",
    )


def _sample_error() -> ErrorInfo:
    return ErrorInfo(code="rate_limit", message="rate limited", details={"category": "rate_limit"})


def _sample_events() -> list[SessionEvent]:
    request = _sample_model_request()
    response = _sample_model_response()
    tool_call = _sample_tool_call()
    tool_result = _sample_tool_result()
    usage = Usage(tokens=TokenUsage(input_tokens=1, output_tokens=1), model_requests=1)

    return [
        SessionStartedEvent(
            event_id="evt_0000000000000001",
            session_id="sess_1",
            sequence=0,
            timestamp=_TIMESTAMP,
            payload=SessionStartedPayload(cwd="/work"),
        ),
        SessionResumedEvent(
            event_id="evt_0000000000000002",
            parent_event_id="evt_0000000000000001",
            session_id="sess_1",
            sequence=1,
            timestamp=_TIMESTAMP,
            payload=SessionResumedPayload(
                resumed_from_event_id="evt_0000000000000001",
                state_event_count=1,
            ),
        ),
        TurnStartedEvent(
            event_id="evt_0000000000000003",
            parent_event_id="evt_0000000000000002",
            session_id="sess_1",
            turn_id="turn_1",
            sequence=2,
            timestamp=_TIMESTAMP,
            payload=TurnStartedPayload(prompt="hello", input_message_id="msg_user_1"),
        ),
        ModelRequestStartedEvent(
            event_id="evt_0000000000000004",
            parent_event_id="evt_0000000000000003",
            session_id="sess_1",
            turn_id="turn_1",
            sequence=3,
            timestamp=_TIMESTAMP,
            payload=ModelRequestStartedPayload(request_id="model_req_1", request=request),
        ),
        ModelResponseCompletedEvent(
            event_id="evt_0000000000000005",
            parent_event_id="evt_0000000000000004",
            session_id="sess_1",
            turn_id="turn_1",
            sequence=4,
            timestamp=_TIMESTAMP,
            payload=ModelResponseCompletedPayload(
                request_id="model_req_1",
                response_id="model_resp_1",
                response=response,
                usage=usage,
            ),
        ),
        ModelRequestFailedEvent(
            event_id="evt_0000000000000006",
            session_id="sess_1",
            turn_id="turn_1",
            sequence=5,
            timestamp=_TIMESTAMP,
            payload=ModelRequestFailedPayload(
                request_id="model_req_2",
                attempt=1,
                error=_sample_error(),
                retryable=True,
            ),
        ),
        RetryScheduledEvent(
            event_id="evt_0000000000000007",
            session_id="sess_1",
            turn_id="turn_1",
            sequence=6,
            timestamp=_TIMESTAMP,
            payload=RetryScheduledPayload(
                request_id="model_req_2",
                attempt=1,
                next_attempt=2,
                delay_seconds=0.5,
                error=_sample_error(),
            ),
        ),
        ToolCallStartedEvent(
            event_id="evt_0000000000000008",
            session_id="sess_1",
            turn_id="turn_1",
            sequence=7,
            timestamp=_TIMESTAMP,
            payload=ToolCallStartedPayload(tool_call=tool_call),
        ),
        ToolCallCompletedEvent(
            event_id="evt_0000000000000009",
            session_id="sess_1",
            turn_id="turn_1",
            sequence=8,
            timestamp=_TIMESTAMP,
            payload=ToolCallCompletedPayload(result=tool_result),
        ),
        CompactionStartedEvent(
            event_id="evt_0000000000000010",
            session_id="sess_1",
            turn_id="turn_1",
            sequence=9,
            timestamp=_TIMESTAMP,
            payload=CompactionStartedPayload(
                compaction_id="compact_1",
                reason="threshold_exceeded",
                planned_message_ids=["msg_old_1"],
            ),
        ),
        CompactionCompletedEvent(
            event_id="evt_0000000000000011",
            session_id="sess_1",
            turn_id="turn_1",
            sequence=10,
            timestamp=_TIMESTAMP,
            payload=CompactionCompletedPayload(
                compaction_id="compact_1",
                summary="Goal and progress summarized.",
                summary_message_id="msg_summary_1",
                covered_message_ids=["msg_old_1"],
                usage=usage,
            ),
        ),
        TurnInterruptedEvent(
            event_id="evt_0000000000000012",
            session_id="sess_1",
            turn_id="turn_1",
            sequence=11,
            timestamp=_TIMESTAMP,
            payload=TurnInterruptedPayload(
                message="interrupted by signal",
                incomplete_event_id="evt_0000000000000008",
            ),
        ),
        TurnCompletedEvent(
            event_id="evt_0000000000000013",
            session_id="sess_1",
            turn_id="turn_1",
            sequence=12,
            timestamp=_TIMESTAMP,
            payload=TurnCompletedPayload(
                stop_reason=StopReason.FINAL_RESPONSE,
                final_response="done",
                usage=usage,
                model_request_count=1,
                tool_call_count=1,
            ),
        ),
    ]


def test_serialization_deserialization_for_each_event_type() -> None:
    for event in _sample_events():
        dumped = dump_event_json(event)
        restored = parse_event_json(dumped)

        assert restored == event


def test_discriminated_union_parses_to_specific_event_type() -> None:
    raw_event = TurnCompletedEvent(
        event_id="evt_done",
        session_id="sess_1",
        turn_id="turn_1",
        timestamp=_TIMESTAMP,
        payload=TurnCompletedPayload(stop_reason=StopReason.FINAL_RESPONSE, final_response="ok"),
    ).model_dump(mode="json")

    restored = parse_event(raw_event)

    assert isinstance(restored, TurnCompletedEvent)
    assert restored.event_type is EventType.TURN_COMPLETED
    assert restored.payload.final_response == "ok"


def test_unsupported_schema_version_fails_clearly() -> None:
    raw_event = _sample_events()[0].model_dump(mode="json")
    raw_event["schema_version"] = 999

    with pytest.raises(UnsupportedSchemaVersionError, match="unsupported event schema version"):
        parse_event(raw_event)


def test_schema_version_bool_fails_as_unsupported_version() -> None:
    raw_event = _sample_events()[0].model_dump(mode="json")
    raw_event["schema_version"] = True

    with pytest.raises(UnsupportedSchemaVersionError, match="unsupported event schema version"):
        parse_event(raw_event)


def test_strict_event_models_reject_unknown_fields() -> None:
    raw_event = _sample_events()[0].model_dump(mode="json")
    raw_event["unexpected"] = "not allowed"

    with pytest.raises(ValidationError):
        parse_event(raw_event)


def test_event_sequence_helpers_order_linear_history_and_leave_none_last() -> None:
    events = [
        SessionStartedEvent(event_id="evt_no_sequence", session_id="sess_1", timestamp=_TIMESTAMP),
        SessionStartedEvent(
            event_id="evt_0000000000000002",
            session_id="sess_1",
            sequence=2,
            timestamp=_TIMESTAMP,
        ),
        SessionStartedEvent(
            event_id="evt_0000000000000001",
            session_id="sess_1",
            sequence=1,
            timestamp=_TIMESTAMP,
        ),
    ]

    assert next_event_sequence(events) == 3
    assert [event.event_id for event in sorted(events, key=event_order_key)] == [
        "evt_0000000000000001",
        "evt_0000000000000002",
        "evt_no_sequence",
    ]


def test_payload_consistency_validators() -> None:
    with pytest.raises(ValidationError, match="request_id must match"):
        ModelRequestStartedPayload(
            request_id="different",
            request=ModelRequest(request_id="model_req_1"),
        )

    with pytest.raises(ValidationError, match="response_id must match"):
        ModelResponseCompletedPayload(
            request_id="model_req_1",
            response_id="different",
            response=_sample_model_response(),
        )

    with pytest.raises(ValidationError, match="next_attempt must be greater"):
        RetryScheduledPayload(
            request_id="model_req_1",
            attempt=2,
            next_attempt=2,
            delay_seconds=0.1,
        )


def test_failed_tool_result_payload_round_trips() -> None:
    failed_result = ToolResult(
        tool_call_id="call_failed",
        tool_name="bash",
        success=False,
        error=ToolError(error_type="handler_exception", message="boom"),
    )
    event = ToolCallCompletedEvent(
        event_id="evt_tool_failed",
        session_id="sess_1",
        turn_id="turn_1",
        timestamp=_TIMESTAMP,
        payload=ToolCallCompletedPayload(result=failed_result),
    )

    restored = parse_event_json(dump_event_json(event))

    assert isinstance(restored, ToolCallCompletedEvent)
    assert restored.payload.result.success is False
    assert restored.payload.result.error is not None
    assert restored.payload.result.error.error_type == "handler_exception"
