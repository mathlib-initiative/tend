from __future__ import annotations

from pathlib import Path

import pytest

from tend._common.errors import ErrorInfo, PersistenceError
from tend._common.types import StopReason
from tend.agent.persistence.events import (
    ModelRequestFailedEvent,
    ModelRequestFailedPayload,
    ModelRequestStartedEvent,
    ModelRequestStartedPayload,
    ModelResponseCompletedEvent,
    ModelResponseCompletedPayload,
    SessionStartedEvent,
    ToolCallCompletedEvent,
    ToolCallCompletedPayload,
    ToolCallStartedEvent,
    ToolCallStartedPayload,
    TurnCompletedEvent,
    TurnCompletedPayload,
    TurnInterruptedEvent,
    TurnInterruptedPayload,
    TurnStartedEvent,
)
from tend.agent.persistence.replay import replay_events
from tend.agent.persistence.state import SessionState
from tend.agent.session import Session
from tend.llm.models import ModelRequest, ModelResponse, ToolCall
from tend.llm.models.provider import ProviderMetadata
from tend.llm.models.tools import ToolResult

_TIMESTAMP = "2026-05-04T12:00:00Z"


def _started(event_id: str = "evt_0000000000000001") -> SessionStartedEvent:
    return SessionStartedEvent(
        event_id=event_id,
        session_id="sess_1",
        sequence=0,
        timestamp=_TIMESTAMP,
    )


def _turn_started() -> TurnStartedEvent:
    return TurnStartedEvent(
        event_id="evt_0000000000000002",
        parent_event_id="evt_0000000000000001",
        session_id="sess_1",
        turn_id="turn_1",
        sequence=1,
        timestamp=_TIMESTAMP,
    )


def _model_request_started(
    *,
    event_id: str = "evt_0000000000000003",
    request_id: str = "model_req_1",
    sequence: int = 2,
    attempt: int = 1,
) -> ModelRequestStartedEvent:
    return ModelRequestStartedEvent(
        event_id=event_id,
        session_id="sess_1",
        turn_id="turn_1",
        sequence=sequence,
        timestamp=_TIMESTAMP,
        payload=ModelRequestStartedPayload(
            request_id=request_id,
            attempt=attempt,
            request=ModelRequest(request_id=request_id, model_name="scripted"),
        ),
    )


def _model_response_completed() -> ModelResponseCompletedEvent:
    return ModelResponseCompletedEvent(
        event_id="evt_0000000000000004",
        session_id="sess_1",
        turn_id="turn_1",
        sequence=3,
        timestamp=_TIMESTAMP,
        payload=ModelResponseCompletedPayload(
            request_id="model_req_1",
            response_id="model_resp_1",
            response=ModelResponse(
                response_id="model_resp_1",
                request_id="model_req_1",
                stop_reason=StopReason.FINAL_RESPONSE,
                provider_metadata=ProviderMetadata(
                    provider_name="openai_responses",
                    response_id="resp_provider_1",
                ),
            ),
        ),
    )


def _tool_call_started(
    *,
    event_id: str = "evt_0000000000000005",
    sequence: int = 4,
) -> ToolCallStartedEvent:
    return ToolCallStartedEvent(
        event_id=event_id,
        session_id="sess_1",
        turn_id="turn_1",
        sequence=sequence,
        timestamp=_TIMESTAMP,
        payload=ToolCallStartedPayload(
            tool_call=ToolCall(
                call_id="call_1",
                tool_name="ls",
                arguments={"path": "."},
                order=0,
                provider_call_id="provider_call_1",
            )
        ),
    )


def _tool_call_completed() -> ToolCallCompletedEvent:
    return ToolCallCompletedEvent(
        event_id="evt_0000000000000006",
        session_id="sess_1",
        turn_id="turn_1",
        sequence=5,
        timestamp=_TIMESTAMP,
        payload=ToolCallCompletedPayload(
            result=ToolResult(
                tool_call_id="call_1",
                tool_name="ls",
                arguments={"path": "."},
                success=True,
                output="README.md",
                order=0,
                provider_call_id="provider_call_1",
            )
        ),
    )


def test_replay_complete_turn_tracks_completed_model_and_tool_operations() -> None:
    events = [
        _started(),
        _turn_started(),
        _model_request_started(),
        _model_response_completed(),
        _tool_call_started(),
        _tool_call_completed(),
        TurnCompletedEvent(
            event_id="evt_0000000000000007",
            session_id="sess_1",
            turn_id="turn_1",
            sequence=6,
            timestamp=_TIMESTAMP,
            payload=TurnCompletedPayload(
                stop_reason=StopReason.FINAL_RESPONSE,
                final_response="done",
                model_request_count=1,
                tool_call_count=1,
            ),
        ),
    ]

    state = replay_events(events)

    assert state.event_count == 7
    assert state.last_event_id == "evt_0000000000000007"
    assert state.incomplete_model_requests == {}
    assert state.interrupted_tool_calls == {}

    completed_model = state.completed_model_requests["model_req_1"]
    assert completed_model.started_event_id == "evt_0000000000000003"
    assert completed_model.completed_event_id == "evt_0000000000000004"
    assert completed_model.response_id == "model_resp_1"
    assert completed_model.provider_response_id == "resp_provider_1"
    assert completed_model.provider_name == "openai_responses"
    assert state.model_request_start_event_ids == {
        "evt_0000000000000003": "model_req_1"
    }
    assert state.model_response_completed_event_ids == {
        "evt_0000000000000004": "model_req_1"
    }
    assert state.provider_response_ids == {"resp_provider_1": "model_req_1"}

    completed_tool = state.completed_tool_calls["call_1"]
    assert completed_tool.started_event_id == "evt_0000000000000005"
    assert completed_tool.completed_event_id == "evt_0000000000000006"
    assert completed_tool.result.success is True
    assert state.tool_call_start_event_ids == {"evt_0000000000000005": "call_1"}
    assert state.tool_call_completed_event_ids == {"evt_0000000000000006": "call_1"}


def test_replay_detects_incomplete_model_request() -> None:
    state = replay_events([_started(), _turn_started(), _model_request_started()])

    assert state.completed_model_requests == {}
    incomplete = state.incomplete_model_requests["model_req_1"]
    assert incomplete.started_event_id == "evt_0000000000000003"
    assert incomplete.request is not None
    assert incomplete.request.request_id == "model_req_1"
    assert incomplete.attempt == 1


def test_replay_tracks_failed_attempts_before_incomplete_model_retry() -> None:
    error = ErrorInfo(code="rate_limit", message="rate limited")
    first_start = _model_request_started(event_id="evt_start_1", sequence=2, attempt=1)
    failed = ModelRequestFailedEvent(
        event_id="evt_failed_1",
        session_id="sess_1",
        turn_id="turn_1",
        sequence=3,
        timestamp=_TIMESTAMP,
        payload=ModelRequestFailedPayload(
            request_id="model_req_1",
            attempt=1,
            error=error,
            retryable=True,
        ),
    )
    retry_start = _model_request_started(
        event_id="evt_start_2",
        sequence=4,
        attempt=2,
    )

    state = replay_events([_started(), _turn_started(), first_start, failed, retry_start])

    incomplete = state.incomplete_model_requests["model_req_1"]
    assert incomplete.started_event_id == "evt_start_2"
    assert incomplete.attempt == 2
    assert incomplete.failed_event_ids == ["evt_failed_1"]
    assert incomplete.last_failed_event_id == "evt_failed_1"
    assert incomplete.last_error == error


def test_replay_marks_incomplete_tool_call_as_interrupted_result() -> None:
    tool_started = _tool_call_started()
    interrupted = TurnInterruptedEvent(
        event_id="evt_0000000000000006",
        session_id="sess_1",
        turn_id="turn_1",
        sequence=5,
        timestamp=_TIMESTAMP,
        payload=TurnInterruptedPayload(
            message="interrupted by signal",
            incomplete_event_id=tool_started.event_id,
        ),
    )

    state = replay_events([_started(), _turn_started(), tool_started, interrupted])

    interrupted_tool = state.interrupted_tool_calls["call_1"]
    assert interrupted_tool.started_event_id == tool_started.event_id
    assert interrupted_tool.interrupted_event_id == interrupted.event_id
    assert interrupted_tool.result.success is False
    assert interrupted_tool.result.error is not None
    assert interrupted_tool.result.error.error_type == "interrupted"
    assert interrupted_tool.result.provider_call_id == "provider_call_1"
    assert state.completed_tool_calls == {}


def test_replay_state_json_cache_can_be_rebuilt_from_events(tmp_path: Path) -> None:
    created = Session.create(tmp_path, session_id="sess_1", sync_writes=False)
    created.close()
    stale = SessionState(session_id="sess_1", event_count=99, last_event_id="evt_stale")
    (tmp_path / "state.json").write_text(stale.model_dump_json() + "\n", encoding="utf-8")

    resumed = Session.open(tmp_path, sync_writes=False)
    try:
        assert resumed.event_count == 2
        assert resumed.state.event_count == 2
        assert resumed.state.last_event_id != "evt_stale"
        assert resumed.snapshot_store.read() == resumed.state
    finally:
        resumed.close()


def test_replay_rejects_inconsistent_terminal_events() -> None:
    response_without_start = _model_response_completed()

    with pytest.raises(PersistenceError, match="no matching start"):
        replay_events([_started(), _turn_started(), response_without_start])

    with pytest.raises(PersistenceError, match="no matching start"):
        replay_events([_started(), _turn_started(), _tool_call_completed()])


def test_replay_empty_events_requires_session_id() -> None:
    with pytest.raises(ValueError, match="session_id is required"):
        replay_events([])

    assert replay_events([], session_id="sess_empty") == SessionState(session_id="sess_empty")
