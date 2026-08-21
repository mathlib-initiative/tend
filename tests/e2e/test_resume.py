from __future__ import annotations

from pathlib import Path

from pydantic import Field

from tend import Agent
from tend._common.types import StopReason, StrictModel
from tend.agent.config import CompactionConfig, RuntimeConfig
from tend.agent.context import assistant_tool_calls
from tend.agent.persistence.events import (
    EventType,
    ModelRequestStartedEvent,
    ModelRequestStartedPayload,
    ModelResponseCompletedEvent,
    ModelResponseCompletedPayload,
    SessionEvent,
    ToolCallCompletedEvent,
    ToolCallCompletedPayload,
    ToolCallStartedEvent,
    ToolCallStartedPayload,
    TurnStartedEvent,
    TurnStartedPayload,
)
from tend.agent.persistence.state import session_state_from_events
from tend.agent.session import Session
from tend.agent.tools import Tool, ToolContext
from tend.llm.models import (
    AssistantMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    ToolCall,
    ToolResult,
    ToolResultMessage,
)
from tend.llm.testing import ScriptedModel


class EchoArguments(StrictModel):
    text: str = Field(min_length=1)


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(compaction=CompactionConfig(enabled=False))


def _final_response(text: str = "done", *, response_id: str = "model_resp_final") -> ModelResponse:
    return ModelResponse(
        response_id=response_id,
        assistant_message=AssistantMessage(content=[TextContent(text=text)]),
        stop_reason=StopReason.FINAL_RESPONSE,
    )


def _echo_tool(seen: list[str] | None = None) -> Tool[EchoArguments]:
    async def handler(_context: ToolContext, arguments: EchoArguments) -> dict[str, str]:
        if seen is not None:
            seen.append(arguments.text)
        return {"echo": arguments.text}

    return Tool.from_arguments_model(
        name="echo",
        description="Echo text for persistence/resume tests.",
        arguments_model=EchoArguments,
        handler=handler,
    )


def _tool_call(call_id: str, text: str, *, order: int) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        tool_name="echo",
        arguments={"text": text},
        order=order,
    )


def _append_turn_started(session: Session, *, turn_id: str) -> TurnStartedEvent:
    event = TurnStartedEvent(
        session_id=session.session_id,
        turn_id=turn_id,
        parent_event_id=session.last_event_id,
        sequence=session.next_sequence,
        payload=TurnStartedPayload(prompt="crashed turn", input_message_id=f"msg_{turn_id}"),
    )
    session.append_event(event)
    return event


def _append_model_request_started(
    session: Session,
    *,
    turn_id: str,
    request_id: str,
) -> ModelRequestStartedEvent:
    request = ModelRequest(request_id=request_id, model_name="scripted")
    event = ModelRequestStartedEvent(
        session_id=session.session_id,
        turn_id=turn_id,
        parent_event_id=session.last_event_id,
        sequence=session.next_sequence,
        payload=ModelRequestStartedPayload(request_id=request_id, request=request),
    )
    session.append_event(event)
    return event


def _append_model_response_completed(
    session: Session,
    *,
    turn_id: str,
    request_id: str,
    response: ModelResponse,
) -> ModelResponseCompletedEvent:
    event = ModelResponseCompletedEvent(
        session_id=session.session_id,
        turn_id=turn_id,
        parent_event_id=session.last_event_id,
        sequence=session.next_sequence,
        payload=ModelResponseCompletedPayload(
            request_id=request_id,
            response_id=response.response_id,
            response=response.model_copy(update={"request_id": request_id}, deep=True),
        ),
    )
    session.append_event(event)
    return event


def _append_tool_call_started(
    session: Session,
    *,
    turn_id: str,
    tool_call: ToolCall,
) -> ToolCallStartedEvent:
    event = ToolCallStartedEvent(
        session_id=session.session_id,
        turn_id=turn_id,
        parent_event_id=session.last_event_id,
        sequence=session.next_sequence,
        payload=ToolCallStartedPayload(tool_call=tool_call),
    )
    session.append_event(event)
    return event


def _append_tool_call_completed(
    session: Session,
    *,
    turn_id: str,
    tool_call: ToolCall,
    output: str,
) -> ToolCallCompletedEvent:
    event = ToolCallCompletedEvent(
        session_id=session.session_id,
        turn_id=turn_id,
        parent_event_id=session.last_event_id,
        sequence=session.next_sequence,
        payload=ToolCallCompletedPayload(
            result=ToolResult(
                tool_call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                arguments=tool_call.arguments,
                success=True,
                output=output,
                order=tool_call.order,
            )
        ),
    )
    session.append_event(event)
    return event


def _events_of_type(events: list[SessionEvent], event_type: EventType) -> list[SessionEvent]:
    return [event for event in events if event.event_type is event_type]


async def test_completed_turn_event_log_replays_to_equivalent_state_snapshot(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    model = ScriptedModel([_final_response("persisted done", response_id="model_resp_done")])
    agent = Agent("System prompt.", model=model, model_name="scripted")

    with Session.create(session_dir, session_id="sess_replay", sync_writes=False) as session:
        result = await agent.run_turn(
            "Persist this turn.",
            session=session,
            config=_runtime_config(),
        )
        events = session.event_store.read_all()
        snapshot = session.snapshot_store.read()

    rebuilt = session_state_from_events(events)

    assert result.final_response == "persisted done"
    assert snapshot == rebuilt
    assert result.session_state == rebuilt
    assert rebuilt.incomplete_model_requests == {}
    assert rebuilt.interrupted_tool_calls == {}
    assert len(rebuilt.completed_model_requests) == 1
    completed_request = next(iter(rebuilt.completed_model_requests.values()))
    assert completed_request.response_id == "model_resp_done"

    lines = (session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == rebuilt.event_count
    assert '"event_type":"TurnCompleted"' in lines[-1]
    assert _events_of_type(events, EventType.MODEL_REQUEST_STARTED)
    assert _events_of_type(events, EventType.MODEL_RESPONSE_COMPLETED)


async def test_resume_after_incomplete_model_request_keeps_retry_state_and_starts_fresh_request(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    with Session.create(
        session_dir,
        session_id="sess_incomplete_model",
        sync_writes=False,
    ) as session:
        _append_turn_started(session, turn_id="turn_crashed_model")
        _append_model_request_started(
            session,
            turn_id="turn_crashed_model",
            request_id="model_req_crashed",
        )

    model = ScriptedModel([_final_response("after crash", response_id="model_resp_after")])
    agent = Agent("System prompt.", model=model, model_name="scripted")

    with Session.resume(session_dir, sync_writes=False) as resumed:
        before = resumed.state
        assert set(before.incomplete_model_requests) == {"model_req_crashed"}
        assert before.incomplete_model_requests["model_req_crashed"].request is not None

        result = await agent.run_turn(
            "Continue safely.",
            session=resumed,
            config=_runtime_config(),
        )
        after = resumed.state

    assert result.final_response == "after crash"
    assert len(model.requests) == 1
    assert model.requests[0].request_id != "model_req_crashed"
    assert "model_req_crashed" in after.incomplete_model_requests
    assert any(
        completed.response_id == "model_resp_after"
        for completed in after.completed_model_requests.values()
    )


async def test_resume_after_incomplete_tool_call_surfaces_interruption_without_rerun(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    interrupted_call = _tool_call("call_crashed", "must not run", order=0)
    with Session.create(
        session_dir,
        session_id="sess_incomplete_tool",
        sync_writes=False,
    ) as session:
        _append_turn_started(session, turn_id="turn_crashed_tool")
        _append_tool_call_started(
            session,
            turn_id="turn_crashed_tool",
            tool_call=interrupted_call,
        )

    seen: list[str] = []
    model = ScriptedModel([_final_response("recovered")])
    agent = Agent(
        "System prompt.",
        model=model,
        tools=[_echo_tool(seen)],
        model_name="scripted",
    )

    with Session.resume(session_dir, sync_writes=False) as resumed:
        assert set(resumed.state.interrupted_tool_calls) == {"call_crashed"}
        interrupted = resumed.state.interrupted_tool_calls["call_crashed"]
        assert interrupted.interrupted_event_id is None

        result = await agent.run_turn(
            "Recover from crash.",
            session=resumed,
            config=_runtime_config(),
        )
        events = resumed.event_store.read_all()

    assert result.final_response == "recovered"
    assert seen == []
    first_request = model.requests[0]
    assert [message.role.value for message in first_request.messages] == [
        "system",
        "assistant",
        "tool",
        "user",
    ]
    assistant = first_request.messages[1]
    assert isinstance(assistant, AssistantMessage)
    assert assistant_tool_calls(assistant) == (interrupted_call,)
    tool_message = first_request.messages[2]
    assert isinstance(tool_message, ToolResultMessage)
    assert tool_message.tool_call_id == "call_crashed"
    assert tool_message.result.success is False
    assert tool_message.result.error is not None
    assert tool_message.result.error.error_type == "interrupted"
    assert [
        event.payload.result.tool_call_id
        for event in events
        if isinstance(event, ToolCallCompletedEvent)
    ] == []


async def test_multi_tool_resume_preserves_completed_tool_and_does_not_rerun_effects(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    completed_call = _tool_call("call_completed", "already done", order=0)
    interrupted_call = _tool_call("call_pending", "was interrupted", order=1)
    with Session.create(session_dir, session_id="sess_multi_tool", sync_writes=False) as session:
        _append_turn_started(session, turn_id="turn_multi_tool")
        _append_model_request_started(
            session,
            turn_id="turn_multi_tool",
            request_id="model_req_tools",
        )
        _append_model_response_completed(
            session,
            turn_id="turn_multi_tool",
            request_id="model_req_tools",
            response=ModelResponse(
                response_id="model_resp_tools",
                tool_calls=[completed_call, interrupted_call],
            ),
        )
        _append_tool_call_started(session, turn_id="turn_multi_tool", tool_call=completed_call)
        _append_tool_call_completed(
            session,
            turn_id="turn_multi_tool",
            tool_call=completed_call,
            output="first result",
        )
        _append_tool_call_started(session, turn_id="turn_multi_tool", tool_call=interrupted_call)

    seen: list[str] = []
    model = ScriptedModel([_final_response("continued")])
    agent = Agent(
        "System prompt.",
        model=model,
        tools=[_echo_tool(seen)],
        model_name="scripted",
    )

    with Session.resume(session_dir, sync_writes=False) as resumed:
        before = resumed.state
        assert set(before.completed_tool_calls) == {"call_completed"}
        assert set(before.interrupted_tool_calls) == {"call_pending"}

        result = await agent.run_turn(
            "Continue without redoing completed tools.",
            session=resumed,
            config=_runtime_config(),
        )
        after = resumed.state
        events = resumed.event_store.read_all()

    assert result.final_response == "continued"
    assert seen == []
    assert set(after.completed_tool_calls) == {"call_completed"}
    assert set(after.interrupted_tool_calls) == {"call_pending"}
    assert after.completed_tool_calls["call_completed"].result.output == "first result"
    started_tool_call_ids = [
        event.payload.tool_call.call_id
        for event in events
        if isinstance(event, ToolCallStartedEvent)
    ]
    assert started_tool_call_ids == ["call_completed", "call_pending"]
    assert [
        event.payload.result.tool_call_id
        for event in events
        if isinstance(event, ToolCallCompletedEvent)
    ] == ["call_completed"]
