"""Deterministic event replay for resumable sessions."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import TypeAdapter

from tend._common.errors import PersistenceError
from tend._common.types import JsonObject
from tend.agent.persistence.events import (
    CompactionCompletedEvent,
    CompactionStartedEvent,
    EventBase,
    ModelRequestFailedEvent,
    ModelRequestStartedEvent,
    ModelResponseCompletedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    TurnInterruptedEvent,
)
from tend.agent.persistence.state import (
    CompletedCompaction,
    CompletedModelRequest,
    CompletedToolCall,
    IncompleteModelRequest,
    InterruptedToolCall,
    SessionState,
)
from tend.llm.context_estimation import (
    CONTEXT_ESTIMATE_METADATA_KEY,
    ContextEstimate,
    context_estimate_from_metadata,
)
from tend.llm.models.requests import ModelRequest, ModelResponse
from tend.llm.models.tools import ToolError, ToolResult
from tend.llm.usage import Usage, usage_with_model_request_count

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


def replay_events(
    events: Iterable[EventBase],
    *,
    session_id: str | None = None,
) -> SessionState:
    """Replay canonical events into a side-effect-free session state cache.

    Completed model responses and tool results are recorded so future resume
    logic can avoid rerunning completed nondeterministic effects. Started model
    requests without a response/failure are marked incomplete. Started tool
    calls without a completion event become interrupted model-visible results.
    """

    resolved_session_id = session_id
    event_count = 0
    last_event_id: str | None = None
    last_sequence: int | None = None

    open_model_requests: dict[str, ModelRequestStartedEvent] = {}
    model_failed_event_ids: dict[str, list[str]] = {}
    model_last_failure: dict[str, ModelRequestFailedEvent] = {}
    completed_model_requests: dict[str, CompletedModelRequest] = {}
    model_request_start_event_ids: dict[str, str] = {}
    model_response_completed_event_ids: dict[str, str] = {}
    provider_response_ids: dict[str, str] = {}

    open_tool_calls: dict[str, ToolCallStartedEvent] = {}
    completed_tool_calls: dict[str, CompletedToolCall] = {}
    tool_call_start_event_ids: dict[str, str] = {}
    tool_call_completed_event_ids: dict[str, str] = {}

    interruption_event_ids: dict[str, str] = {}

    open_compactions: dict[str, CompactionStartedEvent] = {}
    completed_compactions: dict[str, CompletedCompaction] = {}

    session_usage = Usage()
    turn_usage: dict[str, Usage] = {}
    model_request_usage: dict[str, Usage] = {}
    compaction_usage: dict[str, Usage] = {}
    model_request_context_estimates: dict[str, ContextEstimate] = {}
    latest_context_estimate: ContextEstimate | None = None

    for event in events:
        resolved_session_id = _resolve_session_id(
            current=resolved_session_id,
            event=event,
        )
        event_count += 1
        last_event_id = event.event_id
        last_sequence = event.sequence

        if isinstance(event, ModelRequestStartedEvent):
            _record_model_request_started(
                event,
                open_model_requests=open_model_requests,
                completed_model_requests=completed_model_requests,
                model_request_start_event_ids=model_request_start_event_ids,
            )
            context_estimate = _context_estimate_from_request(event.payload.request)
            if context_estimate is not None:
                model_request_context_estimates[event.payload.request_id] = context_estimate
                latest_context_estimate = context_estimate
        elif isinstance(event, ModelRequestFailedEvent):
            _record_model_request_failed(
                event,
                open_model_requests=open_model_requests,
                model_failed_event_ids=model_failed_event_ids,
                model_last_failure=model_last_failure,
            )
            usage_delta = event.payload.usage
            if not _is_empty_usage(usage_delta):
                session_usage = session_usage.add(usage_delta)
                _add_turn_usage(turn_usage, event.turn_id, usage_delta)
                model_request_usage[event.payload.request_id] = usage_delta
        elif isinstance(event, ModelResponseCompletedEvent):
            _record_model_response_completed(
                event,
                open_model_requests=open_model_requests,
                model_failed_event_ids=model_failed_event_ids,
                completed_model_requests=completed_model_requests,
                model_response_completed_event_ids=model_response_completed_event_ids,
                provider_response_ids=provider_response_ids,
            )
            usage_delta = _model_response_event_usage(event)
            session_usage = session_usage.add(usage_delta)
            _add_turn_usage(turn_usage, event.turn_id, usage_delta)
            model_request_usage[event.payload.request_id] = usage_delta
        elif isinstance(event, ToolCallStartedEvent):
            _record_tool_call_started(
                event,
                open_tool_calls=open_tool_calls,
                completed_tool_calls=completed_tool_calls,
                tool_call_start_event_ids=tool_call_start_event_ids,
            )
        elif isinstance(event, ToolCallCompletedEvent):
            _record_tool_call_completed(
                event,
                open_tool_calls=open_tool_calls,
                completed_tool_calls=completed_tool_calls,
                tool_call_completed_event_ids=tool_call_completed_event_ids,
            )
            usage_delta = Usage(tool_calls=1)
            session_usage = session_usage.add(usage_delta)
            _add_turn_usage(turn_usage, event.turn_id, usage_delta)
        elif isinstance(event, TurnInterruptedEvent):
            incomplete_event_id = event.payload.incomplete_event_id
            if incomplete_event_id is not None:
                interruption_event_ids[incomplete_event_id] = event.event_id
        elif isinstance(event, CompactionStartedEvent):
            _record_compaction_started(
                event,
                open_compactions=open_compactions,
                completed_compactions=completed_compactions,
            )
        elif isinstance(event, CompactionCompletedEvent):
            completed = _record_compaction_completed(
                event,
                open_compactions=open_compactions,
                completed_compactions=completed_compactions,
            )
            usage_delta = completed.usage
            if not _is_empty_usage(usage_delta):
                session_usage = session_usage.add(usage_delta)
                _add_turn_usage(turn_usage, event.turn_id, usage_delta)
                compaction_usage[event.payload.compaction_id] = usage_delta

    if resolved_session_id is None:
        raise ValueError("session_id is required when replaying an empty event log")

    incomplete_model_requests = _incomplete_model_requests(
        open_model_requests=open_model_requests,
        model_failed_event_ids=model_failed_event_ids,
        model_last_failure=model_last_failure,
        interruption_event_ids=interruption_event_ids,
    )
    interrupted_tool_calls = _interrupted_tool_calls(
        open_tool_calls=open_tool_calls,
        interruption_event_ids=interruption_event_ids,
    )

    return SessionState(
        session_id=resolved_session_id,
        event_count=event_count,
        last_event_id=last_event_id,
        last_sequence=last_sequence,
        usage=session_usage,
        turn_usage=turn_usage,
        model_request_usage=model_request_usage,
        compaction_usage=compaction_usage,
        completed_compactions=completed_compactions,
        model_request_context_estimates=model_request_context_estimates,
        latest_context_estimate=latest_context_estimate,
        completed_model_requests=completed_model_requests,
        incomplete_model_requests=incomplete_model_requests,
        model_request_start_event_ids=model_request_start_event_ids,
        model_response_completed_event_ids=model_response_completed_event_ids,
        provider_response_ids=provider_response_ids,
        completed_tool_calls=completed_tool_calls,
        interrupted_tool_calls=interrupted_tool_calls,
        tool_call_start_event_ids=tool_call_start_event_ids,
        tool_call_completed_event_ids=tool_call_completed_event_ids,
    )


def _resolve_session_id(*, current: str | None, event: EventBase) -> str:
    if current is None:
        return event.session_id
    if event.session_id != current:
        raise ValueError("all events in a session replay must share one session_id")
    return current


def _record_model_request_started(
    event: ModelRequestStartedEvent,
    *,
    open_model_requests: dict[str, ModelRequestStartedEvent],
    completed_model_requests: dict[str, CompletedModelRequest],
    model_request_start_event_ids: dict[str, str],
) -> None:
    request_id = event.payload.request_id
    if request_id in completed_model_requests:
        raise PersistenceError(
            f"model request {request_id!r} was already completed before a later start event"
        )
    if request_id in open_model_requests:
        raise PersistenceError(
            f"model request {request_id!r} has multiple starts without a terminal event"
        )
    open_model_requests[request_id] = event
    model_request_start_event_ids[event.event_id] = request_id


def _record_model_request_failed(
    event: ModelRequestFailedEvent,
    *,
    open_model_requests: dict[str, ModelRequestStartedEvent],
    model_failed_event_ids: dict[str, list[str]],
    model_last_failure: dict[str, ModelRequestFailedEvent],
) -> None:
    request_id = event.payload.request_id
    started = open_model_requests.pop(request_id, None)
    if started is None:
        raise PersistenceError(
            f"model request failure {event.event_id!r} has no matching start for {request_id!r}"
        )
    model_failed_event_ids.setdefault(request_id, []).append(event.event_id)
    model_last_failure[request_id] = event


def _record_model_response_completed(
    event: ModelResponseCompletedEvent,
    *,
    open_model_requests: dict[str, ModelRequestStartedEvent],
    model_failed_event_ids: dict[str, list[str]],
    completed_model_requests: dict[str, CompletedModelRequest],
    model_response_completed_event_ids: dict[str, str],
    provider_response_ids: dict[str, str],
) -> None:
    request_id = event.payload.request_id
    started = open_model_requests.pop(request_id, None)
    if started is None:
        raise PersistenceError(
            f"model response {event.event_id!r} has no matching start for {request_id!r}"
        )
    provider_response_id, provider_name = _provider_response_info(event.payload.response)
    completed_model_requests[request_id] = CompletedModelRequest(
        request_id=request_id,
        started_event_id=started.event_id,
        completed_event_id=event.event_id,
        turn_id=event.turn_id or started.turn_id,
        attempt=started.payload.attempt,
        response_id=event.payload.response_id,
        provider_response_id=provider_response_id,
        provider_name=provider_name,
        failed_event_ids=list(model_failed_event_ids.get(request_id, [])),
    )
    model_response_completed_event_ids[event.event_id] = request_id
    if provider_response_id is not None:
        if provider_response_id in provider_response_ids:
            raise PersistenceError(
                f"provider response ID {provider_response_id!r} appears more than once"
            )
        provider_response_ids[provider_response_id] = request_id


def _provider_response_info(response: ModelResponse | None) -> tuple[str | None, str | None]:
    if response is None or response.provider_metadata is None:
        return (None, None)
    return (response.provider_metadata.response_id, response.provider_metadata.provider_name)


def _model_response_event_usage(event: ModelResponseCompletedEvent) -> Usage:
    usage = event.payload.usage
    response = event.payload.response
    if _is_empty_usage(usage) and response is not None:
        usage = response.usage
    return usage_with_model_request_count(usage)


def _context_estimate_from_request(request: ModelRequest | None) -> ContextEstimate | None:
    if request is None:
        return None
    return context_estimate_from_metadata(
        request.request_metadata.get(CONTEXT_ESTIMATE_METADATA_KEY)
    )


def _add_turn_usage(
    turn_usage: dict[str, Usage],
    turn_id: str | None,
    usage_delta: Usage,
) -> None:
    if turn_id is None:
        return
    current = turn_usage.get(turn_id, Usage())
    turn_usage[turn_id] = current.add(usage_delta)


def _is_empty_usage(usage: Usage) -> bool:
    return usage == Usage()


def _record_tool_call_started(
    event: ToolCallStartedEvent,
    *,
    open_tool_calls: dict[str, ToolCallStartedEvent],
    completed_tool_calls: dict[str, CompletedToolCall],
    tool_call_start_event_ids: dict[str, str],
) -> None:
    tool_call_id = event.payload.tool_call.call_id
    if tool_call_id in completed_tool_calls:
        raise PersistenceError(
            f"tool call {tool_call_id!r} was already completed before a later start event"
        )
    if tool_call_id in open_tool_calls:
        raise PersistenceError(
            f"tool call {tool_call_id!r} has multiple starts without a completion event"
        )
    open_tool_calls[tool_call_id] = event
    tool_call_start_event_ids[event.event_id] = tool_call_id


def _record_tool_call_completed(
    event: ToolCallCompletedEvent,
    *,
    open_tool_calls: dict[str, ToolCallStartedEvent],
    completed_tool_calls: dict[str, CompletedToolCall],
    tool_call_completed_event_ids: dict[str, str],
) -> None:
    result = event.payload.result
    started = open_tool_calls.pop(result.tool_call_id, None)
    if started is None:
        raise PersistenceError(
            f"tool call completion {event.event_id!r} has no matching start "
            f"for {result.tool_call_id!r}"
        )
    started_tool_call = started.payload.tool_call
    if result.tool_name != started_tool_call.tool_name:
        raise PersistenceError(
            f"tool call completion {event.event_id!r} tool name does not match start event"
        )
    completed_tool_calls[result.tool_call_id] = CompletedToolCall(
        tool_call_id=result.tool_call_id,
        tool_name=result.tool_name,
        started_event_id=started.event_id,
        completed_event_id=event.event_id,
        turn_id=event.turn_id or started.turn_id,
        order=result.order,
        result=result,
    )
    tool_call_completed_event_ids[event.event_id] = result.tool_call_id


def _record_compaction_started(
    event: CompactionStartedEvent,
    *,
    open_compactions: dict[str, CompactionStartedEvent],
    completed_compactions: dict[str, CompletedCompaction],
) -> None:
    compaction_id = event.payload.compaction_id
    if compaction_id in completed_compactions:
        raise PersistenceError(
            f"compaction {compaction_id!r} was already completed before a later start event"
        )
    if compaction_id in open_compactions:
        raise PersistenceError(
            f"compaction {compaction_id!r} has multiple starts without a completion event"
        )
    open_compactions[compaction_id] = event


def _record_compaction_completed(
    event: CompactionCompletedEvent,
    *,
    open_compactions: dict[str, CompactionStartedEvent],
    completed_compactions: dict[str, CompletedCompaction],
) -> CompletedCompaction:
    compaction_id = event.payload.compaction_id
    started = open_compactions.pop(compaction_id, None)
    if started is None:
        raise PersistenceError(
            f"compaction completion {event.event_id!r} has no matching start "
            f"for {compaction_id!r}"
        )
    if compaction_id in completed_compactions:
        raise PersistenceError(f"compaction {compaction_id!r} completed more than once")

    metadata = event.payload.metadata
    plan = _json_object_metadata(metadata, "plan")
    config = _json_object_metadata(metadata, "config")
    completed = CompletedCompaction(
        compaction_id=compaction_id,
        started_event_id=started.event_id,
        completed_event_id=event.event_id,
        turn_id=event.turn_id or started.turn_id,
        reason=started.payload.reason,
        summary=event.payload.summary,
        summary_message_id=event.payload.summary_message_id,
        covered_message_ids=list(event.payload.covered_message_ids),
        planned_message_ids=list(started.payload.planned_message_ids),
        preserved_message_ids=_string_list_metadata(metadata, "preserved_message_ids"),
        compact_start_index=_int_metadata(metadata, "compact_start_index"),
        compact_end_index=_int_metadata(metadata, "compact_end_index"),
        split_turn_prefix=_bool_metadata(metadata, "split_turn_prefix"),
        usage=event.payload.usage,
        config=config,
        plan=plan,
        metadata=metadata.copy(),
    )
    completed_compactions[compaction_id] = completed
    return completed


def _json_object_metadata(metadata: JsonObject, key: str) -> JsonObject:
    value = metadata.get(key)
    if not isinstance(value, dict):
        return {}
    return _JSON_OBJECT_ADAPTER.validate_python(value)


def _string_list_metadata(metadata: JsonObject, key: str) -> list[str]:
    value = metadata.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _int_metadata(metadata: JsonObject, key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _bool_metadata(metadata: JsonObject, key: str) -> bool:
    value = metadata.get(key)
    if not isinstance(value, bool):
        return False
    return value


def _incomplete_model_requests(
    *,
    open_model_requests: dict[str, ModelRequestStartedEvent],
    model_failed_event_ids: dict[str, list[str]],
    model_last_failure: dict[str, ModelRequestFailedEvent],
    interruption_event_ids: dict[str, str],
) -> dict[str, IncompleteModelRequest]:
    incomplete: dict[str, IncompleteModelRequest] = {}
    for request_id, event in open_model_requests.items():
        failed_event_ids = list(model_failed_event_ids.get(request_id, []))
        last_failure = model_last_failure.get(request_id)
        incomplete[request_id] = IncompleteModelRequest(
            request_id=request_id,
            started_event_id=event.event_id,
            turn_id=event.turn_id,
            attempt=event.payload.attempt,
            request=event.payload.request,
            failed_event_ids=failed_event_ids,
            last_failed_event_id=last_failure.event_id if last_failure is not None else None,
            last_error=last_failure.payload.error if last_failure is not None else None,
            interrupted_event_id=interruption_event_ids.get(event.event_id),
        )
    return incomplete


def _interrupted_tool_calls(
    *,
    open_tool_calls: dict[str, ToolCallStartedEvent],
    interruption_event_ids: dict[str, str],
) -> dict[str, InterruptedToolCall]:
    interrupted: dict[str, InterruptedToolCall] = {}
    for tool_call_id, event in open_tool_calls.items():
        interrupted_event_id = interruption_event_ids.get(event.event_id)
        result = _interrupted_result(event, interrupted_event_id=interrupted_event_id)
        tool_call = event.payload.tool_call
        interrupted[tool_call_id] = InterruptedToolCall(
            tool_call_id=tool_call_id,
            tool_name=tool_call.tool_name,
            started_event_id=event.event_id,
            interrupted_event_id=interrupted_event_id,
            turn_id=event.turn_id,
            order=tool_call.order,
            tool_call=tool_call,
            result=result,
        )
    return interrupted


def _interrupted_result(
    event: ToolCallStartedEvent,
    *,
    interrupted_event_id: str | None,
) -> ToolResult:
    tool_call = event.payload.tool_call
    message = (
        f"Tool call '{tool_call.tool_name}' was interrupted before completion and was not rerun."
    )
    details: JsonObject = {
        "tool_call_id": tool_call.call_id,
        "started_event_id": event.event_id,
    }
    if interrupted_event_id is not None:
        details["interrupted_event_id"] = interrupted_event_id
    return ToolResult(
        tool_call_id=tool_call.call_id,
        tool_name=tool_call.tool_name,
        arguments=tool_call.arguments,
        success=False,
        output=f"[Tool interrupted before completion: {tool_call.tool_name}]",
        error=ToolError(
            error_type="interrupted",
            message=message,
            details=details,
        ),
        started_at=event.timestamp,
        ended_at=None,
        duration_ms=None,
        timed_out=False,
        truncated=False,
        order=tool_call.order,
        provider_item_id=tool_call.provider_item_id,
        provider_call_id=tool_call.provider_call_id,
        provider_tool_use_id=tool_call.provider_tool_use_id,
        provider_metadata=tool_call.provider_metadata,
    )


__all__ = ("replay_events",)
