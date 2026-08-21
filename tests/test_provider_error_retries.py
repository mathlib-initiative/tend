from __future__ import annotations

import asyncio
from asyncio import CancelledError
from collections.abc import Iterable
from pathlib import Path

import pytest

from tend import Agent
from tend._common.types import StopReason
from tend.agent.config import RuntimeConfig
from tend.agent.persistence.events import (
    EventType,
    ModelRequestFailedEvent,
    ModelRequestStartedEvent,
    ModelResponseCompletedEvent,
    TurnInterruptedEvent,
)
from tend.agent.session import Session
from tend.llm.config import RetryConfig
from tend.llm.models import AssistantMessage, TextContent
from tend.llm.models.requests import ModelResponse
from tend.llm.providers.errors import ProviderRequestError
from tend.llm.retries import RetryErrorCategory
from tend.llm.testing import ScriptedModel
from tend.llm.usage import Usage


def _final_response(text: str = "done") -> ModelResponse:
    return ModelResponse(
        response_id="model_resp_final",
        assistant_message=AssistantMessage(content=[TextContent(text=text)]),
        stop_reason=StopReason.FINAL_RESPONSE,
    )


def _retryable_error() -> ProviderRequestError:
    return ProviderRequestError(
        category=RetryErrorCategory.TIMEOUT,
        message="temporary timeout",
    )


def _runtime_config(*, delay_seconds: float = 0.0) -> RuntimeConfig:
    return RuntimeConfig(
        retries=RetryConfig(
            max_attempts=2,
            initial_delay_seconds=delay_seconds,
            max_delay_seconds=delay_seconds,
            jitter=False,
        )
    )


async def test_provider_error_retry_reuses_request_id_attempts_and_counts_usage(
    tmp_path: Path,
) -> None:
    model = ScriptedModel([_retryable_error(), _final_response("ok")])
    agent = Agent("System prompt.", model=model)

    with Session.create(tmp_path, session_id="sess_provider_retry", sync_writes=False) as session:
        result = await agent.run_turn(
            "Retry transient provider errors",
            session=session,
            config=_runtime_config(),
        )
        events = session.event_store.read_all()
        state = session.state

    assert result.final_response == "ok"
    assert result.model_request_count == 2
    assert result.usage == Usage(model_requests=2, retry_attempts=1)
    assert state.usage == result.usage

    requests = model.requests
    assert len(requests) == 2
    request_id = requests[0].request_id
    assert requests[1].request_id == request_id

    started = _events(events, ModelRequestStartedEvent)
    failed = _events(events, ModelRequestFailedEvent)
    completed = _events(events, ModelResponseCompletedEvent)

    assert [event.payload.request_id for event in started] == [request_id, request_id]
    assert [event.payload.attempt for event in started] == [1, 2]
    assert len(failed) == 1
    assert failed[0].payload.request_id == request_id
    assert failed[0].payload.attempt == 1
    assert failed[0].payload.usage == Usage(model_requests=1, retry_attempts=1)
    assert len(completed) == 1
    assert completed[0].payload.request_id == request_id

    completed_request = state.completed_model_requests[request_id]
    assert completed_request.attempt == 2
    assert completed_request.failed_event_ids == [failed[0].event_id]


async def test_provider_retry_backoff_cancellation_records_turn_interrupted(
    tmp_path: Path,
) -> None:
    model = ScriptedModel([_retryable_error(), _final_response("not consumed")])
    agent = Agent("System prompt.", model=model)

    with Session.create(
        tmp_path,
        session_id="sess_provider_retry_cancel",
        sync_writes=False,
    ) as session:
        task = asyncio.create_task(
            agent.run_turn(
                "Cancel during retry backoff",
                session=session,
                config=_runtime_config(delay_seconds=60.0),
            )
        )
        for _ in range(10):
            await asyncio.sleep(0)
            if _events(session.event_store.read_all(), ModelRequestFailedEvent):
                break
        else:
            pytest.fail("provider retry was not scheduled")

        task.cancel()
        with pytest.raises(CancelledError):
            await task

        events = session.event_store.read_all()
        state = session.state

    assert [event.event_type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.TURN_STARTED,
        EventType.MODEL_REQUEST_STARTED,
        EventType.MODEL_REQUEST_FAILED,
        EventType.TURN_INTERRUPTED,
    ]
    assert model.remaining_steps == 1
    assert state.usage == Usage(model_requests=1, retry_attempts=1)

    failed = _events(events, ModelRequestFailedEvent)
    interrupted = _events(events, TurnInterruptedEvent)
    assert len(failed) == 1
    assert len(interrupted) == 1
    assert interrupted[0].payload.incomplete_event_id == failed[0].event_id
    assert interrupted[0].payload.usage == Usage(model_requests=1, retry_attempts=1)


def _events[T](events: Iterable[object], event_class: type[T]) -> list[T]:
    return [event for event in events if isinstance(event, event_class)]
