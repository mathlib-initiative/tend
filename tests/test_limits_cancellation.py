from __future__ import annotations

from asyncio import CancelledError
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import Field

from tend import Agent, CancellationState
from tend._common.types import StopReason, StrictModel
from tend.agent.config import RuntimeConfig, RuntimeLimitsConfig
from tend.agent.persistence.events import EventType, TurnInterruptedEvent
from tend.agent.session import Session
from tend.agent.tools import Tool, ToolContext
from tend.llm.models import AssistantMessage, ModelRequest, ModelResponse, TextContent, ToolCall
from tend.llm.models.profiles import ModelProfile
from tend.llm.testing import ScriptedModel
from tend.llm.usage import Cost, TokenUsage, Usage


class EchoArguments(StrictModel):
    message: str = Field(min_length=1)


class StepClock:
    def __init__(self, values: Iterable[float]) -> None:
        self._values = list(values)
        self._last = self._values[-1]

    def __call__(self) -> float:
        if self._values:
            self._last = self._values.pop(0)
        return self._last


def _tool_call(*, call_id: str = "call_echo") -> ToolCall:
    return ToolCall(call_id=call_id, tool_name="echo", arguments={"message": "hello"})


def _tool_response(
    *,
    tool_call: ToolCall | None = None,
    usage: Usage | None = None,
) -> ModelResponse:
    return ModelResponse(
        response_id="model_resp_tool",
        tool_calls=[tool_call or _tool_call()],
        usage=usage or Usage(),
    )


def _final_response(text: str = "done") -> ModelResponse:
    return ModelResponse(
        response_id="model_resp_final",
        assistant_message=AssistantMessage(content=[TextContent(text=text)]),
        stop_reason=StopReason.FINAL_RESPONSE,
    )


def _echo_tool(seen: list[str] | None = None) -> Tool[EchoArguments]:
    async def handler(_context: ToolContext, arguments: EchoArguments) -> str:
        if seen is not None:
            seen.append(arguments.message)
        return arguments.message

    return Tool.from_arguments_model(
        name="echo",
        description="Echo a message.",
        arguments_model=EchoArguments,
        handler=handler,
    )


async def test_max_model_requests_stops_before_another_model_request() -> None:
    seen: list[str] = []
    model = ScriptedModel([_tool_response(), _final_response("not consumed")])
    agent = Agent("System prompt.", model=model, tools=[_echo_tool(seen)])
    config = RuntimeConfig(limits=RuntimeLimitsConfig(max_model_requests=1))

    result = await agent.run_turn("Use one tool", config=config)

    assert result.final_response is None
    assert result.stop_reason is StopReason.MAX_MODEL_REQUESTS
    assert result.stop is not None
    assert result.stop.details["model_request_count"] == 1
    assert result.model_request_count == 1
    assert result.tool_call_count == 1
    assert seen == ["hello"]
    assert model.remaining_steps == 1


async def test_max_tool_calls_stops_before_tool_execution() -> None:
    seen: list[str] = []
    model = ScriptedModel([_tool_response(), _final_response("not consumed")])
    agent = Agent("System prompt.", model=model, tools=[_echo_tool(seen)])
    config = RuntimeConfig(limits=RuntimeLimitsConfig(max_tool_calls=0))

    result = await agent.run_turn("Do not run tools", config=config)

    assert result.final_response is None
    assert result.stop_reason is StopReason.MAX_TOOL_CALLS
    assert result.stop is not None
    assert result.stop.details["requested_tool_call_count"] == 1
    assert [call.call_id for call in result.tool_calls] == ["call_echo"]
    assert result.tool_results == []
    assert result.tool_call_count == 0
    assert seen == []


async def test_wall_clock_limit_uses_injected_clock_before_tool_work() -> None:
    seen: list[str] = []
    model = ScriptedModel([_tool_response(), _final_response("not consumed")])
    agent = Agent("System prompt.", model=model, tools=[_echo_tool(seen)])
    config = RuntimeConfig(limits=RuntimeLimitsConfig(max_wall_time_seconds=1.0))
    clock = StepClock([10.0, 10.0, 12.5])

    result = await agent.run_turn("Clock stops tools", config=config, clock=clock)

    assert result.final_response is None
    assert result.stop_reason is StopReason.MAX_WALL_TIME
    assert result.stop is not None
    assert result.stop.details["boundary"] == "before_tool_execution"
    assert result.stop.details["elapsed_seconds"] == 2.5
    assert result.tool_results == []
    assert seen == []


async def test_token_limit_stops_at_next_work_boundary() -> None:
    usage = Usage(tokens=TokenUsage(input_tokens=4, output_tokens=1))
    model = ScriptedModel([_tool_response(usage=usage), _final_response("not consumed")])
    agent = Agent("System prompt.", model=model, tools=[_echo_tool()])
    config = RuntimeConfig(limits=RuntimeLimitsConfig(max_tokens=5))

    result = await agent.run_turn("Token limit", config=config)

    assert result.stop_reason is StopReason.MAX_TOKENS
    assert result.stop is not None
    assert result.stop.details["token_count"] == 5
    assert result.tool_results == []


async def test_cost_limit_stops_at_next_work_boundary() -> None:
    usage = Usage(cost=Cost(amount=Decimal("0.25"), currency="USD"))
    model = ScriptedModel([_tool_response(usage=usage), _final_response("not consumed")])
    agent = Agent("System prompt.", model=model, tools=[_echo_tool()])
    config = RuntimeConfig(limits=RuntimeLimitsConfig(max_cost=Decimal("0.25")))

    result = await agent.run_turn("Cost limit", config=config)

    assert result.stop_reason is StopReason.MAX_COST
    assert result.stop is not None
    assert result.stop.details["cost_amount"] == "0.25"
    assert result.tool_results == []


async def test_pre_requested_cancellation_records_interruption(tmp_path: Path) -> None:
    cancellation = CancellationState(is_cancelled=True, reason="user requested stop")
    model = ScriptedModel([_final_response("not consumed")])
    agent = Agent("System prompt.", model=model)

    with Session.create(tmp_path, session_id="sess_cancel", sync_writes=False) as session:
        result = await agent.run_turn(
            "Stop before work",
            session=session,
            cancellation=cancellation,
        )
        events = session.event_store.read_all()

    assert result.final_response is None
    assert result.stop_reason is StopReason.INTERRUPTED
    assert result.stop is not None
    assert "user requested stop" in (result.stop.message or "")
    assert model.requests == ()
    assert [event.event_type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.TURN_STARTED,
        EventType.TURN_INTERRUPTED,
        EventType.TURN_COMPLETED,
    ]
    assert isinstance(events[2], TurnInterruptedEvent)
    assert events[2].payload.incomplete_event_id is None


async def test_tool_can_observe_and_request_cancellation(tmp_path: Path) -> None:
    cancellation = CancellationState()
    observed: list[bool] = []

    async def handler(context: ToolContext, arguments: EchoArguments) -> str:
        observed.append(context.is_cancelled)
        context_cancellation = context.cancellation
        assert context_cancellation is not None
        assert context_cancellation is cancellation
        context_cancellation.cancel("tool requested stop")
        return arguments.message

    tool = Tool.from_arguments_model(
        name="echo",
        description="Echo a message.",
        arguments_model=EchoArguments,
        handler=handler,
    )
    model = ScriptedModel([_tool_response(), _final_response("not consumed")])
    agent = Agent("System prompt.", model=model, tools=[tool])

    with Session.create(tmp_path, session_id="sess_tool_cancel", sync_writes=False) as session:
        result = await agent.run_turn("Tool cancels", session=session, cancellation=cancellation)
        events = session.event_store.read_all()

    assert observed == [False]
    assert result.stop_reason is StopReason.INTERRUPTED
    assert result.stop is not None
    assert "tool requested stop" in (result.stop.message or "")
    assert len(result.tool_results) == 1
    assert result.tool_results[0].success is True
    assert model.remaining_steps == 1
    assert EventType.TURN_INTERRUPTED in [event.event_type for event in events]
    assert events[-1].event_type is EventType.TURN_COMPLETED


class CancellingModel:
    @property
    def profile(self) -> ModelProfile | None:
        return None

    async def generate(self, request: ModelRequest) -> ModelResponse:
        _ = request
        raise CancelledError


async def test_task_cancellation_records_interruption_and_reraises(tmp_path: Path) -> None:
    agent = Agent("System prompt.", model=CancellingModel())

    with Session.create(tmp_path, session_id="sess_task_cancel", sync_writes=False) as session:
        with pytest.raises(CancelledError):
            await agent.run_turn("Cancel during model", session=session)
        events = session.event_store.read_all()
        state = session.state

    assert [event.event_type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.TURN_STARTED,
        EventType.MODEL_REQUEST_STARTED,
        EventType.TURN_INTERRUPTED,
    ]
    assert isinstance(events[-1], TurnInterruptedEvent)
    assert events[-1].payload.incomplete_event_id == events[2].event_id
    assert len(state.incomplete_model_requests) == 1
    incomplete = next(iter(state.incomplete_model_requests.values()))
    assert incomplete.interrupted_event_id == events[-1].event_id
