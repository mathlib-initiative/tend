from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import Field

import tend.agent.turn_loop as turn_loop_module
from tend import Agent
from tend._common.types import StopReason, StrictModel
from tend.agent.compaction import CompactionPlan, CompactionTriggerReason, plan_compaction
from tend.agent.config import CompactionConfig, RuntimeConfig, UsageConfig
from tend.agent.context import assistant_tool_calls
from tend.agent.persistence.events import (
    CompactionCompletedEvent,
    CompactionStartedEvent,
    EventType,
    ModelRequestFailedEvent,
)
from tend.agent.session import Session
from tend.agent.tools import Tool, ToolContext
from tend.llm.context_estimation import TokenEstimatorConfig
from tend.llm.models import (
    AssistantMessage,
    CompactionSummaryContent,
    ModelResponse,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from tend.llm.testing import ScriptedModel
from tend.llm.usage import TokenUsage, Usage

ESTIMATOR = TokenEstimatorConfig(
    chars_per_token=1000.0,
    tokens_per_message=1,
    tokens_per_content_part=0,
    tokens_per_tool_call=0,
    tokens_per_tool_result=0,
    tokens_per_tool_schema=0,
    tokens_per_reasoning_settings=0,
)


class EchoArguments(StrictModel):
    message: str = Field(min_length=1)


class ContextOverflowError(Exception):
    code = "context_length_exceeded"


def _echo_tool() -> Tool[EchoArguments]:
    async def handler(_context: ToolContext, arguments: EchoArguments) -> str:
        return arguments.message

    return Tool.from_arguments_model(
        name="echo",
        description="Echo a message.",
        arguments_model=EchoArguments,
        handler=handler,
    )


def _tool_call() -> ToolCall:
    return ToolCall(
        call_id="call_echo",
        tool_name="echo",
        arguments={"message": "hello"},
        order=0,
    )


def _tool_response(*, usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(
        response_id="model_resp_tool",
        tool_calls=[_tool_call()],
        usage=usage or Usage(),
    )


def _text_response(text: str, *, response_id: str, usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(
        response_id=response_id,
        assistant_message=AssistantMessage(content=[TextContent(text=text)]),
        stop_reason=StopReason.FINAL_RESPONSE,
        usage=usage or Usage(),
    )


def _compacting_config(*, threshold_messages: int = 2) -> RuntimeConfig:
    return RuntimeConfig(
        compaction=CompactionConfig(
            threshold_messages=threshold_messages,
            reserve_tokens=0,
            keep_recent_tokens=1,
            target_tokens=1,
        ),
        usage=UsageConfig(token_estimator=ESTIMATOR),
    )


async def test_pre_request_compaction_replaces_active_context_with_summary_plus_tail(
    tmp_path: Path,
) -> None:
    compaction_usage = Usage(tokens=TokenUsage(input_tokens=7, output_tokens=3))
    model = ScriptedModel(
        [
            _tool_response(),
            _text_response(
                "## Goal\nContinue after summarizing the original user prompt.",
                response_id="model_resp_summary",
                usage=compaction_usage,
            ),
            _text_response("done", response_id="model_resp_done"),
        ]
    )
    agent = Agent("System prompt.", model=model, tools=[_echo_tool()], model_name="scripted")

    with Session.create(tmp_path, session_id="sess_compact", sync_writes=False) as session:
        result = await agent.run_turn(
            "Use the tool and then continue.",
            session=session,
            config=_compacting_config(),
        )
        events = session.event_store.read_all()
        state = session.state

    assert result.final_response == "done"
    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.model_request_count == 2
    assert result.tool_call_count == 1
    assert result.usage.model_requests == 3
    assert result.usage.tokens == compaction_usage.tokens

    assert len(model.requests) == 3
    compaction_request = model.requests[1]
    assert compaction_request.request_metadata["purpose"] == "generic_compaction"
    final_request = model.requests[2]
    assert [message.role.value for message in final_request.messages] == [
        "system",
        "assistant",
        "assistant",
        "tool",
    ]
    summary_message = final_request.messages[1]
    assert isinstance(summary_message, AssistantMessage)
    summary_part = summary_message.content[0]
    assert isinstance(summary_part, CompactionSummaryContent)
    assert summary_part.summary.startswith("## Goal")
    tail_assistant = final_request.messages[2]
    assert isinstance(tail_assistant, AssistantMessage)
    assert [call.call_id for call in assistant_tool_calls(tail_assistant)] == ["call_echo"]
    tail_tool_result = final_request.messages[3]
    assert isinstance(tail_tool_result, ToolResultMessage)
    assert tail_tool_result.tool_call_id == "call_echo"

    assert [event.event_type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.TURN_STARTED,
        EventType.MODEL_REQUEST_STARTED,
        EventType.MODEL_RESPONSE_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.COMPACTION_STARTED,
        EventType.COMPACTION_COMPLETED,
        EventType.MODEL_REQUEST_STARTED,
        EventType.MODEL_RESPONSE_COMPLETED,
        EventType.TURN_COMPLETED,
    ]
    assert isinstance(events[6], CompactionStartedEvent)
    assert events[6].payload.reason == "threshold_messages"
    assert events[6].payload.planned_message_ids == summary_part.covered_message_ids
    assert isinstance(events[7], CompactionCompletedEvent)
    assert events[7].payload.summary.startswith("## Goal")
    assert events[7].payload.covered_message_ids == summary_part.covered_message_ids
    assert events[7].payload.usage.tokens == compaction_usage.tokens

    assert set(state.completed_compactions) == {events[7].payload.compaction_id}
    completed = next(iter(state.completed_compactions.values()))
    assert completed.covered_message_ids == summary_part.covered_message_ids
    assert completed.preserved_message_ids
    assert state.compaction_usage == {completed.compaction_id: events[7].payload.usage}


async def test_api_anchor_triggers_pre_request_compaction_above_token_threshold(
    tmp_path: Path,
) -> None:
    model = ScriptedModel(
        [
            _tool_response(usage=Usage(tokens=TokenUsage(input_tokens=100, output_tokens=20))),
            _text_response("Anchored summary.", response_id="model_resp_anchor_summary"),
            _text_response("done", response_id="model_resp_anchor_done"),
        ]
    )
    agent = Agent("System prompt.", model=model, tools=[_echo_tool()], model_name="scripted")
    config = RuntimeConfig(
        compaction=CompactionConfig(
            threshold_tokens=50,
            threshold_messages=100,
            reserve_tokens=0,
            keep_recent_tokens=1,
            target_tokens=1,
        ),
        usage=UsageConfig(token_estimator=ESTIMATOR),
    )

    with Session.create(tmp_path, session_id="sess_anchor_compact", sync_writes=False) as session:
        result = await agent.run_turn("Use the tool.", session=session, config=config)
        events = session.event_store.read_all()

    assert result.final_response == "done"
    started_events = [event for event in events if isinstance(event, CompactionStartedEvent)]
    assert len(started_events) == 1
    started = started_events[0]
    assert started.payload.reason == "threshold_tokens"
    plan_metadata = started.payload.metadata["plan"]
    assert isinstance(plan_metadata, dict)
    estimated_tokens = plan_metadata["estimated_tokens"]
    anchor_estimated_tokens = plan_metadata["anchor_estimated_tokens"]
    assert isinstance(estimated_tokens, int)
    assert isinstance(anchor_estimated_tokens, int)
    assert estimated_tokens < 50
    assert anchor_estimated_tokens >= 120


async def test_anchor_only_trigger_with_realistic_budget_proceeds_without_compaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_plans: list[CompactionPlan] = []

    def record_plan(**kwargs: Any) -> CompactionPlan:
        plan = plan_compaction(**kwargs)
        captured_plans.append(plan)
        return plan

    monkeypatch.setattr(turn_loop_module, "plan_compaction", record_plan)
    model = ScriptedModel(
        [
            _tool_response(usage=Usage(tokens=TokenUsage(input_tokens=100, output_tokens=20))),
            _text_response("done", response_id="model_resp_anchor_skip_done"),
        ]
    )
    agent = Agent("System prompt.", model=model, tools=[_echo_tool()], model_name="scripted")
    config = RuntimeConfig(
        compaction=CompactionConfig(
            threshold_tokens=50,
            threshold_messages=100,
            reserve_tokens=0,
            keep_recent_tokens=16_000,
            target_tokens=4_000,
        ),
        usage=UsageConfig(token_estimator=ESTIMATOR),
    )

    with Session.create(tmp_path, session_id="sess_anchor_skip", sync_writes=False) as session:
        result = await agent.run_turn("Use the tool.", session=session, config=config)
        events = session.event_store.read_all()

    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.final_response == "done"
    assert result.model_request_count == 2
    assert len(model.requests) == 2
    assert not any(isinstance(event, CompactionStartedEvent) for event in events)
    anchor_plans = [plan for plan in captured_plans if plan.anchor_estimated_tokens is not None]
    assert len(anchor_plans) == 1
    anchor_plan = anchor_plans[0]
    anchor_estimated_tokens = anchor_plan.anchor_estimated_tokens
    assert anchor_estimated_tokens is not None
    assert anchor_estimated_tokens >= 120
    assert anchor_plan.char_triggered is False
    assert anchor_plan.trigger_reasons == [CompactionTriggerReason.THRESHOLD_TOKENS]
    assert anchor_plan.should_compact is False
    assert anchor_plan.skip_reason == "no safe compaction range"


async def test_context_overflow_error_compacts_once_and_retries_safely(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            _tool_response(usage=Usage(tokens=TokenUsage(input_tokens=100, output_tokens=20))),
            ContextOverflowError("maximum context length exceeded"),
            _text_response("Summary after overflow.", response_id="model_resp_overflow_summary"),
            _text_response("retried", response_id="model_resp_retried"),
        ]
    )
    agent = Agent("System prompt.", model=model, tools=[_echo_tool()], model_name="scripted")
    config = _compacting_config(threshold_messages=100)

    with Session.create(tmp_path, session_id="sess_overflow", sync_writes=False) as session:
        result = await agent.run_turn("Use the tool.", session=session, config=config)
        events = session.event_store.read_all()

    assert result.final_response == "retried"
    assert result.model_request_count == 3
    assert result.tool_call_count == 1
    assert result.usage.model_requests == 4
    assert len(model.requests) == 4

    failed_events = [event for event in events if isinstance(event, ModelRequestFailedEvent)]
    assert len(failed_events) == 1
    assert failed_events[0].payload.retryable is True
    started_events = [event for event in events if isinstance(event, CompactionStartedEvent)]
    assert len(started_events) == 1
    overflow_plan = started_events[0].payload.metadata["plan"]
    assert isinstance(overflow_plan, dict)
    anchor_estimated_tokens = overflow_plan["anchor_estimated_tokens"]
    assert isinstance(anchor_estimated_tokens, int)
    assert anchor_estimated_tokens >= 120
    assert overflow_plan["char_triggered"] is False
    compaction_events = [event for event in events if isinstance(event, CompactionCompletedEvent)]
    assert len(compaction_events) == 1

    retry_request = model.requests[-1]
    summary_message = retry_request.messages[1]
    assert isinstance(summary_message, AssistantMessage)
    assert isinstance(summary_message.content[0], CompactionSummaryContent)
    assert summary_message.content[0].summary == "Summary after overflow."


async def test_char_triggered_no_range_still_stops_turn(tmp_path: Path) -> None:
    model = ScriptedModel([_text_response("unused", response_id="model_resp_unused")])
    agent = Agent("System prompt.", model=model, model_name="scripted")
    config = RuntimeConfig(
        compaction=CompactionConfig(
            threshold_tokens=1,
            threshold_messages=100,
            reserve_tokens=0,
            keep_recent_tokens=16_000,
            target_tokens=4_000,
        ),
        usage=UsageConfig(token_estimator=ESTIMATOR),
    )

    with Session.create(tmp_path, session_id="sess_char_no_range", sync_writes=False) as session:
        result = await agent.run_turn("Tiny prompt.", session=session, config=config)

    assert result.stop_reason is StopReason.COMPACTION_FAILED
    assert result.model_request_count == 0
    assert len(model.requests) == 0


async def test_compaction_failure_stops_turn_with_structured_reason(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            _tool_response(),
            _text_response("  \n\t", response_id="model_resp_empty_summary"),
            _text_response("unused", response_id="model_resp_unused"),
        ]
    )
    agent = Agent("System prompt.", model=model, tools=[_echo_tool()], model_name="scripted")

    with Session.create(
        tmp_path,
        session_id="sess_compaction_failure",
        sync_writes=False,
    ) as session:
        result = await agent.run_turn(
            "Use the tool, then compact badly.",
            session=session,
            config=_compacting_config(),
        )
        events = session.event_store.read_all()

    assert result.final_response is None
    assert result.stop_reason is StopReason.COMPACTION_FAILED
    assert result.stop is not None
    assert result.stop.error is not None
    assert result.stop.error.code == "compaction_failed"
    assert result.model_request_count == 1
    assert result.tool_call_count == 1
    assert model.remaining_steps == 1
    assert any(isinstance(event, CompactionStartedEvent) for event in events)
    assert not any(isinstance(event, CompactionCompletedEvent) for event in events)
    assert events[-1].event_type is EventType.TURN_COMPLETED
    assert events[-1].payload.stop_reason is StopReason.COMPACTION_FAILED
