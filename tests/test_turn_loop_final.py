from __future__ import annotations

from pathlib import Path

from tend import Agent, TurnResult
from tend._common.types import StopReason
from tend.agent.persistence.events import (
    EventType,
    ModelRequestStartedEvent,
    ModelResponseCompletedEvent,
    SessionStartedEvent,
    TurnCompletedEvent,
    TurnStartedEvent,
)
from tend.agent.results import StopResult
from tend.agent.session import Session
from tend.llm.models import (
    AssistantMessage,
    ModelResponse,
    ProviderCompletionStatus,
    ProviderMetadata,
    TextContent,
)
from tend.llm.testing import ScriptedModel
from tend.llm.usage import TokenUsage, Usage


async def test_agent_run_turn_returns_final_text_without_session() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                response_id="model_resp_1",
                assistant_message=AssistantMessage(content=[TextContent(text="ok")]),
                stop_reason=StopReason.FINAL_RESPONSE,
                usage=Usage(tokens=TokenUsage(input_tokens=3, output_tokens=1)),
            )
        ]
    )
    agent = Agent("You are concise.", model=model, tools=["ls"], model_name="scripted")

    result = await agent.run_turn("Say ok")

    assert isinstance(result, TurnResult)
    assert result.final_response == "ok"
    assert result.final_text == "ok"
    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.stop is None
    assert result.session_id is None
    assert result.session_state is None
    assert result.tool_calls == []
    assert result.tool_results == []
    assert result.model_request_count == 1
    assert result.usage.tokens.input_tokens == 3
    assert result.usage.tokens.output_tokens == 1
    assert result.usage.model_requests == 1

    request = model.last_request
    assert request is not None
    assert request.model_name == "scripted"
    assert [message.role.value for message in request.messages] == ["system", "user"]
    assert request.messages[-1].content[0] == TextContent(text="Say ok")
    assert len(request.tools) == 1
    assert request.tools[0]["name"] == "ls"


async def test_agent_run_turn_writes_session_events_in_order(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                response_id="model_resp_1",
                assistant_message=AssistantMessage(
                    message_id="msg_assistant_1",
                    content=[TextContent(text="done")],
                ),
                stop_reason=StopReason.FINAL_RESPONSE,
            )
        ]
    )
    agent = Agent("System prompt.", model=model, model_name="scripted")

    with Session.create(tmp_path, session_id="sess_1", sync_writes=False) as session:
        result = await agent.run_turn("Finish this", session=session)

        assert result.final_response == "done"
        assert result.session_id == "sess_1"
        assert result.session_state == session.state
        assert result.session_state is not None
        assert len(result.session_state.completed_model_requests) == 1

        events = session.event_store.read_all()

    assert [event.event_type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.TURN_STARTED,
        EventType.MODEL_REQUEST_STARTED,
        EventType.MODEL_RESPONSE_COMPLETED,
        EventType.TURN_COMPLETED,
    ]
    assert [event.sequence for event in events] == [0, 1, 2, 3, 4]
    assert [event.parent_event_id for event in events] == [
        None,
        events[0].event_id,
        events[1].event_id,
        events[2].event_id,
        events[3].event_id,
    ]

    assert isinstance(events[0], SessionStartedEvent)
    assert isinstance(events[1], TurnStartedEvent)
    assert isinstance(events[2], ModelRequestStartedEvent)
    assert isinstance(events[3], ModelResponseCompletedEvent)
    assert isinstance(events[4], TurnCompletedEvent)

    turn_id = events[1].turn_id
    assert turn_id is not None
    assert all(event.turn_id == turn_id for event in events[1:])
    assert events[1].payload.prompt == "Finish this"
    assert events[1].payload.input_message_id is not None
    assert events[2].payload.request is not None
    assert events[2].payload.request_id == events[2].payload.request.request_id
    assert events[2].payload.request.messages[-1].content[0] == TextContent(text="Finish this")
    assert events[3].payload.request_id == events[2].payload.request_id
    assert events[3].payload.response is not None
    assert events[3].payload.response.final_text == "done"
    assert events[4].payload.stop_reason is StopReason.FINAL_RESPONSE
    assert events[4].payload.final_response == "done"
    assert events[4].payload.model_request_count == 1
    assert events[4].payload.tool_call_count == 0


async def test_agent_run_turn_works_without_session(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                response_id="model_resp_1",
                assistant_message=AssistantMessage(content=[TextContent(text="memory only")]),
            )
        ]
    )
    agent = Agent("System prompt.", model=model)

    result = await agent.run_turn("No persistence")

    assert result.final_response == "memory only"
    assert not (tmp_path / "events.jsonl").exists()


async def test_completed_model_response_without_text_returns_empty_final_response() -> None:
    model = ScriptedModel([ModelResponse(response_id="model_resp_1")])
    agent = Agent("System prompt.", model=model)

    result = await agent.run_turn("No final text")

    assert result.final_response == ""
    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.stop is None


async def test_completed_native_response_without_text_returns_empty_final_response() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                response_id="model_resp_1",
                provider_metadata=ProviderMetadata(
                    provider_name="openai",
                    native_stop_reason="completed",
                ),
            )
        ]
    )
    agent = Agent("System prompt.", model=model)

    result = await agent.run_turn("No final text")

    assert result.final_response == ""
    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.stop is None


async def test_non_final_native_stop_reason_without_tools_is_not_empty_final_response() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                response_id="model_resp_1",
                provider_metadata=ProviderMetadata(
                    provider_name="anthropic",
                    native_stop_reason="tool_use",
                ),
            )
        ]
    )
    agent = Agent("System prompt.", model=model)

    result = await agent.run_turn("No final text")

    assert result.final_response is None
    assert result.stop_reason is StopReason.PROVIDER_STOP_REASON
    assert result.stop is not None
    assert result.stop.details["native_stop_reason"] == "tool_use"


async def test_non_final_model_response_returns_structured_stop() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                response_id="model_resp_1",
                stop_reason=StopReason.MAX_TOKENS,
                provider_completion_status=ProviderCompletionStatus.INCOMPLETE,
            )
        ]
    )
    agent = Agent("System prompt.", model=model)

    result = await agent.run_turn("No final text")

    assert result.final_response is None
    assert result.stop_reason is StopReason.MAX_TOKENS
    assert isinstance(result.stop, StopResult)
    assert result.stop.reason is StopReason.MAX_TOKENS
    assert result.stop.message == "Model response did not contain final assistant text."


async def test_turn_result_json_serialization_round_trip() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                response_id="model_resp_1",
                assistant_message=AssistantMessage(content=[TextContent(text="json ok")]),
            )
        ]
    )
    agent = Agent("System prompt.", model=model)

    result = await agent.run_turn("Serialize")
    restored = TurnResult.model_validate_json(result.model_dump_json())

    assert restored == result
