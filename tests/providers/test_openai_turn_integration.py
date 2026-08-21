from __future__ import annotations

from pathlib import Path
from typing import cast

from pydantic import Field, JsonValue

from tend import Agent
from tend._common.types import JsonObject, StopReason, StrictModel
from tend.agent.config import CompactionConfig, RuntimeConfig
from tend.agent.persistence.events import (
    EventType,
    ModelRequestStartedEvent,
    ModelResponseCompletedEvent,
    ToolCallCompletedEvent,
)
from tend.agent.session import Session
from tend.agent.tools import Tool, ToolContext
from tend.llm.models import (
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ProviderApi,
    ProviderCompletionStatus,
    ReasoningEffort,
    ReasoningSettings,
    get_builtin_profile,
)
from tend.llm.providers import JsonPostResponse, OpenAIResponsesAdapter, ScriptedJsonTransport


class EchoArguments(StrictModel):
    text: str = Field(min_length=1)


def _json(value: object) -> JsonValue:
    return cast(JsonValue, value)


def _cloudflare_gpt5_profile() -> ModelProfile:
    profile = get_builtin_profile(ProviderApi.OPENAI_RESPONSES, "cloudflare_openai", "gpt-5")
    assert profile is not None
    return profile


def _empty_json_list() -> list[object]:
    return []


def _body(transport: ScriptedJsonTransport, index: int) -> JsonObject:
    return cast(JsonObject, transport.requests[index].body)


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(compaction=CompactionConfig(enabled=False))


def _echo_tool() -> Tool[EchoArguments]:
    async def handler(_context: ToolContext, arguments: EchoArguments) -> dict[str, str]:
        return {"echo": arguments.text}

    return Tool.from_arguments_model(
        name="echo",
        description="Echo text for provider integration tests.",
        arguments_model=EchoArguments,
        handler=handler,
    )


async def test_openai_responses_adapter_runs_final_response_turn_through_agent(
    tmp_path: Path,
) -> None:
    transport = ScriptedJsonTransport(
        [
            JsonPostResponse(
                status_code=200,
                body=_json(
                    {
                        "id": "resp_final",
                        "status": "completed",
                        "model": "gpt-5-2025-08-07",
                        "output": [
                            {
                                "id": "msg_final",
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "openai final"}],
                            }
                        ],
                        "usage": {
                            "input_tokens": 8,
                            "output_tokens": 3,
                            "total_tokens": 11,
                            "output_tokens_details": {"reasoning_tokens": 1},
                        },
                    }
                ),
            )
        ]
    )
    adapter = OpenAIResponsesAdapter(
        model_name="gpt-5",
        provider_name="cloudflare_openai",
        profile=_cloudflare_gpt5_profile(),
        transport=transport,
    )
    agent = Agent(
        "System prompt.",
        model=adapter,
        reasoning=ReasoningSettings(effort=ReasoningEffort.LOW),
        max_output_tokens=64,
    )

    with Session.create(
        tmp_path / "openai_final",
        session_id="sess_openai_final",
        sync_writes=False,
    ) as session:
        result = await agent.run_turn("Reply once.", session=session, config=_runtime_config())
        events = session.event_store.read_all()

    assert result.final_response == "openai final"
    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.stop is None
    assert result.model_request_count == 1
    assert result.tool_call_count == 0
    assert result.usage.model_requests == 1
    assert result.usage.tokens.input_tokens == 8
    assert result.usage.tokens.output_tokens == 3
    assert result.usage.tokens.reasoning_tokens == 1
    assert result.usage.tokens.provider_details == {"total_tokens": 11}

    request_body = _body(transport, 0)
    assert request_body["input"] == [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Reply once."},
    ]
    assert request_body["reasoning"] == {"effort": "low", "summary": "auto"}
    assert request_body["include"] == ["reasoning.encrypted_content"]
    assert request_body["max_output_tokens"] == 64
    assert "temperature" not in request_body
    assert transport.remaining_steps == 0

    request_events = [event for event in events if isinstance(event, ModelRequestStartedEvent)]
    response_events = [event for event in events if isinstance(event, ModelResponseCompletedEvent)]
    assert len(request_events) == 1
    assert len(response_events) == 1
    persisted_request = request_events[0].payload.request
    persisted_response = response_events[0].payload.response
    assert isinstance(persisted_request, ModelRequest)
    assert isinstance(persisted_response, ModelResponse)
    assert persisted_request.request_metadata["runtime_cwd"] == "."
    assert persisted_request.request_metadata["turn_iteration"] == 0
    assert persisted_response.response_id == "resp_final"
    assert persisted_response.provider_metadata is not None
    assert persisted_response.provider_metadata.provider_name == "cloudflare_openai"
    assert persisted_response.provider_metadata.response_id == "resp_final"
    assert persisted_response.provider_metadata.item_ids == ["msg_final"]


async def test_openai_responses_tool_turn_uses_stateless_function_call_output(
    tmp_path: Path,
) -> None:
    transport = ScriptedJsonTransport(
        [
            JsonPostResponse(
                status_code=200,
                body=_json(
                    {
                        "id": "resp_tool",
                        "status": "completed",
                        "model": "gpt-5",
                        "output": [
                            {
                                "id": "rs_tool",
                                "type": "reasoning",
                                "status": "completed",
                                "summary": [{"type": "summary_text", "text": "Need echo."}],
                                "encrypted_content": "encrypted-openai-reasoning",
                            },
                            {
                                "id": "fc_echo",
                                "type": "function_call",
                                "status": "completed",
                                "call_id": "call_openai_echo",
                                "name": "echo",
                                "arguments": '{"text":"hello"}',
                            },
                        ],
                        "usage": {
                            "input_tokens": 13,
                            "output_tokens": 7,
                            "output_tokens_details": {"reasoning_tokens": 2},
                        },
                    }
                ),
            ),
            JsonPostResponse(
                status_code=200,
                body=_json(
                    {
                        "id": "resp_done",
                        "status": "completed",
                        "model": "gpt-5",
                        "output": [
                            {
                                "id": "msg_done",
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "tool done"}],
                            }
                        ],
                        "usage": {"input_tokens": 9, "output_tokens": 2},
                    }
                ),
            ),
        ]
    )
    adapter = OpenAIResponsesAdapter(
        model_name="gpt-5",
        provider_name="cloudflare_openai",
        profile=_cloudflare_gpt5_profile(),
        transport=transport,
    )
    agent = Agent(
        "System prompt.",
        model=adapter,
        tools=[_echo_tool()],
        max_output_tokens=64,
    )

    with Session.create(
        tmp_path / "openai_tool",
        session_id="sess_openai_tool",
        sync_writes=False,
    ) as session:
        result = await agent.run_turn("Use echo.", session=session, config=_runtime_config())
        events = session.event_store.read_all()

    assert result.final_response == "tool done"
    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert [call.call_id for call in result.tool_calls] == ["call_openai_echo"]
    assert result.tool_calls[0].provider_item_id == "fc_echo"
    assert result.tool_calls[0].provider_call_id == "call_openai_echo"
    assert result.tool_calls[0].arguments == {"text": "hello"}
    assert result.tool_calls[0].provider_metadata["openai_raw_arguments"] == '{"text":"hello"}'
    assert len(result.tool_results) == 1
    assert result.tool_results[0].success is True
    assert result.tool_results[0].output == {"echo": "hello"}
    assert result.tool_results[0].provider_call_id == "call_openai_echo"
    assert result.model_request_count == 2
    assert result.tool_call_count == 1
    assert result.usage.model_requests == 2
    assert result.usage.tool_calls == 1
    assert result.usage.tokens.input_tokens == 22
    assert result.usage.tokens.output_tokens == 9
    assert result.usage.tokens.reasoning_tokens == 2

    first_body = _body(transport, 0)
    assert first_body["tools"] == [
        {
            "type": "function",
            "name": "echo",
            "description": "Echo text for provider integration tests.",
            "parameters": _echo_tool().definition.arguments_schema,
            "strict": True,
        }
    ]
    assert first_body["parallel_tool_calls"] is False
    assert "previous_response_id" not in first_body
    assert "temperature" not in first_body

    follow_up_body = _body(transport, 1)
    assert follow_up_body["input"] == [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Use echo."},
        {
            "type": "function_call",
            "call_id": "call_openai_echo",
            "name": "echo",
            "arguments": '{"text":"hello"}',
            "status": "completed",
            "id": "fc_echo",
        },
        {
            "type": "function_call_output",
            "call_id": "call_openai_echo",
            "output": '{"echo":"hello"}',
        },
    ]
    assert "previous_response_id" not in follow_up_body
    assert transport.remaining_steps == 0

    response_events = [event for event in events if isinstance(event, ModelResponseCompletedEvent)]
    tool_completed_events = [event for event in events if isinstance(event, ToolCallCompletedEvent)]
    assert len(response_events) == 2
    first_response = response_events[0].payload.response
    assert first_response is not None
    assert first_response.provider_completion_status is ProviderCompletionStatus.COMPLETED
    assert first_response.provider_metadata is not None
    assert first_response.provider_metadata.item_ids == ["rs_tool", "fc_echo"]
    assert first_response.reasoning is not None
    assert first_response.reasoning.summaries[0].text == "Need echo."
    assert first_response.reasoning.provider_private_continuation[0].encrypted_content == (
        "encrypted-openai-reasoning"
    )
    assert len(tool_completed_events) == 1
    assert tool_completed_events[0].payload.result.provider_call_id == "call_openai_echo"
    assert tool_completed_events[0].payload.result.success is True
    assert [event.event_type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.TURN_STARTED,
        EventType.MODEL_REQUEST_STARTED,
        EventType.MODEL_RESPONSE_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_REQUEST_STARTED,
        EventType.MODEL_RESPONSE_COMPLETED,
        EventType.TURN_COMPLETED,
    ]


async def test_openai_responses_incomplete_max_output_turn_stops_with_limit() -> None:
    transport = ScriptedJsonTransport(
        [
            JsonPostResponse(
                status_code=200,
                body=_json(
                    {
                        "id": "resp_incomplete",
                        "status": "incomplete",
                        "model": "gpt-5",
                        "incomplete_details": {"reason": "max_output_tokens"},
                        "output": [
                            {"id": "rs_limit", "type": "reasoning", "summary": _empty_json_list()},
                            {
                                "id": "msg_partial",
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "partial"}],
                            },
                        ],
                        "usage": {
                            "input_tokens": 5,
                            "output_tokens": 1,
                            "output_tokens_details": {"reasoning_tokens": 1},
                        },
                    }
                ),
            )
        ]
    )
    adapter = OpenAIResponsesAdapter(
        model_name="gpt-5",
        provider_name="cloudflare_openai",
        profile=_cloudflare_gpt5_profile(),
        transport=transport,
    )
    agent = Agent("System prompt.", model=adapter, max_output_tokens=1)

    result = await agent.run_turn("Use too few output tokens.", config=_runtime_config())

    assert result.final_response is None
    assert result.stop_reason is StopReason.MAX_TOKENS
    assert result.stop is not None
    assert result.stop.reason is StopReason.MAX_TOKENS
    assert result.stop.details["response_id"] == "resp_incomplete"
    assert result.stop.details["provider_completion_status"] == "incomplete"
    assert result.stop.details["incomplete_details"] == {"reason": "max_output_tokens"}
    assert result.usage.model_requests == 1
    assert result.usage.tokens.reasoning_tokens == 1
    assert transport.remaining_steps == 0

    request_body = _body(transport, 0)
    assert request_body["reasoning"] == {"effort": "minimal", "summary": "auto"}
    assert request_body["include"] == ["reasoning.encrypted_content"]
    assert request_body["max_output_tokens"] == 1
