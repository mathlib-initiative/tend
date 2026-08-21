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
    ReasoningEffort,
    ReasoningSettings,
    get_builtin_profile,
)
from tend.llm.providers import AnthropicMessagesAdapter, JsonPostResponse, ScriptedJsonTransport


class EchoArguments(StrictModel):
    text: str = Field(min_length=1)


def _json(value: object) -> JsonValue:
    return cast(JsonValue, value)


def _cloudflare_sonnet_profile() -> ModelProfile:
    profile = get_builtin_profile(
        ProviderApi.ANTHROPIC_MESSAGES,
        "cloudflare_anthropic",
        "claude-sonnet-4-5",
    )
    assert profile is not None
    return profile


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


async def test_anthropic_messages_adapter_runs_final_response_turn_through_agent(
    tmp_path: Path,
) -> None:
    transport = ScriptedJsonTransport(
        [
            JsonPostResponse(
                status_code=200,
                body=_json(
                    {
                        "id": "msg_final",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-sonnet-4-5-20250929",
                        "content": [{"type": "text", "text": "anthropic final"}],
                        "stop_reason": "end_turn",
                        "usage": {
                            "input_tokens": 7,
                            "output_tokens": 3,
                            "cache_creation_input_tokens": 1,
                            "cache_read_input_tokens": 2,
                        },
                    }
                ),
            )
        ]
    )
    adapter = AnthropicMessagesAdapter(
        model_name="claude-sonnet-4-5",
        provider_name="cloudflare_anthropic",
        profile=_cloudflare_sonnet_profile(),
        transport=transport,
        default_request_settings={"temperature": 0.0},
    )
    agent = Agent("System prompt.", model=adapter, max_output_tokens=64)

    with Session.create(
        tmp_path / "anthropic_final",
        session_id="sess_anthropic_final",
        sync_writes=False,
    ) as session:
        result = await agent.run_turn("Reply once.", session=session, config=_runtime_config())
        events = session.event_store.read_all()

    assert result.final_response == "anthropic final"
    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.stop is None
    assert result.model_request_count == 1
    assert result.tool_call_count == 0
    assert result.usage.model_requests == 1
    assert result.usage.tokens.input_tokens == 7
    assert result.usage.tokens.output_tokens == 3
    assert result.usage.tokens.cache_write_tokens == 1
    assert result.usage.tokens.cache_read_tokens == 2

    request_body = _body(transport, 0)
    assert request_body["system"] == "System prompt."
    assert request_body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Reply once."}]}
    ]
    assert request_body["max_tokens"] == 64
    assert request_body["temperature"] == 0.0
    assert "tool_choice" not in request_body
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
    assert persisted_response.response_id == "msg_final"
    assert persisted_response.provider_metadata is not None
    assert persisted_response.provider_metadata.provider_name == "cloudflare_anthropic"
    assert persisted_response.provider_metadata.response_id == "msg_final"
    assert persisted_response.provider_metadata.item_ids == []


async def test_anthropic_messages_tool_turn_preserves_thinking_and_tool_use_ids(
    tmp_path: Path,
) -> None:
    transport = ScriptedJsonTransport(
        [
            JsonPostResponse(
                status_code=200,
                body=_json(
                    {
                        "id": "msg_tool",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-sonnet-4-5",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "I should call echo.",
                                "signature": "sig_echo",
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_echo",
                                "name": "echo",
                                "input": {"text": "hello"},
                            },
                        ],
                        "stop_reason": "tool_use",
                        "usage": {
                            "input_tokens": 14,
                            "output_tokens": 8,
                            "thinking_tokens": 4,
                        },
                    }
                ),
            ),
            JsonPostResponse(
                status_code=200,
                body=_json(
                    {
                        "id": "msg_done",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-sonnet-4-5",
                        "content": [{"type": "text", "text": "tool done"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 2},
                    }
                ),
            ),
        ]
    )
    adapter = AnthropicMessagesAdapter(
        model_name="claude-sonnet-4-5",
        provider_name="cloudflare_anthropic",
        profile=_cloudflare_sonnet_profile(),
        transport=transport,
        default_request_settings={"tool_choice": {"type": "tool", "name": "echo"}},
    )
    agent = Agent(
        "System prompt.",
        model=adapter,
        tools=[_echo_tool()],
        reasoning=ReasoningSettings(effort=ReasoningEffort.LOW, max_reasoning_tokens=1_024),
        max_output_tokens=2_048,
    )

    with Session.create(
        tmp_path / "anthropic_tool",
        session_id="sess_anthropic_tool",
        sync_writes=False,
    ) as session:
        result = await agent.run_turn("Use echo.", session=session, config=_runtime_config())
        events = session.event_store.read_all()

    assert result.final_response == "tool done"
    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert [call.call_id for call in result.tool_calls] == ["toolu_echo"]
    assert result.tool_calls[0].provider_tool_use_id == "toolu_echo"
    assert result.tool_calls[0].provider_item_id == "toolu_echo"
    assert result.tool_calls[0].arguments == {"text": "hello"}
    assert result.tool_calls[0].provider_metadata["anthropic_raw_input"] == {"text": "hello"}
    assert len(result.tool_results) == 1
    assert result.tool_results[0].success is True
    assert result.tool_results[0].output == {"echo": "hello"}
    assert result.tool_results[0].provider_tool_use_id == "toolu_echo"
    assert result.model_request_count == 2
    assert result.tool_call_count == 1
    assert result.usage.model_requests == 2
    assert result.usage.tool_calls == 1
    assert result.usage.tokens.input_tokens == 24
    assert result.usage.tokens.output_tokens == 10
    assert result.usage.tokens.reasoning_tokens == 4

    first_body = _body(transport, 0)
    assert first_body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert first_body["tools"] == [
        {
            "name": "echo",
            "description": "Echo text for provider integration tests.",
            "input_schema": _echo_tool().definition.arguments_schema,
        }
    ]
    assert "tool_choice" not in first_body
    assert first_body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Use echo."}]}
    ]

    follow_up_body = _body(transport, 1)
    assert follow_up_body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert "tool_choice" not in follow_up_body
    assert follow_up_body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Use echo."}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "I should call echo.",
                    "signature": "sig_echo",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_echo",
                    "name": "echo",
                    "input": {"text": "hello"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_echo",
                    "content": '{"echo":"hello"}',
                }
            ],
        },
    ]
    assert transport.remaining_steps == 0

    response_events = [event for event in events if isinstance(event, ModelResponseCompletedEvent)]
    tool_completed_events = [event for event in events if isinstance(event, ToolCallCompletedEvent)]
    assert len(response_events) == 2
    first_response = response_events[0].payload.response
    assert first_response is not None
    assert first_response.final_text is None
    assert first_response.provider_metadata is not None
    assert first_response.provider_metadata.item_ids == ["toolu_echo"]
    assert first_response.reasoning is not None
    continuation = first_response.reasoning.provider_private_continuation[0]
    assert continuation.kind == "thinking"
    assert continuation.signature == "sig_echo"
    assert continuation.redacted_details["anthropic_block"] == {
        "type": "thinking",
        "thinking": "I should call echo.",
        "signature": "sig_echo",
    }
    assert len(tool_completed_events) == 1
    assert tool_completed_events[0].payload.result.provider_tool_use_id == "toolu_echo"
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


async def test_anthropic_messages_max_tokens_turn_stops_with_limit() -> None:
    transport = ScriptedJsonTransport(
        [
            JsonPostResponse(
                status_code=200,
                body=_json(
                    {
                        "id": "msg_partial",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-sonnet-4-5",
                        "content": [{"type": "text", "text": "partial"}],
                        "stop_reason": "max_tokens",
                        "usage": {"input_tokens": 5, "output_tokens": 1},
                    }
                ),
            )
        ]
    )
    adapter = AnthropicMessagesAdapter(
        model_name="claude-sonnet-4-5",
        provider_name="cloudflare_anthropic",
        profile=_cloudflare_sonnet_profile(),
        transport=transport,
    )
    agent = Agent("System prompt.", model=adapter, max_output_tokens=1)

    result = await agent.run_turn("Use too few output tokens.", config=_runtime_config())

    assert result.final_response is None
    assert result.stop_reason is StopReason.MAX_TOKENS
    assert result.stop is not None
    assert result.stop.reason is StopReason.MAX_TOKENS
    assert result.stop.details["response_id"] == "msg_partial"
    assert result.stop.details["provider_completion_status"] == "incomplete"
    assert result.stop.details["incomplete_details"] == {"stop_reason": "max_tokens"}
    assert result.usage.model_requests == 1
    assert transport.remaining_steps == 0

    request_body = _body(transport, 0)
    assert request_body["max_tokens"] == 1
