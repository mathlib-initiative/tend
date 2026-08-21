from __future__ import annotations

from typing import cast

import pytest
from pydantic import JsonValue

from tend._common.errors import ProviderProtocolError
from tend._common.types import JsonObject, StopReason
from tend.agent.context import assistant_message_from_response
from tend.llm.models import (
    ModelRequest,
    ProviderCompletionStatus,
    ProviderItemKind,
    ReasoningEffort,
    ReasoningSettings,
    TextContent,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from tend.llm.providers import AnthropicMessagesAdapter, JsonPostResponse, ScriptedJsonTransport


def _json(value: object) -> JsonValue:
    return cast(JsonValue, value)


def _empty_json_list() -> list[object]:
    return []


def test_text_response_extracts_text_usage_stop_and_metadata() -> None:
    adapter = AnthropicMessagesAdapter(
        model_name="claude-sonnet-4-5",
        provider_name="cloudflare_anthropic",
    )
    request = ModelRequest(
        request_id="model_req_1",
        messages=[UserMessage(content=[TextContent(text="Say hi")])],
    )

    response = adapter.parse_response(
        _json(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "cache_creation_input_tokens": 1,
                    "cache_read_input_tokens": 2,
                    "service_tier": "standard",
                    "cache_creation": {"ephemeral_5m_input_tokens": 4},
                    "server_tool_use": {"web_search_requests": 1},
                },
            }
        ),
        request=request,
    )

    assert response.request_id == "model_req_1"
    assert response.response_id == "msg_1"
    assert response.final_text == "hello"
    assert response.stop_reason is StopReason.FINAL_RESPONSE
    assert response.provider_completion_status is ProviderCompletionStatus.COMPLETED
    assert response.usage.tokens.input_tokens == 7
    assert response.usage.tokens.output_tokens == 3
    assert response.usage.tokens.cache_write_tokens == 1
    assert response.usage.tokens.cache_read_tokens == 2
    assert response.usage.tokens.provider_details == {
        "cache_creation.ephemeral_5m_input_tokens": 4,
        "server_tool_use.web_search_requests": 1,
    }
    assert response.response_metadata["service_tier"] == "standard"
    assert response.provider_metadata is not None
    assert response.provider_metadata.response_id == "msg_1"
    assert response.provider_metadata.model_name == "claude-sonnet-4-5-20250929"
    assert response.provider_metadata.native_stop_reason == "end_turn"
    assert response.provider_metadata.items[0].kind is ProviderItemKind.OUTPUT_TEXT


def test_multiple_tool_use_blocks_preserve_ids_arguments_and_order() -> None:
    adapter = AnthropicMessagesAdapter(model_name="claude-sonnet-4-5")

    response = adapter.parse_response(
        _json(
            {
                "id": "msg_tools",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "ls",
                        "input": {"path": "."},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "read_file",
                        "input": {"path": "README.md", "limit": 5},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )
    )

    assert response.final_text is None
    assert response.stop_reason is StopReason.PROVIDER_STOP_REASON
    assert [call.call_id for call in response.tool_calls] == ["toolu_1", "toolu_2"]
    assert [call.order for call in response.tool_calls] == [0, 1]
    assert response.tool_calls[0].provider_tool_use_id == "toolu_1"
    assert response.tool_calls[0].provider_item_id == "toolu_1"
    assert response.tool_calls[0].arguments == {"path": "."}
    assert response.tool_calls[0].provider_metadata["anthropic_raw_input"] == {"path": "."}
    assert response.tool_calls[1].arguments == {"path": "README.md", "limit": 5}
    assert response.provider_metadata is not None
    assert response.provider_metadata.item_ids == ["toolu_1", "toolu_2"]
    assert [item.kind for item in response.provider_metadata.items] == [
        ProviderItemKind.TOOL_USE,
        ProviderItemKind.TOOL_USE,
    ]
    assert response.provider_metadata.items[0].provider_tool_use_id == "toolu_1"


def test_thinking_signature_is_preserved_as_reasoning_not_final_text() -> None:
    adapter = AnthropicMessagesAdapter(model_name="claude-sonnet-4-5")
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="Think briefly")])],
        reasoning=ReasoningSettings(effort=ReasoningEffort.LOW, max_reasoning_tokens=1024),
        max_output_tokens=2048,
    )

    response = adapter.parse_response(
        _json(
            {
                "id": "msg_thinking",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "private reasoning text",
                        "signature": "sig_123",
                    },
                    {"type": "text", "text": "done"},
                ],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 17,
                    "thinking_tokens": 9,
                },
            }
        ),
        request=request,
    )

    assert response.final_text == "done"
    assert "private reasoning" not in response.final_text
    assert response.reasoning is not None
    assert response.reasoning.requested is not None
    assert response.reasoning.requested.effort is ReasoningEffort.LOW
    assert response.reasoning.reasoning_tokens == 9
    assert response.reasoning.native_settings == {
        "anthropic_thinking_block_count": 1,
        "anthropic_thinking_block_types": ["thinking"],
    }
    continuation = response.reasoning.provider_private_continuation[0]
    assert continuation.kind == "thinking"
    assert continuation.signature == "sig_123"
    assert continuation.redacted_details["anthropic_block"] == {
        "type": "thinking",
        "thinking": "private reasoning text",
        "signature": "sig_123",
    }
    assert response.provider_metadata is not None
    thinking_item = response.provider_metadata.items[0]
    assert thinking_item.kind is ProviderItemKind.THINKING
    assert thinking_item.thinking_signature == "sig_123"


def test_empty_thinking_block_is_preserved_for_omitted_display() -> None:
    adapter = AnthropicMessagesAdapter(model_name="anthropic/claude-fable-5")

    response = adapter.parse_response(
        _json(
            {
                "id": "msg_fable",
                "type": "message",
                "role": "assistant",
                "model": "claude-fable-5",
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "sig_omitted"},
                    {"type": "text", "text": "ok"},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 8, "output_tokens": 6},
            }
        )
    )

    assert response.final_text == "ok"
    assert response.reasoning is not None
    continuation = response.reasoning.provider_private_continuation[0]
    assert continuation.signature == "sig_omitted"
    assert continuation.redacted_details["anthropic_block"] == {
        "type": "thinking",
        "thinking": "",
        "signature": "sig_omitted",
    }
    assert response.provider_metadata is not None
    thinking_item = response.provider_metadata.items[0]
    assert thinking_item.kind is ProviderItemKind.THINKING
    assert thinking_item.redacted_details == {
        "type": "thinking",
        "has_signature": True,
        "has_thinking": False,
        "thinking_omitted": True,
    }


def test_adaptive_thinking_tokens_in_output_tokens_details_lift_to_reasoning_tokens() -> None:
    """Adaptive endpoint puts ``thinking_tokens`` under ``output_tokens_details``.

    Older Messages calls (opus-4-5 / sonnet-4-5) keep ``thinking_tokens`` at the
    top of the usage block. Adaptive-thinking models (opus-4-6/4-7/4-8,
    sonnet-4-6) nest it under ``output_tokens_details`` instead. The adapter
    must read both locations and surface the count on ``reasoning_tokens`` so
    the OpenAI Responses and Anthropic Messages adapters agree.
    """

    adapter = AnthropicMessagesAdapter(model_name="claude-opus-4-7")
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="Think")])],
        reasoning=ReasoningSettings(effort=ReasoningEffort.MEDIUM),
        max_output_tokens=2048,
    )

    response = adapter.parse_response(
        _json(
            {
                "id": "msg_adaptive",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-7",
                "content": [
                    {"type": "thinking", "thinking": "private", "signature": "sig"},
                    {"type": "text", "text": "ok"},
                ],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 17,
                    "output_tokens_details": {"thinking_tokens": 40},
                },
            }
        ),
        request=request,
    )

    assert response.usage.tokens.reasoning_tokens == 40
    assert response.reasoning is not None
    assert response.reasoning.reasoning_tokens == 40
    # Breadcrumb survives in provider_details so downstream consumers can still
    # read the raw nested value.
    assert (
        response.usage.tokens.provider_details["output_tokens_details.thinking_tokens"]
        == 40
    )


def test_adaptive_and_top_level_thinking_tokens_sum() -> None:
    """If both locations are populated, sum them (defensive against API drift)."""

    adapter = AnthropicMessagesAdapter(model_name="claude-opus-4-7")
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="Think")])],
        reasoning=ReasoningSettings(effort=ReasoningEffort.LOW),
        max_output_tokens=512,
    )

    response = adapter.parse_response(
        _json(
            {
                "id": "msg_both",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 9,
                    "thinking_tokens": 3,
                    "output_tokens_details": {"thinking_tokens": 7},
                },
            }
        ),
        request=request,
    )

    assert response.usage.tokens.reasoning_tokens == 10


def test_thinking_plus_tool_use_round_trips_for_stateless_continuation() -> None:
    adapter = AnthropicMessagesAdapter(model_name="claude-sonnet-4-5")
    response = adapter.parse_response(
        _json(
            {
                "id": "msg_thinking_tool",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "I should call the tool.",
                        "signature": "sig_tool",
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_calc",
                        "name": "calculator",
                        "input": {"expression": "2+2"},
                    },
                ],
                "stop_reason": "tool_use",
            }
        )
    )
    tool_call = response.tool_calls[0]
    tool_result = ToolResult(
        tool_call_id=tool_call.call_id,
        tool_name=tool_call.tool_name,
        arguments=tool_call.arguments,
        success=True,
        output="4",
        provider_tool_use_id=tool_call.provider_tool_use_id,
    )

    payload = adapter.build_payload(
        ModelRequest(
            messages=[
                UserMessage(content=[TextContent(text="What is 2+2?")]),
                assistant_message_from_response(response),
                ToolResultMessage.from_result(tool_result),
            ],
            max_output_tokens=64,
        )
    )

    messages = cast(list[JsonObject], payload["messages"])
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == [
        {"type": "thinking", "thinking": "I should call the tool.", "signature": "sig_tool"},
        {
            "type": "tool_use",
            "id": "toolu_calc",
            "name": "calculator",
            "input": {"expression": "2+2"},
        },
    ]
    assert messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_calc", "content": "4"}
    ]


def test_empty_thinking_plus_tool_use_round_trips_for_stateless_continuation() -> None:
    adapter = AnthropicMessagesAdapter(model_name="anthropic/claude-fable-5")
    response = adapter.parse_response(
        _json(
            {
                "id": "msg_empty_thinking_tool",
                "type": "message",
                "role": "assistant",
                "model": "claude-fable-5",
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "sig_omitted"},
                    {
                        "type": "tool_use",
                        "id": "toolu_calc",
                        "name": "calculator",
                        "input": {"expression": "2+2"},
                    },
                ],
                "stop_reason": "tool_use",
            }
        )
    )
    tool_call = response.tool_calls[0]
    tool_result = ToolResult(
        tool_call_id=tool_call.call_id,
        tool_name=tool_call.tool_name,
        arguments=tool_call.arguments,
        success=True,
        output="4",
        provider_tool_use_id=tool_call.provider_tool_use_id,
    )

    payload = adapter.build_payload(
        ModelRequest(
            messages=[
                UserMessage(content=[TextContent(text="What is 2+2?")]),
                assistant_message_from_response(response),
                ToolResultMessage.from_result(tool_result),
            ],
            max_output_tokens=64,
        )
    )

    messages = cast(list[JsonObject], payload["messages"])
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == [
        {"type": "thinking", "thinking": "", "signature": "sig_omitted"},
        {
            "type": "tool_use",
            "id": "toolu_calc",
            "name": "calculator",
            "input": {"expression": "2+2"},
        },
    ]
    assert messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_calc", "content": "4"}
    ]


def test_max_tokens_response_maps_to_limit_stop_and_omits_partial_text() -> None:
    adapter = AnthropicMessagesAdapter(model_name="claude-sonnet-4-5")

    response = adapter.parse_response(
        _json(
            {
                "id": "msg_partial",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "partial"}],
                "stop_reason": "max_tokens",
            }
        )
    )

    assert response.provider_completion_status is ProviderCompletionStatus.INCOMPLETE
    assert response.stop_reason is StopReason.MAX_TOKENS
    assert response.incomplete_details == {"stop_reason": "max_tokens"}
    assert response.final_text is None
    assert response.response_metadata["partial_text_omitted_due_to_incomplete_status"] is True


def test_malformed_payloads_raise_protocol_error() -> None:
    adapter = AnthropicMessagesAdapter(model_name="claude-sonnet-4-5")

    with pytest.raises(ProviderProtocolError, match="content list"):
        adapter.parse_response(
            _json(
                {
                    "id": "msg_bad",
                    "type": "message",
                    "role": "assistant",
                    "stop_reason": "end_turn",
                }
            )
        )

    with pytest.raises(ProviderProtocolError, match="tool_use input"):
        adapter.parse_response(
            _json(
                {
                    "id": "msg_bad_tool",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_bad",
                            "name": "ls",
                            "input": _empty_json_list(),
                        }
                    ],
                    "stop_reason": "tool_use",
                }
            )
        )


async def test_generate_posts_request_and_parses_response_with_scripted_transport() -> None:
    transport = ScriptedJsonTransport(
        [
            JsonPostResponse(
                status_code=200,
                body=_json(
                    {
                        "id": "msg_http",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-sonnet-4-5",
                        "content": [{"type": "text", "text": "ok"}],
                        "stop_reason": "end_turn",
                    }
                ),
            )
        ]
    )
    adapter = AnthropicMessagesAdapter(model_name="claude-sonnet-4-5", transport=transport)
    request = ModelRequest(messages=[UserMessage(content=[TextContent(text="Reply ok")])])

    response = await adapter.generate(request)

    captured_body = cast(JsonObject, transport.requests[0].body)
    assert captured_body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Reply ok"}]}
    ]
    assert response.request_id == request.request_id
    assert response.response_id == "msg_http"
    assert response.final_text == "ok"
