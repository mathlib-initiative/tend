from __future__ import annotations

from typing import cast

import pytest
from pydantic import JsonValue

from tend._common.errors import ProviderProtocolError
from tend._common.types import JsonObject, StopReason
from tend.llm.models import (
    ModelRequest,
    ProviderCompletionStatus,
    ProviderItemKind,
    ReasoningEffort,
    ReasoningSettings,
    ReasoningSummaryPreference,
    TextContent,
    UserMessage,
)
from tend.llm.providers import JsonPostResponse, OpenAIResponsesAdapter, ScriptedJsonTransport


def _json(value: object) -> JsonValue:
    return cast(JsonValue, value)


def _empty_json_list() -> list[object]:
    return []


def test_final_text_response_extracts_text_ids_usage_and_metadata() -> None:
    adapter = OpenAIResponsesAdapter(model_name="gpt-5", provider_name="cloudflare_openai")
    request = ModelRequest(
        request_id="model_req_1",
        messages=[UserMessage(content=[TextContent(text="Say hi")])],
        reasoning=ReasoningSettings(effort=ReasoningEffort.MINIMAL),
    )

    response = adapter.parse_response(
        _json(
            {
                "id": "resp_1",
                "object": "response",
                "created_at": 123,
                "status": "completed",
                "model": "gpt-5-2025-08-07",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello"}],
                    }
                ],
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "total_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 2},
                    "output_tokens_details": {"reasoning_tokens": 1},
                },
            }
        ),
        request=request,
    )

    assert response.request_id == "model_req_1"
    assert response.response_id == "resp_1"
    assert response.final_text == "hello"
    assert response.stop_reason is StopReason.FINAL_RESPONSE
    assert response.provider_completion_status is ProviderCompletionStatus.COMPLETED
    assert response.usage.tokens.input_tokens == 5
    assert response.usage.tokens.output_tokens == 3
    assert response.usage.tokens.reasoning_tokens == 1
    assert response.usage.tokens.cache_read_tokens == 2
    assert response.usage.tokens.provider_details == {"total_tokens": 10}
    assert response.provider_metadata is not None
    assert response.provider_metadata.response_id == "resp_1"
    assert response.provider_metadata.model_name == "gpt-5-2025-08-07"
    assert response.provider_metadata.items[0].kind is ProviderItemKind.RESPONSE
    assert response.provider_metadata.items[0].provider_item_id == "msg_1"
    assert response.reasoning is not None
    assert response.reasoning.requested is not None
    assert response.reasoning.requested.effort is ReasoningEffort.MINIMAL
    assert response.reasoning.reasoning_tokens == 1


def test_message_content_skips_empty_output_text_parts() -> None:
    adapter = OpenAIResponsesAdapter(model_name="gpt-5", provider_name="cloudflare_openai")

    response = adapter.parse_response(
        _json(
            {
                "id": "resp_empty_and_text",
                "status": "completed",
                "model": "gpt-5",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": ""},
                            {"type": "output_text", "text": "done"},
                        ],
                    }
                ],
            }
        )
    )

    assert response.final_text == "done"
    assert response.stop_reason is StopReason.FINAL_RESPONSE
    assert response.provider_metadata is not None
    assert response.provider_metadata.items[0].provider_item_id == "msg_1"
    assert response.provider_metadata.items[0].kind is ProviderItemKind.RESPONSE


def test_top_level_output_text_skips_empty_text_parts_and_preserves_metadata() -> None:
    adapter = OpenAIResponsesAdapter(model_name="gpt-5", provider_name="cloudflare_openai")

    response = adapter.parse_response(
        _json(
            {
                "id": "resp_top_empty",
                "status": "completed",
                "model": "gpt-5",
                "output": [
                    {"id": "txt_empty", "type": "output_text", "text": ""},
                    {"id": "txt_done", "type": "output_text", "text": "done"},
                ],
            }
        )
    )

    assert response.final_text == "done"
    assert response.stop_reason is StopReason.FINAL_RESPONSE
    assert response.provider_metadata is not None
    assert response.provider_metadata.item_ids == ["txt_empty", "txt_done"]
    assert [item.kind for item in response.provider_metadata.items] == [
        ProviderItemKind.OUTPUT_TEXT,
        ProviderItemKind.OUTPUT_TEXT,
    ]
    assert [item.provider_item_id for item in response.provider_metadata.items] == [
        "txt_empty",
        "txt_done",
    ]


def test_completed_response_with_only_empty_text_has_no_final_text() -> None:
    adapter = OpenAIResponsesAdapter(model_name="gpt-5", provider_name="cloudflare_openai")

    response = adapter.parse_response(
        _json(
            {
                "id": "resp_empty_only",
                "status": "completed",
                "model": "gpt-5",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": ""}],
                    }
                ],
            }
        )
    )

    assert response.final_text is None
    assert response.assistant_message is None
    assert response.stop_reason is StopReason.PROVIDER_STOP_REASON
    assert response.provider_metadata is not None
    assert response.provider_metadata.item_ids == ["msg_1"]
    assert response.provider_metadata.items[0].provider_item_id == "msg_1"


@pytest.mark.parametrize(
    "output_item",
    [
        {"id": "txt_missing", "type": "output_text"},
        {"id": "txt_null", "type": "output_text", "text": None},
        {"id": "txt_int", "type": "output_text", "text": 123},
    ],
)
def test_malformed_top_level_output_text_requires_string_text(
    output_item: dict[str, object],
) -> None:
    adapter = OpenAIResponsesAdapter(model_name="gpt-5")

    with pytest.raises(ProviderProtocolError, match="output_text item requires string 'text'"):
        adapter.parse_response(
            _json(
                {
                    "id": "resp_bad_text",
                    "status": "completed",
                    "model": "gpt-5",
                    "output": [output_item],
                }
            )
        )


def test_function_tool_calls_decode_arguments_and_preserve_provider_ids_in_order() -> None:
    adapter = OpenAIResponsesAdapter(model_name="gpt-5", provider_name="cloudflare_openai")

    response = adapter.parse_response(
        _json(
            {
                "id": "resp_tools",
                "status": "completed",
                "model": "gpt-5",
                "output": [
                    {"id": "rs_1", "type": "reasoning", "summary": _empty_json_list()},
                    {
                        "id": "fc_1",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": "call_1",
                        "name": "ls",
                        "arguments": '{"path":"."}',
                    },
                    {
                        "id": "fc_2",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": "call_2",
                        "name": "read_file",
                        "arguments": '{"path":"README.md","limit":5}',
                    },
                ],
            }
        )
    )

    assert [call.call_id for call in response.tool_calls] == ["call_1", "call_2"]
    assert [call.order for call in response.tool_calls] == [0, 1]
    assert response.tool_calls[0].provider_item_id == "fc_1"
    assert response.tool_calls[0].provider_call_id == "call_1"
    assert response.tool_calls[0].provider_status == "completed"
    assert response.tool_calls[0].arguments == {"path": "."}
    assert response.tool_calls[0].provider_metadata["openai_raw_arguments"] == '{"path":"."}'
    assert response.tool_calls[1].arguments == {"path": "README.md", "limit": 5}
    assert response.provider_metadata is not None
    assert response.provider_metadata.item_ids == ["rs_1", "fc_1", "fc_2"]
    assert [item.kind for item in response.provider_metadata.items] == [
        ProviderItemKind.REASONING,
        ProviderItemKind.FUNCTION_CALL,
        ProviderItemKind.FUNCTION_CALL,
    ]
    assert response.provider_metadata.items[1].order == 1
    assert response.provider_metadata.items[1].provider_call_id == "call_1"


def test_reasoning_summary_encrypted_continuation_and_usage_details_are_extracted() -> None:
    adapter = OpenAIResponsesAdapter(model_name="gpt-5", provider_name="cloudflare_openai")
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="Think briefly")])],
        reasoning=ReasoningSettings(
            effort=ReasoningEffort.LOW,
            summary=ReasoningSummaryPreference.AUTO,
        ),
    )

    response = adapter.parse_response(
        _json(
            {
                "id": "resp_reasoning",
                "status": "completed",
                "model": "gpt-5",
                "reasoning": {"effort": "low", "summary": "detailed"},
                "output": [
                    {
                        "id": "rs_1",
                        "type": "reasoning",
                        "status": "completed",
                        "summary": [
                            {"type": "summary_text", "text": "Checked constraints."}
                        ],
                        "encrypted_content": "encrypted_reasoning_blob",
                    },
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    },
                ],
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 17,
                    "total_tokens": 28,
                    "input_tokens_details": {
                        "cached_tokens": 4,
                        "audio_tokens": 2,
                    },
                    "output_tokens_details": {
                        "reasoning_tokens": 9,
                        "accepted_prediction_tokens": 1,
                    },
                },
            }
        ),
        request=request,
    )

    assert response.final_text == "done"
    assert response.reasoning is not None
    assert response.reasoning.observed_effort == "low"
    assert response.reasoning.native_settings == {"effort": "low", "summary": "detailed"}
    assert response.reasoning.summaries[0].text == "Checked constraints."
    assert response.reasoning.summaries[0].provider_item_id == "rs_1"
    assert response.reasoning.provider_private_continuation[0].encrypted_content == (
        "encrypted_reasoning_blob"
    )
    assert response.reasoning.provider_private_continuation[0].provider_item_id == "rs_1"
    assert response.usage.tokens.provider_details == {
        "input_tokens_details.audio_tokens": 2,
        "output_tokens_details.accepted_prediction_tokens": 1,
        "total_tokens": 28,
    }
    assert response.provider_metadata is not None
    reasoning_item = response.provider_metadata.items[0]
    assert reasoning_item.kind is ProviderItemKind.REASONING
    assert reasoning_item.encrypted_reasoning_content == "encrypted_reasoning_blob"
    assert response.final_text == "done"


def test_incomplete_max_output_payload_maps_to_limit_stop_without_final_text() -> None:
    adapter = OpenAIResponsesAdapter(model_name="gpt-5", provider_name="cloudflare_openai")

    response = adapter.parse_response(
        _json(
            {
                "id": "resp_incomplete",
                "status": "incomplete",
                "model": "gpt-5",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [
                    {"id": "rs_1", "type": "reasoning", "summary": _empty_json_list()},
                    {
                        "id": "msg_partial",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "partial"}],
                    },
                ],
            }
        )
    )

    assert response.provider_completion_status is ProviderCompletionStatus.INCOMPLETE
    assert response.stop_reason is StopReason.MAX_TOKENS
    assert response.incomplete_details == {"reason": "max_output_tokens"}
    assert response.final_text is None
    assert response.response_metadata["partial_text_omitted_due_to_incomplete_status"] is True
    assert response.provider_metadata is not None
    assert response.provider_metadata.native_stop_reason == "incomplete"


def test_malformed_function_call_arguments_raise_protocol_error() -> None:
    adapter = OpenAIResponsesAdapter(model_name="gpt-5")

    with pytest.raises(ProviderProtocolError, match="function_call arguments"):
        adapter.parse_response(
            _json(
                {
                    "id": "resp_bad",
                    "status": "completed",
                    "model": "gpt-5",
                    "output": [
                        {
                            "id": "fc_bad",
                            "type": "function_call",
                            "call_id": "call_bad",
                            "name": "ls",
                            "arguments": "{not-json}",
                        }
                    ],
                }
            )
        )


def test_malformed_payload_without_output_list_raises_protocol_error() -> None:
    adapter = OpenAIResponsesAdapter(model_name="gpt-5")

    with pytest.raises(ProviderProtocolError, match="output list"):
        adapter.parse_response(_json({"id": "resp_bad", "status": "completed"}))


async def test_generate_posts_request_and_parses_response_with_scripted_transport() -> None:
    transport = ScriptedJsonTransport(
        [
            JsonPostResponse(
                status_code=200,
                body=_json(
                    {
                        "id": "resp_http",
                        "status": "completed",
                        "model": "gpt-5",
                        "output": [
                            {
                                "id": "msg_1",
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "ok"}],
                            }
                        ],
                    }
                ),
            )
        ]
    )
    adapter = OpenAIResponsesAdapter(model_name="gpt-5", transport=transport)
    request = ModelRequest(messages=[UserMessage(content=[TextContent(text="Reply ok")])])

    response = await adapter.generate(request)

    captured_body = cast(JsonObject, transport.requests[0].body)
    assert captured_body["input"] == [{"role": "user", "content": "Reply ok"}]
    assert response.request_id == request.request_id
    assert response.response_id == "resp_http"
    assert response.final_text == "ok"
