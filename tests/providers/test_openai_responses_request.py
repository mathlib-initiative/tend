from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest

from tend._common.errors import ConfigurationError
from tend._common.types import JsonObject
from tend.agent.config import (
    AgentModelConfig,
    HeaderValueSource,
    ModelSettingsConfig,
    RuntimeConfig,
)
from tend.agent.context import assistant_message_from_tool_calls
from tend.agent.tools import export_builtin_tool_schemas
from tend.llm.models import (
    AssistantMessage,
    ContextWindow,
    ContinuationCapabilities,
    ContinuationStrategy,
    ModelMessage,
    ModelProfile,
    ModelRequest,
    ProviderApi,
    ProviderMetadata,
    ReasoningCapabilities,
    ReasoningEffort,
    ReasoningSettings,
    ReasoningSummaryPreference,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolCapabilities,
    ToolResult,
    ToolResultMessage,
    UserMessage,
    get_builtin_profile,
)
from tend.llm.providers import (
    JsonPostResponse,
    OpenAIResponsesAdapter,
    ProviderHTTPStatusError,
    ScriptedJsonTransport,
)
from tend.llm.secrets import REDACTED_VALUE


def _cloudflare_gpt5_profile() -> ModelProfile:
    profile = get_builtin_profile(ProviderApi.OPENAI_RESPONSES, "cloudflare_openai", "gpt-5")
    assert profile is not None
    return profile


@pytest.mark.parametrize("model_name", ["gpt-5", "gpt-5.2", "gpt-5.4-mini", "gpt-5.4-pro"])
def test_profile_output_limit_shapes_openai_request(model_name: str) -> None:
    profile = get_builtin_profile(
        ProviderApi.OPENAI_RESPONSES,
        "cloudflare_openai",
        model_name,
    )
    assert profile is not None
    adapter = OpenAIResponsesAdapter(
        model_name=model_name,
        provider_name="cloudflare_openai",
        profile=profile,
    )
    messages: list[ModelMessage] = [UserMessage(content=[TextContent(text="Reply with ok.")])]

    assert adapter.build_payload(ModelRequest(messages=messages))["max_output_tokens"] == 128_000
    assert (
        adapter.build_payload(ModelRequest(messages=messages, max_output_tokens=32_768))[
            "max_output_tokens"
        ]
        == 32_768
    )
    with pytest.raises(ConfigurationError, match="at most 128000"):
        adapter.build_payload(ModelRequest(messages=messages, max_output_tokens=128_001))


def test_plain_text_request_shape_uses_responses_input_and_minimal_reasoning() -> None:
    adapter = OpenAIResponsesAdapter(
        model_name="gpt-5",
        provider_name="cloudflare_openai",
        base_url="https://gateway.example/v1/account/gateway/openai",
        profile=_cloudflare_gpt5_profile(),
        timeout_seconds=12.5,
    )
    request = ModelRequest(
        messages=[
            SystemMessage(content=[TextContent(text="You are careful.")]),
            UserMessage(content=[TextContent(text="Reply with ok.")]),
        ],
        max_output_tokens=64,
    )

    http_request = adapter.build_http_request(request)

    assert http_request.url == "https://gateway.example/v1/account/gateway/openai/responses"
    assert http_request.timeout_seconds == 12.5
    assert http_request.headers == {"Content-Type": "application/json"}
    assert http_request.body == {
        "model": "gpt-5",
        "input": [
            {"role": "system", "content": "You are careful."},
            {"role": "user", "content": "Reply with ok."},
        ],
        "max_output_tokens": 64,
        "reasoning": {"effort": "minimal", "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
    }


def test_tool_schema_request_shape_uses_strict_function_tools_and_serial_hint() -> None:
    adapter = OpenAIResponsesAdapter(
        model_name="gpt-5",
        provider_name="cloudflare_openai",
        profile=_cloudflare_gpt5_profile(),
    )
    tool_schema = export_builtin_tool_schemas(["ls"])[0]
    expected_parameters = cast(JsonObject, deepcopy(tool_schema["arguments_schema"]))
    expected_parameters["required"] = ["path", "max_entries", "max_output_bytes"]
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="List files.")])],
        tools=[tool_schema],
    )

    payload = adapter.build_payload(request)

    assert payload["tools"] == [
        {
            "type": "function",
            "name": "ls",
            "description": tool_schema["description"],
            "parameters": expected_parameters,
            "strict": True,
        }
    ]
    assert payload["parallel_tool_calls"] is False


def test_force_tool_name_metadata_sets_responses_tool_choice() -> None:
    # Blocking review finding: the turn loop's forced re-ask sets
    # request_metadata["force_tool_name"], which the Responses adapter must translate
    # into a provider tool_choice or the force is a no-op.
    adapter = OpenAIResponsesAdapter(
        model_name="gpt-5",
        provider_name="cloudflare_openai",
        profile=_cloudflare_gpt5_profile(),
    )
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="Finish the task.")])],
        tools=[export_builtin_tool_schemas(["ls"])[0]],
        request_metadata={"force_tool_name": "ls"},
    )

    payload = adapter.build_payload(request)

    assert payload["tool_choice"] == {"type": "function", "name": "ls"}


def test_disable_reasoning_metadata_suppresses_default_reasoning() -> None:
    adapter = OpenAIResponsesAdapter(
        model_name="gpt-5",
        provider_name="cloudflare_openai",
        profile=_cloudflare_gpt5_profile(),
    )
    messages: list[ModelMessage] = [UserMessage(content=[TextContent(text="Reply with ok.")])]

    # gpt-5 otherwise defaults to minimal reasoning in the payload.
    assert "reasoning" in adapter.build_payload(ModelRequest(messages=messages))

    disabled = adapter.build_payload(
        ModelRequest(messages=messages, request_metadata={"disable_reasoning": True})
    )
    assert "reasoning" not in disabled


def test_explicit_reasoning_effort_and_summary_are_mapped() -> None:
    adapter = OpenAIResponsesAdapter(
        model_name="gpt-5",
        provider_name="cloudflare_openai",
        profile=_cloudflare_gpt5_profile(),
    )
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="Think briefly.")])],
        reasoning=ReasoningSettings(
            effort=ReasoningEffort.LOW,
            summary=ReasoningSummaryPreference.AUTO,
        ),
    )

    payload = adapter.build_payload(request)

    assert payload["reasoning"] == {"effort": "low", "summary": "auto"}
    assert payload["include"] == ["reasoning.encrypted_content"]


def test_reasoning_block_defaults_summary_to_auto_when_unset() -> None:
    # Match pi: when emitting reasoning, default the summary to "auto" if the
    # caller hasn't expressed a preference, and add the encrypted-continuation
    # include tag so stateless callers can persist provider-private state.
    adapter = OpenAIResponsesAdapter(
        model_name="gpt-5",
        provider_name="cloudflare_openai",
        profile=_cloudflare_gpt5_profile(),
    )
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="Think briefly.")])],
        reasoning=ReasoningSettings(effort=ReasoningEffort.LOW),
    )

    payload = adapter.build_payload(request)

    assert payload["reasoning"] == {"effort": "low", "summary": "auto"}
    assert payload["include"] == ["reasoning.encrypted_content"]


def test_reasoning_summary_preference_none_opts_out_of_summary() -> None:
    # Explicit NONE must still suppress the summary even though the new default
    # would otherwise emit ``summary: "auto"``.
    adapter = OpenAIResponsesAdapter(
        model_name="gpt-5",
        provider_name="cloudflare_openai",
        profile=_cloudflare_gpt5_profile(),
    )
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="Think briefly.")])],
        reasoning=ReasoningSettings(
            effort=ReasoningEffort.LOW,
            summary=ReasoningSummaryPreference.NONE,
        ),
    )

    payload = adapter.build_payload(request)

    assert payload["reasoning"] == {"effort": "low"}
    assert payload["include"] == ["reasoning.encrypted_content"]


def test_reasoning_summary_preference_none_without_effort_keeps_summary_disabled() -> None:
    # If NONE is the only explicit reasoning setting, the fallback default effort
    # must not re-add ``summary: "auto"``.
    adapter = OpenAIResponsesAdapter(
        model_name="gpt-5",
        provider_name="cloudflare_openai",
        profile=_cloudflare_gpt5_profile(),
    )
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="Think briefly.")])],
        reasoning=ReasoningSettings(summary=ReasoningSummaryPreference.NONE),
    )

    payload = adapter.build_payload(request)

    assert payload["reasoning"] == {"effort": "minimal"}
    assert payload["include"] == ["reasoning.encrypted_content"]


def test_encrypted_content_include_omitted_when_profile_lacks_support() -> None:
    # ``supports_encrypted_reasoning_content=False`` profiles should not get the
    # ``include`` tag (e.g., providers that don't return encrypted reasoning
    # blobs).
    profile = ModelProfile(
        provider_name="custom_openai",
        model_name="no-encrypted",
        api=ProviderApi.OPENAI_RESPONSES,
        tools=ToolCapabilities(
            supports_tool_calling=True,
            supports_strict_tool_schemas=True,
            supports_serial_tool_calls=True,
        ),
        reasoning=ReasoningCapabilities(
            supports_reasoning=True,
            supported_efforts=[ReasoningEffort.LOW],
            supports_reasoning_summaries=True,
            supported_summary_preferences=[
                ReasoningSummaryPreference.NONE,
                ReasoningSummaryPreference.AUTO,
            ],
            supports_encrypted_reasoning_content=False,
        ),
    )
    adapter = OpenAIResponsesAdapter(
        model_name="no-encrypted",
        provider_name="custom_openai",
        profile=profile,
    )
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="Think briefly.")])],
        reasoning=ReasoningSettings(effort=ReasoningEffort.LOW),
    )

    payload = adapter.build_payload(request)

    assert payload["reasoning"] == {"effort": "low", "summary": "auto"}
    assert "include" not in payload


def test_thinking_level_map_remaps_effort_and_can_disable_reasoning() -> None:
    # ``thinking_level_map`` either remaps an effort to a provider-specific level
    # or, with a ``None`` value, suppresses the reasoning block entirely.
    profile = ModelProfile(
        provider_name="custom_openai",
        model_name="mapped-effort",
        api=ProviderApi.OPENAI_RESPONSES,
        tools=ToolCapabilities(
            supports_tool_calling=True,
            supports_strict_tool_schemas=True,
            supports_serial_tool_calls=True,
        ),
        reasoning=ReasoningCapabilities(
            supports_reasoning=True,
            supported_efforts=[ReasoningEffort.LOW, ReasoningEffort.HIGH],
            supports_reasoning_summaries=True,
            supported_summary_preferences=[
                ReasoningSummaryPreference.NONE,
                ReasoningSummaryPreference.AUTO,
            ],
            supports_encrypted_reasoning_content=True,
            thinking_level_map={
                ReasoningEffort.HIGH: ReasoningEffort.MEDIUM,
                ReasoningEffort.LOW: None,
            },
        ),
    )
    adapter = OpenAIResponsesAdapter(
        model_name="mapped-effort",
        provider_name="custom_openai",
        profile=profile,
    )

    remapped = adapter.build_payload(
        ModelRequest(
            messages=[UserMessage(content=[TextContent(text="Think.")])],
            reasoning=ReasoningSettings(effort=ReasoningEffort.HIGH),
        )
    )
    assert remapped["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert remapped["include"] == ["reasoning.encrypted_content"]

    disabled = adapter.build_payload(
        ModelRequest(
            messages=[UserMessage(content=[TextContent(text="Think.")])],
            reasoning=ReasoningSettings(effort=ReasoningEffort.LOW),
        )
    )
    assert "reasoning" not in disabled
    assert "include" not in disabled


def test_stateless_function_call_and_tool_result_history_shape() -> None:
    tool_call = ToolCall(
        call_id="call_local",
        tool_name="echo",
        arguments={"text": "hello"},
        order=0,
        provider_item_id="fc_123",
        provider_call_id="call_provider",
        provider_status="completed",
    )
    tool_result = ToolResult(
        tool_call_id="call_local",
        tool_name="echo",
        arguments={"text": "hello"},
        success=True,
        output={"echoed": "hello"},
        order=0,
        provider_item_id="fc_123",
        provider_call_id="call_provider",
    )
    adapter = OpenAIResponsesAdapter(
        model_name="gpt-5",
        provider_name="cloudflare_openai",
        profile=_cloudflare_gpt5_profile(),
    )
    request = ModelRequest(
        messages=[
            UserMessage(content=[TextContent(text="Use echo.")]),
            assistant_message_from_tool_calls([tool_call]),
            ToolResultMessage.from_result(tool_result),
        ],
    )

    payload = adapter.build_payload(request)

    assert payload["input"] == [
        {"role": "user", "content": "Use echo."},
        {
            "type": "function_call",
            "call_id": "call_provider",
            "name": "echo",
            "arguments": '{"text":"hello"}',
            "status": "completed",
            "id": "fc_123",
        },
        {
            "type": "function_call_output",
            "call_id": "call_provider",
            "output": '{"echoed":"hello"}',
        },
    ]
    assert "previous_response_id" not in payload


def test_previous_response_id_is_capability_gated_and_disabled_for_cloudflare_zdr() -> None:
    safe_profile = ModelProfile(
        provider_name="custom_openai",
        model_name="safe-model",
        api=ProviderApi.OPENAI_RESPONSES,
        context_window=ContextWindow(tokens=8_000),
        tools=ToolCapabilities(supports_tool_calling=True, supports_serial_tool_calls=True),
        reasoning=ReasoningCapabilities(
            supports_reasoning=True,
            supported_efforts=[ReasoningEffort.MINIMAL],
        ),
        continuation=ContinuationCapabilities(
            supports_stateless_replay=True,
            supports_provider_response_id=True,
            provider_side_continuation_safe=True,
            stored_state_available=True,
            preferred_strategy=ContinuationStrategy.PROVIDER_RESPONSE_ID,
        ),
    )
    messages: list[ModelMessage] = [
        AssistantMessage(
            content=[TextContent(text="Earlier answer.")],
            provider_metadata={"model_response_id": "resp_123"},
        ),
        UserMessage(content=[TextContent(text="Continue.")]),
    ]

    disabled_adapter = OpenAIResponsesAdapter(
        model_name="safe-model",
        provider_name="custom_openai",
        profile=safe_profile,
        enable_provider_side_continuation=False,
    )
    enabled_adapter = OpenAIResponsesAdapter(
        model_name="safe-model",
        provider_name="custom_openai",
        profile=safe_profile,
        enable_provider_side_continuation=True,
    )
    cloudflare_adapter = OpenAIResponsesAdapter(
        model_name="gpt-5",
        provider_name="cloudflare_openai",
        profile=_cloudflare_gpt5_profile(),
        enable_provider_side_continuation=True,
    )

    assert "previous_response_id" not in disabled_adapter.build_payload(
        ModelRequest(messages=messages)
    )
    enabled_payload = enabled_adapter.build_payload(ModelRequest(messages=messages))
    assert enabled_payload["previous_response_id"] == "resp_123"
    assert "previous_response_id" not in cloudflare_adapter.build_payload(
        ModelRequest(
            messages=[UserMessage(content=[TextContent(text="Continue.")])],
            provider_metadata=ProviderMetadata(
                provider_name="cloudflare_openai",
                response_id="resp_cloudflare",
            ),
        )
    )


def test_unsupported_temperature_setting_fails_before_http() -> None:
    adapter = OpenAIResponsesAdapter(
        model_name="gpt-5",
        provider_name="cloudflare_openai",
        profile=_cloudflare_gpt5_profile(),
        default_request_settings={"temperature": 0.0},
    )

    with pytest.raises(ConfigurationError, match="temperature"):
        adapter.build_payload(ModelRequest(messages=[UserMessage(content=[TextContent(text="hi")])]))


def test_from_config_resolves_base_url_and_environment_headers_without_global_env_reads() -> None:
    runtime = RuntimeConfig.model_validate(
        {
            "model": {
                "base_url": "https://gateway.example/v1/account/gateway/openai",
                "extra_headers": [
                    {
                        "name": "cf-aig-authorization",
                        "source": HeaderValueSource.ENV,
                        "env_var": "CF_AIG_TOKEN",
                        "secret": True,
                    }
                ],
            }
        }
    )
    model = AgentModelConfig(
        provider="cloudflare_openai",
        api=ProviderApi.OPENAI_RESPONSES,
        model_name="gpt-5",
        settings=ModelSettingsConfig(
            reasoning=ReasoningSettings(effort=ReasoningEffort.MINIMAL)
        ),
    )

    adapter = OpenAIResponsesAdapter.from_config(
        model,
        runtime.to_provider_runtime_config(),
        environment={"CF_AIG_TOKEN": "gateway-secret"},
    )
    http_request = adapter.build_http_request(
        ModelRequest(messages=[UserMessage(content=[TextContent(text="hello")])])
    )

    assert http_request.url == "https://gateway.example/v1/account/gateway/openai/responses"
    assert http_request.headers["cf-aig-authorization"] == "gateway-secret"
    assert adapter.redacted_headers(http_request.headers)["cf-aig-authorization"] == REDACTED_VALUE


async def test_api_key_header_is_redacted_when_provider_error_is_raised() -> None:
    transport = ScriptedJsonTransport(
        [
            JsonPostResponse(
                status_code=400,
                body={"error": {"message": "bad request", "type": "invalid_request_error"}},
            )
        ],
        secret_header_names=["Authorization"],
    )
    adapter = OpenAIResponsesAdapter(
        model_name="gpt-5",
        transport=transport,
        environment={"OPENAI_API_KEY": "sk-secret"},
        api_key_env_var="OPENAI_API_KEY",
    )

    with pytest.raises(ProviderHTTPStatusError) as exc_info:
        awaitable = adapter.generate(
            ModelRequest(messages=[UserMessage(content=[TextContent(text="hello")])])
        )
        # Keep the await expression inside the raises block without hiding the
        # specific provider HTTP error behind Phase-37 parsing.
        await awaitable

    assert "sk-secret" not in str(exc_info.value)
    assert transport.requests[0].headers["Authorization"] == "Bearer sk-secret"
    assert adapter.redacted_headers(transport.requests[0].headers)["Authorization"] == (
        REDACTED_VALUE
    )


async def test_generate_posts_request_and_parses_success_response() -> None:
    transport = ScriptedJsonTransport(
        [
            JsonPostResponse(
                status_code=200,
                body={
                    "id": "resp_1",
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
                },
            )
        ]
    )
    adapter = OpenAIResponsesAdapter(model_name="gpt-5", transport=transport)

    request = ModelRequest(messages=[UserMessage(content=[TextContent(text="hi")])])
    response = await adapter.generate(request)

    captured_body = cast(JsonObject, transport.requests[0].body)
    assert captured_body["input"] == [{"role": "user", "content": "hi"}]
    assert response.response_id == "resp_1"
    assert response.final_text == "ok"
