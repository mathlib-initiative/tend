from __future__ import annotations

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
    DeveloperMessage,
    ModelProfile,
    ModelRequest,
    ProviderApi,
    ReasoningEffort,
    ReasoningSettings,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolCapabilities,
    ToolResult,
    ToolResultMessage,
    UserMessage,
    get_builtin_profile,
)
from tend.llm.providers import AnthropicMessagesAdapter
from tend.llm.secrets import REDACTED_VALUE


def _cloudflare_sonnet_profile() -> ModelProfile:
    profile = get_builtin_profile(
        ProviderApi.ANTHROPIC_MESSAGES,
        "cloudflare_anthropic",
        "claude-sonnet-4-5",
    )
    assert profile is not None
    return profile


def _cloudflare_opus_4_7_profile() -> ModelProfile:
    profile = get_builtin_profile(
        ProviderApi.ANTHROPIC_MESSAGES,
        "cloudflare_anthropic",
        "claude-opus-4-7",
    )
    assert profile is not None
    return profile


def _cloudflare_opus_4_5_profile() -> ModelProfile:
    profile = get_builtin_profile(
        ProviderApi.ANTHROPIC_MESSAGES,
        "cloudflare_anthropic",
        "claude-opus-4-5",
    )
    assert profile is not None
    return profile


def _cloudflare_fable_5_profile() -> ModelProfile:
    profile = get_builtin_profile(
        ProviderApi.ANTHROPIC_MESSAGES,
        "cloudflare_anthropic",
        "anthropic/claude-fable-5",
    )
    assert profile is not None
    return profile


def test_plain_text_request_shape_uses_native_messages_system_and_temperature() -> None:
    adapter = AnthropicMessagesAdapter(
        model_name="claude-sonnet-4-5",
        provider_name="cloudflare_anthropic",
        base_url="https://gateway.example/v1/account/gateway/anthropic/v1",
        profile=_cloudflare_sonnet_profile(),
        timeout_seconds=12.5,
        default_request_settings={"temperature": 0.0},
    )
    request = ModelRequest(
        messages=[
            SystemMessage(content=[TextContent(text="You are careful.")]),
            DeveloperMessage(content=[TextContent(text="Prefer concise answers.")]),
            UserMessage(content=[TextContent(text="Reply with ok.")]),
        ],
        max_output_tokens=64,
    )

    http_request = adapter.build_http_request(request)

    assert http_request.url == "https://gateway.example/v1/account/gateway/anthropic/v1/messages"
    assert http_request.timeout_seconds == 12.5
    assert http_request.headers == {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    assert http_request.body == {
        "model": "claude-sonnet-4-5",
        "max_tokens": 64,
        "cache_control": {"type": "ephemeral"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Reply with ok."}],
            }
        ],
        "system": "You are careful.\n\nDeveloper instructions:\nPrefer concise answers.",
        "temperature": 0.0,
    }


def test_unset_max_output_tokens_uses_adapter_fallback_not_profile_maximum() -> None:
    profile = _cloudflare_sonnet_profile()
    assert profile.max_output_tokens == 64_000
    adapter = AnthropicMessagesAdapter(
        model_name="claude-sonnet-4-5",
        provider_name="cloudflare_anthropic",
        profile=profile,
    )
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="Reply briefly.")])]
    )

    payload = adapter.build_payload(request)

    assert payload["max_tokens"] == 1_024


def test_tool_schema_request_shape_uses_anthropic_native_input_schema() -> None:
    adapter = AnthropicMessagesAdapter(
        model_name="claude-sonnet-4-5",
        provider_name="cloudflare_anthropic",
        profile=_cloudflare_sonnet_profile(),
    )
    schemas = export_builtin_tool_schemas(["ls"])
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="List files.")])],
        tools=list(schemas),
    )

    payload = adapter.build_payload(request)

    assert payload["tools"] == [
        {
            "name": "ls",
            "description": schemas[0]["description"],
            "input_schema": schemas[0]["arguments_schema"],
        }
    ]


def test_tool_use_and_tool_result_continuation_shape_preserves_anthropic_ids() -> None:
    tool_call = ToolCall(
        call_id="call_local",
        tool_name="echo",
        arguments={"text": "hello"},
        order=0,
        provider_tool_use_id="toolu_123",
        provider_metadata={"anthropic_raw_input": {"text": "hello"}},
    )
    tool_result = ToolResult(
        tool_call_id="call_local",
        tool_name="echo",
        arguments={"text": "hello"},
        success=True,
        output={"echoed": "hello"},
        order=0,
        provider_tool_use_id="toolu_123",
    )
    adapter = AnthropicMessagesAdapter(model_name="claude-sonnet-4-5")
    request = ModelRequest(
        messages=[
            UserMessage(content=[TextContent(text="Use echo.")]),
            assistant_message_from_tool_calls(
                [tool_call],
                provider_metadata={
                    "anthropic_content_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "checked the tool to call",
                            "signature": "sig_123",
                        }
                    ]
                },
            ),
            ToolResultMessage.from_result(tool_result),
        ],
        max_output_tokens=64,
    )

    payload = adapter.build_payload(request)

    assert payload["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Use echo."}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "checked the tool to call",
                    "signature": "sig_123",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_123",
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
                    "tool_use_id": "toolu_123",
                    "content": '{"echoed":"hello"}',
                }
            ],
        },
    ]


def test_thinking_settings_shape_and_budget_validation() -> None:
    adapter = AnthropicMessagesAdapter(
        model_name="claude-sonnet-4-5",
        provider_name="cloudflare_anthropic",
        profile=_cloudflare_sonnet_profile(),
    )
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="Think briefly.")])],
        reasoning=ReasoningSettings(
            effort=ReasoningEffort.LOW,
            max_reasoning_tokens=1_024,
        ),
        max_output_tokens=2_048,
    )

    payload = adapter.build_payload(request)

    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 1024}

    with pytest.raises(ConfigurationError, match="thinking budget"):
        adapter.build_payload(
            ModelRequest(
                messages=[UserMessage(content=[TextContent(text="Too small budget.")])],
                reasoning=ReasoningSettings(
                    effort=ReasoningEffort.LOW,
                    max_reasoning_tokens=512,
                ),
                max_output_tokens=2_048,
            )
        )

    with pytest.raises(ConfigurationError, match="max output tokens greater than thinking budget"):
        adapter.build_payload(
            ModelRequest(
                messages=[UserMessage(content=[TextContent(text="Too few output tokens.")])],
                reasoning=ReasoningSettings(
                    effort=ReasoningEffort.LOW,
                    max_reasoning_tokens=1_024,
                ),
                max_output_tokens=1_024,
            )
        )


def test_thinking_with_tools_omits_forced_tool_choice_unless_profile_allows_it() -> None:
    schemas = export_builtin_tool_schemas(["ls"])
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="List files after thinking.")])],
        tools=list(schemas),
        reasoning=ReasoningSettings(effort=ReasoningEffort.LOW, max_reasoning_tokens=1_024),
        max_output_tokens=2_048,
        request_metadata={"force_tool_name": "ls"},
    )
    profile = _cloudflare_sonnet_profile()
    default_adapter = AnthropicMessagesAdapter(
        model_name="claude-sonnet-4-5",
        provider_name="cloudflare_anthropic",
        profile=profile,
    )
    compatible_profile = profile.model_copy(
        update={
            "tools": ToolCapabilities(
                supports_tool_calling=True,
                supports_strict_tool_schemas=True,
                supports_serial_tool_calls=True,
                supports_parallel_tool_calls=True,
                supports_forced_tool_choice=True,
                forced_tool_choice_compatible_with_thinking=True,
            )
        },
        deep=True,
    )
    compatible_adapter = AnthropicMessagesAdapter(
        model_name="claude-sonnet-4-5",
        provider_name="cloudflare_anthropic",
        profile=compatible_profile,
    )

    assert "tool_choice" not in default_adapter.build_payload(request)
    assert compatible_adapter.build_payload(request)["tool_choice"] == {
        "type": "tool",
        "name": "ls",
    }


def test_disable_reasoning_metadata_keeps_forced_tool_choice_despite_default_thinking() -> None:
    # Regression: an adapter configured with default thinking would otherwise resolve
    # request.reasoning=None back to that default, re-enabling thinking and silently
    # dropping the forced tool choice (cloudflare sonnet is thinking-incompatible).
    schemas = export_builtin_tool_schemas(["ls"])
    adapter = AnthropicMessagesAdapter(
        model_name="claude-sonnet-4-5",
        provider_name="cloudflare_anthropic",
        profile=_cloudflare_sonnet_profile(),
        default_reasoning=ReasoningSettings(
            effort=ReasoningEffort.LOW, max_reasoning_tokens=1_024
        ),
    )

    # Default thinking active: the force is dropped (the bug this PR must avoid).
    with_thinking = adapter.build_payload(
        ModelRequest(
            messages=[UserMessage(content=[TextContent(text="List files.")])],
            tools=list(schemas),
            max_output_tokens=2_048,
            request_metadata={"force_tool_name": "ls"},
        )
    )
    assert "thinking" in with_thinking
    assert "tool_choice" not in with_thinking

    # disable_reasoning suppresses the default thinking, so the force survives.
    forced = adapter.build_payload(
        ModelRequest(
            messages=[UserMessage(content=[TextContent(text="List files.")])],
            tools=list(schemas),
            max_output_tokens=2_048,
            request_metadata={"force_tool_name": "ls", "disable_reasoning": True},
        )
    )
    assert "thinking" not in forced
    assert forced["tool_choice"] == {"type": "tool", "name": "ls"}


def test_from_config_builds_required_headers_and_redacts_secrets() -> None:
    runtime = RuntimeConfig.model_validate(
        {
            "model": {
                "base_url": "https://gateway.example/v1/account/gateway/anthropic/v1",
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
        provider="cloudflare_anthropic",
        api=ProviderApi.ANTHROPIC_MESSAGES,
        model_name="claude-sonnet-4-5",
        settings=ModelSettingsConfig(temperature=0.0, max_output_tokens=64),
    )

    adapter = AnthropicMessagesAdapter.from_config(
        model,
        runtime.to_provider_runtime_config(),
        environment={"CF_AIG_TOKEN": "gateway-secret"},
    )
    http_request = adapter.build_http_request(
        ModelRequest(messages=[UserMessage(content=[TextContent(text="hello")])])
    )

    assert http_request.url == "https://gateway.example/v1/account/gateway/anthropic/v1/messages"
    assert http_request.headers["anthropic-version"] == "2023-06-01"
    assert http_request.headers["cf-aig-authorization"] == "gateway-secret"
    assert "x-api-key" not in http_request.headers
    body = cast(JsonObject, http_request.body)
    assert body["temperature"] == 0.0
    assert adapter.redacted_headers(http_request.headers)["cf-aig-authorization"] == REDACTED_VALUE


def test_base_url_falls_back_to_anthropic_base_url_env_var() -> None:
    """If cfg has no base_url, the provider reads ANTHROPIC_BASE_URL from env.

    Mirrors how the OpenAI SDK resolves OPENAI_BASE_URL — letting users route
    through the Cloudflare AI Gateway by exporting one env var per shell rather
    than writing the URL into every cfg.yaml.
    """

    runtime = RuntimeConfig.model_validate(
        {
            "model": {
                "extra_headers": [
                    {
                        "name": "cf-aig-authorization",
                        "source": HeaderValueSource.ENV,
                        "env_var": "CF_AIG_AUTHORIZATION",
                        "secret": True,
                    }
                ],
            }
        }
    )
    model = AgentModelConfig(
        provider="cloudflare_anthropic",
        api=ProviderApi.ANTHROPIC_MESSAGES,
        model_name="claude-opus-4-7",
        settings=ModelSettingsConfig(max_output_tokens=64),
    )
    adapter = AnthropicMessagesAdapter.from_config(
        model,
        runtime.to_provider_runtime_config(),
        environment={
            "CF_AIG_AUTHORIZATION": "gateway-secret",
            "ANTHROPIC_BASE_URL": "https://gateway.example/v1/acct/gw/anthropic/v1",
        },
    )
    http_request = adapter.build_http_request(
        ModelRequest(messages=[UserMessage(content=[TextContent(text="hi")])])
    )
    assert http_request.url == "https://gateway.example/v1/acct/gw/anthropic/v1/messages"


def test_direct_anthropic_api_key_header_is_redacted() -> None:
    adapter = AnthropicMessagesAdapter(
        model_name="claude-sonnet-4-5",
        environment={"ANTHROPIC_API_KEY": "anthropic-secret"},
        api_key_env_var="ANTHROPIC_API_KEY",
    )

    headers = adapter.build_headers()

    assert headers["x-api-key"] == "anthropic-secret"
    assert adapter.redacted_headers(headers)["x-api-key"] == REDACTED_VALUE


def test_adaptive_thinking_emits_output_config_for_newer_claudes() -> None:
    # opus-4-7 (and the other adaptive-thinking profiles) must emit
    # ``thinking: {type: "adaptive", display: "summarized"}`` and
    # ``output_config: {effort: ...}``, never the legacy ``budget_tokens``.
    adapter = AnthropicMessagesAdapter(
        model_name="claude-opus-4-7",
        provider_name="cloudflare_anthropic",
        profile=_cloudflare_opus_4_7_profile(),
    )
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="Think hard.")])],
        reasoning=ReasoningSettings(effort=ReasoningEffort.HIGH),
        max_output_tokens=4_096,
    )

    payload = adapter.build_payload(request)

    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "high"}
    thinking = payload["thinking"]
    assert isinstance(thinking, dict)
    assert "budget_tokens" not in thinking


def test_adaptive_thinking_without_effort_emits_no_thinking_block() -> None:
    # Adaptive thinking is opt-in via an explicit effort; without one, neither
    # ``thinking`` nor ``output_config`` should appear in the request body.
    adapter = AnthropicMessagesAdapter(
        model_name="claude-opus-4-7",
        provider_name="cloudflare_anthropic",
        profile=_cloudflare_opus_4_7_profile(),
    )
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="Reply briefly.")])],
        max_output_tokens=2_048,
    )

    payload = adapter.build_payload(request)

    assert "thinking" not in payload
    assert "output_config" not in payload


def test_adaptive_display_override_via_native_settings() -> None:
    adapter = AnthropicMessagesAdapter(
        model_name="claude-opus-4-7",
        provider_name="cloudflare_anthropic",
        profile=_cloudflare_opus_4_7_profile(),
    )
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="Think hard.")])],
        reasoning=ReasoningSettings(
            effort=ReasoningEffort.MEDIUM,
            native_settings={"thinking": {"display": "raw"}},
        ),
        max_output_tokens=4_096,
    )

    payload = adapter.build_payload(request)

    assert payload["thinking"] == {"type": "adaptive", "display": "raw"}
    assert payload["output_config"] == {"effort": "medium"}


def test_legacy_budget_thinking_preserved_for_non_adaptive_claudes() -> None:
    # opus-4-5 is the non-adaptive baseline: it must keep the legacy
    # ``thinking: {type: "enabled", budget_tokens: N}`` request shape so older
    # deployments continue to work unchanged.
    adapter = AnthropicMessagesAdapter(
        model_name="claude-opus-4-5",
        provider_name="cloudflare_anthropic",
        profile=_cloudflare_opus_4_5_profile(),
    )
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="Think hard.")])],
        reasoning=ReasoningSettings(effort=ReasoningEffort.HIGH),
        max_output_tokens=16_384,
    )

    payload = adapter.build_payload(request)

    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert "output_config" not in payload


def test_fable_5_max_effort_uses_adaptive_output_config() -> None:
    adapter = AnthropicMessagesAdapter(
        model_name="anthropic/claude-fable-5",
        provider_name="cloudflare_anthropic",
        profile=_cloudflare_fable_5_profile(),
    )
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="Think as deeply as needed.")])],
        reasoning=ReasoningSettings(effort=ReasoningEffort.MAX),
        max_output_tokens=4_096,
    )

    payload = adapter.build_payload(request)

    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "max"}


def test_fable_5_rejects_temperature_and_forced_tool_choice() -> None:
    adapter = AnthropicMessagesAdapter(
        model_name="anthropic/claude-fable-5",
        provider_name="cloudflare_anthropic",
        profile=_cloudflare_fable_5_profile(),
        default_request_settings={"temperature": 0.0},
    )

    with pytest.raises(ConfigurationError, match="temperature"):
        adapter.build_payload(
            ModelRequest(messages=[UserMessage(content=[TextContent(text="Reply ok.")])])
        )

    schemas = export_builtin_tool_schemas(["ls"])
    no_temperature_adapter = AnthropicMessagesAdapter(
        model_name="anthropic/claude-fable-5",
        provider_name="cloudflare_anthropic",
        profile=_cloudflare_fable_5_profile(),
    )
    with pytest.raises(ConfigurationError, match="forced tool choice"):
        no_temperature_adapter.build_payload(
            ModelRequest(
                messages=[UserMessage(content=[TextContent(text="List files.")])],
                tools=list(schemas),
                request_metadata={"force_tool_name": "ls"},
            )
        )


def test_fable_5_always_on_adaptive_display_without_effort() -> None:
    adapter = AnthropicMessagesAdapter(
        model_name="anthropic/claude-fable-5",
        provider_name="cloudflare_anthropic",
        profile=_cloudflare_fable_5_profile(),
    )
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="Reply with a summary.")])],
        reasoning=ReasoningSettings(native_settings={"thinking": {"display": "omitted"}}),
        max_output_tokens=4_096,
    )

    payload = adapter.build_payload(request)

    assert payload["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert "output_config" not in payload


def test_thinking_level_map_can_disable_thinking_for_an_effort() -> None:
    # When ``thinking_level_map`` maps an effort to ``None`` ("off"), no
    # thinking block should be emitted — pi behaves the same way.
    base = _cloudflare_opus_4_7_profile()
    profile = base.model_copy(
        update={
            "reasoning": base.reasoning.model_copy(
                update={
                    "thinking_level_map": {ReasoningEffort.LOW: None},
                },
                deep=True,
            )
        },
        deep=True,
    )
    adapter = AnthropicMessagesAdapter(
        model_name="claude-opus-4-7",
        provider_name="cloudflare_anthropic",
        profile=profile,
    )
    request = ModelRequest(
        messages=[UserMessage(content=[TextContent(text="Light touch only.")])],
        reasoning=ReasoningSettings(effort=ReasoningEffort.LOW),
        max_output_tokens=4_096,
    )

    payload = adapter.build_payload(request)

    assert "thinking" not in payload
    assert "output_config" not in payload
