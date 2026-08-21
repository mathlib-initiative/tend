from decimal import Decimal

import pytest
from pydantic import ValidationError

from tend._common.errors import ConfigurationError
from tend.llm.models import (
    CapabilityRequirements,
    ContextWindow,
    ContinuationStrategy,
    InputModality,
    ModelProfile,
    ModelSettingsCapabilities,
    ProviderApi,
    ProviderCompatibility,
    ReasoningCapabilities,
    ReasoningEffort,
    ReasoningSettings,
    ReasoningSummaryPreference,
    TokenPricing,
    ToolCapabilities,
    context_tokens_remaining,
    context_usage_ratio,
    get_builtin_profile,
    list_builtin_profile_keys,
    resolve_max_output_tokens,
    select_continuation_strategy,
    validate_capability_requirements,
)


def test_profile_capability_requirements_pass_for_supported_settings() -> None:
    profile = ModelProfile(
        provider_name="custom_openai",
        model_name="tool-reasoning-model",
        api=ProviderApi.OPENAI_RESPONSES,
        context_window=ContextWindow(tokens=8_000, source="test"),
        max_output_tokens=1_000,
        default_output_tokens=256,
        input_modalities=[InputModality.TEXT],
        tools=ToolCapabilities(
            supports_tool_calling=True,
            supports_strict_tool_schemas=True,
            supports_serial_tool_calls=True,
            supports_forced_tool_choice=True,
        ),
        reasoning=ReasoningCapabilities(
            supports_reasoning=True,
            supported_efforts=[ReasoningEffort.LOW, ReasoningEffort.MEDIUM],
            supports_reasoning_summaries=True,
            supported_summary_preferences=[
                ReasoningSummaryPreference.NONE,
                ReasoningSummaryPreference.AUTO,
            ],
        ),
        settings=ModelSettingsCapabilities(supports_temperature=True),
        pricing=TokenPricing(
            input_per_million_tokens=Decimal("1.25"),
            output_per_million_tokens=Decimal("10.00"),
            source="test fixture",
        ),
    )

    validate_capability_requirements(
        profile,
        CapabilityRequirements(
            reasoning=ReasoningSettings(
                effort=ReasoningEffort.LOW,
                summary=ReasoningSummaryPreference.AUTO,
            ),
            temperature=0.5,
            max_output_tokens=512,
            force_tool_choice=True,
        ),
    )

    assert context_usage_ratio(profile, 2_000) == 0.25
    assert context_tokens_remaining(profile, 2_000) == 6_000
    assert resolve_max_output_tokens(profile) == 256
    assert resolve_max_output_tokens(profile, 128) == 128
    assert ModelProfile.model_validate_json(profile.model_dump_json()) == profile


def test_capability_validation_fails_when_tool_calling_is_absent() -> None:
    profile = ModelProfile(
        provider_name="custom_openai",
        model_name="no-tools",
        api=ProviderApi.OPENAI_RESPONSES,
        tools=ToolCapabilities(
            supports_tool_calling=False,
            supports_strict_tool_schemas=False,
            supports_serial_tool_calls=False,
        ),
    )

    with pytest.raises(ConfigurationError, match="tool calling"):
        validate_capability_requirements(profile, CapabilityRequirements(require_tool_calling=True))


def test_capability_validation_fails_when_reasoning_is_required_but_absent() -> None:
    profile = ModelProfile(
        provider_name="custom_openai",
        model_name="no-reasoning",
        api=ProviderApi.OPENAI_RESPONSES,
        reasoning=ReasoningCapabilities(supports_reasoning=False, supported_efforts=[]),
    )

    with pytest.raises(ConfigurationError, match="explicit reasoning"):
        validate_capability_requirements(
            profile,
            CapabilityRequirements(require_tool_calling=False, require_reasoning=True),
        )


def test_unsupported_reasoning_effort_and_summary_fail_clearly() -> None:
    profile = ModelProfile(
        provider_name="custom_openai",
        model_name="limited-reasoning",
        api=ProviderApi.OPENAI_RESPONSES,
        tools=ToolCapabilities(
            supports_tool_calling=True,
            supports_strict_tool_schemas=True,
            supports_serial_tool_calls=True,
        ),
        reasoning=ReasoningCapabilities(
            supports_reasoning=True,
            supported_efforts=[ReasoningEffort.LOW],
        ),
    )

    with pytest.raises(ConfigurationError, match="reasoning effort"):
        validate_capability_requirements(
            profile,
            CapabilityRequirements(reasoning=ReasoningSettings(effort=ReasoningEffort.HIGH)),
        )

    with pytest.raises(ConfigurationError, match="reasoning summaries"):
        validate_capability_requirements(
            profile,
            CapabilityRequirements(
                reasoning=ReasoningSettings(summary=ReasoningSummaryPreference.AUTO)
            ),
        )


def test_cloudflare_openai_profile_rejects_temperature_and_defaults_to_stateless() -> None:
    profile = get_builtin_profile(ProviderApi.OPENAI_RESPONSES, "cloudflare_openai", "gpt-5")

    assert profile is not None
    assert profile.settings.supports_temperature is False
    assert profile.continuation.stateless_continuation_required is True
    assert select_continuation_strategy(profile, provider_side_enabled=True) == (
        ContinuationStrategy.STATELESS_REPLAY
    )

    with pytest.raises(ConfigurationError, match="temperature"):
        validate_capability_requirements(profile, CapabilityRequirements(temperature=0.0))

    with pytest.raises(ConfigurationError, match="provider-side continuation"):
        validate_capability_requirements(
            profile,
            CapabilityRequirements(require_provider_side_continuation=True),
        )


def test_anthropic_thinking_budget_and_forced_tool_choice_combination_are_validated() -> None:
    profile = get_builtin_profile(
        ProviderApi.ANTHROPIC_MESSAGES,
        "cloudflare_anthropic",
        "claude-sonnet-4-5",
    )

    assert profile is not None
    assert profile.reasoning.supports_anthropic_thinking is True

    with pytest.raises(ConfigurationError, match="thinking budget"):
        validate_capability_requirements(
            profile,
            CapabilityRequirements(
                reasoning=ReasoningSettings(effort=ReasoningEffort.LOW),
                thinking_enabled=True,
                thinking_budget_tokens=512,
                max_output_tokens=2_048,
            ),
        )

    with pytest.raises(ConfigurationError, match="forced tool choice with thinking"):
        validate_capability_requirements(
            profile,
            CapabilityRequirements(
                reasoning=ReasoningSettings(effort=ReasoningEffort.LOW),
                thinking_enabled=True,
                thinking_budget_tokens=1_024,
                max_output_tokens=2_048,
                force_tool_choice=True,
            ),
        )

    validate_capability_requirements(
        profile,
        CapabilityRequirements(
            reasoning=ReasoningSettings(effort=ReasoningEffort.LOW),
            thinking_enabled=True,
            thinking_budget_tokens=1_024,
            max_output_tokens=2_048,
        ),
    )


def test_max_output_limits_are_checked_by_helpers() -> None:
    profile = ModelProfile(
        provider_name="custom_openai",
        model_name="small-output",
        api=ProviderApi.OPENAI_RESPONSES,
        context_window=ContextWindow(tokens=4_096),
        max_output_tokens=512,
    )

    assert resolve_max_output_tokens(profile) == 512
    with pytest.raises(ConfigurationError, match="at most 512"):
        resolve_max_output_tokens(profile, 1_024)
    with pytest.raises(ConfigurationError, match="at most 512"):
        validate_capability_requirements(
            profile,
            CapabilityRequirements(
                require_tool_calling=False,
                require_reasoning=False,
                max_output_tokens=1_024,
            ),
        )


@pytest.mark.parametrize("estimated_tokens", [-1, -10])
def test_context_helpers_reject_negative_estimates(estimated_tokens: int) -> None:
    profile = ModelProfile(
        provider_name="custom_openai",
        model_name="context",
        api=ProviderApi.OPENAI_RESPONSES,
        context_window=ContextWindow(tokens=1_000),
    )

    with pytest.raises(ValueError, match="non-negative"):
        context_usage_ratio(profile, estimated_tokens)
    with pytest.raises(ValueError, match="non-negative"):
        context_tokens_remaining(profile, estimated_tokens)


def test_profile_models_reject_unknown_fields_but_allow_explicit_details() -> None:
    with pytest.raises(ValidationError):
        ProviderCompatibility.model_validate(
            {"zero_data_retention": True, "observed_previous_response_id_rejected": True}
        )

    compatibility = ProviderCompatibility(
        zero_data_retention=True,
        details={"observed_previous_response_id_rejected": True},
    )

    assert compatibility.details == {"observed_previous_response_id_rejected": True}


def test_profile_shape_validation_rejects_duplicates_and_inconsistent_capabilities() -> None:
    with pytest.raises(ValidationError, match="input modalities"):
        ModelProfile(
            provider_name="custom_openai",
            model_name="bad-modalities",
            api=ProviderApi.OPENAI_RESPONSES,
            input_modalities=[InputModality.TEXT, InputModality.TEXT],
        )

    with pytest.raises(ValidationError, match="supports_tool_calling"):
        ToolCapabilities(
            supports_tool_calling=False,
            supports_strict_tool_schemas=True,
        )

    with pytest.raises(ValidationError, match="summary"):
        ReasoningCapabilities(
            supports_reasoning_summaries=False,
            supported_summary_preferences=[ReasoningSummaryPreference.AUTO],
        )


def test_requires_adaptive_thinking_requires_anthropic_thinking_support() -> None:
    # The flag is meaningless without ``supports_anthropic_thinking``; pydantic
    # should reject the inconsistent capability.
    with pytest.raises(ValidationError, match="adaptive thinking"):
        ReasoningCapabilities(
            supports_reasoning=True,
            supported_efforts=[ReasoningEffort.HIGH],
            supports_anthropic_thinking=False,
            requires_adaptive_thinking=True,
        )

    with pytest.raises(ValidationError, match="always-on adaptive thinking"):
        ReasoningCapabilities(
            supports_reasoning=True,
            supported_efforts=[ReasoningEffort.HIGH],
            supports_anthropic_thinking=True,
            adaptive_thinking_always_on=True,
        )

    capabilities = ReasoningCapabilities(
        supports_reasoning=True,
        supported_efforts=[ReasoningEffort.HIGH],
        supports_anthropic_thinking=True,
        requires_adaptive_thinking=True,
        adaptive_thinking_always_on=True,
    )
    assert capabilities.requires_adaptive_thinking is True
    assert capabilities.adaptive_thinking_always_on is True


def test_cloudflare_opus_4_7_profile_flags_adaptive_thinking() -> None:
    profile = get_builtin_profile(
        ProviderApi.ANTHROPIC_MESSAGES,
        "cloudflare_anthropic",
        "claude-opus-4-7",
    )
    assert profile is not None
    assert profile.reasoning.requires_adaptive_thinking is True
    assert profile.reasoning.supports_anthropic_thinking is True

    legacy = get_builtin_profile(
        ProviderApi.ANTHROPIC_MESSAGES,
        "cloudflare_anthropic",
        "claude-opus-4-5",
    )
    assert legacy is not None
    assert legacy.reasoning.requires_adaptive_thinking is False


def test_cloudflare_opus_4_8_profile_flags_adaptive_thinking() -> None:
    profile = get_builtin_profile(
        ProviderApi.ANTHROPIC_MESSAGES,
        "cloudflare_anthropic",
        "claude-opus-4-8",
    )
    assert profile is not None
    assert profile.reasoning.requires_adaptive_thinking is True
    assert profile.reasoning.supports_anthropic_thinking is True
    assert profile.reasoning.supported_efforts == [
        ReasoningEffort.LOW,
        ReasoningEffort.MEDIUM,
        ReasoningEffort.HIGH,
        ReasoningEffort.XHIGH,
    ]


def test_cloudflare_fable_5_profile_captures_gateway_capabilities() -> None:
    profile = get_builtin_profile(
        ProviderApi.ANTHROPIC_MESSAGES,
        "cloudflare_anthropic",
        "anthropic/claude-fable-5",
    )
    assert profile is not None
    assert profile.context_window is not None
    assert profile.context_window.tokens == 1_000_000
    assert profile.max_output_tokens == 128_000
    assert profile.settings.supports_temperature is False
    assert profile.tools.supports_tool_calling is True
    assert profile.tools.supports_forced_tool_choice is False
    assert profile.reasoning.requires_adaptive_thinking is True
    assert profile.reasoning.adaptive_thinking_always_on is True
    assert profile.reasoning.supported_efforts == [
        ReasoningEffort.LOW,
        ReasoningEffort.MEDIUM,
        ReasoningEffort.HIGH,
        ReasoningEffort.XHIGH,
        ReasoningEffort.MAX,
    ]
    assert profile.compatibility.zero_data_retention is False
    assert profile.pricing is not None
    assert profile.pricing.input_per_million_tokens == Decimal("10.00")
    assert profile.pricing.output_per_million_tokens == Decimal("50.00")

    native_alias = get_builtin_profile(
        ProviderApi.ANTHROPIC_MESSAGES,
        "cloudflare_anthropic",
        "claude-fable-5",
    )
    assert native_alias is not None
    assert native_alias.model_name == "claude-fable-5"


def test_all_builtin_profiles_register_valid_sourced_token_limits() -> None:
    for key in list_builtin_profile_keys():
        profile = get_builtin_profile(*key)
        assert profile is not None
        assert profile.context_window is not None
        assert profile.context_window.source
        assert profile.max_output_tokens is not None
        assert profile.max_output_tokens <= profile.context_window.tokens


def test_all_builtin_anthropic_profiles_register_expected_token_limits() -> None:
    expected_limits = {
        (
            ProviderApi.ANTHROPIC_MESSAGES,
            "cloudflare_anthropic",
            "anthropic/claude-fable-5",
        ): (1_000_000, 128_000),
        (
            ProviderApi.ANTHROPIC_MESSAGES,
            "cloudflare_anthropic",
            "claude-fable-5",
        ): (1_000_000, 128_000),
        (
            ProviderApi.ANTHROPIC_MESSAGES,
            "cloudflare_anthropic",
            "claude-opus-4-5",
        ): (200_000, 64_000),
        (
            ProviderApi.ANTHROPIC_MESSAGES,
            "cloudflare_anthropic",
            "claude-opus-4-6",
        ): (1_000_000, 128_000),
        (
            ProviderApi.ANTHROPIC_MESSAGES,
            "cloudflare_anthropic",
            "claude-opus-4-7",
        ): (1_000_000, 128_000),
        (
            ProviderApi.ANTHROPIC_MESSAGES,
            "cloudflare_anthropic",
            "claude-opus-4-8",
        ): (1_000_000, 128_000),
        (
            ProviderApi.ANTHROPIC_MESSAGES,
            "cloudflare_anthropic",
            "claude-sonnet-4-5",
        ): (200_000, 64_000),
        (
            ProviderApi.ANTHROPIC_MESSAGES,
            "cloudflare_anthropic",
            "claude-sonnet-4-6",
        ): (1_000_000, 128_000),
    }
    anthropic_keys = {
        key for key in list_builtin_profile_keys() if key[0] is ProviderApi.ANTHROPIC_MESSAGES
    }

    assert anthropic_keys == set(expected_limits)
    for key, expected in expected_limits.items():
        profile = get_builtin_profile(*key)
        assert profile is not None
        assert profile.context_window is not None
        assert (profile.context_window.tokens, profile.max_output_tokens) == expected


def test_openai_profiles_register_expected_token_limits() -> None:
    expected_limits = {
        (ProviderApi.OPENAI_RESPONSES, "cloudflare_openai", "gpt-5"): (
            400_000,
            128_000,
        ),
        (ProviderApi.OPENAI_RESPONSES, "cloudflare_openai", "gpt-5.2"): (
            400_000,
            128_000,
        ),
        (ProviderApi.OPENAI_RESPONSES, "cloudflare_openai", "gpt-5.5"): (
            1_050_000,
            128_000,
        ),
        (ProviderApi.OPENAI_RESPONSES, "cloudflare_openai", "gpt-5.4-mini"): (
            400_000,
            128_000,
        ),
        (ProviderApi.OPENAI_RESPONSES, "cloudflare_openai", "gpt-5.4-pro"): (
            1_050_000,
            128_000,
        ),
    }
    openai_keys = {
        key for key in list_builtin_profile_keys() if key[0] is ProviderApi.OPENAI_RESPONSES
    }

    assert openai_keys == set(expected_limits)
    for key, expected in expected_limits.items():
        profile = get_builtin_profile(*key)
        assert profile is not None
        assert profile.context_window is not None
        assert (profile.context_window.tokens, profile.max_output_tokens) == expected


def test_builtin_profile_registry_is_deterministic_and_returns_copies() -> None:
    keys = list_builtin_profile_keys()
    first = get_builtin_profile(ProviderApi.OPENAI_RESPONSES, "cloudflare_openai", "gpt-5")
    second = get_builtin_profile(ProviderApi.OPENAI_RESPONSES, "cloudflare_openai", "gpt-5")

    assert (ProviderApi.OPENAI_RESPONSES, "cloudflare_openai", "gpt-5") in keys
    assert first is not None
    assert second is not None
    assert first == second
    assert first is not second
    assert get_builtin_profile(ProviderApi.OPENAI_RESPONSES, "other", "gpt-5") is None
