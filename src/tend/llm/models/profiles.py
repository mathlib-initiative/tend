"""Model profile and capability schemas for provider adapters."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Final

from pydantic import Field, field_validator, model_validator

from tend._common.errors import ConfigurationError
from tend._common.types import JsonObject, StrictModel
from tend.llm.models.provider import ContinuationStrategy
from tend.llm.models.reasoning import (
    ReasoningEffort,
    ReasoningSettings,
    ReasoningSummaryPreference,
)

_PositiveTokenCount = Annotated[int, Field(ge=1)]
_NonNegativePrice = Annotated[Decimal, Field(ge=Decimal("0"))]
_NonNegativeFloat = Annotated[float, Field(ge=0)]


def _default_text_input_modalities() -> list[InputModality]:
    return [InputModality.TEXT]


def _empty_reasoning_efforts() -> list[ReasoningEffort]:
    return []


def _default_summary_preferences() -> list[ReasoningSummaryPreference]:
    return [ReasoningSummaryPreference.NONE]


def _empty_strings() -> list[str]:
    return []


def _empty_thinking_level_map() -> dict[ReasoningEffort, ReasoningEffort | None]:
    return {}


def _profile_label(profile: ModelProfile) -> str:
    return f"{profile.provider_name}/{profile.model_name} ({profile.api.value})"


def _validate_non_negative_tokens(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


class ProviderApi(StrEnum):
    """Provider API family used by a model profile."""

    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"


class InputModality(StrEnum):
    """Input modalities a model profile can accept."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    FILE = "file"


class ContextWindow(StrictModel):
    """Known context-window size for compaction and observability estimates."""

    tokens: _PositiveTokenCount
    source: str | None = Field(default=None, min_length=1)

    def remaining_tokens(self, used_tokens: int) -> int:
        """Return remaining context tokens for an estimate, never below zero."""

        _validate_non_negative_tokens("used_tokens", used_tokens)
        return max(self.tokens - used_tokens, 0)

    def usage_ratio(self, used_tokens: int) -> float:
        """Return estimated context usage as a ratio of the known window."""

        _validate_non_negative_tokens("used_tokens", used_tokens)
        return used_tokens / self.tokens


class ToolCapabilities(StrictModel):
    """Tool-calling behavior supported by a model/provider combination."""

    supports_tool_calling: bool = False
    supports_strict_tool_schemas: bool = False
    supports_serial_tool_calls: bool = False
    supports_parallel_tool_calls: bool = False
    can_request_serial_tool_calls: bool = False
    supports_forced_tool_choice: bool = False
    forced_tool_choice_compatible_with_reasoning: bool = True
    forced_tool_choice_compatible_with_thinking: bool = True

    @model_validator(mode="after")
    def _validate_tool_capabilities(self) -> ToolCapabilities:
        if not self.supports_tool_calling:
            enabled_flags = [
                self.supports_strict_tool_schemas,
                self.supports_serial_tool_calls,
                self.supports_parallel_tool_calls,
                self.can_request_serial_tool_calls,
                self.supports_forced_tool_choice,
            ]
            if any(enabled_flags):
                raise ValueError("tool capability flags require supports_tool_calling=true")
        elif not self.supports_serial_tool_calls and not self.supports_parallel_tool_calls:
            raise ValueError("tool-calling models must support serial or parallel tool calls")
        return self


class ReasoningCapabilities(StrictModel):
    """Reasoning/thinking behavior supported by a model/provider combination."""

    supports_reasoning: bool = False
    supported_efforts: list[ReasoningEffort] = Field(default_factory=_empty_reasoning_efforts)
    supports_reasoning_summaries: bool = False
    supported_summary_preferences: list[ReasoningSummaryPreference] = Field(
        default_factory=_default_summary_preferences
    )
    supports_anthropic_thinking: bool = False
    # Newer Claude models (fable-5, opus-4-6/4-7/4-8, sonnet-4-6) require the
    # ``thinking: {type: "adaptive", display}`` + ``output_config: {effort}``
    # request shape instead of the legacy ``budget_tokens`` form. Mirrors pi's
    # ``compat.forceAdaptiveThinking`` flag.
    requires_adaptive_thinking: bool = False
    min_thinking_budget_tokens: _PositiveTokenCount | None = None
    max_thinking_budget_tokens: _PositiveTokenCount | None = None
    thinking_budget_must_be_less_than_max_output: bool = False
    supports_provider_private_reasoning_continuation: bool = False
    supports_encrypted_reasoning_content: bool = False
    # Some adaptive-only Anthropic models apply adaptive thinking even when the
    # request omits the ``thinking`` parameter. The flag is metadata for adapters
    # that need to distinguish "adaptive opt-in" from "adaptive always-on".
    adaptive_thinking_always_on: bool = False
    # Optional per-profile remap from our unified ``ReasoningEffort`` to a
    # provider-specific level. A value of ``None`` means "off" (suppress the
    # reasoning/thinking block entirely for that effort). Default empty map =
    # no remap (the unified effort is sent as-is).
    thinking_level_map: dict[ReasoningEffort, ReasoningEffort | None] = Field(
        default_factory=_empty_thinking_level_map
    )

    @field_validator("supported_efforts")
    @classmethod
    def _validate_unique_efforts(
        cls, efforts: list[ReasoningEffort]
    ) -> list[ReasoningEffort]:
        if len(set(efforts)) != len(efforts):
            raise ValueError("supported reasoning efforts must be unique")
        return efforts

    @field_validator("supported_summary_preferences")
    @classmethod
    def _validate_unique_summary_preferences(
        cls, preferences: list[ReasoningSummaryPreference]
    ) -> list[ReasoningSummaryPreference]:
        if len(set(preferences)) != len(preferences):
            raise ValueError("supported reasoning summary preferences must be unique")
        return preferences

    @model_validator(mode="after")
    def _validate_reasoning_capabilities(self) -> ReasoningCapabilities:
        if not self.supports_reasoning:
            if self.supported_efforts:
                raise ValueError("supported efforts require supports_reasoning=true")
            if self.supports_reasoning_summaries:
                raise ValueError("reasoning summaries require supports_reasoning=true")
            if self.supports_anthropic_thinking:
                raise ValueError("anthropic thinking requires supports_reasoning=true")
            if self.supports_provider_private_reasoning_continuation:
                raise ValueError("reasoning continuation requires supports_reasoning=true")
        elif not self.supported_efforts and not self.supports_anthropic_thinking:
            raise ValueError("reasoning support must declare efforts or thinking support")
        if not self.supports_reasoning_summaries:
            disallowed_summaries = [
                preference
                for preference in self.supported_summary_preferences
                if preference is not ReasoningSummaryPreference.NONE
            ]
            if disallowed_summaries:
                raise ValueError("non-none summary preferences require summary support")
        if not self.supports_anthropic_thinking:
            if self.min_thinking_budget_tokens is not None:
                raise ValueError("thinking budgets require anthropic thinking support")
            if self.max_thinking_budget_tokens is not None:
                raise ValueError("thinking budgets require anthropic thinking support")
            if self.requires_adaptive_thinking:
                raise ValueError(
                    "adaptive thinking requires supports_anthropic_thinking=true"
                )
            if self.adaptive_thinking_always_on:
                raise ValueError(
                    "always-on adaptive thinking requires supports_anthropic_thinking=true"
                )
        if self.adaptive_thinking_always_on and not self.requires_adaptive_thinking:
            raise ValueError("always-on adaptive thinking requires adaptive thinking")
        if (
            self.min_thinking_budget_tokens is not None
            and self.max_thinking_budget_tokens is not None
            and self.max_thinking_budget_tokens < self.min_thinking_budget_tokens
        ):
            raise ValueError("max thinking budget must be greater than or equal to min budget")
        return self


class ModelSettingsCapabilities(StrictModel):
    """Provider request settings that may be emitted for this profile."""

    supports_temperature: bool = True
    temperature_min: _NonNegativeFloat | None = 0.0
    temperature_max: _NonNegativeFloat | None = 2.0
    supports_max_output_tokens: bool = True
    supported_extra_settings: list[str] = Field(default_factory=_empty_strings)

    @field_validator("supported_extra_settings")
    @classmethod
    def _validate_extra_settings(cls, settings: list[str]) -> list[str]:
        if len(set(settings)) != len(settings):
            raise ValueError("supported extra settings must be unique")
        if any(not setting for setting in settings):
            raise ValueError("supported extra settings must be non-empty")
        return settings

    @model_validator(mode="after")
    def _validate_temperature_bounds(self) -> ModelSettingsCapabilities:
        if (
            self.temperature_min is not None
            and self.temperature_max is not None
            and self.temperature_max < self.temperature_min
        ):
            raise ValueError("temperature max must be greater than or equal to min")
        return self


class ContinuationCapabilities(StrictModel):
    """Continuation strategies supported by a provider/profile combination."""

    supports_stateless_replay: bool = True
    supports_provider_response_id: bool = False
    provider_side_continuation_safe: bool = False
    stored_state_available: bool = False
    stateless_continuation_required: bool = False
    zero_data_retention_disables_provider_state: bool = False
    preferred_strategy: ContinuationStrategy = ContinuationStrategy.STATELESS_REPLAY

    @model_validator(mode="after")
    def _validate_continuation_capabilities(self) -> ContinuationCapabilities:
        if self.preferred_strategy is ContinuationStrategy.STATELESS_REPLAY:
            if not self.supports_stateless_replay:
                raise ValueError("stateless replay preference requires stateless support")
        if self.preferred_strategy is ContinuationStrategy.PROVIDER_RESPONSE_ID:
            if not self.supports_provider_response_id:
                raise ValueError("provider response ID preference requires provider-side support")
            if not self.provider_side_continuation_safe:
                raise ValueError(
                    "provider response ID preference requires safe provider-side state"
                )
        if self.stateless_continuation_required and self.preferred_strategy is not (
            ContinuationStrategy.STATELESS_REPLAY
        ):
            raise ValueError("stateless-required profiles must prefer stateless replay")
        if (
            self.zero_data_retention_disables_provider_state
            and self.provider_side_continuation_safe
        ):
            raise ValueError("ZDR-disabled provider state cannot be marked safe")
        if self.provider_side_continuation_safe and not self.supports_provider_response_id:
            raise ValueError(
                "safe provider-side continuation requires provider response ID support"
            )
        return self


class TokenPricing(StrictModel):
    """Optional cost metadata for provider-reported token categories."""

    currency: str = Field(default="USD", min_length=3, max_length=8)
    input_per_million_tokens: _NonNegativePrice | None = None
    output_per_million_tokens: _NonNegativePrice | None = None
    reasoning_per_million_tokens: _NonNegativePrice | None = None
    cache_read_per_million_tokens: _NonNegativePrice | None = None
    cache_write_per_million_tokens: _NonNegativePrice | None = None
    source: str | None = Field(default=None, min_length=1)
    provider_details: JsonObject = Field(default_factory=dict)


class ProviderCompatibility(StrictModel):
    """Provider/gateway compatibility flags not captured by generic capability groups."""

    is_gateway: bool = False
    gateway_name: str | None = Field(default=None, min_length=1)
    zero_data_retention: bool = False
    requires_request_header_routing: bool = False
    native_reasoning_default: str | None = Field(default=None, min_length=1)
    details: JsonObject = Field(default_factory=dict)


class ModelProfile(StrictModel):
    """Provider-neutral model capabilities and compatibility metadata."""

    provider_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    api: ProviderApi
    context_window: ContextWindow | None = None
    max_output_tokens: _PositiveTokenCount | None = None
    default_output_tokens: _PositiveTokenCount | None = None
    input_modalities: list[InputModality] = Field(default_factory=_default_text_input_modalities)
    tools: ToolCapabilities = Field(default_factory=ToolCapabilities)
    reasoning: ReasoningCapabilities = Field(default_factory=ReasoningCapabilities)
    settings: ModelSettingsCapabilities = Field(default_factory=ModelSettingsCapabilities)
    continuation: ContinuationCapabilities = Field(default_factory=ContinuationCapabilities)
    pricing: TokenPricing | None = None
    compatibility: ProviderCompatibility = Field(default_factory=ProviderCompatibility)
    details: JsonObject = Field(default_factory=dict)

    @field_validator("input_modalities")
    @classmethod
    def _validate_input_modalities(
        cls, modalities: list[InputModality]
    ) -> list[InputModality]:
        if not modalities:
            raise ValueError("input modalities must not be empty")
        if len(set(modalities)) != len(modalities):
            raise ValueError("input modalities must be unique")
        return modalities

    @model_validator(mode="after")
    def _validate_profile_token_limits(self) -> ModelProfile:
        if (
            self.context_window is not None
            and self.max_output_tokens is not None
            and self.max_output_tokens > self.context_window.tokens
        ):
            raise ValueError("max output tokens cannot exceed context window")
        if (
            self.default_output_tokens is not None
            and self.max_output_tokens is not None
            and self.default_output_tokens > self.max_output_tokens
        ):
            raise ValueError("default output tokens cannot exceed max output tokens")
        return self


class CapabilityRequirements(StrictModel):
    """Requested capabilities/settings to validate before building provider requests."""

    require_tool_calling: bool = True
    require_reasoning: bool = True
    reasoning: ReasoningSettings | None = None
    temperature: _NonNegativeFloat | None = None
    max_output_tokens: _PositiveTokenCount | None = None
    force_tool_choice: bool = False
    thinking_enabled: bool = False
    thinking_budget_tokens: _PositiveTokenCount | None = None
    require_provider_side_continuation: bool = False


ProfileKey = tuple[ProviderApi, str, str]


def context_usage_ratio(profile: ModelProfile, estimated_tokens: int) -> float | None:
    """Return estimated context usage ratio, or ``None`` for unknown windows."""

    _validate_non_negative_tokens("estimated_tokens", estimated_tokens)
    if profile.context_window is None:
        return None
    return profile.context_window.usage_ratio(estimated_tokens)


def context_tokens_remaining(profile: ModelProfile, estimated_tokens: int) -> int | None:
    """Return remaining estimated context tokens, or ``None`` for unknown windows."""

    _validate_non_negative_tokens("estimated_tokens", estimated_tokens)
    if profile.context_window is None:
        return None
    return profile.context_window.remaining_tokens(estimated_tokens)


def resolve_max_output_tokens(
    profile: ModelProfile, requested_tokens: int | None = None
) -> int | None:
    """Resolve and validate request max-output tokens against a profile."""

    if requested_tokens is not None:
        _validate_non_negative_tokens("requested_tokens", requested_tokens)
        if requested_tokens == 0:
            raise ValueError("requested_tokens must be greater than zero")
    selected = requested_tokens or profile.default_output_tokens or profile.max_output_tokens
    if selected is None:
        return None
    if profile.max_output_tokens is not None and selected > profile.max_output_tokens:
        raise ConfigurationError(
            f"model {_profile_label(profile)} supports at most "
            f"{profile.max_output_tokens} output tokens; requested {selected}"
        )
    return selected


def select_continuation_strategy(
    profile: ModelProfile, *, provider_side_enabled: bool = False
) -> ContinuationStrategy:
    """Select the safe continuation strategy for a profile and runtime preference."""

    continuation = profile.continuation
    if (
        provider_side_enabled
        and continuation.supports_provider_response_id
        and continuation.provider_side_continuation_safe
        and not continuation.stateless_continuation_required
    ):
        return ContinuationStrategy.PROVIDER_RESPONSE_ID
    if continuation.supports_stateless_replay:
        return ContinuationStrategy.STATELESS_REPLAY
    return ContinuationStrategy.NONE


def validate_capability_requirements(
    profile: ModelProfile,
    requirements: CapabilityRequirements | None = None,
) -> None:
    """Fail fast when a profile cannot satisfy requested v1 capabilities/settings."""

    requested = requirements or CapabilityRequirements()
    label = _profile_label(profile)
    if requested.require_tool_calling and not profile.tools.supports_tool_calling:
        raise ConfigurationError(f"model {label} must support tool calling")
    if requested.require_reasoning and not profile.reasoning.supports_reasoning:
        raise ConfigurationError(f"model {label} must support explicit reasoning configuration")
    _validate_reasoning_request(profile, requested, label)
    _validate_tool_choice_request(profile, requested, label)
    _validate_setting_request(profile, requested, label)
    if requested.require_provider_side_continuation:
        strategy = select_continuation_strategy(profile, provider_side_enabled=True)
        if strategy is not ContinuationStrategy.PROVIDER_RESPONSE_ID:
            raise ConfigurationError(
                f"model {label} does not support safe provider-side continuation"
            )


def get_builtin_profile(
    api: ProviderApi, provider_name: str, model_name: str
) -> ModelProfile | None:
    """Return a deep copy of a built-in profile, if one is known."""

    profile = _BUILT_IN_PROFILES.get((api, provider_name, model_name))
    if profile is None:
        return None
    return profile.model_copy(deep=True)


def list_builtin_profile_keys() -> tuple[ProfileKey, ...]:
    """Return known built-in profile registry keys in deterministic order."""

    return tuple(sorted(_BUILT_IN_PROFILES, key=lambda key: tuple(str(part) for part in key)))


def _validate_reasoning_request(
    profile: ModelProfile, requested: CapabilityRequirements, label: str
) -> None:
    reasoning_request = requested.reasoning
    if reasoning_request is not None and not profile.reasoning.supports_reasoning:
        raise ConfigurationError(f"model {label} does not support requested reasoning settings")
    if reasoning_request is not None and reasoning_request.effort is not None:
        if reasoning_request.effort not in profile.reasoning.supported_efforts:
            raise ConfigurationError(
                f"model {label} does not support reasoning effort "
                f"{reasoning_request.effort.value!r}"
            )
    if reasoning_request is not None and reasoning_request.summary is not None:
        summary = reasoning_request.summary
        if summary is not ReasoningSummaryPreference.NONE:
            if not profile.reasoning.supports_reasoning_summaries:
                raise ConfigurationError(f"model {label} does not support reasoning summaries")
            if summary not in profile.reasoning.supported_summary_preferences:
                raise ConfigurationError(
                    f"model {label} does not support reasoning summary preference "
                    f"{summary.value!r}"
                )
    if requested.thinking_enabled:
        if not profile.reasoning.supports_anthropic_thinking:
            raise ConfigurationError(f"model {label} does not support Anthropic thinking")
        budget = requested.thinking_budget_tokens
        if budget is None and reasoning_request is not None:
            budget = reasoning_request.max_reasoning_tokens
        if budget is not None:
            _validate_thinking_budget(profile, requested, label, budget)


def _validate_thinking_budget(
    profile: ModelProfile,
    requested: CapabilityRequirements,
    label: str,
    budget: int,
) -> None:
    minimum = profile.reasoning.min_thinking_budget_tokens
    maximum = profile.reasoning.max_thinking_budget_tokens
    if minimum is not None and budget < minimum:
        raise ConfigurationError(
            f"model {label} requires thinking budget >= {minimum}; requested {budget}"
        )
    if maximum is not None and budget > maximum:
        raise ConfigurationError(
            f"model {label} supports thinking budget <= {maximum}; requested {budget}"
        )
    if (
        profile.reasoning.thinking_budget_must_be_less_than_max_output
        and requested.max_output_tokens is not None
        and budget >= requested.max_output_tokens
    ):
        raise ConfigurationError(
            f"model {label} requires max output tokens greater than thinking budget"
        )


def _validate_tool_choice_request(
    profile: ModelProfile, requested: CapabilityRequirements, label: str
) -> None:
    if requested.force_tool_choice and not profile.tools.supports_forced_tool_choice:
        raise ConfigurationError(f"model {label} does not support forced tool choice")
    if requested.force_tool_choice:
        if requested.reasoning is not None and not (
            profile.tools.forced_tool_choice_compatible_with_reasoning
        ):
            raise ConfigurationError(
                f"model {label} does not support forced tool choice with reasoning"
            )
        if requested.thinking_enabled and not (
            profile.tools.forced_tool_choice_compatible_with_thinking
        ):
            raise ConfigurationError(
                f"model {label} does not support forced tool choice with thinking"
            )


def _validate_setting_request(
    profile: ModelProfile, requested: CapabilityRequirements, label: str
) -> None:
    if requested.temperature is not None:
        if not profile.settings.supports_temperature:
            raise ConfigurationError(f"model {label} does not support temperature overrides")
        minimum = profile.settings.temperature_min
        maximum = profile.settings.temperature_max
        if minimum is not None and requested.temperature < minimum:
            raise ConfigurationError(
                f"model {label} requires temperature >= {minimum}; "
                f"requested {requested.temperature}"
            )
        if maximum is not None and requested.temperature > maximum:
            raise ConfigurationError(
                f"model {label} requires temperature <= {maximum}; "
                f"requested {requested.temperature}"
            )
    if requested.max_output_tokens is not None:
        if not profile.settings.supports_max_output_tokens:
            raise ConfigurationError(f"model {label} does not support max output token overrides")
        resolve_max_output_tokens(profile, requested.max_output_tokens)


_ANTHROPIC_PRICING_SOURCE: Final[str] = "https://platform.claude.com/docs/en/about-claude/pricing"
_OPENAI_PRICING_SOURCE: Final[str] = "https://openai.com/api/pricing/"

_ANTHROPIC_OPUS_PRICING: Final[TokenPricing] = TokenPricing(
    input_per_million_tokens=Decimal("5.00"),
    output_per_million_tokens=Decimal("25.00"),
    cache_write_per_million_tokens=Decimal("6.25"),
    cache_read_per_million_tokens=Decimal("0.50"),
    source=_ANTHROPIC_PRICING_SOURCE,
)

_ANTHROPIC_SONNET_PRICING: Final[TokenPricing] = TokenPricing(
    input_per_million_tokens=Decimal("3.00"),
    output_per_million_tokens=Decimal("15.00"),
    cache_write_per_million_tokens=Decimal("3.75"),
    cache_read_per_million_tokens=Decimal("0.30"),
    source=_ANTHROPIC_PRICING_SOURCE,
)

_ANTHROPIC_FABLE_PRICING: Final[TokenPricing] = TokenPricing(
    input_per_million_tokens=Decimal("10.00"),
    output_per_million_tokens=Decimal("50.00"),
    cache_write_per_million_tokens=Decimal("12.50"),
    cache_read_per_million_tokens=Decimal("1.00"),
    source=_ANTHROPIC_PRICING_SOURCE,
)


def _cloudflare_openai_profile(
    model_name: str,
    supported_efforts: list[ReasoningEffort] | None = None,
    pricing: TokenPricing | None = None,
    *,
    context_window_tokens: int | None = None,
    max_output_tokens: int | None = None,
    token_limit_source: str | None = None,
) -> ModelProfile:
    if supported_efforts is None:
        supported_efforts = [
            ReasoningEffort.MINIMAL,
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
        ]
    context_window: ContextWindow | None = None
    if context_window_tokens is not None:
        context_window = ContextWindow(
            tokens=context_window_tokens,
            source=token_limit_source,
        )
    return ModelProfile(
        provider_name="cloudflare_openai",
        model_name=model_name,
        api=ProviderApi.OPENAI_RESPONSES,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        tools=ToolCapabilities(
            supports_tool_calling=True,
            supports_strict_tool_schemas=True,
            supports_serial_tool_calls=True,
            supports_parallel_tool_calls=True,
            can_request_serial_tool_calls=True,
            supports_forced_tool_choice=True,
        ),
        reasoning=ReasoningCapabilities(
            supports_reasoning=True,
            supported_efforts=supported_efforts,
            supports_reasoning_summaries=True,
            supported_summary_preferences=[
                ReasoningSummaryPreference.NONE,
                ReasoningSummaryPreference.AUTO,
                ReasoningSummaryPreference.CONCISE,
                ReasoningSummaryPreference.DETAILED,
            ],
            supports_provider_private_reasoning_continuation=True,
            supports_encrypted_reasoning_content=True,
        ),
        settings=ModelSettingsCapabilities(supports_temperature=False),
        continuation=ContinuationCapabilities(
            supports_stateless_replay=True,
            supports_provider_response_id=True,
            provider_side_continuation_safe=False,
            stored_state_available=False,
            stateless_continuation_required=True,
            zero_data_retention_disables_provider_state=True,
            preferred_strategy=ContinuationStrategy.STATELESS_REPLAY,
        ),
        compatibility=ProviderCompatibility(
            is_gateway=True,
            gateway_name="cloudflare_ai_gateway",
            zero_data_retention=True,
            native_reasoning_default="medium",
            details={"observed_previous_response_id_rejected": True},
        ),
        pricing=pricing,
    )


_ADAPTIVE_THINKING_MODEL_TAGS: Final[tuple[str, ...]] = (
    "fable-5",
    "opus-4-6",
    "opus-4-7",
    "opus-4-8",
    "sonnet-4-6",
)


def _cloudflare_anthropic_profile(
    model_name: str,
    *,
    supported_efforts: list[ReasoningEffort] | None = None,
    pricing_override: TokenPricing | None = None,
    context_window_tokens: int | None = None,
    max_output_tokens: int | None = None,
    token_limit_source: str | None = None,
    supports_temperature: bool = True,
    supports_forced_tool_choice: bool = True,
    forced_tool_choice_compatible_with_thinking: bool = False,
    adaptive_thinking_always_on: bool = False,
    zero_data_retention: bool = True,
    compatibility_details: JsonObject | None = None,
) -> ModelProfile:
    if pricing_override is not None:
        pricing: TokenPricing | None = pricing_override
    elif "fable" in model_name:
        pricing = _ANTHROPIC_FABLE_PRICING
    elif "opus" in model_name:
        pricing = _ANTHROPIC_OPUS_PRICING
    elif "sonnet" in model_name:
        pricing = _ANTHROPIC_SONNET_PRICING
    else:
        pricing = None
    requires_adaptive_thinking = any(tag in model_name for tag in _ADAPTIVE_THINKING_MODEL_TAGS)
    # The legacy budget-based form enforces a per-request budget less than
    # max_output_tokens. Adaptive thinking leaves the decision to the model and
    # does not require/accept a budget, so the constraint is dropped there.
    thinking_budget_constraint = not requires_adaptive_thinking
    min_budget_tokens: int | None = None if requires_adaptive_thinking else 1024
    if supported_efforts is None:
        supported_efforts = [
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
        ]
    context_window: ContextWindow | None = None
    if context_window_tokens is not None:
        context_window = ContextWindow(tokens=context_window_tokens, source=token_limit_source)
    return ModelProfile(
        provider_name="cloudflare_anthropic",
        model_name=model_name,
        api=ProviderApi.ANTHROPIC_MESSAGES,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        tools=ToolCapabilities(
            supports_tool_calling=True,
            supports_strict_tool_schemas=True,
            supports_serial_tool_calls=True,
            supports_parallel_tool_calls=True,
            supports_forced_tool_choice=supports_forced_tool_choice,
            forced_tool_choice_compatible_with_thinking=(
                forced_tool_choice_compatible_with_thinking
            ),
        ),
        reasoning=ReasoningCapabilities(
            supports_reasoning=True,
            supported_efforts=supported_efforts,
            supports_anthropic_thinking=True,
            requires_adaptive_thinking=requires_adaptive_thinking,
            adaptive_thinking_always_on=adaptive_thinking_always_on,
            min_thinking_budget_tokens=min_budget_tokens,
            thinking_budget_must_be_less_than_max_output=thinking_budget_constraint,
            supports_provider_private_reasoning_continuation=True,
        ),
        settings=ModelSettingsCapabilities(supports_temperature=supports_temperature),
        continuation=ContinuationCapabilities(
            supports_stateless_replay=True,
            supports_provider_response_id=False,
            provider_side_continuation_safe=False,
            stored_state_available=False,
            stateless_continuation_required=True,
            preferred_strategy=ContinuationStrategy.STATELESS_REPLAY,
        ),
        compatibility=ProviderCompatibility(
            is_gateway=True,
            gateway_name="cloudflare_ai_gateway",
            zero_data_retention=zero_data_retention,
            details=compatibility_details or {},
        ),
        pricing=pricing,
    )


def _cloudflare_fable_profile(model_name: str) -> ModelProfile:
    return _cloudflare_anthropic_profile(
        model_name,
        supported_efforts=[
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
            ReasoningEffort.XHIGH,
            ReasoningEffort.MAX,
        ],
        context_window_tokens=1_000_000,
        max_output_tokens=128_000,
        token_limit_source=(
            "Cloudflare model catalog and Anthropic Claude Fable 5 launch docs"
        ),
        supports_temperature=False,
        supports_forced_tool_choice=False,
        forced_tool_choice_compatible_with_thinking=True,
        adaptive_thinking_always_on=True,
        zero_data_retention=False,
        compatibility_details={
            "cloudflare_unified_model_id": "anthropic/claude-fable-5",
            "covered_model_data_retention": "30 days",
        },
    )


_BUILT_IN_PROFILES: Final[dict[ProfileKey, ModelProfile]] = {
    (ProviderApi.OPENAI_RESPONSES, "cloudflare_openai", "gpt-5"): _cloudflare_openai_profile(
        "gpt-5",
        pricing=TokenPricing(
            input_per_million_tokens=Decimal("1.25"),
            output_per_million_tokens=Decimal("10.00"),
            cache_read_per_million_tokens=Decimal("0.125"),
            source=_OPENAI_PRICING_SOURCE,
        ),
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        token_limit_source="pi OpenAI model catalog (@earendil-works/pi-ai)",
    ),
    (ProviderApi.OPENAI_RESPONSES, "cloudflare_openai", "gpt-5.5"): _cloudflare_openai_profile(
        "gpt-5.5",
        supported_efforts=[
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
            ReasoningEffort.XHIGH,
        ],
        pricing=TokenPricing(
            input_per_million_tokens=Decimal("5.00"),
            output_per_million_tokens=Decimal("30.00"),
            cache_read_per_million_tokens=Decimal("0.50"),
            source=_OPENAI_PRICING_SOURCE,
        ),
        # The old 272k window came from pi's openai-codex/ChatGPT backend
        # route, not the Cloudflare OpenAI route registered by this profile.
        context_window_tokens=1_050_000,
        max_output_tokens=128_000,
        token_limit_source=(
            "pi cloudflare-ai-gateway gpt-5.5 catalog entry "
            "(@earendil-works/pi-ai) and "
            "https://platform.openai.com/docs/models/gpt-5.5"
        ),
    ),
    (ProviderApi.OPENAI_RESPONSES, "cloudflare_openai", "gpt-5.2"): _cloudflare_openai_profile(
        "gpt-5.2",
        supported_efforts=[
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
            ReasoningEffort.XHIGH,
        ],
        pricing=TokenPricing(
            input_per_million_tokens=Decimal("0.875"),
            output_per_million_tokens=Decimal("7.00"),
            cache_read_per_million_tokens=Decimal("0.0875"),
            source=_OPENAI_PRICING_SOURCE,
        ),
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        token_limit_source=(
            "pi cloudflare-ai-gateway model catalog (@earendil-works/pi-ai)"
        ),
    ),
    (
        ProviderApi.OPENAI_RESPONSES,
        "cloudflare_openai",
        "gpt-5.4-pro",
    ): _cloudflare_openai_profile(
        "gpt-5.4-pro",
        supported_efforts=[
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
            ReasoningEffort.XHIGH,
        ],
        pricing=TokenPricing(
            input_per_million_tokens=Decimal("30.00"),
            output_per_million_tokens=Decimal("180.00"),
            cache_read_per_million_tokens=Decimal("3.00"),
            source=_OPENAI_PRICING_SOURCE,
        ),
        context_window_tokens=1_050_000,
        max_output_tokens=128_000,
        token_limit_source="pi OpenAI model catalog (@earendil-works/pi-ai)",
    ),
    (
        ProviderApi.OPENAI_RESPONSES,
        "cloudflare_openai",
        "gpt-5.4-mini",
    ): _cloudflare_openai_profile(
        "gpt-5.4-mini",
        supported_efforts=[
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
            ReasoningEffort.XHIGH,
        ],
        pricing=TokenPricing(
            input_per_million_tokens=Decimal("0.75"),
            output_per_million_tokens=Decimal("4.50"),
            cache_read_per_million_tokens=Decimal("0.075"),
            source=_OPENAI_PRICING_SOURCE,
        ),
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        token_limit_source="pi OpenAI model catalog (@earendil-works/pi-ai)",
    ),
    (
        ProviderApi.ANTHROPIC_MESSAGES,
        "cloudflare_anthropic",
        "anthropic/claude-fable-5",
    ): _cloudflare_fable_profile("anthropic/claude-fable-5"),
    (
        ProviderApi.ANTHROPIC_MESSAGES,
        "cloudflare_anthropic",
        "claude-fable-5",
    ): _cloudflare_fable_profile("claude-fable-5"),
    (
        ProviderApi.ANTHROPIC_MESSAGES,
        "cloudflare_anthropic",
        "claude-sonnet-4-5",
    ): _cloudflare_anthropic_profile(
        "claude-sonnet-4-5",
        context_window_tokens=200_000,
        max_output_tokens=64_000,
        token_limit_source=(
            "pi cloudflare-ai-gateway model catalog (@earendil-works/pi-ai) and "
            "Anthropic model overview"
        ),
    ),
    (
        ProviderApi.ANTHROPIC_MESSAGES,
        "cloudflare_anthropic",
        "claude-sonnet-4-6",
    ): _cloudflare_anthropic_profile(
        "claude-sonnet-4-6",
        context_window_tokens=1_000_000,
        # Pi's Cloudflare catalog diverges at 64k, while Anthropic's overview and
        # pi's direct-Anthropic catalog both specify 128k. Use the upper documented
        # bound because this value is an enforced local validator cap and Cloudflare
        # documents no gateway-specific lower limit.
        max_output_tokens=128_000,
        token_limit_source=(
            "Anthropic model overview and pi direct-Anthropic model catalog "
            "(@earendil-works/pi-ai)"
        ),
    ),
    (
        ProviderApi.ANTHROPIC_MESSAGES,
        "cloudflare_anthropic",
        "claude-opus-4-5",
    ): _cloudflare_anthropic_profile(
        "claude-opus-4-5",
        context_window_tokens=200_000,
        max_output_tokens=64_000,
        token_limit_source=(
            "pi cloudflare-ai-gateway model catalog (@earendil-works/pi-ai) and "
            "Anthropic model overview"
        ),
    ),
    (
        ProviderApi.ANTHROPIC_MESSAGES,
        "cloudflare_anthropic",
        "claude-opus-4-6",
    ): _cloudflare_anthropic_profile(
        "claude-opus-4-6",
        context_window_tokens=1_000_000,
        max_output_tokens=128_000,
        token_limit_source=(
            "pi cloudflare-ai-gateway model catalog (@earendil-works/pi-ai) and "
            "Anthropic model overview"
        ),
    ),
    (
        ProviderApi.ANTHROPIC_MESSAGES,
        "cloudflare_anthropic",
        "claude-opus-4-7",
    ): _cloudflare_anthropic_profile(
        "claude-opus-4-7",
        context_window_tokens=1_000_000,
        max_output_tokens=128_000,
        token_limit_source=(
            "pi cloudflare-ai-gateway model catalog (@earendil-works/pi-ai) and "
            "Anthropic model overview"
        ),
    ),
    (
        ProviderApi.ANTHROPIC_MESSAGES,
        "cloudflare_anthropic",
        "claude-opus-4-8",
    ): _cloudflare_anthropic_profile(
        "claude-opus-4-8",
        supported_efforts=[
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
            ReasoningEffort.XHIGH,
        ],
        context_window_tokens=1_000_000,
        max_output_tokens=128_000,
        token_limit_source=(
            "pi cloudflare-ai-gateway model catalog (@earendil-works/pi-ai) and "
            "Anthropic model overview"
        ),
    ),
}

__all__ = (
    "CapabilityRequirements",
    "ContextWindow",
    "ContinuationCapabilities",
    "InputModality",
    "ModelProfile",
    "ModelSettingsCapabilities",
    "ProfileKey",
    "ProviderApi",
    "ProviderCompatibility",
    "ReasoningCapabilities",
    "TokenPricing",
    "ToolCapabilities",
    "context_tokens_remaining",
    "context_usage_ratio",
    "get_builtin_profile",
    "list_builtin_profile_keys",
    "resolve_max_output_tokens",
    "select_continuation_strategy",
    "validate_capability_requirements",
)
