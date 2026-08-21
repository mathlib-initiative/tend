"""LLM/provider configuration models and deterministic profile resolution."""

from __future__ import annotations

from re import compile
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from tend._common.env import ENV_NAME_PATTERN
from tend._common.types import JsonObject, StrictModel
from tend.llm.models.profiles import (
    CapabilityRequirements,
    ModelProfile,
    ProviderApi,
    get_builtin_profile,
    validate_capability_requirements,
)
from tend.llm.models.reasoning import ReasoningSettings
from tend.llm.secrets import HeaderValueSource, ProviderHeaderConfig

_PositiveInt = Annotated[int, Field(ge=1)]
_NonNegativeFloat = Annotated[float, Field(ge=0)]


def _empty_strings() -> list[str]:
    return []


def _empty_json_object() -> JsonObject:
    return {}


def _empty_provider_headers() -> list[ProviderHeaderConfig]:
    return []


class ModelSettingsConfig(StrictModel):
    """Durable model request settings stored in agent YAML/JSON config."""

    temperature: _NonNegativeFloat | None = None
    max_output_tokens: _PositiveInt | None = None
    reasoning: ReasoningSettings | None = None
    extra_settings: JsonObject = Field(default_factory=_empty_json_object)


class AgentModelConfig(StrictModel):
    """Durable model identity and settings for an agent."""

    provider: str = Field(min_length=1)
    api: ProviderApi
    model_name: str = Field(min_length=1)
    endpoint: str | None = Field(default=None, min_length=1)
    settings: ModelSettingsConfig = Field(default_factory=ModelSettingsConfig)
    profile: ModelProfile | None = None

    @field_validator("endpoint")
    @classmethod
    def _validate_endpoint(cls, endpoint: str | None) -> str | None:
        if endpoint is None:
            return None
        _validate_http_url_without_embedded_secret(endpoint, field_name="model endpoint")
        return endpoint

    @model_validator(mode="after")
    def _validate_profile_identity_and_capabilities(self) -> AgentModelConfig:
        profile = resolve_agent_model_profile(self)
        if profile is not None:
            if profile.provider_name != self.provider:
                raise ValueError("model profile provider_name must match configured provider")
            if profile.model_name != self.model_name:
                raise ValueError("model profile model_name must match configured model_name")
            if profile.api is not self.api:
                raise ValueError("model profile api must match configured api")
            validate_capability_requirements(profile, _capability_requirements_for_model(self))
        return self


class RetryConfig(StrictModel):
    """Provider-neutral retry/backoff configuration."""

    enabled: bool = True
    max_attempts: _PositiveInt = 10
    initial_delay_seconds: _NonNegativeFloat = 1.0
    max_delay_seconds: _NonNegativeFloat = 60.0
    multiplier: _NonNegativeFloat = 2.0
    jitter: bool = True
    respect_retry_after: bool = True
    max_retry_after_seconds: _NonNegativeFloat | None = 300.0

    @model_validator(mode="after")
    def _validate_retry_delays(self) -> RetryConfig:
        if self.multiplier < 1.0:
            raise ValueError("retry multiplier must be at least 1.0")
        if self.max_delay_seconds < self.initial_delay_seconds:
            raise ValueError("max retry delay must be >= initial retry delay")
        return self


class RetryConfigOverrides(StrictModel):
    """Partial retry overrides."""

    enabled: bool | None = None
    max_attempts: _PositiveInt | None = None
    initial_delay_seconds: _NonNegativeFloat | None = None
    max_delay_seconds: _NonNegativeFloat | None = None
    multiplier: _NonNegativeFloat | None = None
    jitter: bool | None = None
    respect_retry_after: bool | None = None
    max_retry_after_seconds: _NonNegativeFloat | None = None


class RedactionConfig(StrictModel):
    """Configurable redaction posture for provider diagnostics and local logs."""

    redact_secrets: bool = True
    redact_allowed_environment: bool = True
    redact_mildly_sensitive_urls: bool = True
    patterns: list[str] = Field(default_factory=_empty_strings)

    @field_validator("patterns")
    @classmethod
    def _validate_patterns(cls, patterns: list[str]) -> list[str]:
        _validate_unique_strings(patterns, field_name="redaction patterns")
        for pattern in patterns:
            compile(pattern)
        return patterns


class RedactionConfigOverrides(StrictModel):
    """Partial redaction overrides."""

    redact_secrets: bool | None = None
    redact_allowed_environment: bool | None = None
    redact_mildly_sensitive_urls: bool | None = None
    patterns: list[str] | None = None


class ApiKeySourcesConfig(StrictModel):
    """Environment-variable names used as provider secret/base-url sources."""

    openai_api_key_env: str = Field(
        default="OPENAI_API_KEY", min_length=1, pattern=ENV_NAME_PATTERN.pattern
    )
    openai_base_url_env: str = Field(
        default="OPENAI_BASE_URL", min_length=1, pattern=ENV_NAME_PATTERN.pattern
    )
    anthropic_api_key_env: str = Field(
        default="ANTHROPIC_API_KEY", min_length=1, pattern=ENV_NAME_PATTERN.pattern
    )
    anthropic_base_url_env: str = Field(
        default="ANTHROPIC_BASE_URL", min_length=1, pattern=ENV_NAME_PATTERN.pattern
    )

    def names(self) -> tuple[str, ...]:
        """Return configured source names in deterministic order."""

        return (
            self.openai_api_key_env,
            self.openai_base_url_env,
            self.anthropic_api_key_env,
            self.anthropic_base_url_env,
        )


class ApiKeySourcesConfigOverrides(StrictModel):
    """Partial API-key source overrides."""

    openai_api_key_env: str | None = Field(
        default=None, min_length=1, pattern=ENV_NAME_PATTERN.pattern
    )
    openai_base_url_env: str | None = Field(
        default=None, min_length=1, pattern=ENV_NAME_PATTERN.pattern
    )
    anthropic_api_key_env: str | None = Field(
        default=None, min_length=1, pattern=ENV_NAME_PATTERN.pattern
    )
    anthropic_base_url_env: str | None = Field(
        default=None, min_length=1, pattern=ENV_NAME_PATTERN.pattern
    )


class ModelRequestOverridesConfig(StrictModel):
    """Runtime provider request overrides."""

    base_url: str | None = Field(default=None, min_length=1)
    timeout_seconds: _NonNegativeFloat = 60.0
    extra_headers: list[ProviderHeaderConfig] = Field(default_factory=_empty_provider_headers)
    allow_literal_secret_headers: bool = False
    enable_provider_side_continuation: bool = False
    extra_request_settings: JsonObject = Field(default_factory=_empty_json_object)

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, base_url: str | None) -> str | None:
        if base_url is None:
            return None
        _validate_http_url_without_embedded_secret(base_url, field_name="runtime base_url")
        return base_url

    @field_validator("extra_headers")
    @classmethod
    def _validate_unique_header_names(
        cls, headers: list[ProviderHeaderConfig]
    ) -> list[ProviderHeaderConfig]:
        names = [header.name.lower() for header in headers]
        _validate_unique_strings(names, field_name="provider header names")
        return headers


class ModelRequestOverridesPatch(StrictModel):
    """Partial runtime provider request overrides."""

    base_url: str | None = Field(default=None, min_length=1)
    timeout_seconds: _NonNegativeFloat | None = None
    extra_headers: list[ProviderHeaderConfig] | None = None
    allow_literal_secret_headers: bool | None = None
    enable_provider_side_continuation: bool | None = None
    extra_request_settings: JsonObject | None = None


class ProviderRuntimeConfig(StrictModel):
    """Provider-facing subset of resolved runtime configuration."""

    model: ModelRequestOverridesConfig = Field(default_factory=ModelRequestOverridesConfig)
    api_key_sources: ApiKeySourcesConfig = Field(default_factory=ApiKeySourcesConfig)
    redaction: RedactionConfig = Field(default_factory=RedactionConfig)

    def secret_source_names(self) -> tuple[str, ...]:
        """Return env var names that should be redacted as secret sources."""

        names = set(self.api_key_sources.names())
        names.update(header.env_var for header in self.model.extra_headers if header.env_var)
        return tuple(sorted(names))


def provider_runtime_config(
    *,
    model: ModelRequestOverridesConfig | None = None,
    api_key_sources: ApiKeySourcesConfig | None = None,
    redaction: RedactionConfig | None = None,
) -> ProviderRuntimeConfig:
    """Build the provider-facing runtime subset from resolved runtime pieces."""

    return ProviderRuntimeConfig(
        model=ModelRequestOverridesConfig() if model is None else model.model_copy(deep=True),
        api_key_sources=(
            ApiKeySourcesConfig()
            if api_key_sources is None
            else api_key_sources.model_copy(deep=True)
        ),
        redaction=RedactionConfig() if redaction is None else redaction.model_copy(deep=True),
    )


def resolve_agent_model_profile(model: AgentModelConfig) -> ModelProfile | None:
    """Return the explicit or built-in model profile for an agent model."""

    if model.profile is not None:
        return model.profile
    return get_builtin_profile(model.api, model.provider, model.model_name)


def _capability_requirements_for_model(model: AgentModelConfig) -> CapabilityRequirements:
    reasoning = model.settings.reasoning
    return CapabilityRequirements(
        require_tool_calling=True,
        require_reasoning=reasoning is not None,
        reasoning=reasoning,
        temperature=model.settings.temperature,
        max_output_tokens=model.settings.max_output_tokens,
    )


def _validate_http_url_without_embedded_secret(value: str, *, field_name: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{field_name} must not include credentials, query, or fragment")


def _validate_unique_strings(values: list[str], *, field_name: str) -> None:
    if any(not value for value in values):
        raise ValueError(f"{field_name} must not contain empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must be unique")


__all__ = (
    "AgentModelConfig",
    "ApiKeySourcesConfig",
    "ApiKeySourcesConfigOverrides",
    "HeaderValueSource",
    "ModelRequestOverridesConfig",
    "ModelRequestOverridesPatch",
    "ModelSettingsConfig",
    "ProviderHeaderConfig",
    "ProviderRuntimeConfig",
    "RedactionConfig",
    "RedactionConfigOverrides",
    "RetryConfig",
    "RetryConfigOverrides",
    "provider_runtime_config",
    "resolve_agent_model_profile",
)
