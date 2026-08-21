"""Agent and runtime configuration models plus deterministic resolution."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Annotated, cast

from pydantic import Field, ValidationInfo, field_validator, model_validator

from tend._common.env import validate_env_name
from tend._common.types import JsonObject, StrictModel
from tend.agent.outputs import AgentOutputConfig
from tend.agent.tool_names import BUILTIN_TOOL_NAMES, validate_builtin_tool_names
from tend.llm.config import (
    AgentModelConfig,
    ApiKeySourcesConfig,
    ApiKeySourcesConfigOverrides,
    HeaderValueSource,
    ModelRequestOverridesConfig,
    ModelRequestOverridesPatch,
    ModelSettingsConfig,
    ProviderHeaderConfig,
    ProviderRuntimeConfig,
    RedactionConfig,
    RedactionConfigOverrides,
    RetryConfig,
    RetryConfigOverrides,
    provider_runtime_config,
    resolve_agent_model_profile,
)
from tend.llm.context_estimation import TokenEstimatorConfig, TokenEstimatorConfigOverrides
from tend.prompts import builtin_prompts_root, load_prompt

_PositiveInt = Annotated[int, Field(ge=1)]
_NonNegativeInt = Annotated[int, Field(ge=0)]
_NonNegativeFloat = Annotated[float, Field(ge=0)]
_NonNegativeDecimal = Annotated[Decimal, Field(ge=Decimal("0"))]

_SECRET_KEY_NAMES: frozenset[str] = frozenset(
    {
        "apikey",
        "authorization",
        "authtoken",
        "bearertoken",
        "clientsecret",
        "password",
        "passwd",
        "refreshtoken",
        "secret",
        "token",
    }
)

_DEFAULT_ALLOWED_ENV_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
)


def _empty_strings() -> list[str]:
    return []


def _default_allowed_env_vars() -> list[str]:
    return list(_DEFAULT_ALLOWED_ENV_VARS)


class AgentToolsConfig(StrictModel):
    """Enabled built-in tools and durable per-tool options."""

    enabled: list[str] = Field(default_factory=_empty_strings)
    options: dict[str, JsonObject] = Field(default_factory=dict)

    @field_validator("enabled")
    @classmethod
    def _validate_enabled_tools(cls, tool_names: list[str]) -> list[str]:
        _validate_unique_strings(tool_names, field_name="enabled tools")
        validate_builtin_tool_names(tool_names)
        return tool_names

    @field_validator("options")
    @classmethod
    def _validate_tool_options(cls, options: dict[str, JsonObject]) -> dict[str, JsonObject]:
        validate_builtin_tool_names(options.keys())
        return options

    @model_validator(mode="after")
    def _validate_option_targets(self) -> AgentToolsConfig:
        enabled = set(self.enabled)
        unknown_option_targets = sorted(set(self.options) - enabled)
        if unknown_option_targets:
            joined = ", ".join(unknown_option_targets)
            raise ValueError(f"tool options were provided for disabled tools: {joined}")
        return self


class RuntimeLimitsConfig(StrictModel):
    """Finite runtime limits for unattended turns."""

    max_iterations: _PositiveInt | None = None
    max_model_requests: _PositiveInt | None = None
    max_tool_calls: _NonNegativeInt | None = None
    max_wall_time_seconds: _NonNegativeFloat = 3600.0
    max_tokens: _NonNegativeInt | None = None
    max_cost: _NonNegativeDecimal | None = None


class RuntimeLimitsOverrides(StrictModel):
    """Partial runtime limit overrides from agent defaults, cfg, or CLI."""

    max_iterations: _PositiveInt | None = None
    max_model_requests: _PositiveInt | None = None
    max_tool_calls: _NonNegativeInt | None = None
    max_wall_time_seconds: _NonNegativeFloat | None = None
    max_tokens: _NonNegativeInt | None = None
    max_cost: _NonNegativeDecimal | None = None


class CompactionConfig(StrictModel):
    """Generic compaction trigger configuration."""

    enabled: bool = True
    threshold_tokens: _PositiveInt | None = None
    threshold_messages: _PositiveInt | None = None
    reserve_tokens: _NonNegativeInt = 4096
    keep_recent_tokens: _NonNegativeInt = 16_000
    target_tokens: _PositiveInt = 4_000
    trigger_on_context_overflow: bool = True

    @model_validator(mode="after")
    def _validate_compaction_budget(self) -> CompactionConfig:
        if self.keep_recent_tokens < self.target_tokens:
            raise ValueError("keep_recent_tokens must be >= target_tokens")
        return self


class CompactionConfigOverrides(StrictModel):
    """Partial compaction overrides."""

    enabled: bool | None = None
    threshold_tokens: _PositiveInt | None = None
    threshold_messages: _PositiveInt | None = None
    reserve_tokens: _NonNegativeInt | None = None
    keep_recent_tokens: _NonNegativeInt | None = None
    target_tokens: _PositiveInt | None = None
    trigger_on_context_overflow: bool | None = None


class LoggingConfig(StrictModel):
    """Detailed local logging controls."""

    detailed: bool = False
    include_model_payloads: bool = False
    include_tool_outputs: bool = False
    include_prompts: bool = False
    warn_on_full_payload_logging: bool = True

    @model_validator(mode="after")
    def _validate_detailed_payload_flags(self) -> LoggingConfig:
        if not self.detailed:
            enabled_payload_flags = [
                self.include_model_payloads,
                self.include_tool_outputs,
                self.include_prompts,
            ]
            if any(enabled_payload_flags):
                raise ValueError("payload logging flags require detailed logging")
        return self


class LoggingConfigOverrides(StrictModel):
    """Partial detailed logging overrides."""

    detailed: bool | None = None
    include_model_payloads: bool | None = None
    include_tool_outputs: bool | None = None
    include_prompts: bool | None = None
    warn_on_full_payload_logging: bool | None = None


class ArtifactConfig(StrictModel):
    """Artifact storage controls for large payloads."""

    enabled: bool = True
    inline_threshold_bytes: _NonNegativeInt = 32_768
    directory_name: str = Field(default="artifacts", min_length=1)

    @field_validator("directory_name")
    @classmethod
    def _validate_directory_name(cls, directory_name: str) -> str:
        if "/" in directory_name or "\\" in directory_name or directory_name in {".", ".."}:
            raise ValueError("artifact directory name must be a single relative path segment")
        return directory_name


class ArtifactConfigOverrides(StrictModel):
    """Partial artifact overrides."""

    enabled: bool | None = None
    inline_threshold_bytes: _NonNegativeInt | None = None
    directory_name: str | None = Field(default=None, min_length=1)


class UsageConfig(StrictModel):
    """Context-estimation tracking switches.

    Token usage and cost are always recorded from provider responses; the only
    remaining switch is whether to estimate active context tokens per request.
    """

    estimate_context_tokens: bool = True
    token_estimator: TokenEstimatorConfig = Field(default_factory=TokenEstimatorConfig)


class UsageConfigOverrides(StrictModel):
    """Partial context-estimation overrides."""

    estimate_context_tokens: bool | None = None
    token_estimator: TokenEstimatorConfigOverrides | None = None


class EnvironmentConfig(StrictModel):
    """Environment names that may be supplied to provider adapters."""

    allowed_env_vars: list[str] = Field(default_factory=_default_allowed_env_vars)
    safe_to_log_env_vars: list[str] = Field(default_factory=_empty_strings)

    @field_validator("allowed_env_vars", "safe_to_log_env_vars")
    @classmethod
    def _validate_env_vars(cls, names: list[str]) -> list[str]:
        _validate_unique_strings(names, field_name="environment variables")
        for name in names:
            _validate_env_name(name)
        return names

    @model_validator(mode="after")
    def _validate_safe_env_subset(self) -> EnvironmentConfig:
        unknown_safe_names = sorted(set(self.safe_to_log_env_vars) - set(self.allowed_env_vars))
        if unknown_safe_names:
            joined = ", ".join(unknown_safe_names)
            raise ValueError(f"safe-to-log environment variables are not allowed: {joined}")
        return self


class EnvironmentConfigOverrides(StrictModel):
    """Partial environment allowlist overrides."""

    allowed_env_vars: list[str] | None = None
    safe_to_log_env_vars: list[str] | None = None


class RuntimeConfig(StrictModel):
    """Fully resolved runtime configuration used by library/CLI execution."""

    prompt: str | None = Field(default=None, min_length=1)
    cwd: str = "."
    session_dir: str | None = Field(default=None, min_length=1)
    limits: RuntimeLimitsConfig = Field(default_factory=RuntimeLimitsConfig)
    retries: RetryConfig = Field(default_factory=RetryConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    redaction: RedactionConfig = Field(default_factory=RedactionConfig)
    usage: UsageConfig = Field(default_factory=UsageConfig)
    model: ModelRequestOverridesConfig = Field(default_factory=ModelRequestOverridesConfig)
    api_key_sources: ApiKeySourcesConfig = Field(default_factory=ApiKeySourcesConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)

    @field_validator("cwd", "session_dir")
    @classmethod
    def _validate_paths(cls, path: str | None) -> str | None:
        if path is None:
            return None
        _validate_non_empty_path(path, field_name="path")
        return path

    def secret_source_names(self) -> tuple[str, ...]:
        """Return env var names that should be redacted as secret sources."""

        names = set(self.api_key_sources.names())
        names.update(header.env_var for header in self.model.extra_headers if header.env_var)
        return tuple(sorted(names))

    def allowed_environment_names(self) -> tuple[str, ...]:
        """Return env var names explicitly allowed or needed by secret sources."""

        names = set(self.environment.allowed_env_vars)
        names.update(self.api_key_sources.names())
        names.update(header.env_var for header in self.model.extra_headers if header.env_var)
        return tuple(sorted(names))

    def to_provider_runtime_config(self) -> ProviderRuntimeConfig:
        """Return the LLM-provider-facing subset of this runtime config."""

        return provider_runtime_config(
            model=self.model,
            api_key_sources=self.api_key_sources,
            redaction=self.redaction,
        )


class RuntimeConfigOverrides(StrictModel):
    """Partial runtime configuration from ``cfg.yaml``/``cfg.json`` or CLI flags.

    Every field is optional and resolution applies only fields that are actually
    present, allowing deterministic nested precedence without reading the
    environment.
    """

    prompt: str | None = Field(default=None, min_length=1)
    cwd: str | None = None
    session_dir: str | None = Field(default=None, min_length=1)
    limits: RuntimeLimitsOverrides | None = None
    retries: RetryConfigOverrides | None = None
    compaction: CompactionConfigOverrides | None = None
    logging: LoggingConfigOverrides | None = None
    artifacts: ArtifactConfigOverrides | None = None
    redaction: RedactionConfigOverrides | None = None
    usage: UsageConfigOverrides | None = None
    model: ModelRequestOverridesPatch | None = None
    api_key_sources: ApiKeySourcesConfigOverrides | None = None
    environment: EnvironmentConfigOverrides | None = None

    @field_validator("cwd", "session_dir")
    @classmethod
    def _validate_paths(cls, path: str | None) -> str | None:
        if path is None:
            return None
        _validate_non_empty_path(path, field_name="path")
        return path


class AgentConfigSystemPromptRegistry(StrictModel):
    """Pointer to a system prompt under the bundled prompt registry.

    The pointer is a ``<role>/<version>`` slug (e.g. ``reviewer/minimal``) that
    resolves against :func:`tend.prompts.builtin_prompts_root` at config
    load time. The launch-time code snapshot pins the Tend checkout into
    ``<run>/code/``, so the registry travels with the run and the
    pointer stays reproducible without inlining the prompt text into the
    agent yaml.
    """

    registry: str = Field(
        min_length=1,
        description="<role>/<version>, e.g. 'reviewer/minimal' or 'worker/minimal'",
    )


class AgentConfigSystemPromptPath(StrictModel):
    """Pointer to a system prompt markdown file.

    Relative paths resolve against the directory containing the durable
    ``agent.yaml``/``agent.json`` file when loaded through
    :func:`tend._common.config_files.read_config_model` (the path used by the
    standalone ``tend-agent`` CLI). Absolute paths are loaded directly.
    """

    path: str = Field(
        min_length=1,
        description="Path to a system prompt markdown file, relative to agent config.",
    )

    @field_validator("path")
    @classmethod
    def _validate_path(cls, path: str) -> str:
        _validate_non_empty_path(path, field_name="system prompt path")
        return path


def _resolve_registry_prompt(registry: str) -> str:
    """Load ``<builtin_prompts_root>/<role>/<version>/system.md``.

    ``registry`` is a relative ``<role>/<version>`` slug. The resolver only
    consults the bundled registry under :func:`builtin_prompts_root`; for
    custom non-bundled prompts, use the literal-string or ``{path: ...}`` form
    instead.
    """

    relative = Path(registry)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            f"prompt registry pointer must be a relative <role>/<version> slug, got: {registry!r}"
        )
    bundled_root = builtin_prompts_root()
    prompt_dir = (bundled_root / relative).resolve()
    # Defence in depth against prompt-injection via a symlink under the bundled
    # package that resolves to content outside ``builtin_prompts_root``. The
    # lexical check above blocks crafted slugs; this guards the on-disk layer.
    if not prompt_dir.is_relative_to(bundled_root):
        raise ValueError(
            f"prompt registry pointer escapes bundled root: {registry!r} → {prompt_dir}"
        )
    if not prompt_dir.is_dir():
        raise ValueError(f"prompt registry path not found: {prompt_dir}")
    return load_prompt(prompt_dir, "system")


def _resolve_path_prompt(path: str, *, config_root: Path | None) -> str:
    """Load a system prompt markdown file from ``path``.

    Relative paths require the ``agent.yaml`` directory supplied through
    Pydantic validation context by ``read_config_model``. Trailing whitespace is
    stripped to match bundled registry prompt loading and shell command
    substitution used by generated agent launchers.
    """

    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        prompt_path = raw_path.resolve()
    else:
        if config_root is None:
            raise ValueError(
                "relative system prompt path requires an agent config file context"
            )
        prompt_path = (config_root / raw_path).resolve()
    if not prompt_path.is_file():
        raise ValueError(f"system prompt path not found: {prompt_path}")
    try:
        return prompt_path.read_text(encoding="utf-8").rstrip()
    except OSError as exc:
        raise ValueError(
            f"could not read system prompt path {prompt_path}: {exc.strerror or exc}"
        ) from exc


def _config_root_from_validation_context(info: ValidationInfo) -> Path | None:
    context = info.context
    if not isinstance(context, Mapping):
        return None
    root = cast(Mapping[str, object], context).get("config_root")
    if isinstance(root, Path):
        return root
    if isinstance(root, str):
        return Path(root)
    return None


class AgentConfig(StrictModel):
    """Durable ``agent.yaml``/``agent.json`` configuration.

    Secrets are intentionally not represented here. Provider credentials,
    request headers, and environment source names belong to runtime config.

    ``system_prompt`` accepts a literal string (back-compat), a
    ``{registry: "<role>/<version>"}`` mapping that resolves against the
    bundled prompt registry, or a ``{path: "..."}`` mapping that resolves a
    markdown file relative to the agent config at load time. The runtime
    attribute is always a plain string.
    """

    schema_version: str = Field(default="1", min_length=1)
    system_prompt: str = Field(min_length=1)
    model: AgentModelConfig
    tools: AgentToolsConfig = Field(default_factory=AgentToolsConfig)
    output: AgentOutputConfig | None = None
    runtime_defaults: RuntimeConfigOverrides = Field(default_factory=RuntimeConfigOverrides)

    @field_validator("system_prompt", mode="before")
    @classmethod
    def _resolve_system_prompt_source(cls, value: object, info: ValidationInfo) -> object:
        """Resolve system prompt source mappings to prompt text.

        Literal strings pass through unchanged for back-compat with existing
        inlined yamls. ``{registry: "<role>/<version>"}`` loads bundled
        registry text; ``{path: "..."}`` loads a markdown file, resolving
        relative paths against the directory containing the agent config when
        that context is available.
        """

        if isinstance(value, str):
            return value
        if isinstance(value, AgentConfigSystemPromptRegistry):
            return _resolve_registry_prompt(value.registry)
        if isinstance(value, AgentConfigSystemPromptPath):
            return _resolve_path_prompt(
                value.path,
                config_root=_config_root_from_validation_context(info),
            )
        if isinstance(value, dict):
            if "registry" in value:
                pointer = AgentConfigSystemPromptRegistry.model_validate(value)
                return _resolve_registry_prompt(pointer.registry)
            if "path" in value:
                pointer = AgentConfigSystemPromptPath.model_validate(value)
                return _resolve_path_prompt(
                    pointer.path,
                    config_root=_config_root_from_validation_context(info),
                )
            raise ValueError("system_prompt mapping must contain 'registry' or 'path'")
        return value

    @model_validator(mode="after")
    def _validate_no_durable_secrets(self) -> AgentConfig:
        _reject_secret_like_keys(
            self.model_dump(mode="python", exclude={"runtime_defaults"}),
            path="agent",
        )
        _validate_agent_runtime_defaults(self.runtime_defaults)
        return self


class ResolvedConfig(StrictModel):
    """Resolved durable agent plus runtime configuration."""

    agent: AgentConfig
    runtime: RuntimeConfig


def resolve_runtime_config(
    *,
    agent_config: AgentConfig | None = None,
    cfg: RuntimeConfigOverrides | None = None,
    cli_overrides: RuntimeConfigOverrides | None = None,
) -> RuntimeConfig:
    """Resolve runtime config with CLI > cfg.yaml/cfg.json > agent defaults > library defaults."""

    merged = _model_to_object_map(RuntimeConfig())
    if agent_config is not None:
        merged = _deep_merge(merged, _patch_to_object_map(agent_config.runtime_defaults))
    if cfg is not None:
        merged = _deep_merge(merged, _patch_to_object_map(cfg))
    if cli_overrides is not None:
        merged = _deep_merge(merged, _patch_to_object_map(cli_overrides))
    return RuntimeConfig.model_validate(merged)


def resolve_config(
    agent_config: AgentConfig,
    *,
    cfg: RuntimeConfigOverrides | None = None,
    cli_overrides: RuntimeConfigOverrides | None = None,
) -> ResolvedConfig:
    """Resolve durable and runtime configuration together."""

    return ResolvedConfig(
        agent=agent_config,
        runtime=resolve_runtime_config(
            agent_config=agent_config,
            cfg=cfg,
            cli_overrides=cli_overrides,
        ),
    )


def _validate_agent_runtime_defaults(defaults: RuntimeConfigOverrides) -> None:
    if "prompt" in defaults.model_fields_set:
        raise ValueError("agent runtime_defaults must not contain an invocation prompt")
    if defaults.model is not None:
        has_provider_request_overrides = any(
            field_name in defaults.model.model_fields_set
            for field_name in ("extra_headers", "extra_request_settings")
        )
        if has_provider_request_overrides:
            raise ValueError(
                "agent runtime_defaults must not contain provider headers or request payloads"
            )
    if defaults.api_key_sources is not None:
        raise ValueError("agent runtime_defaults must not contain API-key sources")
    if defaults.environment is not None:
        raise ValueError("agent runtime_defaults must not contain environment allowlists")


def _validate_env_name(name: str) -> None:
    validate_env_name(name)


def _validate_non_empty_path(path: str, *, field_name: str) -> None:
    if not path or "\x00" in path:
        raise ValueError(f"{field_name} must be non-empty and must not contain NUL")


def _validate_unique_strings(values: list[str], *, field_name: str) -> None:
    if any(not value for value in values):
        raise ValueError(f"{field_name} must not contain empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must be unique")


def _reject_secret_like_keys(value: object, *, path: str) -> None:
    if isinstance(value, dict):
        for key, item in cast(dict[object, object], value).items():
            key_text = str(key)
            normalized = key_text.replace("_", "").replace("-", "").lower()
            child_path = f"{path}.{key_text}"
            if normalized in _SECRET_KEY_NAMES:
                raise ValueError(
                    f"durable agent config must not contain secret-like key {child_path}"
                )
            _reject_secret_like_keys(item, path=child_path)
    elif isinstance(value, list):
        for index, item in enumerate(cast(list[object], value)):
            _reject_secret_like_keys(item, path=f"{path}[{index}]")


type _ObjectMap = dict[str, object]


def _model_to_object_map(model: StrictModel) -> _ObjectMap:
    return cast(_ObjectMap, model.model_dump(mode="python"))


def _patch_to_object_map(model: StrictModel) -> _ObjectMap:
    return cast(_ObjectMap, model.model_dump(mode="python", exclude_unset=True))


def _deep_merge(left: _ObjectMap, right: _ObjectMap) -> _ObjectMap:
    result: _ObjectMap = dict(left)
    for key, value in right.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(cast(_ObjectMap, existing), cast(_ObjectMap, value))
        else:
            result[key] = value
    return result


__all__ = (
    "AgentConfig",
    "AgentConfigSystemPromptPath",
    "AgentConfigSystemPromptRegistry",
    "AgentModelConfig",
    "AgentOutputConfig",
    "AgentToolsConfig",
    "ApiKeySourcesConfig",
    "ApiKeySourcesConfigOverrides",
    "ArtifactConfig",
    "ArtifactConfigOverrides",
    "BUILTIN_TOOL_NAMES",
    "CompactionConfig",
    "CompactionConfigOverrides",
    "EnvironmentConfig",
    "EnvironmentConfigOverrides",
    "HeaderValueSource",
    "LoggingConfig",
    "LoggingConfigOverrides",
    "ModelRequestOverridesConfig",
    "ModelRequestOverridesPatch",
    "ModelSettingsConfig",
    "ProviderHeaderConfig",
    "RedactionConfig",
    "RedactionConfigOverrides",
    "ResolvedConfig",
    "RetryConfig",
    "RetryConfigOverrides",
    "RuntimeConfig",
    "RuntimeConfigOverrides",
    "RuntimeLimitsConfig",
    "RuntimeLimitsOverrides",
    "TokenEstimatorConfig",
    "TokenEstimatorConfigOverrides",
    "UsageConfig",
    "UsageConfigOverrides",
    "resolve_agent_model_profile",
    "resolve_config",
    "resolve_runtime_config",
    "validate_builtin_tool_names",
)
