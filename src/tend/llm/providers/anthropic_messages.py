"""Native Anthropic Messages API adapter translation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

from pydantic import JsonValue, SecretStr, TypeAdapter, ValidationError

from tend._common.errors import ConfigurationError, ProviderProtocolError
from tend._common.types import JsonObject, StopReason, new_id
from tend.llm.config import (
    AgentModelConfig,
    ProviderRuntimeConfig,
    resolve_agent_model_profile,
)
from tend.llm.history import ASSISTANT_REASONING_METADATA_KEY, assistant_tool_calls
from tend.llm.models.messages import (
    AssistantMessage,
    CompactionSummaryContent,
    DeveloperMessage,
    SystemMessage,
    TextContent,
    UserMessage,
)
from tend.llm.models.profiles import (
    CapabilityRequirements,
    ModelProfile,
    ProviderApi,
    ReasoningEffort,
    resolve_max_output_tokens,
    validate_capability_requirements,
)
from tend.llm.models.provider import (
    ContinuationStrategy,
    ProviderCompletionStatus,
    ProviderItemKind,
    ProviderItemMetadata,
    ProviderMetadata,
)
from tend.llm.models.reasoning import (
    ReasoningContinuationMetadata,
    ReasoningMetadata,
    ReasoningSettings,
)
from tend.llm.models.requests import ModelMessage, ModelRequest, ModelResponse
from tend.llm.models.tools import ToolCall, ToolResultMessage, model_visible_tool_result_text
from tend.llm.providers.errors import DEFAULT_SECRET_HEADER_NAMES
from tend.llm.providers.http import HttpxJsonTransport, JsonPostRequest, JsonPostTransport
from tend.llm.redaction import Redactor, redact_headers
from tend.llm.secrets import (
    EnvironmentSecretSource,
    ProviderHeaderConfig,
    SecretSourceKind,
    SecretValue,
    resolve_provider_headers,
)
from tend.llm.usage import TokenUsage, Usage

_DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
_DEFAULT_MESSAGES_PATH = "messages"
_DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_THINKING_BUDGETS: Mapping[ReasoningEffort, int] = {
    ReasoningEffort.MINIMAL: 1024,
    ReasoningEffort.LOW: 1024,
    ReasoningEffort.MEDIUM: 4096,
    ReasoningEffort.HIGH: 8192,
    ReasoningEffort.XHIGH: 16384,
    ReasoningEffort.MAX: 32768,
}
_AUTH_HEADER_NAMES: frozenset[str] = frozenset(
    {"anthropic-api-key", "authorization", "cf-aig-authorization", "x-api-key"}
)
_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)
_JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


class AnthropicMessagesAdapter:
    """Provider adapter for native Anthropic Messages requests and responses.

    ``build_payload`` and ``build_http_request`` are stable testable seams for
    request-shape tests. ``generate`` posts the translated request and parses the
    Messages payload back into provider-neutral schemas.
    """

    __slots__ = (
        "_anthropic_version",
        "_api_key",
        "_api_key_env_var",
        "_api_key_header_name",
        "_api_key_scheme",
        "_base_url",
        "_default_max_output_tokens",
        "_default_request_settings",
        "_default_reasoning",
        "_environment",
        "_extra_headers",
        "_model_name",
        "_profile",
        "_provider_name",
        "_raw_headers",
        "_redactor",
        "_timeout_seconds",
        "_transport",
    )

    _model_name: str
    _provider_name: str
    _profile: ModelProfile | None
    _base_url: str
    _anthropic_version: str
    _transport: JsonPostTransport
    _timeout_seconds: float | None
    _environment: Mapping[str, str]
    _api_key_env_var: str | None
    _api_key: SecretValue | None
    _api_key_header_name: str
    _api_key_scheme: str | None
    _extra_headers: tuple[ProviderHeaderConfig, ...]
    _raw_headers: dict[str, str]
    _redactor: Redactor | None
    _default_reasoning: ReasoningSettings | None
    _default_request_settings: JsonObject
    _default_max_output_tokens: int | None

    def __init__(
        self,
        *,
        model_name: str,
        provider_name: str = "anthropic",
        base_url: str = _DEFAULT_ANTHROPIC_BASE_URL,
        anthropic_version: str = _DEFAULT_ANTHROPIC_VERSION,
        profile: ModelProfile | None = None,
        transport: JsonPostTransport | None = None,
        timeout_seconds: float | None = 60.0,
        environment: Mapping[str, str] | None = None,
        api_key_env_var: str | None = None,
        api_key: SecretValue | str | None = None,
        api_key_header_name: str = "x-api-key",
        api_key_scheme: str | None = None,
        extra_headers: Iterable[ProviderHeaderConfig] = (),
        raw_headers: Mapping[str, str] | None = None,
        allow_literal_secret_headers: bool = False,
        redactor: Redactor | None = None,
        default_reasoning: ReasoningSettings | None = None,
        default_request_settings: Mapping[str, JsonValue] | None = None,
        default_max_output_tokens: int | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must be non-empty")
        if not provider_name:
            raise ValueError("provider_name must be non-empty")
        if not anthropic_version:
            raise ValueError("anthropic_version must be non-empty")
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative when provided")
        if default_max_output_tokens is not None and default_max_output_tokens < 1:
            raise ValueError("default_max_output_tokens must be positive when provided")
        if profile is not None:
            _validate_anthropic_profile(profile, model_name=model_name, provider_name=provider_name)

        env = {} if environment is None else dict(environment)
        header_configs = tuple(extra_headers)
        self._model_name = model_name
        self._provider_name = provider_name
        self._profile = profile.model_copy(deep=True) if profile is not None else None
        self._base_url = _validate_base_url(base_url)
        self._anthropic_version = anthropic_version
        self._timeout_seconds = timeout_seconds
        self._environment = env
        self._api_key_env_var = api_key_env_var
        self._api_key = _secret_value_from_api_key(api_key)
        self._api_key_header_name = api_key_header_name
        self._api_key_scheme = api_key_scheme
        self._extra_headers = header_configs
        self._raw_headers = {str(name): str(value) for name, value in (raw_headers or {}).items()}
        self._redactor = redactor
        self._default_reasoning = (
            default_reasoning.model_copy(deep=True) if default_reasoning is not None else None
        )
        self._default_request_settings = _json_object(default_request_settings or {})
        self._default_max_output_tokens = default_max_output_tokens

        resolved_headers = resolve_provider_headers(
            header_configs,
            env,
            allow_literal_secrets=allow_literal_secret_headers,
        )
        self._raw_headers.update(resolved_headers.as_dict())
        self._transport = transport or HttpxJsonTransport(
            redactor=redactor,
            secret_header_names=self.secret_header_names,
        )

    @classmethod
    def from_config(
        cls,
        model_config: AgentModelConfig,
        runtime_config: ProviderRuntimeConfig,
        *,
        environment: Mapping[str, str],
        transport: JsonPostTransport | None = None,
    ) -> AnthropicMessagesAdapter:
        """Build an adapter from durable/runtime config and explicit environment."""

        if model_config.api is not ProviderApi.ANTHROPIC_MESSAGES:
            raise ConfigurationError("Anthropic Messages adapter requires anthropic_messages API")

        profile = resolve_agent_model_profile(model_config)
        base_url = _base_url_from_config(model_config, runtime_config, environment)
        redactor = _redactor_from_runtime_config(runtime_config, base_url=base_url)
        extra_headers = tuple(runtime_config.model.extra_headers)
        api_key_env_var = None
        if not _has_auth_header_configs(extra_headers):
            api_key_env_var = runtime_config.api_key_sources.anthropic_api_key_env

        settings = _merge_json_objects(
            model_config.settings.extra_settings,
            runtime_config.model.extra_request_settings,
        )
        if model_config.settings.temperature is not None and "temperature" not in settings:
            settings["temperature"] = model_config.settings.temperature

        return cls(
            model_name=model_config.model_name,
            provider_name=model_config.provider,
            base_url=base_url,
            profile=profile,
            transport=transport,
            timeout_seconds=runtime_config.model.timeout_seconds,
            environment=environment,
            api_key_env_var=api_key_env_var,
            extra_headers=extra_headers,
            allow_literal_secret_headers=runtime_config.model.allow_literal_secret_headers,
            redactor=redactor,
            default_reasoning=model_config.settings.reasoning,
            default_request_settings=settings,
            default_max_output_tokens=model_config.settings.max_output_tokens,
        )

    @property
    def profile(self) -> ModelProfile | None:
        """Return a defensive copy of profile/capability metadata, if configured."""

        if self._profile is None:
            return None
        return self._profile.model_copy(deep=True)

    @property
    def model_name(self) -> str:
        """Return the configured provider model name."""

        return self._model_name

    @property
    def endpoint_url(self) -> str:
        """Return the concrete Anthropic Messages endpoint URL."""

        return _messages_endpoint_url(self._base_url)

    @property
    def secret_header_names(self) -> tuple[str, ...]:
        """Return request header names whose values must be redacted."""

        configured = [header.name for header in self._extra_headers if header.is_sensitive]
        configured.append(self._api_key_header_name)
        for name in DEFAULT_SECRET_HEADER_NAMES:
            configured.append(name)
        return tuple(sorted({name.lower() for name in configured if name}))

    def build_payload(self, request: ModelRequest) -> JsonObject:
        """Translate a provider-neutral request into an Anthropic Messages body."""

        model_name = request.model_name or self._model_name
        if not model_name:
            raise ConfigurationError("Anthropic Messages requests require a model name")

        reasoning = self._resolve_reasoning(request)
        max_tokens = self._resolve_max_tokens(request)
        settings = self._request_settings(request)
        temperature = _pop_optional_float_setting(settings, "temperature")
        requested_tool_choice = _pop_tool_choice_request(settings, request.request_metadata)
        thinking_block = self._thinking_payload(reasoning, max_tokens=max_tokens)
        thinking = thinking_block.thinking if thinking_block is not None else None
        output_config = thinking_block.output_config if thinking_block is not None else None
        tool_choice = _tool_choice_payload(
            requested_tool_choice,
            thinking_enabled=thinking is not None,
            profile=self._profile,
        )
        self._validate_request_capabilities(
            request,
            reasoning=reasoning,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            force_tool_choice=_is_forced_tool_choice(tool_choice),
        )

        messages = _messages_from_model_messages(request.messages)
        body: dict[str, object] = {
            "model": model_name,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        system = _system_text_from_messages(request.messages)
        if system:
            body["system"] = system
        body["cache_control"] = {"type": "ephemeral"}
        if thinking is not None:
            body["thinking"] = thinking
        if output_config is not None:
            body["output_config"] = output_config
        if request.tools:
            body["tools"] = [_tool_payload(tool) for tool in request.tools]
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if temperature is not None:
            body["temperature"] = temperature

        _apply_extra_settings(body, settings, self._profile)
        return _json_object(body)

    def build_headers(self) -> dict[str, str]:
        """Return raw request headers for immediate HTTP use."""

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": self._anthropic_version,
        }
        api_key = self._resolve_api_key()
        if api_key is not None:
            headers[self._api_key_header_name] = _format_secret_header_value(
                api_key.reveal_value(),
                scheme=self._api_key_scheme,
            )
        headers.update(self._raw_headers)
        return headers

    def redacted_headers(self, headers: Mapping[str, str]) -> dict[str, str]:
        """Return diagnostic-safe headers using adapter redaction policy."""

        redacted = redact_headers(
            {str(name): str(value) for name, value in headers.items()},
            secret_header_names=self.secret_header_names,
        )
        if self._redactor is None:
            return redacted
        return self._redactor.redact_headers(redacted)

    def build_http_request(self, request: ModelRequest) -> JsonPostRequest:
        """Build the JSON POST request that would be sent to the provider."""

        return JsonPostRequest(
            url=self.endpoint_url,
            headers=self.build_headers(),
            body=self.build_payload(request),
            timeout_seconds=self._timeout_seconds,
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """POST a translated Messages request and parse the provider payload."""

        http_request = self.build_http_request(request)
        http_response = await self._transport.post_json(http_request)
        return self.parse_response(http_response.body, request=request)

    def parse_response(
        self,
        body: JsonValue,
        *,
        request: ModelRequest | None = None,
    ) -> ModelResponse:
        """Translate an Anthropic Messages payload into ``ModelResponse``.

        Thinking blocks and signatures are kept in reasoning/provider metadata
        for continuation; they are never exposed as normal assistant text by the
        parser.
        """

        payload = _response_payload(body)
        parsed = _parse_response_payload(
            payload,
            provider_name=self._provider_name,
            fallback_model_name=self._model_name,
            requested_reasoning=self._requested_reasoning_metadata(request),
            profile=self._profile,
        )
        if request is None:
            return parsed
        return parsed.model_copy(update={"request_id": request.request_id}, deep=True)

    def _resolve_reasoning(self, request: ModelRequest) -> ReasoningSettings | None:
        # An explicit per-request disable signal overrides adapter defaults. The
        # turn loop sets it on a forced re-ask so thinking is off and the forced
        # tool choice is not silently dropped (thinking is incompatible with
        # forced tool choice on some models).
        if _reasoning_disabled(request.request_metadata):
            return None
        if request.reasoning is not None:
            return request.reasoning.model_copy(deep=True)
        if self._default_reasoning is not None:
            return self._default_reasoning.model_copy(deep=True)
        return None

    def _requested_reasoning_metadata(
        self,
        request: ModelRequest | None,
    ) -> ReasoningSettings | None:
        if request is not None:
            return self._resolve_reasoning(request)
        if self._default_reasoning is not None:
            return self._default_reasoning.model_copy(deep=True)
        return None

    def _resolve_max_tokens(self, request: ModelRequest) -> int:
        requested = request.max_output_tokens or self._default_max_output_tokens
        if requested is None and self._profile is not None:
            requested = self._profile.default_output_tokens
        if requested is None:
            requested = _DEFAULT_MAX_TOKENS
        if self._profile is not None:
            resolved = resolve_max_output_tokens(self._profile, requested)
            if resolved is None:
                return requested
            return resolved
        return requested

    def _request_settings(self, request: ModelRequest) -> JsonObject:
        settings = _copy_json_object(self._default_request_settings)
        metadata_settings = _metadata_settings(request.request_metadata)
        settings.update(metadata_settings)
        return settings

    def _thinking_payload(
        self,
        reasoning: ReasoningSettings | None,
        *,
        max_tokens: int,
    ) -> _ThinkingPayload | None:
        if reasoning is None:
            return None

        native = _copy_json_object(reasoning.native_settings)
        native_thinking = _pop_native_thinking_settings(native)

        effort = _remapped_effort(reasoning.effort, self._profile)
        # An explicit "off" mapping in thinking_level_map disables thinking
        # entirely for this effort, matching pi's semantics.
        if reasoning.effort is not None and effort is None:
            return None

        if _profile_requires_adaptive_thinking(self._profile):
            # Adaptive thinking: pi-compatible request shape. ``display`` defaults
            # to "summarized" but can be overridden via either:
            #   reasoning.native_settings["thinking"]["display"]
            #   reasoning.native_settings["display"]
            # The thinking dict takes precedence over the flat key.
            if effort is None and not _profile_adaptive_thinking_always_on(self._profile):
                # Adaptive thinking is opt-in via an explicit effort on Opus/Sonnet
                # profiles. Always-on models (Fable 5) may still need a thinking
                # block without ``output_config`` to request display settings.
                return None
            display = _adaptive_display_setting(native_thinking, native)
            payload: dict[str, object] = {"type": "adaptive", "display": display}
            payload.update(native_thinking)
            payload.update(native)
            thinking_obj = _json_object(payload)
            output_config = _json_object({"effort": effort.value}) if effort is not None else None
            return _ThinkingPayload(thinking=thinking_obj, output_config=output_config)

        budget = _thinking_budget(reasoning, native_thinking, self._profile, effort=effort)
        legacy_payload: dict[str, object] = {
            "type": "enabled",
            "budget_tokens": budget,
        }
        legacy_payload.update(native_thinking)
        legacy_payload.update(native)
        result = _json_object(legacy_payload)
        self._validate_thinking_payload(result, reasoning=reasoning, max_tokens=max_tokens)
        return _ThinkingPayload(thinking=result, output_config=None)

    def _validate_thinking_payload(
        self,
        thinking: JsonObject,
        *,
        reasoning: ReasoningSettings,
        max_tokens: int,
    ) -> None:
        budget = _thinking_budget_from_payload(thinking)
        if self._profile is None:
            if budget >= max_tokens:
                raise ConfigurationError(
                    "Anthropic thinking requires max_tokens greater than budget_tokens"
                )
            return
        validate_capability_requirements(
            self._profile,
            CapabilityRequirements(
                require_tool_calling=False,
                require_reasoning=True,
                reasoning=reasoning,
                max_output_tokens=max_tokens,
                thinking_enabled=True,
                thinking_budget_tokens=budget,
            ),
        )

    def _validate_request_capabilities(
        self,
        request: ModelRequest,
        *,
        reasoning: ReasoningSettings | None,
        temperature: float | None,
        max_tokens: int,
        thinking: JsonObject | None,
        force_tool_choice: bool,
    ) -> None:
        if self._profile is None:
            return
        validate_capability_requirements(
            self._profile,
            CapabilityRequirements(
                require_tool_calling=bool(request.tools),
                require_reasoning=reasoning is not None,
                reasoning=reasoning,
                temperature=temperature,
                max_output_tokens=max_tokens,
                force_tool_choice=force_tool_choice,
                thinking_enabled=thinking is not None,
                thinking_budget_tokens=(
                    _optional_thinking_budget_from_payload(thinking)
                    if thinking is not None
                    else None
                ),
            ),
        )

    def _resolve_api_key(self) -> SecretValue | None:
        if self._api_key is not None:
            return self._api_key
        if self._api_key_env_var is None:
            return None
        return EnvironmentSecretSource(env_var=self._api_key_env_var).resolve(self._environment)


@dataclass(frozen=True, slots=True)
class _ParsedContent:
    text_parts: list[str]
    tool_calls: list[ToolCall]
    provider_items: list[ProviderItemMetadata]
    item_ids: list[str]
    thinking_blocks: list[JsonObject]
    content_block_types: list[str]
    unknown_content_block_types: list[str]
    partial_text_present: bool


def _parse_response_payload(
    payload: JsonObject,
    *,
    provider_name: str,
    fallback_model_name: str,
    requested_reasoning: ReasoningSettings | None,
    profile: ModelProfile | None,
) -> ModelResponse:
    _validate_message_response_payload(payload)
    native_stop_reason = _optional_provider_string(payload, "stop_reason")
    completion_status = _completion_status(native_stop_reason)
    incomplete_details = _incomplete_details(payload, native_stop_reason)
    usage = _usage_from_payload(payload)
    parsed_content = _parse_content_blocks(payload)
    reasoning = _reasoning_metadata(
        parsed_content,
        requested_reasoning=requested_reasoning,
        usage=usage,
    )
    provider_response_id = _optional_provider_string(payload, "id")
    model_name = _optional_provider_string(payload, "model") or fallback_model_name
    assistant_message = _assistant_message_from_texts(
        parsed_content.text_parts,
        include_text=completion_status is ProviderCompletionStatus.COMPLETED,
    )
    return ModelResponse(
        response_id=provider_response_id or new_id("model_resp"),
        assistant_message=assistant_message,
        tool_calls=parsed_content.tool_calls,
        stop_reason=_normalized_stop_reason(
            completion_status,
            native_stop_reason=native_stop_reason,
            has_final_text=assistant_message is not None,
            has_tool_calls=bool(parsed_content.tool_calls),
        ),
        provider_completion_status=completion_status,
        incomplete_details=incomplete_details,
        usage=usage,
        reasoning=reasoning,
        provider_metadata=_provider_metadata(
            payload,
            provider_name=provider_name,
            model_name=model_name,
            native_stop_reason=native_stop_reason,
            parsed_content=parsed_content,
            profile=profile,
        ),
        response_metadata=_response_metadata(
            payload,
            native_stop_reason=native_stop_reason,
            completion_status=completion_status,
            parsed_content=parsed_content,
        ),
    )


def _response_payload(body: JsonValue) -> JsonObject:
    if not isinstance(body, Mapping):
        raise ProviderProtocolError("Anthropic Messages payload must be a JSON object")
    return _json_object(cast(Mapping[str, object], body))


def _validate_message_response_payload(payload: JsonObject) -> None:
    message_type = _optional_provider_string(payload, "type")
    if message_type is not None and message_type != "message":
        raise ProviderProtocolError(f"Anthropic Messages type {message_type!r} is not supported")
    role = _optional_provider_string(payload, "role")
    if role is not None and role != "assistant":
        raise ProviderProtocolError(f"Anthropic Messages role {role!r} is not supported")


def _completion_status(stop_reason: str | None) -> ProviderCompletionStatus:
    if stop_reason == "max_tokens":
        return ProviderCompletionStatus.INCOMPLETE
    if stop_reason == "pause_turn":
        return ProviderCompletionStatus.INCOMPLETE
    return ProviderCompletionStatus.COMPLETED


def _incomplete_details(payload: JsonObject, stop_reason: str | None) -> JsonObject:
    details: dict[str, object] = {}
    if stop_reason in {"max_tokens", "pause_turn"}:
        details["stop_reason"] = stop_reason
    stop_sequence = payload.get("stop_sequence")
    if stop_sequence is not None:
        details["stop_sequence"] = _json_value(stop_sequence)
    return _json_object(details)


def _parse_content_blocks(payload: JsonObject) -> _ParsedContent:
    raw_content = payload.get("content")
    if not isinstance(raw_content, list):
        raise ProviderProtocolError("Anthropic Messages payload requires a content list")

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    provider_items: list[ProviderItemMetadata] = []
    item_ids: list[str] = []
    thinking_blocks: list[JsonObject] = []
    content_block_types: list[str] = []
    unknown_content_block_types: list[str] = []

    for block_order, raw_block in enumerate(raw_content):
        if not isinstance(raw_block, Mapping):
            raise ProviderProtocolError("Anthropic Messages content blocks must be objects")
        block = _json_object(cast(Mapping[str, object], raw_block))
        block_type = _required_provider_string(block, "type", context="content block")
        content_block_types.append(block_type)
        block_id = _content_block_id(block, block_type=block_type)
        if block_id is not None:
            item_ids.append(block_id)

        if block_type == "text":
            text = _required_provider_string(block, "text", context="text content block")
            text_parts.append(text)
            provider_items.append(
                ProviderItemMetadata(
                    kind=ProviderItemKind.OUTPUT_TEXT,
                    order=block_order,
                    provider_block_id=block_id,
                    redacted_details=_text_block_details(block),
                )
            )
        elif block_type == "tool_use":
            tool_call = _tool_call_from_tool_use_block(block, order=len(tool_calls))
            tool_calls.append(tool_call)
            provider_items.append(
                ProviderItemMetadata(
                    kind=ProviderItemKind.TOOL_USE,
                    order=block_order,
                    provider_item_id=tool_call.provider_item_id,
                    provider_tool_use_id=tool_call.provider_tool_use_id,
                    provider_block_id=block_id,
                    tool_name=tool_call.tool_name,
                    status=tool_call.provider_status,
                    redacted_details=_tool_use_block_details(block, tool_call=tool_call),
                )
            )
        elif block_type == "thinking":
            thinking_block = _thinking_continuation_block(block)
            thinking_blocks.append(thinking_block)
            provider_items.append(
                ProviderItemMetadata(
                    kind=ProviderItemKind.THINKING,
                    order=block_order,
                    provider_block_id=block_id,
                    thinking_signature=_optional_provider_string(block, "signature"),
                    redacted_details=_thinking_provider_item_details(block),
                )
            )
        elif block_type == "redacted_thinking":
            redacted_block = _redacted_thinking_continuation_block(block)
            thinking_blocks.append(redacted_block)
            provider_items.append(
                ProviderItemMetadata(
                    kind=ProviderItemKind.THINKING,
                    order=block_order,
                    provider_block_id=block_id,
                    encrypted_reasoning_content=_required_provider_string(
                        block,
                        "data",
                        context="redacted_thinking content block",
                    ),
                    redacted_details=_thinking_provider_item_details(block),
                )
            )
        else:
            unknown_content_block_types.append(block_type)

    return _ParsedContent(
        text_parts=text_parts,
        tool_calls=tool_calls,
        provider_items=provider_items,
        item_ids=item_ids,
        thinking_blocks=thinking_blocks,
        content_block_types=content_block_types,
        unknown_content_block_types=unknown_content_block_types,
        partial_text_present=bool(text_parts),
    )


def _tool_call_from_tool_use_block(block: JsonObject, *, order: int) -> ToolCall:
    tool_use_id = _required_provider_string(block, "id", context="tool_use content block")
    tool_name = _required_provider_string(block, "name", context="tool_use content block")
    raw_input = block.get("input")
    if not isinstance(raw_input, Mapping):
        raise ProviderProtocolError("Anthropic Messages tool_use input must be a JSON object")
    input_payload = _json_object(cast(Mapping[str, object], raw_input))
    try:
        return ToolCall.from_provider_arguments(
            call_id=tool_use_id,
            tool_name=tool_name,
            arguments=input_payload,
            order=order,
            provider_item_id=tool_use_id,
            provider_tool_use_id=tool_use_id,
            provider_status=_optional_provider_string(block, "status"),
            provider_metadata={
                "anthropic_type": "tool_use",
                "anthropic_raw_input": input_payload,
            },
        )
    except (ValueError, ValidationError) as exc:
        raise ProviderProtocolError(
            f"Anthropic Messages tool_use input for {tool_name!r} is invalid"
        ) from exc


def _thinking_continuation_block(block: JsonObject) -> JsonObject:
    thinking = _required_provider_string_allow_empty(
        block,
        "thinking",
        context="thinking content block",
    )
    payload: dict[str, object] = {"type": "thinking", "thinking": thinking}
    signature = _optional_provider_string(block, "signature")
    if signature is not None:
        payload["signature"] = signature
    block_id = _content_block_id(block, block_type="thinking")
    if block_id is not None:
        payload["id"] = block_id
    return _json_object(payload)


def _redacted_thinking_continuation_block(block: JsonObject) -> JsonObject:
    data = _required_provider_string(block, "data", context="redacted_thinking content block")
    payload: dict[str, object] = {"type": "redacted_thinking", "data": data}
    block_id = _content_block_id(block, block_type="redacted_thinking")
    if block_id is not None:
        payload["id"] = block_id
    return _json_object(payload)


def _assistant_message_from_texts(
    text_parts: list[str],
    *,
    include_text: bool,
) -> AssistantMessage | None:
    if not include_text or not text_parts:
        return None
    return AssistantMessage(content=[TextContent(text=text) for text in text_parts if text])


def _usage_from_payload(payload: JsonObject) -> Usage:
    raw_usage = payload.get("usage")
    if raw_usage is None:
        return Usage()
    if not isinstance(raw_usage, Mapping):
        raise ProviderProtocolError("Anthropic Messages usage must be a JSON object")
    usage = _json_object(cast(Mapping[str, object], raw_usage))
    # Anthropic's older Messages API exposes ``thinking_tokens`` at the top of the
    # usage block; the newer adaptive-thinking endpoint (opus-4-6/4-7/4-8,
    # sonnet-4-6) puts the same value inside ``output_tokens_details``. Accept
    # either location and roll both into ``reasoning_tokens`` for parity with the
    # OpenAI Responses adapter.
    reasoning_tokens = (
        (_optional_usage_non_negative_int(usage, "thinking_tokens") or 0)
        + (_optional_usage_non_negative_int(usage, "reasoning_tokens") or 0)
        + _nested_thinking_tokens(usage)
    )
    return Usage(
        tokens=TokenUsage(
            input_tokens=_optional_usage_non_negative_int(usage, "input_tokens") or 0,
            output_tokens=_optional_usage_non_negative_int(usage, "output_tokens") or 0,
            reasoning_tokens=reasoning_tokens,
            cache_read_tokens=_optional_usage_non_negative_int(
                usage,
                "cache_read_input_tokens",
            )
            or 0,
            cache_write_tokens=_optional_usage_non_negative_int(
                usage,
                "cache_creation_input_tokens",
            )
            or 0,
            provider_details=_usage_provider_details(usage),
        )
    )


def _nested_thinking_tokens(usage: JsonObject) -> int:
    """Return ``output_tokens_details.thinking_tokens`` if present and well-formed.

    Anthropic's adaptive-thinking endpoint nests the thinking-token count under
    ``output_tokens_details`` rather than exposing it at the top of the usage
    block. Older Messages calls keep the value at the top level; we read both.
    """

    raw_details = usage.get("output_tokens_details")
    if not isinstance(raw_details, Mapping):
        return 0
    details = _json_object(cast(Mapping[str, object], raw_details))
    return _optional_usage_non_negative_int(details, "thinking_tokens") or 0


def _usage_provider_details(usage: JsonObject) -> dict[str, int]:
    details: dict[str, int] = {}
    common = {
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "thinking_tokens",
        "reasoning_tokens",
    }
    for key, value in usage.items():
        if key in common:
            continue
        _collect_integer_usage_details(str(key), value, details)
    return details


def _collect_integer_usage_details(prefix: str, value: object, details: dict[str, int]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int):
        if value < 0:
            raise ProviderProtocolError(f"Anthropic Messages usage field {prefix!r} is negative")
        details[prefix] = value
        return
    if isinstance(value, Mapping):
        for child_key, child_value in cast(Mapping[object, object], value).items():
            _collect_integer_usage_details(f"{prefix}.{child_key}", child_value, details)


def _reasoning_metadata(
    parsed_content: _ParsedContent,
    *,
    requested_reasoning: ReasoningSettings | None,
    usage: Usage,
) -> ReasoningMetadata | None:
    continuation = _reasoning_continuation(parsed_content.thinking_blocks)
    reasoning_tokens = usage.tokens.reasoning_tokens or None
    has_reasoning_metadata = bool(requested_reasoning or continuation or reasoning_tokens)
    if not has_reasoning_metadata:
        return None
    return ReasoningMetadata(
        requested=(
            requested_reasoning.model_copy(deep=True)
            if requested_reasoning is not None
            else None
        ),
        native_settings=_reasoning_native_settings(parsed_content),
        reasoning_tokens=reasoning_tokens,
        provider_private_continuation=continuation,
        display_policy=(
            requested_reasoning.display_policy
            if requested_reasoning is not None
            else ReasoningSettings().display_policy
        ),
    )


def _reasoning_native_settings(parsed_content: _ParsedContent) -> JsonObject:
    if not parsed_content.thinking_blocks:
        return {}
    return _json_object(
        {
            "anthropic_thinking_block_count": len(parsed_content.thinking_blocks),
            "anthropic_thinking_block_types": [
                _required_provider_string(block, "type", context="thinking continuation block")
                for block in parsed_content.thinking_blocks
            ],
        }
    )


def _reasoning_continuation(
    thinking_blocks: list[JsonObject],
) -> list[ReasoningContinuationMetadata]:
    continuation: list[ReasoningContinuationMetadata] = []
    for order, block in enumerate(thinking_blocks):
        block_type = _required_provider_string(block, "type", context="thinking continuation block")
        block_id = _optional_provider_string(block, "id")
        if block_type == "thinking":
            continuation.append(
                ReasoningContinuationMetadata(
                    provider_name="anthropic",
                    kind="thinking",
                    order=order,
                    provider_block_id=block_id,
                    signature=_optional_provider_string(block, "signature"),
                    redacted_details=_reasoning_continuation_details(block, order=order),
                )
            )
        elif block_type == "redacted_thinking":
            continuation.append(
                ReasoningContinuationMetadata(
                    provider_name="anthropic",
                    kind="redacted_thinking",
                    order=order,
                    provider_block_id=block_id,
                    encrypted_content=_required_provider_string(
                        block,
                        "data",
                        context="redacted_thinking continuation block",
                    ),
                    redacted_details=_reasoning_continuation_details(block, order=order),
                )
            )
    return continuation


def _provider_metadata(
    payload: JsonObject,
    *,
    provider_name: str,
    model_name: str,
    native_stop_reason: str | None,
    parsed_content: _ParsedContent,
    profile: ModelProfile | None,
) -> ProviderMetadata:
    return ProviderMetadata(
        provider_name=provider_name,
        model_name=model_name,
        response_id=_optional_provider_string(payload, "id"),
        native_stop_reason=native_stop_reason,
        item_ids=list(parsed_content.item_ids),
        items=parsed_content.provider_items,
        continuation_strategy=ContinuationStrategy.STATELESS_REPLAY,
        provider_side_continuation_available=False,
        stateless_continuation_required=_stateless_continuation_required(profile),
        redacted_raw_details=_provider_raw_details(
            payload,
            native_stop_reason=native_stop_reason,
            parsed_content=parsed_content,
        ),
    )


def _stateless_continuation_required(profile: ModelProfile | None) -> bool:
    if profile is None:
        return True
    return profile.continuation.stateless_continuation_required


def _response_metadata(
    payload: JsonObject,
    *,
    native_stop_reason: str | None,
    completion_status: ProviderCompletionStatus,
    parsed_content: _ParsedContent,
) -> JsonObject:
    metadata: dict[str, object] = {
        "anthropic_content_block_types": list(parsed_content.content_block_types),
    }
    if native_stop_reason is not None:
        metadata["anthropic_stop_reason"] = native_stop_reason
    message_type = _optional_provider_string(payload, "type")
    if message_type is not None:
        metadata["anthropic_type"] = message_type
    role = _optional_provider_string(payload, "role")
    if role is not None:
        metadata["anthropic_role"] = role
    stop_sequence = payload.get("stop_sequence")
    if stop_sequence is not None:
        metadata["stop_sequence"] = _json_value(stop_sequence)
    service_tier = _service_tier_from_payload(payload)
    if service_tier is not None:
        metadata["service_tier"] = service_tier
    if parsed_content.unknown_content_block_types:
        metadata["unknown_content_block_types"] = list(
            parsed_content.unknown_content_block_types
        )
    if (
        parsed_content.partial_text_present
        and completion_status is ProviderCompletionStatus.INCOMPLETE
    ):
        metadata["partial_text_omitted_due_to_incomplete_status"] = True
    return _json_object(metadata)


def _provider_raw_details(
    payload: JsonObject,
    *,
    native_stop_reason: str | None,
    parsed_content: _ParsedContent,
) -> JsonObject:
    details: dict[str, object] = {
        "content_block_types": list(parsed_content.content_block_types),
    }
    for key in ("type", "role", "stop_sequence", "stop_details", "service_tier"):
        if key in payload:
            details[key] = _json_value(payload[key])
    if native_stop_reason is not None:
        details["stop_reason"] = native_stop_reason
    service_tier = _service_tier_from_payload(payload)
    if service_tier is not None:
        details["service_tier"] = service_tier
    if parsed_content.unknown_content_block_types:
        details["unknown_content_block_types"] = list(parsed_content.unknown_content_block_types)
    usage_non_token_details = _usage_non_token_details(payload)
    if usage_non_token_details:
        details["usage_non_token_details"] = usage_non_token_details
    return _json_object(details)


def _text_block_details(block: JsonObject) -> JsonObject:
    details: dict[str, object] = {"type": "text"}
    block_id = _content_block_id(block, block_type="text")
    if block_id is not None:
        details["id"] = block_id
    citations = block.get("citations")
    if isinstance(citations, list):
        details["citation_count"] = len(citations)
    return _json_object(details)


def _tool_use_block_details(block: JsonObject, *, tool_call: ToolCall) -> JsonObject:
    details: dict[str, object] = {
        "type": "tool_use",
        "id": tool_call.provider_tool_use_id or tool_call.call_id,
        "input_keys": sorted(tool_call.arguments),
    }
    status = _optional_provider_string(block, "status")
    if status is not None:
        details["status"] = status
    return _json_object(details)


def _thinking_provider_item_details(block: JsonObject) -> JsonObject:
    block_type = _required_provider_string(block, "type", context="thinking content block")
    details: dict[str, object] = {"type": block_type}
    block_id = _content_block_id(block, block_type=block_type)
    if block_id is not None:
        details["id"] = block_id
    signature = _optional_provider_string(block, "signature")
    if signature is not None:
        details["has_signature"] = True
    if block_type == "thinking":
        thinking = _required_provider_string_allow_empty(
            block,
            "thinking",
            context="thinking content block",
        )
        details["has_thinking"] = bool(thinking)
        if not thinking:
            details["thinking_omitted"] = True
    elif block_type == "redacted_thinking":
        details["has_redacted_data"] = True
    return _json_object(details)


def _reasoning_continuation_details(block: JsonObject, *, order: int) -> JsonObject:
    details: dict[str, object] = {
        "anthropic_block": _copy_json_object(block),
        "anthropic_content_order": order,
    }
    signature = _optional_provider_string(block, "signature")
    if signature is not None:
        details["has_signature"] = True
    return _json_object(details)


def _usage_non_token_details(payload: JsonObject) -> JsonObject:
    raw_usage = payload.get("usage")
    if not isinstance(raw_usage, Mapping):
        return {}
    usage = _json_object(cast(Mapping[str, object], raw_usage))
    details: dict[str, object] = {}
    for key, value in usage.items():
        if _is_token_count_like(value):
            continue
        if isinstance(value, Mapping):
            nested_non_token = {
                str(nested_key): _json_value(nested_value)
                for nested_key, nested_value in cast(Mapping[object, object], value).items()
                if not _is_token_count_like(nested_value)
            }
            if nested_non_token:
                details[str(key)] = nested_non_token
        else:
            details[str(key)] = _json_value(value)
    return _json_object(details)


def _is_token_count_like(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _service_tier_from_payload(payload: JsonObject) -> str | None:
    direct = _optional_provider_string(payload, "service_tier")
    if direct is not None:
        return direct
    raw_usage = payload.get("usage")
    if not isinstance(raw_usage, Mapping):
        return None
    usage = _json_object(cast(Mapping[str, object], raw_usage))
    return _optional_provider_string(usage, "service_tier")


def _content_block_id(block: JsonObject, *, block_type: str) -> str | None:
    value = _optional_provider_string(block, "id")
    if value is not None:
        return value
    if block_type == "tool_use":
        return _optional_provider_string(block, "tool_use_id")
    return None


def _normalized_stop_reason(
    completion_status: ProviderCompletionStatus,
    *,
    native_stop_reason: str | None,
    has_final_text: bool,
    has_tool_calls: bool,
) -> StopReason:
    if completion_status is ProviderCompletionStatus.INCOMPLETE:
        if native_stop_reason == "max_tokens":
            return StopReason.MAX_TOKENS
        return StopReason.PROVIDER_STOP_REASON
    if has_tool_calls:
        return StopReason.PROVIDER_STOP_REASON
    if has_final_text:
        return StopReason.FINAL_RESPONSE
    return StopReason.PROVIDER_STOP_REASON


def _optional_provider_string(value: JsonObject, key: str) -> str | None:
    raw = value.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ProviderProtocolError(f"Anthropic Messages field {key!r} must be a string")
    if not raw:
        return None
    return raw


def _required_provider_string_allow_empty(
    value: JsonObject,
    key: str,
    *,
    context: str,
) -> str:
    raw = value.get(key)
    if not isinstance(raw, str):
        raise ProviderProtocolError(f"Anthropic Messages {context} requires string {key!r}")
    return raw


def _required_provider_string(value: JsonObject, key: str, *, context: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw:
        raise ProviderProtocolError(f"Anthropic Messages {context} requires non-empty {key!r}")
    return raw


def _optional_usage_non_negative_int(value: JsonObject, key: str) -> int | None:
    raw = value.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ProviderProtocolError(f"Anthropic Messages usage field {key!r} must be an integer")
    if raw < 0:
        raise ProviderProtocolError(
            f"Anthropic Messages usage field {key!r} must be non-negative"
        )
    return raw


def _validate_anthropic_profile(
    profile: ModelProfile,
    *,
    model_name: str,
    provider_name: str,
) -> None:
    if profile.api is not ProviderApi.ANTHROPIC_MESSAGES:
        raise ConfigurationError(
            "Anthropic Messages adapter requires an anthropic_messages profile"
        )
    if profile.model_name != model_name:
        raise ConfigurationError("profile model_name must match adapter model_name")
    if profile.provider_name != provider_name:
        raise ConfigurationError("profile provider_name must match adapter provider_name")


def _secret_value_from_api_key(value: SecretValue | str | None) -> SecretValue | None:
    if value is None:
        return None
    if isinstance(value, SecretValue):
        return value
    return SecretValue(
        value=SecretStr(value),
        source_kind=SecretSourceKind.LITERAL,
    )


def _validate_base_url(base_url: str) -> str:
    if not base_url:
        raise ValueError("base_url must be non-empty")
    return base_url.rstrip("/")


def _messages_endpoint_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith(f"/{_DEFAULT_MESSAGES_PATH}"):
        return stripped
    return f"{stripped}/{_DEFAULT_MESSAGES_PATH}"


def _base_url_from_config(
    model_config: AgentModelConfig,
    runtime_config: ProviderRuntimeConfig,
    environment: Mapping[str, str],
) -> str:
    if runtime_config.model.base_url is not None:
        return runtime_config.model.base_url
    if model_config.endpoint is not None:
        return model_config.endpoint
    env_name = runtime_config.api_key_sources.anthropic_base_url_env
    env_value = environment.get(env_name)
    if env_value:
        return env_value
    return _DEFAULT_ANTHROPIC_BASE_URL


def _redactor_from_runtime_config(
    runtime_config: ProviderRuntimeConfig,
    *,
    base_url: str,
) -> Redactor:
    return Redactor(
        secret_source_names=(
            runtime_config.secret_source_names() if runtime_config.redaction.redact_secrets else ()
        ),
        secret_header_names=DEFAULT_SECRET_HEADER_NAMES,
        patterns=runtime_config.redaction.patterns,
        mildly_sensitive_urls=(
            (base_url,) if runtime_config.redaction.redact_mildly_sensitive_urls else ()
        ),
    )


def _has_auth_header_configs(headers: Iterable[ProviderHeaderConfig]) -> bool:
    return any(header.name.lower() in _AUTH_HEADER_NAMES for header in headers)


def _system_text_from_messages(messages: Iterable[ModelMessage]) -> str:
    sections: list[str] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            text = _message_content_text(message)
            if text:
                sections.append(text)
        elif isinstance(message, DeveloperMessage):
            text = _message_content_text(message)
            if text:
                sections.append(f"Developer instructions:\n{text}")
    return "\n\n".join(sections)


def _messages_from_model_messages(messages: Iterable[ModelMessage]) -> list[JsonObject]:
    native_messages: list[JsonObject] = []
    for message in messages:
        if isinstance(message, SystemMessage | DeveloperMessage):
            continue
        if isinstance(message, UserMessage):
            _append_native_message(native_messages, role="user", content=_text_blocks(message))
        elif isinstance(message, AssistantMessage):
            _append_native_message(
                native_messages,
                role="assistant",
                content=_assistant_content_blocks(message),
            )
        else:
            _append_native_message(
                native_messages,
                role="user",
                content=[_tool_result_block(message)],
            )
    if not native_messages:
        raise ConfigurationError("Anthropic Messages requests require at least one message")
    return native_messages


def _append_native_message(
    messages: list[JsonObject],
    *,
    role: str,
    content: list[JsonObject],
) -> None:
    if not content:
        return
    if messages and messages[-1].get("role") == role:
        existing = messages[-1].get("content")
        if isinstance(existing, list):
            existing.extend(content)
            messages[-1] = _json_object(messages[-1])
            return
    messages.append(_json_object({"role": role, "content": content}))


def _assistant_content_blocks(message: AssistantMessage) -> list[JsonObject]:
    blocks: list[JsonObject] = []
    blocks.extend(_thinking_blocks_from_message_metadata(message))
    blocks.extend(_text_blocks(message))
    for tool_call in sorted(assistant_tool_calls(message), key=lambda call: call.order):
        blocks.append(_tool_use_block(tool_call))
    return blocks


def _text_blocks(message: UserMessage | AssistantMessage) -> list[JsonObject]:
    blocks: list[JsonObject] = []
    for part in message.content:
        if isinstance(part, TextContent):
            blocks.append(_json_object({"type": "text", "text": part.text}))
        else:
            blocks.append(_json_object({"type": "text", "text": _compaction_summary_text(part)}))
    return blocks


def _message_content_text(message: SystemMessage | DeveloperMessage) -> str:
    parts: list[str] = []
    for part in message.content:
        if isinstance(part, TextContent):
            parts.append(part.text)
        else:
            parts.append(_compaction_summary_text(part))
    return "\n\n".join(parts)


def _compaction_summary_text(part: CompactionSummaryContent) -> str:
    if part.covered_message_ids:
        covered = ", ".join(part.covered_message_ids)
        return f"Compaction summary covering messages [{covered}]:\n{part.summary}"
    return f"Compaction summary:\n{part.summary}"


def _tool_use_block(tool_call: ToolCall) -> JsonObject:
    tool_use_id = tool_call.provider_tool_use_id or tool_call.provider_call_id or tool_call.call_id
    raw_input = tool_call.provider_metadata.get("anthropic_raw_input")
    if isinstance(raw_input, Mapping):
        input_payload = _json_object(cast(Mapping[str, object], raw_input))
    else:
        input_payload = _copy_json_object(tool_call.arguments)
    return _json_object(
        {
            "type": "tool_use",
            "id": tool_use_id,
            "name": tool_call.tool_name,
            "input": input_payload,
        }
    )


def _tool_result_block(message: ToolResultMessage) -> JsonObject:
    result = message.result
    content = _tool_message_content_text(message) or model_visible_tool_result_text(result)
    tool_use_id = (
        result.provider_tool_use_id
        or _tool_use_id_from_message_metadata(message)
        or result.provider_call_id
        or result.tool_call_id
    )
    block: dict[str, object] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if not result.success:
        block["is_error"] = True
    return _json_object(block)


def _tool_message_content_text(message: ToolResultMessage) -> str:
    parts: list[str] = []
    for part in message.content:
        if isinstance(part, TextContent):
            parts.append(part.text)
        else:
            parts.append(_compaction_summary_text(part))
    return "\n\n".join(parts)


def _tool_use_id_from_message_metadata(message: ToolResultMessage) -> str | None:
    for key in ("anthropic_tool_use_id", "tool_use_id"):
        value = message.provider_metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _thinking_blocks_from_message_metadata(message: AssistantMessage) -> list[JsonObject]:
    blocks: list[JsonObject] = []
    raw_blocks = message.provider_metadata.get("anthropic_content_blocks")
    if isinstance(raw_blocks, list):
        for raw_block in raw_blocks:
            block = _thinking_block_from_raw(raw_block)
            if block is not None:
                blocks.append(block)

    raw_reasoning = message.provider_metadata.get(ASSISTANT_REASONING_METADATA_KEY)
    if isinstance(raw_reasoning, Mapping):
        try:
            reasoning = ReasoningMetadata.model_validate(raw_reasoning, strict=False)
        except ValidationError:
            return blocks
        for continuation in reasoning.provider_private_continuation:
            block = _thinking_block_from_continuation(continuation.model_dump(mode="python"))
            if block is not None:
                blocks.append(block)
    return blocks


def _thinking_block_from_raw(value: object) -> JsonObject | None:
    if not isinstance(value, Mapping):
        return None
    block = _json_object(cast(Mapping[str, object], value))
    block_type = block.get("type")
    if block_type == "thinking":
        thinking = block.get("thinking")
        signature = block.get("signature")
        has_signature = isinstance(signature, str) and bool(signature)
        if not isinstance(thinking, str) or (not thinking and not has_signature):
            return None
        payload: dict[str, object] = {"type": "thinking", "thinking": thinking}
        if has_signature:
            payload["signature"] = signature
        return _json_object(payload)
    if block_type == "redacted_thinking":
        data = block.get("data")
        if not isinstance(data, str) or not data:
            return None
        return _json_object({"type": "redacted_thinking", "data": data})
    return None


def _thinking_block_from_continuation(value: JsonObject) -> JsonObject | None:
    provider_name = value.get("provider_name")
    if provider_name is not None and provider_name != "anthropic":
        return None
    details = value.get("redacted_details")
    if isinstance(details, Mapping):
        raw_block = details.get("anthropic_block")
        block = _thinking_block_from_raw(raw_block)
        if block is not None:
            return block
    kind = value.get("kind")
    encrypted_content = value.get("encrypted_content")
    if kind == "redacted_thinking" and isinstance(encrypted_content, str) and encrypted_content:
        return _json_object({"type": "redacted_thinking", "data": encrypted_content})
    if kind != "thinking":
        return None
    thinking_text = _thinking_text_from_details(details)
    signature = value.get("signature")
    if thinking_text is not None:
        payload: dict[str, object] = {"type": "thinking", "thinking": thinking_text}
        if isinstance(signature, str) and signature:
            payload["signature"] = signature
        return _json_object(payload)
    if isinstance(encrypted_content, str) and encrypted_content:
        return _json_object({"type": "redacted_thinking", "data": encrypted_content})
    return None


def _thinking_text_from_details(details: JsonValue | object) -> str | None:
    if not isinstance(details, Mapping):
        return None
    detail_map = cast(Mapping[str, object], details)
    for key in ("thinking", "text"):
        value = detail_map.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _tool_payload(tool: JsonObject) -> JsonObject:
    name = _required_string(tool, "name", context="tool definition")
    description = _required_string(tool, "description", context=f"tool {name!r}")
    input_schema = _required_object(tool, "arguments_schema", context=f"tool {name!r}")
    return _json_object(
        {
            "name": name,
            "description": description,
            "input_schema": _copy_json_object(input_schema),
        }
    )


def _pop_native_thinking_settings(native_settings: JsonObject) -> JsonObject:
    raw_thinking = native_settings.pop("thinking", None)
    if raw_thinking is None:
        return {}
    if not isinstance(raw_thinking, Mapping):
        raise ConfigurationError("ReasoningSettings.native_settings.thinking must be an object")
    return _json_object(cast(Mapping[str, object], raw_thinking))


def _thinking_budget(
    reasoning: ReasoningSettings,
    native_thinking: JsonObject,
    profile: ModelProfile | None,
    *,
    effort: ReasoningEffort | None = None,
) -> int:
    native_budget = _optional_positive_int(native_thinking.get("budget_tokens"), "budget_tokens")
    if native_budget is not None:
        return native_budget
    if reasoning.max_reasoning_tokens is not None and reasoning.max_reasoning_tokens > 0:
        return reasoning.max_reasoning_tokens
    resolved_effort = effort or reasoning.effort or ReasoningEffort.LOW
    budget = _DEFAULT_THINKING_BUDGETS[resolved_effort]
    minimum = profile.reasoning.min_thinking_budget_tokens if profile is not None else None
    if minimum is not None and budget < minimum:
        return minimum
    return budget


def _thinking_budget_from_payload(thinking: JsonObject) -> int:
    budget = _optional_positive_int(thinking.get("budget_tokens"), "budget_tokens")
    if budget is None:
        raise ConfigurationError("Anthropic thinking requires positive budget_tokens")
    return budget


def _optional_thinking_budget_from_payload(thinking: JsonObject) -> int | None:
    """Return ``budget_tokens`` if present; adaptive thinking has none."""

    return _optional_positive_int(thinking.get("budget_tokens"), "budget_tokens")


def _profile_requires_adaptive_thinking(profile: ModelProfile | None) -> bool:
    if profile is None:
        return False
    return profile.reasoning.requires_adaptive_thinking


def _profile_adaptive_thinking_always_on(profile: ModelProfile | None) -> bool:
    if profile is None:
        return False
    return profile.reasoning.adaptive_thinking_always_on


def _remapped_effort(
    effort: ReasoningEffort | None,
    profile: ModelProfile | None,
) -> ReasoningEffort | None:
    """Apply ``thinking_level_map`` if set. ``None`` may signal "off" for an effort."""

    if effort is None or profile is None:
        return effort
    mapping = profile.reasoning.thinking_level_map
    if effort in mapping:
        return mapping[effort]
    return effort


def _adaptive_display_setting(native_thinking: JsonObject, native: JsonObject) -> str:
    """Resolve the ``display`` field for adaptive thinking.

    Default is ``"summarized"`` (matches pi's behaviour for newer Claude
    models). Callers can override via ``reasoning.native_settings`` using either
    ``{"thinking": {"display": ...}}`` or a flat ``{"display": ...}``; the
    nested form wins so the precedence matches the legacy budget path.
    """

    raw = native_thinking.get("display")
    if raw is None:
        raw = native.pop("display", None)
    if raw is None:
        return "summarized"
    if not isinstance(raw, str) or not raw:
        raise ConfigurationError(
            "Anthropic adaptive thinking display must be a non-empty string"
        )
    # Consume the value if it came from native_thinking so it isn't duplicated.
    native_thinking.pop("display", None)
    return raw


@dataclass(frozen=True, slots=True)
class _ThinkingPayload:
    thinking: JsonObject
    output_config: JsonObject | None


def _optional_positive_int(value: JsonValue | object, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"Anthropic thinking {field_name} must be a positive integer")
    return value


def _reasoning_disabled(metadata: JsonObject) -> bool:
    # Produced by the turn loop (and honoured by every adapter) to suppress
    # reasoning/thinking for a single request regardless of adapter defaults.
    return metadata.get("disable_reasoning") is True


def _pop_tool_choice_request(settings: JsonObject, metadata: JsonObject) -> object | None:
    value: object | None = settings.pop("tool_choice", None)
    for key in ("provider_tool_choice", "anthropic_tool_choice", "tool_choice"):
        if key in metadata:
            value = metadata[key]
    force_tool_name = metadata.get("force_tool_name")
    if isinstance(force_tool_name, str) and force_tool_name:
        return {"type": "tool", "name": force_tool_name}
    return value


def _tool_choice_payload(
    value: object | None,
    *,
    thinking_enabled: bool,
    profile: ModelProfile | None,
) -> JsonObject | None:
    if value is None:
        return None
    payload = _normalize_tool_choice(value)
    if thinking_enabled and _is_forced_tool_choice(payload):
        if profile is None or not profile.tools.forced_tool_choice_compatible_with_thinking:
            return None
    return payload


def _normalize_tool_choice(value: object) -> JsonObject:
    if isinstance(value, str):
        if value not in {"auto", "any", "none"}:
            raise ConfigurationError("Anthropic tool_choice string must be auto, any, or none")
        return _json_object({"type": value})
    if not isinstance(value, Mapping):
        raise ConfigurationError("Anthropic tool_choice must be a string or object")
    payload = _json_object(cast(Mapping[str, object], value))
    choice_type = payload.get("type")
    if not isinstance(choice_type, str) or not choice_type:
        raise ConfigurationError("Anthropic tool_choice requires non-empty type")
    if choice_type == "tool":
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigurationError("Anthropic tool_choice type 'tool' requires name")
    elif choice_type not in {"auto", "any", "none"}:
        raise ConfigurationError("Anthropic tool_choice type must be auto, any, none, or tool")
    return payload


def _is_forced_tool_choice(tool_choice: JsonObject | None) -> bool:
    if tool_choice is None:
        return False
    choice_type = tool_choice.get("type")
    return choice_type in {"any", "tool"}


def _metadata_settings(metadata: JsonObject) -> JsonObject:
    merged: JsonObject = {}
    for key in ("provider_request_settings", "anthropic_messages_request_settings"):
        value = metadata.get(key)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ConfigurationError(f"request_metadata.{key} must be a JSON object")
        merged.update(_json_object(cast(Mapping[str, object], value)))
    return merged


def _pop_optional_float_setting(settings: JsonObject, key: str) -> float | None:
    value = settings.pop(key, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"Anthropic Messages setting {key!r} must be a number")
    return float(value)


def _apply_extra_settings(
    body: dict[str, object],
    settings: JsonObject,
    profile: ModelProfile | None,
) -> None:
    for key, value in settings.items():
        _validate_extra_setting_supported(key, profile)
        body[key] = _json_value(value)


def _validate_extra_setting_supported(key: str, profile: ModelProfile | None) -> None:
    if profile is None:
        return
    if key not in profile.settings.supported_extra_settings:
        raise ConfigurationError(
            f"model {profile.provider_name}/{profile.model_name} does not support "
            f"Anthropic Messages request setting {key!r}"
        )


def _format_secret_header_value(value: str, *, scheme: str | None) -> str:
    if scheme is None or not scheme:
        return value
    if value.lower().startswith(f"{scheme.lower()} "):
        return value
    return f"{scheme} {value}"


def _required_string(value: JsonObject, key: str, *, context: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw:
        raise ConfigurationError(f"Anthropic Messages {context} requires non-empty {key!r}")
    return raw


def _required_object(value: JsonObject, key: str, *, context: str) -> JsonObject:
    raw = value.get(key)
    if not isinstance(raw, Mapping):
        raise ConfigurationError(f"Anthropic Messages {context} requires object {key!r}")
    return _json_object(cast(Mapping[str, object], raw))


def _merge_json_objects(left: JsonObject, right: JsonObject) -> JsonObject:
    result = _copy_json_object(left)
    result.update(_copy_json_object(right))
    return result


def _copy_json_object(value: JsonObject) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(deepcopy(value))


def _json_object(value: Mapping[str, object]) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(dict(value))


def _json_value(value: object) -> JsonValue:
    return _JSON_VALUE_ADAPTER.validate_python(value)


__all__ = ("AnthropicMessagesAdapter",)
