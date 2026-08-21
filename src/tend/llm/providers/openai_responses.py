"""OpenAI-compatible Responses API adapter request translation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import cast

from pydantic import Field, JsonValue, SecretStr, TypeAdapter, ValidationError

from tend._common.errors import ConfigurationError, ProviderProtocolError
from tend._common.types import JsonObject, StopReason, StrictModel, new_id
from tend.llm.config import (
    AgentModelConfig,
    ProviderRuntimeConfig,
    resolve_agent_model_profile,
)
from tend.llm.history import (
    ASSISTANT_MODEL_RESPONSE_ID_METADATA_KEY,
    ASSISTANT_PROVIDER_METADATA_KEY,
    assistant_tool_calls,
)
from tend.llm.models.messages import (
    AssistantMessage,
    CompactionSummaryContent,
    ContentPart,
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
    select_continuation_strategy,
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
    ReasoningSummary,
    ReasoningSummaryPreference,
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

_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_RESPONSES_PATH = "responses"
_OPENAI_DEFAULT_REASONING_EFFORT = ReasoningEffort.MINIMAL
_AUTH_HEADER_NAMES: frozenset[str] = frozenset(
    {"authorization", "cf-aig-authorization", "openai-api-key", "x-api-key"}
)
_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)
_JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


class OpenAIResponsesAdapter:
    """Provider adapter for OpenAI-compatible Responses requests.

    ``build_payload`` and ``build_http_request`` are stable testable seams for
    request-shape tests. ``generate`` posts the translated request and parses the
    Responses payload back into provider-neutral schemas.
    """

    __slots__ = (
        "_api_key",
        "_api_key_env_var",
        "_api_key_header_name",
        "_api_key_scheme",
        "_base_url",
        "_default_max_output_tokens",
        "_default_request_settings",
        "_default_reasoning",
        "_enable_provider_side_continuation",
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
    _enable_provider_side_continuation: bool
    _default_reasoning: ReasoningSettings | None
    _default_request_settings: JsonObject
    _default_max_output_tokens: int | None

    def __init__(
        self,
        *,
        model_name: str,
        provider_name: str = "openai",
        base_url: str = _DEFAULT_OPENAI_BASE_URL,
        profile: ModelProfile | None = None,
        transport: JsonPostTransport | None = None,
        timeout_seconds: float | None = 60.0,
        environment: Mapping[str, str] | None = None,
        api_key_env_var: str | None = None,
        api_key: SecretValue | str | None = None,
        api_key_header_name: str = "Authorization",
        api_key_scheme: str | None = "Bearer",
        extra_headers: Iterable[ProviderHeaderConfig] = (),
        raw_headers: Mapping[str, str] | None = None,
        allow_literal_secret_headers: bool = False,
        redactor: Redactor | None = None,
        enable_provider_side_continuation: bool = False,
        default_reasoning: ReasoningSettings | None = None,
        default_request_settings: Mapping[str, JsonValue] | None = None,
        default_max_output_tokens: int | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must be non-empty")
        if not provider_name:
            raise ValueError("provider_name must be non-empty")
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative when provided")
        if default_max_output_tokens is not None and default_max_output_tokens < 1:
            raise ValueError("default_max_output_tokens must be positive when provided")
        if profile is not None:
            _validate_openai_profile(profile, model_name=model_name, provider_name=provider_name)

        env = {} if environment is None else dict(environment)
        header_configs = tuple(extra_headers)
        self._model_name = model_name
        self._provider_name = provider_name
        self._profile = profile.model_copy(deep=True) if profile is not None else None
        self._base_url = _validate_base_url(base_url)
        self._timeout_seconds = timeout_seconds
        self._environment = env
        self._api_key_env_var = api_key_env_var
        self._api_key = _secret_value_from_api_key(api_key)
        self._api_key_header_name = api_key_header_name
        self._api_key_scheme = api_key_scheme
        self._extra_headers = header_configs
        self._raw_headers = {str(name): str(value) for name, value in (raw_headers or {}).items()}
        self._redactor = redactor
        self._enable_provider_side_continuation = enable_provider_side_continuation
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
    ) -> OpenAIResponsesAdapter:
        """Build an adapter from durable/runtime config and explicit environment.

        This method reads only the supplied ``environment`` mapping; it never
        reads process environment variables directly.
        """

        if model_config.api is not ProviderApi.OPENAI_RESPONSES:
            raise ConfigurationError("OpenAI Responses adapter requires openai_responses API")

        profile = resolve_agent_model_profile(model_config)
        base_url = _base_url_from_config(model_config, runtime_config, environment)
        redactor = _redactor_from_runtime_config(runtime_config, base_url=base_url)
        extra_headers = tuple(runtime_config.model.extra_headers)
        api_key_env_var = None
        if not _has_auth_header_configs(extra_headers):
            api_key_env_var = runtime_config.api_key_sources.openai_api_key_env

        settings = _merge_json_objects(
            model_config.settings.extra_settings,
            runtime_config.model.extra_request_settings,
        )
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
            enable_provider_side_continuation=(
                runtime_config.model.enable_provider_side_continuation
            ),
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
        """Return the concrete Responses endpoint URL."""

        return _responses_endpoint_url(self._base_url)

    @property
    def secret_header_names(self) -> tuple[str, ...]:
        """Return request header names whose values must be redacted."""

        configured = [header.name for header in self._extra_headers if header.is_sensitive]
        configured.append(self._api_key_header_name)
        for name in DEFAULT_SECRET_HEADER_NAMES:
            configured.append(name)
        return tuple(sorted({name.lower() for name in configured if name}))

    def build_payload(self, request: ModelRequest) -> JsonObject:
        """Translate a provider-neutral request into an OpenAI Responses body."""

        model_name = request.model_name or self._model_name
        if not model_name:
            raise ConfigurationError("OpenAI Responses requests require a model name")

        reasoning = self._resolve_reasoning(request)
        max_output_tokens = self._resolve_max_output_tokens(request)
        settings = self._request_settings(request)
        temperature = _pop_optional_float_setting(settings, "temperature")
        tool_choice = _tool_choice_payload(
            _pop_tool_choice_request(settings, request.request_metadata)
        )
        self._validate_request_capabilities(
            request,
            reasoning=reasoning,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            force_tool_choice=_is_forced_tool_choice(tool_choice),
        )

        body: dict[str, object] = {
            "model": model_name,
            "input": _input_items_from_messages(request.messages),
        }
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens
        if reasoning is not None:
            reasoning_payload = _reasoning_payload(reasoning, self._profile)
            if reasoning_payload is not None:
                body["reasoning"] = reasoning_payload
                if _should_include_encrypted_reasoning_content(self._profile):
                    body["include"] = ["reasoning.encrypted_content"]
        if request.tools:
            body["tools"] = [_function_tool_payload(tool, self._profile) for tool in request.tools]
            if _can_request_serial_tool_calls(self._profile):
                body["parallel_tool_calls"] = False
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
        if temperature is not None:
            body["temperature"] = temperature

        _apply_extra_settings(body, settings, self._profile)
        previous_response_id = self._previous_response_id_for_request(request)
        if previous_response_id is not None:
            body["previous_response_id"] = previous_response_id
        return _json_object(body)

    def build_headers(self) -> dict[str, str]:
        """Return raw request headers for immediate HTTP use."""

        headers: dict[str, str] = {"Content-Type": "application/json"}
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
        """POST a translated Responses request and parse the provider payload."""

        http_request = self.build_http_request(request)
        http_response = await self._transport.post_json(http_request)
        return self.parse_response(http_response.body, request=request)

    def parse_response(
        self,
        body: JsonValue,
        *,
        request: ModelRequest | None = None,
    ) -> ModelResponse:
        """Translate an OpenAI Responses payload into ``ModelResponse``.

        The parser keeps OpenAI-specific item IDs, call IDs, native statuses,
        usage details, and reasoning continuation material in metadata fields so
        the turn loop remains provider-neutral.
        """

        payload = _response_payload(body)
        requested_reasoning = self._requested_reasoning_metadata(request)
        parsed = _parse_response_payload(
            payload,
            provider_name=self._provider_name,
            fallback_model_name=self._model_name,
            requested_reasoning=requested_reasoning,
            profile=self._profile,
            provider_side_enabled=self._enable_provider_side_continuation,
        )
        if request is None:
            return parsed
        return parsed.model_copy(update={"request_id": request.request_id}, deep=True)

    def _resolve_reasoning(self, request: ModelRequest) -> ReasoningSettings | None:
        # An explicit per-request disable signal overrides adapter defaults, so the
        # turn loop can drop reasoning for a forced re-ask even when the adapter was
        # configured with default reasoning.
        if _reasoning_disabled(request.request_metadata):
            return None
        if request.reasoning is not None:
            return request.reasoning.model_copy(deep=True)
        if self._default_reasoning is not None:
            return self._default_reasoning.model_copy(deep=True)
        if _should_default_minimal_reasoning(self._profile):
            return ReasoningSettings(effort=_OPENAI_DEFAULT_REASONING_EFFORT)
        if self._profile is None:
            return ReasoningSettings(effort=_OPENAI_DEFAULT_REASONING_EFFORT)
        return None

    def _requested_reasoning_metadata(
        self,
        request: ModelRequest | None,
    ) -> ReasoningSettings | None:
        if request is not None:
            return self._resolve_reasoning(request)
        if self._default_reasoning is not None:
            return self._default_reasoning.model_copy(deep=True)
        if _should_default_minimal_reasoning(self._profile) or self._profile is None:
            return ReasoningSettings(effort=_OPENAI_DEFAULT_REASONING_EFFORT)
        return None

    def _resolve_max_output_tokens(self, request: ModelRequest) -> int | None:
        requested = request.max_output_tokens or self._default_max_output_tokens
        if self._profile is None:
            return requested
        return resolve_max_output_tokens(self._profile, requested)

    def _request_settings(self, request: ModelRequest) -> JsonObject:
        settings = _copy_json_object(self._default_request_settings)
        metadata_settings = _metadata_settings(request.request_metadata)
        settings.update(metadata_settings)
        return settings

    def _validate_request_capabilities(
        self,
        request: ModelRequest,
        *,
        reasoning: ReasoningSettings | None,
        temperature: float | None,
        max_output_tokens: int | None,
        force_tool_choice: bool = False,
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
                max_output_tokens=max_output_tokens,
                force_tool_choice=force_tool_choice,
            ),
        )

    def _previous_response_id_for_request(self, request: ModelRequest) -> str | None:
        if self._profile is None:
            return None
        strategy = select_continuation_strategy(
            self._profile,
            provider_side_enabled=self._enable_provider_side_continuation,
        )
        if strategy is not ContinuationStrategy.PROVIDER_RESPONSE_ID:
            return None
        return _latest_previous_response_id(request)

    def _resolve_api_key(self) -> SecretValue | None:
        if self._api_key is not None:
            return self._api_key
        if self._api_key_env_var is None:
            return None
        return EnvironmentSecretSource(env_var=self._api_key_env_var).resolve(self._environment)


def _parse_response_payload(
    payload: JsonObject,
    *,
    provider_name: str,
    fallback_model_name: str,
    requested_reasoning: ReasoningSettings | None,
    profile: ModelProfile | None,
    provider_side_enabled: bool,
) -> ModelResponse:
    native_status = _optional_string(payload, "status") or "completed"
    completion_status = _completion_status(native_status)
    incomplete_details = _incomplete_details(payload)
    usage = _usage_from_payload(payload)
    parsed_output = _parse_output_items(payload)
    reasoning = _reasoning_metadata(
        payload,
        parsed_output=parsed_output,
        requested_reasoning=requested_reasoning,
        usage=usage,
    )
    provider_response_id = _optional_string(payload, "id")
    model_name = _optional_string(payload, "model") or fallback_model_name
    provider_metadata = _provider_metadata(
        payload,
        provider_name=provider_name,
        model_name=model_name,
        native_status=native_status,
        parsed_output=parsed_output,
        profile=profile,
        provider_side_enabled=provider_side_enabled,
    )
    response_metadata = _response_metadata(
        payload,
        native_status=native_status,
        parsed_output=parsed_output,
    )
    assistant_message = _assistant_message_from_texts(
        parsed_output.text_parts,
        include_text=completion_status is ProviderCompletionStatus.COMPLETED,
    )
    return ModelResponse(
        response_id=provider_response_id or new_id("model_resp"),
        assistant_message=assistant_message,
        tool_calls=parsed_output.tool_calls,
        stop_reason=_normalized_stop_reason(
            completion_status,
            incomplete_details=incomplete_details,
            has_final_text=assistant_message is not None,
        ),
        provider_completion_status=completion_status,
        incomplete_details=incomplete_details,
        usage=usage,
        reasoning=reasoning,
        provider_metadata=provider_metadata,
        response_metadata=response_metadata,
    )


def _empty_text_list() -> list[str]:
    return []


def _empty_tool_call_list() -> list[ToolCall]:
    return []


def _empty_provider_item_list() -> list[ProviderItemMetadata]:
    return []


def _empty_json_object_list() -> list[JsonObject]:
    return []


class _ParsedOutput(StrictModel):
    text_parts: list[str] = Field(default_factory=_empty_text_list)
    tool_calls: list[ToolCall] = Field(default_factory=_empty_tool_call_list)
    provider_items: list[ProviderItemMetadata] = Field(default_factory=_empty_provider_item_list)
    item_ids: list[str] = Field(default_factory=_empty_text_list)
    reasoning_items: list[JsonObject] = Field(default_factory=_empty_json_object_list)
    output_item_types: list[str] = Field(default_factory=_empty_text_list)
    unknown_output_item_types: list[str] = Field(default_factory=_empty_text_list)
    partial_text_present: bool = False


def _response_payload(body: JsonValue) -> JsonObject:
    if not isinstance(body, Mapping):
        raise ProviderProtocolError("OpenAI Responses payload must be a JSON object")
    return _json_object(cast(Mapping[str, object], body))


def _completion_status(status: str) -> ProviderCompletionStatus:
    if status == "completed":
        return ProviderCompletionStatus.COMPLETED
    if status == "incomplete":
        return ProviderCompletionStatus.INCOMPLETE
    if status == "failed":
        return ProviderCompletionStatus.FAILED
    raise ProviderProtocolError(f"OpenAI Responses status {status!r} is not supported")


def _incomplete_details(payload: JsonObject) -> JsonObject:
    details = _optional_object(payload, "incomplete_details")
    if details is not None:
        return details
    error = payload.get("error")
    if error is None:
        return {}
    return {"error": _json_value(error)}


def _parse_output_items(payload: JsonObject) -> _ParsedOutput:
    output = payload.get("output")
    if not isinstance(output, list):
        raise ProviderProtocolError("OpenAI Responses payload requires an output list")

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    provider_items: list[ProviderItemMetadata] = []
    item_ids: list[str] = []
    reasoning_items: list[JsonObject] = []
    output_item_types: list[str] = []
    unknown_types: list[str] = []

    for order, raw_item in enumerate(output):
        if not isinstance(raw_item, Mapping):
            raise ProviderProtocolError("OpenAI Responses output items must be JSON objects")
        item = _json_object(cast(Mapping[str, object], raw_item))
        item_type = _required_provider_string(item, "type", context="output item")
        output_item_types.append(item_type)
        provider_item_id = _optional_string(item, "id")
        if provider_item_id is not None:
            item_ids.append(provider_item_id)

        if item_type == "message":
            text_parts.extend(_text_parts_from_message_item(item))
            provider_items.append(
                ProviderItemMetadata(
                    kind=ProviderItemKind.RESPONSE,
                    order=order,
                    provider_item_id=provider_item_id,
                    status=_optional_string(item, "status"),
                    redacted_details=_item_details(item, include_content=False),
                )
            )
        elif item_type == "output_text":
            text = _provider_text(item, "text", context="output_text item")
            if text is not None:
                text_parts.append(text)
            provider_items.append(
                ProviderItemMetadata(
                    kind=ProviderItemKind.OUTPUT_TEXT,
                    order=order,
                    provider_item_id=provider_item_id,
                    status=_optional_string(item, "status"),
                    redacted_details=_item_details(item, include_content=False),
                )
            )
        elif item_type == "function_call":
            tool_call = _tool_call_from_function_item(item, order=len(tool_calls))
            tool_calls.append(tool_call)
            provider_items.append(
                ProviderItemMetadata(
                    kind=ProviderItemKind.FUNCTION_CALL,
                    order=order,
                    provider_item_id=tool_call.provider_item_id,
                    provider_call_id=tool_call.provider_call_id,
                    tool_name=tool_call.tool_name,
                    status=tool_call.provider_status,
                    redacted_details=_item_details(item, include_content=False),
                )
            )
        elif item_type == "reasoning":
            reasoning_items.append(item)
            provider_items.append(
                ProviderItemMetadata(
                    kind=ProviderItemKind.REASONING,
                    order=order,
                    provider_item_id=provider_item_id,
                    status=_optional_string(item, "status"),
                    encrypted_reasoning_content=_optional_string(item, "encrypted_content"),
                    redacted_details=_reasoning_item_details(item),
                )
            )
        else:
            unknown_types.append(item_type)

    return _ParsedOutput(
        text_parts=text_parts,
        tool_calls=tool_calls,
        provider_items=provider_items,
        item_ids=item_ids,
        reasoning_items=reasoning_items,
        output_item_types=output_item_types,
        unknown_output_item_types=unknown_types,
        partial_text_present=bool(text_parts),
    )


def _text_parts_from_message_item(item: JsonObject) -> list[str]:
    role = _optional_string(item, "role")
    if role is not None and role != "assistant":
        raise ProviderProtocolError(f"OpenAI Responses message role {role!r} is not supported")
    content = item.get("content")
    if not isinstance(content, list):
        raise ProviderProtocolError("OpenAI Responses message output requires a content list")
    parts: list[str] = []
    for raw_part in content:
        if not isinstance(raw_part, Mapping):
            raise ProviderProtocolError("OpenAI Responses message content parts must be objects")
        part = _json_object(cast(Mapping[str, object], raw_part))
        part_type = _optional_string(part, "type")
        if part_type == "output_text":
            text = _provider_text(part, "text", context="output_text content part")
            if text is not None:
                parts.append(text)
        elif part_type == "text":
            text = _provider_text(part, "text", context="text content part")
            if text is not None:
                parts.append(text)
    return parts


def _tool_call_from_function_item(item: JsonObject, *, order: int) -> ToolCall:
    provider_item_id = _optional_string(item, "id")
    provider_call_id = _required_provider_string(item, "call_id", context="function_call item")
    tool_name = _required_provider_string(item, "name", context="function_call item")
    arguments = _required_provider_string(item, "arguments", context="function_call item")
    try:
        return ToolCall.from_provider_arguments(
            call_id=provider_call_id,
            tool_name=tool_name,
            arguments=arguments,
            order=order,
            provider_item_id=provider_item_id,
            provider_call_id=provider_call_id,
            provider_status=_optional_string(item, "status"),
            provider_metadata={
                "openai_type": "function_call",
                "openai_raw_arguments": arguments,
            },
        )
    except (ValueError, ValidationError) as exc:
        raise ProviderProtocolError(
            f"OpenAI Responses function_call arguments for {tool_name!r} are invalid"
        ) from exc


def _assistant_message_from_texts(
    text_parts: list[str],
    *,
    include_text: bool,
) -> AssistantMessage | None:
    if not include_text:
        return None
    content: list[ContentPart] = [TextContent(text=text) for text in text_parts if text]
    if not content:
        return None
    return AssistantMessage(content=content)


def _usage_from_payload(payload: JsonObject) -> Usage:
    raw_usage = payload.get("usage")
    if raw_usage is None:
        return Usage()
    if not isinstance(raw_usage, Mapping):
        raise ProviderProtocolError("OpenAI Responses usage must be a JSON object")
    usage = _json_object(cast(Mapping[str, object], raw_usage))
    input_details = _optional_object(usage, "input_tokens_details") or {}
    output_details = _optional_object(usage, "output_tokens_details") or {}
    provider_details = _usage_provider_details(usage, input_details, output_details)
    total_input = _optional_non_negative_int(usage, "input_tokens") or 0
    cache_read = _optional_non_negative_int(input_details, "cached_tokens") or 0
    return Usage(
        tokens=TokenUsage(
            input_tokens=total_input - cache_read,
            output_tokens=_optional_non_negative_int(usage, "output_tokens") or 0,
            reasoning_tokens=_optional_non_negative_int(output_details, "reasoning_tokens") or 0,
            cache_read_tokens=cache_read,
            cache_write_tokens=_optional_non_negative_int(input_details, "cache_write_tokens") or 0,
            provider_details=provider_details,
        )
    )


def _usage_provider_details(
    usage: JsonObject,
    input_details: JsonObject,
    output_details: JsonObject,
) -> dict[str, int]:
    details: dict[str, int] = {}
    for key, value in usage.items():
        if key in {
            "input_tokens",
            "output_tokens",
            "input_tokens_details",
            "output_tokens_details",
        }:
            continue
        count = _json_non_negative_int(value, context=f"usage.{key}")
        if count is not None:
            details[key] = count
    for key, value in input_details.items():
        if key in {"cached_tokens", "cache_write_tokens"}:
            continue
        count = _json_non_negative_int(value, context=f"usage.input_tokens_details.{key}")
        if count is not None:
            details[f"input_tokens_details.{key}"] = count
    for key, value in output_details.items():
        if key == "reasoning_tokens":
            continue
        count = _json_non_negative_int(value, context=f"usage.output_tokens_details.{key}")
        if count is not None:
            details[f"output_tokens_details.{key}"] = count
    return details


def _reasoning_metadata(
    payload: JsonObject,
    *,
    parsed_output: _ParsedOutput,
    requested_reasoning: ReasoningSettings | None,
    usage: Usage,
) -> ReasoningMetadata | None:
    native_settings = _optional_object(payload, "reasoning") or {}
    summaries = _reasoning_summaries(parsed_output.reasoning_items)
    continuation = _reasoning_continuation(parsed_output.reasoning_items)
    reasoning_tokens = usage.tokens.reasoning_tokens or None
    has_reasoning_metadata = bool(
        requested_reasoning or native_settings or summaries or continuation or reasoning_tokens
    )
    if not has_reasoning_metadata:
        return None
    return ReasoningMetadata(
        requested=(
            requested_reasoning.model_copy(deep=True)
            if requested_reasoning is not None
            else None
        ),
        observed_effort=_optional_string(native_settings, "effort"),
        native_settings=native_settings,
        summaries=summaries,
        reasoning_tokens=reasoning_tokens,
        provider_private_continuation=continuation,
        display_policy=(
            requested_reasoning.display_policy
            if requested_reasoning is not None
            else ReasoningSettings().display_policy
        ),
    )


def _reasoning_summaries(items: list[JsonObject]) -> list[ReasoningSummary]:
    summaries: list[ReasoningSummary] = []
    for item in items:
        provider_item_id = _optional_string(item, "id")
        raw_summary = item.get("summary")
        if raw_summary is None:
            continue
        if isinstance(raw_summary, str):
            if raw_summary:
                summaries.append(
                    ReasoningSummary(text=raw_summary, provider_item_id=provider_item_id)
                )
            continue
        if not isinstance(raw_summary, list):
            raise ProviderProtocolError("OpenAI Responses reasoning summary must be a list")
        for raw_part in raw_summary:
            text = _reasoning_summary_text(raw_part)
            if text:
                summaries.append(ReasoningSummary(text=text, provider_item_id=provider_item_id))
    return summaries


def _reasoning_summary_text(value: object) -> str | None:
    if isinstance(value, str):
        return value or None
    if not isinstance(value, Mapping):
        raise ProviderProtocolError("OpenAI Responses reasoning summary entries must be objects")
    part = _json_object(cast(Mapping[str, object], value))
    text = _optional_string(part, "text")
    if text is None:
        return None
    return text


def _reasoning_continuation(items: list[JsonObject]) -> list[ReasoningContinuationMetadata]:
    continuation: list[ReasoningContinuationMetadata] = []
    for order, item in enumerate(items):
        encrypted_content = _optional_string(item, "encrypted_content")
        if encrypted_content is None:
            continue
        continuation.append(
            ReasoningContinuationMetadata(
                provider_name="openai",
                kind="reasoning",
                order=order,
                provider_item_id=_optional_string(item, "id"),
                encrypted_content=encrypted_content,
                redacted_details=_reasoning_item_details(item),
            )
        )
    return continuation


def _provider_metadata(
    payload: JsonObject,
    *,
    provider_name: str,
    model_name: str,
    native_status: str,
    parsed_output: _ParsedOutput,
    profile: ModelProfile | None,
    provider_side_enabled: bool,
) -> ProviderMetadata:
    strategy = _continuation_strategy(profile, provider_side_enabled=provider_side_enabled)
    provider_response_id = _optional_string(payload, "id")
    return ProviderMetadata(
        provider_name=provider_name,
        model_name=model_name,
        response_id=provider_response_id,
        previous_response_id=_optional_string(payload, "previous_response_id"),
        native_stop_reason=native_status,
        item_ids=parsed_output.item_ids,
        items=parsed_output.provider_items,
        continuation_strategy=strategy,
        provider_side_continuation_available=(
            strategy is ContinuationStrategy.PROVIDER_RESPONSE_ID
            and provider_response_id is not None
        ),
        stateless_continuation_required=_stateless_continuation_required(profile, strategy),
        redacted_raw_details=_provider_raw_details(
            payload,
            parsed_output=parsed_output,
            native_status=native_status,
        ),
    )


def _continuation_strategy(
    profile: ModelProfile | None,
    *,
    provider_side_enabled: bool,
) -> ContinuationStrategy:
    if profile is None:
        return ContinuationStrategy.STATELESS_REPLAY
    return select_continuation_strategy(
        profile,
        provider_side_enabled=provider_side_enabled,
    )


def _stateless_continuation_required(
    profile: ModelProfile | None,
    strategy: ContinuationStrategy,
) -> bool:
    if profile is None:
        return strategy is ContinuationStrategy.STATELESS_REPLAY
    return (
        profile.continuation.stateless_continuation_required
        or strategy is ContinuationStrategy.STATELESS_REPLAY
    )


def _response_metadata(
    payload: JsonObject,
    *,
    native_status: str,
    parsed_output: _ParsedOutput,
) -> JsonObject:
    metadata: dict[str, object] = {
        "openai_status": native_status,
        "output_item_types": list(parsed_output.output_item_types),
    }
    object_value = _optional_string(payload, "object")
    if object_value is not None:
        metadata["openai_object"] = object_value
    created_at = _optional_number(payload, "created_at")
    if created_at is not None:
        metadata["created_at"] = created_at
    if parsed_output.unknown_output_item_types:
        metadata["unknown_output_item_types"] = list(parsed_output.unknown_output_item_types)
    if parsed_output.partial_text_present and native_status != "completed":
        metadata["partial_text_omitted_due_to_incomplete_status"] = True
    return _json_object(metadata)


def _provider_raw_details(
    payload: JsonObject,
    *,
    parsed_output: _ParsedOutput,
    native_status: str,
) -> JsonObject:
    details: dict[str, object] = {
        "status": native_status,
        "output_item_types": list(parsed_output.output_item_types),
    }
    for key in (
        "object",
        "created_at",
        "parallel_tool_calls",
        "store",
        "service_tier",
    ):
        if key in payload:
            details[key] = _json_value(payload[key])
    incomplete = _optional_object(payload, "incomplete_details")
    if incomplete is not None:
        details["incomplete_details"] = incomplete
    error = payload.get("error")
    if error is not None:
        details["error"] = _json_value(error)
    if parsed_output.unknown_output_item_types:
        details["unknown_output_item_types"] = list(parsed_output.unknown_output_item_types)
    return _json_object(details)


def _item_details(item: JsonObject, *, include_content: bool) -> JsonObject:
    details: dict[str, object] = {}
    for key in ("type", "role", "status"):
        if key in item:
            details[key] = _json_value(item[key])
    if include_content and "content" in item:
        details["content"] = _json_value(item["content"])
    return _json_object(details)


def _reasoning_item_details(item: JsonObject) -> JsonObject:
    details = _item_details(item, include_content=False)
    if "summary" in item:
        raw_summary = item["summary"]
        if isinstance(raw_summary, list):
            details["summary_count"] = len(raw_summary)
        elif isinstance(raw_summary, str):
            details["summary_count"] = 1 if raw_summary else 0
    details["has_encrypted_content"] = _optional_string(item, "encrypted_content") is not None
    return _json_object(details)


def _normalized_stop_reason(
    completion_status: ProviderCompletionStatus,
    *,
    incomplete_details: JsonObject,
    has_final_text: bool,
) -> StopReason:
    if completion_status is ProviderCompletionStatus.INCOMPLETE:
        reason = _optional_string(incomplete_details, "reason")
        if reason in {"max_output_tokens", "max_tokens"}:
            return StopReason.MAX_TOKENS
        return StopReason.PROVIDER_STOP_REASON
    if completion_status is ProviderCompletionStatus.FAILED:
        return StopReason.MODEL_ERROR
    if has_final_text:
        return StopReason.FINAL_RESPONSE
    return StopReason.PROVIDER_STOP_REASON


def _optional_string(value: JsonObject, key: str) -> str | None:
    raw = value.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ProviderProtocolError(f"OpenAI Responses field {key!r} must be a string")
    if not raw:
        return None
    return raw


def _provider_text(value: JsonObject, key: str, *, context: str) -> str | None:
    raw = value.get(key)
    if not isinstance(raw, str):
        raise ProviderProtocolError(f"OpenAI Responses {context} requires string {key!r}")
    if raw == "":
        return None
    return raw


def _required_provider_string(value: JsonObject, key: str, *, context: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw:
        raise ProviderProtocolError(f"OpenAI Responses {context} requires non-empty {key!r}")
    return raw


def _optional_object(value: JsonObject, key: str) -> JsonObject | None:
    raw = value.get(key)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ProviderProtocolError(f"OpenAI Responses field {key!r} must be an object")
    return _json_object(cast(Mapping[str, object], raw))


def _optional_number(value: JsonObject, key: str) -> int | float | None:
    raw = value.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ProviderProtocolError(f"OpenAI Responses field {key!r} must be a number")
    return raw


def _optional_non_negative_int(value: JsonObject, key: str) -> int | None:
    return _json_non_negative_int(value.get(key), context=f"OpenAI Responses field {key!r}")


def _json_non_negative_int(value: JsonValue | object, *, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProviderProtocolError(f"{context} must be a non-negative integer")
    if value < 0:
        raise ProviderProtocolError(f"{context} must be a non-negative integer")
    return value


def _validate_openai_profile(
    profile: ModelProfile,
    *,
    model_name: str,
    provider_name: str,
) -> None:
    if profile.api is not ProviderApi.OPENAI_RESPONSES:
        raise ConfigurationError("OpenAI Responses adapter requires an openai_responses profile")
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


def _responses_endpoint_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith(f"/{_DEFAULT_RESPONSES_PATH}"):
        return stripped
    return f"{stripped}/{_DEFAULT_RESPONSES_PATH}"


def _base_url_from_config(
    model_config: AgentModelConfig,
    runtime_config: ProviderRuntimeConfig,
    environment: Mapping[str, str],
) -> str:
    if runtime_config.model.base_url is not None:
        return runtime_config.model.base_url
    if model_config.endpoint is not None:
        return model_config.endpoint
    env_name = runtime_config.api_key_sources.openai_base_url_env
    env_value = environment.get(env_name)
    if env_value:
        return env_value
    return _DEFAULT_OPENAI_BASE_URL


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


def _input_items_from_messages(messages: Iterable[ModelMessage]) -> list[JsonObject]:
    items: list[JsonObject] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            _append_role_message(items, role="system", content=_message_content_text(message))
        elif isinstance(message, DeveloperMessage):
            _append_role_message(items, role="developer", content=_message_content_text(message))
        elif isinstance(message, UserMessage):
            _append_role_message(items, role="user", content=_message_content_text(message))
        elif isinstance(message, AssistantMessage):
            content = _message_content_text(message)
            if content:
                _append_role_message(items, role="assistant", content=content)
            for tool_call in sorted(assistant_tool_calls(message), key=lambda call: call.order):
                items.append(_function_call_input_item(tool_call))
        else:
            items.append(_function_call_output_input_item(message))
    return items


def _append_role_message(items: list[JsonObject], *, role: str, content: str) -> None:
    if not content:
        return
    items.append(_json_object({"role": role, "content": content}))


def _message_content_text(
    message: SystemMessage | DeveloperMessage | UserMessage | AssistantMessage,
) -> str:
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


def _function_call_input_item(tool_call: ToolCall) -> JsonObject:
    call_id = _openai_call_id_from_tool_call(tool_call)
    item: dict[str, object] = {
        "type": "function_call",
        "call_id": call_id,
        "name": tool_call.tool_name,
        "arguments": _openai_arguments_json_from_tool_call(tool_call),
        "status": tool_call.provider_status or "completed",
    }
    if tool_call.provider_item_id is not None:
        item["id"] = tool_call.provider_item_id
    return _json_object(item)


def _function_call_output_input_item(message: ToolResultMessage) -> JsonObject:
    result = message.result
    call_id = result.provider_call_id or result.tool_call_id
    content_text = _tool_message_content_text(message)
    output = content_text if content_text else model_visible_tool_result_text(result)
    return _json_object(
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        }
    )


def _tool_message_content_text(message: ToolResultMessage) -> str:
    parts: list[str] = []
    for part in message.content:
        if isinstance(part, TextContent):
            parts.append(part.text)
        else:
            parts.append(_compaction_summary_text(part))
    return "\n\n".join(parts)


def _openai_call_id_from_tool_call(tool_call: ToolCall) -> str:
    return tool_call.provider_call_id or tool_call.call_id


def _arguments_json(arguments: JsonObject) -> str:
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"))


def _openai_arguments_json_from_tool_call(tool_call: ToolCall) -> str:
    raw_arguments = tool_call.provider_metadata.get("openai_raw_arguments")
    if isinstance(raw_arguments, str) and raw_arguments:
        return raw_arguments
    return _arguments_json(tool_call.arguments)


def _function_tool_payload(tool: JsonObject, profile: ModelProfile | None) -> JsonObject:
    name = _required_string(tool, "name", context="tool definition")
    description = _required_string(tool, "description", context=f"tool {name!r}")
    parameters = _required_object(tool, "arguments_schema", context=f"tool {name!r}")
    strict = _supports_strict_tool_schemas(profile)
    payload: dict[str, object] = {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": (
            _strict_tool_parameters(parameters) if strict else _copy_json_object(parameters)
        ),
    }
    if strict:
        payload["strict"] = True
    return _json_object(payload)


def _strict_tool_parameters(parameters: JsonObject) -> JsonObject:
    """Return OpenAI strict-mode compatible tool parameters.

    OpenAI strict function schemas require every object schema with
    ``properties`` to include a ``required`` array containing all property keys.
    Canonical tend tool schemas preserve Pydantic defaults, so defaulted
    arguments may be omitted from ``required`` until this provider-specific
    translation step.
    """

    return _JSON_OBJECT_ADAPTER.validate_python(_strict_tool_schema_value(parameters))


def _strict_tool_schema_value(value: object) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        copied: dict[str, object] = {
            str(key): _strict_tool_schema_value(child) for key, child in mapping.items()
        }
        properties = copied.get("properties")
        if isinstance(properties, Mapping):
            property_map = cast(Mapping[str, object], properties)
            copied["required"] = [str(key) for key in property_map.keys()]
        return copied
    if isinstance(value, list):
        items = cast(list[object], value)
        return [_strict_tool_schema_value(item) for item in items]
    return deepcopy(value)


def _supports_strict_tool_schemas(profile: ModelProfile | None) -> bool:
    if profile is None:
        return True
    return profile.tools.supports_strict_tool_schemas


def _can_request_serial_tool_calls(profile: ModelProfile | None) -> bool:
    if profile is None:
        return True
    return profile.tools.can_request_serial_tool_calls


def _reasoning_payload(
    reasoning: ReasoningSettings,
    profile: ModelProfile | None,
) -> JsonObject | None:
    """Build the OpenAI Responses ``reasoning`` body field.

    Returns ``None`` when ``thinking_level_map`` remaps the requested effort
    to "off" (i.e., the entry is present with a value of ``None``). In that
    case the caller should omit the reasoning block entirely; matches pi's
    handling of an explicit off-mapping.
    """

    effort = reasoning.effort
    if effort is not None and profile is not None:
        mapping = profile.reasoning.thinking_level_map
        if effort in mapping:
            mapped = mapping[effort]
            if mapped is None:
                return None
            effort = mapped

    payload: dict[str, object] = {}
    if effort is not None:
        payload["effort"] = effort.value
    # OpenAI Responses defaults: always request a reasoning summary
    # (``summary: "auto"``) unless the caller explicitly opts out via
    # ``ReasoningSummaryPreference.NONE``. ``None`` (unset) gets the default.
    if reasoning.summary is None:
        payload["summary"] = ReasoningSummaryPreference.AUTO.value
    elif reasoning.summary is not ReasoningSummaryPreference.NONE:
        payload["summary"] = reasoning.summary.value
    payload.update(_copy_json_object(reasoning.native_settings))
    if not payload:
        payload["effort"] = _OPENAI_DEFAULT_REASONING_EFFORT.value
        if reasoning.summary is not ReasoningSummaryPreference.NONE:
            payload["summary"] = ReasoningSummaryPreference.AUTO.value
    return _json_object(payload)


def _should_include_encrypted_reasoning_content(profile: ModelProfile | None) -> bool:
    """Whether to add ``include: ["reasoning.encrypted_content"]`` to the body.

    Mirrors pi's default for reasoning-capable models. Profiles that do not
    expose encrypted continuation (``supports_encrypted_reasoning_content``
    False) opt out automatically; with no profile, we still emit it so
    stateless callers can persist continuation material.
    """

    if profile is None:
        return True
    return profile.reasoning.supports_encrypted_reasoning_content


def _should_default_minimal_reasoning(profile: ModelProfile | None) -> bool:
    if profile is None:
        return True
    return (
        profile.reasoning.supports_reasoning
        and _OPENAI_DEFAULT_REASONING_EFFORT in profile.reasoning.supported_efforts
    )


def _reasoning_disabled(metadata: JsonObject) -> bool:
    # Produced by the turn loop (and honoured by every adapter) to suppress
    # reasoning/thinking for a single request regardless of adapter defaults.
    return metadata.get("disable_reasoning") is True


def _pop_tool_choice_request(settings: JsonObject, metadata: JsonObject) -> JsonValue | None:
    value: JsonValue | None = settings.pop("tool_choice", None)
    for key in ("provider_tool_choice", "openai_tool_choice", "tool_choice"):
        if key in metadata:
            value = metadata[key]
    force_tool_name = metadata.get("force_tool_name")
    if isinstance(force_tool_name, str) and force_tool_name:
        return {"type": "function", "name": force_tool_name}
    return value


def _tool_choice_payload(value: JsonValue | None) -> JsonValue | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value not in {"auto", "none", "required"}:
            raise ConfigurationError(
                "OpenAI Responses tool_choice string must be auto, none, or required"
            )
        return value
    if not isinstance(value, Mapping):
        raise ConfigurationError("OpenAI Responses tool_choice must be a string or object")
    payload = _json_object(cast(Mapping[str, object], value))
    choice_type = payload.get("type")
    if not isinstance(choice_type, str) or not choice_type:
        raise ConfigurationError("OpenAI Responses tool_choice requires a non-empty type")
    if choice_type == "function":
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigurationError("OpenAI Responses tool_choice type 'function' requires a name")
    return payload


def _is_forced_tool_choice(tool_choice: JsonValue | None) -> bool:
    if isinstance(tool_choice, str):
        return tool_choice == "required"
    if isinstance(tool_choice, Mapping):
        return tool_choice.get("type") == "function"
    return False


def _metadata_settings(metadata: JsonObject) -> JsonObject:
    merged: JsonObject = {}
    for key in ("provider_request_settings", "openai_responses_request_settings"):
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
        raise ConfigurationError(f"OpenAI Responses setting {key!r} must be a number")
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
            f"OpenAI Responses request setting {key!r}"
        )


def _latest_previous_response_id(request: ModelRequest) -> str | None:
    if request.provider_metadata is not None:
        direct = (
            request.provider_metadata.previous_response_id
            or request.provider_metadata.response_id
        )
        if direct is not None:
            return direct
    for message in reversed(request.messages):
        if not isinstance(message, AssistantMessage):
            continue
        from_provider_metadata = _response_id_from_assistant_provider_metadata(message)
        if from_provider_metadata is not None:
            return from_provider_metadata
        direct_value = message.provider_metadata.get(ASSISTANT_MODEL_RESPONSE_ID_METADATA_KEY)
        if isinstance(direct_value, str) and direct_value:
            return direct_value
    return None


def _response_id_from_assistant_provider_metadata(message: AssistantMessage) -> str | None:
    raw_metadata = message.provider_metadata.get(ASSISTANT_PROVIDER_METADATA_KEY)
    if isinstance(raw_metadata, Mapping):
        try:
            metadata = ProviderMetadata.model_validate(raw_metadata)
        except ValidationError:
            direct = raw_metadata.get("response_id")
            if isinstance(direct, str) and direct:
                return direct
            return None
        return metadata.response_id
    return None


def _format_secret_header_value(value: str, *, scheme: str | None) -> str:
    if scheme is None or not scheme:
        return value
    if value.lower().startswith(f"{scheme.lower()} "):
        return value
    return f"{scheme} {value}"


def _required_string(value: JsonObject, key: str, *, context: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw:
        raise ConfigurationError(f"OpenAI Responses {context} requires non-empty {key!r}")
    return raw


def _required_object(value: JsonObject, key: str, *, context: str) -> JsonObject:
    raw = value.get(key)
    if not isinstance(raw, Mapping):
        raise ConfigurationError(f"OpenAI Responses {context} requires object {key!r}")
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


__all__ = ("OpenAIResponsesAdapter",)
