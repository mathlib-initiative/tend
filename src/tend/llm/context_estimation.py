"""Simple provider-neutral context/token estimation helpers.

The estimates in this module are intentionally approximate. They are used for
observability and early compaction decisions before provider-specific token
counting exists; provider-reported usage remains authoritative when available.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from math import ceil
from typing import Annotated

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from tend._common.types import JsonObject, StrictModel
from tend.llm.models.messages import AssistantMessage, ContentPart, TextContent
from tend.llm.models.profiles import ModelProfile
from tend.llm.models.reasoning import ReasoningSettings
from tend.llm.models.requests import ModelMessage, ModelRequest
from tend.llm.models.tools import ToolCall, ToolResultMessage

CONTEXT_ESTIMATE_METADATA_KEY = "context_estimate"
_ASSISTANT_TOOL_CALLS_METADATA_KEY = "tool_calls"
_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)

_NonNegativeInt = Annotated[int, Field(ge=0)]
_PositiveFloat = Annotated[float, Field(gt=0)]


def _empty_json_object() -> JsonObject:
    return {}


class TokenEstimatorConfig(StrictModel):
    """Knobs for the built-in simple token estimator.

    ``chars_per_token`` is the main approximation. Small fixed overheads make
    role/message/tool boundaries visible enough for compaction heuristics while
    staying provider-agnostic.
    """

    chars_per_token: _PositiveFloat = 2.0
    tokens_per_message: _NonNegativeInt = 4
    tokens_per_content_part: _NonNegativeInt = 1
    tokens_per_tool_call: _NonNegativeInt = 8
    tokens_per_tool_result: _NonNegativeInt = 4
    tokens_per_tool_schema: _NonNegativeInt = 8
    tokens_per_reasoning_settings: _NonNegativeInt = 2
    estimator_name: str = Field(default="simple_chars", min_length=1)


class TokenEstimatorConfigOverrides(StrictModel):
    """Partial overrides for :class:`TokenEstimatorConfig`."""

    chars_per_token: _PositiveFloat | None = None
    tokens_per_message: _NonNegativeInt | None = None
    tokens_per_content_part: _NonNegativeInt | None = None
    tokens_per_tool_call: _NonNegativeInt | None = None
    tokens_per_tool_result: _NonNegativeInt | None = None
    tokens_per_tool_schema: _NonNegativeInt | None = None
    tokens_per_reasoning_settings: _NonNegativeInt | None = None
    estimator_name: str | None = Field(default=None, min_length=1)


class ContextEstimate(StrictModel):
    """Approximate active-context size for a provider-neutral request."""

    estimated_tokens: _NonNegativeInt
    message_tokens: _NonNegativeInt
    tool_schema_tokens: _NonNegativeInt = 0
    reasoning_setting_tokens: _NonNegativeInt = 0
    context_window_tokens: _NonNegativeInt | None = None
    remaining_context_tokens: _NonNegativeInt | None = None
    context_usage_ratio: float | None = Field(default=None, ge=0)
    context_usage_percent: float | None = Field(default=None, ge=0)
    estimator: str = Field(default="simple_chars", min_length=1)
    is_estimate: bool = True
    # Explicit producer provenance; unlike ``estimator``, this cannot be forged
    # through the user-configurable TokenEstimatorConfig.estimator_name.
    is_api_anchored: bool = False
    metadata: JsonObject = Field(default_factory=_empty_json_object)

    @model_validator(mode="after")
    def _validate_parts_sum(self) -> ContextEstimate:
        parts_total = self.message_tokens + self.tool_schema_tokens + self.reasoning_setting_tokens
        if self.estimated_tokens < parts_total:
            raise ValueError("estimated_tokens must be at least the sum of component estimates")
        return self


class ContextEstimateParts(StrictModel):
    """Intermediate token-estimate components before profile window metadata."""

    message_tokens: _NonNegativeInt
    tool_schema_tokens: _NonNegativeInt = 0
    reasoning_setting_tokens: _NonNegativeInt = 0

    @property
    def total_tokens(self) -> int:
        """Return the sum of all estimate components."""

        return self.message_tokens + self.tool_schema_tokens + self.reasoning_setting_tokens


def estimate_text_tokens(text: str, config: TokenEstimatorConfig | None = None) -> int:
    """Estimate tokens for plain text using the configured chars/token ratio."""

    estimator = config or TokenEstimatorConfig()
    if not text:
        return 0
    return ceil(len(text) / estimator.chars_per_token)


def estimate_content_part_tokens(
    part: ContentPart,
    config: TokenEstimatorConfig | None = None,
) -> int:
    """Estimate tokens for one provider-neutral content part."""

    estimator = config or TokenEstimatorConfig()
    if isinstance(part, TextContent):
        return estimator.tokens_per_content_part + estimate_text_tokens(part.text, estimator)
    covered_id_tokens = sum(
        estimate_text_tokens(item, estimator) for item in part.covered_message_ids
    )
    return (
        estimator.tokens_per_content_part
        + estimate_text_tokens(part.summary, estimator)
        + covered_id_tokens
    )


def estimate_message_tokens(
    message: ModelMessage,
    config: TokenEstimatorConfig | None = None,
) -> int:
    """Estimate tokens for one provider-neutral message."""

    estimator = config or TokenEstimatorConfig()
    content_tokens = sum(estimate_content_part_tokens(part, estimator) for part in message.content)
    total = (
        estimator.tokens_per_message
        + estimate_text_tokens(message.role.value, estimator)
        + content_tokens
    )
    if isinstance(message, AssistantMessage):
        total += _estimate_assistant_tool_call_metadata(message, estimator)
    elif isinstance(message, ToolResultMessage):
        total += estimator.tokens_per_tool_result
    return total


def estimate_messages_tokens(
    messages: Iterable[ModelMessage],
    config: TokenEstimatorConfig | None = None,
) -> int:
    """Estimate tokens for provider-neutral messages."""

    estimator = config or TokenEstimatorConfig()
    return sum(estimate_message_tokens(message, estimator) for message in messages)


def estimate_tool_schema_tokens(
    tool_schema: JsonObject,
    config: TokenEstimatorConfig | None = None,
) -> int:
    """Estimate tokens for one provider-neutral tool schema."""

    estimator = config or TokenEstimatorConfig()
    return estimator.tokens_per_tool_schema + _estimate_json_tokens(tool_schema, estimator)


def estimate_tool_schemas_tokens(
    tool_schemas: Iterable[JsonObject],
    config: TokenEstimatorConfig | None = None,
) -> int:
    """Estimate tokens for all tool schemas included with a request."""

    estimator = config or TokenEstimatorConfig()
    return sum(estimate_tool_schema_tokens(schema, estimator) for schema in tool_schemas)


def estimate_reasoning_settings_tokens(
    reasoning: ReasoningSettings | None,
    config: TokenEstimatorConfig | None = None,
) -> int:
    """Estimate small request overhead for explicit reasoning settings."""

    if reasoning is None:
        return 0
    estimator = config or TokenEstimatorConfig()
    return estimator.tokens_per_reasoning_settings + _estimate_json_tokens(
        _JSON_OBJECT_ADAPTER.validate_python(reasoning.model_dump(mode="json")),
        estimator,
    )


def estimate_context_parts(
    *,
    messages: Iterable[ModelMessage],
    tools: Iterable[JsonObject] = (),
    reasoning: ReasoningSettings | None = None,
    config: TokenEstimatorConfig | None = None,
) -> ContextEstimateParts:
    """Estimate request context components before applying profile metadata."""

    estimator = config or TokenEstimatorConfig()
    return ContextEstimateParts(
        message_tokens=estimate_messages_tokens(messages, estimator),
        tool_schema_tokens=estimate_tool_schemas_tokens(tools, estimator),
        reasoning_setting_tokens=estimate_reasoning_settings_tokens(reasoning, estimator),
    )


def estimate_context(
    *,
    messages: Iterable[ModelMessage],
    tools: Iterable[JsonObject] = (),
    reasoning: ReasoningSettings | None = None,
    profile: ModelProfile | None = None,
    config: TokenEstimatorConfig | None = None,
    metadata: JsonObject | None = None,
) -> ContextEstimate:
    """Estimate active request context and include profile-window percentages."""

    estimator = config or TokenEstimatorConfig()
    parts = estimate_context_parts(
        messages=messages,
        tools=tools,
        reasoning=reasoning,
        config=estimator,
    )
    return _estimate_from_parts(parts, profile=profile, config=estimator, metadata=metadata)


def estimate_model_request_context(
    request: ModelRequest,
    *,
    profile: ModelProfile | None = None,
    config: TokenEstimatorConfig | None = None,
    metadata: JsonObject | None = None,
) -> ContextEstimate:
    """Estimate context tokens for a complete provider-neutral model request."""

    return estimate_context(
        messages=request.messages,
        tools=request.tools,
        reasoning=request.reasoning,
        profile=profile,
        config=config,
        metadata=metadata,
    )


def estimate_context_from_api_anchor(
    *,
    anchor_tokens: int,
    new_messages: Iterable[ModelMessage],
    profile: ModelProfile | None = None,
    config: TokenEstimatorConfig | None = None,
    metadata: JsonObject | None = None,
) -> ContextEstimate:
    """Estimate active context using API-reported totals as an anchor.

    ``anchor_tokens`` is ``input_tokens + cache_read_tokens + cache_write_tokens
    + output_tokens`` from the previous response — the exact context size the
    model processed plus what it wrote.  ``new_messages`` are messages appended
    since then (typically tool results); their sizes are estimated conservatively
    using ``chars_per_token``.  Tool schemas and stable history are already
    captured by the anchor, so they don't need re-estimation.
    """
    estimator = config or TokenEstimatorConfig()
    new_msg_tokens = estimate_messages_tokens(new_messages, estimator)
    estimated_tokens = anchor_tokens + new_msg_tokens
    # All tokens are collapsed into message_tokens; tool_schema and reasoning
    # overhead are already baked into the anchor from the previous response.
    parts = ContextEstimateParts(message_tokens=estimated_tokens)
    base = _estimate_from_parts(parts, profile=profile, config=estimator, metadata=metadata)
    return base.model_copy(update={"estimator": "api_anchor", "is_api_anchored": True})


def context_estimate_to_metadata(estimate: ContextEstimate) -> JsonObject:
    """Return a JSON-compatible payload for request/event metadata."""

    return _JSON_OBJECT_ADAPTER.validate_python(estimate.model_dump(mode="json"))


def context_estimate_from_metadata(value: object) -> ContextEstimate | None:
    """Parse a context-estimate metadata value, returning ``None`` when absent."""

    if value is None:
        return None
    return ContextEstimate.model_validate(value)


def _estimate_from_parts(
    parts: ContextEstimateParts,
    *,
    profile: ModelProfile | None,
    config: TokenEstimatorConfig,
    metadata: JsonObject | None,
) -> ContextEstimate:
    estimated_tokens = parts.total_tokens
    context_window_tokens: int | None = None
    remaining_context_tokens: int | None = None
    usage_ratio: float | None = None
    usage_percent: float | None = None
    estimate_metadata: JsonObject = {} if metadata is None else metadata.copy()
    if profile is not None:
        estimate_metadata.setdefault("model_name", profile.model_name)
        estimate_metadata.setdefault("provider_name", profile.provider_name)
        if profile.context_window is not None:
            context_window_tokens = profile.context_window.tokens
            remaining_context_tokens = profile.context_window.remaining_tokens(estimated_tokens)
            usage_ratio = profile.context_window.usage_ratio(estimated_tokens)
            usage_percent = usage_ratio * 100.0
    return ContextEstimate(
        estimated_tokens=estimated_tokens,
        message_tokens=parts.message_tokens,
        tool_schema_tokens=parts.tool_schema_tokens,
        reasoning_setting_tokens=parts.reasoning_setting_tokens,
        context_window_tokens=context_window_tokens,
        remaining_context_tokens=remaining_context_tokens,
        context_usage_ratio=usage_ratio,
        context_usage_percent=usage_percent,
        estimator=config.estimator_name,
        metadata=estimate_metadata,
    )


def _estimate_assistant_tool_call_metadata(
    message: AssistantMessage,
    config: TokenEstimatorConfig,
) -> int:
    value = message.provider_metadata.get(_ASSISTANT_TOOL_CALLS_METADATA_KEY)
    if value is None:
        return 0
    if not isinstance(value, list):
        return _estimate_json_tokens({_ASSISTANT_TOOL_CALLS_METADATA_KEY: value}, config)

    total = 0
    for item in value:
        if isinstance(item, Mapping):
            try:
                tool_call = ToolCall.model_validate(item)
            except ValidationError:
                total += _estimate_json_tokens(
                    _JSON_OBJECT_ADAPTER.validate_python(dict(item)),
                    config,
                )
            else:
                total += estimate_tool_call_tokens(tool_call, config)
        else:
            total += estimate_text_tokens(str(item), config)
    return total


def estimate_tool_call_tokens(
    tool_call: ToolCall,
    config: TokenEstimatorConfig | None = None,
) -> int:
    """Estimate tokens for a provider-neutral assistant tool call."""

    estimator = config or TokenEstimatorConfig()
    return (
        estimator.tokens_per_tool_call
        + estimate_text_tokens(tool_call.tool_name, estimator)
        + _estimate_json_tokens(tool_call.arguments, estimator)
    )


def _estimate_json_tokens(value: JsonObject, config: TokenEstimatorConfig) -> int:
    if not value:
        return 0
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return estimate_text_tokens(text, config)


__all__ = (
    "CONTEXT_ESTIMATE_METADATA_KEY",
    "ContextEstimate",
    "ContextEstimateParts",
    "TokenEstimatorConfig",
    "TokenEstimatorConfigOverrides",
    "context_estimate_from_metadata",
    "context_estimate_to_metadata",
    "estimate_content_part_tokens",
    "estimate_context",
    "estimate_context_from_api_anchor",
    "estimate_context_parts",
    "estimate_message_tokens",
    "estimate_messages_tokens",
    "estimate_model_request_context",
    "estimate_reasoning_settings_tokens",
    "estimate_text_tokens",
    "estimate_tool_call_tokens",
    "estimate_tool_schema_tokens",
    "estimate_tool_schemas_tokens",
)
