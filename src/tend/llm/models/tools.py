"""Provider-neutral tool-call and tool-result schemas."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import Field, JsonValue, TypeAdapter, model_validator

from tend._common.errors import ErrorInfo
from tend._common.types import JsonObject, StrictModel, new_id
from tend.llm.models.messages import ContentPart, MessageRole, TextContent
from tend.llm.truncation import TruncationInfo

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)
_NonNegativeOrder = Annotated[int, Field(ge=0)]
_NonNegativeDuration = Annotated[float, Field(ge=0)]


class ToolCall(StrictModel):
    """Provider-neutral assistant-requested tool call.

    Provider adapters normalize native argument payloads before constructing this
    model, so ``arguments`` is always a JSON object in core code. OpenAI
    Responses ``function_call.id`` and ``call_id`` plus Anthropic ``tool_use.id``
    are preserved in explicit metadata fields for later stateless continuation.
    """

    call_id: str = Field(default_factory=lambda: new_id("call"), min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: JsonObject = Field(default_factory=dict)
    order: _NonNegativeOrder = 0
    provider_item_id: str | None = Field(default=None, min_length=1)
    provider_call_id: str | None = Field(default=None, min_length=1)
    provider_tool_use_id: str | None = Field(default=None, min_length=1)
    provider_status: str | None = Field(default=None, min_length=1)
    provider_metadata: JsonObject = Field(default_factory=dict)

    @classmethod
    def from_provider_arguments(
        cls,
        *,
        tool_name: str,
        arguments: str | JsonObject,
        call_id: str | None = None,
        order: int = 0,
        provider_item_id: str | None = None,
        provider_call_id: str | None = None,
        provider_tool_use_id: str | None = None,
        provider_status: str | None = None,
        provider_metadata: JsonObject | None = None,
    ) -> ToolCall:
        """Create a tool call from OpenAI JSON-string or Anthropic object args."""

        normalized_arguments = normalize_tool_arguments(arguments)
        return cls(
            call_id=call_id or new_id("call"),
            tool_name=tool_name,
            arguments=normalized_arguments,
            order=order,
            provider_item_id=provider_item_id,
            provider_call_id=provider_call_id,
            provider_tool_use_id=provider_tool_use_id,
            provider_status=provider_status,
            provider_metadata=provider_metadata or {},
        )


class ToolError(StrictModel):
    """Structured model-visible tool failure."""

    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    details: JsonObject = Field(default_factory=dict)

    @classmethod
    def from_error_info(cls, error: ErrorInfo) -> ToolError:
        """Convert a public/persisted error payload into a tool error."""

        return cls(error_type=error.code, message=error.message, details=error.details)


def _empty_content_parts() -> list[ContentPart]:
    return []


class ToolResult(StrictModel):
    """Structured result of a tool call.

    Later tool-execution phases will add backend-specific output and truncation
    helpers around this schema. The v1 core fields needed by model adapters and
    persistence are present now: linkage, success/error state, output, timing,
    truncation/timeout flags, order, and provider IDs.
    """

    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: JsonObject = Field(default_factory=dict)
    success: bool
    output: JsonValue | None = None
    error: ToolError | None = None
    started_at: str | None = Field(default=None, min_length=1)
    ended_at: str | None = Field(default=None, min_length=1)
    duration_ms: _NonNegativeDuration | None = None
    timed_out: bool = False
    truncated: bool = False
    truncation: TruncationInfo | None = None
    order: _NonNegativeOrder = 0
    provider_item_id: str | None = Field(default=None, min_length=1)
    provider_call_id: str | None = Field(default=None, min_length=1)
    provider_tool_use_id: str | None = Field(default=None, min_length=1)
    provider_metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_success_error_pair(self) -> ToolResult:
        if self.success and self.error is not None:
            raise ValueError("successful tool results must not include an error")
        if not self.success and self.error is None:
            raise ValueError("failed tool results must include an error")
        if self.truncated and self.truncation is None:
            raise ValueError("truncated tool results must include truncation metadata")
        if self.truncation is not None and self.truncation.truncated != self.truncated:
            raise ValueError("tool result truncated flag must match truncation metadata")
        return self


class ToolResultMessage(StrictModel):
    """Model-visible message that returns a tool result to the model."""

    message_id: str = Field(default_factory=lambda: new_id("msg"), min_length=1)
    sequence: int | None = Field(default=None, ge=0)
    role: Literal[MessageRole.TOOL] = MessageRole.TOOL
    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    content: list[ContentPart] = Field(default_factory=_empty_content_parts)
    result: ToolResult
    provider_metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_result_linkage(self) -> ToolResultMessage:
        if self.result.tool_call_id != self.tool_call_id:
            raise ValueError("tool result message call ID must match result call ID")
        if self.result.tool_name != self.tool_name:
            raise ValueError("tool result message tool name must match result tool name")
        return self

    @classmethod
    def from_result(
        cls,
        result: ToolResult,
        *,
        message_id: str | None = None,
        sequence: int | None = None,
        text: str | None = None,
        provider_metadata: JsonObject | None = None,
    ) -> ToolResultMessage:
        """Create a model-visible tool-result message from a structured result."""

        visible_text = text if text is not None else model_visible_tool_result_text(result)
        return cls(
            message_id=message_id or new_id("msg"),
            sequence=sequence,
            tool_call_id=result.tool_call_id,
            tool_name=result.tool_name,
            content=[TextContent(text=visible_text)],
            result=result,
            provider_metadata=provider_metadata or {},
        )


def normalize_tool_arguments(value: str | JsonObject) -> JsonObject:
    """Normalize provider-native tool arguments to a strict JSON object.

    OpenAI Responses function calls provide ``arguments`` as a JSON string;
    Anthropic tool-use blocks provide ``input`` as a JSON object. Core schemas use
    only the normalized object form.
    """

    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("tool arguments JSON string is invalid") from exc
        return _validate_json_object(decoded)
    return _validate_json_object(value)


def model_visible_tool_result_text(result: ToolResult) -> str:
    """Return concise text suitable for a provider tool-result message."""

    if result.success:
        if result.output is None:
            return "Tool completed successfully with no output."
        if isinstance(result.output, str):
            return result.output
        return json.dumps(result.output, sort_keys=True, separators=(",", ":"))
    if result.error is None:  # Defensive; model validation normally prevents this.
        return "Tool failed."
    return f"Tool error ({result.error.error_type}): {result.error.message}"


def _validate_json_object(value: object) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError("tool arguments must be a JSON object")
    return _JSON_OBJECT_ADAPTER.validate_python(value)


__all__ = (
    "ToolCall",
    "ToolError",
    "ToolResult",
    "ToolResultMessage",
    "model_visible_tool_result_text",
    "normalize_tool_arguments",
)
