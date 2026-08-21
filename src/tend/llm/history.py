"""Provider-neutral assistant history metadata helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import TypeAdapter, ValidationError

from tend._common.types import JsonObject, new_id
from tend.llm.models.messages import AssistantMessage, ContentPart, TextContent
from tend.llm.models.requests import ModelResponse
from tend.llm.models.tools import ToolCall

ASSISTANT_TOOL_CALLS_METADATA_KEY = "tool_calls"
ASSISTANT_MODEL_RESPONSE_ID_METADATA_KEY = "model_response_id"
ASSISTANT_PROVIDER_METADATA_KEY = "provider_response_metadata"
ASSISTANT_REASONING_METADATA_KEY = "reasoning_metadata"
ASSISTANT_RESPONSE_METADATA_KEY = "response_metadata"

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


def assistant_message_from_response(
    response: ModelResponse,
    *,
    message_id: str | None = None,
    sequence: int | None = None,
) -> AssistantMessage:
    """Return an assistant history message carrying response tool metadata."""

    if response.assistant_message is None:
        message = AssistantMessage(
            message_id=message_id or new_id("msg"),
            sequence=sequence,
        )
    else:
        message = response.assistant_message.model_copy(deep=True)
        updates: dict[str, object] = {}
        if message_id is not None:
            updates["message_id"] = message_id
        if sequence is not None:
            updates["sequence"] = sequence
        if updates:
            message = message.model_copy(update=updates)

    metadata: dict[str, object] = dict(message.provider_metadata)
    metadata[ASSISTANT_MODEL_RESPONSE_ID_METADATA_KEY] = response.response_id
    if response.provider_metadata is not None:
        metadata[ASSISTANT_PROVIDER_METADATA_KEY] = response.provider_metadata.model_dump(
            mode="json"
        )
    if response.reasoning is not None:
        metadata[ASSISTANT_REASONING_METADATA_KEY] = response.reasoning.model_dump(mode="json")
    if response.response_metadata:
        metadata[ASSISTANT_RESPONSE_METADATA_KEY] = response.response_metadata.copy()

    message = message.model_copy(update={"provider_metadata": _json_object(metadata)})
    if not response.tool_calls:
        return message
    return assistant_message_with_tool_calls(message, response.tool_calls)


def assistant_message_from_tool_calls(
    tool_calls: Iterable[ToolCall],
    *,
    text: str | None = None,
    message_id: str | None = None,
    sequence: int | None = None,
    provider_metadata: JsonObject | None = None,
) -> AssistantMessage:
    """Create a minimal assistant message that preserves requested tool calls."""

    content: list[ContentPart] = [] if text is None else [TextContent(text=text)]
    message = AssistantMessage(
        message_id=message_id or new_id("msg"),
        sequence=sequence,
        content=content,
        provider_metadata={} if provider_metadata is None else provider_metadata.copy(),
    )
    return assistant_message_with_tool_calls(message, tool_calls)


def assistant_message_with_tool_calls(
    message: AssistantMessage,
    tool_calls: Iterable[ToolCall],
) -> AssistantMessage:
    """Return a copy of ``message`` with serialized provider-neutral tool calls."""

    calls = tuple(tool_call.model_copy(deep=True) for tool_call in tool_calls)
    if not calls:
        return message.model_copy(deep=True)

    seen: set[str] = set()
    for tool_call in calls:
        if tool_call.call_id in seen:
            raise ValueError(f"duplicate assistant tool call ID: {tool_call.call_id}")
        seen.add(tool_call.call_id)

    metadata: dict[str, object] = dict(message.provider_metadata)
    metadata[ASSISTANT_TOOL_CALLS_METADATA_KEY] = [
        tool_call.model_dump(mode="json") for tool_call in calls
    ]
    return message.model_copy(update={"provider_metadata": _json_object(metadata)}, deep=True)


def assistant_tool_calls(message: AssistantMessage) -> tuple[ToolCall, ...]:
    """Extract serialized provider-neutral tool calls from assistant metadata."""

    value = message.provider_metadata.get(ASSISTANT_TOOL_CALLS_METADATA_KEY)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("assistant tool-call metadata must be a list")

    calls: list[ToolCall] = []
    for index, raw_call in enumerate(value):
        if not isinstance(raw_call, Mapping):
            raise ValueError(f"assistant tool-call metadata item {index} must be an object")
        try:
            calls.append(ToolCall.model_validate(raw_call))
        except ValidationError as exc:
            raise ValueError(f"assistant tool-call metadata item {index} is invalid") from exc
    return tuple(calls)


def _json_object(value: Mapping[str, object]) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(dict(value))


__all__ = (
    "ASSISTANT_MODEL_RESPONSE_ID_METADATA_KEY",
    "ASSISTANT_PROVIDER_METADATA_KEY",
    "ASSISTANT_REASONING_METADATA_KEY",
    "ASSISTANT_RESPONSE_METADATA_KEY",
    "ASSISTANT_TOOL_CALLS_METADATA_KEY",
    "assistant_message_from_response",
    "assistant_message_from_tool_calls",
    "assistant_message_with_tool_calls",
    "assistant_tool_calls",
)
