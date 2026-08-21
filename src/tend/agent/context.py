"""Provider-neutral active-context construction.

The context builder is the boundary between persisted/replayed session state and
model requests. It operates only on core message/tool/result schemas; provider
adapters remain responsible for translating the resulting history to native
OpenAI Responses or Anthropic Messages payloads.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated

from pydantic import Field, TypeAdapter

from tend._common.types import JsonObject, StrictModel, new_id
from tend.agent.persistence.state import InterruptedToolCall, SessionState
from tend.llm.history import (
    ASSISTANT_MODEL_RESPONSE_ID_METADATA_KEY,
    ASSISTANT_PROVIDER_METADATA_KEY,
    ASSISTANT_REASONING_METADATA_KEY,
    ASSISTANT_RESPONSE_METADATA_KEY,
    ASSISTANT_TOOL_CALLS_METADATA_KEY,
    assistant_message_from_response,
    assistant_message_from_tool_calls,
    assistant_message_with_tool_calls,
    assistant_tool_calls,
)
from tend.llm.models.messages import (
    AssistantMessage,
    CompactionSummaryContent,
    DeveloperMessage,
    SystemMessage,
    TextContent,
    UserMessage,
)
from tend.llm.models.provider import ProviderMetadata
from tend.llm.models.reasoning import ReasoningSettings
from tend.llm.models.requests import ModelMessage, ModelRequest
from tend.llm.models.tools import ToolResultMessage

_NonNegativeSequence = Annotated[int, Field(ge=0)]
_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


def _empty_strings() -> list[str]:
    return []


def _empty_json_object() -> JsonObject:
    return {}


def _empty_model_messages() -> list[ModelMessage]:
    return []


class ActiveCompactionSummary(StrictModel):
    """A compaction summary ready to be inserted into active model context."""

    summary: str = Field(min_length=1)
    covered_message_ids: list[str] = Field(default_factory=_empty_strings)
    message_id: str | None = Field(default=None, min_length=1)
    sequence: _NonNegativeSequence | None = None
    provider_metadata: JsonObject = Field(default_factory=_empty_json_object)

    def to_message(self) -> AssistantMessage:
        """Return the dedicated compaction-summary assistant message."""

        return AssistantMessage(
            message_id=self.message_id or new_id("msg"),
            sequence=self.sequence,
            content=[
                CompactionSummaryContent(
                    summary=self.summary,
                    covered_message_ids=list(self.covered_message_ids),
                )
            ],
            provider_metadata=self.provider_metadata.copy(),
        )


class ContextToolPairing(StrictModel):
    """Tool-call/result pairing information found in an active context."""

    tool_call_ids: list[str] = Field(default_factory=_empty_strings)
    tool_result_ids: list[str] = Field(default_factory=_empty_strings)
    unresolved_tool_call_ids: list[str] = Field(default_factory=_empty_strings)
    orphan_tool_result_ids: list[str] = Field(default_factory=_empty_strings)


class ActiveContext(StrictModel):
    """Provider-neutral active context plus bookkeeping for the new prompt."""

    messages: list[ModelMessage] = Field(default_factory=_empty_model_messages)
    input_message_id: str = Field(min_length=1)
    tool_pairing: ContextToolPairing = Field(default_factory=ContextToolPairing)

    @property
    def unresolved_tool_call_ids(self) -> tuple[str, ...]:
        """Return assistant tool calls without matching tool-result messages."""

        return tuple(self.tool_pairing.unresolved_tool_call_ids)


def build_active_context(
    *,
    system_prompt: str,
    new_user_prompt: str,
    session_state: SessionState | None = None,
    compaction_summaries: Iterable[ActiveCompactionSummary] = (),
    tail_messages: Iterable[ModelMessage] = (),
    developer_prompt: str | None = None,
    input_message_id: str | None = None,
    include_interrupted_tool_results: bool = True,
    allow_unresolved_tool_calls: bool = True,
) -> ActiveContext:
    """Build the provider-neutral message list for the next model request.

    ``tail_messages`` is the already-active uncompacted history. Compaction
    summaries are inserted before that tail as dedicated summary content. When
    replay state contains interrupted tool calls, their synthetic model-visible
    results are inserted after the matching assistant tool call when possible;
    if the matching assistant message is not present, a minimal assistant
    tool-call message is synthesized so the tool result is never orphaned.
    """

    _validate_non_empty_text(system_prompt, field_name="system_prompt")
    _validate_non_empty_text(new_user_prompt, field_name="new_user_prompt")
    if developer_prompt is not None:
        _validate_non_empty_text(developer_prompt, field_name="developer_prompt")

    messages: list[ModelMessage] = [
        SystemMessage(content=[TextContent(text=system_prompt)]),
    ]
    if developer_prompt is not None:
        messages.append(DeveloperMessage(content=[TextContent(text=developer_prompt)]))

    for summary in compaction_summaries:
        messages.append(summary.to_message())

    active_tail = [_copy_model_message(message) for message in tail_messages]
    if include_interrupted_tool_results and session_state is not None:
        active_tail = _merge_interrupted_tool_results(active_tail, session_state)
    messages.extend(active_tail)

    user_message = UserMessage(
        message_id=input_message_id or new_id("msg"),
        content=[TextContent(text=new_user_prompt)],
    )
    messages.append(user_message)

    tool_pairing = validate_context_messages(
        messages,
        allow_unresolved_tool_calls=allow_unresolved_tool_calls,
        allow_orphan_tool_results=False,
    )
    return ActiveContext(
        messages=messages,
        input_message_id=user_message.message_id,
        tool_pairing=tool_pairing,
    )


def build_model_request(
    *,
    system_prompt: str,
    new_user_prompt: str,
    session_state: SessionState | None = None,
    compaction_summaries: Iterable[ActiveCompactionSummary] = (),
    tail_messages: Iterable[ModelMessage] = (),
    developer_prompt: str | None = None,
    input_message_id: str | None = None,
    model_name: str | None = None,
    tools: Iterable[JsonObject] = (),
    reasoning: ReasoningSettings | None = None,
    max_output_tokens: int | None = None,
    provider_metadata: ProviderMetadata | None = None,
    request_metadata: JsonObject | None = None,
) -> ModelRequest:
    """Convenience wrapper that packages active context into ``ModelRequest``."""

    context = build_active_context(
        system_prompt=system_prompt,
        new_user_prompt=new_user_prompt,
        session_state=session_state,
        compaction_summaries=compaction_summaries,
        tail_messages=tail_messages,
        developer_prompt=developer_prompt,
        input_message_id=input_message_id,
    )
    return ModelRequest(
        model_name=model_name,
        messages=context.messages,
        tools=[_copy_json_object(tool) for tool in tools],
        reasoning=reasoning.model_copy(deep=True) if reasoning is not None else None,
        max_output_tokens=max_output_tokens,
        provider_metadata=(
            provider_metadata.model_copy(deep=True) if provider_metadata is not None else None
        ),
        request_metadata={} if request_metadata is None else request_metadata.copy(),
    )


def inspect_context_tool_pairing(messages: Iterable[ModelMessage]) -> ContextToolPairing:
    """Inspect assistant tool calls and linked tool-result messages in order."""

    tool_call_ids: list[str] = []
    tool_result_ids: list[str] = []
    orphan_tool_result_ids: list[str] = []
    seen_calls: set[str] = set()
    seen_results: set[str] = set()

    for message in messages:
        if isinstance(message, AssistantMessage):
            for tool_call in assistant_tool_calls(message):
                if tool_call.call_id in seen_calls:
                    raise ValueError(f"duplicate assistant tool call ID: {tool_call.call_id}")
                seen_calls.add(tool_call.call_id)
                tool_call_ids.append(tool_call.call_id)
        elif isinstance(message, ToolResultMessage):
            tool_call_id = message.tool_call_id
            if tool_call_id in seen_results:
                raise ValueError(f"duplicate tool result for call ID: {tool_call_id}")
            seen_results.add(tool_call_id)
            tool_result_ids.append(tool_call_id)
            if tool_call_id not in seen_calls:
                orphan_tool_result_ids.append(tool_call_id)

    unresolved_tool_call_ids = [
        tool_call_id for tool_call_id in tool_call_ids if tool_call_id not in seen_results
    ]
    return ContextToolPairing(
        tool_call_ids=tool_call_ids,
        tool_result_ids=tool_result_ids,
        unresolved_tool_call_ids=unresolved_tool_call_ids,
        orphan_tool_result_ids=orphan_tool_result_ids,
    )


def validate_context_messages(
    messages: Sequence[ModelMessage],
    *,
    allow_unresolved_tool_calls: bool = True,
    allow_orphan_tool_results: bool = False,
) -> ContextToolPairing:
    """Validate tool-call/result pairing invariants for active context."""

    pairing = inspect_context_tool_pairing(messages)
    if pairing.orphan_tool_result_ids and not allow_orphan_tool_results:
        joined = ", ".join(pairing.orphan_tool_result_ids)
        raise ValueError(f"tool result messages without prior assistant tool calls: {joined}")
    if pairing.unresolved_tool_call_ids and not allow_unresolved_tool_calls:
        joined = ", ".join(pairing.unresolved_tool_call_ids)
        raise ValueError(f"assistant tool calls without tool results: {joined}")
    return pairing


def _merge_interrupted_tool_results(
    messages: list[ModelMessage],
    session_state: SessionState,
) -> list[ModelMessage]:
    interrupted_calls = _ordered_interrupted_tool_calls(session_state)
    if not interrupted_calls:
        return messages

    interrupted_by_id = {item.tool_call_id: item for item in interrupted_calls}
    existing_result_ids = {
        message.tool_call_id for message in messages if isinstance(message, ToolResultMessage)
    }
    inserted_ids: set[str] = set()
    merged: list[ModelMessage] = []

    for message in messages:
        merged.append(message)
        if not isinstance(message, AssistantMessage):
            continue
        for tool_call in assistant_tool_calls(message):
            interrupted = interrupted_by_id.get(tool_call.call_id)
            if interrupted is None or tool_call.call_id in existing_result_ids:
                continue
            merged.append(ToolResultMessage.from_result(interrupted.result))
            existing_result_ids.add(tool_call.call_id)
            inserted_ids.add(tool_call.call_id)

    for interrupted in interrupted_calls:
        already_has_result = interrupted.tool_call_id in existing_result_ids
        already_inserted = interrupted.tool_call_id in inserted_ids
        if already_has_result or already_inserted:
            continue
        merged.append(assistant_message_from_tool_calls([interrupted.tool_call]))
        merged.append(ToolResultMessage.from_result(interrupted.result))
        existing_result_ids.add(interrupted.tool_call_id)
        inserted_ids.add(interrupted.tool_call_id)

    return merged


def _ordered_interrupted_tool_calls(session_state: SessionState) -> tuple[InterruptedToolCall, ...]:
    return tuple(
        sorted(
            session_state.interrupted_tool_calls.values(),
            key=lambda item: (item.turn_id or "", item.order, item.started_event_id),
        )
    )


def _copy_model_message(message: ModelMessage) -> ModelMessage:
    return message.model_copy(deep=True)


def _copy_json_object(value: JsonObject) -> JsonObject:
    return _json_object(value)


def _json_object(value: Mapping[str, object]) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(dict(value))


def _validate_non_empty_text(value: str, *, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


__all__ = (
    "ASSISTANT_MODEL_RESPONSE_ID_METADATA_KEY",
    "ASSISTANT_PROVIDER_METADATA_KEY",
    "ASSISTANT_REASONING_METADATA_KEY",
    "ASSISTANT_RESPONSE_METADATA_KEY",
    "ASSISTANT_TOOL_CALLS_METADATA_KEY",
    "ActiveCompactionSummary",
    "ActiveContext",
    "ContextToolPairing",
    "assistant_message_from_response",
    "assistant_message_from_tool_calls",
    "assistant_message_with_tool_calls",
    "assistant_tool_calls",
    "build_active_context",
    "build_model_request",
    "inspect_context_tool_pairing",
    "validate_context_messages",
)
