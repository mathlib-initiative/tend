"""Provider-neutral model request and response containers."""

from __future__ import annotations

from typing import Annotated

from pydantic import Discriminator, Field, model_validator

from tend._common.types import JsonObject, StopReason, StrictModel, new_id
from tend.llm.models.messages import (
    AssistantMessage,
    DeveloperMessage,
    SystemMessage,
    TextContent,
    UserMessage,
)
from tend.llm.models.provider import ProviderCompletionStatus, ProviderMetadata
from tend.llm.models.reasoning import ReasoningMetadata, ReasoningSettings
from tend.llm.models.tools import ToolCall, ToolResultMessage
from tend.llm.usage import Usage

_MODEL_MESSAGE_DISCRIMINATOR: Discriminator = Discriminator("role")

type ModelMessage = Annotated[
    SystemMessage | DeveloperMessage | UserMessage | AssistantMessage | ToolResultMessage,
    _MODEL_MESSAGE_DISCRIMINATOR,
]


def _empty_model_messages() -> list[ModelMessage]:
    return []


def _empty_tool_definitions() -> list[JsonObject]:
    return []


def _empty_tool_calls() -> list[ToolCall]:
    return []


class ModelRequest(StrictModel):
    """Provider-neutral request consumed by model adapters."""

    request_id: str = Field(default_factory=lambda: new_id("model_req"), min_length=1)
    model_name: str | None = Field(default=None, min_length=1)
    messages: list[ModelMessage] = Field(default_factory=_empty_model_messages)
    tools: list[JsonObject] = Field(default_factory=_empty_tool_definitions)
    reasoning: ReasoningSettings | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    provider_metadata: ProviderMetadata | None = None
    request_metadata: JsonObject = Field(default_factory=dict)


class ModelResponse(StrictModel):
    """Provider-neutral response returned by model adapters."""

    response_id: str = Field(default_factory=lambda: new_id("model_resp"), min_length=1)
    request_id: str | None = Field(default=None, min_length=1)
    assistant_message: AssistantMessage | None = None
    tool_calls: list[ToolCall] = Field(default_factory=_empty_tool_calls)
    stop_reason: StopReason | None = None
    provider_completion_status: ProviderCompletionStatus = ProviderCompletionStatus.COMPLETED
    incomplete_details: JsonObject = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    reasoning: ReasoningMetadata | None = None
    provider_metadata: ProviderMetadata | None = None
    response_metadata: JsonObject = Field(default_factory=dict)

    @property
    def final_text(self) -> str | None:
        """Return concatenated normal assistant text, excluding reasoning metadata."""

        if self.assistant_message is None:
            return None
        text_parts = [
            part.text for part in self.assistant_message.content if isinstance(part, TextContent)
        ]
        if not text_parts:
            return None
        return "\n".join(text_parts)

    @model_validator(mode="after")
    def _validate_tool_call_ids(self) -> ModelResponse:
        seen_call_ids: set[str] = set()
        for tool_call in self.tool_calls:
            if tool_call.call_id in seen_call_ids:
                raise ValueError("model response tool call IDs must be unique")
            seen_call_ids.add(tool_call.call_id)
        return self


__all__ = ("ModelMessage", "ModelRequest", "ModelResponse")
