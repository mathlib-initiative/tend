"""Provider metadata schemas kept at model-adapter boundaries."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field

from tend._common.types import JsonObject, StrictModel

_NonNegativeOrder = Annotated[int, Field(ge=0)]


class ContinuationStrategy(StrEnum):
    """How a follow-up request should preserve provider conversation state."""

    STATELESS_REPLAY = "stateless_replay"
    PROVIDER_RESPONSE_ID = "provider_response_id"
    NONE = "none"


class ProviderCompletionStatus(StrEnum):
    """Provider-level completion status before turn-loop stop normalization."""

    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


class ProviderItemKind(StrEnum):
    """Known provider item kinds that may be needed for continuation."""

    RESPONSE = "response"
    OUTPUT_TEXT = "output_text"
    FUNCTION_CALL = "function_call"
    FUNCTION_CALL_OUTPUT = "function_call_output"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    REASONING = "reasoning"
    THINKING = "thinking"


class ProviderItemMetadata(StrictModel):
    """Ordered provider item/block identity needed to rebuild native history."""

    kind: ProviderItemKind
    order: _NonNegativeOrder | None = None
    provider_item_id: str | None = Field(default=None, min_length=1)
    provider_call_id: str | None = Field(default=None, min_length=1)
    provider_tool_use_id: str | None = Field(default=None, min_length=1)
    provider_block_id: str | None = Field(default=None, min_length=1)
    tool_name: str | None = Field(default=None, min_length=1)
    status: str | None = Field(default=None, min_length=1)
    thinking_signature: str | None = Field(default=None, min_length=1)
    encrypted_reasoning_content: str | None = Field(default=None, min_length=1)
    redacted_details: JsonObject = Field(default_factory=dict)


def _empty_provider_items() -> list[ProviderItemMetadata]:
    return []


class ProviderMetadata(StrictModel):
    """Provider-neutral response metadata with deliberate escape hatches."""

    provider_name: str = Field(min_length=1)
    model_name: str | None = Field(default=None, min_length=1)
    response_id: str | None = Field(default=None, min_length=1)
    previous_response_id: str | None = Field(default=None, min_length=1)
    native_stop_reason: str | None = Field(default=None, min_length=1)
    item_ids: list[str] = Field(default_factory=list)
    items: list[ProviderItemMetadata] = Field(default_factory=_empty_provider_items)
    continuation_strategy: ContinuationStrategy = ContinuationStrategy.STATELESS_REPLAY
    provider_side_continuation_available: bool | None = None
    stateless_continuation_required: bool = False
    redacted_raw_details: JsonObject = Field(default_factory=dict)
    artifact_reference_ids: list[str] = Field(default_factory=list)


__all__ = (
    "ContinuationStrategy",
    "ProviderCompletionStatus",
    "ProviderItemKind",
    "ProviderItemMetadata",
    "ProviderMetadata",
)
