"""Provider-neutral reasoning settings and metadata schemas."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field

from tend._common.types import JsonObject, StrictModel

_NonNegativeTokenCount = Annotated[int, Field(ge=0)]
_NonNegativeOrder = Annotated[int, Field(ge=0)]


class ReasoningEffort(StrEnum):
    """Unified reasoning-effort values understood by model profiles/adapters."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class ReasoningSummaryPreference(StrEnum):
    """Requested reasoning-summary behavior."""

    NONE = "none"
    AUTO = "auto"
    CONCISE = "concise"
    DETAILED = "detailed"


class ReasoningDisplayPolicy(StrEnum):
    """Controls how reasoning metadata may be displayed to callers."""

    HIDDEN = "hidden"
    SUMMARY_ONLY = "summary_only"
    VISIBLE = "visible"


class ReasoningSettings(StrictModel):
    """Provider-neutral reasoning settings requested by callers/config."""

    effort: ReasoningEffort | None = None
    summary: ReasoningSummaryPreference | None = None
    display_policy: ReasoningDisplayPolicy = ReasoningDisplayPolicy.SUMMARY_ONLY
    max_reasoning_tokens: _NonNegativeTokenCount | None = None
    include_provider_private_metadata: bool = True
    native_settings: JsonObject = Field(default_factory=dict)


class ReasoningSummary(StrictModel):
    """Provider-exposed reasoning summary that is safe to persist/display by policy."""

    text: str = Field(min_length=1)
    provider_item_id: str | None = Field(default=None, min_length=1)
    redacted: bool = False


class ReasoningContinuationMetadata(StrictModel):
    """Provider-private reasoning continuation data, not normal assistant text."""

    provider_name: str | None = Field(default=None, min_length=1)
    kind: str = Field(min_length=1)
    order: _NonNegativeOrder | None = None
    provider_item_id: str | None = Field(default=None, min_length=1)
    provider_block_id: str | None = Field(default=None, min_length=1)
    encrypted_content: str | None = Field(default=None, min_length=1)
    signature: str | None = Field(default=None, min_length=1)
    redacted_details: JsonObject = Field(default_factory=dict)


_ReasoningSummaryList = list[ReasoningSummary]
_ReasoningContinuationList = list[ReasoningContinuationMetadata]


def _empty_reasoning_summaries() -> _ReasoningSummaryList:
    return []


def _empty_reasoning_continuation() -> _ReasoningContinuationList:
    return []


class ReasoningMetadata(StrictModel):
    """Reasoning metadata returned by a provider.

    Hidden chain-of-thought or provider-private continuation material is kept in
    explicit metadata fields and is never represented as normal assistant text by
    these schemas.
    """

    requested: ReasoningSettings | None = None
    observed_effort: str | None = Field(default=None, min_length=1)
    native_settings: JsonObject = Field(default_factory=dict)
    summaries: _ReasoningSummaryList = Field(default_factory=_empty_reasoning_summaries)
    reasoning_tokens: _NonNegativeTokenCount | None = None
    provider_private_continuation: _ReasoningContinuationList = Field(
        default_factory=_empty_reasoning_continuation
    )
    display_policy: ReasoningDisplayPolicy = ReasoningDisplayPolicy.SUMMARY_ONLY


__all__ = (
    "ReasoningContinuationMetadata",
    "ReasoningDisplayPolicy",
    "ReasoningEffort",
    "ReasoningMetadata",
    "ReasoningSettings",
    "ReasoningSummary",
    "ReasoningSummaryPreference",
)
