"""Deterministic compaction trigger and safe-cut planning.

This module decides *where* generic compaction may run; it never calls a
model and never mutates history. The main safety invariant is that assistant
requested tool calls must not be separated from their corresponding tool-result
messages, and unresolved tool calls must remain in the preserved active tail.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import ceil
from typing import Annotated

from pydantic import Field, model_validator

from tend._common.types import JsonObject, StrictModel
from tend.agent.config import CompactionConfig
from tend.agent.context import assistant_tool_calls
from tend.llm.context_estimation import (
    RequestTokenEstimator,
    TokenEstimatorConfig,
    estimate_request_tokens,
)
from tend.llm.models.messages import (
    AssistantMessage,
    CompactionSummaryContent,
    DeveloperMessage,
    SystemMessage,
    UserMessage,
)
from tend.llm.models.profiles import ModelProfile
from tend.llm.models.reasoning import ReasoningSettings
from tend.llm.models.requests import ModelMessage, ModelRequest
from tend.llm.models.tools import ToolResultMessage

_NonNegativeInt = Annotated[int, Field(ge=0)]


def _empty_strings() -> list[str]:
    return []


def _empty_trigger_reasons() -> list[CompactionTriggerReason]:
    return []


class CompactionTriggerReason(StrEnum):
    """Reasons that a compaction threshold or forced retry path was triggered."""

    THRESHOLD_TOKENS = "threshold_tokens"
    THRESHOLD_MESSAGES = "threshold_messages"
    CONTEXT_WINDOW = "context_window"
    CONTEXT_OVERFLOW = "context_overflow"


class CompactionPlan(StrictModel):
    """Pure description of a possible compaction operation.

    ``compact_start_index`` and ``compact_end_index`` use normal Python slicing
    semantics over the input message sequence. When ``should_compact`` is true,
    the covered range is ``messages[compact_start_index:compact_end_index]``.
    Leading system/developer instruction messages are intentionally outside the
    compacted range and remain available verbatim.
    """

    enabled: bool
    should_compact: bool
    # Whether the ordinary character/message estimate would have triggered
    # without an API anchor. The default preserves replay of older plan metadata.
    char_triggered: bool = False
    trigger_reasons: list[CompactionTriggerReason] = Field(default_factory=_empty_trigger_reasons)
    skip_reason: str | None = Field(default=None, min_length=1)
    message_count: _NonNegativeInt
    estimated_tokens: _NonNegativeInt
    anchor_estimated_tokens: _NonNegativeInt | None = None
    token_scale_factor: float = Field(default=1.0, ge=1)
    token_target_tokens: _NonNegativeInt | None = None
    projected_tokens: _NonNegativeInt | None = None
    message_token_estimate: _NonNegativeInt
    effective_threshold_tokens: _NonNegativeInt | None = None
    context_limit_tokens: _NonNegativeInt | None = None
    reserve_tokens: _NonNegativeInt
    keep_recent_tokens: _NonNegativeInt
    effective_keep_recent_tokens: _NonNegativeInt
    target_tokens: _NonNegativeInt
    compact_start_index: _NonNegativeInt | None = None
    compact_end_index: _NonNegativeInt | None = None
    compact_message_ids: list[str] = Field(default_factory=_empty_strings)
    preserved_message_ids: list[str] = Field(default_factory=_empty_strings)
    compact_token_estimate: _NonNegativeInt = 0
    preserved_token_estimate: _NonNegativeInt = 0
    recent_token_estimate: _NonNegativeInt = 0
    split_turn_prefix: bool = False

    @model_validator(mode="after")
    def _validate_compaction_range(self) -> CompactionPlan:
        if self.should_compact:
            if self.compact_start_index is None or self.compact_end_index is None:
                raise ValueError("compaction plans must include compact range indices")
            if self.compact_end_index <= self.compact_start_index:
                raise ValueError("compaction range must be non-empty")
            if not self.compact_message_ids:
                raise ValueError("compaction plans must include compact message IDs")
            if self.skip_reason is not None:
                raise ValueError("compaction plans must not include a skip reason")
        else:
            if self.compact_start_index is not None or self.compact_end_index is not None:
                raise ValueError("skipped compaction plans must not include range indices")
            if self.compact_message_ids:
                raise ValueError("skipped compaction plans must not include compact message IDs")
        return self


@dataclass(frozen=True, slots=True)
class _ToolBoundary:
    call_id: str
    call_index: int
    result_index: int | None


def plan_compaction(
    *,
    messages: Sequence[ModelMessage],
    config: CompactionConfig,
    profile: ModelProfile | None = None,
    estimator_config: TokenEstimatorConfig | None = None,
    tools: Iterable[JsonObject] = (),
    reasoning: ReasoningSettings | None = None,
    anchor_estimated_tokens: int | None = None,
    force_context_overflow: bool = False,
    token_estimator: RequestTokenEstimator | None = None,
) -> CompactionPlan:
    """Return a deterministic pre-request compaction plan for active messages.

    Triggering uses configured token/message thresholds plus the known model
    context window minus reserve tokens when a profile provides a window. When
    an API-anchored token estimate is available, token triggers use the larger
    of it and the local estimate. Per-message and fixed costs are calibrated to
    that same metric before selecting a suffix. Token-triggered plans must leave
    10% headroom, including the summary, instructions and fixed request costs.
    Tool pairs and unresolved calls remain protected. Forced overflow recovery
    instead uses the smallest safe suffix, even if the projection is pessimistic.
    """

    estimator = estimator_config or TokenEstimatorConfig()
    estimate = estimate_request_tokens(
        ModelRequest(messages=list(messages), tools=list(tools), reasoning=reasoning),
        config=estimator,
        token_estimator=token_estimator,
    )
    message_tokens = estimate.message_tokens
    message_token_estimate = sum(message_tokens)
    local_total = estimate.parts.total_tokens
    context_limit_tokens = _context_limit_tokens(config, profile)
    effective_threshold_tokens = _effective_threshold_tokens(config, context_limit_tokens)
    trigger_estimated_tokens = max(
        local_total,
        anchor_estimated_tokens if anchor_estimated_tokens is not None else 0,
    )
    token_scale_factor = max(1.0, trigger_estimated_tokens / max(local_total, 1))
    calibrated_tokens = [ceil(tokens * token_scale_factor) for tokens in message_tokens]
    fixed_tokens = max(
        ceil((local_total - message_token_estimate) * token_scale_factor),
        trigger_estimated_tokens - sum(calibrated_tokens),
    )
    char_triggered = bool(
        _trigger_reasons(
            estimated_tokens=local_total,
            message_count=len(messages),
            config=config,
            context_limit_tokens=context_limit_tokens,
        )
    )
    trigger_reasons = _trigger_reasons(
        estimated_tokens=trigger_estimated_tokens,
        message_count=len(messages),
        config=config,
        context_limit_tokens=context_limit_tokens,
    )
    if force_context_overflow:
        trigger_reasons = [*trigger_reasons, CompactionTriggerReason.CONTEXT_OVERFLOW]
    token_target_tokens = (
        effective_threshold_tokens * 9 // 10
        if effective_threshold_tokens is not None
        and trigger_estimated_tokens > effective_threshold_tokens
        and not force_context_overflow
        else None
    )
    effective_keep_recent_tokens = _effective_keep_recent_tokens(config, profile)
    if token_target_tokens is not None:
        effective_keep_recent_tokens = min(
            effective_keep_recent_tokens,
            max(
                token_target_tokens
                - fixed_tokens
                - config.target_tokens
                - sum(calibrated_tokens[: leading_instruction_end(messages)]),
                0,
            ),
        )
    if force_context_overflow:
        # Provider-reported overflow means our estimate/budget was too optimistic.
        # Preserve the minimum safe suffix and compact as aggressively as the
        # tool-call/result boundary rules allow for the retry.
        effective_keep_recent_tokens = 0

    common = _CommonPlanFields(
        enabled=config.enabled,
        char_triggered=char_triggered,
        trigger_reasons=trigger_reasons,
        message_count=len(messages),
        estimated_tokens=local_total,
        anchor_estimated_tokens=anchor_estimated_tokens,
        token_scale_factor=token_scale_factor,
        token_target_tokens=token_target_tokens,
        message_token_estimate=message_token_estimate,
        effective_threshold_tokens=effective_threshold_tokens,
        context_limit_tokens=context_limit_tokens,
        reserve_tokens=config.reserve_tokens,
        keep_recent_tokens=config.keep_recent_tokens,
        effective_keep_recent_tokens=effective_keep_recent_tokens,
        target_tokens=config.target_tokens,
        preserved_message_ids=[message.message_id for message in messages],
        preserved_token_estimate=message_token_estimate,
        recent_token_estimate=message_token_estimate,
    )

    if not config.enabled:
        return _skipped_plan(common, skip_reason="compaction disabled")
    if not trigger_reasons:
        return _skipped_plan(common, skip_reason=None)

    compact_start = leading_instruction_end(messages)
    if compact_start >= len(messages):
        return _skipped_plan(common, skip_reason="no compactable messages")

    preferred_end = initial_recent_start(
        messages=messages,
        compactable_start=compact_start,
        token_estimates=calibrated_tokens,
        keep_recent_tokens=effective_keep_recent_tokens,
    )
    compact_end = find_safe_compaction_end(
        messages=messages,
        compactable_start=compact_start,
        preferred_end=preferred_end,
    )
    projected_tokens: int | None = None
    if token_target_tokens is not None:
        # A tool boundary can expand the suffix beyond its budget. Try later
        # safe cuts before giving up, always preserving the newest message.
        for candidate in range(max(compact_end, compact_start + 1), len(messages)):
            if not is_safe_compaction_range(messages, compact_start, candidate):
                continue
            projected_tokens = _projected_tokens(
                messages=messages,
                start=compact_start,
                end=candidate,
                calibrated_tokens=calibrated_tokens,
                fixed_tokens=fixed_tokens,
                target_tokens=config.target_tokens,
                scale=token_scale_factor,
                estimator=estimator,
                token_estimator=token_estimator,
            )
            if projected_tokens <= token_target_tokens:
                compact_end = candidate
                break
        else:
            return _skipped_plan(
                common.model_copy(update={"projected_tokens": projected_tokens}),
                skip_reason=(
                    "insufficient token reduction"
                    if projected_tokens is not None
                    else "no safe compaction range"
                ),
            )
    elif compact_end <= compact_start:
        return _skipped_plan(common, skip_reason="no safe compaction range")

    compact_message_ids = [message.message_id for message in messages[compact_start:compact_end]]
    preserved_message_ids = [
        message.message_id
        for index, message in enumerate(messages)
        if index < compact_start or index >= compact_end
    ]
    compact_token_estimate = sum(message_tokens[compact_start:compact_end])
    recent_token_estimate = sum(message_tokens[compact_end:])
    preserved_token_estimate = message_token_estimate - compact_token_estimate
    latest_user_index = latest_user_message_index(messages, start=compact_start)

    return CompactionPlan(
        enabled=config.enabled,
        should_compact=True,
        char_triggered=char_triggered,
        trigger_reasons=trigger_reasons,
        skip_reason=None,
        message_count=len(messages),
        estimated_tokens=local_total,
        anchor_estimated_tokens=anchor_estimated_tokens,
        token_scale_factor=token_scale_factor,
        token_target_tokens=token_target_tokens,
        projected_tokens=projected_tokens,
        message_token_estimate=message_token_estimate,
        effective_threshold_tokens=effective_threshold_tokens,
        context_limit_tokens=context_limit_tokens,
        reserve_tokens=config.reserve_tokens,
        keep_recent_tokens=config.keep_recent_tokens,
        effective_keep_recent_tokens=effective_keep_recent_tokens,
        target_tokens=config.target_tokens,
        compact_start_index=compact_start,
        compact_end_index=compact_end,
        compact_message_ids=compact_message_ids,
        preserved_message_ids=preserved_message_ids,
        compact_token_estimate=compact_token_estimate,
        preserved_token_estimate=preserved_token_estimate,
        recent_token_estimate=recent_token_estimate,
        split_turn_prefix=(
            latest_user_index is not None
            and compact_start <= latest_user_index < compact_end < len(messages)
        ),
    )


def _projected_tokens(
    *,
    messages: Sequence[ModelMessage],
    start: int,
    end: int,
    calibrated_tokens: Sequence[int],
    fixed_tokens: int,
    target_tokens: int,
    scale: float,
    estimator: TokenEstimatorConfig,
    token_estimator: RequestTokenEstimator | None,
) -> int:
    # The output limit bounds summary text, but the inserted message also has
    # a role, wrapper and covered IDs. Estimate those through the same adapter.
    summary = AssistantMessage(
        content=[
            CompactionSummaryContent(
                summary=" ",
                covered_message_ids=[message.message_id for message in messages[start:end]],
            )
        ]
    )
    summary_overhead = estimate_request_tokens(
        ModelRequest(messages=[summary]),
        config=estimator,
        token_estimator=token_estimator,
    ).message_tokens[0]
    return (
        sum(calibrated_tokens[:start])
        + sum(calibrated_tokens[end:])
        + fixed_tokens
        + target_tokens
        + ceil(summary_overhead * scale)
    )


def leading_instruction_end(messages: Sequence[ModelMessage]) -> int:
    """Return the first index after leading system/developer instructions."""

    index = 0
    while index < len(messages) and isinstance(messages[index], SystemMessage | DeveloperMessage):
        index += 1
    return index


def latest_user_message_index(
    messages: Sequence[ModelMessage],
    *,
    start: int = 0,
) -> int | None:
    """Return the latest user-message index at or after ``start``."""

    _validate_index_range(len(messages), start, len(messages))
    for index in range(len(messages) - 1, start - 1, -1):
        if isinstance(messages[index], UserMessage):
            return index
    return None


def initial_recent_start(
    *,
    messages: Sequence[ModelMessage],
    compactable_start: int,
    token_estimates: Sequence[int],
    keep_recent_tokens: int,
) -> int:
    """Choose an initial suffix boundary by walking backward from newest messages.

    At least the newest compactable message is preserved when any compactable
    message exists, even when that message alone exceeds the recent-token
    budget. Later safety adjustment may preserve more messages.
    """

    if len(token_estimates) != len(messages):
        raise ValueError("token_estimates length must match messages length")
    _validate_index_range(len(messages), compactable_start, len(messages))
    if compactable_start >= len(messages):
        return len(messages)

    recent_start = len(messages)
    recent_tokens = 0
    for index in range(len(messages) - 1, compactable_start - 1, -1):
        token_estimate = token_estimates[index]
        if token_estimate < 0:
            raise ValueError("token estimates must be non-negative")
        candidate_tokens = recent_tokens + token_estimate
        if recent_start != len(messages) and candidate_tokens > keep_recent_tokens:
            break
        recent_start = index
        recent_tokens = candidate_tokens
    return recent_start


def find_safe_compaction_end(
    *,
    messages: Sequence[ModelMessage],
    compactable_start: int,
    preferred_end: int,
) -> int:
    """Move a preferred cut point backward until the compacted range is safe."""

    _validate_index_range(len(messages), compactable_start, preferred_end)
    for compact_end in range(preferred_end, compactable_start - 1, -1):
        if is_safe_compaction_range(messages, compactable_start, compact_end):
            return compact_end
    return compactable_start


def is_safe_compaction_range(messages: Sequence[ModelMessage], start: int, end: int) -> bool:
    """Return whether ``messages[start:end]`` respects tool-pair invariants.

    A range is safe only if every completed assistant-tool/tool-result pair is
    either wholly inside or wholly outside the range. Assistant tool calls that
    do not yet have a result are unresolved and must remain outside the range.
    Orphan result messages are treated as protected and are not compacted.
    """

    _validate_index_range(len(messages), start, end)
    boundaries, orphan_result_indexes = _tool_boundaries(messages)
    for boundary in boundaries:
        call_inside = start <= boundary.call_index < end
        if boundary.result_index is None:
            if call_inside:
                return False
            continue
        result_inside = start <= boundary.result_index < end
        if call_inside != result_inside:
            return False
    for result_index in orphan_result_indexes:
        if start <= result_index < end:
            return False
    return True


def _tool_boundaries(messages: Sequence[ModelMessage]) -> tuple[list[_ToolBoundary], list[int]]:
    call_indexes: dict[str, int] = {}
    result_indexes: dict[str, int] = {}
    for index, message in enumerate(messages):
        if isinstance(message, AssistantMessage):
            for tool_call in assistant_tool_calls(message):
                if tool_call.call_id in call_indexes:
                    raise ValueError(f"duplicate assistant tool call ID: {tool_call.call_id}")
                call_indexes[tool_call.call_id] = index
        elif isinstance(message, ToolResultMessage):
            if message.tool_call_id in result_indexes:
                raise ValueError(f"duplicate tool result for call ID: {message.tool_call_id}")
            result_indexes[message.tool_call_id] = index

    boundaries = [
        _ToolBoundary(
            call_id=call_id,
            call_index=call_index,
            result_index=result_indexes.get(call_id),
        )
        for call_id, call_index in call_indexes.items()
    ]
    orphan_result_indexes = [
        result_index
        for call_id, result_index in result_indexes.items()
        if call_id not in call_indexes
    ]
    return (boundaries, orphan_result_indexes)


def _trigger_reasons(
    *,
    estimated_tokens: int,
    message_count: int,
    config: CompactionConfig,
    context_limit_tokens: int | None,
) -> list[CompactionTriggerReason]:
    reasons: list[CompactionTriggerReason] = []
    if config.threshold_tokens is not None and estimated_tokens > config.threshold_tokens:
        reasons.append(CompactionTriggerReason.THRESHOLD_TOKENS)
    if config.threshold_messages is not None and message_count > config.threshold_messages:
        reasons.append(CompactionTriggerReason.THRESHOLD_MESSAGES)
    if context_limit_tokens is not None and estimated_tokens > context_limit_tokens:
        reasons.append(CompactionTriggerReason.CONTEXT_WINDOW)
    return reasons


def _context_limit_tokens(config: CompactionConfig, profile: ModelProfile | None) -> int | None:
    if profile is None or profile.context_window is None:
        return None
    if profile.context_window.tokens <= config.reserve_tokens:
        # A reserve larger than the advertised window is not a useful trigger;
        # callers can configure explicit thresholds for tiny test/local models.
        return None
    return profile.context_window.tokens - config.reserve_tokens


def _effective_threshold_tokens(
    config: CompactionConfig,
    context_limit_tokens: int | None,
) -> int | None:
    thresholds: list[int] = []
    if config.threshold_tokens is not None:
        thresholds.append(config.threshold_tokens)
    if context_limit_tokens is not None:
        thresholds.append(context_limit_tokens)
    if not thresholds:
        return None
    return min(thresholds)


def _effective_keep_recent_tokens(config: CompactionConfig, profile: ModelProfile | None) -> int:
    if profile is None or profile.context_window is None:
        return config.keep_recent_tokens
    window_keep_budget = max(
        profile.context_window.tokens - config.reserve_tokens - config.target_tokens,
        0,
    )
    return min(config.keep_recent_tokens, window_keep_budget)


class _CommonPlanFields(StrictModel):
    enabled: bool
    char_triggered: bool
    trigger_reasons: list[CompactionTriggerReason]
    message_count: _NonNegativeInt
    estimated_tokens: _NonNegativeInt
    anchor_estimated_tokens: _NonNegativeInt | None
    token_scale_factor: float = Field(ge=1)
    token_target_tokens: _NonNegativeInt | None
    projected_tokens: _NonNegativeInt | None = None
    message_token_estimate: _NonNegativeInt
    effective_threshold_tokens: _NonNegativeInt | None
    context_limit_tokens: _NonNegativeInt | None
    reserve_tokens: _NonNegativeInt
    keep_recent_tokens: _NonNegativeInt
    effective_keep_recent_tokens: _NonNegativeInt
    target_tokens: _NonNegativeInt
    preserved_message_ids: list[str]
    preserved_token_estimate: _NonNegativeInt
    recent_token_estimate: _NonNegativeInt


def _skipped_plan(common: _CommonPlanFields, *, skip_reason: str | None) -> CompactionPlan:
    return CompactionPlan(
        enabled=common.enabled,
        should_compact=False,
        char_triggered=common.char_triggered,
        trigger_reasons=list(common.trigger_reasons),
        skip_reason=skip_reason,
        message_count=common.message_count,
        estimated_tokens=common.estimated_tokens,
        anchor_estimated_tokens=common.anchor_estimated_tokens,
        token_scale_factor=common.token_scale_factor,
        token_target_tokens=common.token_target_tokens,
        projected_tokens=common.projected_tokens,
        message_token_estimate=common.message_token_estimate,
        effective_threshold_tokens=common.effective_threshold_tokens,
        context_limit_tokens=common.context_limit_tokens,
        reserve_tokens=common.reserve_tokens,
        keep_recent_tokens=common.keep_recent_tokens,
        effective_keep_recent_tokens=common.effective_keep_recent_tokens,
        target_tokens=common.target_tokens,
        preserved_message_ids=list(common.preserved_message_ids),
        preserved_token_estimate=common.preserved_token_estimate,
        recent_token_estimate=common.recent_token_estimate,
    )


def _validate_index_range(length: int, start: int, end: int) -> None:
    if start < 0 or end < start or end > length:
        raise ValueError("invalid message index range")


__all__ = (
    "CompactionPlan",
    "CompactionTriggerReason",
    "find_safe_compaction_end",
    "initial_recent_start",
    "is_safe_compaction_range",
    "latest_user_message_index",
    "leading_instruction_end",
    "plan_compaction",
)
