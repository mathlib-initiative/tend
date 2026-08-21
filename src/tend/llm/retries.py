"""Provider-neutral retry and exponential backoff helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from math import isfinite
from random import random
from typing import Annotated

from pydantic import Field, model_validator

from tend._common.errors import ErrorInfo
from tend._common.types import StrictModel
from tend.llm.config import RetryConfig

_PositiveInt = Annotated[int, Field(ge=1)]
_NonNegativeSeconds = Annotated[float, Field(ge=0)]

type RandomSource = Callable[[], float]
type SleepFunction = Callable[[float], Awaitable[None]]
type RetryAfterValue = str | int | float | None


class RetryErrorCategory(StrEnum):
    """Provider-neutral model/request failure categories used by retry policy."""

    RATE_LIMIT = "rate_limit"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    OVERLOADED = "overloaded"
    PROTOCOL_ERROR = "protocol_error"
    CONTEXT_OVERFLOW = "context_overflow"
    UNSUPPORTED_PARAMETER = "unsupported_parameter"
    CONTINUATION_UNAVAILABLE = "continuation_unavailable"
    NON_RETRYABLE = "non_retryable"


RETRYABLE_ERROR_CATEGORIES: frozenset[RetryErrorCategory] = frozenset(
    {
        RetryErrorCategory.RATE_LIMIT,
        RetryErrorCategory.CONNECTION_ERROR,
        RetryErrorCategory.TIMEOUT,
        RetryErrorCategory.SERVER_ERROR,
        RetryErrorCategory.OVERLOADED,
        RetryErrorCategory.PROTOCOL_ERROR,
    }
)


class RetryDecisionReason(StrEnum):
    """Machine-readable reason for a retry policy decision."""

    SCHEDULED = "scheduled"
    DISABLED = "disabled"
    MAX_ATTEMPTS_EXHAUSTED = "max_attempts_exhausted"
    NON_RETRYABLE_CATEGORY = "non_retryable_category"
    UNSAFE_TO_RETRY = "unsafe_to_retry"
    RETRY_AFTER_TOO_LONG = "retry_after_too_long"


class RetryDecision(StrictModel):
    """Structured retry decision suitable for event payloads and tests."""

    should_retry: bool
    reason: RetryDecisionReason
    category: RetryErrorCategory
    attempt: _PositiveInt
    max_attempts: _PositiveInt
    next_attempt: _PositiveInt | None = None
    delay_seconds: _NonNegativeSeconds | None = None
    retry_after_seconds: _NonNegativeSeconds | None = None
    request_id: str | None = Field(default=None, min_length=1)
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> RetryDecision:
        if self.should_retry:
            if self.reason is not RetryDecisionReason.SCHEDULED:
                raise ValueError("scheduled retry decisions must use the scheduled reason")
            if self.next_attempt != self.attempt + 1:
                raise ValueError("next_attempt must equal attempt + 1 for scheduled retries")
            if self.delay_seconds is None:
                raise ValueError("scheduled retry decisions require delay_seconds")
        else:
            if self.next_attempt is not None:
                raise ValueError("non-retry decisions must not include next_attempt")
            if self.delay_seconds is not None:
                raise ValueError("non-retry decisions must not include delay_seconds")
        return self


def is_retryable_category(category: RetryErrorCategory) -> bool:
    """Return whether a category is retryable when the operation is safe to retry."""

    return category in RETRYABLE_ERROR_CATEGORIES


def parse_retry_after(value: RetryAfterValue, *, now: datetime | None = None) -> float | None:
    """Parse a provider ``Retry-After`` value into seconds.

    Supports delta-second values and HTTP-date values. Invalid values return
    ``None`` so callers can fall back to local exponential backoff policy.
    Dates in the past are treated as an immediate retry hint of ``0.0``.
    """

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return _valid_non_negative_seconds(float(value))

    stripped = value.strip()
    if not stripped:
        return None

    numeric_seconds = _parse_numeric_retry_after(stripped)
    if numeric_seconds is not None:
        return numeric_seconds

    try:
        retry_at = parsedate_to_datetime(stripped)
    except (TypeError, ValueError):
        return None
    if retry_at.tzinfo is None or retry_at.utcoffset() is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    current = _coerce_aware_utc(now) if now is not None else datetime.now(UTC)
    return max(0.0, (retry_at.astimezone(UTC) - current).total_seconds())


def calculate_exponential_delay(config: RetryConfig, *, attempt: int) -> float:
    """Calculate the capped exponential delay for a failed 1-based attempt."""

    _validate_attempt(attempt)
    delay = config.initial_delay_seconds * (config.multiplier ** (attempt - 1))
    return float(min(delay, config.max_delay_seconds))


def calculate_retry_delay(
    config: RetryConfig,
    *,
    attempt: int,
    retry_after_seconds: float | None = None,
    random_source: RandomSource | None = None,
) -> float:
    """Calculate the sleep delay for a scheduled retry.

    A valid provider ``Retry-After`` hint is used exactly when configured; local
    jitter is applied only to locally calculated exponential delays.
    """

    _validate_attempt(attempt)
    if config.respect_retry_after and retry_after_seconds is not None:
        if retry_after_seconds < 0 or not isfinite(retry_after_seconds):
            raise ValueError("retry_after_seconds must be finite and non-negative")
        return float(retry_after_seconds)

    delay = calculate_exponential_delay(config, attempt=attempt)
    if not config.jitter:
        return delay
    source = random if random_source is None else random_source
    factor = source()
    if factor < 0.0 or factor > 1.0 or not isfinite(factor):
        raise ValueError("retry jitter random source must return a finite value in [0.0, 1.0]")
    return delay * factor


def decide_retry(
    config: RetryConfig,
    *,
    category: RetryErrorCategory,
    attempt: int,
    retry_after: RetryAfterValue = None,
    safe_to_retry: bool = True,
    request_id: str | None = None,
    error: ErrorInfo | None = None,
    now: datetime | None = None,
    random_source: RandomSource | None = None,
) -> RetryDecision:
    """Return the retry policy decision after one failed request attempt."""

    _validate_attempt(attempt)
    retry_after_seconds = parse_retry_after(retry_after, now=now)

    if not config.enabled:
        return _no_retry(
            reason=RetryDecisionReason.DISABLED,
            category=category,
            attempt=attempt,
            config=config,
            retry_after_seconds=retry_after_seconds,
            request_id=request_id,
            error=error,
        )
    if not safe_to_retry:
        return _no_retry(
            reason=RetryDecisionReason.UNSAFE_TO_RETRY,
            category=category,
            attempt=attempt,
            config=config,
            retry_after_seconds=retry_after_seconds,
            request_id=request_id,
            error=error,
        )
    if not is_retryable_category(category):
        return _no_retry(
            reason=RetryDecisionReason.NON_RETRYABLE_CATEGORY,
            category=category,
            attempt=attempt,
            config=config,
            retry_after_seconds=retry_after_seconds,
            request_id=request_id,
            error=error,
        )
    if attempt >= config.max_attempts:
        return _no_retry(
            reason=RetryDecisionReason.MAX_ATTEMPTS_EXHAUSTED,
            category=category,
            attempt=attempt,
            config=config,
            retry_after_seconds=retry_after_seconds,
            request_id=request_id,
            error=error,
        )
    if _retry_after_exceeds_configured_max(config, retry_after_seconds):
        return _no_retry(
            reason=RetryDecisionReason.RETRY_AFTER_TOO_LONG,
            category=category,
            attempt=attempt,
            config=config,
            retry_after_seconds=retry_after_seconds,
            request_id=request_id,
            error=error,
        )

    delay_seconds = calculate_retry_delay(
        config,
        attempt=attempt,
        retry_after_seconds=retry_after_seconds,
        random_source=random_source,
    )
    return RetryDecision(
        should_retry=True,
        reason=RetryDecisionReason.SCHEDULED,
        category=category,
        attempt=attempt,
        max_attempts=config.max_attempts,
        next_attempt=attempt + 1,
        delay_seconds=delay_seconds,
        retry_after_seconds=retry_after_seconds,
        request_id=request_id,
        error=error,
    )


async def sleep_for_retry(
    delay_seconds: float,
    *,
    sleeper: SleepFunction | None = None,
) -> None:
    """Sleep for a retry delay using an injectable async sleeper."""

    if delay_seconds < 0.0 or not isfinite(delay_seconds):
        raise ValueError("retry sleep delay must be finite and non-negative")
    sleep_impl = asyncio.sleep if sleeper is None else sleeper
    await sleep_impl(delay_seconds)


async def wait_for_retry(
    decision: RetryDecision,
    *,
    sleeper: SleepFunction | None = None,
) -> None:
    """Sleep for a scheduled retry decision; no-op for non-retry decisions."""

    if not decision.should_retry:
        return
    if decision.delay_seconds is None:
        raise ValueError("scheduled retry decision is missing delay_seconds")
    await sleep_for_retry(decision.delay_seconds, sleeper=sleeper)


def _no_retry(
    *,
    reason: RetryDecisionReason,
    category: RetryErrorCategory,
    attempt: int,
    config: RetryConfig,
    retry_after_seconds: float | None,
    request_id: str | None,
    error: ErrorInfo | None,
) -> RetryDecision:
    return RetryDecision(
        should_retry=False,
        reason=reason,
        category=category,
        attempt=attempt,
        max_attempts=config.max_attempts,
        retry_after_seconds=retry_after_seconds,
        request_id=request_id,
        error=error,
    )


def _parse_numeric_retry_after(value: str) -> float | None:
    try:
        seconds = float(value)
    except ValueError:
        return None
    return _valid_non_negative_seconds(seconds)


def _valid_non_negative_seconds(seconds: float) -> float | None:
    if not isfinite(seconds) or seconds < 0.0:
        return None
    return float(seconds)


def _coerce_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _validate_attempt(attempt: int) -> None:
    if isinstance(attempt, bool) or attempt < 1:
        raise ValueError("retry attempt must be a positive 1-based integer")


def _retry_after_exceeds_configured_max(
    config: RetryConfig,
    retry_after_seconds: float | None,
) -> bool:
    if not config.respect_retry_after or retry_after_seconds is None:
        return False
    if config.max_retry_after_seconds is None:
        return False
    return retry_after_seconds > config.max_retry_after_seconds


__all__ = (
    "RETRYABLE_ERROR_CATEGORIES",
    "RandomSource",
    "RetryAfterValue",
    "RetryConfig",
    "RetryDecision",
    "RetryDecisionReason",
    "RetryErrorCategory",
    "SleepFunction",
    "calculate_exponential_delay",
    "calculate_retry_delay",
    "decide_retry",
    "is_retryable_category",
    "parse_retry_after",
    "sleep_for_retry",
    "wait_for_retry",
)
