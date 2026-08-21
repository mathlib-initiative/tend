from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest
from pydantic import ValidationError

from tend._common.errors import ErrorInfo
from tend.agent.config import RetryConfig
from tend.llm.retries import (
    RetryDecision,
    RetryDecisionReason,
    RetryErrorCategory,
    calculate_exponential_delay,
    calculate_retry_delay,
    decide_retry,
    parse_retry_after,
    wait_for_retry,
)


def test_exponential_delay_sequence_is_capped() -> None:
    config = RetryConfig(
        jitter=False,
        max_attempts=6,
        initial_delay_seconds=1.0,
        multiplier=2.0,
        max_delay_seconds=5.0,
    )

    assert [calculate_exponential_delay(config, attempt=attempt) for attempt in range(1, 6)] == [
        1.0,
        2.0,
        4.0,
        5.0,
        5.0,
    ]

    decisions = [
        decide_retry(config, category=RetryErrorCategory.RATE_LIMIT, attempt=attempt)
        for attempt in range(1, 7)
    ]

    assert [decision.delay_seconds for decision in decisions[:5]] == [1.0, 2.0, 4.0, 5.0, 5.0]
    assert decisions[0].next_attempt == 2
    assert decisions[4].next_attempt == 6
    assert decisions[5].should_retry is False
    assert decisions[5].reason is RetryDecisionReason.MAX_ATTEMPTS_EXHAUSTED


def test_disabled_retries_do_not_schedule_even_for_retryable_categories() -> None:
    config = RetryConfig(enabled=False)

    decision = decide_retry(config, category=RetryErrorCategory.CONNECTION_ERROR, attempt=1)

    assert decision.should_retry is False
    assert decision.reason is RetryDecisionReason.DISABLED
    assert decision.delay_seconds is None
    assert decision.next_attempt is None


def test_nonretryable_and_unsafe_categories_do_not_schedule() -> None:
    config = RetryConfig(jitter=False)

    context_overflow = decide_retry(
        config,
        category=RetryErrorCategory.CONTEXT_OVERFLOW,
        attempt=1,
    )
    unsafe_protocol = decide_retry(
        config,
        category=RetryErrorCategory.PROTOCOL_ERROR,
        attempt=1,
        safe_to_retry=False,
    )

    assert context_overflow.should_retry is False
    assert context_overflow.reason is RetryDecisionReason.NON_RETRYABLE_CATEGORY
    assert unsafe_protocol.should_retry is False
    assert unsafe_protocol.reason is RetryDecisionReason.UNSAFE_TO_RETRY


def test_retry_after_delta_and_date_are_respected() -> None:
    now = datetime(2026, 5, 4, 12, 0, tzinfo=UTC)
    http_date = format_datetime(now + timedelta(seconds=4), usegmt=True)
    config = RetryConfig(
        jitter=True,
        respect_retry_after=True,
        max_retry_after_seconds=10.0,
        initial_delay_seconds=1.0,
    )

    delta_decision = decide_retry(
        config,
        category=RetryErrorCategory.RATE_LIMIT,
        attempt=1,
        retry_after="7",
        random_source=lambda: pytest.fail("Retry-After delay must not be jittered"),
    )
    date_decision = decide_retry(
        config,
        category=RetryErrorCategory.RATE_LIMIT,
        attempt=1,
        retry_after=http_date,
        now=now,
        random_source=lambda: pytest.fail("Retry-After delay must not be jittered"),
    )

    assert delta_decision.should_retry is True
    assert delta_decision.delay_seconds == 7.0
    assert delta_decision.retry_after_seconds == 7.0
    assert date_decision.should_retry is True
    assert date_decision.delay_seconds == 4.0
    assert date_decision.retry_after_seconds == 4.0
    assert parse_retry_after("not a retry after", now=now) is None
    past_retry_after = format_datetime(now - timedelta(seconds=3), usegmt=True)
    assert parse_retry_after(past_retry_after, now=now) == 0.0


def test_retry_after_above_configured_max_fails_clearly() -> None:
    config = RetryConfig(
        jitter=False,
        respect_retry_after=True,
        max_retry_after_seconds=5.0,
    )

    decision = decide_retry(
        config,
        category=RetryErrorCategory.RATE_LIMIT,
        attempt=1,
        retry_after="6",
    )

    assert decision.should_retry is False
    assert decision.reason is RetryDecisionReason.RETRY_AFTER_TOO_LONG
    assert decision.retry_after_seconds == 6.0


def test_retry_after_can_be_ignored_by_config() -> None:
    config = RetryConfig(
        jitter=False,
        respect_retry_after=False,
        initial_delay_seconds=2.0,
        max_delay_seconds=10.0,
    )

    decision = decide_retry(
        config,
        category=RetryErrorCategory.RATE_LIMIT,
        attempt=2,
        retry_after="30",
    )

    assert decision.should_retry is True
    assert decision.delay_seconds == 4.0
    assert decision.retry_after_seconds == 30.0


def test_jitter_uses_deterministic_injected_random_source() -> None:
    config = RetryConfig(
        jitter=True,
        initial_delay_seconds=8.0,
        multiplier=2.0,
        max_delay_seconds=60.0,
    )

    delay = calculate_retry_delay(config, attempt=1, random_source=lambda: 0.25)
    decision = decide_retry(
        config,
        category=RetryErrorCategory.OVERLOADED,
        attempt=2,
        random_source=lambda: 0.5,
    )

    assert delay == 2.0
    assert decision.should_retry is True
    assert decision.delay_seconds == 8.0

    with pytest.raises(ValueError, match="random source"):
        calculate_retry_delay(config, attempt=1, random_source=lambda: 2.0)


def test_retry_decision_preserves_error_and_request_metadata() -> None:
    config = RetryConfig(jitter=False)
    error = ErrorInfo(code="rate_limit", message="provider asked us to slow down")

    decision = decide_retry(
        config,
        category=RetryErrorCategory.RATE_LIMIT,
        attempt=1,
        request_id="req_1",
        error=error,
    )

    assert decision.should_retry is True
    assert decision.request_id == "req_1"
    assert decision.error == error
    dumped = decision.model_dump(mode="json")
    assert dumped["attempt"] == 1
    assert dumped["next_attempt"] == 2
    assert dumped["delay_seconds"] == 1.0
    assert dumped["error"] == {
        "code": "rate_limit",
        "message": "provider asked us to slow down",
        "details": {},
    }


def test_retry_decision_shape_is_validated() -> None:
    with pytest.raises(ValidationError, match="next_attempt"):
        RetryDecision(
            should_retry=True,
            reason=RetryDecisionReason.SCHEDULED,
            category=RetryErrorCategory.RATE_LIMIT,
            attempt=1,
            max_attempts=3,
            next_attempt=3,
            delay_seconds=1.0,
        )

    with pytest.raises(ValidationError, match="non-retry decisions"):
        RetryDecision(
            should_retry=False,
            reason=RetryDecisionReason.DISABLED,
            category=RetryErrorCategory.RATE_LIMIT,
            attempt=1,
            max_attempts=3,
            next_attempt=2,
        )


async def test_wait_for_retry_uses_injected_async_sleeper() -> None:
    config = RetryConfig(jitter=False, initial_delay_seconds=1.5)
    decision = decide_retry(config, category=RetryErrorCategory.TIMEOUT, attempt=1)
    calls: list[float] = []

    async def fake_sleep(delay_seconds: float) -> None:
        calls.append(delay_seconds)

    await wait_for_retry(decision, sleeper=fake_sleep)
    await wait_for_retry(
        decide_retry(RetryConfig(enabled=False), category=RetryErrorCategory.TIMEOUT, attempt=1),
        sleeper=fake_sleep,
    )

    assert calls == [1.5]
