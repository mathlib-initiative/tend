"""Usage and cost schemas shared by model requests, turns, and sessions."""

from __future__ import annotations

from decimal import Decimal

from tend._common.usage import Cost, TokenUsage, Usage
from tend.llm.models.profiles import TokenPricing

_MILLION = Decimal("1000000")


def calculate_token_cost(tokens: TokenUsage, pricing: TokenPricing | None) -> Cost | None:
    """Calculate token cost from configured per-million-token pricing.

    ``None`` is returned when no pricing is configured or when there are no
    reported tokens for priced categories. This avoids fabricating a monetary
    value from missing provider usage.
    """

    if pricing is None:
        return None

    amount = Decimal("0")
    billable_tokens = 0
    amount += _priced_component(tokens.input_tokens, pricing.input_per_million_tokens)
    if pricing.input_per_million_tokens is not None:
        billable_tokens += tokens.input_tokens
    amount += _priced_component(tokens.output_tokens, pricing.output_per_million_tokens)
    if pricing.output_per_million_tokens is not None:
        billable_tokens += tokens.output_tokens
    amount += _priced_component(tokens.reasoning_tokens, pricing.reasoning_per_million_tokens)
    if pricing.reasoning_per_million_tokens is not None:
        billable_tokens += tokens.reasoning_tokens
    amount += _priced_component(tokens.cache_read_tokens, pricing.cache_read_per_million_tokens)
    if pricing.cache_read_per_million_tokens is not None:
        billable_tokens += tokens.cache_read_tokens
    amount += _priced_component(tokens.cache_write_tokens, pricing.cache_write_per_million_tokens)
    if pricing.cache_write_per_million_tokens is not None:
        billable_tokens += tokens.cache_write_tokens

    if billable_tokens == 0:
        return None
    return Cost(amount=amount, currency=pricing.currency, pricing_source=pricing.source)


def usage_with_model_request_count(usage: Usage) -> Usage:
    """Return usage with at least one model request counted."""

    if usage.model_requests >= 1:
        return usage.model_copy(deep=True)
    return usage.model_copy(update={"model_requests": 1}, deep=True)


def usage_with_retry_attempt_count(usage: Usage) -> Usage:
    """Return usage with at least one retry attempt counted."""

    if usage.retry_attempts >= 1:
        return usage.model_copy(deep=True)
    return usage.model_copy(update={"retry_attempts": 1}, deep=True)


def _priced_component(token_count: int, per_million_tokens: Decimal | None) -> Decimal:
    if per_million_tokens is None or token_count == 0:
        return Decimal("0")
    return Decimal(token_count) * per_million_tokens / _MILLION


__all__ = (
    "Cost",
    "TokenUsage",
    "Usage",
    "calculate_token_cost",
    "usage_with_model_request_count",
    "usage_with_retry_attempt_count",
)
