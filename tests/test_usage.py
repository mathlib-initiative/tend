from decimal import Decimal

import pytest
from pydantic import ValidationError

from tend.llm.usage import Cost, TokenUsage, Usage


def test_token_usage_aggregation_preserves_provider_details_deterministically() -> None:
    left = TokenUsage(
        input_tokens=1,
        output_tokens=2,
        reasoning_tokens=3,
        cache_read_tokens=4,
        cache_write_tokens=5,
        provider_details={"z_tokens": 6, "audio_tokens": 7},
    )
    right = TokenUsage(
        input_tokens=10,
        output_tokens=20,
        reasoning_tokens=30,
        cache_read_tokens=40,
        cache_write_tokens=50,
        provider_details={"audio_tokens": 70, "other_tokens": 80},
    )

    combined = left.add(right)

    assert combined == TokenUsage(
        input_tokens=11,
        output_tokens=22,
        reasoning_tokens=33,
        cache_read_tokens=44,
        cache_write_tokens=55,
        provider_details={"audio_tokens": 77, "other_tokens": 80, "z_tokens": 6},
    )
    assert list(combined.provider_details) == ["audio_tokens", "other_tokens", "z_tokens"]
    assert left.provider_details == {"z_tokens": 6, "audio_tokens": 7}
    assert right.provider_details == {"audio_tokens": 70, "other_tokens": 80}


def test_token_usage_json_round_trip() -> None:
    usage = TokenUsage(
        input_tokens=3,
        output_tokens=5,
        reasoning_tokens=2,
        cache_read_tokens=1,
        provider_details={"accepted_prediction_tokens": 4},
    )

    restored = TokenUsage.model_validate_json(usage.model_dump_json())

    assert restored == usage


def test_token_usage_rejects_invalid_provider_details() -> None:
    with pytest.raises(ValidationError):
        TokenUsage.model_validate({"provider_details": {"bad": -1}})

    with pytest.raises(ValidationError):
        TokenUsage.model_validate({"provider_details": {"bad": "1"}})

    with pytest.raises(ValidationError):
        TokenUsage.model_validate({"provider_details": {"": 1}})


def test_cost_aggregation_and_json_round_trip() -> None:
    first = Cost(amount=Decimal("0.01"), currency="USD", pricing_source="profile")
    second = Cost(amount=Decimal("0.02"), currency="USD", pricing_source="profile")

    combined = first.add(second)

    assert combined == Cost(amount=Decimal("0.03"), currency="USD", pricing_source="profile")
    assert Cost.model_validate_json(combined.model_dump_json()) == combined


def test_cost_aggregation_marks_mixed_pricing_sources() -> None:
    first = Cost(amount=Decimal("0.01"), currency="USD", pricing_source="profile")
    second = Cost(amount=Decimal("0.02"), currency="USD", pricing_source="provider")

    assert first.add(second) == Cost(
        amount=Decimal("0.03"),
        currency="USD",
        pricing_source="mixed",
    )


def test_cost_aggregation_rejects_currency_mismatch() -> None:
    usd = Cost(amount=Decimal("0.01"), currency="USD")
    eur = Cost(amount=Decimal("0.01"), currency="EUR")

    with pytest.raises(ValueError):
        usd.add(eur)


def test_usage_aggregation_is_side_effect_free() -> None:
    left = Usage(
        tokens=TokenUsage(input_tokens=1, provider_details={"x": 2}),
        cost=Cost(amount=Decimal("0.01"), currency="USD"),
        model_requests=1,
        retry_attempts=2,
        tool_calls=3,
    )
    right = Usage(
        tokens=TokenUsage(output_tokens=4, provider_details={"x": 5, "y": 6}),
        cost=Cost(amount=Decimal("0.02"), currency="USD"),
        model_requests=10,
        retry_attempts=20,
        tool_calls=30,
    )

    combined = left.add(right)

    assert combined == Usage(
        tokens=TokenUsage(input_tokens=1, output_tokens=4, provider_details={"x": 7, "y": 6}),
        cost=Cost(amount=Decimal("0.03"), currency="USD"),
        model_requests=11,
        retry_attempts=22,
        tool_calls=33,
    )
    assert left.tokens.provider_details == {"x": 2}
    assert right.tokens.provider_details == {"x": 5, "y": 6}


def test_usage_json_round_trip() -> None:
    usage = Usage(
        tokens=TokenUsage(input_tokens=1, output_tokens=2),
        cost=Cost(amount=Decimal("0.0123"), currency="USD", pricing_source="test"),
        model_requests=1,
        retry_attempts=0,
        tool_calls=2,
    )

    restored = Usage.model_validate_json(usage.model_dump_json())

    assert restored == usage


def test_usage_rejects_negative_counts() -> None:
    with pytest.raises(ValidationError):
        Usage.model_validate({"model_requests": -1})
