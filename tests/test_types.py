from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from tend._common.types import (
    IdGenerator,
    StopReason,
    StrictModel,
    format_sequence_id,
    format_utc_timestamp,
    utc_now,
)


class StopReasonExample(StrictModel):
    reason: StopReason


def test_stop_reason_serializes_as_string() -> None:
    model = StopReasonExample(reason=StopReason.FINAL_RESPONSE)

    assert model.model_dump(mode="json") == {"reason": "final_response"}
    assert StopReasonExample.model_validate_json(model.model_dump_json()) == model


def test_stop_reason_rejects_unknown_values() -> None:
    with pytest.raises(ValidationError):
        StopReasonExample.model_validate({"reason": "not_a_stop_reason"})


def test_id_generator_produces_monotonic_sortable_ids() -> None:
    generator = IdGenerator(start=7)

    assert generator.next_sequence_id() == 7
    assert generator.new_id("turn") == "turn_000008"
    assert generator.new_event_id() == "evt_000009"


def test_id_generator_advance_to_never_rewinds() -> None:
    generator = IdGenerator(start=100)

    assert generator.new_event_id() == "evt_000100"
    generator.advance_to(10)
    assert generator.new_event_id() == "evt_000101"

    generator.advance_to(200)
    assert generator.new_event_id() == "evt_000200"


def test_format_sequence_id_validates_inputs() -> None:
    assert format_sequence_id("event", 42) == "event_000042"
    assert format_sequence_id("event", 42, width=3) == "event_042"

    with pytest.raises(ValueError):
        format_sequence_id("Event", 1)
    with pytest.raises(ValueError):
        format_sequence_id("event", -1)


def test_utc_now_returns_timezone_aware_utc_datetime() -> None:
    now = utc_now()

    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_format_utc_timestamp_normalizes_to_z_suffix() -> None:
    non_utc = datetime(2026, 5, 4, 12, 30, tzinfo=timezone(timedelta(hours=2)))

    assert format_utc_timestamp(non_utc) == "2026-05-04T10:30:00Z"

    with pytest.raises(ValueError):
        format_utc_timestamp(datetime(2026, 5, 4, 10, 30))


def test_format_utc_timestamp_accepts_utc_datetimes() -> None:
    timestamp = format_utc_timestamp(datetime(2026, 5, 4, 10, 30, tzinfo=UTC))

    assert timestamp == "2026-05-04T10:30:00Z"
