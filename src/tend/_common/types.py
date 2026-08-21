"""Common Pydantic, ID, timestamp, and JSON types shared across modules."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from re import Pattern, compile
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, JsonValue

type JsonObject = dict[str, JsonValue]

_ID_PREFIX_PATTERN: Pattern[str] = compile(r"^[a-z][a-z0-9_]*$")


class StrictModel(BaseModel):
    """Base class for public and persisted Pydantic schemas.

    The default posture is intentionally strict: schemas reject unknown fields and
    avoid implicit type coercion. Later modules can opt into explicit metadata
    fields such as ``dict[str, JsonValue]`` when provider-specific details must
    be preserved.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class StopReason(StrEnum):
    """Structured turn stop reasons used across loops, sessions, and results."""

    FINAL_RESPONSE = "final_response"
    FINAL_RESULT = "final_result"
    PROVIDER_STOP_REASON = "provider_stop_reason"
    MAX_MODEL_REQUESTS = "max_model_requests"
    MAX_TOOL_CALLS = "max_tool_calls"
    MAX_ITERATIONS = "max_iterations"
    MAX_WALL_TIME = "max_wall_time"
    MAX_TOKENS = "max_tokens"
    MAX_COST = "max_cost"
    MODEL_ERROR = "model_error"
    INTERRUPTED = "interrupted"
    COMPACTION_FAILED = "compaction_failed"


class IdGenerator:
    """Small deterministic monotonic ID generator.

    Later persistence and turn-loop code can inject an instance in tests instead
    of depending on the module-level helper functions.
    """

    __slots__ = ("_next",)

    _next: int

    def __init__(self, *, start: int = 1) -> None:
        if start < 0:
            raise ValueError("ID generator start must be non-negative")
        self._next = start

    def next_sequence_id(self) -> int:
        """Return the next monotonic integer sequence ID."""

        sequence = self._next
        self._next += 1
        return sequence

    def new_id(self, prefix: str, *, width: int = 6) -> str:
        """Return a prefixed monotonic string ID."""

        return format_sequence_id(prefix, self.next_sequence_id(), width=width)

    def new_event_id(self) -> str:
        """Return a monotonic event ID."""

        return self.new_id("evt")

    def advance_to(self, min_next: int) -> None:
        """Advance the counter so the next value produced is >= min_next."""

        if min_next < 0:
            raise ValueError("ID generator minimum must be non-negative")
        self._next = max(self._next, min_next)


_default_id_generator = IdGenerator()


def format_sequence_id(prefix: str, sequence: int, *, width: int = 6) -> str:
    """Format a prefix and sequence number as a stable sortable string ID."""

    if not _ID_PREFIX_PATTERN.fullmatch(prefix):
        raise ValueError("ID prefix must match [a-z][a-z0-9_]*")
    if sequence < 0:
        raise ValueError("ID sequence must be non-negative")
    return f"{prefix}_{sequence:0{width}d}"


def next_sequence_id() -> int:
    """Return the next module-level monotonic integer sequence ID."""

    return _default_id_generator.next_sequence_id()


def new_id(prefix: str, *, width: int = 6) -> str:
    """Return the next module-level prefixed monotonic string ID."""

    return _default_id_generator.new_id(prefix, width=width)


def new_event_id() -> str:
    """Return the next module-level monotonic event ID."""

    return _default_id_generator.new_event_id()


def advance_id_counter(min_next: int) -> None:
    """Advance the module-level counter so the next ID allocation is >= min_next.

    Call this when resuming a session to prevent ID collisions with IDs that
    were already allocated in a previous process run.
    """

    _default_id_generator.advance_to(min_next)


def utc_now() -> datetime:
    """Return an aware UTC datetime for persisted timestamps."""

    return datetime.now(UTC)


def format_utc_timestamp(value: datetime) -> str:
    """Return an ISO-8601 UTC timestamp string using a trailing ``Z``."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def utc_timestamp() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""

    return format_utc_timestamp(utc_now())


__all__ = (
    "IdGenerator",
    "JsonObject",
    "StopReason",
    "StrictModel",
    "format_sequence_id",
    "format_utc_timestamp",
    "new_event_id",
    "new_id",
    "advance_id_counter",
    "next_sequence_id",
    "utc_now",
    "utc_timestamp",
)
