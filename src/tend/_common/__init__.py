"""Internal dependency-free primitives shared by tend layers."""

from tend._common.errors import (
    ConfigurationError,
    ErrorInfo,
    FrameworkError,
    PersistenceError,
    ProviderProtocolError,
    UnsupportedSchemaVersionError,
)
from tend._common.types import (
    IdGenerator,
    JsonObject,
    StopReason,
    StrictModel,
    advance_id_counter,
    format_sequence_id,
    format_utc_timestamp,
    new_event_id,
    new_id,
    next_sequence_id,
    utc_now,
    utc_timestamp,
)

__all__ = (
    "ConfigurationError",
    "ErrorInfo",
    "FrameworkError",
    "IdGenerator",
    "JsonObject",
    "PersistenceError",
    "ProviderProtocolError",
    "StopReason",
    "StrictModel",
    "UnsupportedSchemaVersionError",
    "advance_id_counter",
    "format_sequence_id",
    "format_utc_timestamp",
    "new_event_id",
    "new_id",
    "next_sequence_id",
    "utc_now",
    "utc_timestamp",
)
