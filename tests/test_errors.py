import pytest
from pydantic import ValidationError

from tend._common.errors import (
    ConfigurationError,
    ErrorInfo,
    FrameworkError,
    ProviderProtocolError,
    UnsupportedSchemaVersionError,
)


def test_error_info_serialization_round_trip() -> None:
    error = ErrorInfo(
        code="configuration_error",
        message="Model provider is required",
        details={"field": "provider", "retryable": False, "nested": {"attempt": 1}},
    )

    serialized = error.model_dump_json()
    restored = ErrorInfo.model_validate_json(serialized)

    assert restored == error


def test_error_info_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ErrorInfo.model_validate(
            {"code": "bad", "message": "Invalid", "details": {}, "unexpected": "nope"}
        )


def test_framework_exceptions_are_runtime_exceptions() -> None:
    exc = ConfigurationError("Missing model provider")

    assert isinstance(exc, FrameworkError)
    assert str(exc) == "Missing model provider"


def test_framework_exception_subclasses_are_distinct_types() -> None:
    protocol_error = ProviderProtocolError("Malformed provider payload")
    version_error = UnsupportedSchemaVersionError("Unsupported events.jsonl schema version: 99")

    assert isinstance(protocol_error, FrameworkError)
    assert isinstance(version_error, FrameworkError)
    assert type(protocol_error) is ProviderProtocolError
    assert type(version_error) is UnsupportedSchemaVersionError
