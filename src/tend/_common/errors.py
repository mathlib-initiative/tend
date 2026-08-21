"""Serializable domain errors and framework exception classes."""

from __future__ import annotations

from pydantic import Field

from tend._common.types import JsonObject, StrictModel


class ErrorInfo(StrictModel):
    """Serializable public/persisted description of a domain error."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    details: JsonObject = Field(default_factory=dict)


class FrameworkError(Exception):
    """Base class for framework control/fatal errors."""


class ConfigurationError(FrameworkError):
    """Configuration is invalid or incomplete."""


class PersistenceError(FrameworkError):
    """Session persistence failed or stored data is corrupt."""


class ProviderProtocolError(FrameworkError):
    """A provider response could not be interpreted as the expected protocol."""


class UnsupportedSchemaVersionError(FrameworkError):
    """A persisted or configured schema version is not supported by this build."""


__all__ = (
    "ConfigurationError",
    "ErrorInfo",
    "FrameworkError",
    "PersistenceError",
    "ProviderProtocolError",
    "UnsupportedSchemaVersionError",
)
