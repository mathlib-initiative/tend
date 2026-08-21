"""Secret-source and provider request header value handling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Literal

from pydantic import Field, SecretStr, field_serializer, field_validator, model_validator

from tend._common.env import ENV_NAME_PATTERN
from tend._common.errors import ConfigurationError
from tend._common.types import StrictModel

REDACTED_VALUE = "[REDACTED]"


class HeaderValueSource(StrEnum):
    """Where a provider request header value comes from."""

    LITERAL = "literal"
    ENV = "env"


class SecretSourceKind(StrEnum):
    """Secret source kinds supported without provider/network behavior."""

    LITERAL = "literal"
    ENV = "env"


class SecretValue(StrictModel):
    """Resolved secret value with safe default representation/serialization.

    The raw value is intentionally accessible only through ``reveal_value()`` for
    request construction. Pydantic's ``SecretStr`` keeps repr and JSON dumps
    redacted for diagnostics and tests.
    """

    value: SecretStr = Field(repr=False)
    source_kind: SecretSourceKind
    source_name: str | None = Field(default=None, min_length=1)

    def reveal_value(self) -> str:
        """Return the raw secret for immediate request construction."""

        return self.value.get_secret_value()


class EnvironmentSecretSource(StrictModel):
    """Environment-sourced secret descriptor that stores only the variable name."""

    kind: Literal[SecretSourceKind.ENV] = SecretSourceKind.ENV
    env_var: str = Field(min_length=1, pattern=ENV_NAME_PATTERN.pattern)

    def resolve(self, environment: Mapping[str, str]) -> SecretValue:
        """Resolve the source from an explicit environment mapping."""

        return resolve_environment_secret(self, environment)


class LiteralSecretSource(StrictModel):
    """Runtime-only literal secret descriptor with redacted serialization."""

    kind: Literal[SecretSourceKind.LITERAL] = SecretSourceKind.LITERAL
    value: SecretStr = Field(repr=False)

    def resolve(self) -> SecretValue:
        """Resolve the literal secret without exposing it in repr/serialization."""

        return SecretValue(value=self.value, source_kind=SecretSourceKind.LITERAL)


class ProviderHeaderConfig(StrictModel):
    """Runtime provider request header descriptor.

    Headers distinguish literal non-secret values, literal secret values, and
    environment-sourced secret values. Literal secret resolution is guarded by an
    explicit ``allow_literal_secrets`` flag so provider adapters do not
    accidentally accept durable or logged secret material.
    """

    name: str = Field(min_length=1)
    source: HeaderValueSource
    value: str | None = Field(default=None, min_length=1, repr=False)
    env_var: str | None = Field(default=None, min_length=1, pattern=ENV_NAME_PATTERN.pattern)
    secret: bool = True

    @field_validator("name")
    @classmethod
    def _validate_header_name(cls, name: str) -> str:
        if any(ord(char) < 33 or ord(char) > 126 or char == ":" for char in name):
            raise ValueError("header names must be visible ASCII without ':'")
        return name

    @model_validator(mode="after")
    def _validate_source_fields(self) -> ProviderHeaderConfig:
        if self.source is HeaderValueSource.LITERAL:
            if self.value is None or self.env_var is not None:
                raise ValueError("literal headers require value and must not set env_var")
        if self.source is HeaderValueSource.ENV:
            if self.env_var is None or self.value is not None:
                raise ValueError("env headers require env_var and must not set value")
        return self

    @field_serializer("value", when_used="json")
    def _serialize_value(self, value: str | None) -> str | None:
        if value is not None and self.is_sensitive:
            return REDACTED_VALUE
        return value

    @property
    def is_sensitive(self) -> bool:
        """Return whether logs/artifacts should redact this header value."""

        return self.secret or self.source is HeaderValueSource.ENV

    def resolve(
        self,
        environment: Mapping[str, str],
        *,
        allow_literal_secrets: bool = False,
    ) -> ResolvedHeaderValue:
        """Resolve this descriptor into a request-ready header value."""

        return resolve_provider_header(
            self,
            environment,
            allow_literal_secrets=allow_literal_secrets,
        )

    def safe_dump(self) -> dict[str, object]:
        """Return a diagnostic-safe serialized descriptor."""

        data = self.model_dump(mode="json")
        if self.is_sensitive and self.source is HeaderValueSource.LITERAL:
            data["value"] = REDACTED_VALUE
        return data


class ResolvedHeaderValue(StrictModel):
    """Request-ready provider header value with safe repr/serialization."""

    name: str = Field(min_length=1)
    value: SecretStr = Field(repr=False)
    source: HeaderValueSource
    source_name: str | None = Field(default=None, min_length=1)
    secret: bool = True

    @field_validator("name")
    @classmethod
    def _validate_header_name(cls, name: str) -> str:
        if any(ord(char) < 33 or ord(char) > 126 or char == ":" for char in name):
            raise ValueError("header names must be visible ASCII without ':'")
        return name

    @field_serializer("value")
    def _serialize_value(self, value: SecretStr) -> str:
        if self.secret:
            return REDACTED_VALUE
        return value.get_secret_value()

    def reveal_value(self) -> str:
        """Return the raw header value for immediate request construction."""

        return self.value.get_secret_value()

    def as_header_pair(self) -> tuple[str, str]:
        """Return ``(name, value)`` for HTTP request construction."""

        return (self.name, self.reveal_value())


def _empty_resolved_headers() -> list[ResolvedHeaderValue]:
    return []


class ResolvedHeaders(StrictModel):
    """Small wrapper around resolved request headers with safe serialization."""

    headers: list[ResolvedHeaderValue] = Field(default_factory=_empty_resolved_headers)

    def as_dict(self) -> dict[str, str]:
        """Return raw headers for immediate request construction."""

        return {header.name: header.reveal_value() for header in self.headers}


def resolve_environment_secret(
    source: EnvironmentSecretSource, environment: Mapping[str, str]
) -> SecretValue:
    """Resolve one environment-sourced secret from an explicit mapping."""

    try:
        value = environment[source.env_var]
    except KeyError as exc:
        raise ConfigurationError(
            f"required secret environment variable is missing: {source.env_var}"
        ) from exc
    if value == "":
        raise ConfigurationError(
            f"required secret environment variable is empty: {source.env_var}"
        )
    return SecretValue(
        value=SecretStr(value),
        source_kind=SecretSourceKind.ENV,
        source_name=source.env_var,
    )


def resolve_provider_header(
    header: ProviderHeaderConfig,
    environment: Mapping[str, str],
    *,
    allow_literal_secrets: bool = False,
) -> ResolvedHeaderValue:
    """Resolve one provider header descriptor for request construction."""

    if header.source is HeaderValueSource.LITERAL:
        if header.value is None:  # Defensive; model validation normally prevents this.
            raise ConfigurationError(f"literal header {header.name!r} is missing a value")
        if header.secret and not allow_literal_secrets:
            raise ConfigurationError(
                f"literal secret header {header.name!r} is not allowed by runtime config"
            )
        return ResolvedHeaderValue(
            name=header.name,
            value=SecretStr(header.value),
            source=HeaderValueSource.LITERAL,
            secret=header.secret,
        )

    if header.env_var is None:  # Defensive; model validation normally prevents this.
        raise ConfigurationError(f"environment header {header.name!r} is missing env_var")
    secret = resolve_environment_secret(
        EnvironmentSecretSource(env_var=header.env_var),
        environment,
    )
    return ResolvedHeaderValue(
        name=header.name,
        value=SecretStr(secret.reveal_value()),
        source=HeaderValueSource.ENV,
        source_name=header.env_var,
        secret=True,
    )


def resolve_provider_headers(
    headers: Iterable[ProviderHeaderConfig],
    environment: Mapping[str, str],
    *,
    allow_literal_secrets: bool = False,
) -> ResolvedHeaders:
    """Resolve provider header descriptors without logging raw values."""

    return ResolvedHeaders(
        headers=[
            resolve_provider_header(
                header,
                environment,
                allow_literal_secrets=allow_literal_secrets,
            )
            for header in headers
        ]
    )


__all__ = (
    "ENV_NAME_PATTERN",
    "HeaderValueSource",
    "EnvironmentSecretSource",
    "LiteralSecretSource",
    "ProviderHeaderConfig",
    "REDACTED_VALUE",
    "ResolvedHeaderValue",
    "ResolvedHeaders",
    "SecretSourceKind",
    "SecretValue",
    "resolve_environment_secret",
    "resolve_provider_header",
    "resolve_provider_headers",
)
