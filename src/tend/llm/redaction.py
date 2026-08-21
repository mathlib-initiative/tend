"""Redaction helpers for headers, payloads, and diagnostic text."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from re import Pattern, compile, escape
from typing import cast
from urllib.parse import urlsplit, urlunsplit

from pydantic import JsonValue

from tend.llm.secrets import (
    REDACTED_VALUE,
    HeaderValueSource,
    ProviderHeaderConfig,
    ResolvedHeaderValue,
)

type JsonLike = JsonValue | Mapping[str, object] | Sequence[object]

_SECRET_SOURCE_VALUE_REDACTION = REDACTED_VALUE
_MILD_URL_REDACTION_PATH = "/[REDACTED]"


class Redactor:
    """Deterministic redactor for diagnostic strings and JSON-like payloads."""

    __slots__ = (
        "_mildly_sensitive_urls",
        "_patterns",
        "_secret_header_names",
        "_secret_source_names",
        "_secret_values",
    )

    _secret_values: tuple[str, ...]
    _secret_source_names: tuple[str, ...]
    _secret_header_names: frozenset[str]
    _patterns: tuple[Pattern[str], ...]
    _mildly_sensitive_urls: tuple[str, ...]

    def __init__(
        self,
        *,
        secret_values: Iterable[str] = (),
        secret_source_names: Iterable[str] = (),
        secret_header_names: Iterable[str] = (),
        patterns: Iterable[str | Pattern[str]] = (),
        mildly_sensitive_urls: Iterable[str] = (),
    ) -> None:
        self._secret_values = _unique_non_empty_by_length(secret_values)
        self._secret_source_names = tuple(sorted(_unique_non_empty(secret_source_names)))
        self._secret_header_names = frozenset(name.lower() for name in secret_header_names if name)
        self._patterns = tuple(
            pattern if isinstance(pattern, Pattern) else compile(pattern) for pattern in patterns
        )
        self._mildly_sensitive_urls = _url_redaction_candidates(mildly_sensitive_urls)

    def redact_text(self, text: str) -> str:
        """Redact configured secrets, source assignments, URLs, and patterns."""

        redacted = text
        for secret_value in self._secret_values:
            redacted = redacted.replace(secret_value, REDACTED_VALUE)
        for url in self._mildly_sensitive_urls:
            redacted = redacted.replace(url, redact_mildly_sensitive_url(url))
        for source_name in self._secret_source_names:
            redacted = _redact_source_assignment(redacted, source_name)
        for pattern in self._patterns:
            redacted = pattern.sub(REDACTED_VALUE, redacted)
        return redacted

    def redact_headers(self, headers: Mapping[str, str]) -> dict[str, str]:
        """Redact configured secret/env-sourced request headers."""

        result: dict[str, str] = {}
        for name, value in headers.items():
            if name.lower() in self._secret_header_names:
                result[name] = REDACTED_VALUE
            else:
                result[name] = self.redact_text(value)
        return result

    def redact_payload(self, payload: object) -> object:
        """Recursively redact strings in a JSON-like payload."""

        if isinstance(payload, str):
            return self.redact_text(payload)
        if isinstance(payload, Mapping):
            mapping = cast(Mapping[object, object], payload)
            redacted_mapping: dict[object, object] = {}
            for key, value in mapping.items():
                key_text = str(key)
                if self._is_secret_key(key_text):
                    redacted_mapping[key] = REDACTED_VALUE
                else:
                    redacted_mapping[key] = self.redact_payload(value)
            return redacted_mapping
        if isinstance(payload, list):
            items = cast(list[object], payload)
            return [self.redact_payload(item) for item in items]
        if isinstance(payload, tuple):
            tuple_items = cast(tuple[object, ...], payload)
            return tuple(self.redact_payload(item) for item in tuple_items)
        return payload

    def _is_secret_key(self, key: str) -> bool:
        return key in self._secret_source_names or key.lower() in self._secret_header_names


class HeaderRedactionPolicy:
    """Redaction policy derived from provider header descriptors."""

    __slots__ = ("_secret_header_names",)

    _secret_header_names: frozenset[str]

    def __init__(self, secret_header_names: Iterable[str]) -> None:
        self._secret_header_names = frozenset(name.lower() for name in secret_header_names)

    @classmethod
    def from_provider_headers(
        cls, headers: Iterable[ProviderHeaderConfig]
    ) -> HeaderRedactionPolicy:
        """Build a policy from configured provider headers."""

        return cls(header.name for header in headers if header.is_sensitive)

    @classmethod
    def from_resolved_headers(
        cls, headers: Iterable[ResolvedHeaderValue]
    ) -> HeaderRedactionPolicy:
        """Build a policy from resolved request headers."""

        return cls(
            header.name
            for header in headers
            if header.secret or header.source is HeaderValueSource.ENV
        )

    def secret_header_names(self) -> tuple[str, ...]:
        """Return case-folded secret header names in deterministic order."""

        return tuple(sorted(self._secret_header_names))


def redact_text(
    text: str,
    *,
    secret_values: Iterable[str] = (),
    secret_source_names: Iterable[str] = (),
    patterns: Iterable[str | Pattern[str]] = (),
    mildly_sensitive_urls: Iterable[str] = (),
) -> str:
    """One-shot text redaction helper."""

    return Redactor(
        secret_values=secret_values,
        secret_source_names=secret_source_names,
        patterns=patterns,
        mildly_sensitive_urls=mildly_sensitive_urls,
    ).redact_text(text)


def redact_headers(
    headers: Mapping[str, str],
    *,
    secret_header_names: Iterable[str] = (),
    secret_values: Iterable[str] = (),
    secret_source_names: Iterable[str] = (),
    patterns: Iterable[str | Pattern[str]] = (),
) -> dict[str, str]:
    """One-shot request-header redaction helper."""

    return Redactor(
        secret_header_names=secret_header_names,
        secret_values=secret_values,
        secret_source_names=secret_source_names,
        patterns=patterns,
    ).redact_headers(headers)


def redact_provider_headers(
    headers: Mapping[str, str],
    header_configs: Iterable[ProviderHeaderConfig],
    *,
    secret_values: Iterable[str] = (),
) -> dict[str, str]:
    """Redact raw headers using provider header config descriptors."""

    policy = HeaderRedactionPolicy.from_provider_headers(header_configs)
    return redact_headers(
        headers,
        secret_header_names=policy.secret_header_names(),
        secret_values=secret_values,
    )


def redact_resolved_headers(headers: Iterable[ResolvedHeaderValue]) -> dict[str, str]:
    """Return a redacted dict from resolved request header wrappers."""

    resolved = list(headers)
    policy = HeaderRedactionPolicy.from_resolved_headers(resolved)
    raw_headers = {header.name: header.reveal_value() for header in resolved}
    return redact_headers(raw_headers, secret_header_names=policy.secret_header_names())


def redact_payload(
    payload: object,
    *,
    secret_values: Iterable[str] = (),
    secret_source_names: Iterable[str] = (),
    secret_header_names: Iterable[str] = (),
    patterns: Iterable[str | Pattern[str]] = (),
    mildly_sensitive_urls: Iterable[str] = (),
) -> object:
    """One-shot JSON-like payload redaction helper."""

    return Redactor(
        secret_values=secret_values,
        secret_source_names=secret_source_names,
        secret_header_names=secret_header_names,
        patterns=patterns,
        mildly_sensitive_urls=mildly_sensitive_urls,
    ).redact_payload(payload)


def redact_mildly_sensitive_url(url: str) -> str:
    """Redact account/gateway identifiers while preserving scheme and host."""

    parsed = urlsplit(url)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return urlunsplit((parsed.scheme, parsed.netloc, _MILD_URL_REDACTION_PATH, "", ""))
    return REDACTED_VALUE


def header_names_requiring_redaction(headers: Iterable[ProviderHeaderConfig]) -> tuple[str, ...]:
    """Return provider header names whose values should be redacted."""

    return tuple(sorted(header.name for header in headers if header.is_sensitive))


def _redact_source_assignment(text: str, source_name: str) -> str:
    name = escape(source_name)
    pattern = compile(
        rf"(?P<prefix>[\"']?{name}[\"']?\s*[:=]\s*[\"']?)"
        rf"(?P<value>[^\"'\s,;}}]+)"
        rf"(?P<suffix>[\"']?)"
    )
    return pattern.sub(
        lambda match: (
            f"{match.group('prefix')}"
            f"{_SECRET_SOURCE_VALUE_REDACTION}"
            f"{match.group('suffix')}"
        ),
        text,
    )


def _unique_non_empty(values: Iterable[str]) -> frozenset[str]:
    return frozenset(value for value in values if value)


def _unique_non_empty_by_length(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(_unique_non_empty(values), key=lambda value: (-len(value), value)))


def _url_redaction_candidates(values: Iterable[str]) -> tuple[str, ...]:
    candidates: set[str] = set()
    for value in values:
        if not value:
            continue
        candidates.add(value)
        stripped = value.rstrip("/")
        if stripped:
            candidates.add(stripped)
            candidates.add(f"{stripped}/")
    return tuple(sorted(candidates, key=lambda value: (-len(value), value)))


__all__ = (
    "HeaderRedactionPolicy",
    "JsonLike",
    "Redactor",
    "header_names_requiring_redaction",
    "redact_headers",
    "redact_mildly_sensitive_url",
    "redact_payload",
    "redact_provider_headers",
    "redact_resolved_headers",
    "redact_text",
)
