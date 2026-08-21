"""Provider error taxonomy and redacted diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from pydantic import Field, JsonValue, TypeAdapter, ValidationError

from tend._common.errors import ErrorInfo, FrameworkError
from tend._common.types import JsonObject, StrictModel
from tend.llm.redaction import Redactor
from tend.llm.retries import RetryErrorCategory

ProviderErrorCategory = RetryErrorCategory

DEFAULT_SECRET_HEADER_NAMES: tuple[str, ...] = (
    "anthropic-api-key",
    "api-key",
    "authorization",
    "cf-aig-authorization",
    "cookie",
    "openai-api-key",
    "proxy-authorization",
    "set-cookie",
    "x-api-key",
)

_CONTEXT_OVERFLOW_MARKERS: tuple[str, ...] = (
    "context_length_exceeded",
    "context overflow",
    "context window",
    "exceeds the context",
    "input is too long",
    "max context",
    "max_context_length_exceeded",
    "maximum context length",
    "prompt is too long",
    "too many tokens",
)
_CONTINUATION_MARKERS: tuple[str, ...] = (
    "previous_response_id",
    "previous response",
    "response id",
)
_CONTINUATION_FAILURE_MARKERS: tuple[str, ...] = (
    "cannot",
    "disabled",
    "does not exist",
    "not allowed",
    "not available",
    "not found",
    "rejected",
    "stored",
    "unavailable",
    "unsupported",
    "zero data retention",
    "zdr",
)
_UNSUPPORTED_PARAMETER_MARKERS: tuple[str, ...] = (
    "capability mismatch",
    "does not support",
    "invalid parameter",
    "not supported",
    "unsupported parameter",
    "unsupported_parameter",
    "unrecognized request argument",
    "unknown parameter",
)
_RATE_LIMIT_MARKERS: tuple[str, ...] = (
    "rate limit",
    "rate_limit",
    "too many requests",
)
_OVERLOADED_MARKERS: tuple[str, ...] = (
    "overloaded",
    "overloaded_error",
)
_TIMEOUT_MARKERS: tuple[str, ...] = (
    "request timeout",
    "timed out",
    "timeout",
)

_JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


class ProviderErrorInfo(StrictModel):
    """Serializable provider failure classification."""

    category: RetryErrorCategory
    message: str = Field(min_length=1)
    status_code: int | None = Field(default=None, ge=100, le=599)
    provider_error_type: str | None = Field(default=None, min_length=1)
    retry_after: str | None = Field(default=None, min_length=1)
    request_url: str | None = Field(default=None, min_length=1)
    request_headers: dict[str, str] = Field(default_factory=dict)
    response_headers: dict[str, str] = Field(default_factory=dict)
    response_body: JsonValue | None = None


class ProviderRequestError(FrameworkError):
    """Provider request failed with redacted request/response diagnostics."""

    __slots__ = (
        "category",
        "message",
        "provider_error_type",
        "request_headers",
        "request_url",
        "response_body",
        "response_headers",
        "retry_after",
        "status_code",
    )

    category: RetryErrorCategory
    message: str
    status_code: int | None
    provider_error_type: str | None
    retry_after: str | None
    request_url: str | None
    request_headers: dict[str, str]
    response_headers: dict[str, str]
    response_body: JsonValue | None

    def __init__(
        self,
        *,
        category: RetryErrorCategory,
        message: str,
        status_code: int | None = None,
        provider_error_type: str | None = None,
        retry_after: str | None = None,
        request_url: str | None = None,
        request_headers: Mapping[str, str] | None = None,
        response_headers: Mapping[str, str] | None = None,
        response_body: object | None = None,
        redactor: Redactor | None = None,
        secret_header_names: Iterable[str] = (),
    ) -> None:
        self.category = category
        self.message = _redact_text(message, redactor=redactor)
        self.status_code = status_code
        self.provider_error_type = provider_error_type
        self.retry_after = retry_after
        self.request_url = _redact_text(request_url, redactor=redactor) if request_url else None
        self.request_headers = _redact_headers(
            request_headers or {},
            redactor=redactor,
            secret_header_names=secret_header_names,
        )
        self.response_headers = _redact_headers(
            response_headers or {},
            redactor=redactor,
            secret_header_names=secret_header_names,
        )
        self.response_body = _json_value_or_string(
            _redact_payload(response_body, redactor=redactor)
        ) if response_body is not None else None
        super().__init__(self._summary())

    def to_provider_error_info(self) -> ProviderErrorInfo:
        """Return a structured provider error payload with redacted details."""

        return ProviderErrorInfo(
            category=self.category,
            message=self.message,
            status_code=self.status_code,
            provider_error_type=self.provider_error_type,
            retry_after=self.retry_after,
            request_url=self.request_url,
            request_headers=dict(self.request_headers),
            response_headers=dict(self.response_headers),
            response_body=self.response_body,
        )

    def to_error_info(self) -> ErrorInfo:
        """Return a generic serialized error payload for public/event boundaries."""

        info = self.to_provider_error_info()
        details: dict[str, object] = {
            "category": info.category.value,
            "request_headers": info.request_headers,
            "response_headers": info.response_headers,
        }
        if info.status_code is not None:
            details["status_code"] = info.status_code
        if info.provider_error_type is not None:
            details["provider_error_type"] = info.provider_error_type
        if info.retry_after is not None:
            details["retry_after"] = info.retry_after
        if info.request_url is not None:
            details["request_url"] = info.request_url
        if info.response_body is not None:
            details["response_body"] = info.response_body
        return ErrorInfo(
            code=f"provider_{info.category.value}",
            message=info.message,
            details=_JSON_OBJECT_ADAPTER.validate_python(details),
        )

    def _summary(self) -> str:
        parts = [f"provider request failed ({self.category.value})", self.message]
        if self.status_code is not None:
            parts.append(f"HTTP {self.status_code}")
        if self.provider_error_type is not None:
            parts.append(f"type={self.provider_error_type}")
        return ": ".join(parts)

    def __repr__(self) -> str:
        return (
            "ProviderRequestError("
            f"category={self.category.value!r}, "
            f"message={self.message!r}, "
            f"status_code={self.status_code!r}, "
            f"provider_error_type={self.provider_error_type!r}"
            ")"
        )


class ProviderHTTPStatusError(ProviderRequestError):
    """Provider returned a non-success HTTP status for a JSON request."""


def classify_http_status(status_code: int, payload: object | None = None) -> RetryErrorCategory:
    """Classify a provider HTTP status and optional error payload."""

    payload_category = classify_provider_error_payload(payload)
    if payload_category is not None:
        return payload_category
    if status_code == 408:
        return RetryErrorCategory.TIMEOUT
    if status_code == 429:
        return RetryErrorCategory.RATE_LIMIT
    if 500 <= status_code <= 599:
        return RetryErrorCategory.SERVER_ERROR
    if 100 <= status_code <= 599:
        return RetryErrorCategory.NON_RETRYABLE
    return RetryErrorCategory.PROTOCOL_ERROR


def classify_provider_error_payload(payload: object | None) -> RetryErrorCategory | None:
    """Classify known provider error payload shapes by message/type content."""

    text = _payload_text(payload)
    if not text:
        return None
    if _contains_any(text, _CONTEXT_OVERFLOW_MARKERS):
        return RetryErrorCategory.CONTEXT_OVERFLOW
    if _looks_like_continuation_unavailable(text):
        return RetryErrorCategory.CONTINUATION_UNAVAILABLE
    if _contains_any(text, _UNSUPPORTED_PARAMETER_MARKERS):
        return RetryErrorCategory.UNSUPPORTED_PARAMETER
    if _contains_any(text, _RATE_LIMIT_MARKERS):
        return RetryErrorCategory.RATE_LIMIT
    if _contains_any(text, _OVERLOADED_MARKERS):
        return RetryErrorCategory.OVERLOADED
    if _contains_any(text, _TIMEOUT_MARKERS):
        return RetryErrorCategory.TIMEOUT
    return None


def provider_error_message_from_payload(payload: object | None) -> str | None:
    """Extract a concise provider error message from common payload shapes."""

    for value in _candidate_key_strings(payload, "message"):
        return value
    return None


def provider_error_type_from_payload(payload: object | None) -> str | None:
    """Extract a provider error type/code from common payload shapes."""

    error_object = _mapping_value(payload, "error")
    if isinstance(error_object, Mapping):
        error_mapping = cast(Mapping[object, object], error_object)
        for key in ("type", "code", "param"):
            value = error_mapping.get(key)
            if isinstance(value, str) and value:
                return value
    for key in ("code", "type", "param"):
        for value in _candidate_key_strings(payload, key):
            if value != "error":
                return value
    return None


def is_context_overflow_error(error: BaseException) -> bool:
    """Return whether an exception is classified as provider context overflow."""

    category = getattr(error, "category", None)
    if _category_value(category) == RetryErrorCategory.CONTEXT_OVERFLOW.value:
        return True
    return classify_provider_error_payload(str(error)) is RetryErrorCategory.CONTEXT_OVERFLOW


def is_continuation_unavailable_error(error: BaseException) -> bool:
    """Return whether an exception indicates provider-side continuation is unavailable."""

    category = getattr(error, "category", None)
    if _category_value(category) == RetryErrorCategory.CONTINUATION_UNAVAILABLE.value:
        return True
    return (
        classify_provider_error_payload(str(error))
        is RetryErrorCategory.CONTINUATION_UNAVAILABLE
    )


def retry_after_from_headers(headers: Mapping[str, str]) -> str | None:
    """Return a Retry-After header value using case-insensitive lookup."""

    for name, value in headers.items():
        if name.lower() == "retry-after" and value:
            return value
    return None


def _redact_headers(
    headers: Mapping[str, str],
    *,
    redactor: Redactor | None,
    secret_header_names: Iterable[str],
) -> dict[str, str]:
    raw = {str(name): str(value) for name, value in headers.items()}
    secret_redactor = Redactor(
        secret_header_names=(*DEFAULT_SECRET_HEADER_NAMES, *tuple(secret_header_names))
    )
    redacted = secret_redactor.redact_headers(raw)
    if redactor is not None:
        return redactor.redact_headers(redacted)
    return redacted


def _redact_text(text: str, *, redactor: Redactor | None) -> str:
    if redactor is None:
        return text
    return redactor.redact_text(text)


def _redact_payload(payload: object, *, redactor: Redactor | None) -> object:
    if redactor is None:
        return payload
    return redactor.redact_payload(payload)


def _json_value_or_string(value: object) -> JsonValue:
    try:
        return _JSON_VALUE_ADAPTER.validate_python(value)
    except ValidationError:
        return str(value)


def _payload_text(payload: object | None) -> str:
    return " ".join(_flatten_strings(payload)).lower()


def _contains_any(text: str, markers: Iterable[str]) -> bool:
    return any(marker in text for marker in markers)


def _looks_like_continuation_unavailable(text: str) -> bool:
    if _contains_any(text, _CONTINUATION_MARKERS) and _contains_any(
        text, _CONTINUATION_FAILURE_MARKERS
    ):
        return True
    return "zero data retention" in text and any(
        marker in text for marker in ("conversation", "response", "state", "stored")
    )


def _flatten_strings(value: object | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, bool | int | float):
        return (str(value),)
    if isinstance(value, Mapping):
        strings: list[str] = []
        for key, item in cast(Mapping[object, object], value).items():
            strings.append(str(key))
            strings.extend(_flatten_strings(item))
        return tuple(strings)
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        strings = []
        for item in cast(Sequence[object], value):
            strings.extend(_flatten_strings(item))
        return tuple(strings)
    return (str(value),)


def _candidate_key_strings(value: object | None, key: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        strings: list[str] = []
        mapping = cast(Mapping[object, object], value)
        for item_key, item in mapping.items():
            if str(item_key) == key and isinstance(item, str) and item:
                strings.append(item)
            strings.extend(_candidate_key_strings(item, key))
        return tuple(strings)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        strings = []
        for item in cast(Sequence[object], value):
            strings.extend(_candidate_key_strings(item, key))
        return tuple(strings)
    return ()


def _mapping_value(value: object | None, key: str) -> object | None:
    if isinstance(value, Mapping):
        return cast(Mapping[object, object], value).get(key)
    return None


def _category_value(category: object) -> str | None:
    if isinstance(category, RetryErrorCategory):
        return category.value
    if isinstance(category, str):
        return category
    return None


__all__ = (
    "DEFAULT_SECRET_HEADER_NAMES",
    "ProviderErrorCategory",
    "ProviderErrorInfo",
    "ProviderHTTPStatusError",
    "ProviderRequestError",
    "classify_http_status",
    "classify_provider_error_payload",
    "is_context_overflow_error",
    "is_continuation_unavailable_error",
    "provider_error_message_from_payload",
    "provider_error_type_from_payload",
    "retry_after_from_headers",
)
