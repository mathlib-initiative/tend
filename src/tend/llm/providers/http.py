"""Small async JSON HTTP boundary for provider adapters."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from typing import Protocol, runtime_checkable

import httpx
from pydantic import Field, JsonValue

from tend._common.types import StrictModel
from tend.llm.providers.errors import (
    ProviderHTTPStatusError,
    ProviderRequestError,
    classify_http_status,
    provider_error_message_from_payload,
    provider_error_type_from_payload,
    retry_after_from_headers,
)
from tend.llm.redaction import Redactor
from tend.llm.retries import RetryErrorCategory


def _empty_headers() -> dict[str, str]:
    return {}


class JsonPostRequest(StrictModel):
    """One provider JSON POST request captured at the HTTP boundary."""

    url: str = Field(min_length=1)
    headers: dict[str, str] = Field(default_factory=_empty_headers)
    body: JsonValue
    timeout_seconds: float | None = Field(default=None, ge=0)


class JsonPostResponse(StrictModel):
    """One provider JSON HTTP response."""

    status_code: int = Field(ge=100, le=599)
    headers: dict[str, str] = Field(default_factory=_empty_headers)
    body: JsonValue

    @property
    def is_success(self) -> bool:
        """Return whether the HTTP status is in the 2xx success range."""

        return 200 <= self.status_code <= 299


@runtime_checkable
class JsonPostTransport(Protocol):
    """Minimal replaceable async transport for provider JSON POST requests."""

    async def post_json(self, request: JsonPostRequest) -> JsonPostResponse:
        """POST JSON and return a decoded JSON response or raise a provider error."""
        ...


class HttpxJsonTransport(JsonPostTransport):
    """``httpx`` implementation of the provider JSON transport."""

    __slots__ = ("_client", "_redactor", "_secret_header_names")

    _client: httpx.AsyncClient | None
    _redactor: Redactor | None
    _secret_header_names: tuple[str, ...]

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        redactor: Redactor | None = None,
        secret_header_names: Iterable[str] = (),
    ) -> None:
        self._client = client
        self._redactor = redactor
        self._secret_header_names = tuple(secret_header_names)

    async def post_json(self, request: JsonPostRequest) -> JsonPostResponse:
        """POST JSON through ``httpx`` with timeout and error classification."""

        try:
            if self._client is not None:
                response = await self._client.post(
                    request.url,
                    headers=request.headers,
                    json=request.body,
                    timeout=request.timeout_seconds,
                )
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        request.url,
                        headers=request.headers,
                        json=request.body,
                        timeout=request.timeout_seconds,
                    )
        except httpx.TimeoutException as exc:
            raise self._request_error(
                request,
                category=RetryErrorCategory.TIMEOUT,
                message="provider request timed out",
            ) from exc
        except httpx.RequestError as exc:
            message = (
                "provider request failed before a response was received: "
                f"{type(exc).__name__}"
            )
            raise self._request_error(
                request,
                category=RetryErrorCategory.CONNECTION_ERROR,
                message=message,
            ) from exc

        json_response = self._decode_response_json(request, response)
        _raise_for_status(
            request,
            json_response,
            redactor=self._redactor,
            secret_header_names=self._secret_header_names,
        )
        return json_response

    def _decode_response_json(
        self,
        request: JsonPostRequest,
        response: httpx.Response,
    ) -> JsonPostResponse:
        try:
            body = response.json()
        except ValueError as exc:
            raise self._request_error(
                request,
                category=RetryErrorCategory.PROTOCOL_ERROR,
                message="provider response was not valid JSON",
                status_code=response.status_code,
                response_headers=response.headers,
                response_body=response.text,
            ) from exc
        return JsonPostResponse(
            status_code=response.status_code,
            headers=_headers_to_dict(response.headers),
            body=body,
        )

    def _request_error(
        self,
        request: JsonPostRequest,
        *,
        category: RetryErrorCategory,
        message: str,
        status_code: int | None = None,
        response_headers: Mapping[str, str] | None = None,
        response_body: object | None = None,
    ) -> ProviderRequestError:
        return ProviderRequestError(
            category=category,
            message=message,
            status_code=status_code,
            request_url=request.url,
            request_headers=request.headers,
            response_headers=response_headers,
            response_body=response_body,
            redactor=self._redactor,
            secret_header_names=self._secret_header_names,
        )


class ScriptedTransportExhaustedError(AssertionError):
    """Raised when a scripted transport receives more requests than scripted steps."""


type ScriptedJsonStep = JsonPostResponse | ProviderRequestError | Exception


class ScriptedJsonTransport(JsonPostTransport):
    """Deterministic transport for provider adapter tests.

    Scripted responses follow the same status handling as ``HttpxJsonTransport``:
    any non-2xx response becomes a classified ``ProviderHTTPStatusError`` while
    2xx JSON bodies, including provider-level ``status: incomplete`` payloads,
    are returned unchanged.
    """

    __slots__ = ("_redactor", "_requests", "_secret_header_names", "_steps")

    _requests: list[JsonPostRequest]
    _steps: deque[ScriptedJsonStep]
    _redactor: Redactor | None
    _secret_header_names: tuple[str, ...]

    def __init__(
        self,
        steps: Iterable[ScriptedJsonStep] = (),
        *,
        redactor: Redactor | None = None,
        secret_header_names: Iterable[str] = (),
    ) -> None:
        self._steps = deque(_copy_step(step) for step in steps)
        self._requests = []
        self._redactor = redactor
        self._secret_header_names = tuple(secret_header_names)

    @property
    def requests(self) -> tuple[JsonPostRequest, ...]:
        """Return defensive copies of requests received so far."""

        return tuple(request.model_copy(deep=True) for request in self._requests)

    @property
    def remaining_steps(self) -> int:
        """Return the number of unconsumed scripted steps."""

        return len(self._steps)

    def append_response(self, response: JsonPostResponse) -> None:
        """Append one scripted response step."""

        self._steps.append(response.model_copy(deep=True))

    def append_status(
        self,
        status_code: int,
        body: JsonValue,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """Append one scripted HTTP status/body response step."""

        self.append_response(
            JsonPostResponse(
                status_code=status_code,
                headers=dict(headers or {}),
                body=body,
            )
        )

    def append_exception(self, exception: Exception) -> None:
        """Append one scripted exception step."""

        self._steps.append(exception)

    def clear_requests(self) -> None:
        """Clear recorded requests without modifying remaining scripted steps."""

        self._requests.clear()

    async def post_json(self, request: JsonPostRequest) -> JsonPostResponse:
        """Record ``request`` and consume the next scripted response/error step."""

        self._requests.append(request.model_copy(deep=True))
        if not self._steps:
            raise ScriptedTransportExhaustedError(
                "scripted JSON transport has no remaining response steps"
            )

        step = self._steps.popleft()
        if isinstance(step, ProviderRequestError):
            raise step
        if isinstance(step, Exception):
            raise step

        response = step.model_copy(deep=True)
        _raise_for_status(
            request,
            response,
            redactor=self._redactor,
            secret_header_names=self._secret_header_names,
        )
        return response


def _raise_for_status(
    request: JsonPostRequest,
    response: JsonPostResponse,
    *,
    redactor: Redactor | None,
    secret_header_names: Iterable[str],
) -> None:
    if response.is_success:
        return
    category = classify_http_status(response.status_code, response.body)
    message = provider_error_message_from_payload(response.body) or (
        f"provider returned HTTP {response.status_code}"
    )
    raise ProviderHTTPStatusError(
        category=category,
        message=message,
        status_code=response.status_code,
        provider_error_type=provider_error_type_from_payload(response.body),
        retry_after=retry_after_from_headers(response.headers),
        request_url=request.url,
        request_headers=request.headers,
        response_headers=response.headers,
        response_body=response.body,
        redactor=redactor,
        secret_header_names=secret_header_names,
    )


def _headers_to_dict(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(name): str(value) for name, value in headers.items()}


def _copy_step(step: ScriptedJsonStep) -> ScriptedJsonStep:
    if isinstance(step, JsonPostResponse):
        return step.model_copy(deep=True)
    return step


__all__ = (
    "HttpxJsonTransport",
    "JsonPostRequest",
    "JsonPostResponse",
    "JsonPostTransport",
    "ScriptedJsonStep",
    "ScriptedJsonTransport",
    "ScriptedTransportExhaustedError",
)
