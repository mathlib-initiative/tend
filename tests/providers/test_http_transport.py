import httpx
import pytest

from tend.llm.providers import (
    HttpxJsonTransport,
    JsonPostRequest,
    JsonPostResponse,
    ProviderErrorCategory,
    ProviderHTTPStatusError,
    ProviderRequestError,
    ScriptedJsonTransport,
    ScriptedTransportExhaustedError,
    classify_http_status,
    classify_provider_error_payload,
    is_context_overflow_error,
    is_continuation_unavailable_error,
)
from tend.llm.redaction import Redactor
from tend.llm.retries import RetryErrorCategory, is_retryable_category
from tend.llm.secrets import REDACTED_VALUE


def _request() -> JsonPostRequest:
    return JsonPostRequest(
        url="https://provider.example/v1/responses",
        headers={"Authorization": "Bearer fake-secret", "x-public": "ok"},
        body={"input": "hello"},
        timeout_seconds=3.0,
    )


async def test_scripted_transport_captures_requests_and_returns_defensive_responses() -> None:
    scripted_response = JsonPostResponse(
        status_code=200,
        headers={"x-request-id": "req_1"},
        body={"ok": True},
    )
    transport = ScriptedJsonTransport([scripted_response])
    request = _request()

    response = await transport.post_json(request)
    response.headers["changed"] = "yes"
    request.headers["after"] = "mutation"

    assert response.body == {"ok": True}
    assert transport.requests == (
        JsonPostRequest(
            url="https://provider.example/v1/responses",
            headers={"Authorization": "Bearer fake-secret", "x-public": "ok"},
            body={"input": "hello"},
            timeout_seconds=3.0,
        ),
    )
    assert scripted_response.headers == {"x-request-id": "req_1"}
    assert transport.remaining_steps == 0

    with pytest.raises(ScriptedTransportExhaustedError, match="no remaining"):
        await transport.post_json(_request())


async def test_scripted_transport_classifies_non_success_status_and_redacts_headers() -> None:
    transport = ScriptedJsonTransport(
        [
            JsonPostResponse(
                status_code=429,
                headers={"Retry-After": "4"},
                body={"error": {"message": "too many requests", "type": "rate_limit_error"}},
            )
        ],
        secret_header_names=["Authorization"],
    )

    with pytest.raises(ProviderHTTPStatusError) as exc_info:
        await transport.post_json(_request())

    error = exc_info.value
    assert error.category is RetryErrorCategory.RATE_LIMIT
    assert error.retry_after == "4"
    assert error.request_headers == {"Authorization": REDACTED_VALUE, "x-public": "ok"}
    assert "fake-secret" not in str(error)
    assert "fake-secret" not in repr(error)
    assert error.to_error_info().details["request_headers"] == {
        "Authorization": REDACTED_VALUE,
        "x-public": "ok",
    }


async def test_scripted_transport_returns_http_200_provider_incomplete_payloads() -> None:
    transport = ScriptedJsonTransport(
        [
            JsonPostResponse(
                status_code=200,
                body={
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
            )
        ]
    )

    response = await transport.post_json(_request())

    assert response.is_success is True
    assert response.body == {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    }


def test_http_status_and_payload_classification() -> None:
    assert ProviderErrorCategory.RATE_LIMIT is RetryErrorCategory.RATE_LIMIT
    assert classify_http_status(429) is RetryErrorCategory.RATE_LIMIT
    assert classify_http_status(408) is RetryErrorCategory.TIMEOUT
    assert classify_http_status(500) is RetryErrorCategory.SERVER_ERROR
    assert classify_http_status(
        529,
        {"error": {"type": "overloaded_error", "message": "server overloaded"}},
    ) is RetryErrorCategory.OVERLOADED
    assert classify_http_status(
        400,
        {"error": {"code": "context_length_exceeded", "message": "maximum context length"}},
    ) is RetryErrorCategory.CONTEXT_OVERFLOW
    assert classify_http_status(
        400,
        {"error": {"message": "unsupported parameter: temperature"}},
    ) is RetryErrorCategory.UNSUPPORTED_PARAMETER
    assert classify_http_status(
        400,
        {
            "error": {
                "message": "previous_response_id is rejected under Zero Data Retention"
            }
        },
    ) is RetryErrorCategory.CONTINUATION_UNAVAILABLE
    assert classify_http_status(404, {"error": {"message": "not found"}}) is (
        RetryErrorCategory.NON_RETRYABLE
    )


def test_context_overflow_and_continuation_error_hooks() -> None:
    overflow = ProviderRequestError(
        category=RetryErrorCategory.CONTEXT_OVERFLOW,
        message="maximum context length exceeded",
    )
    continuation = ProviderRequestError(
        category=RetryErrorCategory.CONTINUATION_UNAVAILABLE,
        message="previous_response_id unavailable under zero data retention",
    )

    assert is_context_overflow_error(overflow) is True
    assert is_context_overflow_error(RuntimeError("maximum context length exceeded")) is True
    assert is_continuation_unavailable_error(continuation) is True
    assert is_continuation_unavailable_error(RuntimeError("ordinary failure")) is False


def test_retryable_vs_nonretryable_provider_categories() -> None:
    overloaded = classify_provider_error_payload({"error": {"message": "server overloaded"}})
    context_overflow = classify_provider_error_payload(
        {"error": {"message": "prompt exceeds the context window"}}
    )
    unsupported = classify_provider_error_payload({"error": {"message": "unsupported parameter"}})

    assert overloaded is not None
    assert context_overflow is not None
    assert unsupported is not None
    assert is_retryable_category(
        classify_http_status(503, {"error": {"message": "temporary server error"}})
    ) is True
    assert is_retryable_category(overloaded) is True
    assert is_retryable_category(context_overflow) is False
    assert is_retryable_category(unsupported) is False


def test_provider_request_error_redacts_configured_secrets_and_headers() -> None:
    redactor = Redactor(secret_values=["fake-secret"], secret_header_names=["x-secret"])

    error = ProviderRequestError(
        category=RetryErrorCategory.NON_RETRYABLE,
        message="provider rejected fake-secret",
        status_code=400,
        request_url="https://gateway.ai.cloudflare.com/v1/account/gateway/openai/responses",
        request_headers={"x-secret": "fake-secret", "x-public": "fake-secret"},
        response_headers={"set-cookie": "fake-secret"},
        response_body={"error": {"message": "fake-secret"}},
        redactor=redactor,
        secret_header_names=["x-secret"],
    )

    assert "fake-secret" not in str(error)
    assert "fake-secret" not in repr(error)
    dumped = error.to_error_info().model_dump(mode="json")
    assert "fake-secret" not in repr(dumped)
    assert dumped["details"]["request_headers"] == {
        "x-secret": REDACTED_VALUE,
        "x-public": REDACTED_VALUE,
    }
    assert dumped["details"]["response_headers"] == {"set-cookie": REDACTED_VALUE}
    assert dumped["details"]["response_body"] == {
        "error": {"message": REDACTED_VALUE}
    }


async def test_httpx_transport_success_with_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer fake-secret"
        return httpx.Response(200, json={"ok": True}, headers={"x-provider-id": "resp_1"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = HttpxJsonTransport(client=client, secret_header_names=["Authorization"])
        response = await transport.post_json(_request())

    assert response.status_code == 200
    assert response.headers["x-provider-id"] == "resp_1"
    assert response.body == {"ok": True}


async def test_httpx_transport_classifies_status_and_invalid_json_protocol_errors() -> None:
    responses = iter(
        [
            httpx.Response(
                500,
                json={"error": {"type": "overloaded_error", "message": "server overloaded"}},
            ),
            httpx.Response(200, content=b"not-json"),
        ]
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return next(responses)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = HttpxJsonTransport(client=client, secret_header_names=["Authorization"])
        with pytest.raises(ProviderHTTPStatusError) as status_exc:
            await transport.post_json(_request())
        with pytest.raises(ProviderRequestError) as protocol_exc:
            await transport.post_json(_request())

    assert status_exc.value.category is RetryErrorCategory.OVERLOADED
    assert status_exc.value.provider_error_type == "overloaded_error"
    assert "fake-secret" not in status_exc.value.to_error_info().model_dump_json()
    assert protocol_exc.value.category is RetryErrorCategory.PROTOCOL_ERROR
    assert protocol_exc.value.status_code == 200


async def test_httpx_transport_classifies_timeout_and_connection_errors() -> None:
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    def connection_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connect failed", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler)) as client:
        transport = HttpxJsonTransport(client=client)
        with pytest.raises(ProviderRequestError) as timeout_exc:
            await transport.post_json(_request())

    async with httpx.AsyncClient(transport=httpx.MockTransport(connection_handler)) as client:
        transport = HttpxJsonTransport(client=client)
        with pytest.raises(ProviderRequestError) as connection_exc:
            await transport.post_json(_request())

    assert timeout_exc.value.category is RetryErrorCategory.TIMEOUT
    assert connection_exc.value.category is RetryErrorCategory.CONNECTION_ERROR
    assert "fake-secret" not in str(timeout_exc.value)
    assert "fake-secret" not in str(connection_exc.value)
