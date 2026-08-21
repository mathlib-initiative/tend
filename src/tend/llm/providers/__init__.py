"""Provider adapter support primitives."""

from tend.llm.providers.anthropic_messages import AnthropicMessagesAdapter
from tend.llm.providers.errors import (
    DEFAULT_SECRET_HEADER_NAMES,
    ProviderErrorCategory,
    ProviderErrorInfo,
    ProviderHTTPStatusError,
    ProviderRequestError,
    classify_http_status,
    classify_provider_error_payload,
    is_context_overflow_error,
    is_continuation_unavailable_error,
    provider_error_message_from_payload,
    provider_error_type_from_payload,
    retry_after_from_headers,
)
from tend.llm.providers.http import (
    HttpxJsonTransport,
    JsonPostRequest,
    JsonPostResponse,
    JsonPostTransport,
    ScriptedJsonStep,
    ScriptedJsonTransport,
    ScriptedTransportExhaustedError,
)
from tend.llm.providers.openai_responses import OpenAIResponsesAdapter

__all__ = (
    "DEFAULT_SECRET_HEADER_NAMES",
    "AnthropicMessagesAdapter",
    "HttpxJsonTransport",
    "JsonPostRequest",
    "JsonPostResponse",
    "JsonPostTransport",
    "OpenAIResponsesAdapter",
    "ProviderErrorCategory",
    "ProviderErrorInfo",
    "ProviderHTTPStatusError",
    "ProviderRequestError",
    "ScriptedJsonStep",
    "ScriptedJsonTransport",
    "ScriptedTransportExhaustedError",
    "classify_http_status",
    "classify_provider_error_payload",
    "is_context_overflow_error",
    "is_continuation_unavailable_error",
    "provider_error_message_from_payload",
    "provider_error_type_from_payload",
    "retry_after_from_headers",
)
