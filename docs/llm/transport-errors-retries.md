# HTTP Transport, Provider Errors, Redaction, and Retry Helpers

## JSON POST transport

Provider adapters depend on `JsonPostTransport`:

```python
async def post_json(request: JsonPostRequest) -> JsonPostResponse: ...
```

Implemented transports:

- `HttpxJsonTransport`: uses `httpx.AsyncClient`, decodes JSON, classifies HTTP/status/protocol failures.
- `ScriptedJsonTransport`: deterministic test transport that records requests and consumes scripted responses/errors.

`JsonPostRequest` contains URL, headers, JSON body, and timeout seconds. `JsonPostResponse` contains status code, headers, body, and `is_success`.

## Provider errors

Non-2xx HTTP responses and request/protocol failures raise `ProviderRequestError` subclasses with redacted diagnostics.

`ProviderHTTPStatusError` is used for non-2xx statuses. Error categories are shared with retry helpers:

- `rate_limit`
- `connection_error`
- `timeout`
- `server_error`
- `overloaded`
- `protocol_error`
- `context_overflow`
- `unsupported_parameter`
- `continuation_unavailable`
- `non_retryable`

The classifier inspects status code plus provider payload text/type/code. Helpers include `is_context_overflow_error(...)` and `is_continuation_unavailable_error(...)`.

## Redaction

Known secret header names include authorization/API-key/cookie headers and Cloudflare gateway auth. Provider errors redact configured secret headers, secret values, secret source assignments, and mildly sensitive gateway URLs.

Adapters expose:

- `secret_header_names`
- `redacted_headers(headers)`

Secret values are resolved from explicit environment mappings or runtime-only `SecretValue` wrappers. They are not read implicitly from global environment by adapter `from_config` methods.

## Provider header config

Runtime config can define extra provider headers:

```json
{
  "name": "cf-aig-authorization",
  "source": "env",
  "env_var": "CF_AIG_AUTHORIZATION",
  "secret": true
}
```

Literal secret headers are rejected unless `RuntimeConfig.model.allow_literal_secret_headers` is true.

## Retry helpers

`tend.llm.retries` implements provider-neutral retry decisions:

- retryable category detection,
- `Retry-After` parsing,
- capped exponential backoff,
- optional jitter,
- max retry-after guard,
- async sleep helpers.

`RetryConfig` defaults to enabled, max 5 attempts, 1s initial delay, 60s max delay, multiplier 2, jitter enabled, and `Retry-After` respected up to 300s.

Current integration note: the shared turn loop does not yet apply general retry/backoff to all provider errors. It only performs the special context-overflow compaction retry. Provider classification and retry helpers are available for future loop integration and tests.
