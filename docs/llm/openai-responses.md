# OpenAI Responses Adapter

`OpenAIResponsesAdapter` translates provider-neutral requests to an OpenAI-compatible Responses API request and parses Responses payloads back to `ModelResponse`.

## Construction

```python
from tend.llm.providers import OpenAIResponsesAdapter

adapter = OpenAIResponsesAdapter(
    model_name="gpt-5",
    provider_name="cloudflare_openai",
    base_url="https://gateway.example/v1/account/gateway/openai",
    profile=profile,
    environment={"OPENAI_API_KEY": "..."},
    api_key_env_var="OPENAI_API_KEY",
)
```

`from_config(model_config, runtime_config, environment=...)` reads only the supplied environment mapping. It does not read `os.environ` directly.

Base URL resolution order in `from_config`:

1. `runtime_config.model.base_url`
2. `model_config.endpoint`
3. `environment[runtime_config.api_key_sources.openai_base_url_env]`
4. `https://api.openai.com/v1`

The adapter appends `/responses` unless the base URL already ends in `/responses`.

## Request mapping

`build_payload(ModelRequest(...))` produces:

- `model`
- `input`: system/developer/user/assistant messages plus function-call and function-call-output items for stateless continuation
- `max_output_tokens` when supplied/resolved
- `reasoning` when supplied or defaulted
- `tools` as `type: function`
- `parallel_tool_calls: false` when the profile can request serial tool calls
- `temperature` and extra settings only when supported by the profile
- `previous_response_id` only when provider-side continuation is explicitly enabled and the profile marks it safe

Cloudflare/ZDR built-in OpenAI profiles require stateless replay, so `previous_response_id` is not sent even if runtime config enables provider-side continuation.

## Reasoning

If request/default reasoning is absent, the adapter defaults to `{"effort": "minimal"}` when the profile supports minimal reasoning or when no profile is present. Explicit `ReasoningSettings.summary` values other than `none` map to the Responses `reasoning.summary` field.

## Tools

Provider-neutral tool schemas map to Responses function tools:

```json
{
  "type": "function",
  "name": "ls",
  "description": "...",
  "parameters": {...},
  "strict": true
}
```

`strict` is included when the profile supports strict tool schemas, or by default when no profile is present.

Function call outputs are sent as `type: "function_call_output"` with the preserved provider `call_id` when available.

## Response parsing

The parser handles:

- final assistant text from message/output text items;
- function calls with JSON-string arguments;
- response IDs, item IDs, call IDs, statuses, and native status metadata;
- usage fields including input/output/reasoning/cache tokens and provider-specific integer details;
- reasoning summaries and encrypted reasoning continuation metadata;
- incomplete and failed provider statuses.

Empty assistant text parts from the provider are ignored rather than represented as normalized content. Missing or non-string assistant text fields are still malformed and raise `ProviderProtocolError`. A completed response with no non-empty assistant text produces no `final_text` and remains a provider stop instead of a final response.

Incomplete Responses with `incomplete_details.reason` of `max_output_tokens` or `max_tokens` map to `StopReason.MAX_TOKENS` and omit partial text from `final_text`.

Malformed payloads or invalid function-call arguments raise `ProviderProtocolError`.

## Headers and secrets

`build_headers()` includes `Content-Type: application/json`, an API-key header when configured, and extra headers. Environment-sourced and known secret headers are redacted by `redacted_headers(...)` and provider errors.

For Cloudflare AI Gateway, pass `cf-aig-authorization` as an extra env-sourced header or raw header in tests.
