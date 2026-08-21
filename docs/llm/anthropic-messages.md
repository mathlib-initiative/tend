# Anthropic Messages Adapter

`AnthropicMessagesAdapter` translates provider-neutral requests to the native Anthropic Messages API and parses Messages payloads back to `ModelResponse`.

## Construction

```python
from tend.llm.providers import AnthropicMessagesAdapter

adapter = AnthropicMessagesAdapter(
    model_name="claude-sonnet-4-5",
    provider_name="cloudflare_anthropic",
    base_url="https://gateway.example/v1/account/gateway/anthropic/v1",
    profile=profile,
    environment={"ANTHROPIC_API_KEY": "..."},
    api_key_env_var="ANTHROPIC_API_KEY",
)
```

`from_config(model_config, runtime_config, environment=...)` reads only the supplied environment mapping. Base URL resolution order is:

1. `runtime_config.model.base_url`
2. `model_config.endpoint`
3. the environment variable named by `runtime_config.api_key_sources.anthropic_base_url_env` (default `ANTHROPIC_BASE_URL`)
4. `https://api.anthropic.com/v1`

The adapter appends `/messages` unless the base URL already ends in `/messages`.

## Request mapping

`build_payload(ModelRequest(...))` produces:

- `model`
- `max_tokens` from request/default/profile or `1024`
- `messages` with native Anthropic user/assistant turns
- `system` from system messages plus developer messages prefixed as `Developer instructions:`
- `thinking` when `ReasoningSettings` is present
- `tools` with native `input_schema`
- `tool_choice` when requested and compatible
- `temperature` and extra settings only when supported by the profile

System and developer messages are not included in the native `messages` array; they become the native `system` string.

## Thinking

Reasoning settings map to Anthropic thinking:

```json
{"type": "enabled", "budget_tokens": 1024}
```

Budget selection order:

1. `ReasoningSettings.native_settings.thinking.budget_tokens`
2. `ReasoningSettings.max_reasoning_tokens`
3. effort-based defaults (`low`/`minimal` 1024, `medium` 4096, `high` 8192, `xhigh` 16384, `max` 32768)
4. profile minimum when larger

Adaptive-thinking profiles use `thinking: {"type": "adaptive", "display": "summarized"}` plus `output_config.effort` instead of `budget_tokens`. Always-on adaptive models such as Claude Fable 5 may also emit an adaptive `thinking` block without `output_config` when callers only request display settings.

The adapter validates profile thinking constraints, including the common requirement that `max_tokens` must be greater than `budget_tokens`.

When thinking is enabled, forced tool choice is omitted unless the profile marks forced tool choice compatible with thinking.

## Tools and continuation

Provider-neutral tool schemas map to native tools:

```json
{
  "name": "ls",
  "description": "...",
  "input_schema": {...}
}
```

Assistant tool calls become native `tool_use` blocks with preserved `tool_use.id`. Tool results become user `tool_result` blocks. If prior assistant metadata contains Anthropic thinking/redacted thinking continuation blocks, the adapter emits them before text/tool-use blocks for stateless continuation.

## Response parsing

The parser handles:

- text content blocks as final assistant text only for completed responses;
- `tool_use` blocks as ordered `ToolCall` values;
- `thinking` and `redacted_thinking` blocks as reasoning/provider metadata, not final text;
- thinking signatures and redacted data for continuation;
- usage fields including input/output/cache/thinking tokens and nested provider integer details;
- native stop reasons and service-tier metadata.

`stop_reason=max_tokens` maps to `StopReason.MAX_TOKENS` and omits partial text from `final_text`. `tool_use` responses have no final text and normally map to `StopReason.PROVIDER_STOP_REASON` so the turn loop executes tools.

Malformed payloads raise `ProviderProtocolError`.

## Headers and secrets

`build_headers()` includes `Content-Type: application/json` and `anthropic-version: 2023-06-01`. It adds `x-api-key` when an Anthropic API key source is configured, unless an auth extra header such as `cf-aig-authorization` is configured. Secret headers are redacted by `redacted_headers(...)` and provider errors.
