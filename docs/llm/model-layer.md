# Provider-Neutral Model Layer

The turn loop depends only on `ModelAdapter.generate(ModelRequest) -> ModelResponse`. Concrete provider adapters translate at the boundary.

## Adapter protocol

```python
class ModelAdapter(Protocol):
    @property
    def profile(self) -> ModelProfile | None: ...
    async def generate(self, request: ModelRequest) -> ModelResponse: ...
```

Tests use `tend.llm.testing.ScriptedModel` and `ScriptedJsonTransport` to avoid live network calls.

Adapters can optionally implement `RequestTokenEstimator` from
`tend.llm.context_estimation`:

```python
def estimate_request_tokens(
    self, request: ModelRequest, config: TokenEstimatorConfig
) -> RequestTokenEstimate: ...
```

This synchronous method must not make network calls. Its result contains one
nonnegative token estimate per request message, in the same order, plus separate
`tool_schema_tokens` and `reasoning_setting_tokens` fixed costs. It must also
accept partial requests (including a single tool result or summary message),
which the runtime uses to estimate additions and replacement summaries.

Estimates should follow the adapter's serialization logic: include replayed
continuations, but exclude metadata that is only stored locally. When opaque
retained reasoning has a recorded token count, use that count in place of the
serialized payload estimate, without adding duplicate metadata representations.
Both built-in adapters implement this capability. Custom adapters without it
continue to use the generic character estimator. API usage calibrates compaction
budgets for all adapters; local estimates remain approximate.

## Messages and content

Roles:

- `system`
- `developer`
- `user`
- `assistant`
- `tool`

Content parts:

- `TextContent(kind="text", text=...)`
- `CompactionSummaryContent(kind="compaction_summary", summary=..., covered_message_ids=[...])`

`ModelMessage` is a discriminated union over system, developer, user, assistant, and tool-result messages.

Assistant tool calls are not normal text. They are stored in assistant `provider_metadata` by context helpers such as `assistant_message_from_response(...)` and extracted by provider adapters for stateless continuation.

## Requests

`ModelRequest` fields:

- `request_id`
- `model_name`
- `messages`
- `tools`: provider-neutral objects with `name`, `description`, and `arguments_schema`
- `reasoning`
- `max_output_tokens`
- `provider_metadata`
- `request_metadata`

Provider adapters may read provider-specific request settings from `request_metadata`, for example `openai_responses_request_settings` or `anthropic_messages_request_settings`.

## Responses

`ModelResponse` fields:

- `response_id` and optional `request_id`
- optional `assistant_message`
- `tool_calls`
- normalized `stop_reason`
- `provider_completion_status`
- `incomplete_details`
- `usage`
- `reasoning`
- `provider_metadata`
- `response_metadata`

`ModelResponse.final_text` concatenates normal assistant `TextContent` parts. It excludes reasoning/thinking metadata.

## Tool calls/results

`ToolCall` normalizes provider tool arguments to a JSON object and preserves provider IDs:

- OpenAI: item ID and `call_id`
- Anthropic: `tool_use.id`

`ToolResult` preserves linkage and execution metadata. `ToolResultMessage.from_result(...)` creates a model-visible message containing concise text or JSON output.

## Reasoning

`ReasoningSettings` supports:

- effort: `minimal`, `low`, `medium`, `high`, `xhigh`, `max`
- summary preference: `none`, `auto`, `concise`, `detailed`
- display policy
- max reasoning tokens
- provider-native settings

`ReasoningMetadata` records requested settings, observed/native settings, safe summaries, reasoning token counts, and provider-private continuation metadata. Hidden chain-of-thought is not represented as normal assistant text.

## Model profiles

`ModelProfile` describes capabilities and compatibility:

- provider/API/model identity
- optional context window and output limits
- tool-calling capabilities
- reasoning/thinking capabilities
- supported request settings
- continuation capabilities
- optional pricing
- gateway/ZDR compatibility flags

Built-in profiles currently cover Cloudflare-routed OpenAI and Anthropic models used by tests. Unknown/custom models can use explicit profiles or run with less capability validation.
