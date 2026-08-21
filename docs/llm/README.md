# LLM API Calling Layer

The LLM layer converts provider-neutral `ModelRequest` values into provider-native HTTP requests and parses provider-native responses back into `ModelResponse` values.

## Main modules

- `tend.llm.models`: provider-neutral message, request/response, tool, reasoning, provider metadata, and model profile schemas.
- `tend.llm.providers.openai_responses`: OpenAI-compatible Responses adapter.
- `tend.llm.providers.anthropic_messages`: native Anthropic Messages adapter.
- `tend.llm.providers.http`: replaceable async JSON POST transport.
- `tend.llm.providers.errors`: provider error classification and redacted diagnostics.

## Contents

- [Provider-neutral model layer](model-layer.md)
- [OpenAI Responses adapter](openai-responses.md)
- [Anthropic Messages adapter](anthropic-messages.md)
- [HTTP transport, errors, redaction, and retry helpers](transport-errors-retries.md)
- [Cloudflare AI Gateway live checks](cloudflare-live.md)
