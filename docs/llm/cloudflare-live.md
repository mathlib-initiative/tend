# Cloudflare AI Gateway Live Checks

Normal tests are deterministic and do not call providers. Live compatibility tests are opt-in and marked `live`.

## Required environment

- `CF_AIG_URL`: Cloudflare AI Gateway base URL, without provider suffix.
- `CF_AIG_TOKEN`: gateway token.

The tests derive:

- OpenAI base URL: `${CF_AIG_URL}/openai`
- Anthropic provider-native base URL: `${CF_AIG_URL}/anthropic/v1`
- Header: `cf-aig-authorization: Bearer ${CF_AIG_TOKEN}`

Do not print these values in logs.

Claude Fable 5 is exposed in Cloudflare's model catalog through the Cloudflare REST AI Gateway as `anthropic/claude-fable-5`. Use the Anthropic-compatible REST base URL `https://api.cloudflare.com/client/v4/accounts/<account>/ai/v1` with `Authorization: Bearer <token>` and `cf-aig-gateway-id: <gateway>` when using that catalog route; the legacy provider-native `/anthropic/v1` route may require a stored/BYOK Anthropic key for the new model.

## Run live tests

```bash
uv run pytest --run-live -m live
```

Without `--run-live`, or without required environment variables, live tests are skipped.

## Covered live behavior

OpenAI Responses through Cloudflare:

- plain final-response turn using `gpt-5`;
- one forced function call and stateless follow-up continuation;
- minimal reasoning request;
- usage capture.

Anthropic Messages through Cloudflare:

- plain final-response turn using `claude-sonnet-4-5`;
- one forced native tool use and tool-result continuation;
- usage capture.

Live tests use tiny prompts, bounded runtime config, no compaction, and a small deterministic `echo` tool.
