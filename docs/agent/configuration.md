# Agent Configuration

tend separates durable agent definition from per-invocation runtime settings.

## Durable `agent.yaml`

Validated by `AgentConfig`. JSON files are still accepted for compatibility.

```yaml
schema_version: '1'
system_prompt:
  path: prompts/system.md
model:
  provider: openai
  api: openai_responses
  model_name: gpt-5
  settings:
    reasoning:
      effort: minimal
    max_output_tokens: 256
tools:
  enabled:
    - ls
    - read_file
    - grep
    - glob
```

Important rules:

- `system_prompt`, `model.provider`, and `model.model_name` must be non-empty.
- `system_prompt` may be a literal string, `{path: ...}` (relative to the agent config file), or `{registry: "worker/minimal"}` / `{registry: "reviewer/minimal"}` for bundled prompts. File-backed prompts are read when the config is loaded, so editing the file affects the next `tend-agent` launch.
- `model.api` is either `openai_responses` or `anthropic_messages`.
- `model.profile` may provide a custom `ModelProfile`; otherwise built-in profiles are used when available.
- `tools.enabled` and `tools.options` are validated against the closed built-in registry.
- `agent.yaml` must not contain secret-like keys. API keys, request headers, and environment allowlists belong in runtime config.
- `runtime_defaults` may contain sparse runtime overrides, but not prompts, provider headers/request payloads, API-key sources, or environment allowlists.

## Runtime `cfg.yaml`

Validated as `RuntimeConfigOverrides` and resolved to `RuntimeConfig`. JSON files are still accepted for compatibility.

```yaml
cwd: .
session_dir: .tend/session
limits:
  max_iterations: null        # unbounded by default
  max_model_requests: null    # unbounded by default
  max_tool_calls: null        # unbounded by default
  max_wall_time_seconds: 3600
model:
  timeout_seconds: 60
environment:
  allowed_env_vars:
    - OPENAI_API_KEY
```

Major runtime sections:

- `prompt`, `cwd`, `session_dir`
- `limits`: iterations, model requests, tool calls, wall time, tokens, cost
- `retries`: provider-neutral retry/backoff policy helpers
- `compaction`: generic compaction triggers and budgets
- `logging`, `artifacts`, `redaction`, `usage`
- `model`: base URL, timeout, extra headers, provider-side continuation opt-in, extra request settings
- `api_key_sources`: default environment variable names for provider API keys/base URL
- `environment`: host environment names that may be supplied to provider adapter construction

The standalone `tend-agent` CLI does not sandbox execution. Process isolation belongs to the orchestrator or another external launcher.

## Resolution precedence

`resolve_runtime_config(...)` merges config in this order:

1. library defaults
2. `agent.yaml.runtime_defaults`
3. `cfg.yaml`
4. CLI overrides

Later entries override earlier entries recursively.

## Defaults worth knowing

- Limits: iteration, model-request, and tool-call counts are unbounded by default (`null`); wall time defaults to 3600 seconds. `max_tokens` and `max_cost` are unset by default.
- Compaction: enabled, `reserve_tokens=4096`, `keep_recent_tokens=16000`, `target_tokens=4000`.
- Usage/context estimates: enabled.
- Default secret/env source names: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`.

## CLI overrides

The CLI maps flags such as `--cwd`, `--session-dir`, `--max-model-requests`, `--model-base-url`, and `--no-compaction` into sparse `RuntimeConfigOverrides` before resolution.
