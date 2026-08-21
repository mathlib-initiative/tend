# CLI

The console script is `tend-agent = tend.agent.cli:main`. It runs exactly one agent turn in the current process environment.

`tend-agent` does not create an OS sandbox. Callers that need filesystem, process, network, or environment isolation must provide it outside the agent layer, normally through the orchestration layer.

## Basic usage

```bash
uv run tend-agent --agent agent.yaml --config cfg.yaml --prompt "Inspect the project."
```

Prompt source priority:

1. `--prompt`
2. `RuntimeConfig.prompt` from `cfg.yaml`
3. stdin

Use `--json` to emit a serialized `TurnResult` to stdout. Without `--json`, final responses go to stdout and non-final stop diagnostics go to stderr. YAML (`.yaml`/`.yml`) and legacy JSON config files are accepted.

## Common flags

- `--agent PATH` (required)
- `--config PATH`
- `--prompt TEXT`
- `--json`
- `--cwd PATH`
- `--session-dir PATH`
- `--session-id ID`
- `--resume-session`
- `--max-iterations N`
- `--max-model-requests N`
- `--max-tool-calls N`
- `--max-wall-time-seconds SECONDS`
- `--max-tokens N`
- `--max-cost DECIMAL`
- `--model-base-url URL`
- `--model-timeout-seconds SECONDS`
- `--no-compaction`

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | Final response produced. |
| 1 | Structured non-final stop. |
| 2 | Configuration or CLI usage error. |
| 70 | Internal/framework software error. |
| 130 | Interrupted by SIGINT/SIGTERM or cancellation. |

Errors written by the CLI are JSON `ErrorInfo` values.

## Environment handling

The CLI builds provider adapters from the resolved runtime config. Only names returned by `RuntimeConfig.allowed_environment_names()` are passed from the host environment to provider adapter construction. This is provider-secret plumbing, not a process sandbox.

## Signals

The CLI installs SIGINT/SIGTERM handlers while a turn is active. Handlers request cooperative cancellation and cancel the active task so interruption can be persisted and reported consistently.
