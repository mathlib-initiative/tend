# Tend

Tend coordinates durable swarms of coding agents over a Git repository. It turns a YAML task graph into isolated worktrees, runs workers and reviewers concurrently, persists progress in SQLite, and validates and merges approved contributions.

The package also includes the typed, provider-neutral agent runtime used by the generated workers: resumable sessions, built-in filesystem tools, context compaction, structured outputs, and OpenAI Responses and Anthropic Messages adapters.

## Quickstart

Install the project and its development dependencies:

```bash
uv sync --locked --all-groups
```

Initialize an orchestration root outside the repository the agents will edit:

```bash
export ANTHROPIC_API_KEY="<anthropic-api-key>"

uv run tend init \
  --root ./tend-root \
  --entrypoint ./project \
  --agent tend \
  --build-command "uv run pytest"

uv run tend validate-config --root ./tend-root
uv run tend run --root ./tend-root --dry-run
```

Commit task files under `./project/tasks/` before starting a live run. Use `uv run tend-task verify ./project/tasks` to validate them. The [orchestrator quickstart](docs/orchestrator/quickstart.md) covers task files, generated agent configuration, and running a swarm.

Tend does not impose a build command by default. Pass `--build-command` during initialization or configure `pre_merge_validation_commands` before a live run.

## Agent runtime

The deterministic scripted model is useful for embedding Tend and for tests:

```python
import asyncio

from tend import Agent
from tend.llm.models import AssistantMessage, ModelResponse, TextContent
from tend.llm.testing import ScriptedModel


async def main() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                assistant_message=AssistantMessage(
                    content=[TextContent(text="Done.")],
                ),
            )
        ]
    )
    result = await Agent("You are concise.", model=model).run_turn("Say done")
    print(result.final_response)


asyncio.run(main())
```

The standalone `tend-agent` process is not sandboxed. Run it through Tend or another process isolation boundary when executing untrusted work.

## Commands

- `tend`: initialize, run, inspect, and clean durable orchestration roots.
- `tend-agent`: execute one provider-neutral agent turn.
- `tend-task`: validate and inspect YAML task graphs.
- `tend-control`: inspect and steer active runs.

## Documentation

- [Agent runtime](docs/agent/README.md)
- [LLM provider layer](docs/llm/README.md)
- [Orchestrator](docs/orchestrator/README.md)

## Development

```bash
uv run pyright
uv run ruff check
uv run pytest -m "not live"
```

Live provider tests are opt-in and are never part of the default test run.

## License

Tend is licensed under the [Apache License 2.0](LICENSE).
