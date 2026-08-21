# Orchestrator Public API

Most public orchestrator symbols are re-exported from `tend.orchestrator`.

## Main run API

```python
import asyncio
from pathlib import Path

from tend.orchestrator import (
    AsyncOrchestrator,
    AsyncOrchestratorAgentCommandConfig,
    AsyncOrchestratorConfig,
    AsyncOrchestratorValidationCommandConfig,
)


async def main() -> None:
    config = AsyncOrchestratorConfig(
        root=Path("/tmp/async-root"),
        entrypoint=Path("/path/to/repo"),
        worker_agent_command=AsyncOrchestratorAgentCommandConfig(
            argv=("/tmp/async-root/bin/worker-agent.sh",),
            resume_argv=("--resume",),
        ),
        reviewer_agent_command=AsyncOrchestratorAgentCommandConfig(
            argv=("/tmp/async-root/bin/reviewer-agent.sh",),
            resume_argv=("--resume",),
        ),
        validation_commands=(
            AsyncOrchestratorValidationCommandConfig(argv=("uv", "run", "ruff", "check")),
        ),
        pre_merge_validation_commands=(
            AsyncOrchestratorValidationCommandConfig(argv=("uv", "run", "pytest", "-m", "not live")),
        ),
    )
    result = await AsyncOrchestrator(config).run()
    print(result.root, result.entrypoint, result.usage)


asyncio.run(main())
```

`AsyncOrchestrator.run()` runs until cancelled or until all loaded tasks are complete.

`AsyncOrchestratorRunResult` contains:

| Field | Meaning |
| --- | --- |
| `root` | Runtime root path. |
| `entrypoint` | Entrypoint repository path. |
| `usage` | Aggregated `tend.llm.usage.Usage` from managed tend sessions. |

## Configuration models

| Symbol | Module | Meaning |
| --- | --- | --- |
| `AsyncOrchestratorConfig` | `config` | Runtime config passed to `AsyncOrchestrator`. |
| `AsyncOrchestratorProjectConfig` | `config` | Config loaded from `<root>/config.yaml`. |
| `AsyncOrchestratorAgentCommandConfig` | `config` | Shell-free agent command argv plus resume argv. |
| `AsyncOrchestratorValidationCommandConfig` | `config` | Shell-free validation command used for worker or pre-merge validation gates. |
| `AsyncOrchestratorWorktreeSetupCommandConfig` | `config` | Shell-free setup command with `{entrypoint}` / `{worktree}` placeholders. |

Convenience constructor:

```python
from tend.orchestrator import AsyncOrchestratorConfig

config = AsyncOrchestratorConfig.from_paths(
    root="./async-root",
    entrypoint="./repo",
    worker_agent_command=["./async-root/bin/worker-agent.sh"],
    reviewer_agent_command=["./async-root/bin/reviewer-agent.sh"],
    validation_commands=[["uv", "run", "ruff", "check"]],
    pre_merge_validation_commands=[["uv", "run", "pytest", "-m", "not live"]],
    max_concurrent_worker_agents=2,
)
```

## Task API

| Symbol | Meaning |
| --- | --- |
| `Task` | Pydantic model for one YAML task. |
| `TaskStatus` | `open` or `complete`. |
| `TaskPriority` | `default`, `high`, or `max` ready-task scheduling priority. |
| `TaskManager` | Validated dependency DAG and readiness helper. |
| `TaskValidationFailure` | Structured task-directory validation failure details. |
| `task_directory(entrypoint)` | Returns `<entrypoint>/tasks`. |
| `load_task(path)` | Load one YAML task. |
| `load_tasks(directory, glob="*.yaml")` | Load sorted task files. |
| `load_entrypoint_tasks(entrypoint)` | Load `<entrypoint>/tasks/*.yaml`. |
| `load_entrypoint_task_manager(entrypoint)` | Load and validate all entrypoint tasks. |
| `validate_task_directory(directory)` | Strictly validate task files and the dependency graph, returning a task validation failure or `None`. |
| `write_task(path, task)` | Write one task YAML file. |
| `dump_task_yaml(task)` | Serialize task YAML. |

Example:

```python
from pathlib import Path

from tend.orchestrator import Task, TaskManager, TaskPriority, TaskStatus, write_task

entrypoint = Path("/path/to/repo")
seed = Task(
    id="task-001",
    summary="Seed",
    description="Do one focused change.",
    status=TaskStatus.OPEN,
    priority=TaskPriority.HIGH,
)
write_task(entrypoint / "tasks" / "001-seed.yaml", seed)
manager = TaskManager(tasks=[seed])
assert manager.ready_tasks() == (seed,)
```

## State and runtime types

| Symbol | Meaning |
| --- | --- |
| `SQLiteAsyncOrchestratorStore` | Unified `<root>/orchestrator.sqlite` control/state store. |
| `AsyncOrchestratorWorktree` | Worktree record: id, path, head, task id, state, discussion, session flags, and usage snapshots. |
| `WorktreeState` | `pending`, `worker_running`, `review`, `merge`, `closed`. |
| `AsyncOrchestratorDiscussionMessage` | One role/message entry in the discussion. |
| `AsyncOrchestratorAgentRole` | `worker`, `reviewer`, or `orchestrator`. |
| `AsyncOrchestratorRuntime` | Queues, locks, and active agent tasks. Usually internal. |

`AsyncOrchestrator` persists durable state through `SQLiteAsyncOrchestratorStore`. Resume uses `orchestrator.sqlite`: `worker_running` worktrees are moved back to `pending`, non-closed worktree paths are health-checked, and unhealthy worktrees are skipped for that run without mutating their durable rows.

`AsyncOrchestrator` exposes read-only inspection helpers:

```python
orchestrator.worktree_ids
orchestrator.worktrees_by_id
orchestrator.task_queue
orchestrator.worker_queue
orchestrator.review_queue
orchestrator.merge_queue
orchestrator.usage
```

## Agent output models

| Symbol | Meaning |
| --- | --- |
| `WorkerContributionOutput` | The shared `worker_contribution` contract: `schema_version`, `status` (`completed`, `blocked`, or `needs_review`), a non-blank `summary`, and optional `files_changed`, `validation`, `tasks_created`, and `notes`. |
| `ReviewVerdictOutput` | The shared `review_verdict` contract: `schema_version`, `verdict` (`approve` or `request_changes`), `notes`, `feedback_text` (required iff `request_changes`), and optional `comments`. |

Generated worker agents are seeded from the bundled worker prompt version selected at init (`worker/<version>`, default `minimal`) and emit the shared `worker_contribution` output schema. `completed` and `needs_review` route the worktree to review; `blocked` returns it to the worker (pending).

Generated reviewer agents are seeded from the bundled reviewer prompt version selected at init (`reviewer/<version>`, default `minimal`) and emit the shared `review_verdict` output schema. `approve` routes the worktree to the merge queue; `request_changes` returns it to the worker (pending). There is no `deny`.

Example:

```python
from tend.orchestrator import ReviewVerdictOutput

output = ReviewVerdictOutput.model_validate_json(
    '{"schema_version":1,"verdict":"request_changes",'
    '"notes":"Criterion 2 FAIL.","feedback_text":"Please add tests."}'
)
assert output.verdict == "request_changes"
```

## Usage helpers

| Symbol | Meaning |
| --- | --- |
| `agent_session_dir(root, worktree_id, role)` | Returns `<root>/sessions/<worktree_id>/<role>`. |
| `load_agent_session_usage(root, worktree_id, role)` | Reads usage from a managed tend session, if present. |
| `aggregate_agent_session_usage(root, worktrees)` | Aggregates usage across worker/reviewer sessions. |
| `format_usage_summary(usage)` | Formats a compact log message. |

Only tend session event files are understood. Empty or foreign session directories are ignored.

## Module map

```text
tend.orchestrator.__init__       public re-exports
tend.orchestrator.cli            tend CLI
tend.orchestrator.config         config models and argv validation
tend.orchestrator.tasks          Task, TaskPriority, and TaskStatus
tend.orchestrator.task_manager   dependency DAG validation/readiness
tend.orchestrator.task_io        YAML task I/O helpers
tend.orchestrator.state          worktree state value models
tend.orchestrator.control_store  unified <root>/orchestrator.sqlite control/state store
tend.orchestrator.runtime        queues, locks, active asyncio tasks
tend.orchestrator.queues         de-duplicating FIFO queue helper
tend.orchestrator.agent_runner   subprocess environment/launch helper
tend.orchestrator.discussion     .tend/discussion.md rendering
tend.orchestrator.usage          managed tend session usage aggregation
tend.orchestrator.orchestrator   parent workflow implementation
```
