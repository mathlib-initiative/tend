# Orchestrator Tasks

The orchestrator reads task files from:

```text
<entrypoint>/tasks/*.yaml
```

Each file is parsed as `tend.orchestrator.tasks.Task`, and all loaded tasks are validated together by `TaskManager`.

## Task schema

```yaml
schema_version: 1
id: task-001
title: "Add pagination to the API"
status: open
priority: default
depends_on: []
summary: Add pagination to the API
description: |
  Implement cursor pagination for the list endpoint.
  Update tests and documentation.
```

Fields:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `schema_version` | `1` | `1` | Task file format version. |
| `id` | non-blank string | required | Stable task identifier. Must be unique across all loaded tasks. |
| `title` | non-blank string | required (defaults from `summary` when absent) | Short human-readable title. |
| `summary` | non-blank string | required | Short human-readable summary. |
| `description` | non-blank string | required | Worker instructions / free-form body. Use a `description: |` literal block scalar for real tasks so colons and special characters are safe. |
| `status` | `open` or `complete` | `open` | Lifecycle status used by dependency readiness and run completion. |
| `priority` | `default`, `high`, or `max` | `default` | Scheduling priority for ready tasks. `max` ready tasks are picked before `high`, and `high` before `default`. |
| `depends_on` | list of task ids | `[]` | Tasks that must be `complete` before this task is ready. |

## Dependency rules

Across the whole `tasks/*.yaml` set:

- task IDs must be unique;
- every dependency ID must reference another loaded task;
- dependency cycles are rejected;
- a `complete` task may not depend, directly or transitively, on an `open` task.

An `open` task is ready when all of its dependencies are `complete`.

## Verifying a task folder

Use `tend-task verify` to validate task YAML before committing or before starting an orchestrated run:

```bash
uv run tend-task verify <entrypoint>/tasks
```

The command applies the same strict task-file parsing and dependency-graph checks used by the orchestrator's post-merge task validation gate.

Example:

```yaml
# tasks/001-design.yaml
schema_version: 1
id: design
title: "Decide API shape"
status: complete
priority: default
depends_on: []
summary: Decide API shape
description: |
  Write down the API shape.
```

```yaml
# tasks/002-implement.yaml
schema_version: 1
id: implement
title: "Implement API"
status: open
priority: high
depends_on:
  - design
summary: Implement API
description: |
  Implement the API from the design.
```

`implement` is ready because `design` is `complete`.

## Priority and file ordering

`load_entrypoint_tasks()` loads matching files in sorted path order. This order is preserved by `TaskManager.tasks`. `TaskManager.ready_tasks()` and the runtime ready-task queue then order ready work by priority first (`max`, then `high`, then `default`) and preserve file order among tasks with the same priority. Use filename prefixes such as `001-`, `002-`, etc. when deterministic same-priority queue order matters.

## How tasks drive orchestration

For each polling cycle, the orchestrator:

1. reloads and validates `<entrypoint>/tasks/*.yaml`;
2. finds ready tasks;
3. orders queued ready tasks by priority (`max` before `high` before `default`);
4. skips a ready task if it already has a non-closed worktree;
5. creates a detached worktree for every remaining queued ready task;
6. attaches the task ID to that worktree and queues it for a worker.

The run stops only when the loaded task set is non-empty and every loaded task is `complete`. An empty `tasks/` directory makes the run wait indefinitely.

## Completing a task

A worker completes a task by changing its task file in the worktree:

```yaml
status: complete
```

After reviewer approval, the orchestrator merges the worktree's committed changes. The entrypoint copy of the task file then becomes `complete`. On the next poll, if every task is `complete`, the run exits.

## Decomposing broad tasks

Workers may decompose broad work by editing task YAML in the worktree and adding new task files. A common pattern is:

```yaml
# tasks/001-umbrella.yaml
schema_version: 1
id: umbrella
title: "Complete the reporting feature"
status: open
priority: default
depends_on:
  - report-db
  - report-api
  - report-ui
summary: Complete the reporting feature
description: |
  Coordinate the reporting feature through smaller tasks.
```

```yaml
# tasks/010-report-db.yaml
schema_version: 1
id: report-db
title: "Add report tables"
status: open
priority: default
depends_on: []
summary: Add report tables
description: |
  Create schema and migrations for reports.
```

```yaml
# tasks/020-report-api.yaml
schema_version: 1
id: report-api
title: "Add report API"
status: open
priority: default
depends_on:
  - report-db
summary: Add report API
description: |
  Implement API endpoints after the database layer exists.
```

After the decomposition worktree is approved and merged, the orchestrator sees the new ready leaf tasks and starts workers for them.

## Programmatic helpers

```python
from pathlib import Path

from tend.orchestrator import (
    Task,
    TaskManager,
    TaskPriority,
    TaskStatus,
    load_entrypoint_task_manager,
    write_task,
)

entrypoint = Path("/path/to/repo")

task = Task(
    id="task-001",
    title="Seed task",
    summary="Seed task",
    description="Do one focused change.",
    status=TaskStatus.OPEN,
    priority=TaskPriority.HIGH,
)
write_task(entrypoint / "tasks" / "001-seed.yaml", task)

manager = load_entrypoint_task_manager(entrypoint)
print([task.id for task in manager.ready_tasks()])
```
