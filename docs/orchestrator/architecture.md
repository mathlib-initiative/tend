# Orchestrator Architecture

The orchestrator is a small local coordination loop around YAML tasks, git worktrees, and external worker/reviewer commands. It is implemented in `src/tend/orchestrator/` and exposed by the `tend` CLI.

It intentionally keeps durable state small and leaves agent behavior to external commands.

## Main pieces

```text
tend CLI
  -> load config.yaml and optional orchestrator.sqlite
  -> AsyncOrchestrator
      -> task discovery service
      -> task queue service
      -> worker agent service
      -> reviewer agent service
      -> merge queue service
```

The CLI in `cli.py` owns command-line parsing, project initialization, config loading, read-only status reporting, state resume, logging, and signal handling. The orchestration loop itself lives in `orchestrator.py`.

## Configuration

`tend init` creates an orchestration root with a marker file and `config.yaml`. The config records:

- the entrypoint repository containing `tasks/*.yaml`;
- optional worker and reviewer argv commands;
- an optional worktree setup argv command;
- optional validation argv commands;
- the merge target branch, defaulting to `main`;
- worker and reviewer concurrency limits.

`tend run` loads this config, applies CLI overrides, initializes `<root>/orchestrator.sqlite`, and starts `AsyncOrchestrator.run()`. `tend status` and `tend export-state --json` only read the SQLite store.

## Runtime services

`AsyncOrchestrator.run()` starts five services in one `asyncio.TaskGroup`:

1. task discovery polls `<entrypoint>/tasks/*.yaml`;
2. task queue processing creates worktrees for ready tasks;
3. worker spawning runs worker commands for pending worktrees;
4. reviewer spawning runs reviewer commands for review worktrees;
5. merge processing merges approved worktrees back into the entrypoint.

Only task discovery polls. The other services wait on runtime queues.

Runtime coordination is held in `AsyncOrchestratorRuntime`:

- `task_queue`, `worker_queue`, `review_queue`, and `merge_queue` are de-duplicating FIFO queues;
- SQLite transactions protect durable state updates;
- `worktree_creation_lock` serializes worktree allocation/creation;
- `merge_lock` serializes entrypoint git mutations and merge-result transitions;
- active worker and reviewer `asyncio.Task` objects are runtime-only.

## Tasks and readiness

Tasks live in `<entrypoint>/tasks/*.yaml` and are loaded into a `TaskManager`. The task manager validates that task IDs are unique, dependencies exist, dependencies form a DAG, and complete tasks do not depend on open tasks.

A task is ready when it is `open` and all dependencies are `complete`. Discovery enqueues ready task IDs only when that task has no non-closed worktree. Ready task IDs are picked from the task queue by priority: `max`, then `high`, then `default`, preserving file order within each priority.

## Durable state and resume

Durable async state lives in the unified SQLite database at `<root>/orchestrator.sqlite` (with normal WAL sidecars `orchestrator.sqlite-wal` and `orchestrator.sqlite-shm` while active). It contains the control-plane heartbeat/command tables plus:

- the last loaded task-manager snapshot;
- the next worktree sequence number;
- worktree rows keyed by worktree ID;
- worktree task associations and lifecycle states;
- discussion messages and review verdicts;
- worker/reviewer session-started flags and captured per-role usage snapshots.

State transitions are short SQLite transactions, typically guarded compare-and-swap updates. Aggregate usage is derived from the stored worktree session snapshots plus any currently active managed session logs; there is no separate `usage.json` snapshot.

Queues and subprocess objects are not persisted. On process resume, the CLI reuses `orchestrator.sqlite`, converts saved `worker_running` worktrees back to `pending`, checks non-closed worktree paths for basic git-worktree health, and constructs a fresh runtime that re-enqueues healthy worktrees according to their saved states. Missing or invalid non-closed worktrees are logged as warnings, left unchanged in persisted state, and not queued for that run.

## Worktree lifecycle

Worktrees are detached git worktrees under `<root>/worktrees/`.

```text
pending -> worker_running -> review -> merge -> closed
                  ^            |
                  |            v
                  +------ request_changes
```

Validation failure after worker success, merge failure, and pre-merge validation failure all return the same worktree to `pending` with an orchestrator discussion message. The reviewer verdict is `approve` (queue for merge) or `request_changes` (return to the worker as `pending`); there is no `deny`.

When a ready task is claimed, the orchestrator records the entrypoint `HEAD`, provisions the git worktree (`git worktree add --detach`, `.tend/` git exclude, and optional setup command), then records the worktree row in `orchestrator.sqlite` after successful provisioning and queues it for a worker.

## Agents and discussions

Workers and reviewers are shell-free subprocess commands launched by `agent_runner.py` with `cwd` set to the worktree. They inherit the host environment plus `TEND_*` variables describing the root, entrypoint, worktree, role, resume state, session directory, and discussion path.

Before each agent run, the orchestrator writes the current discussion to:

```text
<worktree>/.tend/discussion.md
```

Workers emit the shared `worker_contribution` payload (`schema_version`, `status` in `completed`/`blocked`/`needs_review`, a non-blank `summary`, and optional `files_changed`, `validation`, `tasks_created`, and `notes`). Reviewers emit the shared `review_verdict` payload (`schema_version`, `verdict` in `approve`/`request_changes`, `notes`, and `feedback_text` for `request_changes`). Agent stdout is parsed as this protocol. Raw stdout and stderr are also written as per-attempt diagnostics under `<root>/logs/agents/<worktree_id>/`; stderr is not parsed as protocol output.

A worker `status` of `completed` or `needs_review` sends the worktree on to review; `blocked` returns it to the worker queue (`pending`) for another pass. After a worker succeeds and is bound for review, configured validation commands run sequentially in the same worktree before review. After an approved worktree merge is assembled, configured pre-merge validation commands run before close. With the default staging validation worktree, this happens in `<root>/staging` and the entrypoint only fast-forwards after success; with staging disabled, it happens in the entrypoint with rollback on failure. On the first non-zero exit or start failure, the orchestrator appends command/output feedback to the worktree discussion and returns the worktree to `pending`. Validation results are not stored anywhere else.

Agent subprocesses start in their own process group/session. On normal cancellation, the orchestrator sends `SIGTERM` to the process group, waits briefly, then sends `SIGKILL` if needed.

## Merge path

Approved worktrees enter the merge queue. A runtime-only merge lock wraps the full approved-worktree merge path, including git mutations, staging validation, discussion feedback, and final state transitions. With the default `merge_validation_worktree: true`, the merge step:

1. checks `git status --porcelain` in the entrypoint repository;
2. if the entrypoint is dirty, skips checkout/merge, appends an orchestrator discussion message, and returns the worktree to `pending`;
3. requires the worker worktree to be clean, so only already-committed worker changes can land;
4. assembles the committed candidate(s) on `<root>/staging` starting from the configured merge target branch;
5. runs configured `pre_merge_validation_commands` in `<root>/staging`;
6. fast-forwards the entrypoint to the validated staging head;
7. marks the worktree `closed` on success.

If the worker worktree is dirty before review, it is returned to `pending` before merge so the worker can commit or revert the changes. If staging merge assembly fails, the orchestrator appends trimmed git output to the worktree discussion and returns the worktree to `pending`. If pre-merge validation fails in staging, the entrypoint is unchanged; the orchestrator appends validation output to the discussion and returns the worktree to `pending`. When `merge_validation_worktree: false`, the direct-entrypoint fallback performs the merge and validation in the entrypoint and resets it to the pre-merge `HEAD` on validation failure.

## Filesystem layout

The async root owns operational files:

```text
<root>/
├── .tend-root
├── config.yaml
├── orchestrator.sqlite
├── orchestrator.sqlite-wal  # present while SQLite WAL has uncheckpointed pages
├── orchestrator.sqlite-shm  # present while SQLite WAL is active
├── logs.txt
├── logs/
│   └── agents/
├── worktrees/
└── sessions/
```

Generated `tend-agent` or `pi` profiles may also add `bin/`, `prompts/`, and `.tend/`.

Worktree-local orchestrator metadata lives under `<worktree>/.tend/` and is excluded from commits. `<root>/.tend.lock` is still a root-level `flock` used by `run`/`clean` to serialize git and filesystem side effects; it is not durable state.

## Boundaries

Resume health checks are warning-only and do not repair state. The durable model is the current task snapshot plus worktree lifecycle state in `orchestrator.sqlite`.
