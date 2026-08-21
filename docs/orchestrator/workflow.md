# Orchestrator Workflow, Worktrees, Merges, and Cleanup

`AsyncOrchestrator.run()` starts five async services in one `asyncio.TaskGroup`:

1. ready-task discovery;
2. ready-task queue processing;
3. worker agent spawning;
4. reviewer agent spawning;
5. merge queue processing.

Ready-task discovery polls disk every 1.0 seconds. The other services block on queues and wake when work is available.

## Worktree lifecycle

| State | Meaning | Queue |
| --- | --- | --- |
| `pending` | Worktree is ready for a worker. | worker queue when a task is attached |
| `worker_running` | Worker subprocess is currently running. | none |
| `review` | Worker succeeded; reviewer should inspect. | review queue |
| `merge` | Reviewer approved; merge should run. | merge queue |
| `closed` | Worktree is terminal for this run. | none |

Typical route:

```text
ready task -> pending -> worker_running -> review -> merge -> closed
```

Validation failure route:

```text
worker_running --validation failure--> pending -> worker_running -> review
```

Revision route:

```text
review --request_changes--> pending -> worker_running -> review
```

Merge-failure route:

```text
merge --conflict or pre-merge validation failure--> pending -> worker_running -> review -> merge
```

The reviewer verdict is either `approve` (queues the worktree for merge) or `request_changes` (returns the worktree to the worker as `pending`); there is no `deny`. A worktree only leaves the active set by merging successfully or by an operator editing/closing the task in the entrypoint.

## Task discovery

The discovery service reloads `<entrypoint>/tasks/*.yaml` into a new `TaskManager` every poll. It enqueues ready task IDs that have no active worktree.

Queued ready task IDs are ordered by task priority: `max` before `high` before `default`, with file order preserved among tasks of the same priority. Updating a queued task's YAML priority reorders the in-memory task queue on the next discovery pass.

A task with a `pending`, `worker_running`, `review`, or `merge` worktree is not duplicated. A task with only closed worktrees can receive a fresh worktree if it is still open and ready.

## Worktree creation

For each queued ready task, the orchestrator:

1. allocates an ID such as `worktree_000001`;
2. creates `<root>/worktrees/<worktree_id>`;
3. records the current entrypoint `HEAD`;
4. runs `git worktree add --detach <path> <head>`;
5. adds `.tend/` to the worktree git exclude file;
6. runs `worktree_setup_command`, if configured;
7. records the worktree row in `orchestrator.sqlite` after successful provisioning;
8. queues it for a worker.

Worktrees are detached rather than branch-based. The merge step lands only the worker's already-committed changes.

## Worker and reviewer queues

Worker and reviewer queues are de-duplicating FIFO queues. Role-specific concurrency is controlled by:

```yaml
max_concurrent_worker_agents: 20
max_concurrent_reviewer_agents: 20
```

or by the matching CLI overrides.

When the same role runs again for the same worktree, the orchestrator appends that role's `resume_argv` and sets `TEND_AGENT_RESUME=1`. This is how validation failures, requested changes, and merge conflicts continue the same worktree discussion.

## Validation behavior

If `validation_commands` are configured, the orchestrator runs them sequentially after a worker exits successfully and emits valid JSON, but before routing the worktree to review.

Each validation command:

- is a shell-free argv array;
- runs with `cwd=<worktree>`;
- captures stdout and stderr;
- stops the validation sequence on start failure or non-zero exit.

On failure, the worker's message and an orchestrator feedback message are appended to the discussion, the worktree moves back to `pending`, and the worker queue receives the same worktree. The feedback includes the command, exit code or start error, and trimmed stdout/stderr. Passing validation commands do not create durable records.

## Merge behavior

Approved worktrees are published through the merge path. With the default `merge_validation_worktree: true`, the orchestrator:

1. checks `git status --porcelain` in the entrypoint repository;
2. requires each worker worktree to be clean, so only already-committed worker changes can land;
3. ignores `.tend` discussion metadata, which is excluded from commits;
4. assembles the committed candidate(s) on `<root>/staging` from `merge_target_branch`;
5. runs configured `pre_merge_validation_commands` in `<root>/staging`;
6. fast-forwards the entrypoint to the validated staging head;
7. marks the worktree `closed` when merge assembly and pre-merge validation both pass.

If the worktree has no commits ahead of `merge_target_branch`, the orchestrator appends feedback, returns it to `pending`, and does not attempt a merge. When `merge_validation_worktree: false`, the direct-entrypoint fallback checks out `merge_target_branch`, runs `git merge --no-edit -m <message> <worktree_commit>`, and validates in the entrypoint.

The target branch comes from `merge_target_branch` in `<root>/config.yaml` and defaults to `main`. Ensure the branch already exists locally. The orchestrator does not create branches or sync remotes.

If the entrypoint repository is dirty at the preflight check, checkout and merge are not attempted. The orchestrator appends a discussion message with `git status --porcelain` output, moves the worktree back to `pending`, and queues the worker to retry later after the entrypoint is clean.

## Merge failures

If a worker worktree is dirty before review, it is returned to `pending` before merge so the worker can commit or revert the changes. If merge assembly fails, the orchestrator:

- appends an orchestrator message to the worktree discussion with the failed command, exit code, and trimmed stdout/stderr;
- moves the worktree back to `pending`;
- queues the worker to resolve the issue in the same worktree.

Pre-merge validation commands are skipped when the git merge fails. If a `pre_merge_validation_commands` command fails or cannot start after the merge is assembled, the orchestrator appends validation command/output feedback to the discussion, moves the worktree back to `pending`, and queues the worker. With the default staging validation worktree, the entrypoint is unchanged on failure; with the direct-entrypoint fallback, the orchestrator resets the entrypoint to the pre-merge `HEAD`.

Inspect `<worktree>/.tend/discussion.md` for merge feedback.

## Run completion

`run()` exits successfully when the task set is non-empty and every loaded task has `status: complete`.

It returns aggregate usage from managed tend sessions under:

```text
<root>/sessions/<worktree_id>/worker/events.jsonl
<root>/sessions/<worktree_id>/reviewer/events.jsonl
```

Only commands that write tend-compatible session events are counted. The generated `--agent tend` scripts do this. Arbitrary commands and generated `--agent pi` runs are ignored unless they write compatible events in the managed session directory.

Aggregate `Usage` is derived from `<root>/orchestrator.sqlite` plus any active managed session logs. Operators can monitor it with `tend status` or `tend export-state --json`.

## Live controls

A running async orchestrator publishes heartbeat and control metadata to:

```text
<root>/orchestrator.sqlite
```

Use `tend-control` from another terminal to inspect or enqueue operator intents. The
external CLI writes commands to the SQLite store, and the live orchestrator applies
them from inside its scheduler.

Examples:

```bash
uv run tend-control status --root ./async-root
uv run tend-control pause --root ./async-root --wait
uv run tend-control resume --root ./async-root --wait
uv run tend-control limits --root ./async-root --workers 4 --reviewers 2 --wait
uv run tend-control budget --root ./async-root --max-cost 25 --wait
uv run tend-control drain --root ./async-root --wait
uv run tend-control stop --root ./async-root --now --wait
```

`status` combines the control heartbeat with worktree state and derived usage from
`orchestrator.sqlite`, including the latest active worker/reviewer agent snapshot
sampled by the live run. Active-agent rows are heartbeat-scoped, best-effort snapshots: the row role
is the runtime agent role, while `worktree_state` reflects the worktree's current
state, which may have advanced to `review` or `merge` while the agent task is
still counted until the next heartbeat or cleanup.

Control semantics:

- `pause` stops new ready-task, worktree, worker, and reviewer launches while
  allowing already-running work and merges to settle.
- `resume` reopens those non-terminal gates after a pause.
- `limits` changes live worker/reviewer launch limits; `0` is accepted and means
  no new launches for that role, except that a limit cannot be set to `0` while
  drain is active.
- `budget` changes the live max-cost ceiling in the run's configured budget
  currency. If the current accumulated cost already meets the new ceiling, the
  usual budget drain starts immediately.
- `drain` stops claiming fresh ready tasks / creating fresh task worktrees, but
  keeps launching already-created worker and reviewer worktrees and keeps
  processing merges. It exits once worker, review, and merge queues are settled.
  If a worker or reviewer launch limit is `0`, raise it before draining or use
  stop semantics instead; limits also cannot be lowered to `0` during drain.
- `stop --now` requests terminal stop semantics and cancels running worker and
  reviewer agent tasks.

With `--wait`, `tend-control` waits for the command to reach a terminal command
status. For `drain` and `stop`, it then also waits for the targeted run to reach
a terminal run status. Use `--wait-timeout SECONDS` to bound the wait.

## Cleanup

`tend clean` removes a whole initialized root. Without `--skip-git`, it first loops over directories under `<root>/worktrees/` and runs:

```bash
git -C <entrypoint> worktree remove --force <worktree>
git -C <entrypoint> worktree prune
```

Then it removes the root directory.

Use dry-run first:

```bash
uv run tend clean --root ./async-root --dry-run
uv run tend clean --root ./async-root
```

If the entrypoint repository is unavailable, `--skip-git` removes the root only. You may need to run `git worktree prune` manually later.

## State persistence and resume

The orchestrator persists durable state to a single SQLite database:

```text
<root>/orchestrator.sqlite
<root>/orchestrator.sqlite-wal  # transient WAL sidecar
<root>/orchestrator.sqlite-shm  # transient WAL sidecar
```

It stores the last loaded task-manager snapshot, next worktree sequence, worktree lifecycle rows, task associations, discussion messages, review verdicts, role-session-started flags, per-role usage snapshots, and live control metadata. Aggregate usage is derived from those rows plus any active managed session logs; there is no separate `usage.json`.

Runtime queues and subprocess objects are not persisted. On `tend run`, the CLI automatically resumes when `orchestrator.sqlite` exists. A saved `worker_running` worktree is changed back to `pending` before resume because the child process from the prior CLI invocation no longer exists. The worker session flag remains set, so generated agents receive resume arguments and `TEND_AGENT_RESUME=1`.

Before rebuilding queues, the resumed runtime checks every non-closed worktree path. If a path is missing, is not a directory, or does not look like the git worktree recorded in SQLite, the orchestrator logs a warning, leaves persisted state unchanged, and does not queue that worktree for the current run. This avoids a destructive repair loop; fix the filesystem/state issue and restart, or clean the run root when you want to abandon it.

Use `--fresh` to ignore saved state and start over without deleting the root:

```bash
uv run tend run --root ./async-root --fresh
```

For JSON inspectability similar to the removed snapshots, run:

```bash
uv run tend export-state --root ./async-root --json
```

`--fresh` does not remove old worktrees or sessions. The new run clears durable worktree/task state in `orchestrator.sqlite` before starting. Use `tend clean` when you want to remove the root and registered worktrees.
