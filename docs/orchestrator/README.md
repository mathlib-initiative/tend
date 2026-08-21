# Orchestrator

The orchestrator is the lightweight queue-driven multi-agent runner implemented in `tend.orchestrator` and exposed by the `tend` console script.

It watches YAML task files in a real git repository, creates detached git worktrees for ready tasks, launches worker and reviewer commands concurrently, records worker/reviewer discussion, and merges approved worktree changes back into the configured target branch.

## Current status

This is implemented and tested as a local queue-driven orchestration loop:

- task state comes from `tasks/*.yaml` in the configured entrypoint repository;
- orchestration state is persisted to `<root>/orchestrator.sqlite` after every state change;
- aggregate managed agent usage is derived from SQLite rows plus active managed session logs;
- saved state is automatically loaded from `orchestrator.sqlite` by `tend run` unless `--fresh` is supplied;
- worktrees and agent session directories are written under the async orchestration root;
- workers and reviewers are arbitrary shell-free commands that must print strict JSON to stdout;
- optional validation commands can run after worker success and before review;
- optional pre-merge validation commands can run on an assembled merge before close; by default this happens in `<root>/staging` before the entrypoint fast-forwards;
- approved worktree commits are merged into `merge_target_branch` (default `main`);
- `tend status` prints a read-only summary from `orchestrator.sqlite`;
- `tend clean` removes the orchestration root and registered worktrees.

It is **not** a daemon: subprocesses and queue reservations are still runtime-only. On process restart, `tend run` rebuilds queues from `orchestrator.sqlite`; any worktree saved as `worker_running` is returned to `pending` so the worker can resume from its managed session and discussion log. Non-closed worktree paths that are missing or no longer look like git worktrees are logged as resume health warnings and are not queued for that run.

## Contents

- [Quickstart for real-life runs](quickstart.md)
- [Architecture](architecture.md)
- [CLI](cli.md)
- [Configuration](configuration.md)
- [Tasks and dependency readiness](tasks.md)
- [Worker/reviewer agents](agents.md)
- [Workflow, worktrees, merges, and cleanup](workflow.md)
- [Public API](api.md)

## Runtime layout

`tend init --root <root>` creates the orchestration root. Keep `<root>` outside the entrypoint repository; `init` refuses roots that resolve to the entrypoint itself or a child path so runtime files do not dirty the source repository. `run` later adds/updates `orchestrator.sqlite` (and SQLite WAL sidecars), and log files:

```text
<root>/
├── .tend-root
├── config.yaml
├── orchestrator.sqlite     # created/updated by run
├── orchestrator.sqlite-wal # present while WAL has uncheckpointed pages
├── orchestrator.sqlite-shm # present while WAL is active
├── logs.txt                # appended by run
├── logs/
│   └── agents/             # raw per-agent stdout/stderr diagnostics
├── worktrees/
└── sessions/
```

With `--agent tend` or `--agent pi`, it also creates:

```text
<root>/
├── bin/
│   ├── worker-agent.sh
│   └── reviewer-agent.sh
├── prompts/
│   ├── worker-system.md
│   ├── worker.md
│   ├── worker-revision.md  # only for --agent tend
│   ├── reviewer-system.md
│   └── reviewer.md
└── .tend/              # only for --agent tend
```

During `run`, the CLI writes durable state to `<root>/orchestrator.sqlite`, derives aggregate managed session usage from that database and active session logs, appends orchestrator logs to `<root>/logs.txt`, and writes raw agent stdout/stderr diagnostics under `<root>/logs/agents/<worktree_id>/`. Each worktree receives a discussion log at `<worktree>/.tend/discussion.md`; `.tend/` is excluded from commits and merges. `<root>/.tend.lock` remains a root-level `flock` for git/filesystem side effects, not durable state.

## Workflow at a glance

1. Poll `<entrypoint>/tasks/*.yaml` and validate them as a dependency DAG.
2. Enqueue open tasks whose dependencies are complete, ordered by `priority` (`max`, then `high`, then `default`).
3. Create one detached git worktree under `<root>/worktrees/` per ready task that has no active worktree.
4. Launch worker commands for pending worktrees, respecting worker concurrency.
5. Parse worker stdout as the shared `worker_contribution` payload: `{ "schema_version": 1, "status": "completed|blocked|needs_review", "summary": "...", ... }`. `blocked` returns the worktree to the worker queue (`pending`).
6. Run configured validation commands for worktrees bound for review; failures return the worktree to `pending` with feedback.
7. Route validated worktrees to review and launch reviewer commands, respecting reviewer concurrency.
8. Parse reviewer stdout as the shared `review_verdict` payload: `{ "schema_version": 1, "verdict": "approve|request_changes", "notes": "...", "feedback_text": "..." }`.
9. Merge approved worktrees into the configured target branch. By default, assemble and validate the merge in `<root>/staging`, then fast-forward the entrypoint only after validation succeeds; return dirty-entrypoint preflight failures, merge conflicts, validation failures, or requested changes to the worker queue.
10. Stop when the task set is non-empty and every task is `complete`.

## Important limitations

- Active subprocesses are not persisted. A restarted run resumes queues from `orchestrator.sqlite`, not from live child processes.
- Resume health checks warn about missing or invalid non-closed worktree paths, but do not recreate, delete, or reset worktrees.
- Process-group cleanup is best-effort for normal `SIGINT`/`SIGTERM`; it cannot clean up after `SIGKILL` or deliberately daemonized descendants.
- The merge target branch defaults to `main`; the orchestrator does not create branches or sync remotes.
- If the entrypoint repository is dirty, approved worktree checkout/merge is skipped and the worktree returns to `pending`.
- Agent stdout must stay valid JSON. Raw stdout/stderr debug copies are written under `<root>/logs/agents/`, but only stdout controls the protocol.
- Agent commands inherit the parent process environment plus the documented `TEND_*` variables; there is no orchestrator-specific sandbox or environment allowlist yet.
- Validation commands are minimal sequential gates only; there are no durable validation records or artifact integrations.
