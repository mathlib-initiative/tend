# Orchestrator CLI

The canonical console script is:

```text
tend = tend.orchestrator.cli:main
```

If the first argument is not `init`, `run`, `status`, `export-state`, `clean`, or `validate-config`, the CLI treats the invocation as `run`.

Task-folder utilities are exposed separately as:

```text
tend-task = tend.orchestrator.task_cli:main
```

For now, `tend-task verify <task-dir>` validates a task folder with the same strict checks used by orchestrated merge validation.

## Commands

```bash
uv run tend init --root ./async-root --entrypoint ./repo
uv run tend init --root ./async-root --entrypoint ./repo --agent tend
uv run tend run --root ./async-root
uv run tend status --root ./async-root
uv run tend export-state --root ./async-root --json
uv run tend clean --root ./async-root --dry-run
```

## `init`

Initializes an async orchestration root and writes `config.yaml`.

```bash
uv run tend init \
  --root ./async-root \
  --entrypoint ./repo \
  --agent tend \
  --copy-dir .venv \
  --cow
```

Flags:

| Flag | Meaning |
| --- | --- |
| `--root PATH` | Directory to initialize. Defaults to the current directory. |
| `--entrypoint PATH` | Git repository that contains `tasks/*.yaml`. Defaults to the current directory. |
| `--agent pi\|tend` | Generate worker/reviewer prompts and scripts for `pi` or `tend-agent`. |
| `--worker-prompt-version VERSION` | Worker prompt registry variant under `tend/prompts/worker/` for generated worker agents. Defaults to `minimal`. |
| `--reviewer-prompt-version VERSION` | Reviewer prompt registry variant under `tend/prompts/reviewer/` for generated reviewer agents. Defaults to `minimal`. |
| `--copy-dir DIR` / `--copy_dir DIR` | Copy a relative directory from entrypoint into every new worktree. May be repeated. |
| `--cow` | Add `cp --reflink=always` to generated copy setup. Requires at least one `--copy-dir`. |
| `--mirror-enabled` | Enable workspace mirroring after `git worktree add` and before setup commands. |
| `--symlink-path PATH` | With mirroring, create an absolute symlink for this relative path instead of copying it. May be repeated. |
| `--mirror-exclude-name NAME` | With mirroring, skip path components with this name. May be repeated. |
| `--mirror-exclude-path PATH` | With mirroring, skip this relative subtree. May be repeated. |
| `--mirror-reflink MODE` | Mirror copy reflink policy: `auto`, `required`, or `never`. |
| `--build-command COMMAND` | Post-merge validation/build command to write to config. Disabled by default. |
| `--build-timeout-seconds SECONDS` | Timeout for the configured post-merge build gate. |
| `--no-build-gate` | Explicitly omit the post-merge build gate. |
| `--no-merge-validation-worktree` | Disable staging validation worktree; validate directly in the entrypoint. |
| `--seed-worktree-build` | Seed new worktrees from the staging worktree build cache. |
| `--no-batched-merge` | Disable batched staging merges. |
| `--max-merge-batch-size N` | Cap each batched staging validation to N MERGE worktrees; omit for drain-all behavior. |
| `--skip-build-validation-for-task-only-merges` | Skip the post-merge build gate for merges whose diff changed only files under `tasks/`; the task-tree gate still runs first. |
| `--tend-project PATH` | Pin generated `tend-agent` scripts to a Tend checkout; `run` snapshots it into `<root>/code/`. |
| `--force` | Allow initializing a non-empty root and overwrite managed files. |
| `--log-level LEVEL` | Python log level: `CRITICAL`, `ERROR`, `WARNING`, `INFO`, or `DEBUG`. |

`init` refuses to initialize a non-empty unmarked directory unless `--force` is supplied. It also refuses a root that resolves to the entrypoint repository itself or a child of it; choose a sibling or other outside directory so `<root>/orchestrator.sqlite`, logs, sessions, and worktrees never appear in the entrypoint's `git status --porcelain`. An initialized root contains a `.tend-root` marker.

### Generated `--agent tend` files

`--agent tend` writes:

```text
<root>/bin/worker-agent.sh
<root>/bin/reviewer-agent.sh
<root>/prompts/worker-system.md
<root>/prompts/worker.md
<root>/prompts/worker-revision.md
<root>/prompts/reviewer-system.md
<root>/prompts/reviewer.md
<root>/.tend/worker-agent.yaml
<root>/.tend/reviewer-agent.yaml
<root>/.tend/worker-cfg.yaml
<root>/.tend/reviewer-cfg.yaml
<root>/.tend/README.md
```

The generated `.tend/*-agent.yaml` files use `system_prompt: {path: ../prompts/*-system.md}`. The scripts call `tend-agent` with:

- `--cwd "$TEND_WORKTREE_PATH"`;
- `--session-dir "$TEND_AGENT_SESSION_DIR"`;
- `--prompt "$(cat <root>/prompts/<role>.md)"`;
- `--resume-session` when the orchestrator is resuming that role and an `events.jsonl` session exists.

Set `TEND_AGENT_BIN` if `tend-agent` is not on `PATH`. Editing files under `<root>/prompts/` affects the next launch, including resumed sessions.

### Generated `--agent pi` files

`--agent pi` writes system/task prompts under `prompts/` and scripts under `bin/`. The scripts call `pi --print --mode text`, append the editable system prompt with `--append-system-prompt`, use `--session-dir "$TEND_AGENT_SESSION_DIR"`, and pass `--continue` on resumed sessions.

Set `PI_BIN` if `pi` is not on `PATH`.

## `run`

Runs the orchestrator until all tasks are complete, interrupted, or a framework/configuration error occurs.

```bash
uv run tend run \
  --root ./async-root \
  --max-concurrent-worker-agents 2 \
  --max-concurrent-reviewer-agents 1
```

Flags:

| Flag | Meaning |
| --- | --- |
| `--root PATH` | Required orchestration root containing `config.yaml`. |
| `--entrypoint PATH` | Override the entrypoint repository from `config.yaml`. |
| `--worker-agent-command STRING` | Shell-like command string parsed with `shlex.split`. Overrides config. |
| `--reviewer-agent-command STRING` | Shell-like command string parsed with `shlex.split`. Overrides config. |
| `--worktree-setup-command STRING` | Shell-like setup command. `{entrypoint}` and `{worktree}` placeholders are expanded. |
| `--worker-agent-resume-args STRING` | Args appended to the worker command when that role resumes. |
| `--reviewer-agent-resume-args STRING` | Args appended to the reviewer command when that role resumes. |
| `--max-concurrent-worker-agents N` | Override worker concurrency. Must be >= 1. |
| `--max-concurrent-reviewer-agents N` | Override reviewer concurrency. Must be >= 1. |
| `--max-cost AMOUNT` | Stop claiming new work once accumulated managed-agent cost reaches this ceiling. |
| `--fresh` | Clear saved worktree/task state in `<root>/orchestrator.sqlite` and start fresh. |
| `--dry-run` | Resolve and validate config and state without creating worktrees or launching agents. |
| `--detach` | Spawn the orchestrator in a detached background process and return immediately. |
| `--log-file PATH` | Detached-mode stdout/stderr log file. Defaults to `<root>/run.log`; ignored without `--detach`. |
| `--pid-file PATH` | Detached-mode PID file. Defaults to `<root>/run.pid`; ignored without `--detach`. |
| `--log-level LEVEL` | Python log level. Also controls `<root>/logs.txt`. |

With `--detach`, the parent returns immediately after spawning a child `tend run` process in its own session/process group. The child runs the normal foreground path; its stdout/stderr append to `<root>/run.log` by default (`--log-file` overrides), and its PID is written to `<root>/run.pid` by default (`--pid-file` overrides). The child is not a daemon; it exits when the run completes or is signaled. To stop it and any descendants, use the recorded PID as the process-group ID, for example `kill -- -$(cat <root>/run.pid)`.

By default, `run` resumes from `<root>/orchestrator.sqlite` when it exists and records state transitions there. Aggregate managed-session usage is derived from SQLite worktree rows plus active managed session logs. On process resume, worktrees saved as `worker_running` are returned to `pending` because child processes are not durable across CLI invocations.

When resuming, non-closed worktree paths are checked before runtime queues are rebuilt. Missing paths, non-directories, or paths that no longer look like git worktrees produce warnings in `logs.txt`/stderr. The orchestrator leaves persisted state unchanged and does not queue those unhealthy worktrees for that run.

`run` appends orchestrator logs to `<root>/logs.txt`. It also writes raw per-agent stdout/stderr diagnostics under `<root>/logs/agents/<worktree_id>/`. Agent stdout is still parsed as the role's final JSON output; agent stderr remains diagnostic only.

If `validation_commands` are configured in `config.yaml`, they run after worker success and before review. If `pre_merge_validation_commands` are configured, they run after an approved worktree merge is assembled but before close. With the default `merge_validation_worktree: true`, this happens in `<root>/staging` and the entrypoint only fast-forwards after validation succeeds; with `merge_validation_worktree: false`, validation runs in the entrypoint and failure rolls it back to the pre-merge `HEAD`. With `batched_merge: true`, `max_merge_batch_size` can cap how many queued MERGE worktrees enter each staging batch. In all modes, validation failure returns the worktree to `pending`. There is no `run` flag for overriding either validation list.

If a worker command is not configured, pending worktrees remain queued and no worker process starts. If a reviewer command is not configured, review worktrees remain queued. A practical full run should configure both.

## `status`

Prints a read-only summary from the persisted orchestrator SQLite database.

```bash
uv run tend status --root ./async-root
```

Flags:

| Flag | Meaning |
| --- | --- |
| `--root PATH` | Required orchestration root containing optional `orchestrator.sqlite`. |
| `--log-level LEVEL` | Python log level for CLI diagnostics. |

`status` reads `<root>/orchestrator.sqlite` when it exists. It prints task counts by task status, worktree counts by worktree state, inferred worker/reviewer/merge queue counts from persisted worktree state, and derived aggregate usage counters. It does not run resume health checks. A missing database is reported clearly and is not an error.

The command does not repair state, mutate git, run agents, or inspect session directories beyond the usage aggregation needed for active sessions.

## `export-state`

Dumps the durable SQLite state as JSON for operator inspection.

```bash
uv run tend export-state --root ./async-root --json
```

Flags:

| Flag | Meaning |
| --- | --- |
| `--root PATH` | Required orchestration root containing `orchestrator.sqlite`. |
| `--json` | Required; write JSON containing `worktrees`, `task_snapshot`, and aggregate `usage`. |
| `--log-level LEVEL` | Python log level for CLI diagnostics. |

This replaces ad-hoc `cat state.json`/`cat usage.json` inspection. The database may also have SQLite WAL sidecars (`orchestrator.sqlite-wal` and `orchestrator.sqlite-shm`) while active.

## `validate-config`

Validates the root config without running agents or mutating state.

```bash
uv run tend validate-config --root ./async-root
```

Flags:

| Flag | Meaning |
| --- | --- |
| `--root PATH` | Required orchestration root containing `config.yaml`. |
| `--entrypoint PATH` | Override the entrypoint repository from `config.yaml` during validation. |
| `--log-level LEVEL` | Python log level for CLI diagnostics. |

## `clean`

Removes an initialized async orchestration root and attempts to deregister git worktrees under `<root>/worktrees/`.

```bash
uv run tend clean --root ./async-root --dry-run
uv run tend clean --root ./async-root
```

Flags:

| Flag | Meaning |
| --- | --- |
| `--root PATH` | Required initialized root to remove. |
| `--entrypoint PATH` | Entrypoint repository used for `git worktree remove` and `git worktree prune`. Overrides config. |
| `--dry-run` | Print what would be removed without removing it. |
| `--skip-git` | Skip git worktree deregistration and only remove the root directory. |
| `--log-level LEVEL` | Python log level. |

`clean` refuses to remove directories without the `.tend-root` marker.

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | Success. |
| 2 | Configuration, usage, validation, state, or filesystem error. |
| 70 | Internal/framework software error. |
| 130 | Interrupted by SIGINT. |

Errors are written to stderr as `error[code]: message`.
