# Orchestrator Quickstart

This quickstart starts a real local run against an existing git repository. The commands use `tend-agent` because `tend init --agent tend` creates ready-to-edit worker/reviewer scripts and prompts. Use `--agent pi` instead if you want the generated scripts to call `pi`.

## 0. Preconditions

From the repository you want agents to modify:

```bash
git status --short
git branch --show-current
git config user.email
git config user.name
```

For the current implementation, prefer:

- a clean entrypoint working tree before merges;
- a local target branch for approved worktrees (defaults to `main`);
- local git user name/email configured, because the orchestrator creates merge commits;
- at least one committed task file under `tasks/` before you start the run.

## 1. Initialize an orchestration root

Keep the orchestration root outside the source repository. `tend init` refuses a root that resolves to the entrypoint repository itself or one of its children, because runtime files would otherwise make the entrypoint dirty during merge preflight.

```bash
export REPO_ROOT="$(pwd)"
export ASYNC_ROOT="../tend-run"

uv run tend init \
  --root "$ASYNC_ROOT" \
  --entrypoint "$REPO_ROOT" \
  --agent tend
```

This writes `$ASYNC_ROOT/config.yaml`, worker/reviewer prompts, executable scripts, and default `.tend/*.yaml` files.

If your repository depends on a large gitignored directory that should exist in every worktree, include copy setup in the init command instead of the minimal command above. For example, on filesystems with reflink support:

```bash
uv run tend init \
  --root "$ASYNC_ROOT" \
  --entrypoint "$REPO_ROOT" \
  --agent tend \
  --copy-dir .venv \
  --cow
```

`--copy-dir` may be repeated. It creates a worktree setup command equivalent to `cp --archive [--reflink=always] {entrypoint}/DIR {worktree}/`.

## 2. Review generated agent policy

Open these files before the first live run:

```bash
$EDITOR "$ASYNC_ROOT/prompts/worker-system.md"
$EDITOR "$ASYNC_ROOT/prompts/worker.md"
$EDITOR "$ASYNC_ROOT/prompts/worker-revision.md"
$EDITOR "$ASYNC_ROOT/prompts/reviewer-system.md"
$EDITOR "$ASYNC_ROOT/prompts/reviewer.md"
$EDITOR "$ASYNC_ROOT/.tend/worker-agent.yaml"
$EDITOR "$ASYNC_ROOT/.tend/worker-cfg.yaml"
$EDITOR "$ASYNC_ROOT/.tend/reviewer-agent.yaml"
$EDITOR "$ASYNC_ROOT/.tend/reviewer-cfg.yaml"
```

The generated `tend-agent` config defaults to the Anthropic Messages API and a high output budget. Change model/provider settings if needed. The generated scripts honor `TEND_AGENT_BIN` if `tend-agent` is not on `PATH`.

Export the API key before the run:

```bash
export ANTHROPIC_API_KEY="<anthropic-api-key>"
```

If you change the generated `.tend/*.yaml` files to use OpenAI or another provider, export the corresponding variables expected by those files.

## 3. Add and commit task files

Create `tasks/001-seed.yaml` in the entrypoint repository:

```yaml
schema_version: 1
id: task-001
title: "Add a focused production change"
status: open
priority: default
depends_on: []
summary: Add a focused production change
description: |
  Make one small, reviewable change. Update tests or docs as appropriate and
  finish this task by setting status: complete when done.
```

Commit task files before running. Worktrees are created from git `HEAD`, so uncommitted tasks in the entrypoint may not be visible to worker agents.

```bash
mkdir -p tasks
$EDITOR tasks/001-seed.yaml
uv run tend-task verify tasks
git add tasks/001-seed.yaml
git commit -m "seed orchestrator task"
```

Use more files for parallelizable work. A task is ready when it is `open` and every task listed in `depends_on` is `complete`. Ready tasks are picked by `priority` (`max`, then `high`, then `default`), preserving file order within each priority.

## 4. Run the orchestrator

Optionally edit `$ASYNC_ROOT/config.yaml` to add a minimal validation gate before review:

```yaml
validation_commands:
  - argv: [uv, run, ruff, check]
  - argv: [uv, run, pytest, -m, not live]
```

Validation commands run in each worktree after worker success and before reviewer handoff. A failing command returns the worktree to `pending` with stdout/stderr feedback in `.tend/discussion.md`.

You can also validate the merged result before close:

```yaml
pre_merge_validation_commands:
  - argv: [uv, run, pytest, -m, not live]
```

These commands run after an approved worktree merge is assembled. With the default staging validation worktree, the merge and validation happen in `<root>/staging` and the entrypoint fast-forwards only after validation succeeds. If staging validation is disabled, they run in the entrypoint and failure rolls the entrypoint back to the pre-merge `HEAD`. In both modes, failure records feedback in the worktree discussion and returns the worktree to `pending`.

Start with one worker and one reviewer until your prompts and validation expectations are stable:

```bash
uv run tend run --root "$ASYNC_ROOT" --log-level INFO
```

Then raise concurrency when you are comfortable with merge pressure and provider budget:

```bash
uv run tend run \
  --root "$ASYNC_ROOT" \
  --max-concurrent-worker-agents 2 \
  --max-concurrent-reviewer-agents 1 \
  --log-level INFO
```

The process runs until the task set is non-empty and all task files in the entrypoint have `status: complete`. It writes durable state to `<root>/orchestrator.sqlite` (plus normal SQLite `-wal`/`-shm` sidecars while active). If the process exits before completion, the next `tend run --root "$ASYNC_ROOT"` automatically resumes from that database and rebuilds queues; use `--fresh` only when you intentionally want to clear saved worktree/task state and start over.

## 5. Monitor a live run

Useful files and commands:

```bash
# Read-only summary plus persisted orchestrator state, aggregate usage, and logs
uv run tend status --root "$ASYNC_ROOT"
uv run tend export-state --root "$ASYNC_ROOT" --json | jq .
less "$ASYNC_ROOT/logs.txt"
find "$ASYNC_ROOT/logs/agents" -type f 2>/dev/null | sort

# Worktrees and per-worktree discussion logs
find "$ASYNC_ROOT/worktrees" -maxdepth 3 -path '*/.tend/discussion.md' -print

# tend-agent session events/usage for generated --agent tend scripts
find "$ASYNC_ROOT/sessions" -maxdepth 4 -type f | sort

# Source repository task status and merged changes
git -C "$REPO_ROOT" status --short
git -C "$REPO_ROOT" log --oneline --decorate -n 10
```

The worker final response uses the shared `worker_contribution` contract, exactly JSON like:

```json
{"schema_version":1,"status":"completed","summary":"Implemented the task and ran validation."}
```

The reviewer final response uses the shared `review_verdict` contract, exactly JSON like:

```json
{"schema_version":1,"verdict":"approve","notes":"All criteria PASS; looks good."}
```

Any extra stdout makes the agent output invalid and returns the worktree to its previous queue. Use stderr for diagnostics; raw stdout/stderr copies are saved under `<root>/logs/agents/` after each completed agent run.

## 6. Finish or recover

When all tasks are complete, `tend run` exits and logs aggregate usage for managed `tend-agent` sessions.

If a run fails early because agent config or provider environment was wrong, inspect `tend export-state --json`, `logs.txt`, agent debug logs under `logs/agents/`, and the relevant worktree discussion. To continue, fix the environment and run the same command again; `tend` resumes automatically. If resume logs warn that a non-closed worktree path is missing or invalid, fix that worktree/SQLite-state issue before expecting it to be queued again.

To abandon saved state and start from scratch without deleting the root, use:

```bash
uv run tend run --root "$ASYNC_ROOT" --fresh --log-level INFO
```

To remove all run worktrees and sessions, clean the root:

```bash
uv run tend clean --root "$ASYNC_ROOT" --dry-run
uv run tend clean --root "$ASYNC_ROOT"
```

Then re-run `init`, verify provider variables, and start a fresh run. If the entrypoint repo is unavailable, `clean --skip-git` removes the root but leaves git worktree registrations for you to prune manually later.
