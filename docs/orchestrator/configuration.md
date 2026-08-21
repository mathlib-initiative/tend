# Orchestrator Configuration

The orchestrator has one durable project config file at `<root>/config.yaml` and a runtime config model used by library callers.

## `<root>/config.yaml`

`tend init` writes an `AsyncOrchestratorProjectConfig`:

```yaml
entrypoint: /absolute/path/to/repo
worker_agent_command:
  argv:
    - /absolute/path/to/async-root/bin/worker-agent.sh
  resume_argv:
    - --resume
reviewer_agent_command:
  argv:
    - /absolute/path/to/async-root/bin/reviewer-agent.sh
  resume_argv:
    - --resume
worktree_setup_command:
  argv:
    - cp
    - --archive
    - "{entrypoint}/.venv"
    - "{worktree}/"
validation_commands:
  - argv:
      - uv
      - run
      - ruff
      - check
pre_merge_validation_commands:
  - argv:
      - uv
      - run
      - pytest
      - -m
      - not live
merge_target_branch: main
max_merge_batch_size: 8  # optional; omit for uncapped drain-all batches
max_concurrent_worker_agents: 20
max_concurrent_reviewer_agents: 20
```

Fields:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `entrypoint` | path | required | Git repository containing `tasks/*.yaml`; CLI relative paths resolve from the config directory. |
| `worker_agent_command` | command or null | `null` | Command used for pending task worktrees. |
| `reviewer_agent_command` | command or null | `null` | Command used for worktrees in review. |
| `worktree_setup_command` | command or null | `null` | Command run after each worktree is created. |
| `validation_commands` | list of commands | `[]` | Sequential commands run after worker success and before review. |
| `pre_merge_validation_commands` | list of commands | `[]` | Sequential commands run after an approved worktree merge is assembled and before closing it. |
| `merge_target_branch` | string | `main` | Local entrypoint branch checked out and merged into for approved worktrees. |
| `workspace_mirror` | object | disabled | Mirror/symlink selected entrypoint paths into new worktrees. |
| `merge_validation_worktree` | bool | `true` | Validate approved merges in `<root>/staging` before fast-forwarding the entrypoint. |
| `seed_worktree_build` | bool | `false` | Seed new worktrees from the staging worktree build cache. |
| `batched_merge` | bool | `true` | Validate/publish queued approved worktrees in batches when staging is enabled. |
| `max_merge_batch_size` | positive int or null | `null` | Optional cap on MERGE worktrees included in one staging batch; `null`/omitted drains all visible queued merges. |
| `skip_build_validation_for_task_only_merges` | bool | `false` | Skip `pre_merge_validation_commands` for merges whose diff changed only files under `tasks/`; the build-free task-tree gate still runs first. |
| `max_concurrent_worker_agents` | positive int | `20` | Maximum active worker subprocesses. |
| `max_concurrent_reviewer_agents` | positive int | `20` | Maximum active reviewer subprocesses. |
| `budget` | object | no max cost, USD | Optional run-level managed-agent cost ceiling. |
| `agent_oom_score_adj` | int or null | `750` | Linux OOM score adjustment for spawned agent/build subprocesses. |
| `cleanup_closed_worktrees` | bool | `true` | Best-effort removal of safely closed worktree directories. |

A config with missing agent commands is valid, but a full live run will stall at the corresponding queue. The merge target branch must already exist locally; the orchestrator does not create branches or sync remotes.

## Command configuration

Agent commands are shell-free argv arrays:

```yaml
worker_agent_command:
  argv:
    - /opt/bin/my-worker
    - --mode
    - async
  resume_argv:
    - --resume
```

`argv` is used for the first session in a worktree. When the same role is run again for the same worktree, the orchestrator appends `resume_argv` and sets `TEND_AGENT_RESUME=1`.

Validation rules:

- `argv` must have at least one item;
- every argument must be a non-blank string;
- arguments must not contain NUL bytes;
- there is no shell expansion unless your command explicitly invokes a shell.

CLI flags such as `--worker-agent-command "tend-agent --prompt worker"` are convenience strings parsed with `shlex.split` into argv arrays.

## Worktree setup command

`worktree_setup_command` is also shell-free:

```yaml
worktree_setup_command:
  argv:
    - cp
    - --archive
    - --reflink=always
    - "{entrypoint}/.lake"
    - "{worktree}/"
```

It runs with `cwd=<worktree>` immediately after `git worktree add --detach` and before the worktree is queued for a worker. The command must be interruption-safe and incrementally correct when retried over artifacts left by an interrupted attempt. On POSIX, the orchestrator launches it in its own process-group session; on cancellation it sends SIGTERM to the group, waits briefly, then escalates with SIGKILL and waits for one more bounded interval before cleanup and shutdown. On Windows, only the direct child is terminated; descendant containment requires POSIX process groups and remains part of the stronger containment work tracked in issue #152. If command startup itself is unresponsive before publishing a process to signal, the daemon setup worker is abandoned with a warning rather than blocking shutdown.

On POSIX, setup commands must not call `setsid`, daemonize, or otherwise detach descendants into another session. Such descendants—and a command whose process startup is itself unresponsive—can outlive cancellation and overlap cleanup or mutate a later staging checkout; stronger containment remains tracked in issue #152. Treat orchestrator-root metadata, including the sibling `staging.provisioned` readiness sentinel, as reserved even though a staging setup command's `cwd=<root>/staging` makes parent paths reachable.

Only two placeholders are accepted:

| Placeholder | Expands to |
| --- | --- |
| `{entrypoint}` | Absolute path of the entrypoint repository. |
| `{worktree}` | Path of the newly created worktree. |

Format specs, conversions, and other placeholder names are rejected. Use `tend init --copy-dir DIR [--cow]` to generate common copy commands safely.

## Validation commands

`validation_commands` is a list of shell-free argv commands:

```yaml
validation_commands:
  - argv:
      - uv
      - run
      - ruff
      - check
  - argv:
      - uv
      - run
      - pytest
      - -m
      - not live
```

They run sequentially with `cwd=<worktree>` after a worker exits successfully and prints valid JSON, but before the worktree enters review. The first command that cannot start or exits non-zero stops validation. The orchestrator appends the command, exit code or start error, and trimmed stdout/stderr to the worktree discussion, then returns the worktree to `pending` for the worker to fix.

Validation commands do not use shell expansion unless the argv explicitly invokes a shell. They inherit the parent process environment. The orchestrator does not create durable validation records or artifacts.

All validation commands must be interruption-safe and incrementally correct when rerun over caches or other ignored output left by an interrupted predecessor. A command that cannot trust its own partial artifacts must clean them itself. On POSIX, commands must not call `setsid`, daemonize, or otherwise detach descendants: cancellation terminates the command's process group, but escaped processes can outlive it and overlap a retry or later batch. On Windows, only the direct child is terminated; descendant containment requires POSIX process groups and is tracked in issue #152.

## Pre-merge validation commands

`pre_merge_validation_commands` uses the same shell-free argv schema:

```yaml
pre_merge_validation_commands:
  - argv: [uv, run, pytest, -m, not live]
```

These commands run sequentially after an approved worktree merge is assembled, but before the worktree is marked closed. With the default `merge_validation_worktree: true`, the merge and validation happen in `<root>/staging`; the entrypoint is fast-forwarded only after validation succeeds. With `merge_validation_worktree: false`, the merge and validation happen in the entrypoint and failure resets the entrypoint back to the pre-merge `HEAD`. When `batched_merge: true`, the staging path validates queued MERGE worktrees together; set `max_merge_batch_size` to a positive integer to cap each batch while leaving later MERGE worktrees queued for the next batch. In all modes, the first command that cannot start or exits non-zero stops the sequence, appends command/output feedback to the worktree discussion, returns the worktree to `pending`, and queues the worker to retry.

With `skip_build_validation_for_task_only_merges: true`, a merge (or assembled batch) whose entire (non-empty) diff stays under `tasks/` skips these commands: the build-free post-merge task-tree gate (strict YAML parse plus an acyclic `depends_on` graph) still runs first and must pass, and a merge touching any non-task path — or whose endpoint diff is empty — runs the commands exactly as before. **Enable this only if your validation commands consume nothing under `tasks/`** (no Lean `include_str "tasks/..."`, no custom Lake facets or scripts reading task files): the option is an operator assertion of that fact, which tend does not verify. Under that assertion it trims merge-queue latency for task-only contributions without weakening source validation.

In staging-worktree mode, a crash-signal exit such as SIGSEGV triggers a cold rebuild: the orchestrator clears staging readiness first, purges all ignored state with `git clean -ffdx`, and reapplies workspace mirroring and setup before validation resumes. This crash-purge guarantee does **not** apply with `merge_validation_worktree: false`; the orchestrator will not delete ignored files from the user's entrypoint, so the validation command owns crash-leftover cleanup there. In the batched staging path (`batched_merge: true` and `merge_validation_worktree: true`), cancellation-class signals are retried in place rather than crash-purged, making incremental correctness after interruption especially important. Non-batched staging and entrypoint validation instead treat the result as an immediate validation failure and return the worktree to `pending`.

`<root>/staging` is orchestrator-private. `tend run` holds an exclusive advisory lock on the root for the complete run. Before resetting, cleaning, or reusing an existing staging path, the orchestrator requires it to be a real (non-symlink) directory registered as this entrypoint's staging worktree; any unexpected object is rename-quarantined as `staging.invalid-*` without following it, then a fresh staging worktree is provisioned. External tools must not register another worktree at that path while an orchestrator owns the root.

If the git merge itself fails, pre-merge validation commands are not run; the existing merge-failure feedback and rollback path handles that case.

## Runtime overrides from CLI

`tend run` loads `<root>/config.yaml`, resolves the entrypoint, and then applies CLI overrides:

- `--entrypoint`
- `--worker-agent-command`
- `--reviewer-agent-command`
- `--worktree-setup-command`
- `--worker-agent-resume-args`
- `--reviewer-agent-resume-args`
- `--max-concurrent-worker-agents`
- `--max-concurrent-reviewer-agents`
- `--max-cost`

There is no CLI override for `validation_commands` or `pre_merge_validation_commands`; edit `<root>/config.yaml` to change them.

The CLI writes orchestrator logs to `<root>/logs.txt` for every `run` invocation and raw agent stdout/stderr diagnostics under `<root>/logs/agents/<worktree_id>/` after each completed agent process.

## Programmatic config

Library callers can construct `AsyncOrchestratorConfig` directly:

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
        max_concurrent_worker_agents=2,
    )
    result = await AsyncOrchestrator(config).run()
    print(result.usage)


asyncio.run(main())
```

`AsyncOrchestratorConfig.from_paths(...)` also accepts simple `Sequence[str]` command values and coerces them into command config objects.

## Environment inherited by agents

Agent subprocesses inherit the parent process environment and receive these additional retained compatibility variables:

| Variable | Present for | Meaning |
| --- | --- | --- |
| `TEND_ROOT` | worker/reviewer | Absolute orchestration root. |
| `TEND_ENTRYPOINT` | worker/reviewer | Absolute entrypoint repository. |
| `TEND_WORKTREE_PATH` | worker/reviewer | Current worktree path and process cwd. |
| `TEND_WORKTREE_ID` | worker/reviewer | Worktree id such as `worktree_000001`. |
| `TEND_WORKTREE_HEAD` | worker/reviewer | Entry point commit used to create the worktree. |
| `TEND_AGENT_ROLE` | worker/reviewer | `worker` or `reviewer`. |
| `TEND_AGENT_RESUME` | worker/reviewer | `1` when the same role already started in this worktree, else `0`. |
| `TEND_AGENT_SESSION_DIR` | worker/reviewer | Managed session directory for this worktree/role. |
| `TEND_AGENT_DISCUSSION_PATH` | worker/reviewer | Markdown discussion log path inside the worktree. |
| `TEND_TASK_ID` | worker | Task id attached to the worktree, when any. |

There is no orchestrator-specific environment allowlist yet. Do not run untrusted agent commands with sensitive host environment variables present.
