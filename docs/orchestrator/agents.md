# Orchestrator Agents

Workers and reviewers are external commands launched inside an orchestrator worktree. They can be `tend-agent`, `pi`, custom scripts, or any executable that follows the stdout JSON contract.

## Execution model

For each agent run, the orchestrator calls `asyncio.create_subprocess_exec` with shell-free argv:

- `cwd` is the assigned worktree path;
- stdout is captured and parsed after the process exits;
- raw stdout and stderr are copied to per-attempt debug log files;
- the subprocess starts in its own process group/session;
- on orchestrator cancellation, `tend` sends `SIGTERM` to the agent process group, waits briefly, then sends `SIGKILL` if needed;
- the host environment is inherited and augmented with `TEND_*` variables;
- a role-specific session directory is created at `<root>/sessions/<worktree_id>/<role>/`.

Because stdout is the structured protocol, do not print progress logs to stdout. Write diagnostics to stderr, files, a Tend session, or another logging destination. Process-group cleanup covers normal `SIGINT`/`SIGTERM` cancellation; it is not a guarantee for `SIGKILL` or for child processes that deliberately daemonize or escape their process group.

## Worker contract

A worker runs for a worktree in `pending` state and receives `TEND_TASK_ID` when the worktree is associated with a task. Generated agents are seeded from the bundled worker prompt version selected at init (`worker/<version>`, default `minimal`) and use the shared `worker_contribution` output schema.

The final stdout (via the `final_result` tool) is the `worker_contribution` payload:

```json
{"schema_version":1,"status":"completed","summary":"Implemented task-001 and ran uv run pytest -m 'not live'."}
```

Schema (`WorkerContributionOutput`):

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | `1` | Output schema version. |
| `status` | `completed`, `blocked`, or `needs_review` | Self-assessed outcome that drives routing. |
| `summary` | non-blank string | Summary appended to `.tend/discussion.md` and shown to the reviewer. |
| `files_changed` | list | Optional list of changed file paths. |
| `validation` | list | Optional command/exit-code evidence. |
| `tasks_created` | list | Optional list of newly created task ids. |
| `notes` | string | Optional extra notes; appended after `summary` in the discussion log. |

A `status` of `completed` or `needs_review` sends the worktree on to review; `blocked` returns it to the worker queue (`pending`) for another pass. When bound for review, configured validation commands run first. If validation passes, the worktree moves to `review`. On worker non-zero exit, invalid JSON, or validation failure, the worktree returns to `pending` so a worker can try again.

## Reviewer contract

A reviewer runs for a worktree in `review` state. Generated agents are seeded from the bundled reviewer prompt version selected at init (`reviewer/<version>`, default `minimal`) and use the shared `review_verdict` output schema.

The final stdout (via the `final_result` tool) is the `review_verdict` payload:

```json
{"schema_version":1,"verdict":"approve","notes":"All criteria PASS; diff is focused and the build is green."}
```

Schema (`ReviewVerdictOutput`):

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | `1` | Output schema version. |
| `verdict` | `approve` or `request_changes` | Routing decision. There is no `deny`. |
| `notes` | non-blank string | Per-criterion walkthrough; appended to `.tend/discussion.md`. |
| `feedback_text` | string | Required iff `request_changes`; appended after `notes` in the discussion log. |
| `comments` | list | Optional structured per-location comments. |

Routing:

| Verdict | Next state | Effect |
| --- | --- | --- |
| `approve` | `merge` | Worktree is queued for merge assembly, followed by any pre-merge validation commands. By default validation happens in `<root>/staging` before the entrypoint fast-forwards. |
| `request_changes` | `pending` | Same worktree is returned to the worker queue; worker session resumes. |

Use `request_changes` for fixable review issues. For impossible or unsafe tasks, a human should edit or close the task in the entrypoint repository.

## Debug logs

After each completed agent run, the orchestrator writes raw captured output to:

```text
<root>/logs/agents/<worktree_id>/<role>-<attempt>.stdout
<root>/logs/agents/<worktree_id>/<role>-<attempt>.stderr
```

`<role>` is `worker` or `reviewer`; `<attempt>` starts at `1` per role and worktree. Stdout is still the structured JSON protocol. Stderr is diagnostic only and is never parsed as protocol output.

## Discussion log

Before each agent run, the orchestrator writes:

```text
<worktree>/.tend/discussion.md
```

The path is also available as `TEND_AGENT_DISCUSSION_PATH`. It contains worker, reviewer, and orchestrator messages for that worktree. `.tend/` is excluded from commits, so discussion logs do not merge back into the entrypoint.

A worker resumed after validation failure, `request_changes`, or merge conflict feedback should read this file and address the latest message.

## Environment variables

| Variable | Meaning |
| --- | --- |
| `TEND_ROOT` | Absolute orchestration root. |
| `TEND_ENTRYPOINT` | Absolute entrypoint repository. |
| `TEND_WORKTREE_PATH` | Current worktree path. |
| `TEND_WORKTREE_ID` | Worktree id such as `worktree_000001`. |
| `TEND_WORKTREE_HEAD` | Commit used to create the worktree. |
| `TEND_TASK_ID` | Task id, when the worktree has one. |
| `TEND_AGENT_ROLE` | `worker` or `reviewer`. |
| `TEND_AGENT_RESUME` | `1` when this role already started for the worktree, else `0`. |
| `TEND_AGENT_SESSION_DIR` | Role-specific session directory. |
| `TEND_AGENT_DISCUSSION_PATH` | Discussion log path. |

The generated `--agent tend` scripts pass `TEND_AGENT_SESSION_DIR` to `tend-agent --session-dir`, which lets `tend` aggregate usage from tend session events.

## Generated `tend-agent` scripts

`tend init --agent tend` creates scripts that run:

```bash
tend-agent \
  --agent <root>/.tend/worker-agent.yaml \
  --config <root>/.tend/worker-cfg.yaml \
  --cwd "$TEND_WORKTREE_PATH" \
  --session-dir "$TEND_AGENT_SESSION_DIR" \
  --prompt "$(cat <root>/prompts/worker.md)"
```

The generated agent YAML points its system prompt at `<root>/prompts/worker-system.md` or `<root>/prompts/reviewer-system.md` with `system_prompt: {path: ...}`. On resumed sessions, the script adds `--resume-session` when `events.jsonl` exists in the session directory. Editing any prompt file under `<root>/prompts/` affects the next launch.

The generated worker config enables read, write, edit, `copy_lines`, grep/glob/list, and bash tools. The generated reviewer config omits write/edit/`copy_lines` tools but the process is still not sandboxed by `tend`; do not run untrusted commands.

## Generated `pi` scripts

`tend init --agent pi` creates scripts that run `pi --print --mode text` with a session directory, generated task prompt, and `--append-system-prompt` loaded from the editable system prompt file. On resumed sessions, the script adds `--continue`.

## Custom agent command checklist

A custom command should:

1. read task/discussion context from the worktree and `TEND_*` variables;
2. make changes only inside `TEND_WORKTREE_PATH`;
3. keep stdout reserved for the final JSON object;
4. use stderr or files for progress/debug output;
5. exit non-zero only when the same queue should retry later;
6. support resume behavior when `TEND_AGENT_RESUME=1` or when resume args are appended.

Minimal worker script:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Make changes in "$TEND_WORKTREE_PATH"...
printf '{"schema_version":1,"status":"completed","summary":"Made the requested changes."}'
```

Minimal reviewer script:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Inspect git diff, run validation, read "$TEND_AGENT_DISCUSSION_PATH"...
printf '{"schema_version":1,"verdict":"approve","notes":"All criteria PASS; approved."}'
```
