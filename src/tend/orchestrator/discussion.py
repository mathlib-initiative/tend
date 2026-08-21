"""Discussion log helpers for async orchestrator worktrees."""

from __future__ import annotations

import json
from pathlib import Path

from tend._common.agent_outputs import ReviewVerdictOutput
from tend.orchestrator.state import (
    AsyncOrchestratorAgentRole,
    AsyncOrchestratorWorktree,
)

DISCUSSION_DIRECTORY_NAME = ".tend"
DISCUSSION_FILE_NAME = "discussion.md"
REVIEWS_DIRECTORY_NAME = "reviews"


def discussion_path(worktree: AsyncOrchestratorWorktree) -> Path:
    """Return the discussion log path exposed inside ``worktree``."""

    return worktree.path / DISCUSSION_DIRECTORY_NAME / DISCUSSION_FILE_NAME


def reviews_directory(worktree: AsyncOrchestratorWorktree) -> Path:
    """Return the directory holding structured review-verdict artifacts."""

    return worktree.path / DISCUSSION_DIRECTORY_NAME / REVIEWS_DIRECTORY_NAME


def write_review_verdict_artifact(
    worktree: AsyncOrchestratorWorktree,
    verdict: ReviewVerdictOutput,
    *,
    index: int,
) -> Path:
    """Persist one structured ``review_verdict`` as a JSON artifact in ``worktree``.

    The discussion log only keeps a flattened text message; this retains the full
    structured verdict (verdict/notes/feedback_text and the per-comment array) for
    audit and debugging, mirroring the shared ``REVIEW_VERDICT``
    artifact. ``index`` is the 1-based review number for this worktree.
    """

    directory = reviews_directory(worktree)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{index:03d}-review-verdict.json"
    payload = json.dumps(verdict.model_dump(mode="json"), ensure_ascii=False, indent=2)
    path.write_text(payload + "\n", encoding="utf-8")
    return path


def write_discussion_log_file(worktree: AsyncOrchestratorWorktree) -> None:
    """Write a readable discussion log into ``worktree`` for agent context."""

    path = discussion_path(worktree)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_discussion_log(worktree), encoding="utf-8")


def format_discussion_log(worktree: AsyncOrchestratorWorktree) -> str:
    """Render a worktree discussion as markdown."""

    lines = [
        "# Async Orchestrator Discussion",
        "",
        f"Worktree: `{worktree.worktree_id}`",
    ]
    if worktree.task_id is not None:
        lines.append(f"Task: `{worktree.task_id}`")
    lines.append("")
    if not worktree.discussion:
        lines.extend(("No messages yet.", ""))
        return "\n".join(lines)

    for index, entry in enumerate(worktree.discussion, start=1):
        lines.extend(
            (
                f"## {index}. {entry.role.value.title()}",
                "",
                entry.message.strip(),
                "",
            )
        )
    return "\n".join(lines)


def format_feedback_message_for_worker(
    worktree: AsyncOrchestratorWorktree,
) -> str | None:
    """Render the latest non-worker discussion message as worker feedback.

    The async orchestrator routes every worker-actionable signal through the
    discussion log (see ``_record_orchestrator_message_and_transition``):

    * reviewer ``request_changes`` verdicts (role ``reviewer``)
    * merge-into-target failures and merge races (role ``orchestrator``)
    * pre-merge validation failures, e.g. ``lake build`` (role ``orchestrator``)
    * dirty-entrypoint and entrypoint git-status failures (role ``orchestrator``)

    All of these append a markdown message to ``worktree.discussion`` and send
    the worktree back to ``PENDING`` — the worker is then re-spawned on
    resume. The worker's pending action is therefore *exactly* the latest
    discussion message **when it is not from the worker itself**. When the
    most recent message IS from the worker (it has already replied), there is
    no pending feedback and the shim falls back to the initial assignment
    prompt.

    The body of the latest message is already rendered as worker-facing
    markdown by the orchestrator/agent-runner code that produced it (see
    ``_agent_discussion_message``, ``_merge_failure_discussion_message``,
    ``_pre_merge_validation_failure_discussion_message``,
    ``_dirty_entrypoint_discussion_message``); we wrap it with a small
    header so the worker has stable context (worktree id, task id, source
    role, turn number) and return.

    Mirrors the shared ``_format_feedback_for_prompt``, which
    likewise wraps the latest ``FeedbackRecord.message`` regardless of source.
    """

    if not worktree.discussion:
        return None
    last = worktree.discussion[-1]
    if last.role is AsyncOrchestratorAgentRole.WORKER:
        return None

    lines = [
        f"# Revision feedback for `{worktree.worktree_id}`",
        "",
        f"- Worktree: `{worktree.worktree_id}`",
    ]
    if worktree.task_id is not None:
        lines.append(f"- Task: `{worktree.task_id}`")
    lines.append(f"- Source: `{last.role.value}`")
    lines.append(f"- Discussion turn: `{len(worktree.discussion)}`")
    lines.extend(("", "## Feedback", "", last.message.strip(), ""))
    return "\n".join(lines).rstrip() + "\n"
