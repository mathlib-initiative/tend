"""Tests for the discussion / feedback formatter helpers."""

from __future__ import annotations

from pathlib import Path

from tend.orchestrator.discussion import format_feedback_message_for_worker
from tend.orchestrator.state import (
    AsyncOrchestratorAgentRole,
    AsyncOrchestratorDiscussionMessage,
    AsyncOrchestratorWorktree,
)


def _worktree(
    *,
    discussion: tuple[AsyncOrchestratorDiscussionMessage, ...] = (),
) -> AsyncOrchestratorWorktree:
    return AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        path=Path("/tmp/wt"),
        head="abc123",
        task_id="task-001",
        discussion=discussion,
    )


def _msg(role: AsyncOrchestratorAgentRole, message: str) -> AsyncOrchestratorDiscussionMessage:
    return AsyncOrchestratorDiscussionMessage(role=role, message=message)


def test_format_feedback_message_returns_none_when_no_discussion() -> None:
    """Initial assignment: nothing to feed back."""
    assert format_feedback_message_for_worker(_worktree()) is None


def test_format_feedback_message_returns_none_when_latest_is_worker() -> None:
    """If the worker just spoke, there's no pending feedback for them to address.

    This is the normal post-contribution / pre-review state: the worker emitted
    its summary, the reviewer hasn't yet responded. We must not surface the
    worker's own message back to itself as "feedback".
    """
    worktree = _worktree(
        discussion=(
            _msg(AsyncOrchestratorAgentRole.REVIEWER, "request changes"),
            _msg(AsyncOrchestratorAgentRole.WORKER, "addressed the feedback"),
        ),
    )
    assert format_feedback_message_for_worker(worktree) is None


def test_format_feedback_message_renders_reviewer_request_changes() -> None:
    """Reviewer rejection — the canonical case the PR fixes."""
    worktree = _worktree(
        discussion=(
            _msg(AsyncOrchestratorAgentRole.WORKER, "first contribution"),
            _msg(
                AsyncOrchestratorAgentRole.REVIEWER,
                "Criterion 4 FAIL\n\n"
                "Change line 44 to `sorry -- proof: task-001-1-3-1-4`.",
            ),
        ),
    )

    rendered = format_feedback_message_for_worker(worktree)

    assert rendered is not None
    assert "task-001-1-3-1-4" in rendered
    assert "Criterion 4 FAIL" in rendered
    assert "worktree_000001" in rendered
    assert "task-001" in rendered
    assert "reviewer" in rendered  # source role surfaced in the header


def test_format_feedback_message_renders_orchestrator_merge_failure() -> None:
    """A merge-into-target failure message must reach the worker.

    All four orchestrator-emitted feedback sources (merge fail, validation
    fail, dirty entrypoint, status-check fail) come through the same
    discussion-log mechanism with ``role=orchestrator``; verifying merge-fail
    is sufficient to confirm the source-agnostic formatter handles them.
    """
    worktree = _worktree(
        discussion=(
            _msg(AsyncOrchestratorAgentRole.WORKER, "first contribution"),
            _msg(AsyncOrchestratorAgentRole.REVIEWER, "approve"),
            _msg(
                AsyncOrchestratorAgentRole.ORCHESTRATOR,
                "Merge into `main` failed — typically a merge race: another "
                "worker's contribution merged into `main` while this worktree "
                "was being worked on...\n\n"
                "**Your committed work and session context are preserved — do "
                "not redo the task.** Merge `main` into this branch yourself "
                "(`git merge main` or `git rebase main`), resolve any conflict "
                "markers, commit, and then call `final_result` to signal the "
                "contribution is ready again.",
            ),
        ),
    )

    rendered = format_feedback_message_for_worker(worktree)

    assert rendered is not None
    assert "Merge into `main` failed" in rendered
    assert "git merge main" in rendered
    assert "do not redo the task" in rendered
    # The header surfaces the orchestrator as the source.
    assert "orchestrator" in rendered


def test_format_feedback_message_uses_latest_message_only() -> None:
    """The latest message is the active feedback; older turns are session history."""
    worktree = _worktree(
        discussion=(
            _msg(AsyncOrchestratorAgentRole.WORKER, "first attempt"),
            _msg(AsyncOrchestratorAgentRole.REVIEWER, "OLD_ASK should not appear"),
            _msg(AsyncOrchestratorAgentRole.WORKER, "addressed it"),
            _msg(AsyncOrchestratorAgentRole.REVIEWER, "LATEST_ASK is the active request"),
        ),
    )

    rendered = format_feedback_message_for_worker(worktree)

    assert rendered is not None
    assert "LATEST_ASK" in rendered
    assert "OLD_ASK" not in rendered


def test_format_feedback_message_header_includes_provenance() -> None:
    """The header carries worktree id, task id, source role, and turn number.

    These give the worker stable situational awareness so it knows what
    contribution is being revised, which task it owns, who sent the
    feedback, and where the message sits in the conversation history.
    """
    worktree = _worktree(
        discussion=(
            _msg(AsyncOrchestratorAgentRole.WORKER, "x"),
            _msg(AsyncOrchestratorAgentRole.REVIEWER, "y"),
        ),
    )
    rendered = format_feedback_message_for_worker(worktree)
    assert rendered is not None
    assert "Worktree: `worktree_000001`" in rendered
    assert "Task: `task-001`" in rendered
    assert "Source: `reviewer`" in rendered
    assert "Discussion turn: `2`" in rendered
