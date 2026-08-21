"""State value models for the async orchestrator."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator

from tend._common.agent_outputs import ReviewVerdictOutput
from tend._common.types import StrictModel
from tend.llm.usage import Usage

WorktreeId = Annotated[str, Field(pattern=r"^worktree_[0-9]{6,}$")]
TaskId = Annotated[str, Field(min_length=1)]


class WorktreeState(StrEnum):
    """Lifecycle state for an async orchestrator worktree."""

    PENDING = "pending"
    WORKER_RUNNING = "worker_running"
    REVIEW = "review"
    MERGE = "merge"
    CLOSED = "closed"


class AsyncOrchestratorAgentRole(StrEnum):
    """Roles that can contribute to a worktree discussion."""

    WORKER = "worker"
    REVIEWER = "reviewer"
    ORCHESTRATOR = "orchestrator"


class AsyncOrchestratorDiscussionMessage(StrictModel):
    """One worker/reviewer message in a worktree discussion log."""

    role: AsyncOrchestratorAgentRole
    message: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def _validate_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("discussion messages must not be blank")
        return value


class AsyncOrchestratorWorktree(StrictModel):
    """A git worktree created for an async orchestration run."""

    worktree_id: WorktreeId
    path: Path
    head: str = Field(min_length=1)
    task_id: TaskId | None = None
    state: WorktreeState = WorktreeState.PENDING
    discussion: tuple[AsyncOrchestratorDiscussionMessage, ...] = ()
    review_verdicts: tuple[ReviewVerdictOutput, ...] = ()
    worker_session_started: bool = False
    reviewer_session_started: bool = False
    # Snapshot of each role's session usage, captured when that agent stops.
    # ``None`` means "not captured yet" — either the session never ran, or it is
    # currently running (live value must be read from the session log), or it is
    # a pre-existing worktree awaiting a one-time backfill. Aggregation uses this
    # stored value for terminal sessions and only reads the log for sessions
    # whose role is currently active, so it never re-replays immutable logs.
    worker_session_usage: Usage | None = None
    reviewer_session_usage: Usage | None = None

    @field_validator("head", "task_id")
    @classmethod
    def _validate_non_blank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("worktree text fields must not be blank")
        return value

    def agent_session_started(self, role: AsyncOrchestratorAgentRole) -> bool:
        """Return whether ``role`` has started a session in this worktree."""

        if role is AsyncOrchestratorAgentRole.WORKER:
            return self.worker_session_started
        if role is AsyncOrchestratorAgentRole.REVIEWER:
            return self.reviewer_session_started
        if role is AsyncOrchestratorAgentRole.ORCHESTRATOR:
            return False
        raise ValueError(f"unknown async orchestrator agent role: {role}")

    def agent_session_usage(self, role: AsyncOrchestratorAgentRole) -> Usage | None:
        """Return the stored usage snapshot for ``role``'s session, if captured."""

        if role is AsyncOrchestratorAgentRole.WORKER:
            return self.worker_session_usage
        if role is AsyncOrchestratorAgentRole.REVIEWER:
            return self.reviewer_session_usage
        return None


__all__ = (
    "AsyncOrchestratorAgentRole",
    "AsyncOrchestratorDiscussionMessage",
    "AsyncOrchestratorWorktree",
    "TaskId",
    "WorktreeId",
    "WorktreeState",
)
