from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from tend._common.agent_outputs import ReviewVerdictOutput
from tend.llm.usage import Cost, TokenUsage, Usage
from tend.orchestrator.state import (
    AsyncOrchestratorAgentRole,
    AsyncOrchestratorDiscussionMessage,
    AsyncOrchestratorWorktree,
    WorktreeState,
)


def test_worktree_defaults_and_agent_session_accessors() -> None:
    worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        path=Path("/tmp/worktree"),
        head="abc123",
    )

    assert worktree.state is WorktreeState.PENDING
    assert not worktree.agent_session_started(AsyncOrchestratorAgentRole.WORKER)
    assert not worktree.agent_session_started(AsyncOrchestratorAgentRole.REVIEWER)
    assert not worktree.agent_session_started(AsyncOrchestratorAgentRole.ORCHESTRATOR)
    assert worktree.agent_session_usage(AsyncOrchestratorAgentRole.WORKER) is None


def test_worktree_records_agent_session_usage_snapshot() -> None:
    usage = Usage(
        tokens=TokenUsage(input_tokens=10, output_tokens=5),
        cost=Cost(amount=Decimal("0.25"), currency="USD"),
        model_requests=1,
    )
    worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        path=Path("/tmp/worktree"),
        head="abc123",
    )

    updated = worktree.model_copy(update={"worker_session_usage": usage})

    assert updated.agent_session_usage(AsyncOrchestratorAgentRole.WORKER) == usage
    assert updated.agent_session_usage(AsyncOrchestratorAgentRole.REVIEWER) is None
    assert worktree.agent_session_usage(AsyncOrchestratorAgentRole.WORKER) is None


def test_discussion_message_rejects_blank_message() -> None:
    with pytest.raises(ValidationError, match="discussion messages must not be blank"):
        AsyncOrchestratorDiscussionMessage(
            role=AsyncOrchestratorAgentRole.WORKER,
            message="  ",
        )


def test_worktree_rejects_blank_text_fields() -> None:
    with pytest.raises(ValidationError, match="worktree text fields must not be blank"):
        AsyncOrchestratorWorktree(
            worktree_id="worktree_000001",
            path=Path("/tmp/worktree"),
            head="  ",
        )

    with pytest.raises(ValidationError, match="worktree text fields must not be blank"):
        AsyncOrchestratorWorktree(
            worktree_id="worktree_000001",
            path=Path("/tmp/worktree"),
            head="abc123",
            task_id="  ",
        )


def test_worktree_round_trips_review_verdicts() -> None:
    verdict = ReviewVerdictOutput.model_validate(
        {
            "schema_version": 1,
            "verdict": "request_changes",
            "notes": "Criterion 1 FAIL.",
            "feedback_text": "Fix it.",
            "comments": [{"message": "missing case", "severity": "error"}],
        }
    )
    worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        path=Path("/tmp/worktree"),
        head="abc123",
        state=WorktreeState.REVIEW,
        review_verdicts=(verdict,),
    )

    loaded = AsyncOrchestratorWorktree.model_validate_json(worktree.model_dump_json())

    assert loaded.review_verdicts == (verdict,)
