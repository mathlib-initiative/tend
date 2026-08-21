"""Tests for per-session usage snapshots on the worktree.

Usage aggregation must use the worktree's stored per-role usage snapshot for
terminal sessions and only read the (few) currently-active session logs — so it
never re-replays immutable logs as the run accumulates worktrees.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

import tend.orchestrator.usage as usage_module
from tend.llm.usage import Cost, TokenUsage, Usage
from tend.orchestrator.state import (
    AsyncOrchestratorAgentRole,
    AsyncOrchestratorWorktree,
    WorktreeState,
)
from tend.orchestrator.usage import (
    agent_session_is_active,
    aggregate_agent_session_usage,
    resolve_agent_session_usage,
)

WORKER = AsyncOrchestratorAgentRole.WORKER
REVIEWER = AsyncOrchestratorAgentRole.REVIEWER


def _usage(cost: str) -> Usage:
    return Usage(
        tokens=TokenUsage(input_tokens=10, output_tokens=5),
        cost=Cost(amount=Decimal(cost), currency="USD"),
        model_requests=1,
    )


def _worktree(
    wid: str,
    state: WorktreeState,
    *,
    worker_session_started: bool = False,
    reviewer_session_started: bool = False,
    worker_session_usage: Usage | None = None,
    reviewer_session_usage: Usage | None = None,
) -> AsyncOrchestratorWorktree:
    return AsyncOrchestratorWorktree(
        worktree_id=wid,
        path=Path("/tmp") / wid,
        head="abc",
        state=state,
        worker_session_started=worker_session_started,
        reviewer_session_started=reviewer_session_started,
        worker_session_usage=worker_session_usage,
        reviewer_session_usage=reviewer_session_usage,
    )


def test_agent_session_is_active_maps_role_to_state() -> None:
    running = _worktree("worktree_000001", WorktreeState.WORKER_RUNNING)
    review = _worktree("worktree_000002", WorktreeState.REVIEW)
    closed = _worktree("worktree_000003", WorktreeState.CLOSED)
    assert agent_session_is_active(running, WORKER) is True
    assert agent_session_is_active(running, REVIEWER) is False
    assert agent_session_is_active(review, REVIEWER) is True
    assert agent_session_is_active(review, WORKER) is False
    assert agent_session_is_active(closed, WORKER) is False
    assert agent_session_is_active(closed, REVIEWER) is False


def test_model_stores_and_returns_per_role_usage() -> None:
    wt = _worktree("worktree_000001", WorktreeState.CLOSED)
    assert wt.agent_session_usage(WORKER) is None
    wt2 = wt.model_copy(update={"worker_session_usage": _usage("1.00")})
    assert wt2.agent_session_usage(WORKER) == _usage("1.00")
    assert wt2.agent_session_usage(REVIEWER) is None  # untouched


def test_resolve_uses_stored_snapshot_for_terminal_without_reading_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = 0

    def spy(root: Path, worktree_id: str, role: AsyncOrchestratorAgentRole) -> Usage | None:
        nonlocal reads
        reads += 1
        return _usage("999")  # would be wrong if used

    monkeypatch.setattr(usage_module, "load_agent_session_usage", spy)
    wt = _worktree(
        "worktree_000001",
        WorktreeState.MERGE,
        worker_session_started=True,
        worker_session_usage=_usage("2.50"),
    )

    got = resolve_agent_session_usage(Path("/root"), wt, WORKER)

    assert got == _usage("2.50")  # stored, not the spy's value
    assert reads == 0  # terminal + stored => no log read


def test_resolve_skips_never_started_session(monkeypatch: pytest.MonkeyPatch) -> None:
    reads = 0

    def spy(root: Path, worktree_id: str, role: AsyncOrchestratorAgentRole) -> Usage | None:
        nonlocal reads
        reads += 1
        return _usage("1.00")

    monkeypatch.setattr(usage_module, "load_agent_session_usage", spy)
    wt = _worktree("worktree_000001", WorktreeState.CLOSED)  # reviewer never ran

    assert resolve_agent_session_usage(Path("/root"), wt, REVIEWER) is None
    assert reads == 0  # never started => not read


def test_resolve_reads_log_for_active_session(monkeypatch: pytest.MonkeyPatch) -> None:
    reads = 0

    def spy(root: Path, worktree_id: str, role: AsyncOrchestratorAgentRole) -> Usage | None:
        nonlocal reads
        reads += 1
        return _usage("3.00")

    monkeypatch.setattr(usage_module, "load_agent_session_usage", spy)
    # Active session: even with a (stale) stored snapshot, read live.
    wt = _worktree(
        "worktree_000001",
        WorktreeState.WORKER_RUNNING,
        worker_session_started=True,
        worker_session_usage=_usage("1.00"),
    )

    got = resolve_agent_session_usage(Path("/root"), wt, WORKER)

    assert got == _usage("3.00")  # live value
    assert reads == 1


def test_aggregate_sums_stored_snapshots_with_zero_log_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = 0

    def spy(root: Path, worktree_id: str, role: AsyncOrchestratorAgentRole) -> Usage | None:
        nonlocal reads
        reads += 1
        return None

    monkeypatch.setattr(usage_module, "load_agent_session_usage", spy)
    worktrees = [
        _worktree(
            "worktree_000001",
            WorktreeState.CLOSED,
            worker_session_started=True,
            reviewer_session_started=True,
            worker_session_usage=_usage("1.00"),
            reviewer_session_usage=_usage("0.50"),
        ),
        _worktree(
            "worktree_000002",
            WorktreeState.MERGE,
            worker_session_started=True,
            worker_session_usage=_usage("2.00"),
        ),
    ]

    total = aggregate_agent_session_usage(Path("/root"), worktrees)

    assert total.cost is not None
    assert total.cost.amount == Decimal("3.50")  # 1.00 + 0.50 + 2.00
    assert reads == 0  # all terminal + stored (or never-started) => no log touched


def test_aggregate_reads_only_active_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    read_ids: list[str] = []

    def spy(root: Path, worktree_id: str, role: AsyncOrchestratorAgentRole) -> Usage | None:
        read_ids.append(f"{worktree_id}:{role.value}")
        return _usage("5.00")

    monkeypatch.setattr(usage_module, "load_agent_session_usage", spy)
    worktrees = [
        _worktree(
            "worktree_000001",
            WorktreeState.CLOSED,
            worker_session_started=True,
            worker_session_usage=_usage("1.00"),
        ),
        _worktree(
            "worktree_000002",
            WorktreeState.WORKER_RUNNING,
            worker_session_started=True,
        ),  # active, no snapshot yet
    ]

    total = aggregate_agent_session_usage(Path("/root"), worktrees)

    # Only the active session's log was read; the closed one used its snapshot,
    # and never-started reviewer sessions were skipped entirely.
    assert read_ids == ["worktree_000002:worker"]
    assert total.cost is not None
    assert total.cost.amount == Decimal("6.00")  # 1.00 stored + 5.00 live
