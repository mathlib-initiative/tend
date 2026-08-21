"""Usage aggregation helpers for tend managed agent sessions."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from tend.agent.session import Session
from tend.llm.usage import Usage
from tend.orchestrator.state import (
    AsyncOrchestratorAgentRole,
    AsyncOrchestratorWorktree,
    WorktreeState,
)

_LOGGER = logging.getLogger(__name__)
_AGENT_USAGE_ROLES: tuple[AsyncOrchestratorAgentRole, ...] = (
    AsyncOrchestratorAgentRole.WORKER,
    AsyncOrchestratorAgentRole.REVIEWER,
)


def agent_session_dir(
    root: Path,
    worktree_id: str,
    role: AsyncOrchestratorAgentRole,
) -> Path:
    """Return the managed session directory for one worktree/role pair."""

    return _absolute_path(root) / "sessions" / worktree_id / role.value


def load_agent_session_usage(
    root: Path,
    worktree_id: str,
    role: AsyncOrchestratorAgentRole,
) -> Usage | None:
    """Return usage from a managed ``tend-agent`` session, if one exists.

    The async orchestrator can launch arbitrary commands.  Only commands that
    write tend session events under ``TEND_AGENT_SESSION_DIR`` can be
    interpreted here; empty or foreign session directories are ignored.
    """

    session_dir = agent_session_dir(root, worktree_id, role)
    if not (session_dir / "events.jsonl").is_file():
        return None
    try:
        with Session.open(session_dir, writable=False) as session:
            return session.state.usage
    except Exception as exc:
        _LOGGER.warning(
            "could not read async %s session usage for worktree %s from %s: %s",
            role.value,
            worktree_id,
            session_dir,
            exc,
        )
        return None


def agent_session_is_active(
    worktree: AsyncOrchestratorWorktree,
    role: AsyncOrchestratorAgentRole,
) -> bool:
    """Return whether ``role``'s session log can still grow for this worktree.

    A worker session is being written only while the worktree is
    ``WORKER_RUNNING``; a reviewer session only while ``REVIEW``. In any other
    state the session log is immutable (until a possible re-run, which returns
    the worktree to the active state and is captured then).
    """

    if role is AsyncOrchestratorAgentRole.WORKER:
        return worktree.state is WorktreeState.WORKER_RUNNING
    if role is AsyncOrchestratorAgentRole.REVIEWER:
        return worktree.state is WorktreeState.REVIEW
    return False


def resolve_agent_session_usage(
    root: Path,
    worktree: AsyncOrchestratorWorktree,
    role: AsyncOrchestratorAgentRole,
) -> Usage | None:
    """Return ``role``'s session usage without re-replaying immutable logs.

    Uses the worktree's stored snapshot for terminal sessions; reads the live
    session log only when the session is currently active (mid-run) or has no
    snapshot yet (a not-yet-backfilled pre-existing worktree — read once; the
    caller is expected to persist the snapshot so it is not re-read again).
    """

    if not worktree.agent_session_started(role):
        return None  # the session never ran for this role — nothing to read
    stored = worktree.agent_session_usage(role)
    if stored is not None and not agent_session_is_active(worktree, role):
        return stored
    return load_agent_session_usage(root, worktree.worktree_id, role)


def aggregate_agent_session_usage(
    root: Path,
    worktrees: Iterable[AsyncOrchestratorWorktree],
) -> Usage:
    """Aggregate usage across managed worker/reviewer tend sessions.

    Reads each session's stored usage snapshot for terminal sessions and only
    replays the (few) currently-active session logs — so it never re-parses
    immutable logs, keeping cost flat as the run accumulates worktrees.
    """

    result = Usage()
    cost_discarded = False
    for worktree in sorted(worktrees, key=lambda item: item.worktree_id):
        for role in _AGENT_USAGE_ROLES:
            usage = resolve_agent_session_usage(root, worktree, role)
            if usage is None:
                continue
            if cost_discarded:
                result = _add_usage_without_cost(result, usage)
                continue
            try:
                result = result.add(usage)
            except ValueError:
                _LOGGER.warning(
                    "could not aggregate async agent session costs with different "
                    "currencies; omitting aggregate cost"
                )
                result = _add_usage_without_cost(result, usage)
                cost_discarded = True
    return result


def _add_usage_without_cost(left: Usage, right: Usage) -> Usage:
    return Usage(
        tokens=left.tokens.add(right.tokens),
        cost=None,
        model_requests=left.model_requests + right.model_requests,
        retry_attempts=left.retry_attempts + right.retry_attempts,
        tool_calls=left.tool_calls + right.tool_calls,
    )


def format_usage_summary(usage: Usage) -> str:
    """Return a compact human-readable usage summary."""

    tokens = usage.tokens
    parts = [
        f"input={tokens.input_tokens}",
        f"output={tokens.output_tokens}",
    ]
    if tokens.cache_read_tokens:
        parts.append(f"cache_read={tokens.cache_read_tokens}")
    if tokens.cache_write_tokens:
        parts.append(f"cache_write={tokens.cache_write_tokens}")
    if tokens.reasoning_tokens:
        parts.append(f"reasoning={tokens.reasoning_tokens}")
    parts.append(f"model_requests={usage.model_requests}")
    parts.append(f"retry_attempts={usage.retry_attempts}")
    parts.append(f"tool_calls={usage.tool_calls}")
    if usage.cost is not None:
        parts.append(f"cost={usage.cost.amount:.4f} {usage.cost.currency}")
    return "usage: " + ", ".join(parts)


def _absolute_path(path: Path) -> Path:
    return path.expanduser().resolve()


__all__ = (
    "agent_session_dir",
    "aggregate_agent_session_usage",
    "format_usage_summary",
    "load_agent_session_usage",
)
