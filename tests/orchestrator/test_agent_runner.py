from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tend.orchestrator.agent_runner import oom_score_adj_preexec, run_agent_command
from tend.orchestrator.config import (
    AsyncOrchestratorAgentCommandConfig,
    AsyncOrchestratorConfig,
)
from tend.orchestrator.state import AsyncOrchestratorAgentRole, AsyncOrchestratorWorktree


async def test_agent_runner_writes_stdout_and_stderr_debug_logs(tmp_path: Path) -> None:
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")

    root = tmp_path / "root"
    entrypoint = tmp_path / "entrypoint"
    worktree_path = tmp_path / "worktree"
    entrypoint.mkdir()
    worktree_path.mkdir()
    config = AsyncOrchestratorConfig(root=root, entrypoint=entrypoint)
    command = AsyncOrchestratorAgentCommandConfig(
        argv=(
            "sh",
            "-c",
            'printf "%s" "$1"; printf "%s" "$2" >&2',
            "agent",
            '{"message":"done"}',
            "debug details",
        ),
    )
    worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        path=worktree_path,
        head="abc123",
    )

    exit_code, stdout = await run_agent_command(
        config,
        command,
        worktree=worktree,
        role=AsyncOrchestratorAgentRole.WORKER,
        resume=False,
    )

    assert exit_code == 0
    assert stdout == b'{"message":"done"}'
    log_dir = root / "logs" / "agents" / "worktree_000001"
    assert (log_dir / "worker-1.stdout").read_bytes() == b'{"message":"done"}'
    assert (log_dir / "worker-1.stderr").read_bytes() == b"debug details"

    second_command = AsyncOrchestratorAgentCommandConfig(
        argv=(
            "sh",
            "-c",
            'printf "%s" "$1"; printf "%s" "$2" >&2',
            "agent",
            '{"message":"again"}',
            "more debug",
        ),
    )

    await run_agent_command(
        config,
        second_command,
        worktree=worktree,
        role=AsyncOrchestratorAgentRole.WORKER,
        resume=True,
    )

    assert (log_dir / "worker-2.stdout").read_bytes() == b'{"message":"again"}'
    assert (log_dir / "worker-2.stderr").read_bytes() == b"more debug"


async def test_agent_runner_terminates_process_group_on_cancellation(tmp_path: Path) -> None:
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")

    root = tmp_path / "root"
    entrypoint = tmp_path / "entrypoint"
    worktree_path = tmp_path / "worktree"
    entrypoint.mkdir()
    worktree_path.mkdir()
    command = (
        "printf started > started; "
        "trap 'printf terminated > terminated; exit 0' TERM; "
        "while :; do sleep 1; done"
    )
    task = asyncio.create_task(
        run_agent_command(
            AsyncOrchestratorConfig(
                root=root,
                entrypoint=entrypoint,
                worker_agent_command=AsyncOrchestratorAgentCommandConfig(
                    argv=("sh", "-c", command),
                ),
            ),
            AsyncOrchestratorAgentCommandConfig(argv=("sh", "-c", command)),
            worktree=AsyncOrchestratorWorktree(
                worktree_id="worktree_000001",
                path=worktree_path,
                head="abc123",
            ),
            role=AsyncOrchestratorAgentRole.WORKER,
            resume=False,
        )
    )
    await _wait_for_path(worktree_path / "started")

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    await _wait_for_path(worktree_path / "terminated")


async def _wait_for_path(path: Path) -> None:
    for _ in range(50):
        if path.exists():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for path: {path}")


async def test_agent_runner_materialises_revision_prompt_on_worker_resume(
    tmp_path: Path,
) -> None:
    """Worker resume with pending feedback materialises revision-prompt.md.

    Regression test for the v4/v5 prompt-port livelock: the worker's
    ``--resume`` invocation must see a prompt that includes the latest pending
    feedback (reviewer ``request_changes`` in this test; the four
    orchestrator-injected paths are exercised in test_discussion.py), not the
    same initial assignment prompt that drove the rejected contribution. See
    ``REVISION_PROMPT_FILENAME`` and the shim selection logic in
    ``cli._tend_agent_script``.
    """
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")

    from tend.orchestrator.agent_runner import REVISION_PROMPT_FILENAME
    from tend.orchestrator.state import AsyncOrchestratorDiscussionMessage
    from tend.orchestrator.usage import agent_session_dir

    root = tmp_path / "root"
    entrypoint = tmp_path / "entrypoint"
    worktree_path = tmp_path / "worktree"
    entrypoint.mkdir()
    worktree_path.mkdir()

    # The agent runner reads ``<root>/prompts/worker-revision.md`` as the
    # template; ``tend init`` writes it normally. Use a minimal stand-in
    # here so the test is independent of init's prompt content.
    prompts_dir = root / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "worker-revision.md").write_text(
        "Revision turn.\nPrevious feedback:\n{feedback_message}\nApply it.\n",
        encoding="utf-8",
    )

    config = AsyncOrchestratorConfig(root=root, entrypoint=entrypoint)
    command = AsyncOrchestratorAgentCommandConfig(argv=("sh", "-c", "exit 0"))
    worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000007",
        path=worktree_path,
        head="abc123",
        task_id="task-001-1-3-1",
        discussion=(
            AsyncOrchestratorDiscussionMessage(
                role=AsyncOrchestratorAgentRole.WORKER,
                message="first contribution: decomposed the task",
            ),
            AsyncOrchestratorDiscussionMessage(
                role=AsyncOrchestratorAgentRole.REVIEWER,
                message=(
                    "criterion 4 FAIL on sorry-hygiene\n\n"
                    "Re-point the placeholder at the open assembly child: "
                    "change line 44 to `sorry -- proof: task-001-1-3-1-4`."
                ),
            ),
        ),
    )

    await run_agent_command(
        config,
        command,
        worktree=worktree,
        role=AsyncOrchestratorAgentRole.WORKER,
        resume=True,
    )

    revision_prompt = (
        agent_session_dir(root, worktree.worktree_id, AsyncOrchestratorAgentRole.WORKER)
        / REVISION_PROMPT_FILENAME
    )
    assert revision_prompt.is_file(), "shim-target revision prompt should be materialised"
    rendered = revision_prompt.read_text(encoding="utf-8")
    # The substantive feedback_text appears in the rendered prompt.
    assert "task-001-1-3-1-4" in rendered
    # The placeholder was substituted (no literal ``{feedback_message}`` remains).
    assert "{feedback_message}" not in rendered
    # The wrapper text from the template is preserved.
    assert "Apply it." in rendered


async def test_agent_runner_clears_revision_prompt_when_worker_has_already_replied(
    tmp_path: Path,
) -> None:
    """If the worker is the latest discussion turn, the shim falls back to the initial prompt.

    The worker's own message at the tail signals "no pending feedback" — the
    worker already addressed any prior reviewer/orchestrator turn. A stale
    ``revision-prompt.md`` from a prior iteration must therefore be cleared so
    the worker doesn't pick up obsolete feedback the next time the shim
    launches.
    """
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")

    from tend.orchestrator.agent_runner import REVISION_PROMPT_FILENAME
    from tend.orchestrator.usage import agent_session_dir

    root = tmp_path / "root"
    entrypoint = tmp_path / "entrypoint"
    worktree_path = tmp_path / "worktree"
    entrypoint.mkdir()
    worktree_path.mkdir()
    (root / "prompts").mkdir(parents=True)
    (root / "prompts" / "worker-revision.md").write_text(
        "{feedback_message}", encoding="utf-8",
    )

    config = AsyncOrchestratorConfig(root=root, entrypoint=entrypoint)
    command = AsyncOrchestratorAgentCommandConfig(argv=("sh", "-c", "exit 0"))
    session_dir = agent_session_dir(
        root, "worktree_000008", AsyncOrchestratorAgentRole.WORKER,
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    stale = session_dir / REVISION_PROMPT_FILENAME
    stale.write_text("# stale feedback from a prior iteration\n", encoding="utf-8")

    # Worker's own message is the latest → no pending feedback → stale prompt
    # must be cleared.
    from tend.orchestrator.state import AsyncOrchestratorDiscussionMessage

    worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000008",
        path=worktree_path,
        head="abc123",
        task_id="task-001",
        discussion=(
            AsyncOrchestratorDiscussionMessage(
                role=AsyncOrchestratorAgentRole.WORKER,
                message="worker already addressed all prior feedback",
            ),
        ),
    )

    await run_agent_command(
        config,
        command,
        worktree=worktree,
        role=AsyncOrchestratorAgentRole.WORKER,
        resume=True,
    )

    assert not stale.exists(), "stale revision prompt must be cleared when no feedback is pending"


async def test_agent_runner_does_not_touch_revision_prompt_on_non_resume_invocation(
    tmp_path: Path,
) -> None:
    """An initial spawn (``resume=False``) must not write or clear ``revision-prompt.md``.

    The runner only manages the session-dir revision prompt during ``resume=True``.
    If a stale file somehow predates the first spawn (e.g. dirty session dir
    reuse), the shim — separately — gates its selection on actually being a
    resume, so a non-resume invocation reads only the initial prompt.
    """
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")

    from tend.orchestrator.agent_runner import REVISION_PROMPT_FILENAME
    from tend.orchestrator.usage import agent_session_dir

    root = tmp_path / "root"
    entrypoint = tmp_path / "entrypoint"
    worktree_path = tmp_path / "worktree"
    entrypoint.mkdir()
    worktree_path.mkdir()
    (root / "prompts").mkdir(parents=True)
    (root / "prompts" / "worker-revision.md").write_text(
        "{feedback_message}", encoding="utf-8",
    )

    config = AsyncOrchestratorConfig(root=root, entrypoint=entrypoint)
    command = AsyncOrchestratorAgentCommandConfig(argv=("sh", "-c", "exit 0"))
    session_dir = agent_session_dir(
        root, "worktree_000009", AsyncOrchestratorAgentRole.WORKER,
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    sentinel_path = session_dir / REVISION_PROMPT_FILENAME
    sentinel_path.write_text("# pre-existing sentinel\n", encoding="utf-8")

    worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000009",
        path=worktree_path,
        head="abc123",
        task_id="task-001",
    )

    await run_agent_command(
        config,
        command,
        worktree=worktree,
        role=AsyncOrchestratorAgentRole.WORKER,
        resume=False,
    )

    # Sentinel file must be untouched: resume=False short-circuits before the
    # materialise step entirely.
    assert sentinel_path.is_file()
    assert sentinel_path.read_text(encoding="utf-8") == "# pre-existing sentinel\n"


async def test_agent_runner_does_not_materialise_revision_prompt_for_reviewer_role(
    tmp_path: Path,
) -> None:
    """Reviewer resumes must not write ``revision-prompt.md`` — only workers receive it.

    The reviewer agent has no concept of a revision prompt (the reviewer shim
    doesn't even read one); the runner's materialiser gates on
    ``role is WORKER`` so a reviewer-role resume can't accidentally clobber
    or fabricate a worker-side prompt file in the reviewer's session dir.
    """
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")

    from tend.orchestrator.agent_runner import REVISION_PROMPT_FILENAME
    from tend.orchestrator.state import AsyncOrchestratorDiscussionMessage
    from tend.orchestrator.usage import agent_session_dir

    root = tmp_path / "root"
    entrypoint = tmp_path / "entrypoint"
    worktree_path = tmp_path / "worktree"
    entrypoint.mkdir()
    worktree_path.mkdir()
    (root / "prompts").mkdir(parents=True)
    (root / "prompts" / "worker-revision.md").write_text(
        "{feedback_message}", encoding="utf-8",
    )

    config = AsyncOrchestratorConfig(root=root, entrypoint=entrypoint)
    command = AsyncOrchestratorAgentCommandConfig(argv=("sh", "-c", "exit 0"))
    worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000010",
        path=worktree_path,
        head="abc123",
        task_id="task-001",
        discussion=(
            AsyncOrchestratorDiscussionMessage(
                role=AsyncOrchestratorAgentRole.REVIEWER,
                message="fix the dangling sorry placeholder",
            ),
        ),
    )

    await run_agent_command(
        config,
        command,
        worktree=worktree,
        role=AsyncOrchestratorAgentRole.REVIEWER,
        resume=True,
    )

    reviewer_revision_path = (
        agent_session_dir(root, worktree.worktree_id, AsyncOrchestratorAgentRole.REVIEWER)
        / REVISION_PROMPT_FILENAME
    )
    assert not reviewer_revision_path.exists(), (
        "reviewer-role resume must not produce a revision-prompt.md"
    )


def test_oom_score_adj_preexec_none_disables() -> None:
    """``None`` (the opt-out) yields no ``preexec_fn``."""

    assert oom_score_adj_preexec(None) is None


def test_oom_score_adj_preexec_noop_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-Linux platforms get no ``preexec_fn`` (the /proc interface is Linux-only)."""

    monkeypatch.setattr("tend.orchestrator.agent_runner.sys.platform", "darwin")
    assert oom_score_adj_preexec(750) is None


def test_oom_score_adj_preexec_sets_child_value() -> None:
    """The preexec sets the spawned child's oom_score_adj to the configured value."""

    if sys.platform != "linux":
        pytest.skip("oom_score_adj is Linux-only")
    if shutil.which("cat") is None:
        pytest.skip("cat executable is required")
    completed = subprocess.run(
        ["cat", "/proc/self/oom_score_adj"],
        capture_output=True,
        text=True,
        preexec_fn=oom_score_adj_preexec(742),
        check=True,
    )
    assert completed.stdout.strip() == "742"


def test_oom_score_adj_preexec_inherited_by_descendants() -> None:
    """A grandchild inherits the value — the property that covers lake/lean builds."""

    if sys.platform != "linux":
        pytest.skip("oom_score_adj is Linux-only")
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")
    # The preexec is set on the `sh` we spawn; the inner `cat` is a *grandchild*
    # (sh -> sh -c cat), so a matching value proves fork-inheritance down the tree.
    completed = subprocess.run(
        ["sh", "-c", "exec sh -c 'cat /proc/self/oom_score_adj'"],
        capture_output=True,
        text=True,
        preexec_fn=oom_score_adj_preexec(613),
        check=True,
    )
    assert completed.stdout.strip() == "613"
