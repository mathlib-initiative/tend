"""Subprocess launch helpers for async orchestrator agents."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path

from tend.orchestrator.config import (
    AsyncOrchestratorAgentCommandConfig,
    AsyncOrchestratorConfig,
)
from tend.orchestrator.discussion import (
    discussion_path,
    format_feedback_message_for_worker,
)
from tend.orchestrator.state import (
    AsyncOrchestratorAgentRole,
    AsyncOrchestratorWorktree,
)
from tend.orchestrator.usage import agent_session_dir


def oom_score_adj_preexec(adj: int | None) -> Callable[[], None] | None:
    """Return a ``preexec_fn`` that sets the child's ``oom_score_adj`` (Linux only).

    The Linux OOM killer scores each process as roughly ``memory_footprint +
    oom_score_adj`` and, on memory exhaustion, kills the highest scorer. Setting
    a positive ``adj`` on a spawned process makes it (and — because
    ``oom_score_adj`` is inherited across ``fork`` — its entire descendant tree:
    the agent shim, ``tend-agent``, ``lake``, every ``lean``) a preferred victim,
    so the kernel reaps a recomputable build before ever touching the
    orchestrator or the operator's terminal (both left at the default 0).

    Returns ``None`` (i.e. no ``preexec_fn``) when ``adj`` is ``None`` or the
    platform is not Linux. The returned callable runs in the forked child before
    ``exec``; it uses only raw ``os`` syscalls (no allocation-heavy or
    lock-taking work) to stay safe in that post-fork context, and swallows any
    error so a write failure can never block the spawn. Raising ``oom_score_adj``
    for a same-uid process requires no privilege.
    """

    if adj is None or sys.platform != "linux":
        return None

    encoded = str(adj).encode()

    def _apply() -> None:
        try:
            fd = os.open("/proc/self/oom_score_adj", os.O_WRONLY)
            try:
                os.write(fd, encoded)
            finally:
                os.close(fd)
        except OSError:
            pass

    return _apply


_AGENT_DEBUG_LOG_DIRECTORY = "logs/agents"
_AGENT_START_FAILURE_EXIT_CODE = 127
_AGENT_TERMINATION_GRACE_SECONDS = 5.0
# Filename the worker shim consults inside ``$TEND_AGENT_SESSION_DIR`` on
# resume to pick a per-revision prompt over the initial assignment prompt.
# Must match the shim emitted by ``cli._tend_agent_script``.
REVISION_PROMPT_FILENAME = "revision-prompt.md"
_LOGGER = logging.getLogger(__name__)

type _AgentDebugLogPaths = tuple[Path, Path]


async def run_agent_command(
    config: AsyncOrchestratorConfig,
    command: AsyncOrchestratorAgentCommandConfig,
    *,
    worktree: AsyncOrchestratorWorktree,
    role: AsyncOrchestratorAgentRole,
    resume: bool,
) -> tuple[int, bytes]:
    """Run one configured agent command, returning stdout and writing debug logs."""

    session_dir = agent_session_dir(config.root, worktree.worktree_id, role)
    session_dir.mkdir(parents=True, exist_ok=True)
    if role is AsyncOrchestratorAgentRole.WORKER and resume:
        _materialise_worker_revision_prompt(
            config=config,
            worktree=worktree,
            session_dir=session_dir,
        )
    debug_log_paths = _next_agent_debug_log_paths(config.root, worktree.worktree_id, role)
    _LOGGER.debug(
        "starting %s agent subprocess: worktree=%s resume=%s",
        role.value,
        worktree.worktree_id,
        resume,
    )
    try:
        process = await asyncio.create_subprocess_exec(
            *command.argv_for_resume(resume),
            cwd=worktree.path,
            env=_agent_environment(config, worktree, role=role, resume=resume),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            preexec_fn=oom_score_adj_preexec(config.agent_oom_score_adj),
        )
    except OSError as exc:
        _write_agent_debug_logs(debug_log_paths, stdout=b"", stderr=b"")
        _LOGGER.warning(
            "failed to start %s agent subprocess for worktree %s: %s",
            role.value,
            worktree.worktree_id,
            exc,
        )
        return _AGENT_START_FAILURE_EXIT_CODE, b""

    try:
        stdout_bytes, stderr_bytes = await process.communicate()
    except asyncio.CancelledError:
        _LOGGER.info(
            "terminating %s agent subprocess for worktree %s after cancellation",
            role.value,
            worktree.worktree_id,
        )
        await _terminate_process_group_or_process(process)
        raise
    _write_agent_debug_logs(debug_log_paths, stdout=stdout_bytes, stderr=stderr_bytes)
    exit_code = process.returncode if process.returncode is not None else 1
    _LOGGER.debug(
        "%s agent subprocess exited: worktree=%s exit_code=%d",
        role.value,
        worktree.worktree_id,
        exit_code,
    )
    return exit_code, stdout_bytes


def _next_agent_debug_log_paths(
    root: Path,
    worktree_id: str,
    role: AsyncOrchestratorAgentRole,
) -> _AgentDebugLogPaths | None:
    """Return paths for the next best-effort raw stdout/stderr debug logs."""

    log_dir = _absolute_path(root) / _AGENT_DEBUG_LOG_DIRECTORY / worktree_id
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _LOGGER.warning(
            "could not create async agent debug log directory %s: %s",
            log_dir,
            exc,
        )
        return None

    attempt = 1
    while True:
        stdout_path = log_dir / f"{role.value}-{attempt}.stdout"
        stderr_path = log_dir / f"{role.value}-{attempt}.stderr"
        try:
            if not stdout_path.exists() and not stderr_path.exists():
                return stdout_path, stderr_path
        except OSError as exc:
            _LOGGER.warning(
                "could not inspect async agent debug log paths %s / %s: %s",
                stdout_path,
                stderr_path,
                exc,
            )
            return None
        attempt += 1


def _write_agent_debug_logs(
    paths: _AgentDebugLogPaths | None,
    *,
    stdout: bytes,
    stderr: bytes,
) -> None:
    """Write raw agent stdout/stderr diagnostics without affecting agent routing."""

    if paths is None:
        return
    stdout_path, stderr_path = paths
    try:
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
    except OSError as exc:
        _LOGGER.warning(
            "could not write async agent debug logs %s / %s: %s",
            stdout_path,
            stderr_path,
            exc,
        )


def _worker_revision_template_path(config: AsyncOrchestratorConfig) -> Path:
    """Return the path of the on-disk worker revision template materialised at init.

    The template lives at ``<root>/prompts/worker-revision.md``, written once by
    ``tend init`` from the bundled prompt registry. Returning the run-root
    path (rather than the launch-time code snapshot at
    ``<root>/code/src/tend/prompts/...``) keeps the read consistent with how
    the worker shim reads ``<root>/prompts/worker.md`` today.
    """

    return _absolute_path(config.root) / "prompts" / "worker-revision.md"


def _materialise_worker_revision_prompt(
    *,
    config: AsyncOrchestratorConfig,
    worktree: AsyncOrchestratorWorktree,
    session_dir: Path,
) -> None:
    """Render ``revision-prompt.md`` into ``session_dir`` for the worker shim.

    The worker shim (emitted by ``cli._tend_agent_script``) consults
    ``$TEND_AGENT_SESSION_DIR/revision-prompt.md`` ahead of the initial
    assignment prompt on ``--resume``. We write that file here, substituting
    ``{feedback_message}`` in the on-disk revision template with the rendered
    latest non-worker discussion message (covering reviewer ``request_changes``
    and the four orchestrator-injected feedback paths in
    ``_record_orchestrator_message_and_transition``: merge failure, post-merge
    validation failure, dirty entrypoint, entrypoint status-check failure).
    Best-effort: if the template is missing (older ``tend init`` predating
    this feature) or there is no pending feedback (initial assignment or the
    worker has already replied to prior feedback), we delete any stale prompt
    and the shim falls back to the initial prompt.
    """

    revision_prompt_path = session_dir / REVISION_PROMPT_FILENAME
    feedback = format_feedback_message_for_worker(worktree)
    if feedback is None:
        # No pending feedback to convey; clear any stale prompt from a prior
        # iteration so the shim falls back to the initial assignment prompt.
        try:
            revision_prompt_path.unlink(missing_ok=True)
        except OSError as exc:
            _LOGGER.warning(
                "could not remove stale revision prompt %s: %s",
                revision_prompt_path,
                exc,
            )
        return

    template_path = _worker_revision_template_path(config)
    try:
        template = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        _LOGGER.warning(
            "worker revision template missing or unreadable at %s (%s); "
            "skipping revision-prompt materialisation; worker shim falls back to "
            "the initial assignment prompt",
            template_path,
            exc,
        )
        try:
            revision_prompt_path.unlink(missing_ok=True)
        except OSError:
            pass
        return

    rendered = template.replace("{feedback_message}", feedback)
    try:
        revision_prompt_path.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        _LOGGER.warning(
            "could not write revision prompt %s for worktree %s: %s",
            revision_prompt_path,
            worktree.worktree_id,
            exc,
        )


def _agent_environment(
    config: AsyncOrchestratorConfig,
    worktree: AsyncOrchestratorWorktree,
    *,
    role: AsyncOrchestratorAgentRole,
    resume: bool,
) -> dict[str, str]:
    """Build environment variables shared by worker and reviewer agents."""

    environment = dict(os.environ)
    environment.update(
        {
            "TEND_AGENT_DISCUSSION_PATH": str(discussion_path(worktree)),
            "TEND_AGENT_RESUME": "1" if resume else "0",
            "TEND_AGENT_ROLE": role.value,
            "TEND_AGENT_SESSION_DIR": str(
                agent_session_dir(config.root, worktree.worktree_id, role)
            ),
            "TEND_ENTRYPOINT": str(_absolute_path(config.entrypoint)),
            "TEND_ROOT": str(_absolute_path(config.root)),
            "TEND_WORKTREE_HEAD": worktree.head,
            "TEND_WORKTREE_ID": worktree.worktree_id,
            "TEND_WORKTREE_PATH": str(worktree.path),
        }
    )
    if worktree.task_id is not None:
        environment["TEND_TASK_ID"] = worktree.task_id
    return environment


async def _terminate_process_group_or_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    _signal_process_group_or_process(process, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=_AGENT_TERMINATION_GRACE_SECONDS)
    except TimeoutError:
        _kill_process_group_or_process(process)
        await process.wait()


def _signal_process_group_or_process(
    process: asyncio.subprocess.Process,
    signum: int,
) -> None:
    if process.returncode is not None:
        return
    if os.name != "nt":
        try:
            os.killpg(process.pid, signum)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    try:
        process.send_signal(signum)
    except ProcessLookupError:
        return


def _kill_process_group_or_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    try:
        process.kill()
    except ProcessLookupError:
        return


def _absolute_path(path: Path) -> Path:
    return path.expanduser().resolve()
