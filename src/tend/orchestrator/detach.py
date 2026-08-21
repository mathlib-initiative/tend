"""Shared detached-process launcher for orchestrator commands."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path


def spawn_detached(
    argv: Sequence[str],
    *,
    log_file: Path,
    pid_file: Path,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """Spawn ``argv`` as a detached background process and return its PID.

    The child reads stdin from ``/dev/null``, appends stdout and stderr to
    ``log_file``, starts in its own session/process group, and has its PID
    recorded in ``pid_file``. Parent directories for both files are created.
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("ab") as log_fd:
        process = subprocess.Popen(
            argv,
            stdout=log_fd,
            stderr=log_fd,
            stdin=subprocess.DEVNULL,
            cwd=cwd,
            env=None if env is None else dict(env),
            start_new_session=True,
            close_fds=True,
        )
    pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
    return process.pid
