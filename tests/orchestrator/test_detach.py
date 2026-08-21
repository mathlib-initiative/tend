from __future__ import annotations

import os
import signal
import time
from pathlib import Path

from tend.orchestrator.detach import spawn_detached


def test_spawn_detached_records_pid_and_appends_log(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "run.log"
    pid_file = tmp_path / "pids" / "run.pid"

    pid = spawn_detached(
        ["sh", "-c", "printf detached-helper-ok"],
        log_file=log_file,
        pid_file=pid_file,
    )

    assert int(pid_file.read_text(encoding="utf-8").strip()) == pid
    status: int | None = None
    deadline = time.monotonic() + 2.0
    try:
        while time.monotonic() < deadline:
            waited_pid, wait_status = os.waitpid(pid, os.WNOHANG)
            if waited_pid == pid:
                status = wait_status
                break
            time.sleep(0.01)
    finally:
        if status is None:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            else:
                os.waitpid(pid, 0)

    assert status is not None
    assert os.waitstatus_to_exitcode(status) == 0
    assert log_file.read_text(encoding="utf-8") == "detached-helper-ok"
