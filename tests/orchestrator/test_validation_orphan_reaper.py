"""Tests for the orphaned-validation-process reaper (#132/#146).

The group SIGTERM/SIGKILL sent when a validation command is killed can miss
workers forked concurrently with signal delivery; the reaper sweeps ``/proc``
for such survivors afterwards. Ownership is the sole kill criterion: because
validations are launched with ``start_new_session=True``, descendants initially
inherit both ``pgid`` and ``sid`` from the leader. A descendant may change its
process group while retaining the session, and either identifier still matching
proves ownership — executable names and argv are captured for log enrichment
only. Same-checkout operator builds (different session) therefore survive no
matter how similar they look, and group members are reaped no matter what
they are called or what their argv spells.

The scanning/identity/signalling logic is unit-tested against a fabricated
proc-like directory tree (the proc root is dependency-injected); the pidfd
delivery path is unit-tested against real disposable child processes. The
integration-style tests spawn real process trees: a different-session
bystander that must survive, group orphans of a dead leader that must die
(including one that forks a replacement when SIGTERMed), and an end-to-end
run through the real validation timeout path whose orphan leaves the leader's
process group so that only the reaper (via the session-id arm of the gate)
can possibly kill it.
"""

from __future__ import annotations

import asyncio
import errno
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import pytest

import tend.orchestrator.orchestrator as orchestrator_module
from tend.orchestrator.config import AsyncOrchestratorValidationCommandConfig

# Linux PID_MAX_LIMIT is 4194304 and pids are allocated in [1, pid_max), so no
# live process can ever hold this pid; ``os.kill``/``os.pidfd_open`` on it
# always raise ESRCH.
_UNALLOCATABLE_PID = 4_194_304 + 1
# Leader pid used by fabricated proc trees. Fabricated survivors carry it as
# their pgid/sid so they pass the reaper's ownership gate.
_LEADER_PID = 424_242
_START_TIME = 777_777


def _stat_line(
    pid: int,
    *,
    comm: str = "lean",
    state: str = "S",
    pgid: int = _LEADER_PID,
    sid: int = _LEADER_PID,
    start_time: int = _START_TIME,
) -> bytes:
    """Render a ``/proc/<pid>/stat`` line: pid (comm) state ppid pgrp session ...

    Fields 7 (tty_nr) through 21 (itrealvalue) are zeros; field 22 is
    ``starttime``, followed by a couple of trailing fields for realism.
    """

    middle = " ".join(["0"] * 15)
    return f"{pid} ({comm}) {state} 1 {pgid} {sid} {middle} {start_time} 10240 100".encode()


def _write_proc_entry(
    proc_root: Path,
    pid: int | str,
    *,
    argv: tuple[str, ...] = ("/toolchains/bin/lean", "Foo.lean"),
    comm: str = "lean",
    state: str = "S",
    pgid: int = _LEADER_PID,
    sid: int = _LEADER_PID,
    start_time: int = _START_TIME,
    with_stat: bool = True,
    with_cmdline: bool = True,
) -> None:
    entry = proc_root / str(pid)
    entry.mkdir(parents=True)
    if with_cmdline:
        (entry / "cmdline").write_bytes(b"".join(arg.encode() + b"\0" for arg in argv))
    if with_stat:
        numeric_pid = pid if isinstance(pid, int) else 0
        (entry / "stat").write_bytes(
            _stat_line(
                numeric_pid,
                comm=comm,
                state=state,
                pgid=pgid,
                sid=sid,
                start_time=start_time,
            )
        )


def _found_pids(leader_pid: int, proc_root: Path) -> tuple[int, ...]:
    orphans, _scan_complete = orchestrator_module._find_orphaned_validation_processes(  # pyright: ignore[reportPrivateUsage]
        leader_pid,
        proc_root,
    )
    return tuple(orphan.pid for orphan in orphans)


def _orphan_record(
    pid: int, start_time: int = _START_TIME
) -> orchestrator_module._OrphanedValidationProcess:  # pyright: ignore[reportPrivateUsage]
    return orchestrator_module._OrphanedValidationProcess(  # pyright: ignore[reportPrivateUsage]
        pid=pid,
        start_time=start_time,
        cmdline_head="test-orphan",
    )


@pytest.fixture
def force_pidfd_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``os.pidfd_open`` raise so the stat+``os.kill`` fallback runs.

    Fabricated proc trees use pids that do not exist (or are unrelated real
    processes), so the pidfd fast path would short-circuit before the code
    under test; forcing the fallback also covers platforms without pidfds.
    """

    def unavailable(pid: int, flags: int = 0) -> int:
        raise OSError(errno.ENOSYS, "pidfd_open not supported")

    monkeypatch.setattr(os, "pidfd_open", unavailable)


def _wait_until_exec(pid: int, basename: bytes, timeout_seconds: float = 5.0) -> None:
    """Wait until ``pid`` has exec'd into ``basename`` (fork/exec are not atomic)."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            argv0 = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[0]
        except OSError:
            return
        if argv0.endswith(basename):
            return
        time.sleep(0.01)


def _wait_until_dead(pid: int, timeout_seconds: float = 5.0) -> bool:
    """True once ``pid`` no longer exists or is an unreaped zombie.

    Processes killed by the reaper reparent to init when their parent is dead;
    until init reaps them they linger as zombies, which still "exist" for
    ``os.kill(pid, 0)`` — treat that state as dead too.
    """

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        try:
            _, _, tail = Path(f"/proc/{pid}/stat").read_bytes().rpartition(b")")
            if tail.split()[0] == b"Z":
                return True
        except (OSError, IndexError):
            return True
        time.sleep(0.05)
    return False


def _wait_for_file(path: Path, timeout_seconds: float = 5.0) -> str:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.exists():
            content = path.read_text(encoding="utf-8")
            if content:
                return content
        time.sleep(0.01)
    raise AssertionError(f"file never appeared: {path}")


# --- /proc/<pid>/stat parsing -------------------------------------------------


def test_read_process_stat_fields_survives_hostile_comm(tmp_path: Path) -> None:
    # comm is an arbitrary byte string: spaces, parentheses, and newlines are
    # all legal. Everything after the LAST ')' must be what gets parsed.
    hostile_comm = "we) (ird\nlean 7 7"
    stat_path = tmp_path / "stat"
    stat_path.write_bytes(
        _stat_line(4242, comm=hostile_comm, state="R", pgid=10, sid=11, start_time=12345)
    )

    fields = orchestrator_module._read_process_stat_fields(stat_path)  # pyright: ignore[reportPrivateUsage]

    assert isinstance(fields, orchestrator_module._ProcessStatFields)  # pyright: ignore[reportPrivateUsage]
    assert (fields.comm, fields.state) == (hostile_comm, "R")
    assert (fields.pgid, fields.sid, fields.start_time) == (10, 11, 12345)


def test_read_process_stat_fields_missing_file_is_gone(tmp_path: Path) -> None:
    # ENOENT means the proc entry vanished: the process is confirmed gone.
    assert (
        orchestrator_module._read_process_stat_fields(tmp_path / "missing")  # pyright: ignore[reportPrivateUsage]
        is None
    )


def test_read_process_stat_fields_malformed_is_unverifiable(tmp_path: Path) -> None:
    truncated = tmp_path / "truncated"
    truncated.write_bytes(b"12 (lean) S 1 2 3")
    non_numeric = tmp_path / "non-numeric"
    non_numeric.write_bytes(b"12 (lean) S 1 x y " + b" ".join([b"0"] * 16))

    for path in (truncated, non_numeric):
        result = orchestrator_module._read_process_stat_fields(path)  # pyright: ignore[reportPrivateUsage]
        assert isinstance(result, orchestrator_module._UnverifiableStat)  # pyright: ignore[reportPrivateUsage]
        assert result.detail == "malformed stat"


def test_read_process_stat_fields_permission_denied_is_unverifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # EACCES does not confirm exit: the result must be unverifiable, not gone.
    stat_path = tmp_path / "stat"
    stat_path.write_bytes(_stat_line(4242))
    real_read_bytes = Path.read_bytes

    def denying_read_bytes(self: Path) -> bytes:
        if self == stat_path:
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", denying_read_bytes)

    result = orchestrator_module._read_process_stat_fields(stat_path)  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result, orchestrator_module._UnverifiableStat)  # pyright: ignore[reportPrivateUsage]
    assert result.detail == f"errno={errno.EACCES}"


# --- scanning: ownership is the sole criterion ---------------------------------


def test_find_orphans_matches_any_group_or_session_member(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    # Names and argv are irrelevant: lean workers, lake leaders, bash helpers,
    # relative paths, no paths — every live group/session member is a candidate.
    _write_proc_entry(proc_root, 101, argv=("/toolchains/bin/lake", "build"), comm="lake")
    _write_proc_entry(proc_root, 102, argv=("lean", "Foo.lean"))
    _write_proc_entry(proc_root, 103, argv=("bash", "-c", "while true; do lake build; done"))
    # pgid matches but sid does not, and vice versa: either arm is proof of
    # membership in the killed validation's tree.
    _write_proc_entry(proc_root, 104, argv=("sleep", "3600"), pgid=_LEADER_PID, sid=99)
    _write_proc_entry(proc_root, 105, argv=("sleep", "3600"), pgid=99, sid=_LEADER_PID)

    assert _found_pids(_LEADER_PID, proc_root) == (101, 102, 103, 104, 105)


def test_find_orphans_rejects_unowned_and_dead_processes(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    # A maximal lookalike (lean name, absolute paths in argv) in a different
    # group AND session: an operator build or another validation. Spared.
    _write_proc_entry(
        proc_root,
        201,
        argv=("/toolchains/bin/lean", "/tmp/staging/Foo.lean"),
        pgid=201,
        sid=201,
    )
    # Zombies/dead processes cannot be signalled into anything.
    _write_proc_entry(proc_root, 202, state="Z")
    _write_proc_entry(proc_root, 203, state="X")
    # No stat at all (vanished mid-scan) and a garbled stat (unverifiable):
    # ownership cannot be established, so neither becomes a candidate.
    _write_proc_entry(proc_root, 204, with_stat=False)
    (proc_root / "205").mkdir()
    (proc_root / "205" / "stat").write_bytes(b"garbage")
    # Non-pid entries (``self``, ``meminfo``, ...) are skipped.
    _write_proc_entry(proc_root, "self")
    (proc_root / "300").write_bytes(b"")  # digit-named plain file, not a pid dir
    # The scanning process itself is never a reap candidate.
    _write_proc_entry(proc_root, os.getpid())

    assert _found_pids(_LEADER_PID, proc_root) == ()


def test_find_orphans_reports_descriptor_and_start_time(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _write_proc_entry(proc_root, 101, argv=("/bin/lake", "build"), start_time=555)
    # Empty/unreadable cmdline falls back to the stat comm — enrichment only.
    _write_proc_entry(proc_root, 102, argv=(), comm="lean-worker", start_time=556)
    _write_proc_entry(proc_root, 103, with_cmdline=False, comm="ghost", start_time=557)

    orphans, scan_complete = orchestrator_module._find_orphaned_validation_processes(  # pyright: ignore[reportPrivateUsage]
        _LEADER_PID,
        proc_root,
    )

    assert scan_complete
    assert [(o.pid, o.start_time, o.cmdline_head) for o in orphans] == [
        (101, 555, "/bin/lake build"),
        (102, 556, "[lean-worker]"),
        (103, 557, "[ghost]"),
    ]


def test_find_orphans_without_proc_root_is_empty_but_incomplete(tmp_path: Path) -> None:
    orphans, scan_complete = orchestrator_module._find_orphaned_validation_processes(  # pyright: ignore[reportPrivateUsage]
        _LEADER_PID,
        tmp_path / "missing-proc",
    )

    assert orphans == ()
    assert not scan_complete


# --- pre-signal identity verification -------------------------------------------


def test_fallback_skips_kill_on_starttime_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force_pidfd_fallback: None
) -> None:
    # The scanned process exited and its pid was recycled: the stat file exists
    # but reports a different starttime. No signal may be sent.
    proc_root = tmp_path / "proc"
    _write_proc_entry(proc_root, 101, start_time=_START_TIME)

    def forbidden_kill(pid: int, sig: int) -> None:
        raise AssertionError(f"os.kill({pid}, {sig}) must not be reached")

    monkeypatch.setattr(os, "kill", forbidden_kill)

    outcome = orchestrator_module._verify_and_signal_orphan(  # pyright: ignore[reportPrivateUsage]
        _orphan_record(101, start_time=_START_TIME + 1),
        _LEADER_PID,
        signal.SIGTERM,
        proc_root,
    )

    assert outcome == "gone"


def test_fallback_skips_kill_when_process_left_leader_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force_pidfd_fallback: None
) -> None:
    # Same starttime (same process) but it no longer sits in the killed
    # leader's group/session — the accepted setsid-escape narrowing.
    proc_root = tmp_path / "proc"
    _write_proc_entry(proc_root, 101, pgid=9, sid=9)

    def forbidden_kill(pid: int, sig: int) -> None:
        raise AssertionError(f"os.kill({pid}, {sig}) must not be reached")

    monkeypatch.setattr(os, "kill", forbidden_kill)

    outcome = orchestrator_module._verify_and_signal_orphan(  # pyright: ignore[reportPrivateUsage]
        _orphan_record(101),
        _LEADER_PID,
        signal.SIGTERM,
        proc_root,
    )

    assert outcome == "escaped"


def test_fallback_delivers_via_os_kill_when_pidfd_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force_pidfd_fallback: None
) -> None:
    proc_root = tmp_path / "proc"
    _write_proc_entry(proc_root, 101)
    kills: list[tuple[int, int]] = []
    real_kill = os.kill

    def recording_kill(pid: int, sig: int) -> None:
        if pid == 101:
            kills.append((pid, sig))
            return
        real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", recording_kill)

    outcome = orchestrator_module._verify_and_signal_orphan(  # pyright: ignore[reportPrivateUsage]
        _orphan_record(101),
        _LEADER_PID,
        signal.SIGTERM,
        proc_root,
    )

    assert outcome == "delivered"
    assert kills == [(101, signal.SIGTERM)]


def test_pidfd_open_oserror_falls_back_to_os_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    _write_proc_entry(proc_root, 101)
    kills: list[tuple[int, int]] = []

    def exhausted_pidfd_open(pid: int, flags: int = 0) -> int:
        raise OSError(errno.EMFILE, "too many open files")

    def unused_pidfd_send_signal(pidfd: int, sig: int) -> None:
        pass

    def recording_kill(pid: int, sig: int) -> None:
        kills.append((pid, sig))

    monkeypatch.setattr(os, "pidfd_open", exhausted_pidfd_open)
    monkeypatch.setattr(signal, "pidfd_send_signal", unused_pidfd_send_signal)
    monkeypatch.setattr(os, "kill", recording_kill)

    outcome = orchestrator_module._verify_and_signal_orphan(  # pyright: ignore[reportPrivateUsage]
        _orphan_record(101),
        _LEADER_PID,
        signal.SIGTERM,
        proc_root,
    )

    assert outcome == "delivered"
    assert kills == [(101, signal.SIGTERM)]


def test_pidfd_send_failure_is_warned_and_not_reaped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    proc_root = tmp_path / "proc"
    _write_proc_entry(proc_root, 101)
    fake_pidfd = 987_654
    closed: list[int] = []
    real_close = os.close

    def fake_pidfd_open(pid: int, flags: int = 0) -> int:
        return fake_pidfd

    def failing_send(pidfd: int, sig: int) -> None:
        raise PermissionError(errno.EPERM, "operation not permitted")

    def recording_close(fd: int) -> None:
        if fd == fake_pidfd:
            closed.append(fd)
            return
        real_close(fd)

    monkeypatch.setattr(os, "pidfd_open", fake_pidfd_open)
    monkeypatch.setattr(signal, "pidfd_send_signal", failing_send)
    monkeypatch.setattr(os, "close", recording_close)

    with caplog.at_level("WARNING", logger=orchestrator_module.__name__):
        outcome = orchestrator_module._verify_and_signal_orphan(  # pyright: ignore[reportPrivateUsage]
            _orphan_record(101),
            _LEADER_PID,
            signal.SIGKILL,
            proc_root,
        )

    survivors, scan_complete, reaped = orchestrator_module._sigkill_validation_orphans_once(  # pyright: ignore[reportPrivateUsage]
        tmp_path / "staging", _LEADER_PID, proc_root
    )

    assert outcome == "failed"
    assert scan_complete
    assert tuple(orphan.pid for orphan in survivors) == (101,)
    assert reaped == ()
    assert closed == [fake_pidfd, fake_pidfd]
    assert "failed to signal orphaned validation process" in caplog.text
    assert f"errno={errno.EPERM}" in caplog.text


def test_pidfd_close_failure_is_suppressed_and_sweep_result_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    _write_proc_entry(proc_root, 101)
    fake_pidfd = 987_654
    real_close = os.close

    def fake_pidfd_open(pid: int, flags: int = 0) -> int:
        return fake_pidfd

    def delivering_send(pidfd: int, sig: int) -> None:
        assert (pidfd, sig) == (fake_pidfd, signal.SIGKILL)
        shutil.rmtree(proc_root / "101")

    def failing_close(fd: int) -> None:
        if fd == fake_pidfd:
            raise OSError(errno.EIO, "close failed")
        real_close(fd)

    monkeypatch.setattr(os, "pidfd_open", fake_pidfd_open)
    monkeypatch.setattr(signal, "pidfd_send_signal", delivering_send)
    monkeypatch.setattr(os, "close", failing_close)

    survivors, scan_complete, reaped = orchestrator_module._sigkill_validation_orphans_once(  # pyright: ignore[reportPrivateUsage]
        tmp_path / "staging", _LEADER_PID, proc_root
    )

    assert tuple(orphan.pid for orphan in survivors) == (101,)
    assert scan_complete
    assert reaped == (101,)


def test_pidfd_path_delivers_signal_to_verified_process(tmp_path: Path) -> None:
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        pytest.skip("pidfd support is required")
    sleep_binary = shutil.which("sleep")
    if sleep_binary is None:
        pytest.skip("sleep executable is required")

    # A real disposable child; the fabricated stat vouches for its ownership
    # and starttime so verification passes and pidfd_send_signal really fires.
    child = subprocess.Popen([sleep_binary, "3600"])
    proc_root = tmp_path / "proc"
    _write_proc_entry(proc_root, child.pid)
    try:
        outcome = orchestrator_module._verify_and_signal_orphan(  # pyright: ignore[reportPrivateUsage]
            _orphan_record(child.pid),
            _LEADER_PID,
            signal.SIGTERM,
            proc_root,
        )

        assert outcome == "delivered"
        assert child.wait(timeout=5) == -signal.SIGTERM
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_pidfd_path_skips_signal_on_identity_mismatch(tmp_path: Path) -> None:
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        pytest.skip("pidfd support is required")
    sleep_binary = shutil.which("sleep")
    if sleep_binary is None:
        pytest.skip("sleep executable is required")

    # The pidfd opens successfully, but the stat re-verification sees a
    # different starttime (as after pid recycling): no signal may go through
    # the open fd. The child surviving a would-be SIGKILL proves it.
    child = subprocess.Popen([sleep_binary, "3600"])
    proc_root = tmp_path / "proc"
    _write_proc_entry(proc_root, child.pid, start_time=_START_TIME)
    try:
        outcome = orchestrator_module._verify_and_signal_orphan(  # pyright: ignore[reportPrivateUsage]
            _orphan_record(child.pid, start_time=_START_TIME + 1),
            _LEADER_PID,
            signal.SIGKILL,
            proc_root,
        )

        assert outcome == "gone"
        assert child.poll() is None
    finally:
        child.kill()
        child.wait(timeout=5)


def test_pidfd_path_skips_signal_when_stat_unverifiable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        pytest.skip("pidfd support is required")
    sleep_binary = shutil.which("sleep")
    if sleep_binary is None:
        pytest.skip("sleep executable is required")

    # A garbled stat proves neither identity nor exit: no signal, WARNING, and
    # the outcome must not read as reaped.
    child = subprocess.Popen([sleep_binary, "3600"])
    proc_root = tmp_path / "proc"
    entry = proc_root / str(child.pid)
    entry.mkdir(parents=True)
    (entry / "stat").write_bytes(b"garbage")
    try:
        with caplog.at_level("WARNING", logger=orchestrator_module.__name__):
            outcome = orchestrator_module._verify_and_signal_orphan(  # pyright: ignore[reportPrivateUsage]
                _orphan_record(child.pid),
                _LEADER_PID,
                signal.SIGKILL,
                proc_root,
            )

        assert outcome == "unverifiable"
        assert child.poll() is None
        assert "cannot verify orphaned validation process" in caplog.text
    finally:
        child.kill()
        child.wait(timeout=5)


# --- reaper outcome reporting and kill-pass loop ---------------------------------


async def test_reap_is_noop_without_proc_root(tmp_path: Path) -> None:
    # Graceful no-op on hosts without /proc (non-Linux).
    reaped = await orchestrator_module._reap_orphaned_validation_processes(  # pyright: ignore[reportPrivateUsage]
        tmp_path / "staging",
        _LEADER_PID,
        proc_root=tmp_path / "missing-proc",
        grace_seconds=0.0,
    )

    assert reaped == ()


async def test_reap_returns_pid_confirmed_gone_without_claiming_delivery(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    proc_root = tmp_path / "proc"
    # The fabricated entry names a pid that cannot exist, so every signalling
    # attempt confirms it is gone (ESRCH). That is the reaper's goal state —
    # the pid is returned as confirmed gone — but no "reaping ..." delivery
    # line may be logged for it, and the kill loop must terminate promptly
    # even though the fabricated /proc entry never disappears.
    _write_proc_entry(proc_root, _UNALLOCATABLE_PID)

    with caplog.at_level("WARNING", logger=orchestrator_module.__name__):
        reaped = await orchestrator_module._reap_orphaned_validation_processes(  # pyright: ignore[reportPrivateUsage]
            tmp_path / "staging",
            _LEADER_PID,
            proc_root=proc_root,
            grace_seconds=0.0,
        )

    assert reaped == (_UNALLOCATABLE_PID,)
    assert "reaping orphaned validation process" not in caplog.text


async def test_reap_excludes_pid_whose_kill_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    force_pidfd_fallback: None,
) -> None:
    proc_root = tmp_path / "proc"
    _write_proc_entry(proc_root, _UNALLOCATABLE_PID)
    real_kill = os.kill

    def eperm_kill(pid: int, sig: int) -> None:
        if pid == _UNALLOCATABLE_PID:
            raise PermissionError(errno.EPERM, "Operation not permitted")
        real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", eperm_kill)

    with caplog.at_level("WARNING", logger=orchestrator_module.__name__):
        reaped = await orchestrator_module._reap_orphaned_validation_processes(  # pyright: ignore[reportPrivateUsage]
            tmp_path / "staging",
            _LEADER_PID,
            proc_root=proc_root,
            grace_seconds=0.0,
        )

    # An EPERM process was NOT killed, so it must not be reported as reaped —
    # but the failure must be logged with its errno.
    assert reaped == ()
    assert "failed to signal orphaned validation process" in caplog.text
    assert f"errno={errno.EPERM}" in caplog.text


async def test_kill_loop_reaps_replacement_forked_between_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force_pidfd_fallback: None
) -> None:
    proc_root = tmp_path / "proc"
    first_pid, second_pid = 101, 102
    _write_proc_entry(proc_root, first_pid, argv=("bash", "forker"), comm="bash")
    kills: list[tuple[int, int]] = []
    real_kill = os.kill

    # The first orphan ignores SIGTERM and, exactly when SIGKILLed, "forks" a
    # replacement that only a subsequent scan can see. A single post-grace
    # snapshot (the round-1 design) would return with the replacement alive;
    # the loop must catch it in the next pass.
    def forking_kill(pid: int, sig: int) -> None:
        if pid not in (first_pid, second_pid):
            real_kill(pid, sig)
            return
        kills.append((pid, sig))
        if sig != signal.SIGKILL:
            return
        shutil.rmtree(proc_root / str(pid))
        if pid == first_pid:
            _write_proc_entry(proc_root, second_pid, argv=("lean", "Replacement.lean"))

    monkeypatch.setattr(os, "kill", forking_kill)

    reaped = await orchestrator_module._reap_orphaned_validation_processes(  # pyright: ignore[reportPrivateUsage]
        tmp_path / "staging",
        _LEADER_PID,
        proc_root=proc_root,
        grace_seconds=0.0,
    )

    assert set(reaped) == {first_pid, second_pid}
    assert kills == [
        (first_pid, signal.SIGTERM),
        (first_pid, signal.SIGKILL),
        (second_pid, signal.SIGKILL),
    ]


async def test_kill_loop_rescans_when_candidate_is_gone_before_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force_pidfd_fallback: None
) -> None:
    proc_root = tmp_path / "proc"
    first_pid, replacement_pid = 101, 102
    _write_proc_entry(proc_root, first_pid, argv=("bash", "forker"), comm="bash")
    real_reverify = orchestrator_module._reverify_orphan_identity  # pyright: ignore[reportPrivateUsage]
    real_kill = os.kill
    replaced = False
    kills: list[tuple[int, int]] = []

    # The scan sees generation A. Before its identity check, A forks B and
    # exits, so A's outcome is "gone" without any delivery. The loop must still
    # rescan and discover B rather than treating no delivery as convergence.
    def replacing_reverify(
        orphan: orchestrator_module._OrphanedValidationProcess,  # pyright: ignore[reportPrivateUsage]
        leader_pid: int,
        root: Path,
    ) -> Literal["verified", "gone", "escaped", "unverifiable"]:
        nonlocal replaced
        if orphan.pid == first_pid and not replaced:
            replaced = True
            shutil.rmtree(proc_root / str(first_pid))
            _write_proc_entry(proc_root, replacement_pid, argv=("lean", "Replacement.lean"))
        return real_reverify(orphan, leader_pid, root)

    def killing_replacement(pid: int, sig: int) -> None:
        if pid == replacement_pid:
            kills.append((pid, sig))
            shutil.rmtree(proc_root / str(pid))
            return
        real_kill(pid, sig)

    monkeypatch.setattr(orchestrator_module, "_reverify_orphan_identity", replacing_reverify)
    monkeypatch.setattr(os, "kill", killing_replacement)

    reaped = await orchestrator_module._sigkill_surviving_validation_orphans(  # pyright: ignore[reportPrivateUsage]
        tmp_path / "staging", _LEADER_PID, proc_root
    )

    assert reaped == (first_pid, replacement_pid)
    assert kills == [(replacement_pid, signal.SIGKILL)]


async def test_kill_loop_retries_stable_candidate_after_transient_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force_pidfd_fallback: None
) -> None:
    proc_root = tmp_path / "proc"
    pid = 101
    _write_proc_entry(proc_root, pid)
    real_reverify = orchestrator_module._reverify_orphan_identity  # pyright: ignore[reportPrivateUsage]
    attempts = 0

    # Two identical scans with transiently unverifiable stat reads do not prove
    # that this candidate is permanently stuck. The old stable-set heuristic
    # stopped after attempt two; the pass-cap-only loop must retry and kill it.
    def transient_reverify(
        orphan: orchestrator_module._OrphanedValidationProcess,  # pyright: ignore[reportPrivateUsage]
        leader_pid: int,
        root: Path,
    ) -> Literal["verified", "gone", "escaped", "unverifiable"]:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            return "unverifiable"
        return real_reverify(orphan, leader_pid, root)

    def killing_orphan(target_pid: int, sig: int) -> None:
        assert (target_pid, sig) == (pid, signal.SIGKILL)
        shutil.rmtree(proc_root / str(pid))

    monkeypatch.setattr(orchestrator_module, "_reverify_orphan_identity", transient_reverify)
    monkeypatch.setattr(os, "kill", killing_orphan)

    reaped = await orchestrator_module._sigkill_surviving_validation_orphans(  # pyright: ignore[reportPrivateUsage]
        tmp_path / "staging", _LEADER_PID, proc_root
    )

    assert attempts == 3
    assert reaped == (pid,)


async def test_reap_retries_transient_eio_during_discovery_and_kills_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force_pidfd_fallback: None
) -> None:
    proc_root = tmp_path / "proc"
    pid = 101
    stat_path = proc_root / str(pid) / "stat"
    _write_proc_entry(proc_root, pid)
    real_read_bytes = Path.read_bytes
    real_kill = os.kill
    stat_reads = 0
    kills: list[tuple[int, int]] = []

    # The initial discovery read fails before the process can become a
    # candidate. An empty-but-incomplete scan must enter the grace/kill loop;
    # the next discovery succeeds and the durable orphan is SIGKILLed.
    def transient_discovery_read(self: Path) -> bytes:
        nonlocal stat_reads
        if self == stat_path:
            stat_reads += 1
            if stat_reads == 1:
                raise OSError(errno.EIO, "transient discovery failure")
        return real_read_bytes(self)

    def killing_orphan(target_pid: int, sig: int) -> None:
        if target_pid == pid:
            kills.append((target_pid, sig))
            shutil.rmtree(proc_root / str(pid))
            return
        real_kill(target_pid, sig)

    monkeypatch.setattr(Path, "read_bytes", transient_discovery_read)
    monkeypatch.setattr(os, "kill", killing_orphan)

    reaped = await orchestrator_module._reap_orphaned_validation_processes(  # pyright: ignore[reportPrivateUsage]
        tmp_path / "staging",
        _LEADER_PID,
        proc_root=proc_root,
        grace_seconds=0.0,
    )

    assert stat_reads >= 3  # failed discovery, successful discovery, reverify
    assert kills == [(pid, signal.SIGKILL)]
    assert reaped == (pid,)


async def test_reap_confirms_empty_scan_after_fork_during_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force_pidfd_fallback: None
) -> None:
    proc_root = tmp_path / "proc"
    first_pid, replacement_pid = 101, 102
    first_stat = proc_root / str(first_pid) / "stat"
    _write_proc_entry(proc_root, first_pid, argv=("bash", "forker"), comm="bash")
    real_read_bytes = Path.read_bytes
    real_kill = os.kill
    replaced = False
    kills: list[tuple[int, int]] = []

    # Scan i enumerates A. During A's stat read it forks durable B and exits:
    # A is gone and B was absent from the enumeration, so scan i is empty and
    # complete. The initial fast path must perform scan i+1, discover B, and
    # reap it rather than accepting the first point-in-time snapshot.
    def replacing_discovery_read(self: Path) -> bytes:
        nonlocal replaced
        if self == first_stat and not replaced:
            replaced = True
            _write_proc_entry(
                proc_root,
                replacement_pid,
                argv=("lean", "Replacement.lean"),
            )
            shutil.rmtree(proc_root / str(first_pid))
        return real_read_bytes(self)

    def killing_replacement(target_pid: int, sig: int) -> None:
        if target_pid == replacement_pid:
            kills.append((target_pid, sig))
            shutil.rmtree(proc_root / str(target_pid))
            return
        real_kill(target_pid, sig)

    monkeypatch.setattr(Path, "read_bytes", replacing_discovery_read)
    monkeypatch.setattr(os, "kill", killing_replacement)

    reaped = await orchestrator_module._reap_orphaned_validation_processes(  # pyright: ignore[reportPrivateUsage]
        tmp_path / "staging",
        _LEADER_PID,
        proc_root=proc_root,
        grace_seconds=0.0,
    )

    assert replaced
    assert kills == [(replacement_pid, signal.SIGTERM)]
    assert reaped == (replacement_pid,)
    assert not (proc_root / str(replacement_pid)).exists()


async def test_kill_loop_warns_when_final_scan_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    proc_root = tmp_path / "proc"
    entry = proc_root / "101"
    entry.mkdir(parents=True)
    (entry / "stat").write_bytes(b"garbage")
    monkeypatch.setattr(orchestrator_module, "_VALIDATION_ORPHAN_KILL_PASS_LIMIT", 1)

    with caplog.at_level("WARNING", logger=orchestrator_module.__name__):
        reaped = await orchestrator_module._sigkill_surviving_validation_orphans(  # pyright: ignore[reportPrivateUsage]
            tmp_path / "staging", _LEADER_PID, proc_root
        )

    assert reaped == ()
    assert "could not be fully verified" in caplog.text
    assert "known_survivors=none discovered" in caplog.text


async def test_kill_loop_warns_at_cap_when_only_final_scan_is_empty_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    candidate = _orphan_record(101)
    pass_calls = 0
    final_scan_calls = 0

    # Every bounded pass is nonempty and incomplete, so none contributes even
    # one clean scan toward convergence. A first empty+complete result from the
    # post-cap reporting scan must not make the cap exit look converged.
    def nonconverged_pass(
        repo: Path, leader_pid: int, root: Path
    ) -> tuple[
        tuple[orchestrator_module._OrphanedValidationProcess, ...],  # pyright: ignore[reportPrivateUsage]
        bool,
        tuple[int, ...],
    ]:
        nonlocal pass_calls
        pass_calls += 1
        return (candidate,), False, ()

    def empty_final_scan(
        leader_pid: int, root: Path
    ) -> tuple[
        tuple[orchestrator_module._OrphanedValidationProcess, ...],  # pyright: ignore[reportPrivateUsage]
        bool,
    ]:
        nonlocal final_scan_calls
        final_scan_calls += 1
        return (), True

    monkeypatch.setattr(
        orchestrator_module, "_sigkill_validation_orphans_once", nonconverged_pass
    )
    monkeypatch.setattr(
        orchestrator_module, "_find_orphaned_validation_processes", empty_final_scan
    )
    monkeypatch.setattr(orchestrator_module, "_VALIDATION_ORPHAN_KILL_PASS_LIMIT", 2)
    monkeypatch.setattr(orchestrator_module, "_VALIDATION_ORPHAN_KILL_PASS_INTERVAL_SECONDS", 0.0)

    with caplog.at_level("WARNING", logger=orchestrator_module.__name__):
        reaped = await orchestrator_module._sigkill_surviving_validation_orphans(  # pyright: ignore[reportPrivateUsage]
            tmp_path / "staging", _LEADER_PID, tmp_path / "proc"
        )

    convergence_warnings = [
        record
        for record in caplog.records
        if "two-scan convergence not established" in record.getMessage()
    ]
    assert reaped == ()
    assert pass_calls == 2
    assert final_scan_calls == 1
    assert len(convergence_warnings) > 0
    assert "bounded by pass cap" in caplog.text
    assert "final scan found no survivors" in caplog.text


async def test_kill_loop_clean_convergence_does_not_warn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pass_calls = 0

    def empty_complete_pass(
        repo: Path, leader_pid: int, root: Path
    ) -> tuple[
        tuple[orchestrator_module._OrphanedValidationProcess, ...],  # pyright: ignore[reportPrivateUsage]
        bool,
        tuple[int, ...],
    ]:
        nonlocal pass_calls
        pass_calls += 1
        return (), True, ()

    def unexpected_final_scan(
        leader_pid: int, root: Path
    ) -> tuple[
        tuple[orchestrator_module._OrphanedValidationProcess, ...],  # pyright: ignore[reportPrivateUsage]
        bool,
    ]:
        raise AssertionError("converged cleanup must not perform a warning scan")

    monkeypatch.setattr(
        orchestrator_module, "_sigkill_validation_orphans_once", empty_complete_pass
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_find_orphaned_validation_processes",
        unexpected_final_scan,
    )
    monkeypatch.setattr(orchestrator_module, "_VALIDATION_ORPHAN_KILL_PASS_INTERVAL_SECONDS", 0.0)

    with caplog.at_level("WARNING", logger=orchestrator_module.__name__):
        reaped = await orchestrator_module._sigkill_surviving_validation_orphans(  # pyright: ignore[reportPrivateUsage]
            tmp_path / "staging", _LEADER_PID, tmp_path / "proc"
        )

    convergence_warnings = [
        record
        for record in caplog.records
        if "two-scan convergence not established" in record.getMessage()
    ]
    assert reaped == ()
    assert pass_calls == 2
    assert convergence_warnings == []


async def test_kill_loop_final_warning_uses_fresh_scan_after_pass_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    force_pidfd_fallback: None,
) -> None:
    proc_root = tmp_path / "proc"
    first_pid, replacement_pid = 101, 102
    _write_proc_entry(proc_root, first_pid)
    real_reverify = orchestrator_module._reverify_orphan_identity  # pyright: ignore[reportPrivateUsage]

    # With a one-pass cap, A disappears as "gone" during verification and B
    # cannot be signalled. The post-cap scan must nevertheless name B rather
    # than warning only about candidates whose signalling failed in the pass.
    def replacing_reverify(
        orphan: orchestrator_module._OrphanedValidationProcess,  # pyright: ignore[reportPrivateUsage]
        leader_pid: int,
        root: Path,
    ) -> Literal["verified", "gone", "escaped", "unverifiable"]:
        shutil.rmtree(proc_root / str(first_pid))
        _write_proc_entry(proc_root, replacement_pid)
        return real_reverify(orphan, leader_pid, root)

    monkeypatch.setattr(orchestrator_module, "_reverify_orphan_identity", replacing_reverify)
    monkeypatch.setattr(orchestrator_module, "_VALIDATION_ORPHAN_KILL_PASS_LIMIT", 1)

    with caplog.at_level("WARNING", logger=orchestrator_module.__name__):
        reaped = await orchestrator_module._sigkill_surviving_validation_orphans(  # pyright: ignore[reportPrivateUsage]
            tmp_path / "staging", _LEADER_PID, proc_root
        )

    assert reaped == (first_pid,)
    assert "orphaned validation processes were not reaped" in caplog.text
    assert f"pid={replacement_pid}" in caplog.text


def test_sync_kill_loop_warns_at_cap_when_only_final_scan_is_empty_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    candidate = _orphan_record(101)
    pass_calls = 0
    final_scan_calls = 0

    def nonconverged_pass(
        repo: Path, leader_pid: int, root: Path
    ) -> tuple[
        tuple[orchestrator_module._OrphanedValidationProcess, ...],  # pyright: ignore[reportPrivateUsage]
        bool,
        tuple[int, ...],
    ]:
        nonlocal pass_calls
        pass_calls += 1
        return (candidate,), False, ()

    def empty_final_scan(
        leader_pid: int, root: Path
    ) -> tuple[
        tuple[orchestrator_module._OrphanedValidationProcess, ...],  # pyright: ignore[reportPrivateUsage]
        bool,
    ]:
        nonlocal final_scan_calls
        final_scan_calls += 1
        return (), True

    monkeypatch.setattr(
        orchestrator_module, "_sigkill_validation_orphans_once", nonconverged_pass
    )
    monkeypatch.setattr(
        orchestrator_module, "_find_orphaned_validation_processes", empty_final_scan
    )
    monkeypatch.setattr(orchestrator_module, "_VALIDATION_ORPHAN_KILL_PASS_LIMIT", 2)
    monkeypatch.setattr(orchestrator_module, "_VALIDATION_ORPHAN_KILL_PASS_INTERVAL_SECONDS", 0.0)

    with caplog.at_level("WARNING", logger=orchestrator_module.__name__):
        reaped = orchestrator_module._sigkill_surviving_validation_orphans_sync(  # pyright: ignore[reportPrivateUsage]
            tmp_path / "staging", _LEADER_PID, tmp_path / "proc"
        )

    convergence_warnings = [
        record
        for record in caplog.records
        if "two-scan convergence not established" in record.getMessage()
    ]
    assert reaped == ()
    assert pass_calls == 2
    assert final_scan_calls == 1
    assert len(convergence_warnings) > 0
    assert "bounded by pass cap" in caplog.text
    assert "final scan found no survivors" in caplog.text


def test_sync_kill_loop_stops_starting_slow_passes_at_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    proc_root = tmp_path / "proc"
    pid = 101
    _write_proc_entry(proc_root, pid)
    candidate = _orphan_record(pid)
    pass_calls = 0
    post_deadline_scan_calls = 0

    # Model a pathologically expensive scan/signal pass. The current pass is
    # uninterruptible, but once it carries cleanup beyond the deadline the
    # synchronous path must start neither another pass nor the formerly
    # unconditional fresh warning scan (which would double the delay).
    def slow_pass(
        repo: Path, leader_pid: int, root: Path
    ) -> tuple[
        tuple[orchestrator_module._OrphanedValidationProcess, ...],  # pyright: ignore[reportPrivateUsage]
        bool,
        tuple[int, ...],
    ]:
        nonlocal pass_calls
        pass_calls += 1
        time.sleep(0.06)
        return (candidate,), True, ()

    def expensive_post_deadline_scan(
        leader_pid: int, root: Path
    ) -> tuple[
        tuple[orchestrator_module._OrphanedValidationProcess, ...],  # pyright: ignore[reportPrivateUsage]
        bool,
    ]:
        nonlocal post_deadline_scan_calls
        post_deadline_scan_calls += 1
        time.sleep(0.06)
        return (candidate,), True

    monkeypatch.setattr(orchestrator_module, "_sigkill_validation_orphans_once", slow_pass)
    monkeypatch.setattr(
        orchestrator_module,
        "_find_orphaned_validation_processes",
        expensive_post_deadline_scan,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_VALIDATION_ORPHAN_SYNC_CLEANUP_DEADLINE_SECONDS",
        0.05,
    )
    monkeypatch.setattr(orchestrator_module, "_VALIDATION_ORPHAN_KILL_PASS_INTERVAL_SECONDS", 0.0)
    started = time.monotonic()

    with caplog.at_level("WARNING", logger=orchestrator_module.__name__):
        reaped = orchestrator_module._sigkill_surviving_validation_orphans_sync(  # pyright: ignore[reportPrivateUsage]
            tmp_path / "staging", _LEADER_PID, proc_root
        )

    elapsed = time.monotonic() - started
    assert reaped == ()
    assert pass_calls == 1
    assert post_deadline_scan_calls == 0
    assert elapsed < 0.25
    assert "bounded by synchronous cleanup deadline" in caplog.text
    assert "two-scan convergence not established" in caplog.text
    assert "survivor_list_may_be_stale=true" in caplog.text
    assert f"pid={pid}" in caplog.text


async def test_reap_escalates_to_sigkill_when_cancelled_during_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force_pidfd_fallback: None
) -> None:
    proc_root = tmp_path / "proc"
    _write_proc_entry(proc_root, _UNALLOCATABLE_PID)
    kills: list[tuple[int, int]] = []
    real_kill = os.kill

    def recording_kill(pid: int, sig: int) -> None:
        if pid == _UNALLOCATABLE_PID:
            kills.append((pid, sig))
            if sig == signal.SIGKILL:
                shutil.rmtree(proc_root / str(pid))  # simulate death
            return
        real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", recording_kill)

    task = asyncio.create_task(
        orchestrator_module._reap_orphaned_validation_processes(  # pyright: ignore[reportPrivateUsage]
            tmp_path / "staging",
            _LEADER_PID,
            proc_root=proc_root,
            grace_seconds=60.0,
        )
    )
    # One yield lets the task run its synchronous scan + SIGTERM pass and park
    # in the grace sleep; cancelling there must still run the SIGKILL passes
    # before the CancelledError propagates.
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert kills == [
        (_UNALLOCATABLE_PID, signal.SIGTERM),
        (_UNALLOCATABLE_PID, signal.SIGKILL),
    ]


async def test_reap_cancellation_sweep_kills_replacement_forked_during_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force_pidfd_fallback: None
) -> None:
    proc_root = tmp_path / "proc"
    first_pid, replacement_pid = 101, 102
    _write_proc_entry(proc_root, first_pid, argv=("bash", "forker"), comm="bash")
    kills: list[tuple[int, int]] = []
    real_kill = os.kill
    real_asyncio_sleep = asyncio.sleep
    interpass_sleep_started = asyncio.Event()

    def forking_kill(pid: int, sig: int) -> None:
        if pid not in (first_pid, replacement_pid):
            real_kill(pid, sig)
            return
        kills.append((pid, sig))
        if sig != signal.SIGKILL:
            return
        # The normal loop's first SIGKILL leaves the forker visible and enters
        # its async sleep. During cancellation cleanup, killing that forker
        # creates a replacement which only another synchronous pass can see.
        if pid == first_pid and kills.count((first_pid, signal.SIGKILL)) == 2:
            shutil.rmtree(proc_root / str(first_pid))
            _write_proc_entry(proc_root, replacement_pid, argv=("lean", "Replacement.lean"))
        elif pid == replacement_pid:
            shutil.rmtree(proc_root / str(replacement_pid))

    async def controlled_sleep(delay: float) -> None:
        if delay == orchestrator_module._VALIDATION_ORPHAN_KILL_PASS_INTERVAL_SECONDS:  # pyright: ignore[reportPrivateUsage]
            interpass_sleep_started.set()
            await real_asyncio_sleep(60.0)
        else:
            await real_asyncio_sleep(delay)

    monkeypatch.setattr(os, "kill", forking_kill)
    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    task = asyncio.create_task(
        orchestrator_module._reap_orphaned_validation_processes(  # pyright: ignore[reportPrivateUsage]
            tmp_path / "staging",
            _LEADER_PID,
            proc_root=proc_root,
            grace_seconds=0.0,
        )
    )
    await asyncio.wait_for(interpass_sleep_started.wait(), timeout=5.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert kills == [
        (first_pid, signal.SIGTERM),
        (first_pid, signal.SIGKILL),
        (first_pid, signal.SIGKILL),
        (replacement_pid, signal.SIGKILL),
    ]
    assert not (proc_root / str(replacement_pid)).exists()


# --- integration: real process trees ---------------------------------------------


async def test_reap_spares_same_repo_bystander_in_different_session(tmp_path: Path) -> None:
    # The primary false-positive regression test: a maximal lookalike (binary
    # named ``lean``, absolute repo path in argv) working on the SAME repo but
    # in a different session — e.g. an operator's manual build — must survive
    # the sweep for a killed validation leader.
    tail = shutil.which("tail")
    true_binary = shutil.which("true")
    if tail is None or true_binary is None or not Path("/proc").exists():
        pytest.skip("tail, true, and /proc are required")
    fake_lean = tmp_path / "bin" / "lean"
    fake_lean.parent.mkdir()
    shutil.copy(tail, fake_lean)
    repo = tmp_path / "staging"
    repo.mkdir()
    marker = repo / "Foo.lean"
    marker.write_text("-- marker\n", encoding="utf-8")

    # A validation leader that has already died, leaving an empty group.
    leader = subprocess.Popen([true_binary], start_new_session=True)
    leader.wait(timeout=5)
    bystander = subprocess.Popen(
        [str(fake_lean), "-f", str(marker)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        reaped = await orchestrator_module._reap_orphaned_validation_processes(  # pyright: ignore[reportPrivateUsage]
            repo,
            leader.pid,
            grace_seconds=0.2,
        )

        assert bystander.pid not in reaped
        assert bystander.poll() is None
    finally:
        if bystander.poll() is None:
            bystander.kill()
        bystander.wait(timeout=5)


async def test_reap_kills_group_orphan_with_no_repo_path_and_foreign_name(
    tmp_path: Path,
) -> None:
    # The round-2 false-negative shape: a group member whose argv contains no
    # repo path and whose basename is not lean/lake (here: plain ``tail``)
    # must be reaped — ownership alone convicts it.
    bash = shutil.which("bash")
    tail = shutil.which("tail")
    if bash is None or tail is None or not Path("/proc").exists():
        pytest.skip("bash, tail, and /proc are required")
    repo = tmp_path / "staging"
    repo.mkdir()

    leader = subprocess.Popen(
        [bash, "-c", '"$0" -f /dev/null >/dev/null 2>&1 & echo "$!"', tail],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        text=True,
    )
    stdout, _ = leader.communicate(timeout=5)
    orphan_pid = int(stdout.strip())
    assert leader.wait(timeout=5) == 0  # Leader is dead; the orphan lives on.
    _wait_until_exec(orphan_pid, basename=b"tail")

    try:
        reaped = await orchestrator_module._reap_orphaned_validation_processes(  # pyright: ignore[reportPrivateUsage]
            repo,
            leader.pid,
            grace_seconds=0.2,
        )

        assert orphan_pid in reaped
        assert _wait_until_dead(orphan_pid)
    finally:
        try:
            os.kill(orphan_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


async def test_reap_kills_orphan_that_forks_replacement_on_term(tmp_path: Path) -> None:
    # Finding-2 regression: a surviving group member whose SIGTERM handler
    # forks a replacement child (one generation suffices to prove the loop).
    # Both generations must be dead once the reaper returns.
    bash = shutil.which("bash")
    if bash is None or not Path("/proc").exists():
        pytest.skip("bash and /proc are required")
    repo = tmp_path / "staging"
    repo.mkdir()
    helper = tmp_path / "regenerating_helper.py"
    helper.write_text(
        "import os\n"
        "import signal\n"
        "import sys\n"
        "import time\n"
        "\n"
        "out_dir = sys.argv[1]\n"
        "\n"
        "\n"
        "def handler(signum, frame):\n"
        "    pid = os.fork()\n"
        "    if pid == 0:\n"
        '        os.execvp("sleep", ["sleep", "3600"])\n'
        '    with open(os.path.join(out_dir, "replacement.pid"), "w") as f:\n'
        "        f.write(str(pid))\n"
        "\n"
        "\n"
        "signal.signal(signal.SIGTERM, handler)\n"
        'with open(os.path.join(out_dir, "helper.ready"), "w") as f:\n'
        "    f.write(str(os.getpid()))\n"
        "while True:\n"
        "    time.sleep(60)\n",
        encoding="utf-8",
    )

    leader_argv = [
        bash,
        "-c",
        '"$0" "$1" "$2" >/dev/null 2>&1 & echo "$!"',
        sys.executable,
        str(helper),
        str(tmp_path),
    ]
    leader = subprocess.Popen(
        leader_argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        text=True,
    )
    stdout, _ = leader.communicate(timeout=5)
    helper_pid = int(stdout.strip())
    assert leader.wait(timeout=5) == 0
    assert int(_wait_for_file(tmp_path / "helper.ready")) == helper_pid

    replacement_pid: int | None = None
    try:
        reaped = await orchestrator_module._reap_orphaned_validation_processes(  # pyright: ignore[reportPrivateUsage]
            repo,
            leader.pid,
            grace_seconds=0.2,
        )
        replacement_pid = int(_wait_for_file(tmp_path / "replacement.pid"))

        assert helper_pid in reaped
        assert replacement_pid in reaped
        assert _wait_until_dead(helper_pid)
        assert _wait_until_dead(replacement_pid)
    finally:
        for pid in (helper_pid, replacement_pid):
            if pid is None:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


async def test_validation_timeout_reaps_setpgid_term_ignoring_child(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # End-to-end through the real timeout path, deterministically exercising
    # reaper delivery: the background child moves itself into its OWN process
    # group (same session), so the ordinary killpg escalation cannot reach it
    # regardless of scheduler timing — only the reaper, via the session-id arm
    # of the ownership gate, can kill it. It also ignores SIGTERM so both the
    # reaper's SIGTERM and SIGKILL deliveries must appear in the log.
    bash = shutil.which("bash")
    if bash is None or not Path("/proc").exists():
        pytest.skip("bash and /proc are required")
    repo = tmp_path / "staging"
    repo.mkdir()

    child_code = (
        "import os, signal, time\n"
        "os.setpgid(0, 0)\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "while True:\n"
        "    time.sleep(60)\n"
    )
    script = '"$0" -c "$1" >/dev/null 2>&1 & echo "child=$!"; exec sleep 60'
    command = AsyncOrchestratorValidationCommandConfig(
        argv=(bash, "-c", script, sys.executable, child_code),
        timeout_seconds=2.0,
    )

    with caplog.at_level("WARNING", logger=orchestrator_module.__name__):
        failure = await orchestrator_module._run_validation_commands_async(  # pyright: ignore[reportPrivateUsage]
            [command],
            cwd=repo,
        )

    assert failure is not None
    assert "timed out" in failure.error
    match = re.search(r"child=(\d+)", failure.stdout)
    assert match is not None
    child_pid = int(match.group(1))
    try:
        assert _wait_until_dead(child_pid)
        assert f"pid={child_pid} repo={repo} sig=SIGTERM" in caplog.text
        assert f"pid={child_pid} repo={repo} sig=SIGKILL" in caplog.text
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
