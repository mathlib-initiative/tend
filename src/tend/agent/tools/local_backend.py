"""Local filesystem and process backend implementations."""

from __future__ import annotations

import asyncio
import glob as glob_module
import os
import signal
import time
from contextlib import suppress
from pathlib import Path

from tend.agent.tools.backends import DirectoryEntry, FileStat, ProcessResult, ToolPath

_PROCESS_TERMINATION_GRACE_SECONDS = 5.0


class LocalFilesystemBackend:
    """Local filesystem backend with reliability plumbing only.

    Relative paths are interpreted under ``cwd`` for convenience. The backend
    deliberately does not resolve paths into an allowlist, block ``..`` segments,
    restrict file extensions, or otherwise duplicate sandbox policy.
    """

    __slots__ = ("cwd",)

    cwd: Path

    def __init__(self, *, cwd: ToolPath = ".") -> None:
        self.cwd = Path(cwd)

    async def list_dir(self, path: ToolPath) -> tuple[DirectoryEntry, ...]:
        """List one directory in deterministic name order."""

        return await asyncio.to_thread(self._list_dir_sync, path)

    async def read_bytes(self, path: ToolPath) -> bytes:
        """Read raw bytes from the local filesystem."""

        return await asyncio.to_thread(self._path(path).read_bytes)

    async def read_text(self, path: ToolPath, *, encoding: str = "utf-8") -> str:
        """Read text from the local filesystem."""

        return await asyncio.to_thread(self._path(path).read_text, encoding=encoding)

    async def write_bytes(
        self,
        path: ToolPath,
        data: bytes,
        *,
        create_parents: bool = False,
    ) -> None:
        """Write raw bytes to the local filesystem."""

        await asyncio.to_thread(self._write_bytes_sync, path, data, create_parents)

    async def write_text(
        self,
        path: ToolPath,
        text: str,
        *,
        encoding: str = "utf-8",
        create_parents: bool = False,
    ) -> None:
        """Write text to the local filesystem."""

        await asyncio.to_thread(self._write_text_sync, path, text, encoding, create_parents)

    async def stat(self, path: ToolPath) -> FileStat:
        """Return a small stat record for a local path."""

        return await asyncio.to_thread(self._stat_sync, path)

    async def glob(self, pattern: str, *, root: ToolPath = ".") -> tuple[str, ...]:
        """Return local glob matches in deterministic order."""

        return await asyncio.to_thread(self._glob_sync, pattern, root)

    def _path(self, path: ToolPath) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return self.cwd / candidate

    def _list_dir_sync(self, path: ToolPath) -> tuple[DirectoryEntry, ...]:
        directory = self._path(path)
        entries: list[DirectoryEntry] = []
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            stat = _stat_path(child)
            entries.append(
                DirectoryEntry(
                    name=child.name,
                    path=stat.path,
                    is_file=stat.is_file,
                    is_dir=stat.is_dir,
                    is_symlink=stat.is_symlink,
                    size_bytes=stat.size_bytes,
                )
            )
        return tuple(entries)

    def _write_bytes_sync(self, path: ToolPath, data: bytes, create_parents: bool) -> None:
        target = self._path(path)
        if create_parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def _write_text_sync(
        self,
        path: ToolPath,
        text: str,
        encoding: str,
        create_parents: bool,
    ) -> None:
        target = self._path(path)
        if create_parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding=encoding)

    def _stat_sync(self, path: ToolPath) -> FileStat:
        return _stat_path(self._path(path))

    def _glob_sync(self, pattern: str, root: ToolPath) -> tuple[str, ...]:
        pattern_path = Path(pattern)
        if pattern_path.is_absolute():
            search_pattern = str(pattern_path)
        else:
            search_pattern = str(self._path(root) / pattern)
        return tuple(sorted(glob_module.glob(search_pattern, recursive=True)))


class LocalProcessBackend:
    """Local shell process backend with timeout/capture reliability plumbing only.

    Commands are passed directly to the platform shell. This class intentionally
    does not inspect command syntax, block network-capable commands, or enforce a
    path policy; those controls belong to the process/orchestration sandbox boundary.
    """

    __slots__ = ("cwd", "encoding")

    cwd: Path
    encoding: str

    def __init__(self, *, cwd: ToolPath = ".", encoding: str = "utf-8") -> None:
        self.cwd = Path(cwd)
        self.encoding = encoding

    async def run(
        self,
        command: str,
        *,
        cwd: ToolPath = ".",
        timeout_seconds: float | None = None,
    ) -> ProcessResult:
        """Run a shell command and capture stdout/stderr."""

        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")

        run_cwd = self._path(cwd)
        started = time.monotonic()
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(run_cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        timed_out = False
        communicate_task = asyncio.create_task(process.communicate())
        try:
            if timeout_seconds is None:
                stdout_bytes, stderr_bytes = await communicate_task
            else:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    asyncio.shield(communicate_task),
                    timeout=timeout_seconds,
                )
        except TimeoutError:
            timed_out = True
            _kill_process_group_or_process(process)
            stdout_bytes, stderr_bytes = await communicate_task
        except asyncio.CancelledError:
            await _terminate_process_group_or_process(process)
            communicate_task.cancel()
            with suppress(asyncio.CancelledError):
                await communicate_task
            raise

        duration_ms = (time.monotonic() - started) * 1000
        return ProcessResult(
            command=command,
            cwd=str(run_cwd),
            exit_code=process.returncode,
            stdout=stdout_bytes.decode(self.encoding, errors="replace"),
            stderr=stderr_bytes.decode(self.encoding, errors="replace"),
            timed_out=timed_out,
            duration_ms=duration_ms,
        )

    def _path(self, path: ToolPath) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return self.cwd / candidate


def _stat_path(path: Path) -> FileStat:
    stat_result = path.lstat()
    return FileStat(
        path=str(path),
        is_file=path.is_file(),
        is_dir=path.is_dir(),
        is_symlink=path.is_symlink(),
        size_bytes=stat_result.st_size,
    )


async def _terminate_process_group_or_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    _signal_process_group_or_process(process, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=_PROCESS_TERMINATION_GRACE_SECONDS)
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


__all__ = ("LocalFilesystemBackend", "LocalProcessBackend")
