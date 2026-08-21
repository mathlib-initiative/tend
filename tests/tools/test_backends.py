from __future__ import annotations

from collections.abc import Mapping
from fnmatch import fnmatch
from pathlib import Path

import pytest

from tend.agent.tools.backends import (
    DirectoryEntry,
    FileStat,
    FilesystemBackend,
    ProcessBackend,
    ProcessResult,
    ToolPath,
)
from tend.agent.tools.local_backend import LocalFilesystemBackend, LocalProcessBackend


class InMemoryFilesystemBackend:
    __slots__ = ("files",)

    files: dict[str, bytes]

    def __init__(self, files: Mapping[str, bytes] | None = None) -> None:
        self.files = dict(files or {})

    async def list_dir(self, path: ToolPath) -> tuple[DirectoryEntry, ...]:
        key = self._key(path)
        prefix = "" if key == "." else f"{key.rstrip('/')}/"
        children: dict[str, DirectoryEntry] = {}
        for file_path, data in self.files.items():
            if not file_path.startswith(prefix):
                continue
            remainder = file_path[len(prefix) :]
            if not remainder:
                continue
            child_name = remainder.split("/", 1)[0]
            child_path = child_name if key == "." else f"{prefix}{child_name}"
            is_nested = "/" in remainder
            existing = children.get(child_name)
            if existing is not None and existing.is_dir:
                continue
            children[child_name] = DirectoryEntry(
                name=child_name,
                path=child_path,
                is_file=not is_nested,
                is_dir=is_nested,
                is_symlink=False,
                size_bytes=None if is_nested else len(data),
            )
        return tuple(children[name] for name in sorted(children))

    async def read_bytes(self, path: ToolPath) -> bytes:
        return self.files[self._key(path)]

    async def read_text(self, path: ToolPath, *, encoding: str = "utf-8") -> str:
        return (await self.read_bytes(path)).decode(encoding)

    async def write_bytes(
        self,
        path: ToolPath,
        data: bytes,
        *,
        create_parents: bool = False,
    ) -> None:
        _ = create_parents
        self.files[self._key(path)] = data

    async def write_text(
        self,
        path: ToolPath,
        text: str,
        *,
        encoding: str = "utf-8",
        create_parents: bool = False,
    ) -> None:
        await self.write_bytes(path, text.encode(encoding), create_parents=create_parents)

    async def stat(self, path: ToolPath) -> FileStat:
        key = self._key(path)
        if key in self.files:
            return FileStat(
                path=key,
                is_file=True,
                is_dir=False,
                is_symlink=False,
                size_bytes=len(self.files[key]),
            )
        prefix = "" if key == "." else f"{key.rstrip('/')}/"
        if any(file_path.startswith(prefix) for file_path in self.files):
            return FileStat(
                path=key,
                is_file=False,
                is_dir=True,
                is_symlink=False,
                size_bytes=None,
            )
        raise FileNotFoundError(key)

    async def glob(self, pattern: str, *, root: ToolPath = ".") -> tuple[str, ...]:
        root_key = self._key(root)
        prefix = "" if root_key == "." else f"{root_key.rstrip('/')}/"
        full_pattern = pattern if root_key == "." else f"{prefix}{pattern}"
        return tuple(sorted(path for path in self.files if fnmatch(path, full_pattern)))

    def _key(self, path: ToolPath) -> str:
        text = Path(path).as_posix()
        return "." if text in {"", "."} else text


class ScriptedProcessBackend:
    __slots__ = ("calls", "results")

    calls: list[tuple[str, str, float | None]]
    results: dict[str, ProcessResult]

    def __init__(self, results: Mapping[str, ProcessResult]) -> None:
        self.results = dict(results)
        self.calls = []

    async def run(
        self,
        command: str,
        *,
        cwd: ToolPath = ".",
        timeout_seconds: float | None = None,
    ) -> ProcessResult:
        self.calls.append((command, str(cwd), timeout_seconds))
        return self.results[command]


async def test_fake_backends_implement_protocols_and_are_injectable() -> None:
    filesystem = InMemoryFilesystemBackend({"a.txt": b"alpha", "dir/b.txt": b"beta"})
    filesystem_backend: FilesystemBackend = filesystem

    assert isinstance(filesystem_backend, FilesystemBackend)
    assert [entry.name for entry in await filesystem_backend.list_dir(".")] == ["a.txt", "dir"]
    assert await filesystem_backend.read_text("a.txt") == "alpha"

    await filesystem_backend.write_text("dir/c.txt", "gamma", create_parents=True)
    assert await filesystem_backend.read_bytes("dir/c.txt") == b"gamma"
    assert (await filesystem_backend.stat("dir")).is_dir is True
    assert await filesystem_backend.glob("dir/*.txt") == ("dir/b.txt", "dir/c.txt")

    scripted_result = ProcessResult(command="echo ok", cwd=".", exit_code=0, stdout="ok\n")
    process = ScriptedProcessBackend({"echo ok": scripted_result})
    process_backend: ProcessBackend = process

    assert isinstance(process_backend, ProcessBackend)
    result = await process_backend.run("echo ok", cwd="sandbox", timeout_seconds=1.5)
    assert result == scripted_result
    assert process.calls == [("echo ok", "sandbox", 1.5)]


async def test_local_filesystem_backend_operates_on_temporary_directories(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    backend = LocalFilesystemBackend(cwd=work)

    await backend.write_text("nested/file.txt", "hello", create_parents=True)
    await backend.write_bytes("nested/data.bin", b"data")

    assert await backend.read_text("nested/file.txt") == "hello"
    assert await backend.read_bytes("nested/data.bin") == b"data"

    listing = await backend.list_dir("nested")
    assert [entry.name for entry in listing] == ["data.bin", "file.txt"]
    assert all(entry.is_file for entry in listing)

    stat = await backend.stat("nested/file.txt")
    assert stat.is_file is True
    assert stat.size_bytes == len(b"hello")

    glob_matches = await backend.glob("**/*.txt")
    assert glob_matches == (str(work / "nested" / "file.txt"),)


async def test_local_filesystem_backend_does_not_enforce_path_filtering(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    backend = LocalFilesystemBackend(cwd=work)

    assert await backend.read_text("../outside.txt") == "outside"

    await backend.write_text("../created.txt", "created")
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "created"


async def test_local_process_backend_captures_shell_results(tmp_path: Path) -> None:
    backend = LocalProcessBackend(cwd=tmp_path)

    result = await backend.run("printf 'out'; printf 'err' >&2", timeout_seconds=5)

    assert result.command == "printf 'out'; printf 'err' >&2"
    assert result.cwd == str(tmp_path)
    assert result.exit_code == 0
    assert result.stdout == "out"
    assert result.stderr == "err"
    assert result.timed_out is False
    assert result.duration_ms is not None


async def test_local_process_backend_does_not_filter_shell_commands(tmp_path: Path) -> None:
    backend = LocalProcessBackend(cwd=tmp_path)

    result = await backend.run("printf left && printf right", timeout_seconds=5)

    assert result.exit_code == 0
    assert result.stdout == "leftright"


async def test_local_process_backend_timeout_is_reliability_plumbing(tmp_path: Path) -> None:
    backend = LocalProcessBackend(cwd=tmp_path)

    result = await backend.run("sleep 5", timeout_seconds=0.05)

    assert result.timed_out is True
    assert result.exit_code is not None


async def test_local_process_backend_rejects_negative_timeout_only(tmp_path: Path) -> None:
    backend = LocalProcessBackend(cwd=tmp_path)

    with pytest.raises(ValueError, match="timeout_seconds"):
        await backend.run("printf ok", timeout_seconds=-1)
