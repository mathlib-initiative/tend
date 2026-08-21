"""Tool operation backend protocols and shared result models."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Protocol, runtime_checkable

from pydantic import Field

from tend._common.types import StrictModel

type ToolPath = str | Path

_NonNegativeInt = Annotated[int, Field(ge=0)]
_NonNegativeFloat = Annotated[float, Field(ge=0)]


class FileStat(StrictModel):
    """Small provider-neutral-ish file stat record used by tool backends."""

    path: str = Field(min_length=1)
    is_file: bool
    is_dir: bool
    is_symlink: bool
    size_bytes: _NonNegativeInt | None = None


class DirectoryEntry(FileStat):
    """Directory listing entry returned by filesystem backends."""

    name: str = Field(min_length=1)


class ProcessResult(StrictModel):
    """Captured shell process result returned by process backends."""

    command: str
    cwd: str = Field(min_length=1)
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_ms: _NonNegativeFloat | None = None


@runtime_checkable
class FilesystemBackend(Protocol):
    """Async filesystem operations needed by built-in file/search tools.

    The protocol is intentionally about operation plumbing only. Implementations
    must not add sandbox policy such as path allowlists or extension rules; the
    process/orchestration sandbox boundary owns those restrictions.
    """

    async def list_dir(self, path: ToolPath) -> tuple[DirectoryEntry, ...]:
        """List one directory without recursive traversal."""
        ...

    async def read_bytes(self, path: ToolPath) -> bytes:
        """Read raw bytes from a file."""
        ...

    async def read_text(self, path: ToolPath, *, encoding: str = "utf-8") -> str:
        """Read text from a file using the requested encoding."""
        ...

    async def write_bytes(
        self,
        path: ToolPath,
        data: bytes,
        *,
        create_parents: bool = False,
    ) -> None:
        """Write raw bytes to a file."""
        ...

    async def write_text(
        self,
        path: ToolPath,
        text: str,
        *,
        encoding: str = "utf-8",
        create_parents: bool = False,
    ) -> None:
        """Write text to a file using the requested encoding."""
        ...

    async def stat(self, path: ToolPath) -> FileStat:
        """Return a small stat record for a file or directory."""
        ...

    async def glob(self, pattern: str, *, root: ToolPath = ".") -> tuple[str, ...]:
        """Return paths matching a glob pattern in deterministic order."""
        ...


@runtime_checkable
class ProcessBackend(Protocol):
    """Async shell process execution needed by the built-in bash tool.

    Implementations capture stdout/stderr and enforce reliability timeouts only.
    They must not filter commands or implement network/filesystem restrictions.
    """

    async def run(
        self,
        command: str,
        *,
        cwd: ToolPath = ".",
        timeout_seconds: float | None = None,
    ) -> ProcessResult:
        """Run a shell command and capture stdout/stderr."""
        ...


__all__ = (
    "DirectoryEntry",
    "FileStat",
    "FilesystemBackend",
    "ProcessBackend",
    "ProcessResult",
    "ToolPath",
)
