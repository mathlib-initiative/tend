from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tend._common.types import JsonObject
from tend.agent.tools import ToolContext, get_builtin_tool
from tend.agent.tools.backends import DirectoryEntry, FileStat, ToolPath
from tend.agent.tools.builtin.write_file import WriteFileResult


async def test_write_file_creates_new_utf8_file(tmp_path: Path) -> None:
    tool = get_builtin_tool("write_file")

    result = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments({"path": "notes.txt", "content": "hé\n"}),
    )

    assert isinstance(result, WriteFileResult)
    assert result.success is True
    assert result.error is None
    assert result.path == "notes.txt"
    assert result.bytes_written == len("hé\n".encode())
    assert result.chars_written == len("hé\n")
    assert result.overwritten is False
    assert result.output == "Wrote 4 bytes (3 characters) to notes.txt."
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hé\n"


async def test_write_file_creates_parent_directories_by_default(tmp_path: Path) -> None:
    tool = get_builtin_tool("write_file")

    result = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments({"path": "nested/deep/file.txt", "content": "created"}),
    )

    assert isinstance(result, WriteFileResult)
    assert result.success is True
    assert result.create_parents is True
    assert (tmp_path / "nested" / "deep" / "file.txt").read_text(encoding="utf-8") == "created"


async def test_write_file_overwrites_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("old", encoding="utf-8")
    tool = get_builtin_tool("write_file")

    result = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments({"path": "notes.txt", "content": "new"}),
    )

    assert isinstance(result, WriteFileResult)
    assert result.success is True
    assert result.overwritten is True
    assert result.bytes_written == 3
    assert target.read_text(encoding="utf-8") == "new"


async def test_write_file_reports_missing_parent_when_parent_creation_disabled(
    tmp_path: Path,
) -> None:
    tool = get_builtin_tool("write_file")

    result = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments(
            {"path": "missing_parent/file.txt", "content": "nope", "create_parents": False}
        ),
    )

    assert isinstance(result, WriteFileResult)
    assert result.success is False
    assert result.error is not None
    assert result.error.error_type == "parent_not_found"
    assert "FileNotFoundError" in result.error.message
    assert "[File write error:" in result.output
    assert not (tmp_path / "missing_parent").exists()


async def test_write_file_backend_failure_becomes_structured_error() -> None:
    tool = get_builtin_tool("write_file")

    result = await tool.run(
        ToolContext(filesystem_backend=FailingWriteBackend()),
        tool.validate_arguments({"path": "out.txt", "content": "data"}),
    )

    assert isinstance(result, WriteFileResult)
    assert result.success is False
    assert result.error is not None
    assert result.error.error_type == "write_error"
    assert "disk full" in result.error.message
    assert result.error.details["path"] == "out.txt"
    assert result.bytes_written == 0
    assert result.chars_written == 0


@pytest.mark.parametrize(
    "arguments",
    (
        {"content": "missing path"},
        {"path": "", "content": "empty path"},
        {"path": "out.txt", "content": "ok", "unexpected": True},
        {"path": "out.txt", "content": 1},
        {"path": "out.txt", "content": "ok", "create_parents": "yes"},
    ),
)
def test_write_file_invalid_arguments_fail_at_tool_validation_layer(arguments: JsonObject) -> None:
    tool = get_builtin_tool("write_file")

    with pytest.raises(ValidationError):
        tool.validate_arguments(arguments)


class FailingWriteBackend:
    async def list_dir(self, path: ToolPath) -> tuple[DirectoryEntry, ...]:
        raise NotImplementedError

    async def read_bytes(self, path: ToolPath) -> bytes:
        raise NotImplementedError

    async def read_text(self, path: ToolPath, *, encoding: str = "utf-8") -> str:
        raise NotImplementedError

    async def write_bytes(
        self,
        path: ToolPath,
        data: bytes,
        *,
        create_parents: bool = False,
    ) -> None:
        raise OSError("disk full")

    async def write_text(
        self,
        path: ToolPath,
        text: str,
        *,
        encoding: str = "utf-8",
        create_parents: bool = False,
    ) -> None:
        raise OSError("disk full")

    async def stat(self, path: ToolPath) -> FileStat:
        raise FileNotFoundError(str(path))

    async def glob(self, pattern: str, *, root: ToolPath = ".") -> tuple[str, ...]:
        raise NotImplementedError
