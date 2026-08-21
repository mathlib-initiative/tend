from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tend._common.types import JsonObject
from tend.agent.tools import ToolContext, get_builtin_tool
from tend.agent.tools.backends import DirectoryEntry, FileStat, ToolPath
from tend.agent.tools.builtin.copy_lines import CopyLinesResult


async def _copy(tmp_path: Path, args: JsonObject) -> CopyLinesResult:
    tool = get_builtin_tool("copy_lines")
    result = await tool.run(ToolContext(cwd=tmp_path), tool.validate_arguments(args))
    assert isinstance(result, CopyLinesResult)
    return result


async def test_copy_lines_copies_block_byte_exact(tmp_path: Path) -> None:
    (tmp_path / "src.lean").write_text(
        "A1\nA2\nCOPY_a\nCOPY_b\nCOPY_c\nA6\n", encoding="utf-8"
    )
    (tmp_path / "dst.lean").write_text("B1\nB2\nB3\n", encoding="utf-8")

    result = await _copy(
        tmp_path,
        {
            "source_path": "src.lean",
            "start_line": 3,
            "end_line": 5,
            "dest_path": "dst.lean",
            "dest_after_line": 2,
        },
    )

    assert result.success is True
    assert result.error is None
    assert result.lines_copied == 3
    # source is untouched; removal is left to edit_file/follow-up edits
    assert (tmp_path / "src.lean").read_text(encoding="utf-8") == (
        "A1\nA2\nCOPY_a\nCOPY_b\nCOPY_c\nA6\n"
    )
    # dest has the block inserted after line 2, byte-exact
    assert (tmp_path / "dst.lean").read_text(encoding="utf-8") == (
        "B1\nB2\nCOPY_a\nCOPY_b\nCOPY_c\nB3\n"
    )


async def test_copy_lines_append_to_end(tmp_path: Path) -> None:
    (tmp_path / "src.txt").write_text("keep\ncopy\n", encoding="utf-8")
    (tmp_path / "dst.txt").write_text("a\nb\n", encoding="utf-8")

    result = await _copy(
        tmp_path,
        {
            "source_path": "src.txt",
            "start_line": 2,
            "end_line": 2,
            "dest_path": "dst.txt",
            "dest_after_line": 2,  # == line count → append
        },
    )

    assert result.success is True
    assert (tmp_path / "dst.txt").read_text(encoding="utf-8") == "a\nb\ncopy\n"
    assert (tmp_path / "src.txt").read_text(encoding="utf-8") == "keep\ncopy\n"


async def test_copy_lines_no_trailing_newline_preserved(tmp_path: Path) -> None:
    (tmp_path / "src.txt").write_text("p\nq", encoding="utf-8")  # no final newline
    (tmp_path / "dst.txt").write_text("d\n", encoding="utf-8")

    result = await _copy(
        tmp_path,
        {
            "source_path": "src.txt",
            "start_line": 1,
            "end_line": 1,
            "dest_path": "dst.txt",
            "dest_after_line": 0,  # prepend
        },
    )

    assert result.success is True
    assert (tmp_path / "src.txt").read_text(encoding="utf-8") == "p\nq"
    assert (tmp_path / "dst.txt").read_text(encoding="utf-8") == "p\nd\n"


async def test_copy_lines_rejects_same_file(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("a\nb\n", encoding="utf-8")
    result = await _copy(
        tmp_path,
        {"source_path": "f.txt", "start_line": 1, "end_line": 1, "dest_path": "f.txt"},
    )
    assert result.success is False
    assert result.error is not None
    assert result.error.error_type == "same_file"
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "a\nb\n"  # untouched


async def test_copy_lines_rejects_same_file_alias(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("a\nb\n", encoding="utf-8")
    result = await _copy(
        tmp_path,
        {"source_path": "f.txt", "start_line": 1, "end_line": 1, "dest_path": "./f.txt"},
    )
    assert result.success is False
    assert result.error is not None
    assert result.error.error_type == "same_file"
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "a\nb\n"  # untouched


async def test_copy_lines_range_out_of_bounds(tmp_path: Path) -> None:
    (tmp_path / "src.txt").write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "dst.txt").write_text("z\n", encoding="utf-8")
    result = await _copy(
        tmp_path,
        {"source_path": "src.txt", "start_line": 1, "end_line": 9, "dest_path": "dst.txt"},
    )
    assert result.success is False
    assert result.error is not None
    assert result.error.error_type == "range_out_of_bounds"
    # both files untouched on validation failure
    assert (tmp_path / "src.txt").read_text(encoding="utf-8") == "a\nb\n"
    assert (tmp_path / "dst.txt").read_text(encoding="utf-8") == "z\n"


async def test_copy_lines_dest_after_out_of_bounds(tmp_path: Path) -> None:
    (tmp_path / "src.txt").write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "dst.txt").write_text("z\n", encoding="utf-8")
    result = await _copy(
        tmp_path,
        {
            "source_path": "src.txt",
            "start_line": 1,
            "end_line": 1,
            "dest_path": "dst.txt",
            "dest_after_line": 5,
        },
    )
    assert result.success is False
    assert result.error is not None
    assert result.error.error_type == "dest_line_out_of_bounds"
    assert (tmp_path / "src.txt").read_text(encoding="utf-8") == "a\nb\n"
    assert (tmp_path / "dst.txt").read_text(encoding="utf-8") == "z\n"


async def test_copy_lines_source_not_found(tmp_path: Path) -> None:
    (tmp_path / "dst.txt").write_text("z\n", encoding="utf-8")
    result = await _copy(
        tmp_path,
        {"source_path": "nope.txt", "start_line": 1, "end_line": 1, "dest_path": "dst.txt"},
    )
    assert result.success is False
    assert result.error is not None
    assert result.error.error_type == "source_not_found"


async def test_copy_lines_source_untouched_on_destination_write_failure() -> None:
    tool = get_builtin_tool("copy_lines")
    backend = FailingDestinationWriteBackend()

    result = await tool.run(
        ToolContext(filesystem_backend=backend),
        tool.validate_arguments(
            {
                "source_path": "src.txt",
                "start_line": 2,
                "end_line": 2,
                "dest_path": "dst.txt",
                "dest_after_line": 1,
            }
        ),
    )

    assert isinstance(result, CopyLinesResult)
    assert result.success is False
    assert result.error is not None
    assert result.error.error_type == "write_error"
    assert backend.files["src.txt"] == b"a\nb\nc\n"
    assert backend.files["dst.txt"] == b"x\ny\n"


@pytest.mark.parametrize(
    "arguments",
    (
        {"source_path": "a", "start_line": 5, "end_line": 2, "dest_path": "b"},
        {
            "source_path": "a",
            "start_line": 1,
            "end_line": 1,
            "dest_path": "b",
            "cut": True,
        },
    ),
)
def test_copy_lines_invalid_arguments_fail_at_tool_validation_layer(arguments: JsonObject) -> None:
    tool = get_builtin_tool("copy_lines")
    with pytest.raises(ValidationError):
        tool.validate_arguments(arguments)


class FailingDestinationWriteBackend:
    files: dict[str, bytes]

    def __init__(self) -> None:
        self.files = {"src.txt": b"a\nb\nc\n", "dst.txt": b"x\ny\n"}

    async def list_dir(self, path: ToolPath) -> tuple[DirectoryEntry, ...]:
        raise NotImplementedError

    async def read_bytes(self, path: ToolPath) -> bytes:
        return self.files[str(path)]

    async def read_text(self, path: ToolPath, *, encoding: str = "utf-8") -> str:
        return (await self.read_bytes(path)).decode(encoding)

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
        data = self.files[str(path)]
        return FileStat(
            path=str(path),
            is_file=True,
            is_dir=False,
            is_symlink=False,
            size_bytes=len(data),
        )

    async def glob(self, pattern: str, *, root: ToolPath = ".") -> tuple[str, ...]:
        raise NotImplementedError
