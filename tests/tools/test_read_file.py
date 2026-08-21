from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tend._common.types import JsonObject
from tend.agent.tools import ToolContext, get_builtin_tool
from tend.agent.tools.builtin.read_file import ReadFileResult


async def test_read_file_reads_full_utf8_text(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("alpha\nbeta\n", encoding="utf-8")

    tool = get_builtin_tool("read_file")
    arguments = tool.validate_arguments({"path": "notes.txt"})
    result = await tool.run(ToolContext(cwd=tmp_path), arguments)

    assert isinstance(result, ReadFileResult)
    assert result.output == "alpha\nbeta"
    assert result.size_bytes == len(b"alpha\nbeta\n")
    assert result.total_lines == 2
    assert result.returned_lines == 2
    assert result.start_line == 1
    assert result.end_line == 2
    assert result.has_more is False
    assert result.continuation_offset is None
    assert result.omitted is False
    assert result.truncated is False


async def test_read_file_supports_line_pagination(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

    tool = get_builtin_tool("read_file")
    result = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments({"path": "notes.txt", "offset": 2, "limit": 2}),
    )

    assert isinstance(result, ReadFileResult)
    assert result.output == "two\nthree"
    assert result.total_lines == 4
    assert result.returned_lines == 2
    assert result.start_line == 2
    assert result.end_line == 3
    assert result.has_more is True
    assert result.continuation_offset == 4
    assert result.limit == 2


async def test_read_file_omits_binary_and_non_utf8_content(tmp_path: Path) -> None:
    (tmp_path / "nul.bin").write_bytes(b"abc\x00def")
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe")

    tool = get_builtin_tool("read_file")
    binary_result = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments({"path": "nul.bin"}),
    )
    non_utf8_result = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments({"path": "bad.txt"}),
    )

    assert isinstance(binary_result, ReadFileResult)
    assert binary_result.omitted is True
    assert binary_result.omission_reason == "binary"
    assert "Binary file omitted" in binary_result.output
    assert "\x00" not in binary_result.output

    assert isinstance(non_utf8_result, ReadFileResult)
    assert non_utf8_result.omitted is True
    assert non_utf8_result.omission_reason == "non_utf8"
    assert "Non-UTF-8 file omitted" in non_utf8_result.output
    assert "\ufffd" not in non_utf8_result.output


async def test_read_file_reports_missing_file_as_model_visible_error(tmp_path: Path) -> None:
    tool = get_builtin_tool("read_file")

    result = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments({"path": "missing.txt"}),
    )

    assert isinstance(result, ReadFileResult)
    assert result.omitted is True
    assert result.omission_reason == "not_found"
    assert result.output == "[File read error: file not found: missing.txt]"
    assert result.returned_lines == 0


async def test_read_file_reports_offset_beyond_eof_without_reading_raw_bytes(
    tmp_path: Path,
) -> None:
    (tmp_path / "notes.txt").write_text("one\ntwo\n", encoding="utf-8")
    tool = get_builtin_tool("read_file")

    result = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments({"path": "notes.txt", "offset": 5}),
    )

    assert isinstance(result, ReadFileResult)
    assert result.omitted is True
    assert result.omission_reason == "offset_out_of_range"
    assert "offset 5 is beyond EOF" in result.output
    assert result.total_lines == 2


async def test_read_file_uses_head_truncation_metadata(tmp_path: Path) -> None:
    (tmp_path / "long.txt").write_text("abcdef\nghijkl\n", encoding="utf-8")
    tool = get_builtin_tool("read_file")

    result = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments({"path": "long.txt", "max_output_bytes": 5}),
    )

    assert isinstance(result, ReadFileResult)
    assert result.truncated is True
    assert result.truncation is not None
    assert result.truncation.policy == "head"
    assert result.output.startswith("abcde")
    assert "[Output truncated:" in result.output
    assert result.has_more is True


@pytest.mark.parametrize(
    "arguments",
    (
        {"path": "notes.txt", "unexpected": True},
        {"path": "notes.txt", "offset": 0},
        {"path": "notes.txt", "limit": 0},
        {"path": "notes.txt", "max_output_bytes": 0},
    ),
)
def test_read_file_invalid_arguments_fail_at_tool_validation_layer(arguments: JsonObject) -> None:
    tool = get_builtin_tool("read_file")

    with pytest.raises(ValidationError):
        tool.validate_arguments(arguments)
