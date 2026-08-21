from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from tend._common.types import JsonObject
from tend.agent.tools import ToolContext, get_builtin_tool
from tend.agent.tools.builtin.edit_file import EditFileResult


async def test_edit_file_applies_single_exact_replacement(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("hello world", encoding="utf-8")
    tool = get_builtin_tool("edit_file")

    result = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments(
            {"path": "notes.txt", "edits": [{"old_text": "world", "new_text": "there"}]}
        ),
    )

    assert isinstance(result, EditFileResult)
    assert result.success is True
    assert result.error is None
    assert result.replacement_count == 1
    assert result.bytes_written == len(b"hello there")
    assert result.chars_written == len("hello there")
    assert result.original_size_bytes == len(b"hello world")
    assert result.edited_size_bytes == len(b"hello there")
    assert result.line_ending == "lf"
    assert result.had_utf8_bom is False
    assert result.output == "Edited 1 replacement in notes.txt. Wrote 11 bytes (11 characters)."
    assert target.read_text(encoding="utf-8") == "hello there"


async def test_edit_file_applies_multiple_disjoint_replacements_against_original_order(
    tmp_path: Path,
) -> None:
    target = tmp_path / "config.txt"
    target.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
    tool = get_builtin_tool("edit_file")

    result = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments(
            {
                "path": "config.txt",
                "edits": [
                    {"old_text": "gamma", "new_text": "GAMMA"},
                    {"old_text": "alpha", "new_text": "ALPHA"},
                ],
            }
        ),
    )

    assert isinstance(result, EditFileResult)
    assert result.success is True
    assert result.replacement_count == 2
    assert target.read_text(encoding="utf-8") == "ALPHA\nbeta\nGAMMA\ndelta\n"


async def test_edit_file_matches_all_replacements_against_original_content(
    tmp_path: Path,
) -> None:
    target = tmp_path / "chain.txt"
    target.write_text("alpha", encoding="utf-8")
    tool = get_builtin_tool("edit_file")

    result = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments(
            {
                "path": "chain.txt",
                "edits": [
                    {"old_text": "alpha", "new_text": "beta"},
                    {"old_text": "beta", "new_text": "gamma"},
                ],
            }
        ),
    )

    assert isinstance(result, EditFileResult)
    assert result.success is False
    assert result.error is not None
    assert result.error.error_type == "missing_text"
    assert result.error.details["edit_index"] == 1
    assert "Could not find edits[1].old_text" in result.error.message
    assert target.read_text(encoding="utf-8") == "alpha"


async def test_edit_file_rejects_missing_duplicate_overlap_and_noop_without_writing(
    tmp_path: Path,
) -> None:
    cases = [
        (
            "missing.txt",
            "alpha\nbeta\n",
            [{"old_text": "gamma", "new_text": "GAMMA"}],
            "missing_text",
        ),
        (
            "duplicate.txt",
            "needle\nhay\nneedle\n",
            [{"old_text": "needle", "new_text": "pin"}],
            "duplicate_match",
        ),
        (
            "overlap.txt",
            "abcdef",
            [
                {"old_text": "abc", "new_text": "X"},
                {"old_text": "bcd", "new_text": "Y"},
            ],
            "overlapping_edits",
        ),
        (
            "empty.txt",
            "alpha",
            [{"old_text": "", "new_text": "beta"}],
            "empty_old_text",
        ),
        (
            "noop.txt",
            "alpha",
            [{"old_text": "alpha", "new_text": "alpha"}],
            "no_op",
        ),
    ]
    tool = get_builtin_tool("edit_file")

    for file_name, original, edits, expected_error in cases:
        target = tmp_path / file_name
        target.write_text(original, encoding="utf-8")

        result = await tool.run(
            ToolContext(cwd=tmp_path),
            tool.validate_arguments(cast(JsonObject, {"path": file_name, "edits": edits})),
        )

        assert isinstance(result, EditFileResult)
        assert result.success is False
        assert result.error is not None
        assert result.error.error_type == expected_error
        assert result.output.startswith("[File edit error:")
        assert target.read_text(encoding="utf-8") == original


async def test_edit_file_failure_is_all_or_nothing_for_multiple_edits(tmp_path: Path) -> None:
    target = tmp_path / "all_or_nothing.txt"
    target.write_text("first\nsecond\n", encoding="utf-8")
    tool = get_builtin_tool("edit_file")

    result = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments(
            {
                "path": "all_or_nothing.txt",
                "edits": [
                    {"old_text": "first", "new_text": "FIRST"},
                    {"old_text": "missing", "new_text": "MISSING"},
                ],
            }
        ),
    )

    assert isinstance(result, EditFileResult)
    assert result.success is False
    assert result.error is not None
    assert result.error.error_type == "missing_text"
    assert target.read_text(encoding="utf-8") == "first\nsecond\n"


async def test_edit_file_preserves_crlf_line_endings(tmp_path: Path) -> None:
    target = tmp_path / "windows.txt"
    target.write_bytes(b"alpha\r\nbeta\r\n")
    tool = get_builtin_tool("edit_file")

    result = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments(
            {"path": "windows.txt", "edits": [{"old_text": "beta", "new_text": "gamma"}]}
        ),
    )

    assert isinstance(result, EditFileResult)
    assert result.success is True
    assert result.line_ending == "crlf"
    assert target.read_bytes() == b"alpha\r\ngamma\r\n"


async def test_edit_file_preserves_utf8_bom(tmp_path: Path) -> None:
    target = tmp_path / "bom.txt"
    target.write_bytes(b"\xef\xbb\xbfalpha\nbeta\n")
    tool = get_builtin_tool("edit_file")

    result = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments(
            {"path": "bom.txt", "edits": [{"old_text": "beta", "new_text": "gamma"}]}
        ),
    )

    assert isinstance(result, EditFileResult)
    assert result.success is True
    assert result.had_utf8_bom is True
    data = target.read_bytes()
    assert data.startswith(b"\xef\xbb\xbf")
    assert data.decode("utf-8") == "\ufeffalpha\ngamma\n"


async def test_edit_file_reports_non_utf8_binary_and_missing_files_without_raw_bytes(
    tmp_path: Path,
) -> None:
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe")
    (tmp_path / "nul.bin").write_bytes(b"abc\x00def")
    tool = get_builtin_tool("edit_file")

    non_utf8 = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments(
            {"path": "bad.txt", "edits": [{"old_text": "x", "new_text": "y"}]}
        ),
    )
    binary = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments(
            {"path": "nul.bin", "edits": [{"old_text": "x", "new_text": "y"}]}
        ),
    )
    missing = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments(
            {"path": "missing.txt", "edits": [{"old_text": "x", "new_text": "y"}]}
        ),
    )

    assert isinstance(non_utf8, EditFileResult)
    assert non_utf8.success is False
    assert non_utf8.error is not None
    assert non_utf8.error.error_type == "non_utf8"
    assert "\ufffd" not in non_utf8.output

    assert isinstance(binary, EditFileResult)
    assert binary.success is False
    assert binary.error is not None
    assert binary.error.error_type == "binary"
    assert "\x00" not in binary.output

    assert isinstance(missing, EditFileResult)
    assert missing.success is False
    assert missing.error is not None
    assert missing.error.error_type == "not_found"
    assert "File not found" in missing.error.message


@pytest.mark.parametrize(
    "arguments",
    (
        {"edits": [{"old_text": "a", "new_text": "b"}]},
        {"path": "", "edits": [{"old_text": "a", "new_text": "b"}]},
        {"path": "notes.txt", "edits": []},
        {"path": "notes.txt", "edits": [{"old_text": "a"}]},
        {"path": "notes.txt", "edits": [{"old_text": "a", "new_text": "b", "extra": True}]},
        {"path": "notes.txt", "edits": [{"old_text": 1, "new_text": "b"}]},
        {"path": "notes.txt", "edits": [{"old_text": "a", "new_text": "b"}], "extra": True},
    ),
)
def test_edit_file_invalid_arguments_fail_at_tool_validation_layer(arguments: JsonObject) -> None:
    tool = get_builtin_tool("edit_file")

    with pytest.raises(ValidationError):
        tool.validate_arguments(arguments)
