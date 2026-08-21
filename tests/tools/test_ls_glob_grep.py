from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tend._common.types import JsonObject
from tend.agent.tools import ToolContext, get_builtin_tool
from tend.agent.tools.builtin.glob import GlobResult
from tend.agent.tools.builtin.grep import GrepResult
from tend.agent.tools.builtin.ls import LsResult


async def test_ls_lists_directory_in_deterministic_order(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")

    tool = get_builtin_tool("ls")
    arguments = tool.validate_arguments({"path": "."})
    result = await tool.run(ToolContext(cwd=tmp_path), arguments)

    assert isinstance(result, LsResult)
    assert result.output.splitlines() == [
        "file\ta.txt\t5 bytes",
        "file\tb.txt\t4 bytes",
        "dir\tnested/\t-",
    ]
    assert result.total_entries == 3
    assert result.returned_entries == 3
    assert result.truncated is False
    assert result.truncation is None


async def test_glob_finds_matches_in_deterministic_order(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "z.py").write_text("z", encoding="utf-8")
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("notes", encoding="utf-8")

    tool = get_builtin_tool("glob")
    arguments = tool.validate_arguments({"pattern": "**/*.py", "root": "."})
    result = await tool.run(ToolContext(cwd=tmp_path), arguments)

    assert isinstance(result, GlobResult)
    assert result.output.splitlines() == [
        str(tmp_path / "a.py"),
        str(tmp_path / "pkg" / "z.py"),
    ]
    assert result.total_matches == 2
    assert result.returned_matches == 2
    assert result.truncated is False


async def test_grep_searches_utf8_files_in_path_and_line_order(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("skip\nalpha beta\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("alpha\nnope\n", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"\xff\xfe")

    tool = get_builtin_tool("grep")
    arguments = tool.validate_arguments(
        {"pattern": "alpha", "path": ".", "glob": "*.txt", "case_sensitive": True}
    )
    result = await tool.run(ToolContext(cwd=tmp_path), arguments)

    assert isinstance(result, GrepResult)
    assert result.output.splitlines() == [
        f"{tmp_path / 'a.txt'}:1:alpha",
        f"{tmp_path / 'b.txt'}:2:alpha beta",
    ]
    assert result.total_candidate_files == 2
    assert result.searched_files == 2
    assert result.total_matches == 2
    assert result.returned_matches == 2
    assert result.omitted_non_utf8_files == ()


async def test_result_limits_use_head_truncation_metadata(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"file_{index}.txt").write_text(f"match {index}\n", encoding="utf-8")

    ls_tool = get_builtin_tool("ls")
    ls_result = await ls_tool.run(
        ToolContext(cwd=tmp_path),
        ls_tool.validate_arguments({"path": ".", "max_entries": 1}),
    )
    assert isinstance(ls_result, LsResult)
    assert ls_result.truncated is True
    assert ls_result.truncation is not None
    assert ls_result.truncation.policy == "head"
    assert "[Output truncated:" in ls_result.output

    glob_tool = get_builtin_tool("glob")
    glob_result = await glob_tool.run(
        ToolContext(cwd=tmp_path),
        glob_tool.validate_arguments({"pattern": "*.txt", "max_results": 1}),
    )
    assert isinstance(glob_result, GlobResult)
    assert glob_result.truncated is True
    assert glob_result.truncation is not None
    assert "[Output truncated:" in glob_result.output

    grep_tool = get_builtin_tool("grep")
    grep_result = await grep_tool.run(
        ToolContext(cwd=tmp_path),
        grep_tool.validate_arguments(
            {"pattern": "match", "path": ".", "glob": "*.txt", "max_matches": 1}
        ),
    )
    assert isinstance(grep_result, GrepResult)
    assert grep_result.truncated is True
    assert grep_result.truncation is not None
    assert "[Output truncated:" in grep_result.output


async def test_no_match_behavior_is_explicit(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")

    glob_tool = get_builtin_tool("glob")
    glob_result = await glob_tool.run(
        ToolContext(cwd=tmp_path),
        glob_tool.validate_arguments({"pattern": "*.missing"}),
    )
    assert isinstance(glob_result, GlobResult)
    assert glob_result.output == "[No matches]"
    assert glob_result.total_matches == 0

    grep_tool = get_builtin_tool("grep")
    grep_result = await grep_tool.run(
        ToolContext(cwd=tmp_path),
        grep_tool.validate_arguments({"pattern": "missing", "path": ".", "glob": "*.txt"}),
    )
    assert isinstance(grep_result, GrepResult)
    assert grep_result.output == "[No matches]"
    assert grep_result.total_matches == 0


async def test_grep_omits_non_utf8_files_without_returning_raw_bytes(tmp_path: Path) -> None:
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe")

    tool = get_builtin_tool("grep")
    result = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments({"pattern": "anything", "path": ".", "glob": "*.txt"}),
    )

    assert isinstance(result, GrepResult)
    assert result.output == "[No matches]"
    assert result.omitted_non_utf8_files == (str(tmp_path / "bad.txt"),)


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    (
        ("ls", {"path": ".", "unexpected": True}),
        ("glob", {"pattern": "*.py", "max_results": 0}),
        ("grep", {"pattern": "[", "path": "."}),
    ),
)
def test_invalid_arguments_fail_at_tool_validation_layer(
    tool_name: str,
    arguments: JsonObject,
) -> None:
    tool = get_builtin_tool(tool_name)

    with pytest.raises(ValidationError):
        tool.validate_arguments(arguments)
