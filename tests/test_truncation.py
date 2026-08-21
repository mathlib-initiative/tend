from __future__ import annotations

import pytest
from pydantic import ValidationError

from tend.llm.artifacts import ArtifactRef
from tend.llm.models import ToolResult
from tend.llm.truncation import TruncationInfo, TruncationPolicy, truncate_head, truncate_tail


def _size_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def test_no_truncation_returns_original_text_and_metadata() -> None:
    text = "alpha\nbeta"

    result = truncate_head(text, max_lines=5, max_bytes=100)

    assert result.text == text
    assert result.info.truncated is False
    assert result.info.policy is TruncationPolicy.HEAD
    assert result.info.original_size_bytes == _size_bytes(text)
    assert result.info.original_line_count == 2
    assert result.info.returned_size_bytes == _size_bytes(text)
    assert result.info.returned_line_count == 2
    assert result.info.omitted_size_bytes is None
    assert result.info.omitted_line_count is None
    assert result.info.artifact is None


def test_head_truncation_keeps_prefix_and_appends_notice() -> None:
    text = "one\ntwo\nthree\nfour\n"

    result = truncate_head(text, max_lines=2)

    assert result.text == (
        "one\n"
        "two\n"
        "[Output truncated: showing first 2 of 4 lines and 8 of 19 bytes.]"
    )
    assert result.info.truncated is True
    assert result.info.policy is TruncationPolicy.HEAD
    assert result.info.original_size_bytes == 19
    assert result.info.original_line_count == 4
    assert result.info.returned_size_bytes == _size_bytes(result.text)
    assert result.info.returned_line_count == 3
    assert result.info.omitted_size_bytes == 11
    assert result.info.omitted_line_count == 2


def test_tail_truncation_keeps_suffix_and_prepends_notice() -> None:
    text = "one\ntwo\nthree\nfour\n"

    result = truncate_tail(text, max_lines=2)

    assert result.text == (
        "[Output truncated: showing last 2 of 4 lines and 11 of 19 bytes.]\n"
        "three\n"
        "four\n"
    )
    assert result.info.truncated is True
    assert result.info.policy is TruncationPolicy.TAIL
    assert result.info.original_size_bytes == 19
    assert result.info.original_line_count == 4
    assert result.info.returned_size_bytes == _size_bytes(result.text)
    assert result.info.returned_line_count == 3
    assert result.info.omitted_size_bytes == 8
    assert result.info.omitted_line_count == 2


def test_byte_truncation_preserves_utf8_boundaries() -> None:
    text = "ééé"

    result = truncate_head(text, max_bytes=5)

    assert result.text.startswith("éé\n[Output truncated:")
    assert "�" not in result.text
    assert result.info.original_size_bytes == 6
    assert result.info.original_line_count == 1
    assert result.info.omitted_size_bytes == 2
    assert result.info.omitted_line_count == 0


def test_truncation_notice_references_artifact_without_writing_files() -> None:
    artifact = ArtifactRef(
        artifact_id="art_tool_output_1",
        kind="tool_output",
        path="tool_outputs/art_tool_output_1.txt",
        size_bytes=123,
        content_type="text/plain",
    )

    result = truncate_tail("a\nb\nc\n", max_lines=1, artifact=artifact)

    assert result.info.artifact == artifact
    assert "Full output artifact: art_tool_output_1." in result.text
    assert result.info.model_dump(mode="json")["artifact"] == {
        "artifact_id": "art_tool_output_1",
        "kind": "tool_output",
        "path": "tool_outputs/art_tool_output_1.txt",
        "size_bytes": 123,
        "content_type": "text/plain",
        "sha256": None,
        "metadata": {},
    }


def test_invalid_limits_and_untruncated_artifact_metadata_are_rejected() -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        truncate_head("text", max_bytes=0)

    with pytest.raises(ValueError, match="max_lines"):
        truncate_tail("text", max_lines=0)

    with pytest.raises(ValidationError, match="untruncated output"):
        TruncationInfo(
            truncated=False,
            policy=TruncationPolicy.HEAD,
            returned_size_bytes=4,
            returned_line_count=1,
            omitted_size_bytes=1,
        )


def test_tool_result_can_carry_explicit_truncation_metadata() -> None:
    truncated = truncate_head("one\ntwo\n", max_lines=1)

    result = ToolResult(
        tool_call_id="call_1",
        tool_name="read_file",
        success=True,
        output=truncated.text,
        truncated=True,
        truncation=truncated.info,
    )

    assert result.truncated is True
    assert result.truncation == truncated.info

    with pytest.raises(ValidationError, match="truncation metadata"):
        ToolResult(
            tool_call_id="call_2",
            tool_name="read_file",
            success=True,
            truncated=True,
        )

    with pytest.raises(ValidationError, match="truncated flag"):
        ToolResult(
            tool_call_id="call_3",
            tool_name="read_file",
            success=True,
            output="complete",
            truncated=False,
            truncation=truncated.info,
        )
