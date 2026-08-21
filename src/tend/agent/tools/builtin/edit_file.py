"""Concrete ``edit_file`` built-in tool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field, model_validator

from tend._common.types import JsonObject, StrictModel
from tend.agent.tools.base import Tool
from tend.agent.tools.builtin._common import NonNegativeCount, TextToolOutput, filesystem_backend
from tend.agent.tools.context import ToolContext
from tend.llm.models.tools import ToolError

EditFileErrorType = Literal[
    "binary",
    "duplicate_match",
    "empty_old_text",
    "encoding_error",
    "is_directory",
    "missing_text",
    "no_change",
    "no_op",
    "no_replacements",
    "non_utf8",
    "not_found",
    "overlapping_edits",
    "permission_denied",
    "read_error",
    "write_error",
]
LineEndingStyle = Literal["lf", "crlf"]
EditList = Annotated[list["EditReplacement"], Field(min_length=1)]


@dataclass(frozen=True, slots=True)
class _MatchedEdit:
    edit_index: int
    match_index: int
    match_length: int
    new_text: str


@dataclass(frozen=True, slots=True)
class AppliedEditPlan:
    """Pure exact-replacement plan produced before any file write occurs."""

    content: str
    matches: tuple[_MatchedEdit, ...]


class EditPlanError(ValueError):
    """Semantic edit validation failure raised by the pure planning helper."""

    error_type: EditFileErrorType
    details: JsonObject

    def __init__(self, error_type: EditFileErrorType, message: str, details: JsonObject) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.details = details


class EditReplacement(StrictModel):
    """One exact text replacement to apply within the original file content."""

    old_text: str
    new_text: str


class EditFileArguments(StrictModel):
    """Arguments for atomic exact-match file edits.

    Every ``edits[].old_text`` is matched against the original file content, not
    against incrementally edited text. Each match must be unique and
    non-overlapping. The tool does not enforce path allowlists or extension
    restrictions; sandbox policy belongs to the process boundary.
    """

    path: str = Field(min_length=1)
    edits: EditList


class EditFileResult(TextToolOutput):
    """Structured ``edit_file`` result returned by the built-in handler."""

    path: str
    success: bool
    replacement_count: NonNegativeCount = 0
    bytes_written: NonNegativeCount = 0
    chars_written: NonNegativeCount = 0
    original_size_bytes: NonNegativeCount | None = None
    edited_size_bytes: NonNegativeCount | None = None
    line_ending: LineEndingStyle | None = None
    had_utf8_bom: bool | None = None
    error: ToolError | None = None

    @model_validator(mode="after")
    def _validate_success_error_pair(self) -> EditFileResult:
        if self.success and self.error is not None:
            raise ValueError("successful edit results must not include an error")
        if not self.success and self.error is None:
            raise ValueError("failed edit results must include an error")
        return self


async def _run_edit_file(context: ToolContext, arguments: EditFileArguments) -> EditFileResult:
    backend = filesystem_backend(context)

    try:
        data = await backend.read_bytes(arguments.path)
    except Exception as exc:
        return _error_result(
            arguments,
            error_type=_classify_read_error(exc),
            message=_read_error_message(arguments.path, exc),
            details=_exception_details(arguments.path, exc, operation="read"),
        )

    original_size_bytes = len(data)
    if b"\x00" in data:
        return _error_result(
            arguments,
            error_type="binary",
            message=f"Binary file cannot be edited as UTF-8 text: {arguments.path}",
            details={"path": arguments.path, "size_bytes": original_size_bytes},
            original_size_bytes=original_size_bytes,
        )

    try:
        raw_text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return _error_result(
            arguments,
            error_type="non_utf8",
            message=f"File is not valid UTF-8: {arguments.path}",
            details={
                "path": arguments.path,
                "size_bytes": original_size_bytes,
                "encoding": "utf-8",
                "exception_message": str(exc),
            },
            original_size_bytes=original_size_bytes,
        )

    bom, text = strip_utf8_bom(raw_text)
    line_ending = detect_line_ending(text)
    normalized_content = normalize_to_lf(text)

    try:
        plan = apply_replacements_to_normalized_content(
            normalized_content,
            arguments.edits,
            path=arguments.path,
        )
    except EditPlanError as exc:
        return _error_result(
            arguments,
            error_type=exc.error_type,
            message=str(exc),
            details=exc.details,
            original_size_bytes=original_size_bytes,
            line_ending=line_ending,
            had_utf8_bom=bool(bom),
        )

    final_text = bom + restore_line_endings(plan.content, line_ending)
    try:
        final_bytes = final_text.encode("utf-8")
    except UnicodeEncodeError as exc:
        return _error_result(
            arguments,
            error_type="encoding_error",
            message=f"Edited content could not be encoded as UTF-8: {exc}",
            details={
                "path": arguments.path,
                "encoding": "utf-8",
                "exception_message": str(exc),
            },
            original_size_bytes=original_size_bytes,
            line_ending=line_ending,
            had_utf8_bom=bool(bom),
        )

    try:
        await backend.write_bytes(arguments.path, final_bytes, create_parents=False)
    except Exception as exc:
        return _error_result(
            arguments,
            error_type=_classify_write_error(exc),
            message=f"{type(exc).__name__}: {exc}",
            details=_exception_details(arguments.path, exc, operation="write"),
            original_size_bytes=original_size_bytes,
            line_ending=line_ending,
            had_utf8_bom=bool(bom),
        )

    replacement_count = len(plan.matches)
    bytes_written = len(final_bytes)
    chars_written = len(final_text)
    return EditFileResult(
        path=arguments.path,
        success=True,
        replacement_count=replacement_count,
        bytes_written=bytes_written,
        chars_written=chars_written,
        original_size_bytes=original_size_bytes,
        edited_size_bytes=bytes_written,
        line_ending=_line_ending_style(line_ending),
        had_utf8_bom=bool(bom),
        output=(
            f"Edited {replacement_count} {_plural(replacement_count, 'replacement')} "
            f"in {arguments.path}. Wrote {bytes_written} {_plural(bytes_written, 'byte')} "
            f"({chars_written} {_plural(chars_written, 'character')})."
        ),
    )


def detect_line_ending(text: str) -> str:
    """Detect whether a file's first newline uses LF or CRLF."""

    first_lf = text.find("\n")
    if first_lf == -1:
        return "\n"
    if first_lf > 0 and text[first_lf - 1] == "\r":
        return "\r\n"
    return "\n"


def normalize_to_lf(text: str) -> str:
    """Normalize text to LF so matching is independent of file line endings."""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    """Restore LF-normalized content to the detected file line-ending style."""

    if ending == "\r\n":
        return text.replace("\n", "\r\n")
    return text


def strip_utf8_bom(text: str) -> tuple[str, str]:
    """Return the UTF-8 BOM prefix, if present, and text without that prefix."""

    if text.startswith("\ufeff"):
        return "\ufeff", text.removeprefix("\ufeff")
    return "", text


def count_occurrences(content: str, old_text: str) -> int:
    """Count possibly overlapping exact matches of ``old_text`` in ``content``."""

    if old_text == "":
        return 0

    count = 0
    start = 0
    while True:
        match_index = content.find(old_text, start)
        if match_index == -1:
            return count
        count += 1
        start = match_index + 1


def apply_replacements_to_normalized_content(
    normalized_content: str,
    edits: list[EditReplacement],
    *,
    path: str,
) -> AppliedEditPlan:
    """Plan exact replacements against one immutable original content string."""

    if not edits:
        raise EditPlanError(
            "no_replacements",
            f"edits must contain at least one replacement for {path}",
            {"path": path},
        )

    matched_edits: list[_MatchedEdit] = []
    for edit_index, edit in enumerate(edits):
        old_text = normalize_to_lf(edit.old_text)
        new_text = normalize_to_lf(edit.new_text)
        details: JsonObject = {
            "path": path,
            "edit_index": edit_index,
            "old_text_length": len(old_text),
            "new_text_length": len(new_text),
        }

        if old_text == "":
            raise EditPlanError(
                "empty_old_text",
                f"edits[{edit_index}].old_text must not be empty in {path}",
                details,
            )
        if old_text == new_text:
            raise EditPlanError(
                "no_op",
                (
                    f"edits[{edit_index}] would make no change in {path}; "
                    "old_text and new_text are identical"
                ),
                details,
            )

        occurrences = count_occurrences(normalized_content, old_text)
        if occurrences == 0:
            missing_details: JsonObject = {**details, "occurrence_count": occurrences}
            raise EditPlanError(
                "missing_text",
                f"Could not find edits[{edit_index}].old_text in {path}",
                missing_details,
            )
        if occurrences > 1:
            duplicate_details: JsonObject = {**details, "occurrence_count": occurrences}
            raise EditPlanError(
                "duplicate_match",
                (
                    f"Found {occurrences} occurrences of edits[{edit_index}].old_text "
                    f"in {path}; old_text must be unique"
                ),
                duplicate_details,
            )

        match_index = normalized_content.find(old_text)
        matched_edits.append(
            _MatchedEdit(
                edit_index=edit_index,
                match_index=match_index,
                match_length=len(old_text),
                new_text=new_text,
            )
        )

    matched_edits.sort(key=lambda item: item.match_index)
    for index in range(1, len(matched_edits)):
        previous = matched_edits[index - 1]
        current = matched_edits[index]
        previous_end = previous.match_index + previous.match_length
        if previous_end > current.match_index:
            raise EditPlanError(
                "overlapping_edits",
                f"edits[{previous.edit_index}] and edits[{current.edit_index}] overlap in {path}",
                {
                    "path": path,
                    "first_edit_index": previous.edit_index,
                    "second_edit_index": current.edit_index,
                    "first_start": previous.match_index,
                    "first_end": previous_end,
                    "second_start": current.match_index,
                    "second_end": current.match_index + current.match_length,
                },
            )

    new_content = normalized_content
    for matched_edit in reversed(matched_edits):
        new_content = (
            new_content[: matched_edit.match_index]
            + matched_edit.new_text
            + new_content[matched_edit.match_index + matched_edit.match_length :]
        )

    if new_content == normalized_content:
        raise EditPlanError(
            "no_change",
            f"No changes made to {path}",
            {"path": path, "edit_count": len(edits)},
        )

    return AppliedEditPlan(content=new_content, matches=tuple(matched_edits))


def _classify_read_error(exc: Exception) -> EditFileErrorType:
    if isinstance(exc, FileNotFoundError):
        return "not_found"
    if isinstance(exc, IsADirectoryError):
        return "is_directory"
    if isinstance(exc, PermissionError):
        return "permission_denied"
    return "read_error"


def _classify_write_error(exc: Exception) -> EditFileErrorType:
    if isinstance(exc, IsADirectoryError):
        return "is_directory"
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if isinstance(exc, UnicodeError):
        return "encoding_error"
    return "write_error"


def _read_error_message(path: str, exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return f"File not found: {path}"
    if isinstance(exc, IsADirectoryError):
        return f"Path is a directory: {path}"
    return f"{type(exc).__name__}: {exc}"


def _exception_details(path: str, exc: Exception, *, operation: str) -> JsonObject:
    return {
        "path": path,
        "operation": operation,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    }


def _error_result(
    arguments: EditFileArguments,
    *,
    error_type: EditFileErrorType,
    message: str,
    details: JsonObject,
    original_size_bytes: int | None = None,
    line_ending: str | None = None,
    had_utf8_bom: bool | None = None,
) -> EditFileResult:
    return EditFileResult(
        path=arguments.path,
        success=False,
        replacement_count=0,
        bytes_written=0,
        chars_written=0,
        original_size_bytes=original_size_bytes,
        edited_size_bytes=None,
        line_ending=_line_ending_style(line_ending) if line_ending is not None else None,
        had_utf8_bom=had_utf8_bom,
        output=f"[File edit error: {message}]",
        error=ToolError(error_type=error_type, message=message, details=details),
    )


def _line_ending_style(ending: str) -> LineEndingStyle:
    if ending == "\r\n":
        return "crlf"
    return "lf"


def _plural(count: int, singular: str) -> str:
    if count == 1:
        return singular
    return f"{singular}s"


edit_file_tool: Tool[EditFileArguments] = Tool.from_arguments_model(
    name="edit_file",
    description=(
        "Edit an existing UTF-8 text file with one or more exact replacements in a "
        "single all-or-nothing operation. Every edits[].old_text must match exactly "
        "one non-overlapping region of the original file content, and each edit must "
        "change text. Existing CRLF line endings and a UTF-8 BOM are preserved when "
        "practical. This tool does not enforce path allowlists; sandbox policy belongs "
        "to the process/orchestration sandbox boundary."
    ),
    arguments_model=EditFileArguments,
    handler=_run_edit_file,
    metadata={"built_in": True},
)


__all__ = (
    "AppliedEditPlan",
    "EditFileArguments",
    "EditFileResult",
    "EditPlanError",
    "EditReplacement",
    "apply_replacements_to_normalized_content",
    "count_occurrences",
    "detect_line_ending",
    "edit_file_tool",
    "normalize_to_lf",
    "restore_line_endings",
    "strip_utf8_bom",
)
