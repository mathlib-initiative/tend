"""Concrete built-in tool implementations."""

from tend.agent.tools.builtin.bash import BashArguments, BashResult, bash_tool
from tend.agent.tools.builtin.copy_lines import (
    CopyLinesArguments,
    CopyLinesResult,
    copy_lines_tool,
)
from tend.agent.tools.builtin.edit_file import (
    EditFileArguments,
    EditFileResult,
    EditReplacement,
    edit_file_tool,
)
from tend.agent.tools.builtin.glob import GlobArguments, GlobResult, glob_tool
from tend.agent.tools.builtin.grep import GrepArguments, GrepResult, grep_tool
from tend.agent.tools.builtin.ls import LsArguments, LsResult, ls_tool
from tend.agent.tools.builtin.read_file import ReadFileArguments, ReadFileResult, read_file_tool
from tend.agent.tools.builtin.write_file import (
    WriteFileArguments,
    WriteFileResult,
    write_file_tool,
)

__all__ = (
    "BashArguments",
    "BashResult",
    "bash_tool",
    "CopyLinesArguments",
    "CopyLinesResult",
    "copy_lines_tool",
    "EditFileArguments",
    "EditFileResult",
    "EditReplacement",
    "edit_file_tool",
    "GlobArguments",
    "GlobResult",
    "GrepArguments",
    "GrepResult",
    "LsArguments",
    "LsResult",
    "ReadFileArguments",
    "ReadFileResult",
    "glob_tool",
    "grep_tool",
    "ls_tool",
    "read_file_tool",
    "WriteFileArguments",
    "WriteFileResult",
    "write_file_tool",
)
