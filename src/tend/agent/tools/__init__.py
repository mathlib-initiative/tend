"""Built-in tool definitions and runtime wrappers."""

from tend.agent.tools.backends import (
    DirectoryEntry,
    FileStat,
    FilesystemBackend,
    ProcessBackend,
    ProcessResult,
    ToolPath,
)
from tend.agent.tools.base import ArgumentPreparer, Tool, ToolDefinition, ToolHandler
from tend.agent.tools.context import ToolCancellationState, ToolContext, ToolEventCallback
from tend.agent.tools.executor import (
    TOOL_CALL_COMPLETED_EVENT,
    TOOL_CALL_STARTED_EVENT,
    EnabledTools,
    execute_tool_calls,
)
from tend.agent.tools.local_backend import LocalFilesystemBackend, LocalProcessBackend
from tend.agent.tools.registry import (
    BUILTIN_TOOL_NAMES,
    export_builtin_tool_schemas,
    get_builtin_tool,
    get_builtin_tools,
    list_builtin_tool_names,
    unknown_builtin_tool_names,
    validate_builtin_tool_names,
)
from tend.llm.models.tools import ToolError, ToolResult

__all__ = (
    "ArgumentPreparer",
    "DirectoryEntry",
    "FileStat",
    "FilesystemBackend",
    "LocalFilesystemBackend",
    "LocalProcessBackend",
    "ProcessBackend",
    "ProcessResult",
    "ToolPath",
    "Tool",
    "ToolCancellationState",
    "ToolContext",
    "ToolDefinition",
    "ToolError",
    "ToolEventCallback",
    "ToolHandler",
    "ToolResult",
    "TOOL_CALL_COMPLETED_EVENT",
    "TOOL_CALL_STARTED_EVENT",
    "EnabledTools",
    "execute_tool_calls",
    "BUILTIN_TOOL_NAMES",
    "export_builtin_tool_schemas",
    "get_builtin_tool",
    "get_builtin_tools",
    "list_builtin_tool_names",
    "unknown_builtin_tool_names",
    "validate_builtin_tool_names",
)
