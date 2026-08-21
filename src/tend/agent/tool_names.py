"""Static v1 built-in tool names and validation helpers."""

from __future__ import annotations

from collections.abc import Iterable

BUILTIN_TOOL_NAMES: tuple[str, ...] = (
    "ls",
    "read_file",
    "grep",
    "glob",
    "write_file",
    "edit_file",
    "copy_lines",
    "bash",
)


def list_builtin_tool_names() -> tuple[str, ...]:
    """Return v1 built-in tool names in stable registry order."""

    return BUILTIN_TOOL_NAMES


def unknown_builtin_tool_names(tool_names: Iterable[str]) -> tuple[str, ...]:
    """Return unknown built-in tool names in deterministic order."""

    allowed = set(BUILTIN_TOOL_NAMES)
    return tuple(sorted({name for name in tool_names if name not in allowed}))


def validate_builtin_tool_names(tool_names: Iterable[str]) -> None:
    """Validate names against the closed v1 built-in tool name set."""

    unknown = unknown_builtin_tool_names(tool_names)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"unknown built-in tool name(s): {joined}")


__all__ = (
    "BUILTIN_TOOL_NAMES",
    "list_builtin_tool_names",
    "unknown_builtin_tool_names",
    "validate_builtin_tool_names",
)
