"""Closed built-in tool registry and provider-neutral schema export."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import cast

from pydantic import TypeAdapter

from tend._common.types import JsonObject, StrictModel
from tend.agent.tool_names import (
    BUILTIN_TOOL_NAMES,
    list_builtin_tool_names,
    unknown_builtin_tool_names,
    validate_builtin_tool_names,
)
from tend.agent.tools.base import Tool
from tend.agent.tools.builtin.bash import bash_tool
from tend.agent.tools.builtin.copy_lines import copy_lines_tool
from tend.agent.tools.builtin.edit_file import edit_file_tool
from tend.agent.tools.builtin.glob import glob_tool
from tend.agent.tools.builtin.grep import grep_tool
from tend.agent.tools.builtin.ls import ls_tool
from tend.agent.tools.builtin.read_file import read_file_tool
from tend.agent.tools.builtin.write_file import write_file_tool

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


def _copy_json_object(value: JsonObject) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(deepcopy(value))


_CONCRETE_TOOLS: Mapping[str, Tool[StrictModel]] = MappingProxyType(
    {
        "ls": cast(Tool[StrictModel], ls_tool),
        "grep": cast(Tool[StrictModel], grep_tool),
        "glob": cast(Tool[StrictModel], glob_tool),
        "read_file": cast(Tool[StrictModel], read_file_tool),
        "write_file": cast(Tool[StrictModel], write_file_tool),
        "edit_file": cast(Tool[StrictModel], edit_file_tool),
        "copy_lines": cast(Tool[StrictModel], copy_lines_tool),
        "bash": cast(Tool[StrictModel], bash_tool),
    }
)

_REGISTERED_TOOL_NAMES = frozenset(_CONCRETE_TOOLS)
_EXPECTED_TOOL_NAMES = frozenset(BUILTIN_TOOL_NAMES)
if _REGISTERED_TOOL_NAMES != _EXPECTED_TOOL_NAMES:
    missing = ", ".join(sorted(_EXPECTED_TOOL_NAMES - _REGISTERED_TOOL_NAMES))
    extra = ", ".join(sorted(_REGISTERED_TOOL_NAMES - _EXPECTED_TOOL_NAMES))
    details_parts: list[str] = []
    if missing:
        details_parts.append(f"missing: {missing}")
    if extra:
        details_parts.append(f"extra: {extra}")
    raise RuntimeError(f"built-in tool registry mismatch ({'; '.join(details_parts)})")

_BUILTIN_TOOLS: Mapping[str, Tool[StrictModel]] = _CONCRETE_TOOLS


def get_builtin_tool(name: str) -> Tool[StrictModel]:
    """Return a built-in tool by stable name, failing clearly for unknown names."""

    validate_builtin_tool_names((name,))
    return _BUILTIN_TOOLS[name]


def get_builtin_tools(names: Iterable[str]) -> tuple[Tool[StrictModel], ...]:
    """Return enabled built-in tools in caller-provided order."""

    requested = tuple(names)
    validate_builtin_tool_names(requested)
    return tuple(_BUILTIN_TOOLS[name] for name in requested)


def export_builtin_tool_schemas(names: Iterable[str]) -> tuple[JsonObject, ...]:
    """Export provider-neutral strict tool-definition schemas for enabled tools.

    Provider adapters later translate this neutral shape to native OpenAI
    Responses or Anthropic Messages tool declarations. The argument schema is a
    defensive copy so callers cannot mutate the closed registry definitions.
    """

    return tuple(_export_tool_schema(tool) for tool in get_builtin_tools(names))


def _export_tool_schema(tool: Tool[StrictModel]) -> JsonObject:
    definition = tool.definition
    return {
        "name": definition.name,
        "description": definition.description,
        "arguments_schema": _copy_json_object(definition.arguments_schema),
    }


__all__ = (
    "BUILTIN_TOOL_NAMES",
    "export_builtin_tool_schemas",
    "get_builtin_tool",
    "get_builtin_tools",
    "list_builtin_tool_names",
    "unknown_builtin_tool_names",
    "validate_builtin_tool_names",
)
