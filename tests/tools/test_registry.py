from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from tend._common.types import JsonObject
from tend.agent.config import AgentConfig
from tend.agent.tools import (
    BUILTIN_TOOL_NAMES,
    export_builtin_tool_schemas,
    get_builtin_tool,
    get_builtin_tools,
    list_builtin_tool_names,
    unknown_builtin_tool_names,
    validate_builtin_tool_names,
)


def test_registry_contains_exact_v1_builtin_names() -> None:
    assert list_builtin_tool_names() == (
        "ls",
        "read_file",
        "grep",
        "glob",
        "write_file",
        "edit_file",
        "copy_lines",
        "bash",
    )
    assert BUILTIN_TOOL_NAMES == list_builtin_tool_names()


@pytest.mark.parametrize("name", BUILTIN_TOOL_NAMES)
def test_get_builtin_tool_returns_closed_registry_entries(name: str) -> None:
    tool = get_builtin_tool(name)

    assert tool.name == name
    assert tool.definition.metadata["built_in"] is True
    assert tool.definition.arguments_schema["type"] == "object"
    assert tool.definition.arguments_schema["additionalProperties"] is False


def test_get_builtin_tools_preserves_enabled_order() -> None:
    tools = get_builtin_tools(["bash", "read_file", "ls"])

    assert [tool.name for tool in tools] == ["bash", "read_file", "ls"]


def test_unknown_builtin_names_fail_clearly() -> None:
    assert unknown_builtin_tool_names(["ls", "missing", "also_missing", "missing"]) == (
        "also_missing",
        "missing",
    )

    with pytest.raises(ValueError, match="unknown built-in tool name"):
        validate_builtin_tool_names(["ls", "missing"])

    with pytest.raises(ValueError, match="unknown built-in tool name"):
        get_builtin_tool("missing")


def test_config_validation_uses_closed_registry() -> None:
    with pytest.raises(ValidationError, match="unknown built-in tool"):
        AgentConfig.model_validate(
            {
                "system_prompt": "Prompt.",
                "model": {
                    "provider": "custom",
                    "api": "openai_responses",
                    "model_name": "custom-model",
                },
                "tools": {"enabled": ["ls", "unknown"]},
            }
        )


def test_exported_tool_schemas_are_strict_and_defensive_copies() -> None:
    exported = export_builtin_tool_schemas(["ls", "bash"])

    assert [schema["name"] for schema in exported] == ["ls", "bash"]
    for schema in exported:
        arguments_schema = schema["arguments_schema"]
        assert isinstance(arguments_schema, dict)
        assert arguments_schema["type"] == "object"
        assert arguments_schema["additionalProperties"] is False

    first_arguments_schema = cast(JsonObject, exported[0]["arguments_schema"])
    first_arguments_schema["additionalProperties"] = True

    exported_again = export_builtin_tool_schemas(["ls"])
    exported_again_arguments_schema = cast(JsonObject, exported_again[0]["arguments_schema"])
    assert exported_again_arguments_schema["additionalProperties"] is False
