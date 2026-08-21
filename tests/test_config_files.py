from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from tend._common.config_files import (
    ConfigFileError,
    dump_yaml_data,
    read_config_model,
    read_yaml_config_data,
)
from tend.agent.config import AgentConfig, RuntimeConfigOverrides


def _agent_data() -> dict[str, object]:
    return {
        "schema_version": "1",
        "system_prompt": "You are a careful coding agent.",
        "model": {
            "provider": "cloudflare_openai",
            "api": "openai_responses",
            "model_name": "gpt-5",
            "settings": {"reasoning": {"effort": "minimal"}},
        },
        "tools": {"enabled": ["read_file", "bash"]},
    }


def test_read_config_model_accepts_yaml(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        """schema_version: "1"
system_prompt: |
  You are a careful coding agent.
model:
  provider: cloudflare_openai
  api: openai_responses
  model_name: gpt-5
  settings:
    reasoning:
      effort: minimal
tools:
  enabled:
    - read_file
    - bash
""",
        encoding="utf-8",
    )

    config = read_config_model(path, AgentConfig, kind="agent config")

    assert config.schema_version == "1"
    assert config.system_prompt == "You are a careful coding agent.\n"
    assert config.tools.enabled == ["read_file", "bash"]


def test_read_config_model_keeps_json_compatibility(tmp_path: Path) -> None:
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(_agent_data()), encoding="utf-8")

    config = read_config_model(path, AgentConfig, kind="agent config")

    assert config.model.model_name == "gpt-5"


def test_yaml_loader_rejects_duplicate_and_non_string_mapping_keys() -> None:
    with pytest.raises(ConfigFileError, match="duplicate key"):
        read_yaml_config_data("a: 1\na: 2\n", kind="test config")

    with pytest.raises(ConfigFileError, match="non-string key"):
        read_yaml_config_data("1: value\n", kind="test config")


def test_yaml_loader_avoids_yaml_1_1_implicit_surprises() -> None:
    data = read_yaml_config_data(
        """
enabled: true
word: on
other_word: yes
date_like: 2026-05-07
normal_int: 12
normal_float: 1.20
octal_like: 012
sexagesimal_like: 1:20
""",
        kind="test config",
    )

    assert data == {
        "enabled": True,
        "word": "on",
        "other_word": "yes",
        "date_like": "2026-05-07",
        "normal_int": 12,
        "normal_float": 1.20,
        "octal_like": "012",
        "sexagesimal_like": "1:20",
    }


def test_yaml_loader_rejects_yaml_1_1_numeric_spellings_for_schema(
    tmp_path: Path,
) -> None:
    octal_path = tmp_path / "octal.yaml"
    octal_path.write_text("limits:\n  max_model_requests: 012\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="max_model_requests"):
        read_config_model(octal_path, RuntimeConfigOverrides, kind="runtime config")

    sexagesimal_path = tmp_path / "sexagesimal.yaml"
    sexagesimal_path.write_text("limits:\n  max_wall_time_seconds: 1:20\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="max_wall_time_seconds"):
        read_config_model(sexagesimal_path, RuntimeConfigOverrides, kind="runtime config")


def test_dump_yaml_data_uses_literal_blocks_for_multiline_strings() -> None:
    dumped = dump_yaml_data({"schema_version": "1", "prompt": "line one\nline two"})

    assert "schema_version: '1'" in dumped
    assert "prompt: |-\n" in dumped
    assert "  line one\n  line two\n" in dumped


def test_dump_yaml_data_quotes_strings_matching_json_numeric_resolvers() -> None:
    dumped = dump_yaml_data({"model_name": "1e3", "keys": {"1e3": "still a key"}})

    assert "model_name: '1e3'" in dumped
    assert "'1e3': still a key" in dumped
    assert read_yaml_config_data(dumped, kind="test config") == {
        "model_name": "1e3",
        "keys": {"1e3": "still a key"},
    }


def test_yaml_validation_uses_pydantic_json_mode(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text('limits:\n  max_cost: "1.25"\n', encoding="utf-8")

    cfg = read_config_model(path, RuntimeConfigOverrides, kind="runtime config")

    assert cfg.limits is not None
    assert cfg.limits.max_cost == Decimal("1.25")
