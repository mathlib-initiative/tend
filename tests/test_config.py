import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tend._common.config_files import read_config_model
from tend._common.errors import ConfigurationError
from tend.agent.config import (
    AgentConfig,
    AgentModelConfig,
    HeaderValueSource,
    ModelSettingsConfig,
    ProviderHeaderConfig,
    RuntimeConfig,
    RuntimeConfigOverrides,
    RuntimeLimitsConfig,
    RuntimeLimitsOverrides,
    resolve_config,
    resolve_runtime_config,
)
from tend.agent.outputs import AgentOutputSchemaName
from tend.llm.models import ProviderApi, ReasoningEffort, ReasoningSettings
from tend.prompts import builtin_prompts_root, load_prompt


def _minimal_agent_json(**updates: object) -> str:
    data: dict[str, object] = {
        "system_prompt": "You are a careful coding agent.",
        "model": {
            "provider": "cloudflare_openai",
            "api": "openai_responses",
            "model_name": "gpt-5",
            "settings": {"reasoning": {"effort": "minimal"}},
        },
        "tools": {"enabled": ["read_file", "bash"], "options": {"bash": {"timeout": 10}}},
    }
    data.update(updates)
    return json.dumps(data)


def test_valid_minimal_agent_config_from_json() -> None:
    config = AgentConfig.model_validate_json(_minimal_agent_json())

    assert config.system_prompt == "You are a careful coding agent."
    assert config.model.provider == "cloudflare_openai"
    assert config.model.api is ProviderApi.OPENAI_RESPONSES
    assert config.model.settings.reasoning is not None
    assert config.model.settings.reasoning.effort is ReasoningEffort.MINIMAL
    assert config.tools.enabled == ["read_file", "bash"]
    assert config.tools.options == {"bash": {"timeout": 10}}


def test_agent_config_accepts_typed_output_schema_name() -> None:
    config = AgentConfig.model_validate_json(
        _minimal_agent_json(
            output={
                "tool_name": "final_result",
                "schema_name": "review_verdict",
                "required": True,
            }
        )
    )

    assert config.output is not None
    assert config.output.tool_name == "final_result"
    assert config.output.schema_name is AgentOutputSchemaName.REVIEW_VERDICT
    assert config.output.required is True


def test_agent_output_config_rejects_unknown_schema_name() -> None:
    with pytest.raises(ValidationError, match="schema_name"):
        AgentConfig.model_validate_json(
            _minimal_agent_json(
                output={
                    "tool_name": "final_result",
                    "schema_name": "unknown_output",
                    "required": True,
                }
            )
        )


def test_valid_runtime_config_and_header_descriptors_from_json() -> None:
    runtime = RuntimeConfig.model_validate_json(
        json.dumps(
            {
                "cwd": "/work/project",
                "session_dir": "/work/session",
                "limits": {"max_iterations": 7, "max_tokens": 1234},
                "model": {
                    "base_url": "https://gateway.example.test/v1/account/gateway",
                    "extra_headers": [
                        {
                            "name": "cf-aig-authorization",
                            "source": "env",
                            "env_var": "CF_AIG_TOKEN",
                            "secret": True,
                        }
                    ],
                },
                "environment": {"allowed_env_vars": ["CF_AIG_TOKEN"]},
            }
        )
    )

    assert runtime.cwd == "/work/project"
    assert runtime.limits.max_iterations == 7
    assert runtime.limits.max_tokens == 1234
    assert runtime.model.base_url == "https://gateway.example.test/v1/account/gateway"
    assert runtime.secret_source_names() == (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "CF_AIG_TOKEN",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    )
    assert "CF_AIG_TOKEN" in runtime.allowed_environment_names()


def test_runtime_limit_defaults_are_unbounded_for_counts() -> None:
    # Iteration / model-request / tool-call ceilings are unbounded by default so
    # long-running orchestrated workers are bounded only by wall time, tokens,
    # and cost. A resolved config from empty overrides must preserve this.
    limits = RuntimeLimitsConfig()
    assert limits.max_iterations is None
    assert limits.max_model_requests is None
    assert limits.max_tool_calls is None
    assert limits.max_wall_time_seconds == 3600.0
    assert limits.max_tokens is None
    assert limits.max_cost is None

    resolved = resolve_runtime_config().limits
    assert resolved.max_iterations is None
    assert resolved.max_model_requests is None
    assert resolved.max_tool_calls is None
    assert resolved.max_wall_time_seconds == 3600.0


def test_runtime_resolution_precedence_is_nested_and_deterministic() -> None:
    agent = AgentConfig.model_validate_json(
        _minimal_agent_json(
            runtime_defaults={
                "cwd": "/agent-default",
                "limits": {"max_iterations": 5, "max_model_requests": 6},
                "retries": {"max_attempts": 2},
            }
        )
    )
    cfg = RuntimeConfigOverrides.model_validate(
        {
            "cwd": "/cfg",
            "limits": {"max_iterations": 8},
            "retries": {"initial_delay_seconds": 3.0},
        }
    )
    cli = RuntimeConfigOverrides(limits=RuntimeLimitsOverrides(max_tool_calls=1))

    resolved = resolve_config(agent, cfg=cfg, cli_overrides=cli)

    assert resolved.runtime.cwd == "/cfg"
    assert resolved.runtime.limits.max_iterations == 8
    assert resolved.runtime.limits.max_model_requests == 6
    assert resolved.runtime.limits.max_tool_calls == 1
    assert resolved.runtime.retries.max_attempts == 2
    assert resolved.runtime.retries.initial_delay_seconds == 3.0


def test_runtime_resolution_can_clear_optional_agent_defaults() -> None:
    agent = AgentConfig.model_validate_json(
        _minimal_agent_json(runtime_defaults={"session_dir": "/agent-session"})
    )
    cfg = RuntimeConfigOverrides.model_validate({"session_dir": None})

    resolved = resolve_runtime_config(agent_config=agent, cfg=cfg)

    assert resolved.session_dir is None


def test_unknown_tool_names_fail_configuration_validation() -> None:
    with pytest.raises(ValidationError, match="unknown built-in tool"):
        AgentConfig.model_validate_json(
            _minimal_agent_json(tools={"enabled": ["read_file", "unknown_tool"]})
        )

    with pytest.raises(ValidationError, match="disabled tools"):
        AgentConfig.model_validate_json(
            _minimal_agent_json(tools={"enabled": ["read_file"], "options": {"bash": {}}})
        )


def test_secrets_are_rejected_from_durable_agent_config() -> None:
    with pytest.raises(ValidationError, match="api_key"):
        AgentConfig.model_validate_json(
            _minimal_agent_json(model={"api_key": "sk-test", "provider": "x"})
        )

    with pytest.raises(ValidationError, match="secret-like key"):
        AgentConfig.model_validate_json(
            _minimal_agent_json(
                model={
                    "provider": "custom_openai",
                    "api": "openai_responses",
                    "model_name": "custom",
                    "settings": {"extra_settings": {"authorization": "Bearer secret"}},
                }
            )
        )

    with pytest.raises(ValidationError, match="provider headers"):
        AgentConfig.model_validate_json(
            _minimal_agent_json(
                runtime_defaults={
                    "model": {
                        "extra_headers": [
                            {
                                "name": "Authorization",
                                "source": "literal",
                                "value": "Bearer secret",
                            }
                        ]
                    }
                }
            )
        )


def test_header_values_have_safe_repr_and_strict_source_validation() -> None:
    header = ProviderHeaderConfig(
        name="Authorization",
        source=HeaderValueSource.LITERAL,
        value="Bearer fake-secret",
        secret=True,
    )

    assert "fake-secret" not in repr(header)
    assert header.model_dump()["value"] == "Bearer fake-secret"

    with pytest.raises(ValidationError, match="literal headers require value"):
        ProviderHeaderConfig(name="Authorization", source=HeaderValueSource.LITERAL)


def test_known_profile_rejects_unsupported_agent_model_settings() -> None:
    with pytest.raises(ConfigurationError, match="temperature"):
        AgentConfig(
            system_prompt="Prompt.",
            model=AgentModelConfig(
                provider="cloudflare_openai",
                api=ProviderApi.OPENAI_RESPONSES,
                model_name="gpt-5",
                settings=ModelSettingsConfig(temperature=0.0),
            ),
        )


def test_config_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate({"cwd": ".", "surprise": True})

    with pytest.raises(ValidationError):
        RuntimeConfigOverrides.model_validate({"limits": {"max_iterations": 1, "extra": 2}})


def test_agent_runtime_config_rejects_removed_bwrap_key() -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate({"bwrap": {"enabled": False}})

    with pytest.raises(ValidationError):
        RuntimeConfigOverrides.model_validate({"bwrap": {"enabled": False}})

    with pytest.raises(ValidationError):
        AgentConfig.model_validate_json(
            _minimal_agent_json(runtime_defaults={"bwrap": {"enabled": False}})
        )


def test_agent_config_does_not_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-secret-that-must-not-be-read")

    config = AgentConfig.model_validate_json(_minimal_agent_json())
    runtime = resolve_runtime_config(agent_config=config)

    assert runtime.api_key_sources.openai_api_key_env == "OPENAI_API_KEY"
    assert "fake-secret-that-must-not-be-read" not in repr(runtime)


def test_reasoning_settings_can_be_supplied_with_python_api() -> None:
    config = AgentConfig(
        system_prompt="Prompt.",
        model=AgentModelConfig(
            provider="cloudflare_openai",
            api=ProviderApi.OPENAI_RESPONSES,
            model_name="gpt-5",
            settings=ModelSettingsConfig(
                reasoning=ReasoningSettings(effort=ReasoningEffort.LOW)
            ),
        ),
    )

    assert config.model.settings.reasoning is not None
    assert config.model.settings.reasoning.effort is ReasoningEffort.LOW


def test_system_prompt_accepts_literal_string() -> None:
    """Back-compat: an inlined literal system prompt round-trips verbatim."""

    literal = (
        "You are a careful coding agent.\n"
        "Always cite the file you edited.\n"
        "Stop when the task is complete."
    )
    config = AgentConfig.model_validate_json(_minimal_agent_json(system_prompt=literal))

    assert config.system_prompt == literal


def test_system_prompt_accepts_registry_reference() -> None:
    """A ``{registry: ...}`` pointer resolves against the bundled prompt registry."""

    config = AgentConfig.model_validate_json(
        _minimal_agent_json(system_prompt={"registry": "reviewer/minimal"})
    )

    expected = load_prompt(builtin_prompts_root() / "reviewer" / "minimal", "system")
    assert config.system_prompt == expected


def test_system_prompt_accepts_path_reference_relative_to_agent_config(
    tmp_path: Path,
) -> None:
    """A ``{path: ...}`` pointer loads editable markdown next to the agent config."""

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    system_prompt_path = prompts / "worker-system.md"
    system_prompt_path.write_text("Editable system prompt.\n", encoding="utf-8")
    config_dir = tmp_path / ".tend"
    config_dir.mkdir()
    config_path = config_dir / "worker-agent.yaml"
    config_path.write_text(
        """schema_version: "1"
system_prompt:
  path: ../prompts/worker-system.md
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

    config = read_config_model(config_path, AgentConfig, kind="agent config")
    assert config.system_prompt == "Editable system prompt."

    system_prompt_path.write_text("Edited mid-session prompt.\n", encoding="utf-8")
    reread = read_config_model(config_path, AgentConfig, kind="agent config")
    assert reread.system_prompt == "Edited mid-session prompt."


def test_system_prompt_registry_unknown_role_or_version_errors() -> None:
    """An unknown registry pointer reports the missing on-disk path."""

    with pytest.raises(ValidationError, match="reviewer/v99"):
        AgentConfig.model_validate_json(
            _minimal_agent_json(system_prompt={"registry": "reviewer/v99"})
        )


def test_system_prompt_registry_rejects_absolute_or_traversal_pointer() -> None:
    """Registry pointers must be relative ``<role>/<version>`` slugs."""

    with pytest.raises(ValidationError, match="relative"):
        AgentConfig.model_validate_json(
            _minimal_agent_json(system_prompt={"registry": "../escape"})
        )


def test_system_prompt_registry_rejects_symlink_escape(tmp_path: Path) -> None:
    """A symlink under the bundled root pointing outside it must not be followed.

    The lexical ``..``/absolute check on the slug is not enough on its own: a
    legitimate-looking slug like ``reviewer/sneaky`` would otherwise load
    whatever a planted symlink resolves to. Catch this at the on-disk layer
    too.
    """

    # Plant a symlink inside the bundled registry pointing at an external
    # directory containing a ``system.md``. The validator must refuse to load
    # it instead of silently following the symlink.
    bundled = builtin_prompts_root()
    external = tmp_path / "external_prompts"
    external.mkdir()
    (external / "system.md").write_text("INJECTED", encoding="utf-8")
    sneaky = bundled / "reviewer" / "sneaky_symlink_escape"
    if sneaky.exists() or sneaky.is_symlink():
        sneaky.unlink()
    sneaky.symlink_to(external, target_is_directory=True)
    try:
        with pytest.raises(ValidationError, match="escapes bundled root"):
            AgentConfig.model_validate_json(
                _minimal_agent_json(
                    system_prompt={"registry": "reviewer/sneaky_symlink_escape"}
                )
            )
    finally:
        sneaky.unlink()


def test_all_bundled_system_prompts_are_placeholder_free() -> None:
    """No ``{...}`` placeholder may leak from any registry ``system.md``.

    System prompts are loaded verbatim by the registry-pointer resolver — there
    is no per-invocation substitution step. Any ``{task_path}``-style token
    that survives in a system.md would reach the model as literal text. Task
    prompts (``task.md``) keep their per-invocation substitution and are not
    checked here.
    """

    import re

    bundled = builtin_prompts_root()
    offenders: list[str] = []
    for system_md in bundled.glob("*/*/system.md"):
        text = system_md.read_text(encoding="utf-8")
        if matches := re.findall(r"\{[a-z_]+\}", text):
            offenders.append(f"{system_md}: {sorted(set(matches))}")
    assert not offenders, "Unresolved placeholders in registry system prompts:\n" + "\n".join(
        offenders
    )
