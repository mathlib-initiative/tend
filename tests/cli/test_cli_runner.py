from __future__ import annotations

import json
from collections.abc import Mapping
from io import StringIO
from pathlib import Path

from tend._common.config_files import dump_yaml_data
from tend._common.types import StopReason
from tend.agent.cli import (
    CliRunOptions,
    ModelAdapterFactory,
    load_cli_config,
    run_cli,
)
from tend.agent.config import AgentModelConfig, RuntimeConfig
from tend.agent.results import TurnResult
from tend.llm.models import (
    AssistantMessage,
    ModelAdapter,
    ModelRequest,
    ModelResponse,
    TextContent,
    UserMessage,
)
from tend.llm.testing import ScriptedModel


def _agent_data() -> dict[str, object]:
    return {
        "system_prompt": "You are a concise test agent.",
        "model": {
            "provider": "cloudflare_openai",
            "api": "openai_responses",
            "model_name": "gpt-5",
            "settings": {"reasoning": {"effort": "minimal"}, "max_output_tokens": 64},
        },
        "tools": {"enabled": []},
    }


def _write_agent(path: Path) -> Path:
    path.write_text(json.dumps(_agent_data()), encoding="utf-8")
    return path


def _write_cfg(path: Path, data: dict[str, object]) -> Path:
    content = dump_yaml_data(data) if path.suffix in (".yaml", ".yml") else json.dumps(data)
    path.write_text(content, encoding="utf-8")
    return path


def _write_agent_yaml(path: Path) -> Path:
    path.write_text(dump_yaml_data(_agent_data()), encoding="utf-8")
    return path


def _final_response(text: str = "scripted final") -> ModelResponse:
    return ModelResponse(
        assistant_message=AssistantMessage(content=[TextContent(text=text)]),
        stop_reason=StopReason.FINAL_RESPONSE,
    )


def _factory_for(scripted: ScriptedModel) -> tuple[list[RuntimeConfig], ModelAdapterFactory]:
    seen_runtime: list[RuntimeConfig] = []

    def factory(
        model_config: AgentModelConfig,
        runtime_config: RuntimeConfig,
        environment: Mapping[str, str],
    ) -> ModelAdapter:
        assert model_config.model_name == "gpt-5"
        seen_runtime.append(runtime_config)
        assert "UNRELATED_SECRET" not in environment
        return scripted

    return seen_runtime, factory


def _last_user_text(request: ModelRequest) -> str:
    message = request.messages[-1]
    assert isinstance(message, UserMessage)
    part = message.content[0]
    assert isinstance(part, TextContent)
    return part.text


def test_load_cli_config_resolves_agent_cfg_and_cli_precedence(tmp_path: Path) -> None:
    agent_path = _write_agent_yaml(tmp_path / "agent.yaml")
    cfg_path = _write_cfg(
        tmp_path / "cfg.yaml",
        {
            "prompt": "from cfg",
            "cwd": "/cfg-cwd",
            "limits": {"max_model_requests": 2},
            "model": {"timeout_seconds": 12.0},
        },
    )

    resolved = load_cli_config(
        CliRunOptions(
            agent_path=agent_path,
            config_path=cfg_path,
            cwd="/cli-cwd",
            max_iterations=3,
            disable_compaction=True,
        )
    )

    assert resolved.agent.system_prompt == "You are a concise test agent."
    assert resolved.runtime.prompt == "from cfg"
    assert resolved.runtime.cwd == "/cli-cwd"
    assert resolved.runtime.limits.max_iterations == 3
    assert resolved.runtime.limits.max_model_requests == 2
    assert resolved.runtime.model.timeout_seconds == 12.0
    assert resolved.runtime.compaction.enabled is False


async def test_cli_runner_prompt_precedence_cli_over_config_over_stdin(
    tmp_path: Path,
) -> None:
    agent_path = _write_agent(tmp_path / "agent.json")
    cfg_with_prompt = _write_cfg(tmp_path / "cfg_with_prompt.json", {"prompt": "from cfg"})
    cfg_without_prompt = _write_cfg(tmp_path / "cfg_without_prompt.json", {})

    assert await _captured_prompt(
        agent_path,
        ["--config", str(cfg_with_prompt), "--prompt", "from cli"],
        stdin_text="from stdin",
    ) == "from cli"
    assert await _captured_prompt(
        agent_path,
        ["--config", str(cfg_with_prompt)],
        stdin_text="from stdin",
    ) == "from cfg"
    assert await _captured_prompt(
        agent_path,
        ["--config", str(cfg_without_prompt)],
        stdin_text="from stdin",
    ) == "from stdin"


async def _captured_prompt(agent_path: Path, extra_args: list[str], *, stdin_text: str) -> str:
    scripted = ScriptedModel([_final_response()])
    _seen_runtime, factory = _factory_for(scripted)
    stdout = StringIO()
    stderr = StringIO()

    code = await run_cli(
        ["--agent", str(agent_path), *extra_args, "--no-compaction"],
        stdin=StringIO(stdin_text),
        stdout=stdout,
        stderr=stderr,
        environment={"UNRELATED_SECRET": "must-not-be-forwarded"},
        model_factory=factory,
    )

    assert code == 0, stderr.getvalue()
    assert stdout.getvalue() == "scripted final\n"
    request = scripted.requests[0]
    return _last_user_text(request)


async def test_cli_runner_runs_one_scripted_turn_and_writes_json_result(
    tmp_path: Path,
) -> None:
    agent_path = _write_agent(tmp_path / "agent.json")
    session_dir = tmp_path / "session"
    scripted = ScriptedModel([_final_response("done")])
    seen_runtime, factory = _factory_for(scripted)
    stdout = StringIO()
    stderr = StringIO()

    code = await run_cli(
        [
            "--agent",
            str(agent_path),
            "--prompt",
            "Run once.",
            "--session-dir",
            str(session_dir),
            "--session-id",
            "sess_cli_test",
            "--json",
            "--no-compaction",
        ],
        stdin=StringIO("ignored"),
        stdout=stdout,
        stderr=stderr,
        environment={},
        model_factory=factory,
    )

    assert code == 0, stderr.getvalue()
    result = TurnResult.model_validate_json(stdout.getvalue())
    assert result.final_response == "done"
    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.session_id == "sess_cli_test"
    assert result.model_request_count == 1
    assert result.tool_call_count == 0
    assert scripted.remaining_steps == 0
    assert _last_user_text(scripted.requests[0]) == "Run once."
    assert len(seen_runtime) == 1
    assert seen_runtime[0].session_dir == str(session_dir)
    assert (session_dir / "events.jsonl").exists()
    assert (session_dir / "state.json").exists()


async def test_cli_runner_config_and_usage_errors_are_structured(
    tmp_path: Path,
) -> None:
    missing_stderr = StringIO()
    code = await run_cli(
        ["--agent", str(tmp_path / "missing-agent.json"), "--prompt", "hello"],
        stdin=StringIO(""),
        stdout=StringIO(),
        stderr=missing_stderr,
        environment={},
    )
    assert code == 2
    missing_error = json.loads(missing_stderr.getvalue())
    assert missing_error["code"] == "configuration_error"
    assert missing_error["details"]["kind"] == "agent config"

    agent_path = _write_agent(tmp_path / "agent.json")
    usage_stderr = StringIO()
    code = await run_cli(
        ["--agent", str(agent_path)],
        stdin=StringIO(""),
        stdout=StringIO(),
        stderr=usage_stderr,
        environment={},
    )
    assert code == 2
    usage_error = json.loads(usage_stderr.getvalue())
    assert usage_error["code"] == "cli_usage_error"
    assert "prompt" in usage_error["message"]


async def test_cli_rejects_removed_inner_flag(tmp_path: Path) -> None:
    agent_path = _write_agent(tmp_path / "agent.json")
    stderr = StringIO()

    code = await run_cli(
        ["--inner", "--agent", str(agent_path), "--prompt", "hello"],
        stdin=StringIO(""),
        stdout=StringIO(),
        stderr=stderr,
        environment={},
    )

    assert code == 2
    error = json.loads(stderr.getvalue())
    assert error["code"] == "cli_usage_error"
    assert "--inner" in error["message"]
