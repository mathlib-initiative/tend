from __future__ import annotations

import json
from collections.abc import Mapping
from io import StringIO
from pathlib import Path

import pytest

import tend.agent.cli as agent_cli
from tend._common.errors import FrameworkError
from tend._common.types import StopReason
from tend.agent.cli import ExitCode, ModelAdapterFactory, run_cli
from tend.agent.config import AgentModelConfig, RuntimeConfig
from tend.agent.results import FinalResultOutput, TurnResult
from tend.llm.models import (
    AssistantMessage,
    ModelAdapter,
    ModelResponse,
    ProviderCompletionStatus,
    TextContent,
)
from tend.llm.testing import ScriptedModel


def _write_agent(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "system_prompt": "You are a concise test agent.",
                "model": {
                    "provider": "cloudflare_openai",
                    "api": "openai_responses",
                    "model_name": "gpt-5",
                    "settings": {"reasoning": {"effort": "minimal"}},
                },
                "tools": {"enabled": []},
            }
        ),
        encoding="utf-8",
    )
    return path


def _final_response(text: str = "final text") -> ModelResponse:
    return ModelResponse(
        assistant_message=AssistantMessage(content=[TextContent(text=text)]),
        stop_reason=StopReason.FINAL_RESPONSE,
    )


def _scripted_factory(response: ModelResponse) -> ModelAdapterFactory:
    scripted = ScriptedModel([response])

    def factory(
        model_config: AgentModelConfig,
        runtime_config: RuntimeConfig,
        environment: Mapping[str, str],
    ) -> ModelAdapter:
        del model_config, runtime_config, environment
        return scripted

    return factory


async def test_default_output_prints_only_final_response_to_stdout(tmp_path: Path) -> None:
    agent_path = _write_agent(tmp_path / "agent.json")
    stdout = StringIO()
    stderr = StringIO()

    code = await run_cli(
        ["--agent", str(agent_path), "--prompt", "hello", "--no-compaction"],
        stdout=stdout,
        stderr=stderr,
        environment={},
        model_factory=_scripted_factory(_final_response("hello back")),
    )

    assert code == ExitCode.FINAL_RESPONSE
    assert stdout.getvalue() == "hello back\n"
    assert stderr.getvalue() == ""


async def test_default_output_prints_nothing_for_empty_final_response(tmp_path: Path) -> None:
    agent_path = _write_agent(tmp_path / "agent.json")
    stdout = StringIO()
    stderr = StringIO()

    code = await run_cli(
        ["--agent", str(agent_path), "--prompt", "hello", "--no-compaction"],
        stdout=stdout,
        stderr=stderr,
        environment={},
        model_factory=_scripted_factory(ModelResponse()),
    )

    assert code == ExitCode.FINAL_RESPONSE
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == ""


async def test_json_output_emits_only_turn_result_json_to_stdout(tmp_path: Path) -> None:
    agent_path = _write_agent(tmp_path / "agent.json")
    stdout = StringIO()
    stderr = StringIO()

    code = await run_cli(
        ["--agent", str(agent_path), "--prompt", "hello", "--json", "--no-compaction"],
        stdout=stdout,
        stderr=stderr,
        environment={},
        model_factory=_scripted_factory(_final_response("json final")),
    )

    assert code == ExitCode.FINAL_RESPONSE
    payload = json.loads(stdout.getvalue())
    assert payload["final_response"] == "json final"
    assert payload["stop_reason"] == "final_response"
    assert stderr.getvalue() == ""


async def test_default_output_prints_final_result_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FinalResultAgent:
        async def run_turn(
            self,
            prompt: str,
            *,
            session: object | None = None,
            config: object | None = None,
            cancellation: object | None = None,
            clock: object | None = None,
        ) -> TurnResult:
            del prompt, session, config, cancellation, clock
            return TurnResult(
                turn_id="turn_1",
                stop_reason=StopReason.FINAL_RESULT,
                final_result=FinalResultOutput(
                    tool_call_id="call_final",
                    output={"message": "ok", "count": 1},
                    arguments={"message": "ok"},
                ),
            )

    def from_config(config: object, *, model: object) -> FinalResultAgent:
        del config, model
        return FinalResultAgent()

    monkeypatch.setattr(agent_cli.Agent, "from_config", from_config)
    agent_path = _write_agent(tmp_path / "agent.json")
    stdout = StringIO()
    stderr = StringIO()

    code = await run_cli(
        ["--agent", str(agent_path), "--prompt", "hello", "--no-compaction"],
        stdout=stdout,
        stderr=stderr,
        environment={},
        model_factory=_scripted_factory(_final_response("unused")),
    )

    assert code == ExitCode.FINAL_RESPONSE
    assert stdout.getvalue() == '{"count":1,"message":"ok"}\n'
    assert stderr.getvalue() == ""


async def test_non_final_stop_returns_exit_code_one_and_diagnostic_stderr(
    tmp_path: Path,
) -> None:
    agent_path = _write_agent(tmp_path / "agent.json")
    stdout = StringIO()
    stderr = StringIO()

    code = await run_cli(
        ["--agent", str(agent_path), "--prompt", "hello", "--no-compaction"],
        stdout=stdout,
        stderr=stderr,
        environment={},
        model_factory=_scripted_factory(
            ModelResponse(
                stop_reason=StopReason.MAX_TOKENS,
                provider_completion_status=ProviderCompletionStatus.INCOMPLETE,
            )
        ),
    )

    assert code == ExitCode.NON_FINAL_STOP
    assert stdout.getvalue() == ""
    assert "max_tokens" in stderr.getvalue()


async def test_interrupted_stop_returns_interrupted_exit_code(tmp_path: Path) -> None:
    agent_path = _write_agent(tmp_path / "agent.json")
    stdout = StringIO()
    stderr = StringIO()

    code = await run_cli(
        ["--agent", str(agent_path), "--prompt", "hello", "--no-compaction"],
        stdout=stdout,
        stderr=stderr,
        environment={},
        model_factory=_scripted_factory(ModelResponse(stop_reason=StopReason.INTERRUPTED)),
    )

    assert code == ExitCode.INTERRUPTED
    assert stdout.getvalue() == ""
    assert "interrupted" in stderr.getvalue()


async def test_internal_framework_error_returns_internal_software_exit_code(
    tmp_path: Path,
) -> None:
    agent_path = _write_agent(tmp_path / "agent.json")
    stdout = StringIO()
    stderr = StringIO()

    def factory(
        model_config: AgentModelConfig,
        runtime_config: RuntimeConfig,
        environment: Mapping[str, str],
    ) -> ModelAdapter:
        del model_config, runtime_config, environment
        raise FrameworkError("boom")

    code = await run_cli(
        ["--agent", str(agent_path), "--prompt", "hello", "--no-compaction"],
        stdout=stdout,
        stderr=stderr,
        environment={},
        model_factory=factory,
    )

    assert code == ExitCode.INTERNAL_SOFTWARE
    assert stdout.getvalue() == ""
    error = json.loads(stderr.getvalue())
    assert error["code"] == "framework_error"
