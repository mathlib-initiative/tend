from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import Mapping
from io import StringIO
from pathlib import Path

from tend.agent.cli import ExitCode, ModelAdapterFactory, run_cli
from tend.agent.config import AgentModelConfig, RuntimeConfig
from tend.agent.persistence.events import EventType, parse_event_json
from tend.llm.models import ModelAdapter, ModelRequest, ModelResponse


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


class _SignalOnGenerateModel:
    @property
    def profile(self) -> None:
        return None

    async def generate(self, request: ModelRequest) -> ModelResponse:
        del request
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.sleep(30.0)
        raise AssertionError("signal should cancel the model request")


def _signal_factory() -> ModelAdapterFactory:
    def factory(
        model_config: AgentModelConfig,
        runtime_config: RuntimeConfig,
        environment: Mapping[str, str],
    ) -> ModelAdapter:
        del model_config, runtime_config, environment
        return _SignalOnGenerateModel()

    return factory


async def test_cli_signal_cancels_turn_and_records_interruption(
    tmp_path: Path,
) -> None:
    agent_path = _write_agent(tmp_path / "agent.json")
    session_dir = tmp_path / "session"
    stdout = StringIO()
    stderr = StringIO()

    code = await run_cli(
        [
            "--agent",
            str(agent_path),
            "--prompt",
            "wait",
            "--session-dir",
            str(session_dir),
            "--no-compaction",
        ],
        stdout=stdout,
        stderr=stderr,
        environment={},
        model_factory=_signal_factory(),
    )

    assert code == ExitCode.INTERRUPTED
    assert stdout.getvalue() == ""
    error = json.loads(stderr.getvalue())
    assert error["code"] == "interrupted"
    events = [
        parse_event_json(line)
        for line in (session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event.event_type is EventType.TURN_INTERRUPTED for event in events)
