from __future__ import annotations

import json
from collections.abc import Mapping
from io import StringIO
from pathlib import Path

from tend import Agent
from tend._common.types import StopReason
from tend.agent.cli import ExitCode, ModelAdapterFactory, run_cli
from tend.agent.config import AgentModelConfig, CompactionConfig, RuntimeConfig
from tend.agent.persistence.events import EventType, TurnCompletedEvent
from tend.agent.session import Session
from tend.llm.models import AssistantMessage, ModelAdapter, ModelResponse, TextContent
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


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(compaction=CompactionConfig(enabled=False))


def _final_response(text: str, *, response_id: str) -> ModelResponse:
    return ModelResponse(
        response_id=response_id,
        assistant_message=AssistantMessage(content=[TextContent(text=text)]),
        stop_reason=StopReason.FINAL_RESPONSE,
    )


def _factory_for(scripted: ScriptedModel) -> ModelAdapterFactory:
    def factory(
        model_config: AgentModelConfig,
        runtime_config: RuntimeConfig,
        environment: Mapping[str, str],
    ) -> ModelAdapter:
        del model_config, runtime_config, environment
        return scripted

    return factory


async def test_cli_created_session_can_be_resumed_by_library_api(tmp_path: Path) -> None:
    agent_path = _write_agent(tmp_path / "agent.json")
    session_dir = tmp_path / "cli_session"
    cli_model = ScriptedModel([_final_response("cli done", response_id="model_resp_cli")])
    stdout = StringIO()
    stderr = StringIO()

    code = await run_cli(
        [
            "--agent",
            str(agent_path),
            "--prompt",
            "Create a CLI session.",
            "--session-dir",
            str(session_dir),
            "--session-id",
            "sess_cli_to_library",
            "--no-compaction",
        ],
        stdin=StringIO("ignored"),
        stdout=stdout,
        stderr=stderr,
        environment={},
        model_factory=_factory_for(cli_model),
    )

    assert code == ExitCode.FINAL_RESPONSE, stderr.getvalue()
    assert stdout.getvalue() == "cli done\n"

    library_model = ScriptedModel(
        [_final_response("library resumed", response_id="model_resp_library")]
    )
    library_agent = Agent(
        "You are a concise test agent.",
        model=library_model,
        model_name="scripted",
    )
    with Session.resume(session_dir, sync_writes=False) as session:
        result = await library_agent.run_turn(
            "Resume through the library.",
            session=session,
            config=_runtime_config(),
        )
        events = session.event_store.read_all()

    assert result.final_response == "library resumed"
    assert [event.event_type for event in events].count(EventType.SESSION_STARTED) == 1
    assert [event.event_type for event in events].count(EventType.SESSION_RESUMED) == 1
    completed_turns = [event for event in events if isinstance(event, TurnCompletedEvent)]
    assert [event.payload.final_response for event in completed_turns] == [
        "cli done",
        "library resumed",
    ]


async def test_library_created_session_can_be_resumed_by_cli(tmp_path: Path) -> None:
    agent_path = _write_agent(tmp_path / "agent.json")
    session_dir = tmp_path / "library_session"
    library_model = ScriptedModel(
        [_final_response("library done", response_id="model_resp_library_first")]
    )
    library_agent = Agent("You are a concise test agent.", model=library_model)

    with Session.create(
        session_dir,
        session_id="sess_library_to_cli",
        sync_writes=False,
    ) as session:
        first = await library_agent.run_turn(
            "Create a library session.",
            session=session,
            config=_runtime_config(),
        )

    assert first.final_response == "library done"

    cli_model = ScriptedModel([_final_response("cli resumed", response_id="model_resp_cli")])
    stdout = StringIO()
    stderr = StringIO()
    code = await run_cli(
        [
            "--agent",
            str(agent_path),
            "--prompt",
            "Resume through the CLI.",
            "--session-dir",
            str(session_dir),
            "--resume-session",
            "--no-compaction",
        ],
        stdin=StringIO("ignored"),
        stdout=stdout,
        stderr=stderr,
        environment={},
        model_factory=_factory_for(cli_model),
    )

    assert code == ExitCode.FINAL_RESPONSE, stderr.getvalue()
    assert stdout.getvalue() == "cli resumed\n"

    with Session.open(session_dir, writable=False, sync_writes=False) as read_only:
        events = read_only.event_store.read_all()
        state = read_only.state

    assert [event.event_type for event in events].count(EventType.SESSION_STARTED) == 1
    assert [event.event_type for event in events].count(EventType.SESSION_RESUMED) == 1
    completed_turns = [event for event in events if isinstance(event, TurnCompletedEvent)]
    assert [event.payload.final_response for event in completed_turns] == [
        "library done",
        "cli resumed",
    ]
    assert len(state.completed_model_requests) == 2
    assert state.event_count == len(events)
