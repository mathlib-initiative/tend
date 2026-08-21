from __future__ import annotations

import pytest

from tend import Agent
from tend._common.types import StopReason
from tend.llm.models import (
    ModelProfile,
    ProviderApi,
    ReasoningEffort,
    ReasoningSettings,
    get_builtin_profile,
)
from tend.llm.providers import OpenAIResponsesAdapter
from tests.live._helpers import (
    RecordingModelAdapter,
    cloudflare_auth_headers,
    cloudflare_openai_base_url,
    echo_tool,
    json_object,
    live_runtime_config,
    redactor_for_base_url,
)

_OPENAI_PROVIDER = "cloudflare_openai"
_OPENAI_MODEL = "gpt-5"

pytestmark = pytest.mark.live


async def test_cloudflare_openai_responses_plain_text_smoke() -> None:
    """Run one tiny OpenAI Responses final-response turn through Agent.run_turn."""

    adapter = _openai_adapter(profile=_cloudflare_gpt5_profile())
    recording = RecordingModelAdapter(adapter)
    agent = Agent(
        "You are a concise live compatibility smoke-test assistant.",
        model=recording,
        reasoning=ReasoningSettings(effort=ReasoningEffort.MINIMAL),
        max_output_tokens=64,
    )

    result = await agent.run_turn("Reply with exactly: ok", config=live_runtime_config())

    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.final_response is not None
    assert result.final_response.strip()
    assert result.model_request_count == 1
    assert result.tool_call_count == 0
    assert result.usage.model_requests == 1
    assert result.usage.tokens.input_tokens > 0
    assert result.usage.tokens.output_tokens > 0

    responses = recording.responses
    assert len(responses) == 1
    assert responses[0].response_id is not None


async def test_cloudflare_openai_responses_tool_call_and_stateless_continuation() -> None:
    """Force one tiny function call, then let the follow-up request produce final text."""

    first_request_metadata = json_object(
        {
            "openai_responses_request_settings": {
                "tool_choice": {"type": "function", "name": "echo"}
            }
        }
    )
    adapter = _openai_adapter(profile=_cloudflare_gpt5_profile(allow_tool_choice=True))
    recording = RecordingModelAdapter(adapter, first_metadata=first_request_metadata)
    agent = Agent(
        "You are a concise live compatibility tool-call assistant.",
        model=recording,
        tools=[echo_tool()],
        reasoning=ReasoningSettings(effort=ReasoningEffort.MINIMAL),
        max_output_tokens=128,
    )

    result = await agent.run_turn(
        "Call the echo tool once with text live-ping, then answer briefly.",
        config=live_runtime_config(),
    )

    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.final_response is not None
    assert result.final_response.strip()
    assert result.model_request_count == 2
    assert result.tool_call_count == 1
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "echo"
    assert len(result.tool_results) == 1
    assert result.tool_results[0].success is True
    assert result.usage.model_requests == 2
    assert result.usage.tool_calls == 1
    assert result.usage.tokens.input_tokens > 0
    assert result.usage.tokens.output_tokens > 0

    responses = recording.responses
    assert len(responses) == 2
    assert responses[0].response_id is not None
    assert responses[0].tool_calls[0].tool_name == "echo"
    assert responses[1].response_id is not None


async def test_cloudflare_openai_responses_force_tool_name_metadata_forces_tool() -> None:
    """The turn loop's ``force_tool_name`` metadata must translate into a live forced tool call.

    Uses the plain profile (no ``tool_choice`` in supported_extra_settings) to prove the
    adapter forces the tool from metadata alone, which the metadata-only path did not do
    before this fix.
    """

    first_request_metadata = json_object({"force_tool_name": "echo"})
    adapter = _openai_adapter(profile=_cloudflare_gpt5_profile())
    recording = RecordingModelAdapter(adapter, first_metadata=first_request_metadata)
    agent = Agent(
        "You are a concise live compatibility tool-call assistant.",
        model=recording,
        tools=[echo_tool()],
        reasoning=ReasoningSettings(effort=ReasoningEffort.MINIMAL),
        max_output_tokens=128,
    )

    result = await agent.run_turn(
        "Answer the user briefly.",
        config=live_runtime_config(),
    )

    responses = recording.responses
    assert len(responses) >= 1
    assert responses[0].tool_calls, "force_tool_name metadata did not force a tool call"
    assert responses[0].tool_calls[0].tool_name == "echo"
    assert result.tool_call_count >= 1


def _openai_adapter(*, profile: ModelProfile) -> OpenAIResponsesAdapter:
    base_url = cloudflare_openai_base_url()
    return OpenAIResponsesAdapter(
        model_name=_OPENAI_MODEL,
        provider_name=_OPENAI_PROVIDER,
        base_url=base_url,
        profile=profile,
        raw_headers=cloudflare_auth_headers(),
        timeout_seconds=90.0,
        redactor=redactor_for_base_url(base_url),
    )


def _cloudflare_gpt5_profile(*, allow_tool_choice: bool = False) -> ModelProfile:
    profile = get_builtin_profile(ProviderApi.OPENAI_RESPONSES, _OPENAI_PROVIDER, _OPENAI_MODEL)
    assert profile is not None
    if not allow_tool_choice:
        return profile

    settings = profile.settings.model_copy(update={"supported_extra_settings": ["tool_choice"]})
    return profile.model_copy(update={"settings": settings}, deep=True)
