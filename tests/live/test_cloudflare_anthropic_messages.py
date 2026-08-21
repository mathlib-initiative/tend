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
from tend.llm.providers import AnthropicMessagesAdapter
from tests.live._helpers import (
    RecordingModelAdapter,
    cloudflare_anthropic_base_url,
    cloudflare_auth_headers,
    echo_tool,
    json_object,
    live_runtime_config,
    redactor_for_base_url,
)

_ANTHROPIC_PROVIDER = "cloudflare_anthropic"
_ANTHROPIC_MODEL = "claude-sonnet-4-5"

pytestmark = pytest.mark.live


async def test_cloudflare_anthropic_messages_plain_text_smoke() -> None:
    """Run one tiny native Anthropic Messages final-response turn through Agent.run_turn."""

    adapter = _anthropic_adapter(profile=_cloudflare_sonnet_profile())
    recording = RecordingModelAdapter(adapter)
    agent = Agent(
        "You are a concise live compatibility smoke-test assistant.",
        model=recording,
        max_output_tokens=32,
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


async def test_cloudflare_anthropic_messages_tool_use_and_tool_result_continuation() -> None:
    """Force one tiny native tool use without thinking, then continue to final text."""

    first_request_metadata = json_object(
        {"anthropic_tool_choice": {"type": "tool", "name": "echo"}}
    )
    adapter = _anthropic_adapter(profile=_cloudflare_sonnet_profile())
    recording = RecordingModelAdapter(adapter, first_metadata=first_request_metadata)
    agent = Agent(
        "You are a concise live compatibility tool-use assistant.",
        model=recording,
        tools=[echo_tool()],
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


async def test_cloudflare_anthropic_force_tool_name_with_disable_reasoning_overrides_thinking() -> (
    None
):
    """force_tool_name + disable_reasoning must force a live tool call despite default thinking.

    Cloudflare sonnet marks forced tool choice incompatible with thinking, so an adapter
    configured with default reasoning would otherwise re-enable thinking on a
    ``reasoning=None`` request and silently drop the forced tool choice. ``disable_reasoning``
    (the turn loop's authoritative signal) suppresses the default thinking so the force lands.
    """

    first_request_metadata = json_object(
        {"force_tool_name": "echo", "disable_reasoning": True}
    )
    adapter = _anthropic_adapter(
        profile=_cloudflare_sonnet_profile(),
        default_reasoning=ReasoningSettings(
            effort=ReasoningEffort.LOW, max_reasoning_tokens=1_024
        ),
        set_default_temperature=False,
    )
    recording = RecordingModelAdapter(adapter, first_metadata=first_request_metadata)
    agent = Agent(
        "You are a concise live compatibility tool-use assistant.",
        model=recording,
        tools=[echo_tool()],
        max_output_tokens=2_048,
    )

    result = await agent.run_turn(
        "Answer the user briefly.",
        config=live_runtime_config(),
    )

    responses = recording.responses
    assert len(responses) >= 1
    assert responses[0].tool_calls, "forced tool choice was dropped despite disable_reasoning"
    assert responses[0].tool_calls[0].tool_name == "echo"
    assert result.tool_call_count >= 1


def _anthropic_adapter(
    *,
    profile: ModelProfile,
    default_reasoning: ReasoningSettings | None = None,
    set_default_temperature: bool = True,
) -> AnthropicMessagesAdapter:
    base_url = cloudflare_anthropic_base_url()
    # Thinking is incompatible with an explicit temperature on Anthropic, so callers that
    # configure default reasoning opt out of the default temperature.
    default_request_settings = {"temperature": 0.0} if set_default_temperature else {}
    return AnthropicMessagesAdapter(
        model_name=_ANTHROPIC_MODEL,
        provider_name=_ANTHROPIC_PROVIDER,
        base_url=base_url,
        profile=profile,
        raw_headers=cloudflare_auth_headers(),
        timeout_seconds=90.0,
        redactor=redactor_for_base_url(base_url),
        default_reasoning=default_reasoning,
        default_request_settings=default_request_settings,
    )


def _cloudflare_sonnet_profile() -> ModelProfile:
    profile = get_builtin_profile(
        ProviderApi.ANTHROPIC_MESSAGES,
        _ANTHROPIC_PROVIDER,
        _ANTHROPIC_MODEL,
    )
    assert profile is not None
    return profile
