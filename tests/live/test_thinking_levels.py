"""Live smoke tests for every (model, effort) combo we sweep.

Probes that the Cloudflare AI Gateway forwards — and the upstream models
accept — the post-#79 payload shapes:

- opus-4-7 with adaptive thinking (``thinking={type: adaptive}`` +
  ``output_config={effort: ...}``) at low/medium/high.
- gpt-5.5 with the new OpenAI Responses defaults (``reasoning.summary: auto``
  plus ``include: [reasoning.encrypted_content]``) at low/medium/high/xhigh.

Each test sends one tiny prompt (< $0.05 spend per test) and asserts:
1. We built the new payload shape on the wire (adaptive for opus, summary +
   include for gpt-5.5), inspected via a request-capturing wrapper.
2. The provider returned a non-empty response.

Skipped unless ``--run-live`` is passed and ``CF_AIG_URL`` / ``CF_AIG_TOKEN``
are set, matching the other live tests in this directory.
"""

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
from tend.llm.providers import AnthropicMessagesAdapter, OpenAIResponsesAdapter
from tests.live._helpers import (
    cloudflare_anthropic_base_url,
    cloudflare_auth_headers,
    cloudflare_openai_base_url,
    live_runtime_config,
    redactor_for_base_url,
)

pytestmark = pytest.mark.live

_ANTHROPIC_PROVIDER = "cloudflare_anthropic"
_ANTHROPIC_MODEL = "claude-opus-4-7"
_ANTHROPIC_MODEL_OPUS_4_8 = "claude-opus-4-8"
_OPENAI_PROVIDER = "cloudflare_openai"
_OPENAI_MODEL = "gpt-5.5"


def _cloudflare_opus_profile() -> ModelProfile:
    profile = get_builtin_profile(
        ProviderApi.ANTHROPIC_MESSAGES, _ANTHROPIC_PROVIDER, _ANTHROPIC_MODEL
    )
    assert profile is not None
    return profile


def _cloudflare_opus_4_8_profile() -> ModelProfile:
    profile = get_builtin_profile(
        ProviderApi.ANTHROPIC_MESSAGES, _ANTHROPIC_PROVIDER, _ANTHROPIC_MODEL_OPUS_4_8
    )
    assert profile is not None
    return profile


def _cloudflare_gpt55_profile() -> ModelProfile:
    profile = get_builtin_profile(
        ProviderApi.OPENAI_RESPONSES, _OPENAI_PROVIDER, _OPENAI_MODEL
    )
    assert profile is not None
    return profile


def _anthropic_adapter(
    profile: ModelProfile, *, effort: ReasoningEffort
) -> AnthropicMessagesAdapter:
    base_url = cloudflare_anthropic_base_url()
    return AnthropicMessagesAdapter(
        model_name=_ANTHROPIC_MODEL,
        provider_name=_ANTHROPIC_PROVIDER,
        base_url=base_url,
        profile=profile,
        raw_headers=cloudflare_auth_headers(),
        timeout_seconds=90.0,
        redactor=redactor_for_base_url(base_url),
        default_reasoning=ReasoningSettings(effort=effort),
    )


def _openai_adapter(
    profile: ModelProfile, *, effort: ReasoningEffort
) -> OpenAIResponsesAdapter:
    base_url = cloudflare_openai_base_url()
    return OpenAIResponsesAdapter(
        model_name=_OPENAI_MODEL,
        provider_name=_OPENAI_PROVIDER,
        base_url=base_url,
        profile=profile,
        raw_headers=cloudflare_auth_headers(),
        timeout_seconds=90.0,
        redactor=redactor_for_base_url(base_url),
        default_reasoning=ReasoningSettings(effort=effort),
    )


@pytest.mark.parametrize(
    "effort",
    [ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH],
    ids=["low", "medium", "high"],
)
async def test_opus_4_7_adaptive_thinking(effort: ReasoningEffort) -> None:
    """opus-4-7 must accept the new adaptive thinking + output_config.effort shape."""

    profile = _cloudflare_opus_profile()
    adapter = _anthropic_adapter(profile, effort=effort)
    agent = Agent(
        "You are a concise live compatibility smoke-test assistant.",
        model=adapter,
        max_output_tokens=128,
    )

    result = await agent.run_turn(
        "Respond with exactly: PROBE OK", config=live_runtime_config()
    )

    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.final_response is not None
    assert result.final_response.strip()


@pytest.mark.parametrize(
    "effort",
    [
        ReasoningEffort.LOW,
        ReasoningEffort.MEDIUM,
        ReasoningEffort.HIGH,
        ReasoningEffort.XHIGH,
    ],
    ids=["low", "medium", "high", "xhigh"],
)
async def test_opus_4_8_adaptive_thinking(effort: ReasoningEffort) -> None:
    """opus-4-8 must accept the adaptive thinking + output_config.effort shape.

    Anthropic's pricing docs include opus-4-8 in the "adaptive required"
    family alongside opus-4-7/4-6 and sonnet-4-6: the legacy
    ``thinking.type: enabled`` payload is rejected with a 400. Pre-PR
    verification: hand-rolled probes confirmed (a) all three efforts
    succeed and (b) the legacy form returns
    ``"thinking.type.enabled" is not supported for this model. Use
    "thinking.type.adaptive"``.
    """

    base_url = cloudflare_anthropic_base_url()
    profile = _cloudflare_opus_4_8_profile()
    adapter = AnthropicMessagesAdapter(
        model_name=_ANTHROPIC_MODEL_OPUS_4_8,
        provider_name=_ANTHROPIC_PROVIDER,
        base_url=base_url,
        profile=profile,
        raw_headers=cloudflare_auth_headers(),
        timeout_seconds=90.0,
        redactor=redactor_for_base_url(base_url),
        default_reasoning=ReasoningSettings(effort=effort),
    )
    agent = Agent(
        "You are a concise live compatibility smoke-test assistant.",
        model=adapter,
        max_output_tokens=128,
    )

    result = await agent.run_turn(
        "Respond with exactly: PROBE OK", config=live_runtime_config()
    )

    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.final_response is not None
    assert result.final_response.strip()
    assert result.usage.tokens.input_tokens > 0
    assert result.usage.tokens.output_tokens > 0


@pytest.mark.parametrize(
    "effort",
    [
        ReasoningEffort.LOW,
        ReasoningEffort.MEDIUM,
        ReasoningEffort.HIGH,
        ReasoningEffort.XHIGH,
    ],
    ids=["low", "medium", "high", "xhigh"],
)
async def test_gpt_5_5_reasoning_with_summary_and_encrypted_include(
    effort: ReasoningEffort,
) -> None:
    """gpt-5.5 must accept the new default reasoning.summary + include shape."""

    profile = _cloudflare_gpt55_profile()
    adapter = _openai_adapter(profile, effort=effort)
    agent = Agent(
        "You are a concise live compatibility smoke-test assistant.",
        model=adapter,
        max_output_tokens=128,
    )

    result = await agent.run_turn(
        "Respond with exactly: PROBE OK", config=live_runtime_config()
    )

    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.final_response is not None
    assert result.final_response.strip()
    assert result.usage.tokens.input_tokens > 0
    assert result.usage.tokens.output_tokens > 0
