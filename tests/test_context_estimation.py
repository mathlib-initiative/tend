from __future__ import annotations

from tend.llm.context_estimation import (
    ContextEstimate,
    TokenEstimatorConfig,
    estimate_context,
    estimate_context_from_api_anchor,
)
from tend.llm.models import (
    ContextWindow,
    ModelProfile,
    ProviderApi,
    TextContent,
    UserMessage,
)


def _message(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)])


def test_api_anchor_estimate_is_anchor_plus_new_message_tokens() -> None:
    config = TokenEstimatorConfig(chars_per_token=2.0)

    estimate = estimate_context_from_api_anchor(
        anchor_tokens=1000,
        new_messages=[_message("abcdefgh")],
        config=config,
    )

    # 4 (per message) + 2 ("user" / 2) + 1 (content part) + 4 (8 chars / 2) = 11.
    assert estimate.estimated_tokens == 1011
    assert estimate.message_tokens == 1011
    # Tool schemas and stable history are already baked into the anchor.
    assert estimate.tool_schema_tokens == 0
    assert estimate.reasoning_setting_tokens == 0
    assert estimate.estimator == "api_anchor"
    assert estimate.is_api_anchored is True


def test_configured_estimator_label_cannot_forge_api_anchor_provenance() -> None:
    estimate = estimate_context(
        messages=[_message("tiny")],
        config=TokenEstimatorConfig(estimator_name="api_anchor"),
    )

    assert estimate.estimator == "api_anchor"
    assert estimate.is_api_anchored is False


def test_old_context_estimate_metadata_defaults_to_non_anchored() -> None:
    estimate = ContextEstimate.model_validate(
        {
            "estimated_tokens": 10,
            "message_tokens": 10,
            "estimator": "api_anchor",
        }
    )

    assert estimate.is_api_anchored is False


def test_api_anchor_estimate_with_no_new_messages_equals_anchor() -> None:
    estimate = estimate_context_from_api_anchor(anchor_tokens=1234, new_messages=[])

    assert estimate.estimated_tokens == 1234
    assert estimate.message_tokens == 1234
    assert estimate.estimator == "api_anchor"


def test_api_anchor_estimate_includes_profile_window_percentages() -> None:
    profile = ModelProfile(
        provider_name="scripted_provider",
        model_name="scripted_model",
        api=ProviderApi.OPENAI_RESPONSES,
        context_window=ContextWindow(tokens=2000),
    )

    estimate = estimate_context_from_api_anchor(
        anchor_tokens=1000,
        new_messages=[],
        profile=profile,
    )

    assert estimate.context_window_tokens == 2000
    assert estimate.remaining_context_tokens == 1000
    assert estimate.context_usage_ratio == 0.5
    assert estimate.context_usage_percent == 50.0
