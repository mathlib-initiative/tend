from __future__ import annotations

import pytest

from tend._common.errors import ConfigurationError
from tend.llm.context_estimation import (
    TokenEstimatorConfig,
    estimate_context_from_api_anchor,
    estimate_model_request_context,
    estimate_request_tokens,
)
from tend.llm.history import assistant_message_from_response
from tend.llm.models import (
    AssistantMessage,
    DeveloperMessage,
    ModelRequest,
    ModelResponse,
    ReasoningContinuationMetadata,
    ReasoningMetadata,
    SystemMessage,
    TextContent,
    UserMessage,
)
from tend.llm.providers import AnthropicMessagesAdapter, OpenAIResponsesAdapter


def _assistant(
    *,
    provider: str = "anthropic",
    kind: str = "thinking",
    tokens: int | None = 10_000,
) -> AssistantMessage:
    return assistant_message_from_response(
        ModelResponse(
            assistant_message=AssistantMessage(content=[TextContent(text="answer")]),
            reasoning=ReasoningMetadata(
                reasoning_tokens=tokens,
                provider_private_continuation=[
                    ReasoningContinuationMetadata(
                        provider_name=provider,
                        kind=kind,
                        signature="opaque-signature" * 100,
                        encrypted_content="opaque-data" * 100
                        if kind == "redacted_thinking"
                        else None,
                        redacted_details={"thinking": "retained reasoning" * 100},
                    )
                ],
            ),
        )
    )


@pytest.mark.parametrize("kind", ["thinking", "redacted_thinking"])
@pytest.mark.parametrize("tokens", [None, 10_000])
def test_anthropic_counts_only_replayed_reasoning_once(kind: str, tokens: int | None) -> None:
    adapter = AnthropicMessagesAdapter(model_name="test")
    plain = AssistantMessage(content=[TextContent(text="answer")])
    request = ModelRequest(messages=[_assistant(kind=kind, tokens=tokens)])
    baseline = estimate_request_tokens(ModelRequest(messages=[plain]), token_estimator=adapter)
    estimate = estimate_request_tokens(request, token_estimator=adapter)
    if tokens is not None:
        assert estimate.message_tokens[0] == baseline.message_tokens[0] + tokens
    else:
        assert estimate.message_tokens[0] > baseline.message_tokens[0] + 100
    # The same adapter estimate drives request observability and post-anchor deltas.
    context = estimate_model_request_context(request, token_estimator=adapter)
    anchor = estimate_context_from_api_anchor(
        anchor_tokens=100,
        new_messages=request.messages,
        token_estimator=adapter,
    )
    assert context.message_tokens == estimate.message_tokens[0]
    assert anchor.estimated_tokens == 100 + estimate.message_tokens[0]


@pytest.mark.parametrize(
    "adapter",
    [
        AnthropicMessagesAdapter(model_name="test"),
        OpenAIResponsesAdapter(model_name="test"),
    ],
)
def test_metadata_not_serialized_by_adapter_has_no_token_cost(
    adapter: AnthropicMessagesAdapter | OpenAIResponsesAdapter,
) -> None:
    # OpenAI currently does not replay stored reasoning; Anthropic filters foreign continuations.
    message = _assistant(provider="openai")
    plain = AssistantMessage(content=message.content)
    assert estimate_request_tokens(
        ModelRequest(messages=[message]),
        token_estimator=adapter,
    ) == estimate_request_tokens(ModelRequest(messages=[plain]), token_estimator=adapter)


def test_anthropic_raw_thinking_blocks_are_counted_without_usage() -> None:
    adapter = AnthropicMessagesAdapter(model_name="test")
    message = AssistantMessage(
        provider_metadata={
            "anthropic_content_blocks": [
                {"type": "thinking", "thinking": "x" * 20_000, "signature": "sig"},
            ]
        }
    )
    assert (
        estimate_request_tokens(
            ModelRequest(messages=[message]),
            token_estimator=adapter,
        ).message_tokens[0]
        > 10_000
    )


@pytest.mark.parametrize(
    "adapter",
    [
        AnthropicMessagesAdapter(model_name="test"),
        OpenAIResponsesAdapter(model_name="test"),
    ],
)
def test_adapter_separates_fixed_costs_from_message_costs(
    adapter: AnthropicMessagesAdapter | OpenAIResponsesAdapter,
) -> None:
    request = ModelRequest(messages=[UserMessage(content=[TextContent(text="hello")])])
    with_tools = request.model_copy(
        update={
            "tools": [
                {
                    "name": "tool",
                    "description": "description" * 100,
                    "arguments_schema": {"type": "object", "properties": {}},
                }
            ]
        }
    )
    before = adapter.estimate_request_tokens(request, TokenEstimatorConfig())
    after = adapter.estimate_request_tokens(with_tools, TokenEstimatorConfig())
    assert after.message_tokens == before.message_tokens
    assert after.tool_schema_tokens > before.tool_schema_tokens + 500


@pytest.mark.parametrize(
    "message",
    [
        AssistantMessage(),
        UserMessage(),
        SystemMessage(),
        DeveloperMessage(),
        _assistant(provider="openai").model_copy(update={"content": []}),
    ],
)
def test_anthropic_omitted_messages_have_zero_token_cost(
    message: AssistantMessage | UserMessage | SystemMessage | DeveloperMessage,
) -> None:
    adapter = AnthropicMessagesAdapter(model_name="test")
    request = ModelRequest(
        messages=[
            UserMessage(content=[TextContent(text="before")]),
            message,
            UserMessage(content=[TextContent(text="after")]),
        ]
    )
    assert adapter.build_payload(request)["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "before"}, {"type": "text", "text": "after"}],
        }
    ]
    estimates = estimate_request_tokens(request, token_estimator=adapter)
    assert estimates.message_tokens[1] == 0
    assert (
        estimate_context_from_api_anchor(
            anchor_tokens=100,
            new_messages=[message],
            token_estimator=adapter,
        ).estimated_tokens
        == 100
    )
    # Partial estimation may have no serialized messages. A complete request still may not.
    assert estimate_request_tokens(
        ModelRequest(messages=[message]),
        token_estimator=adapter,
    ).message_tokens == [0]
    with pytest.raises(ConfigurationError, match="require at least one message"):
        adapter.build_payload(ModelRequest(messages=[message]))
