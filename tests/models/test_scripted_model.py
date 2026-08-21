import pytest

from tend._common.errors import ProviderProtocolError
from tend._common.types import StopReason
from tend.llm.models import (
    AssistantMessage,
    ModelAdapter,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ProviderApi,
    TextContent,
    ToolCall,
    UserMessage,
)
from tend.llm.testing import ScriptedModel, ScriptExhaustedError


async def test_scripted_model_returns_final_response_and_records_requests() -> None:
    request = ModelRequest(
        request_id="model_req_1",
        messages=[UserMessage(message_id="msg_user_1", content=[TextContent(text="hello")])],
    )
    scripted_response = ModelResponse(
        response_id="model_resp_1",
        assistant_message=AssistantMessage(
            message_id="msg_assistant_1",
            content=[TextContent(text="ok")],
        ),
        stop_reason=StopReason.FINAL_RESPONSE,
    )
    model = ScriptedModel([scripted_response])
    adapter: ModelAdapter = model

    response = await adapter.generate(request)

    assert response.final_text == "ok"
    assert response.request_id == "model_req_1"
    assert model.requests == (request,)
    assert model.last_request == request
    assert model.remaining_steps == 0


async def test_scripted_model_can_return_tool_call_responses() -> None:
    tool_call = ToolCall(
        call_id="call_1",
        tool_name="read_file",
        arguments={"path": "README.md"},
        order=0,
        provider_tool_use_id="toolu_1",
    )
    model = ScriptedModel([ModelResponse(response_id="model_resp_1", tool_calls=[tool_call])])

    response = await model.generate(ModelRequest(request_id="model_req_1"))

    assert response.tool_calls == [tool_call]
    assert response.final_text is None


async def test_scripted_model_raises_scripted_exceptions_then_continues() -> None:
    model = ScriptedModel(
        [
            ProviderProtocolError("temporary protocol failure"),
            ModelResponse(
                response_id="model_resp_2",
                assistant_message=AssistantMessage(content=[TextContent(text="recovered")]),
            ),
        ]
    )

    with pytest.raises(ProviderProtocolError, match="temporary"):
        await model.generate(ModelRequest(request_id="model_req_1"))

    response = await model.generate(ModelRequest(request_id="model_req_2"))

    assert response.final_text == "recovered"
    assert response.request_id == "model_req_2"
    assert [request.request_id for request in model.requests] == ["model_req_1", "model_req_2"]


async def test_scripted_model_append_helpers_and_exhaustion() -> None:
    model = ScriptedModel()

    with pytest.raises(ScriptExhaustedError, match="no remaining"):
        await model.generate(ModelRequest(request_id="model_req_1"))

    model.append_exception(ProviderProtocolError("still failing"))
    model.append_response(ModelResponse(response_id="model_resp_2"))

    with pytest.raises(ProviderProtocolError, match="still failing"):
        await model.generate(ModelRequest(request_id="model_req_2"))

    assert (await model.generate(ModelRequest(request_id="model_req_3"))).request_id == (
        "model_req_3"
    )


async def test_scripted_model_uses_defensive_copies_for_profile_requests_and_responses() -> None:
    profile = ModelProfile(
        provider_name="test_provider",
        model_name="test-model",
        api=ProviderApi.OPENAI_RESPONSES,
        details={"source": "fixture"},
    )
    response_step = ModelResponse(
        response_id="model_resp_1",
        response_metadata={"mutable": {"value": "original"}},
    )
    request = ModelRequest(request_id="model_req_1", request_metadata={"before": True})
    model = ScriptedModel([response_step], profile=profile, link_response_request_id=False)

    returned_profile = model.profile
    assert returned_profile == profile
    assert returned_profile is not None
    assert returned_profile is not profile

    response = await model.generate(request)
    response.response_metadata["mutable"] = {"value": "changed"}
    request.request_metadata["after"] = True

    assert response.request_id is None
    assert response_step.response_metadata == {"mutable": {"value": "original"}}
    assert model.requests[0].request_metadata == {"before": True}
    assert isinstance(model, ModelAdapter)

    model.clear_requests()
    assert model.requests == ()
