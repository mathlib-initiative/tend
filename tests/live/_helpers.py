from __future__ import annotations

import os
from copy import deepcopy

import pytest
from pydantic import Field, TypeAdapter

from tend._common.types import JsonObject, StrictModel
from tend.agent.config import CompactionConfig, RuntimeConfig, RuntimeLimitsConfig
from tend.agent.tools import Tool, ToolContext
from tend.llm.models import ModelAdapter, ModelProfile, ModelRequest, ModelResponse
from tend.llm.redaction import Redactor

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


class EchoArguments(StrictModel):
    """Tiny live-test tool arguments."""

    text: str = Field(min_length=1)


class RecordingModelAdapter:
    """Small wrapper that records responses and can alter only the first request."""

    __slots__ = ("_first_metadata", "_inner", "_responses", "_used_first_metadata")

    _inner: ModelAdapter
    _first_metadata: JsonObject | None
    _responses: list[ModelResponse]
    _used_first_metadata: bool

    def __init__(
        self,
        inner: ModelAdapter,
        *,
        first_metadata: JsonObject | None = None,
    ) -> None:
        self._inner = inner
        self._first_metadata = _copy_json_object(first_metadata) if first_metadata else None
        self._responses = []
        self._used_first_metadata = False

    @property
    def profile(self) -> ModelProfile | None:
        """Return wrapped profile metadata."""

        return self._inner.profile

    @property
    def responses(self) -> tuple[ModelResponse, ...]:
        """Return defensive copies of provider-neutral responses."""

        return tuple(response.model_copy(deep=True) for response in self._responses)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Optionally decorate the first request, then delegate to the wrapped model."""

        delegated_request = request
        if self._first_metadata is not None and not self._used_first_metadata:
            metadata = _copy_json_object(request.request_metadata)
            metadata.update(_copy_json_object(self._first_metadata))
            delegated_request = request.model_copy(
                update={"request_metadata": metadata},
                deep=True,
            )
            self._used_first_metadata = True

        response = await self._inner.generate(delegated_request)
        self._responses.append(response.model_copy(deep=True))
        return response


def live_runtime_config() -> RuntimeConfig:
    """Return bounded runtime config so live tests cannot loop broadly."""

    return RuntimeConfig(
        compaction=CompactionConfig(enabled=False),
        limits=RuntimeLimitsConfig(
            max_iterations=4,
            max_model_requests=4,
            max_tool_calls=2,
            max_wall_time_seconds=180.0,
        ),
    )


def echo_tool() -> Tool[EchoArguments]:
    """Return a tiny deterministic echo tool for live tool-call checks."""

    async def handler(_context: ToolContext, arguments: EchoArguments) -> dict[str, str]:
        return {"echo": arguments.text}

    return Tool.from_arguments_model(
        name="echo",
        description="Echo a short text value for live provider compatibility checks.",
        arguments_model=EchoArguments,
        handler=handler,
    )


def cloudflare_openai_base_url() -> str:
    """Return the Cloudflare-routed OpenAI provider base URL without printing it."""

    return f"{_gateway_base_url()}/openai"


def cloudflare_anthropic_base_url() -> str:
    """Return the Cloudflare-routed Anthropic provider base URL without printing it."""

    return f"{_gateway_base_url()}/anthropic/v1"


def cloudflare_auth_headers() -> dict[str, str]:
    """Return the opaque Cloudflare auth header; never log the token value."""

    token = _required_environment_variable("CF_AIG_TOKEN")
    return {"cf-aig-authorization": f"Bearer {token}"}


def redactor_for_base_url(base_url: str) -> Redactor:
    """Return a redactor that treats gateway URLs and auth headers as sensitive."""

    return Redactor(
        secret_source_names=("CF_AIG_TOKEN",),
        secret_header_names=("cf-aig-authorization",),
        mildly_sensitive_urls=(base_url,),
    )


def _gateway_base_url() -> str:
    return _required_environment_variable("CF_AIG_URL").rstrip("/")


def _required_environment_variable(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is required for live provider tests")
    return value


def _copy_json_object(value: JsonObject) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(deepcopy(value))


def json_object(value: object) -> JsonObject:
    """Validate a test literal as a JSON object."""

    return _JSON_OBJECT_ADAPTER.validate_python(value)
