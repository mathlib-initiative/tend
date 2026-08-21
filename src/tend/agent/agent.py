"""Public agent runtime boundary."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, TypeAdapter

from tend._common.types import JsonObject
from tend.agent.cancellation import CancellationState
from tend.agent.config import AgentConfig, RuntimeConfig
from tend.agent.limits import MonotonicClock
from tend.agent.outputs import resolve_output_type
from tend.agent.results import TurnResult
from tend.agent.session import Session
from tend.agent.tools.base import Tool
from tend.agent.tools.context import ToolContext
from tend.agent.tools.registry import get_builtin_tools
from tend.agent.turn_loop import run_turn as _run_turn
from tend.llm.models.base import ModelAdapter
from tend.llm.models.reasoning import ReasoningSettings

type AgentToolInput = str | Tool[Any]
type AgentOutputType = type[BaseModel] | TypeAdapter[Any]

_FINAL_RESULT_TOOL_NAME = "final_result"
_FINAL_RESULT_TOOL_DESCRIPTION = "The final response which ends this conversation"
_OUTPUT_TOOL_METADATA: JsonObject = {"tend_tool_kind": "output", "terminates_turn": True}


class Agent:
    """Async-first runtime object for one provider-neutral agent."""

    __slots__ = (
        "max_output_tokens",
        "model",
        "model_name",
        "output_type",
        "reasoning",
        "system_prompt",
        "tools",
    )

    system_prompt: str
    model: ModelAdapter
    tools: tuple[Tool[Any], ...]
    model_name: str | None
    output_type: AgentOutputType | None
    reasoning: ReasoningSettings | None
    max_output_tokens: int | None

    def __init__(
        self,
        system_prompt: str,
        *,
        model: ModelAdapter,
        tools: Iterable[AgentToolInput] = (),
        output_type: AgentOutputType | None = None,
        model_name: str | None = None,
        reasoning: ReasoningSettings | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        if not system_prompt:
            raise ValueError("system_prompt must be non-empty")
        if model_name is not None and not model_name:
            raise ValueError("model_name must be non-empty when provided")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive when provided")

        self.system_prompt = system_prompt
        self.model = model
        self.output_type = output_type
        self.tools = _resolve_tools(tools, output_type=output_type)
        self.model_name = model_name or _profile_model_name(model)
        self.reasoning = reasoning.model_copy(deep=True) if reasoning is not None else None
        self.max_output_tokens = max_output_tokens

    @classmethod
    def from_config(
        cls,
        config: AgentConfig,
        *,
        model: ModelAdapter,
    ) -> Agent:
        """Construct an agent from durable config and an injected model adapter."""

        output_type = (
            None if config.output is None else resolve_output_type(config.output.schema_name)
        )
        return cls(
            config.system_prompt,
            model=model,
            tools=config.tools.enabled,
            output_type=output_type,
            model_name=config.model.model_name,
            reasoning=config.model.settings.reasoning,
            max_output_tokens=config.model.settings.max_output_tokens,
        )

    async def run_turn(
        self,
        prompt: str,
        *,
        session: Session | None = None,
        config: RuntimeConfig | None = None,
        cancellation: CancellationState | None = None,
        clock: MonotonicClock | None = None,
    ) -> TurnResult:
        """Run one full turn through the shared async turn loop."""

        return await _run_turn(
            system_prompt=self.system_prompt,
            model=self.model,
            prompt=prompt,
            tools=self.tools,
            session=session,
            config=config,
            model_name=self.model_name,
            reasoning=self.reasoning,
            max_output_tokens=self.max_output_tokens,
            cancellation=cancellation,
            clock=clock,
        )


def _resolve_tools(
    tools: Iterable[AgentToolInput],
    *,
    output_type: AgentOutputType | None = None,
) -> tuple[Tool[Any], ...]:
    resolved: list[Tool[Any]] = []
    for tool in tools:
        if isinstance(tool, str):
            resolved.extend(get_builtin_tools((tool,)))
        else:
            resolved.append(tool)

    reserved = [tool.name for tool in resolved if tool.name == _FINAL_RESULT_TOOL_NAME]
    if reserved:
        raise ValueError("ordinary tool name 'final_result' is reserved for agent output")

    if output_type is not None:
        resolved.append(_final_result_tool(output_type))

    seen_names: set[str] = set()
    duplicate_names: list[str] = []
    for tool in resolved:
        if tool.name in seen_names:
            duplicate_names.append(tool.name)
        seen_names.add(tool.name)
    if duplicate_names:
        joined = ", ".join(sorted(set(duplicate_names)))
        raise ValueError(f"duplicate enabled tool names: {joined}")
    return tuple(resolved)


def _final_result_tool(output_type: AgentOutputType) -> Tool[Any]:
    async def handler(_context: ToolContext, arguments: Any) -> object:
        return _dump_output_value(output_type, arguments)

    return Tool.from_arguments_model(
        name=_FINAL_RESULT_TOOL_NAME,
        description=_FINAL_RESULT_TOOL_DESCRIPTION,
        arguments_model=output_type,
        handler=handler,
        metadata=_OUTPUT_TOOL_METADATA,
    )


def _dump_output_value(output_type: AgentOutputType, value: Any) -> object:
    if isinstance(output_type, TypeAdapter):
        return output_type.dump_python(value, mode="json")
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _profile_model_name(model: ModelAdapter) -> str | None:
    profile = model.profile
    if profile is None:
        return None
    return profile.model_name


__all__ = ("Agent", "AgentOutputType", "AgentToolInput")
