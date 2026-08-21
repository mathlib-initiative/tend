"""Runtime tool abstraction boundary."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any, cast

from pydantic import BaseModel, Field, TypeAdapter, model_validator
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema

from tend._common.types import JsonObject, StrictModel
from tend.agent.tools.context import ToolContext

type ToolHandler[ArgsT] = Callable[[ToolContext, ArgsT], Awaitable[object]]
type ToolArgumentsValidator[ArgsT] = type[BaseModel] | TypeAdapter[ArgsT]
type ArgumentPreparer = Callable[[JsonObject], JsonObject]

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


class _NoExtraJsonSchema(GenerateJsonSchema):
    """Generate object schemas that match ``extra='forbid'`` validation."""

    def model_fields_schema(self, schema: core_schema.ModelFieldsSchema) -> JsonSchemaValue:
        json_schema = super().model_fields_schema(schema)
        self.resolve_ref_schema(json_schema)["additionalProperties"] = False
        return json_schema

    def typed_dict_schema(self, schema: core_schema.TypedDictSchema) -> JsonSchemaValue:
        json_schema = super().typed_dict_schema(schema)
        self.resolve_ref_schema(json_schema)["additionalProperties"] = False
        return json_schema


def _empty_strict_object_schema() -> JsonObject:
    return {"type": "object", "properties": {}, "additionalProperties": False}


def _empty_json_object() -> JsonObject:
    return {}


class ToolDefinition(StrictModel):
    """Serializable model-visible definition for a built-in tool."""

    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    arguments_schema: JsonObject = Field(default_factory=_empty_strict_object_schema)
    default_timeout_seconds: float | None = Field(default=None, gt=0)
    default_output_limit_bytes: int | None = Field(default=None, ge=1)
    metadata: JsonObject = Field(default_factory=_empty_json_object)

    @classmethod
    def from_arguments_model(
        cls,
        *,
        name: str,
        description: str,
        arguments_model: type[BaseModel] | TypeAdapter[Any],
        default_timeout_seconds: float | None = None,
        default_output_limit_bytes: int | None = None,
        metadata: JsonObject | None = None,
    ) -> ToolDefinition:
        """Build a definition from a strict Pydantic argument model."""

        schema = _arguments_schema(arguments_model)
        return cls(
            name=name,
            description=description,
            arguments_schema=schema,
            default_timeout_seconds=default_timeout_seconds,
            default_output_limit_bytes=default_output_limit_bytes,
            metadata=metadata or {},
        )

    @model_validator(mode="after")
    def _validate_arguments_schema(self) -> ToolDefinition:
        if self.arguments_schema.get("type") != "object":
            raise ValueError("tool argument schema must be a JSON object schema")
        if self.arguments_schema.get("additionalProperties") is not False:
            raise ValueError("tool argument schema must set additionalProperties to false")
        return self


class Tool[ArgumentsT]:
    """Lightweight runtime wrapper for a built-in tool.

    Execution policy intentionally lives outside this class. The sequential
    executor added later owns timing, timeout handling, provider-ID linkage,
    output normalization, and conversion of validation/handler failures into
    model-visible ``ToolResult`` values.
    """

    __slots__ = ("_argument_preparer", "_arguments_model", "_handler", "definition")

    definition: ToolDefinition
    _arguments_model: ToolArgumentsValidator[ArgumentsT]
    _handler: ToolHandler[ArgumentsT]
    _argument_preparer: ArgumentPreparer | None

    def __init__(
        self,
        *,
        definition: ToolDefinition,
        arguments_model: ToolArgumentsValidator[ArgumentsT],
        handler: ToolHandler[ArgumentsT],
        argument_preparer: ArgumentPreparer | None = None,
    ) -> None:
        self.definition = definition
        self._arguments_model = arguments_model
        self._handler = handler
        self._argument_preparer = argument_preparer

    @classmethod
    def from_arguments_model[ArgsT](
        cls,
        *,
        name: str,
        description: str,
        arguments_model: ToolArgumentsValidator[ArgsT],
        handler: ToolHandler[ArgsT],
        argument_preparer: ArgumentPreparer | None = None,
        default_timeout_seconds: float | None = None,
        default_output_limit_bytes: int | None = None,
        metadata: JsonObject | None = None,
    ) -> Tool[ArgsT]:
        """Create a tool and serializable definition from an argument model."""

        definition = ToolDefinition.from_arguments_model(
            name=name,
            description=description,
            arguments_model=arguments_model,
            default_timeout_seconds=default_timeout_seconds,
            default_output_limit_bytes=default_output_limit_bytes,
            metadata=metadata,
        )
        return Tool(
            definition=definition,
            arguments_model=arguments_model,
            handler=handler,
            argument_preparer=argument_preparer,
        )

    @property
    def name(self) -> str:
        """Return the stable tool name."""

        return self.definition.name

    @property
    def arguments_model(self) -> ToolArgumentsValidator[ArgumentsT]:
        """Return the Pydantic validator used to validate arguments."""

        return self._arguments_model

    def prepare_arguments(self, arguments: JsonObject) -> JsonObject:
        """Return compatibility-prepared argument data before validation."""

        copied = _copy_json_object(arguments)
        if self._argument_preparer is None:
            return copied
        return _JSON_OBJECT_ADAPTER.validate_python(self._argument_preparer(copied))

    def validate_arguments(self, arguments: JsonObject) -> ArgumentsT:
        """Prepare and validate raw JSON-object arguments for this tool."""

        prepared = self.prepare_arguments(arguments)
        if isinstance(self._arguments_model, TypeAdapter):
            return self._arguments_model.validate_python(prepared, extra="forbid")
        model_type = self._arguments_model
        return cast(ArgumentsT, model_type.model_validate(prepared, extra="forbid"))

    async def run(self, context: ToolContext, arguments: ArgumentsT) -> object:
        """Run the handler with already validated arguments.

        Normal exceptions propagate to the caller. The later sequential executor
        converts them into structured model-visible tool errors.
        """

        return await self._handler(context, arguments)


def _arguments_schema(arguments_model: type[BaseModel] | TypeAdapter[Any]) -> JsonObject:
    if isinstance(arguments_model, TypeAdapter):
        raw_schema = arguments_model.json_schema(schema_generator=_NoExtraJsonSchema)
    else:
        raw_schema = arguments_model.model_json_schema(schema_generator=_NoExtraJsonSchema)
    return _JSON_OBJECT_ADAPTER.validate_python(raw_schema)


def _copy_json_object(value: JsonObject) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(deepcopy(value))


__all__ = (
    "ArgumentPreparer",
    "Tool",
    "ToolDefinition",
    "ToolArgumentsValidator",
    "ToolHandler",
)
