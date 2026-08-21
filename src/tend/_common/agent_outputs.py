"""Typed agent final-output schemas and schema-name resolution."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from tend._common.types import StrictModel

_NonEmptyString = Annotated[str, Field(min_length=1)]
_PositiveInt = Annotated[int, Field(ge=1)]

_FINAL_RESULT_TOOL_NAME = "final_result"


class AgentOutputSchemaName(StrEnum):
    """Stable schema names usable from durable agent/orchestrator config."""

    REVIEW_VERDICT = "review_verdict"
    WORKER_CONTRIBUTION = "worker_contribution"


class AgentOutputConfig(StrictModel):
    """Durable config selecting an agent-scoped structured final output."""

    tool_name: Literal["final_result"] = _FINAL_RESULT_TOOL_NAME
    schema_name: AgentOutputSchemaName
    required: Literal[True] = True

    @field_validator("schema_name", mode="before")
    @classmethod
    def _coerce_schema_name(cls, value: object) -> object:
        if isinstance(value, str):
            return AgentOutputSchemaName(value)
        return value


class ReviewCommentOutput(StrictModel):
    """Optional structured reviewer comment attached to a verdict."""

    message: _NonEmptyString
    path: _NonEmptyString | None = None
    line_start: _PositiveInt | None = None
    line_end: _PositiveInt | None = None
    severity: Literal["info", "warning", "error"] = "info"

    @model_validator(mode="after")
    def _validate_line_range(self) -> ReviewCommentOutput:
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("review comment line_end must be >= line_start")
        return self


def _empty_review_comments() -> list[ReviewCommentOutput]:
    return []


class ReviewVerdictOutput(StrictModel):
    """Reviewer final output submitted through ``final_result``."""

    schema_version: Literal[1]
    verdict: Literal["approve", "request_changes"]
    notes: _NonEmptyString
    feedback_text: _NonEmptyString | None = None
    comments: list[ReviewCommentOutput] = Field(default_factory=_empty_review_comments)

    @model_validator(mode="after")
    def _validate_feedback_contract(self) -> ReviewVerdictOutput:
        if self.verdict == "approve" and self.feedback_text is not None:
            raise ValueError("approve verdicts must not include feedback_text")
        if self.verdict == "request_changes" and self.feedback_text is None:
            raise ValueError("request_changes verdicts require feedback_text")
        return self


class ValidationEvidence(StrictModel):
    """Command or check evidence reported by a worker."""

    command: _NonEmptyString
    exit_code: int
    summary: _NonEmptyString | None = None


def _empty_strings() -> list[str]:
    return []


def _empty_validation_evidence() -> list[ValidationEvidence]:
    return []


class WorkerContributionOutput(StrictModel):
    """Worker final output submitted through ``final_result``."""

    schema_version: Literal[1]
    status: Literal["completed", "blocked", "needs_review"]
    summary: _NonEmptyString
    files_changed: list[_NonEmptyString] = Field(default_factory=_empty_strings)
    validation: list[ValidationEvidence] = Field(default_factory=_empty_validation_evidence)
    tasks_created: list[_NonEmptyString] = Field(default_factory=_empty_strings)
    notes: _NonEmptyString | None = None


_OUTPUT_TYPES: dict[AgentOutputSchemaName, type[BaseModel]] = {
    AgentOutputSchemaName.REVIEW_VERDICT: ReviewVerdictOutput,
    AgentOutputSchemaName.WORKER_CONTRIBUTION: WorkerContributionOutput,
}


def resolve_output_type(schema_name: AgentOutputSchemaName | str) -> type[BaseModel]:
    """Return the Pydantic output model for a durable schema name."""

    name = (
        schema_name
        if isinstance(schema_name, AgentOutputSchemaName)
        else AgentOutputSchemaName(schema_name)
    )
    return _OUTPUT_TYPES[name]


def output_schema_names() -> tuple[str, ...]:
    """Return schema names accepted by ``resolve_output_type``."""

    return tuple(name.value for name in AgentOutputSchemaName)


__all__ = (
    "AgentOutputConfig",
    "AgentOutputSchemaName",
    "ReviewCommentOutput",
    "ReviewVerdictOutput",
    "ValidationEvidence",
    "WorkerContributionOutput",
    "output_schema_names",
    "resolve_output_type",
)
