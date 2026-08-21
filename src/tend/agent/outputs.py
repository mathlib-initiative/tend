"""Typed agent final-output schemas and schema-name resolution."""

from tend._common.agent_outputs import (
    AgentOutputConfig,
    AgentOutputSchemaName,
    ReviewCommentOutput,
    ReviewVerdictOutput,
    ValidationEvidence,
    WorkerContributionOutput,
    output_schema_names,
    resolve_output_type,
)

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
