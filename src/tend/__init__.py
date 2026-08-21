"""Minimal typed Python library for long-horizon AI agents."""

from tend._common.errors import (
    ConfigurationError,
    ErrorInfo,
    FrameworkError,
    PersistenceError,
    ProviderProtocolError,
    UnsupportedSchemaVersionError,
)
from tend.agent import Agent
from tend.agent.cancellation import CancellationState, CancellationToken
from tend.agent.config import (
    AgentConfig,
    ResolvedConfig,
    RuntimeConfig,
    RuntimeConfigOverrides,
    resolve_config,
    resolve_runtime_config,
)
from tend.agent.results import FinalResultOutput, StopResult, TurnResult
from tend.agent.session import Session
from tend.agent.tools import Tool, ToolContext

__all__ = (
    "Agent",
    "Session",
    "Tool",
    "ToolContext",
    "AgentConfig",
    "ResolvedConfig",
    "RuntimeConfig",
    "RuntimeConfigOverrides",
    "CancellationState",
    "CancellationToken",
    "FinalResultOutput",
    "StopResult",
    "TurnResult",
    "resolve_config",
    "resolve_runtime_config",
    "ConfigurationError",
    "ErrorInfo",
    "FrameworkError",
    "PersistenceError",
    "ProviderProtocolError",
    "UnsupportedSchemaVersionError",
    "__version__",
)

__version__ = "0.1.0"
