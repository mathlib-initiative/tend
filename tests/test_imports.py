import ast
import importlib
from pathlib import Path

import tend
from tend import (
    Agent,
    AgentConfig,
    CancellationState,
    CancellationToken,
    ConfigurationError,
    ErrorInfo,
    FinalResultOutput,
    FrameworkError,
    PersistenceError,
    ProviderProtocolError,
    ResolvedConfig,
    RuntimeConfig,
    RuntimeConfigOverrides,
    Session,
    StopResult,
    Tool,
    ToolContext,
    TurnResult,
    UnsupportedSchemaVersionError,
    resolve_config,
    resolve_runtime_config,
)


def test_public_exports_are_explicit() -> None:
    assert tend.__all__ == (
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
    assert tend.Agent is Agent
    assert tend.Session is Session
    assert tend.Tool is Tool
    assert tend.ToolContext is ToolContext
    assert tend.AgentConfig is AgentConfig
    assert tend.ResolvedConfig is ResolvedConfig
    assert tend.RuntimeConfig is RuntimeConfig
    assert tend.RuntimeConfigOverrides is RuntimeConfigOverrides
    assert tend.CancellationState is CancellationState
    assert tend.CancellationToken is CancellationToken
    assert tend.FinalResultOutput is FinalResultOutput
    assert tend.StopResult is StopResult
    assert tend.TurnResult is TurnResult
    assert tend.resolve_config is resolve_config
    assert tend.resolve_runtime_config is resolve_runtime_config
    assert tend.ConfigurationError is ConfigurationError
    assert tend.ErrorInfo is ErrorInfo
    assert tend.FrameworkError is FrameworkError
    assert tend.PersistenceError is PersistenceError
    assert tend.ProviderProtocolError is ProviderProtocolError
    assert tend.UnsupportedSchemaVersionError is UnsupportedSchemaVersionError


def test_public_placeholders_have_stable_modules() -> None:
    assert Agent.__module__ == "tend.agent.agent"
    assert Session.__module__ == "tend.agent.session"
    assert Tool.__module__ == "tend.agent.tools.base"
    assert ToolContext.__module__ == "tend.agent.tools.context"


def test_boundary_modules_import_without_side_effects() -> None:
    module_names = (
        "tend.agent.cancellation",
        "tend.agent.config",
        "tend.agent.limits",
        "tend._common.types",
        "tend._common.errors",
        "tend.llm.models",
        "tend.agent.tools",
        "tend.agent.persistence",
        "tend.agent.compaction",
        "tend.llm",
        "tend.orchestrator",
    )

    for module_name in module_names:
        assert importlib.import_module(module_name).__name__ == module_name


def test_canonical_layer_import_direction_is_enforced() -> None:
    package_root = Path(tend.__file__).resolve().parent
    rules = {
        "_common": ("tend.agent", "tend.llm", "tend.orchestrator"),
        "llm": ("tend.agent", "tend.orchestrator"),
        "agent": ("tend.orchestrator",),
    }

    violations: list[str] = []
    for layer, forbidden_prefixes in rules.items():
        for path in sorted((package_root / layer).rglob("*.py")):
            imported_modules = _tend_imports(path)
            for module_name in imported_modules:
                if any(
                    module_name == prefix or module_name.startswith(f"{prefix}.")
                    for prefix in forbidden_prefixes
                ):
                    relative = path.relative_to(package_root)
                    violations.append(f"{relative}: {module_name}")

    assert violations == []


def _tend_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("tend."):
                    modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.startswith("tend."):
                modules.add(node.module)
    return modules
