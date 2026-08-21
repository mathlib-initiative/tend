# Tools

tend currently exposes a closed built-in tool surface plus the typed `Tool` wrapper used by tests and internal execution.

## `Tool`

A tool has:

- `ToolDefinition`: name, description, strict JSON object argument schema, optional default timeout/output metadata.
- Pydantic argument model derived from `StrictModel`.
- Async handler: `async def handler(context: ToolContext, args) -> object`.
- Optional argument preparer for persisted/legacy compatibility.

Argument schemas must have `type: "object"` and `additionalProperties: false`.

## `ToolContext`

Handlers receive `ToolContext` with:

- `cwd`
- `session_id` and `turn_id`
- resolved `runtime_config`
- optional `event_callback`
- optional `CancellationState`
- optional `filesystem_backend` and `process_backend` test seams

`ToolContext` deliberately does not implement path allowlists, command filtering, or network policy. Sandbox policy belongs to the process/orchestration boundary.

## Execution semantics

`execute_tool_calls(...)`:

- executes sequentially in provider order (`ToolCall.order`, stable input tie-breaker);
- emits `ToolCallStarted` / `ToolCallCompleted` callback events;
- validates arguments before calling a handler;
- converts unknown tools, validation failures, handler exceptions, and structured tool errors into `ToolResult(success=False)`;
- preserves provider IDs from `ToolCall` into `ToolResult`.

The executor does not enforce `ToolDefinition.default_timeout_seconds`. Concrete tools may implement their own reliability bounds; `bash` does.

## Built-in registry

Stable built-in names:

| Tool | Purpose |
| --- | --- |
| `ls` | List one directory with sorted, bounded output. |
| `read_file` | Read UTF-8 text with 1-based line pagination, omission, and truncation metadata. |
| `grep` | Regex search over UTF-8 text files selected by path/glob. |
| `glob` | Deterministic backend glob search. |
| `write_file` | Write UTF-8 text, creating parents by default and overwriting files. |
| `edit_file` | Apply one or more exact unique non-overlapping replacements to an existing UTF-8 file. |
| `copy_lines` | Copy a 1-based inclusive UTF-8 line range into a distinct destination file without retyping the text. |
| `bash` | Run one shell command with stdout/stderr/exit-code/timeout/truncation metadata. |

Use `get_builtin_tool`, `get_builtin_tools`, or `export_builtin_tool_schemas` from `tend.agent.tools`.

## Built-in behavior notes

- File/list/search tools use head truncation.
- `bash` uses tail truncation for stdout and stderr.
- `read_file` omits binary and non-UTF-8 files instead of returning raw bytes.
- `edit_file` matches all `edits[].old_text` against the original normalized file content, not incrementally edited content. Matches must be unique and non-overlapping.
- `edit_file` preserves detected CRLF line endings and a UTF-8 BOM when practical.
- `copy_lines` never mutates the source file; remove copied ranges with `edit_file` when needed.
- `bash` treats nonzero exit codes as command results, not tool-framework failures. Timeouts are failures.
- `TruncationInfo` supports optional artifact references, but the current built-in tools do not automatically write full-output artifacts.

## Minimal custom tool example

```python
from pydantic import BaseModel, ConfigDict, Field
from tend.agent.tools import Tool, ToolContext

class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(min_length=1)

async def echo(_context: ToolContext, args: EchoArgs) -> dict[str, str]:
    return {"echo": args.text}

echo_tool = Tool.from_arguments_model(
    name="echo",
    description="Echo text.",
    arguments_model=EchoArgs,
    handler=echo,
)
```
