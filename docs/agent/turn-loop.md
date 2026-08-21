# Turn Loop

The shared turn loop lives in `tend.agent.turn_loop.run_turn` and is used by `Agent.run_turn`.

## Flow

For one `Agent.run_turn(prompt, ...)` call:

1. Resolve `RuntimeConfig` defaults and create a `TurnLimitTracker`.
2. Build active context from:
   - system prompt,
   - any interrupted-tool synthetic results from `SessionState`,
   - the new user prompt.
3. Export enabled tool schemas into provider-neutral tool definitions.
4. Estimate context tokens when enabled.
5. Optionally run generic compaction before a model request.
6. Build a `ModelRequest` and call `model.generate(request)`.
7. Normalize usage and persist model response events when a session exists.
8. If the response contains tool calls:
   - append an assistant history message carrying tool-call metadata,
   - execute tool calls sequentially in provider order,
   - append `ToolResultMessage` values,
   - loop back to the next model request.
9. If the response contains final assistant text, return a final `TurnResult`.
10. Otherwise return a structured non-final stop.

The in-memory `messages` list is the active context for the current turn. Persisted session events are written as the turn progresses, but completed historical turns are not automatically replayed into future `Agent.run_turn` calls.

## Session events written by the loop

When a session is supplied, the loop appends events such as:

- `TurnStarted`
- `ModelRequestStarted`
- `ModelResponseCompleted`
- `ModelRequestFailed`
- `ToolCallStarted`
- `ToolCallCompleted`
- `CompactionStarted`
- `CompactionCompleted`
- `TurnInterrupted`
- `TurnCompleted`

Each append refreshes `state.json` through session replay.

## Limits

The loop checks limits before model requests and before tool execution:

- `max_iterations`
- `max_model_requests`
- `max_tool_calls`
- `max_wall_time_seconds`
- `max_tokens`
- `max_cost`

Hitting a limit returns `TurnResult(final_response=None, stop=StopResult(...))` and persists `TurnCompleted` if a session is active.

## Tool execution

Tool calls are sorted by `(ToolCall.order, input_order)` and executed sequentially. Unknown tools, argument validation failures, handler-returned errors, and handler exceptions become model-visible `ToolResult(success=False)` values. They do not stop the loop by themselves.

If cooperative cancellation is requested before or after a tool batch, the turn returns `interrupted`. If an asyncio `CancelledError` is raised while waiting for a model, compacting, or executing tools, the loop records `TurnInterrupted` and re-raises.

## Compaction and context overflow

Before each model request, the loop may run `plan_compaction(...)` and `GenericSummarizationCompactor` when configured thresholds or context-window estimates require it.

If a model request raises a context-overflow-looking exception and `compaction.trigger_on_context_overflow` is enabled, the loop records a retryable `ModelRequestFailed`, runs one forced compaction, and retries once with the compacted active context.

## Retries

The implemented turn loop currently does not apply the general retry/backoff helper to all provider errors. General retry policy and `RetryScheduled` schemas exist, and provider errors are classified, but the loop only has the special context-overflow compaction retry described above.
