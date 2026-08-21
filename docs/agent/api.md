# Public Agent API

## `Agent`

`Agent` is async-first and requires an injected `ModelAdapter`.

```python
from tend import Agent

agent = Agent(
    "You are concise.",
    model=model_adapter,
    tools=["ls", "read_file"],
    model_name="gpt-5",
    reasoning=reasoning_settings,
    max_output_tokens=256,
)

result = await agent.run_turn("Inspect the project.")
```

Constructor fields:

- `system_prompt: str`: required, non-empty.
- `model: ModelAdapter`: provider-neutral adapter from `tend.llm.models`.
- `tools`: built-in tool names or `Tool` objects. Duplicate names are rejected.
- `model_name`: optional request model name. If omitted, `model.profile.model_name` is used when available.
- `reasoning`: optional `ReasoningSettings` copied into each model request.
- `max_output_tokens`: optional positive max output token request.

`Agent.from_config(config, model=...)` builds an agent from `AgentConfig` and an injected model adapter.

## `run_turn`

```python
result = await agent.run_turn(
    prompt,
    session=session,
    config=runtime_config,
    cancellation=cancellation_state,
)
```

Parameters:

- `prompt`: required non-empty user prompt.
- `session`: optional `Session`; when supplied, turn/model/tool/compaction events are appended and `state.json` is refreshed.
- `config`: optional `RuntimeConfig`; defaults are used when omitted.
- `cancellation`: optional `CancellationState`; checked at safe turn-loop boundaries and exposed to tools through `ToolContext`.

## `TurnResult`

`TurnResult` is a strict Pydantic model with:

- `turn_id`
- `final_response` / `final_text`
- `stop_reason`
- `stop` for non-final structured stops
- `usage`
- `context_estimate`
- `tool_calls` and `tool_results`
- `session_id` and `session_state` when a session was used
- `model_request_count` and `tool_call_count`

For final responses, `stop_reason == StopReason.FINAL_RESPONSE`, `final_response` is set, and `stop` is `None`.

## Stop reasons

Implemented `StopReason` values:

- `final_response`
- `provider_stop_reason`
- `max_model_requests`
- `max_tool_calls`
- `max_iterations`
- `max_wall_time`
- `max_tokens`
- `max_cost`
- `model_error`
- `interrupted`
- `compaction_failed`

## Sessions

Use `Session.create(path)`, `Session.open(path, writable=...)`, or `Session.resume(path)`. Writable sessions hold an exclusive `session.lock`. Session handles are context managers and should be closed.

```python
from tend import Session

with Session.create(".tend/session") as session:
    result = await agent.run_turn("Start work", session=session)
```
