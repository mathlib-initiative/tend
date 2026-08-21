# Sessions and Persistence

`Session` is the writable/read-only boundary for persisted agent state.

## Directory layout

A session directory contains:

- `events.jsonl`: append-only canonical event log.
- `state.json`: atomic replay snapshot/cache.
- `session.lock`: advisory exclusive lock for writable handles.
- `artifacts/`: created by `Session.create`; artifact helper subdirectories are created for model requests, model responses, tool outputs, and compactions.

## Opening sessions

```python
from tend import Session

with Session.create(".tend/session", session_id="sess_1") as session:
    ...

with Session.resume(".tend/session") as session:
    ...

with Session.open(".tend/session", writable=False) as session:
    state = session.state
```

Writable sessions append a lifecycle event immediately:

- `SessionStarted` for `create`
- `SessionResumed` for writable `open`/`resume`

Read-only opens do not take the lock and do not append events.

## Events

Events are strict Pydantic discriminated unions with `schema_version=1`. Event envelopes include event ID, parent event ID, sequence, session ID, optional turn ID, timestamp, event type, and payload.

Implemented event types:

- `SessionStarted`, `SessionResumed`
- `TurnStarted`, `TurnInterrupted`, `TurnCompleted`
- `ModelRequestStarted`, `ModelResponseCompleted`, `ModelRequestFailed`
- `RetryScheduled` schema exists, but the current turn loop does not emit general retry events
- `ToolCallStarted`, `ToolCallCompleted`
- `CompactionStarted`, `CompactionCompleted`

The current turn loop stores prompt, provider-neutral model request, and provider-neutral model response inline in events. The separate detailed payload recorder/artifact path exists, but is not wired into `Agent.run_turn` as the default event-writing path.

## State replay

`state.json` is rebuilt by replaying `events.jsonl`. Replay records:

- completed and incomplete model requests;
- completed and interrupted tool calls;
- completed compactions;
- session/turn/model/compaction usage aggregates;
- provider response IDs;
- latest context estimates.

Completed model requests and completed tool calls are never rerun by replay. Started tool calls without completion are surfaced as synthetic interrupted `ToolResult` values. Started model requests without a terminal event remain recorded as incomplete metadata.

## Resume behavior today

When `Agent.run_turn` receives a resumed `Session`, it uses `SessionState` to include interrupted tool-call results in the first new request when needed. It does not automatically reconstruct full prior conversation history from completed events, and it does not rerun or continue an incomplete model request. A resumed turn starts a fresh model request for the new prompt.

## Schema versions

Only schema version `1` is supported for events and state snapshots. Unknown persisted versions raise `UnsupportedSchemaVersionError`.

## Artifacts

`ArtifactStore` can write/read redacted JSON, text, or bytes under the session artifact root and returns storage-neutral `ArtifactRef` values. Artifact naming can use event IDs or content hashes. Checksums are verified on reads when present.

`DetailedPayloadRecorder` can decide whether optional model/tool/compaction payloads are inlined, artifact-backed, or omitted based on `RuntimeConfig.logging` and `RuntimeConfig.artifacts`.
