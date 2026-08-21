# Agent Layer

The agent layer is the provider-neutral runtime above the LLM adapters. It owns one-turn execution, strict tool handling, runtime limits, session event persistence, generic compaction, and the one-turn CLI.

## Main entry points

- `tend.Agent`: async-first public runtime object.
- `tend.Session`: writable/read-only handle for a persisted session directory.
- `tend.Tool` and `tend.ToolContext`: built-in/tool execution boundary.
- `tend.agent.config`: `AgentConfig`, `RuntimeConfig`, sparse override models, and config resolution.
- `tend.agent.results`: `TurnResult` and `StopResult`.

## Contents

- [Public API](api.md)
- [Configuration](configuration.md)
- [Turn loop](turn-loop.md)
- [Tools](tools.md)
- [Sessions and persistence](sessions.md)
- [Compaction](compaction.md)
- [CLI](cli.md)

## Important current behavior

`Agent.run_turn(...)` runs exactly one turn. Within that turn, it keeps the active model context in memory across model/tool iterations. When a `Session` is supplied, it persists canonical events and updates `state.json`; it does **not** automatically replay completed previous turns into the next turn's model context. Current session replay is used for resumability metadata and interrupted-tool surfacing.
