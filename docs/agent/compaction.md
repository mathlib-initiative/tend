# Compaction

tend implements generic provider-neutral summarization compaction. Provider-native compaction is not implemented.

## Configuration

`RuntimeConfig.compaction` fields:

- `enabled` (default `true`)
- `threshold_tokens`
- `threshold_messages`
- `reserve_tokens` (default `4096`)
- `keep_recent_tokens` (default `16000`)
- `target_tokens` (default `4000`)
- `trigger_on_context_overflow` (default `true`)

Context estimation uses `RuntimeConfig.usage.token_estimator` and optional `ModelProfile.context_window` metadata.
Adapters may also supply estimates of their serialized messages and request
overhead through the optional `RequestTokenEstimator` capability.

## Planning

`plan_compaction(...)` is pure and deterministic. It triggers on:

- configured token threshold,
- configured message-count threshold,
- known context window minus reserve tokens,
- forced context-overflow retry path.

The planner preserves leading system/developer instruction messages and chooses a compactable older range plus a recent suffix. It moves the cut point backward until tool-call safety invariants hold:

- a completed assistant tool call and its tool result are either both compacted or both preserved;
- unresolved assistant tool calls are preserved;
- orphan tool-result messages are preserved.

Token triggers use the larger of the local estimate and the latest API usage
anchor. The planner scales local message and fixed overhead estimates by
`max(1, anchor / local_total)` before applying `keep_recent_tokens`, so triggering
and retention use the same units. Without an anchor, the scale is one.

For token-triggered compaction, the projected prompt must fit within 90% of the
effective threshold (the smaller of the configured token threshold and the
context window minus reserve). The projection includes preserved instructions,
the recent suffix, fixed request costs, the summary output budget, and its
message wrapper and covered IDs. If the initial cut cannot reach this target,
the planner tries later safe cuts while keeping the newest message. If none can
provide enough headroom, it skips with `insufficient token reduction` and the
turn loop logs a warning. Forced provider-overflow recovery still attempts the
smallest safe suffix even when the projection is pessimistic.

The provider's summary output limit is not an exact bound in the local
estimator's units. When replacing an existing summary, the planner also
estimates that summary's observed text with the replacement's new wrapper and
covered IDs, and uses the larger budget. A first summary can exceed its initial
projection; the next plan must then remove enough additional history to make
room, or skip, rather than repeatedly replacing only that oversized summary.

Plan metadata includes `token_scale_factor`, `token_target_tokens`, and
`projected_tokens`. Existing raw estimate fields remain unscaled; the legacy
`char_triggered` field now describes the local estimator's trigger, including
adapter estimates when available. Message-count-only triggers retain their
existing behavior.

`CompactionPlan.split_turn_prefix` is set when compaction covers a prefix of the latest turn while preserving a later safe suffix.

## Generic summarization

`GenericSummarizationCompactor` builds a `ModelRequest` with:

- the generic compaction system prompt,
- a rendered transcript of the planned message range,
- no tools,
- the configured reasoning settings when supplied,
- `max_output_tokens` from the explicit compactor setting or `plan.target_tokens`.

The model must return final text and no tool calls. Empty summaries fail with `CompactionError`.

`apply_compaction_result(...)` replaces the covered active-context range with one assistant message containing `CompactionSummaryContent`. The original active message list is not mutated.

## Turn-loop integration

Before each model request, the turn loop may compact active context. On context-overflow-looking model exceptions, the loop can force one compaction and retry once.

Compaction usage is added to the turn usage. When a session exists, the loop writes `CompactionStarted` and `CompactionCompleted` events, and replay records `CompletedCompaction` entries in `SessionState`.

## Prompt structure

The current prompt version is `generic_summarization_v1`. Summaries request these Markdown sections:

- Goal
- Constraints / Preferences
- Completed Work
- In-Progress Work
- Blockers
- Key Decisions
- Next Steps
- Critical Context
- Important Read / Modified Files

## Limitations

- Compaction summarizes only the active in-memory context for the current turn.
- The current `Agent.run_turn` does not automatically rebuild future-turn active context from persisted completed compactions; lower-level context helpers can insert `ActiveCompactionSummary` when callers manage tail context themselves.
- Provider-native compaction is future work.
