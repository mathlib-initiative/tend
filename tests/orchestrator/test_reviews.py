from __future__ import annotations

import pytest

from tend._common.agent_outputs import (
    AgentOutputSchemaName,
    ReviewVerdictOutput,
    WorkerContributionOutput,
    output_schema_names,
    resolve_output_type,
)
from tend.orchestrator import (
    ReviewVerdictOutput as ReExportedReviewVerdictOutput,
)
from tend.orchestrator import (
    WorkerContributionOutput as ReExportedWorkerContributionOutput,
)
from tend.orchestrator.orchestrator import (
    _agent_discussion_message,  # pyright: ignore[reportPrivateUsage]
    _agent_success_state,  # pyright: ignore[reportPrivateUsage]
    _parse_agent_output,  # pyright: ignore[reportPrivateUsage]
)
from tend.orchestrator.state import WorktreeState


def _approve_verdict(notes: str = "All criteria PASS.") -> ReviewVerdictOutput:
    return ReviewVerdictOutput.model_validate(
        {"schema_version": 1, "verdict": "approve", "notes": notes}
    )


def _request_changes_verdict(
    notes: str = "Criterion 2 FAIL: build broke.",
    feedback_text: str = "Fix the YAML and re-run lake build.",
) -> ReviewVerdictOutput:
    return ReviewVerdictOutput.model_validate(
        {
            "schema_version": 1,
            "verdict": "request_changes",
            "notes": notes,
            "feedback_text": feedback_text,
        }
    )


def _worker_contribution(
    status: str = "completed",
    summary: str = "Implemented the requested change.",
    notes: str | None = None,
) -> WorkerContributionOutput:
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "summary": summary,
    }
    if notes is not None:
        payload["notes"] = notes
    return WorkerContributionOutput.model_validate(payload)


def test_worker_contribution_output_reads_summary() -> None:
    output = WorkerContributionOutput.model_validate_json(
        '{"schema_version":1,"status":"completed","summary":"Implemented the requested change."}'
    )

    assert output.status == "completed"
    assert output.summary == "Implemented the requested change."


def test_async_reviewer_reuses_sync_review_verdict_schema() -> None:
    # The async reviewer's output type is the shared review_verdict contract.
    assert ReExportedReviewVerdictOutput is ReviewVerdictOutput
    assert resolve_output_type(AgentOutputSchemaName.REVIEW_VERDICT) is ReviewVerdictOutput
    assert resolve_output_type("review_verdict") is ReviewVerdictOutput


def test_async_worker_reuses_sync_worker_contribution_schema() -> None:
    # The async worker's output type is the shared worker_contribution contract.
    assert ReExportedWorkerContributionOutput is WorkerContributionOutput
    assert (
        resolve_output_type(AgentOutputSchemaName.WORKER_CONTRIBUTION)
        is WorkerContributionOutput
    )
    assert resolve_output_type("worker_contribution") is WorkerContributionOutput


def test_review_verdict_output_requires_feedback_for_request_changes() -> None:
    with pytest.raises(ValueError):
        ReviewVerdictOutput.model_validate(
            {"schema_version": 1, "verdict": "request_changes", "notes": "needs work"}
        )


def test_review_verdict_output_rejects_unknown_verdict() -> None:
    with pytest.raises(ValueError):
        ReviewVerdictOutput.model_validate({"schema_version": 1, "verdict": "deny", "notes": "x"})


def test_async_review_decision_schema_is_removed() -> None:
    assert "async_review_decision" not in output_schema_names()
    assert not hasattr(AgentOutputSchemaName, "ASYNC_REVIEW_DECISION")


def test_async_worker_message_schema_is_removed() -> None:
    assert "async_worker_message" not in output_schema_names()
    assert not hasattr(AgentOutputSchemaName, "ASYNC_WORKER_MESSAGE")


def test_parse_agent_output_reads_bare_payload() -> None:
    output = _parse_agent_output(
        '{"schema_version":1,"status":"completed","summary":"done"}',
        WorkerContributionOutput,
    )

    assert isinstance(output, WorkerContributionOutput)
    assert output.status == "completed"
    assert output.summary == "done"


def test_parse_agent_output_unwraps_worker_turn_result_envelope() -> None:
    # tend-agent --json writes a full TurnResult; the validated payload is final_result.output.
    stdout = (
        '{"turn_id":"t1","stop_reason":"final_result",'
        '"final_result":{"tool_name":"final_result","tool_call_id":"c1",'
        '"output":{"schema_version":1,"status":"completed","summary":"Scaffolded the module."},'
        '"arguments":{}}}'
    )

    output = _parse_agent_output(stdout, WorkerContributionOutput)

    assert isinstance(output, WorkerContributionOutput)
    assert output.summary == "Scaffolded the module."


def test_parse_agent_output_unwraps_reviewer_review_verdict_envelope() -> None:
    stdout = (
        '{"turn_id":"t1","stop_reason":"final_result",'
        '"final_result":{"tool_name":"final_result","tool_call_id":"c1",'
        '"output":{"schema_version":1,"verdict":"request_changes",'
        '"notes":"Criterion 2 FAIL.","feedback_text":"Fix the YAML."},'
        '"arguments":{}}}'
    )

    output = _parse_agent_output(stdout, ReviewVerdictOutput)

    assert isinstance(output, ReviewVerdictOutput)
    assert output.verdict == "request_changes"
    assert output.notes == "Criterion 2 FAIL."
    assert output.feedback_text == "Fix the YAML."


def test_parse_agent_output_reads_bare_review_verdict_payload() -> None:
    output = _parse_agent_output(
        '{"schema_version":1,"verdict":"approve","notes":"All criteria PASS."}',
        ReviewVerdictOutput,
    )

    assert isinstance(output, ReviewVerdictOutput)
    assert output.verdict == "approve"


def test_parse_agent_output_rejects_prose_before_json() -> None:
    # The original failure mode: a chatty model narrating before the JSON. With the
    # final_result tool configured this no longer reaches stdout, and raw prose stays rejected.
    stdout = (
        'Here is my review.\n\n{"schema_version":1,"verdict":"approve","notes":"LGTM"}'
    )

    with pytest.raises(ValueError):
        _parse_agent_output(stdout, ReviewVerdictOutput)


def test_parse_agent_output_rejects_turn_result_with_null_output() -> None:
    stdout = (
        '{"turn_id":"t1","stop_reason":"final_result",'
        '"final_result":{"tool_name":"final_result","tool_call_id":"c1",'
        '"output":null,"arguments":{}}}'
    )

    with pytest.raises(ValueError):
        _parse_agent_output(stdout, WorkerContributionOutput)


def test_approve_verdict_routes_to_merge() -> None:
    assert _agent_success_state(_approve_verdict()) is WorktreeState.MERGE


def test_request_changes_verdict_routes_to_pending() -> None:
    assert _agent_success_state(_request_changes_verdict()) is WorktreeState.PENDING


def test_completed_worker_contribution_routes_to_review() -> None:
    assert _agent_success_state(_worker_contribution(status="completed")) is WorktreeState.REVIEW


def test_needs_review_worker_contribution_routes_to_review() -> None:
    assert (
        _agent_success_state(_worker_contribution(status="needs_review")) is WorktreeState.REVIEW
    )


def test_blocked_worker_contribution_routes_to_review() -> None:
    # Under the blocked contract a ``blocked`` worker is expected to have committed
    # task-graph edits + progress; routing it through review (then merge) is how those
    # land. A ``blocked`` return that committed nothing is closed for a fresh respawn,
    # but that branch lives in ``_run_agent_for_worktree_id`` (needs the worktree path),
    # not in ``_agent_success_state``.
    assert _agent_success_state(_worker_contribution(status="blocked")) is WorktreeState.REVIEW


def test_discussion_message_uses_notes_for_approve() -> None:
    assert _agent_discussion_message(_approve_verdict(notes="Looks great.")) == "Looks great."


def test_discussion_message_appends_feedback_for_request_changes() -> None:
    message = _agent_discussion_message(
        _request_changes_verdict(notes="Criterion 1 FAIL.", feedback_text="Add the lemma.")
    )

    assert "Criterion 1 FAIL." in message
    assert "Add the lemma." in message


def test_worker_discussion_message_uses_summary() -> None:
    assert _agent_discussion_message(_worker_contribution(summary="Did the work.")) == (
        "Did the work."
    )


def test_worker_discussion_message_appends_notes() -> None:
    message = _agent_discussion_message(
        _worker_contribution(summary="Did the work.", notes="Watch out for the edge case.")
    )

    assert "Did the work." in message
    assert "Watch out for the edge case." in message
