"""Per-example metric derivation and exact aggregate recomputation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from typing import cast

from deep_research_rl.agent.contracts import AgentRollout
from deep_research_rl.agent.prompting import format_observation
from deep_research_rl.core.metrics import build_episode_metrics
from deep_research_rl.core.models import SearchAction
from deep_research_rl.evaluation.contracts import (
    BASELINE_EVALUATION_PROTOCOL,
    EvaluationExampleMetrics,
    EvaluationFailure,
    EvaluationItem,
    EvaluationRecord,
    EvaluationResultScope,
    PolicyCondition,
)

EVALUATION_PROTOCOL_VERSION = BASELINE_EVALUATION_PROTOCOL


class EvaluationIntegrityError(ValueError):
    """Raised when records cannot support a trustworthy aggregate."""


def _stable_ids_sha256(example_ids: Sequence[str]) -> str:
    payload = json.dumps(list(example_ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate_rollout(
    rollout: AgentRollout,
    *,
    run_id: str,
    policy_condition: str,
    count_tool_tokens: Callable[[str], int],
) -> EvaluationRecord:
    """Derive every metric from one complete rollout without mutating it."""

    if policy_condition not in {"no_search", "prompted_agent", "rl_agent"}:
        raise ValueError(f"unsupported policy condition: {policy_condition}")
    rebuilt = build_episode_metrics(
        rollout.example,
        rollout.final_state,
        len(rollout.steps),
    )
    if rebuilt != rollout.metrics:
        raise EvaluationIntegrityError("embedded rollout metrics do not recompute exactly")

    attempted_searches = sum(isinstance(step.action, SearchAction) for step in rollout.steps)
    executed_searches = sum(step.observation.status == "search_executed" for step in rollout.steps)
    rejected_searches = sum(step.observation.status == "search_rejected" for step in rollout.steps)
    malformed_actions = sum(step.observation.status == "malformed_action" for step in rollout.steps)
    if executed_searches != rollout.final_state.executed_searches:
        raise EvaluationIntegrityError("executed-search observations disagree with final state")

    retrieved_ids: list[str] = []
    tool_tokens = 0
    for index, step in enumerate(rollout.steps, 1):
        if step.observation.status != "search_executed":
            continue
        rendered = format_observation(index, step.observation)
        count = count_tool_tokens(rendered)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise EvaluationIntegrityError("tool token counter must return a non-negative integer")
        tool_tokens += count
        for document in step.observation.documents:
            if document.document_id not in retrieved_ids:
                retrieved_ids.append(document.document_id)

    supporting = set(rollout.example.supporting_document_ids)
    if supporting:
        hits = len(supporting.intersection(retrieved_ids))
        evidence_recall: float | None = hits / len(supporting)
        complete_support_set: bool | None = hits == len(supporting)
        labels_available = True
    else:
        hits = 0
        evidence_recall = None
        complete_support_set = None
        labels_available = False

    metrics = EvaluationExampleMetrics(
        exact_match=rebuilt.exact_match,
        token_f1=rebuilt.token_f1,
        success=rebuilt.exact_match == 1.0,
        completed=rollout.termination_reason == "answered",
        attempted_searches=attempted_searches,
        executed_searches=executed_searches,
        rejected_searches=rejected_searches,
        malformed_actions=malformed_actions,
        step_count=len(rollout.steps),
        prompt_tokens_processed=rollout.prompt_tokens,
        response_tokens_generated=rollout.response_tokens,
        total_model_tokens=rollout.prompt_tokens + rollout.response_tokens,
        tool_tokens_appended=tool_tokens,
        supporting_labels_available=labels_available,
        supporting_documents=len(supporting),
        supporting_document_hits=hits,
        supporting_document_recall=evidence_recall,
        complete_support_set=complete_support_set,
    )
    return EvaluationRecord(
        run_id=run_id,
        policy_condition=cast(PolicyCondition, policy_condition),
        result_scope=cast(EvaluationResultScope, rollout.result_scope),
        rollout=rollout,
        metrics=metrics,
        retrieved_document_ids=tuple(retrieved_ids),
    )


def _total_and_mean(values: Sequence[int], count: int) -> dict[str, int | float]:
    total = sum(values)
    return {"mean": total / count, "total": total}


def aggregate_evaluation(
    items: Sequence[EvaluationItem],
    *,
    expected_example_ids: Sequence[str],
) -> dict[str, object]:
    """Validate exact coverage and recompute one aggregate from per-example records."""

    if not expected_example_ids:
        raise EvaluationIntegrityError("expected evaluation ID set must not be empty")
    expected = tuple(expected_example_ids)
    if len(expected) != len(set(expected)):
        raise EvaluationIntegrityError("requested evaluation IDs contain duplicates")
    if not items:
        raise EvaluationIntegrityError("per-example output is empty")
    failures = [item for item in items if isinstance(item, EvaluationFailure)]
    if failures:
        failed_ids = ", ".join(failure.example_id for failure in failures)
        raise EvaluationIntegrityError(
            f"infrastructure failures invalidate aggregation: {failed_ids}"
        )
    records = [item for item in items if isinstance(item, EvaluationRecord)]
    actual_ids = tuple(record.example_id for record in records)
    duplicates = sorted(
        identifier for identifier, count in Counter(actual_ids).items() if count > 1
    )
    if duplicates:
        raise EvaluationIntegrityError(f"duplicate evaluation IDs: {', '.join(duplicates)}")
    missing = [identifier for identifier in expected if identifier not in set(actual_ids)]
    extras = [identifier for identifier in actual_ids if identifier not in set(expected)]
    if missing or extras:
        details = []
        if missing:
            details.append(f"missing IDs: {', '.join(missing)}")
        if extras:
            details.append(f"unexpected IDs: {', '.join(extras)}")
        raise EvaluationIntegrityError("; ".join(details))
    if actual_ids != expected:
        raise EvaluationIntegrityError("evaluation records are not in requested canonical order")

    run_ids = {record.run_id for record in records}
    conditions = {record.policy_condition for record in records}
    scopes = {record.result_scope for record in records}
    models = {(record.rollout.model.name, record.rollout.model.revision) for record in records}
    sources = {record.rollout.example.source for record in records}
    if any(len(values) != 1 for values in (run_ids, conditions, scopes, models, sources)):
        raise EvaluationIntegrityError(
            "records must share one run, condition, scope, model, and dataset source"
        )

    count = len(records)
    labeled = [record for record in records if record.metrics.supporting_labels_available]
    supporting_documents = sum(record.metrics.supporting_documents for record in labeled)
    supporting_hits = sum(record.metrics.supporting_document_hits for record in labeled)
    if labeled:
        macro_recall: float | None = sum(
            cast(float, record.metrics.supporting_document_recall) for record in labeled
        ) / len(labeled)
        micro_recall: float | None = supporting_hits / supporting_documents
        complete_rate: float | None = sum(
            cast(bool, record.metrics.complete_support_set) for record in labeled
        ) / len(labeled)
    else:
        macro_recall = None
        micro_recall = None
        complete_rate = None

    termination_counts = Counter(record.rollout.termination_reason for record in records)
    model_name, model_revision = next(iter(models))
    return {
        "benchmark_eligible": (
            next(iter(scopes)) == "baseline_validation"
            and all(record.rollout.example.benchmark_eligible for record in records)
        ),
        "coverage": {
            "example_ids_sha256": _stable_ids_sha256(expected),
            "records": count,
            "requested_examples": len(expected),
        },
        "dataset": {"source": next(iter(sources))},
        "metrics": {
            "completion_rate": sum(record.metrics.completed for record in records) / count,
            "evidence": {
                "complete_support_set_rate": complete_rate,
                "excluded_unlabeled_examples": count - len(labeled),
                "labeled_examples": len(labeled),
                "macro_supporting_document_recall": macro_recall,
                "micro_supporting_document_recall": micro_recall,
                "supporting_document_hits": supporting_hits,
                "supporting_documents": supporting_documents,
            },
            "exact_match": sum(record.metrics.exact_match for record in records) / count,
            "malformed_actions": _total_and_mean(
                [record.metrics.malformed_actions for record in records], count
            ),
            "searches": {
                "attempted": _total_and_mean(
                    [record.metrics.attempted_searches for record in records], count
                ),
                "executed": _total_and_mean(
                    [record.metrics.executed_searches for record in records], count
                ),
                "rejected": _total_and_mean(
                    [record.metrics.rejected_searches for record in records], count
                ),
            },
            "steps": _total_and_mean([record.metrics.step_count for record in records], count),
            "success_rate": sum(record.metrics.success for record in records) / count,
            "termination_reasons": dict(sorted(termination_counts.items())),
            "token_f1": sum(record.metrics.token_f1 for record in records) / count,
            "tokens": {
                "prompt_tokens_processed": _total_and_mean(
                    [record.metrics.prompt_tokens_processed for record in records], count
                ),
                "response_tokens_generated": _total_and_mean(
                    [record.metrics.response_tokens_generated for record in records], count
                ),
                "tool_tokens_appended": _total_and_mean(
                    [record.metrics.tool_tokens_appended for record in records], count
                ),
                "total_model_tokens": _total_and_mean(
                    [record.metrics.total_model_tokens for record in records], count
                ),
            },
        },
        "model": {"name": model_name, "revision": model_revision},
        "policy_condition": next(iter(conditions)),
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "record_type": "evaluation_aggregate",
        "result_scope": next(iter(scopes)),
        "run_id": next(iter(run_ids)),
        "schema_version": 1,
    }
