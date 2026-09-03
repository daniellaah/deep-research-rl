from __future__ import annotations

from dataclasses import replace

import pytest
from evaluation_helpers import FixedPolicy, FixedRetriever, example, rollout, search_result

from deep_research_rl.evaluation.contracts import EvaluationFailure, EvaluationRecord
from deep_research_rl.evaluation.metrics import (
    EvaluationIntegrityError,
    aggregate_evaluation,
    evaluate_rollout,
)


def _golden_record(example_id: str = "example-1") -> tuple[EvaluationRecord, list[str]]:
    evaluation_example = example(example_id)
    policy = FixedPolicy(
        responses=(
            "SEARCH(first query)",
            "SEARCH(rejected query)",
            "not an action",
            "ANSWER(The Blue, Whale!)",
        ),
        prompt_lengths=(2, 3, 4, 5),
        response_lengths=(1, 2, 3, 4),
    )
    retriever = FixedRetriever(
        (
            search_result("doc-a", 1, "Alpha"),
            search_result("distractor", 2, "Noise"),
        )
    )
    agent_rollout = rollout(
        evaluation_example,
        policy,
        retriever,
        max_searches=1,
        max_steps=4,
    )
    rendered_tool_observations: list[str] = []

    def count_tool_tokens(text: str) -> int:
        rendered_tool_observations.append(text)
        return 7

    record = evaluate_rollout(
        agent_rollout,
        run_id="golden-run",
        policy_condition="prompted_agent",
        count_tool_tokens=count_tool_tokens,
    )
    return record, rendered_tool_observations


def _max_steps_record(example_id: str = "example-2") -> EvaluationRecord:
    agent_rollout = rollout(
        example(example_id),
        FixedPolicy(
            responses=("malformed",),
            prompt_lengths=(2,),
            response_lengths=(1,),
        ),
        FixedRetriever(()),
        max_searches=1,
        max_steps=1,
    )
    return evaluate_rollout(
        agent_rollout,
        run_id="golden-run",
        policy_condition="prompted_agent",
        count_tool_tokens=lambda _: 0,
    )


def test_hand_checked_per_example_metrics_keep_distinct_semantics() -> None:
    record, rendered = _golden_record()
    metrics = record.metrics

    assert metrics.exact_match == 1.0  # max over aliases after Hotpot normalization
    assert metrics.token_f1 == 1.0
    assert metrics.success is True
    assert metrics.completed is True
    assert metrics.attempted_searches == 2
    assert metrics.executed_searches == 1
    assert metrics.rejected_searches == 1
    assert metrics.malformed_actions == 1
    assert metrics.step_count == 4
    assert metrics.prompt_tokens_processed == 14
    assert metrics.response_tokens_generated == 10
    assert metrics.total_model_tokens == 24
    assert metrics.tool_tokens_appended == 7
    assert metrics.supporting_document_hits == 1
    assert metrics.supporting_document_recall == 0.5
    assert metrics.complete_support_set is False
    assert record.retrieved_document_ids == ("doc-a", "distractor")
    assert len(rendered) == 1
    assert rendered[0].startswith("Observation 1: SEARCH(first query) executed.")


def test_hand_checked_aggregate_uses_macro_micro_and_completion_denominators() -> None:
    correct, _ = _golden_record()
    truncated = _max_steps_record()

    aggregate = aggregate_evaluation(
        (correct, truncated),
        expected_example_ids=("example-1", "example-2"),
    )
    metrics = aggregate["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["exact_match"] == 0.5
    assert metrics["token_f1"] == 0.5
    assert metrics["success_rate"] == 0.5
    assert metrics["completion_rate"] == 0.5
    assert metrics["searches"] == {
        "attempted": {"mean": 1.0, "total": 2},
        "executed": {"mean": 0.5, "total": 1},
        "rejected": {"mean": 0.5, "total": 1},
    }
    evidence = metrics["evidence"]
    assert isinstance(evidence, dict)
    assert evidence["macro_supporting_document_recall"] == 0.25
    assert evidence["micro_supporting_document_recall"] == 0.25
    assert evidence["complete_support_set_rate"] == 0.0
    assert evidence["labeled_examples"] == 2
    assert evidence["supporting_documents"] == 4
    assert evidence["supporting_document_hits"] == 1


def test_no_search_labeled_example_has_zero_evidence_recall() -> None:
    agent_rollout = rollout(
        example("no-search"),
        FixedPolicy(
            responses=("ANSWER(wrong)",),
            prompt_lengths=(3,),
            response_lengths=(2,),
        ),
        FixedRetriever(()),
        max_searches=0,
        max_steps=1,
    )
    record = evaluate_rollout(
        agent_rollout,
        run_id="no-search-run",
        policy_condition="no_search",
        count_tool_tokens=lambda _: 0,
    )

    assert record.metrics.supporting_labels_available is True
    assert record.metrics.supporting_document_recall == 0.0
    assert record.metrics.complete_support_set is False


def test_aggregate_rejects_duplicates_missing_ids_order_and_infrastructure_failures() -> None:
    first, _ = _golden_record()
    second = _max_steps_record()

    with pytest.raises(EvaluationIntegrityError, match="duplicate evaluation IDs"):
        aggregate_evaluation(
            (first, first),
            expected_example_ids=("example-1", "example-2"),
        )
    with pytest.raises(EvaluationIntegrityError, match="missing IDs: example-2"):
        aggregate_evaluation((first,), expected_example_ids=("example-1", "example-2"))
    with pytest.raises(EvaluationIntegrityError, match="canonical order"):
        aggregate_evaluation(
            (second, first),
            expected_example_ids=("example-1", "example-2"),
        )
    failure = EvaluationFailure(
        run_id="golden-run",
        policy_condition="prompted_agent",
        result_scope="debug_validation_not_benchmark",
        example_id="example-2",
        error_type="RuntimeError",
        message="retriever unavailable",
    )
    with pytest.raises(EvaluationIntegrityError, match="infrastructure failures"):
        aggregate_evaluation(
            (first, failure),
            expected_example_ids=("example-1", "example-2"),
        )

    with pytest.raises(EvaluationIntegrityError, match="one run"):
        aggregate_evaluation(
            (first, replace(second, run_id="different-run")),
            expected_example_ids=("example-1", "example-2"),
        )
