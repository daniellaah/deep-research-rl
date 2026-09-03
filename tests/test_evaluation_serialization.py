from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from evaluation_helpers import FixedPolicy, FixedRetriever, example, rollout
from test_evaluation_metrics import _golden_record, _max_steps_record

from deep_research_rl.evaluation.artifacts import write_evaluation_artifacts
from deep_research_rl.evaluation.contracts import EvaluationFailure, PolicyCondition
from deep_research_rl.evaluation.metrics import aggregate_evaluation, evaluate_rollout
from deep_research_rl.evaluation.reporting import write_comparison_csv
from deep_research_rl.evaluation.serialization import (
    EvaluationFormatError,
    evaluation_item_from_dict,
    evaluation_item_to_dict,
    read_evaluation_jsonl,
    write_evaluation_jsonl,
)


def test_aggregate_recomputes_exactly_after_per_example_jsonl_round_trip(tmp_path: Path) -> None:
    first, _ = _golden_record()
    second = _max_steps_record()
    expected_ids = ("example-1", "example-2")
    path = tmp_path / "per-example.jsonl"

    write_evaluation_jsonl(path, (first, second))
    reloaded = read_evaluation_jsonl(path)

    reloaded_aggregate = aggregate_evaluation(reloaded, expected_example_ids=expected_ids)
    original_aggregate = aggregate_evaluation((first, second), expected_example_ids=expected_ids)
    assert reloaded_aggregate == original_aggregate


def test_incomplete_per_example_record_is_rejected() -> None:
    record, _ = _golden_record()
    serialized = evaluation_item_to_dict(record)
    metrics = serialized["metrics"]
    assert isinstance(metrics, dict)
    del metrics["token_f1"]

    with pytest.raises(EvaluationFormatError, match=r"metrics\.token_f1"):
        evaluation_item_from_dict(serialized)


def test_artifacts_capture_source_records_recomputed_aggregates_and_hashes(
    tmp_path: Path,
) -> None:
    first, _ = _golden_record()
    second = _max_steps_record()
    resolved: dict[str, object] = {
        "policy_condition": "prompted_agent",
        "result_scope": "debug_validation_not_benchmark",
        "run_id": "golden-run",
    }

    artifacts = write_evaluation_artifacts(
        tmp_path,
        items=(first, second),
        expected_example_ids=("example-1", "example-2"),
        resolved_config=resolved,
    )

    assert artifacts.status == "completed"
    assert artifacts.aggregate is not None
    assert json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8")) == (
        artifacts.aggregate
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["artifacts"]["per_example"]["records"] == 2
    expected_hash = hashlib.sha256((tmp_path / "per-example.jsonl").read_bytes()).hexdigest()
    assert manifest["artifacts"]["per_example"]["sha256"] == expected_hash
    rows = list(csv.DictReader((tmp_path / "aggregate.csv").open(encoding="utf-8")))
    assert rows[0]["policy_condition"] == "prompted_agent"
    assert rows[0]["success_rate"] == "0.5"


def test_invalid_run_manifest_removes_stale_aggregate_files(tmp_path: Path) -> None:
    (tmp_path / "aggregate.json").write_text("stale", encoding="utf-8")
    (tmp_path / "aggregate.csv").write_text("stale", encoding="utf-8")
    failure = EvaluationFailure(
        run_id="failed-run",
        policy_condition="prompted_agent",
        result_scope="debug_validation_not_benchmark",
        example_id="example-1",
        error_type="RuntimeError",
        message="retriever worker unavailable",
    )
    resolved: dict[str, object] = {
        "policy_condition": "prompted_agent",
        "result_scope": "debug_validation_not_benchmark",
        "run_id": "failed-run",
    }

    artifacts = write_evaluation_artifacts(
        tmp_path,
        items=(failure,),
        expected_example_ids=("example-1",),
        resolved_config=resolved,
    )

    assert artifacts.status == "invalid"
    assert not (tmp_path / "aggregate.json").exists()
    assert not (tmp_path / "aggregate.csv").exists()


def test_comparison_table_has_fixed_distinct_policy_rows(tmp_path: Path) -> None:
    first, _ = _golden_record()
    second = _max_steps_record()
    expected_ids = ("example-1", "example-2")
    aggregates = []
    conditions: tuple[PolicyCondition, ...] = (
        "rl_agent",
        "no_search",
        "prompted_agent",
    )
    for condition in conditions:
        if condition == "no_search":
            no_search_records = []
            for example_id, response in (
                ("example-1", "ANSWER(blue whale)"),
                ("example-2", "malformed"),
            ):
                no_search_rollout = rollout(
                    example(example_id),
                    FixedPolicy(
                        responses=(response,),
                        prompt_lengths=(1,),
                        response_lengths=(1,),
                    ),
                    FixedRetriever(()),
                    max_searches=0,
                    max_steps=1,
                )
                no_search_records.append(
                    evaluate_rollout(
                        no_search_rollout,
                        run_id="no_search-run",
                        policy_condition="no_search",
                        count_tool_tokens=lambda _: 0,
                    )
                )
            records = tuple(no_search_records)
        else:
            records = (
                replace(first, run_id=f"{condition}-run", policy_condition=condition),
                replace(second, run_id=f"{condition}-run", policy_condition=condition),
            )
        aggregates.append(aggregate_evaluation(records, expected_example_ids=expected_ids))

    output = write_comparison_csv(tmp_path / "comparison.csv", aggregates)
    rows = list(csv.DictReader(output.open(encoding="utf-8")))

    assert [row["policy_condition"] for row in rows] == [
        "no_search",
        "prompted_agent",
        "rl_agent",
    ]
    assert all(row["model_name"] == "test/qwen" for row in rows)


def test_unreadable_and_invalid_jsonl_records_fail_with_line_context(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(EvaluationFormatError, match=r"broken.jsonl:1"):
        read_evaluation_jsonl(path)


def test_max_step_without_answer_is_a_valid_record_not_incomplete() -> None:
    agent_rollout = rollout(
        example("bounded"),
        FixedPolicy(
            responses=("malformed",),
            prompt_lengths=(1,),
            response_lengths=(1,),
        ),
        FixedRetriever(()),
        max_searches=5,
        max_steps=1,
    )
    record = evaluate_rollout(
        agent_rollout,
        run_id="bounded-run",
        policy_condition="prompted_agent",
        count_tool_tokens=lambda _: 0,
    )

    assert record.metrics.completed is False
    assert record.metrics.exact_match == 0.0
    assert evaluation_item_from_dict(evaluation_item_to_dict(record)) == record
