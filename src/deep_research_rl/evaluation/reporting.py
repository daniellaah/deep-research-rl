"""Aggregate CSV, comparison tables, and compatibility validation."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from pathlib import Path

from deep_research_rl.evaluation.metrics import EVALUATION_PROTOCOL_VERSION
from deep_research_rl.evaluation.serialization import EvaluationFormatError
from deep_research_rl.retrieval.index import write_bytes_atomic

COMPARISON_CONDITION_ORDER = {"no_search": 0, "prompted_agent": 1, "rl_agent": 2}
AGGREGATE_CSV_COLUMNS = (
    "run_id",
    "policy_condition",
    "result_scope",
    "benchmark_eligible",
    "model_name",
    "model_revision",
    "records",
    "exact_match",
    "token_f1",
    "success_rate",
    "completion_rate",
    "attempted_searches_total",
    "attempted_searches_mean",
    "executed_searches_total",
    "executed_searches_mean",
    "rejected_searches_total",
    "rejected_searches_mean",
    "steps_total",
    "steps_mean",
    "malformed_actions_total",
    "malformed_actions_mean",
    "prompt_tokens_processed_total",
    "prompt_tokens_processed_mean",
    "response_tokens_generated_total",
    "response_tokens_generated_mean",
    "total_model_tokens_total",
    "total_model_tokens_mean",
    "tool_tokens_appended_total",
    "tool_tokens_appended_mean",
    "evidence_labeled_examples",
    "evidence_excluded_unlabeled_examples",
    "macro_supporting_document_recall",
    "micro_supporting_document_recall",
    "complete_support_set_rate",
)


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise EvaluationFormatError(f"{field} must be an object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def aggregate_csv_row(aggregate: dict[str, object]) -> dict[str, object]:
    """Flatten a validated aggregate into the stable comparison row schema."""

    if aggregate.get("schema_version") != 1:
        raise EvaluationFormatError("unsupported aggregate schema_version")
    if aggregate.get("record_type") != "evaluation_aggregate":
        raise EvaluationFormatError("record_type must be evaluation_aggregate")
    if aggregate.get("protocol_version") != EVALUATION_PROTOCOL_VERSION:
        raise EvaluationFormatError("aggregate uses an incompatible evaluation protocol")
    condition = aggregate.get("policy_condition")
    if condition not in COMPARISON_CONDITION_ORDER:
        raise EvaluationFormatError(f"unsupported policy_condition: {condition}")
    coverage = _mapping(aggregate.get("coverage"), "coverage")
    model = _mapping(aggregate.get("model"), "model")
    metrics = _mapping(aggregate.get("metrics"), "metrics")
    searches = _mapping(metrics.get("searches"), "metrics.searches")
    attempted = _mapping(searches.get("attempted"), "metrics.searches.attempted")
    executed = _mapping(searches.get("executed"), "metrics.searches.executed")
    rejected = _mapping(searches.get("rejected"), "metrics.searches.rejected")
    steps = _mapping(metrics.get("steps"), "metrics.steps")
    malformed = _mapping(metrics.get("malformed_actions"), "metrics.malformed_actions")
    tokens = _mapping(metrics.get("tokens"), "metrics.tokens")
    prompt_tokens = _mapping(
        tokens.get("prompt_tokens_processed"), "metrics.tokens.prompt_tokens_processed"
    )
    response_tokens = _mapping(
        tokens.get("response_tokens_generated"), "metrics.tokens.response_tokens_generated"
    )
    total_tokens = _mapping(tokens.get("total_model_tokens"), "metrics.tokens.total_model_tokens")
    tool_tokens = _mapping(
        tokens.get("tool_tokens_appended"), "metrics.tokens.tool_tokens_appended"
    )
    evidence = _mapping(metrics.get("evidence"), "metrics.evidence")
    return {
        "attempted_searches_mean": attempted.get("mean"),
        "attempted_searches_total": attempted.get("total"),
        "benchmark_eligible": aggregate.get("benchmark_eligible"),
        "complete_support_set_rate": evidence.get("complete_support_set_rate"),
        "completion_rate": metrics.get("completion_rate"),
        "evidence_excluded_unlabeled_examples": evidence.get("excluded_unlabeled_examples"),
        "evidence_labeled_examples": evidence.get("labeled_examples"),
        "exact_match": metrics.get("exact_match"),
        "executed_searches_mean": executed.get("mean"),
        "executed_searches_total": executed.get("total"),
        "macro_supporting_document_recall": evidence.get("macro_supporting_document_recall"),
        "malformed_actions_mean": malformed.get("mean"),
        "malformed_actions_total": malformed.get("total"),
        "micro_supporting_document_recall": evidence.get("micro_supporting_document_recall"),
        "model_name": model.get("name"),
        "model_revision": model.get("revision"),
        "policy_condition": condition,
        "prompt_tokens_processed_mean": prompt_tokens.get("mean"),
        "prompt_tokens_processed_total": prompt_tokens.get("total"),
        "records": coverage.get("records"),
        "rejected_searches_mean": rejected.get("mean"),
        "rejected_searches_total": rejected.get("total"),
        "response_tokens_generated_mean": response_tokens.get("mean"),
        "response_tokens_generated_total": response_tokens.get("total"),
        "result_scope": aggregate.get("result_scope"),
        "run_id": aggregate.get("run_id"),
        "steps_mean": steps.get("mean"),
        "steps_total": steps.get("total"),
        "success_rate": metrics.get("success_rate"),
        "token_f1": metrics.get("token_f1"),
        "tool_tokens_appended_mean": tool_tokens.get("mean"),
        "tool_tokens_appended_total": tool_tokens.get("total"),
        "total_model_tokens_mean": total_tokens.get("mean"),
        "total_model_tokens_total": total_tokens.get("total"),
    }


def _csv_bytes(rows: Sequence[dict[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=AGGREGATE_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_aggregate_csv(path: str | Path, aggregate: dict[str, object]) -> Path:
    """Write one flattened aggregate row."""

    output_path = Path(path)
    write_bytes_atomic(output_path, _csv_bytes((aggregate_csv_row(aggregate),)))
    return output_path


def load_aggregate_json(path: str | Path) -> dict[str, object]:
    """Load enough of an aggregate to reject unreadable or incompatible inputs."""

    input_path = Path(path)
    try:
        value: object = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationFormatError(f"could not read aggregate {input_path}: {error}") from error
    aggregate = _mapping(value, "aggregate")
    aggregate_csv_row(aggregate)
    return aggregate


def write_comparison_csv(
    path: str | Path,
    aggregates: Sequence[dict[str, object]],
) -> Path:
    """Write comparable no-search, prompted-agent, and RL-agent rows in fixed order."""

    if not aggregates:
        raise EvaluationFormatError("comparison requires at least one aggregate")
    rows_by_aggregate = [(aggregate_csv_row(aggregate), aggregate) for aggregate in aggregates]
    conditions = [row["policy_condition"] for row, _ in rows_by_aggregate]
    if len(conditions) != len(set(conditions)):
        raise EvaluationFormatError("comparison contains duplicate policy conditions")
    scopes = {aggregate.get("result_scope") for aggregate in aggregates}
    datasets = {
        (
            _mapping(aggregate.get("dataset"), "dataset").get("source"),
            _mapping(aggregate.get("coverage"), "coverage").get("example_ids_sha256"),
        )
        for aggregate in aggregates
    }
    if len(scopes) != 1 or len(datasets) != 1:
        raise EvaluationFormatError(
            "comparison aggregates must use the same scope, dataset source, and example IDs"
        )
    ordered = sorted(
        rows_by_aggregate,
        key=lambda pair: COMPARISON_CONDITION_ORDER[str(pair[0]["policy_condition"])],
    )
    rows = [row for row, _ in ordered]
    output_path = Path(path)
    write_bytes_atomic(output_path, _csv_bytes(rows))
    return output_path
