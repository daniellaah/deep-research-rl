"""Strict JSONL serialization for evaluation source records."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from deep_research_rl.agent.serialization import agent_rollout_from_dict, agent_rollout_to_dict
from deep_research_rl.evaluation.contracts import (
    BASELINE_EVALUATION_PROTOCOL,
    EvaluationExampleMetrics,
    EvaluationFailure,
    EvaluationItem,
    EvaluationRecord,
    EvaluationResultScope,
    PolicyCondition,
)
from deep_research_rl.retrieval.index import stable_json_bytes, write_bytes_atomic

EVALUATION_RECORD_SCHEMA_VERSION = 1


class EvaluationFormatError(ValueError):
    """Raised when an evaluation artifact is unreadable or incomplete."""


def _metrics_to_record(metrics: EvaluationExampleMetrics) -> dict[str, object]:
    return {
        "attempted_searches": metrics.attempted_searches,
        "complete_support_set": metrics.complete_support_set,
        "completed": metrics.completed,
        "exact_match": metrics.exact_match,
        "executed_searches": metrics.executed_searches,
        "malformed_actions": metrics.malformed_actions,
        "prompt_tokens_processed": metrics.prompt_tokens_processed,
        "rejected_searches": metrics.rejected_searches,
        "response_tokens_generated": metrics.response_tokens_generated,
        "step_count": metrics.step_count,
        "success": metrics.success,
        "supporting_document_hits": metrics.supporting_document_hits,
        "supporting_document_recall": metrics.supporting_document_recall,
        "supporting_documents": metrics.supporting_documents,
        "supporting_labels_available": metrics.supporting_labels_available,
        "token_f1": metrics.token_f1,
        "tool_tokens_appended": metrics.tool_tokens_appended,
        "total_model_tokens": metrics.total_model_tokens,
    }


def evaluation_item_to_dict(item: EvaluationItem) -> dict[str, object]:
    """Convert one valid outcome or infrastructure failure to a stable record."""

    common: dict[str, object] = {
        "example_id": item.example_id,
        "policy_condition": item.policy_condition,
        "protocol_version": BASELINE_EVALUATION_PROTOCOL,
        "result_scope": item.result_scope,
        "run_id": item.run_id,
        "schema_version": EVALUATION_RECORD_SCHEMA_VERSION,
    }
    if isinstance(item, EvaluationFailure):
        return {
            **common,
            "error_type": item.error_type,
            "message": item.message,
            "record_type": "evaluation_infrastructure_failure",
        }
    return {
        **common,
        "metrics": _metrics_to_record(item.metrics),
        "record_type": "evaluation_example",
        "retrieved_document_ids": list(item.retrieved_document_ids),
        "trajectory": agent_rollout_to_dict(item.rollout),
    }


def evaluation_item_as_json(item: EvaluationItem) -> str:
    """Return one stable JSONL-ready evaluation line."""

    return json.dumps(evaluation_item_to_dict(item), ensure_ascii=False, sort_keys=True)


def write_evaluation_jsonl(path: str | Path, items: Sequence[EvaluationItem]) -> Path:
    """Atomically write per-example source records in requested order."""

    if not items:
        raise ValueError("at least one evaluation item is required")
    output_path = Path(path)
    content = "".join(f"{evaluation_item_as_json(item)}\n" for item in items).encode("utf-8")
    write_bytes_atomic(output_path, content)
    return output_path


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise EvaluationFormatError(f"{field} must be an object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise EvaluationFormatError(f"{field} must be an array")
    return list(value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise EvaluationFormatError(f"{field} must be a string")
    return value


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise EvaluationFormatError(f"{field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise EvaluationFormatError(f"{field} must be a number")
    return float(value)


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise EvaluationFormatError(f"{field} must be a boolean")
    return value


def _optional_number(value: object, field: str) -> float | None:
    return None if value is None else _number(value, field)


def _optional_boolean(value: object, field: str) -> bool | None:
    return None if value is None else _boolean(value, field)


def _condition(value: object) -> PolicyCondition:
    condition = _string(value, "policy_condition")
    if condition not in {"no_search", "prompted_agent", "rl_agent"}:
        raise EvaluationFormatError(f"unsupported policy_condition: {condition}")
    return cast(PolicyCondition, condition)


def _scope(value: object) -> EvaluationResultScope:
    scope = _string(value, "result_scope")
    if scope not in {"debug_validation_not_benchmark", "baseline_validation"}:
        raise EvaluationFormatError(f"unsupported result_scope: {scope}")
    return cast(EvaluationResultScope, scope)


def _metrics_from_record(value: object) -> EvaluationExampleMetrics:
    record = _mapping(value, "metrics")
    return EvaluationExampleMetrics(
        exact_match=_number(record.get("exact_match"), "metrics.exact_match"),
        token_f1=_number(record.get("token_f1"), "metrics.token_f1"),
        success=_boolean(record.get("success"), "metrics.success"),
        completed=_boolean(record.get("completed"), "metrics.completed"),
        attempted_searches=_integer(record.get("attempted_searches"), "metrics.attempted_searches"),
        executed_searches=_integer(record.get("executed_searches"), "metrics.executed_searches"),
        rejected_searches=_integer(record.get("rejected_searches"), "metrics.rejected_searches"),
        malformed_actions=_integer(record.get("malformed_actions"), "metrics.malformed_actions"),
        step_count=_integer(record.get("step_count"), "metrics.step_count"),
        prompt_tokens_processed=_integer(
            record.get("prompt_tokens_processed"), "metrics.prompt_tokens_processed"
        ),
        response_tokens_generated=_integer(
            record.get("response_tokens_generated"), "metrics.response_tokens_generated"
        ),
        total_model_tokens=_integer(record.get("total_model_tokens"), "metrics.total_model_tokens"),
        tool_tokens_appended=_integer(
            record.get("tool_tokens_appended"), "metrics.tool_tokens_appended"
        ),
        supporting_labels_available=_boolean(
            record.get("supporting_labels_available"),
            "metrics.supporting_labels_available",
        ),
        supporting_documents=_integer(
            record.get("supporting_documents"), "metrics.supporting_documents"
        ),
        supporting_document_hits=_integer(
            record.get("supporting_document_hits"), "metrics.supporting_document_hits"
        ),
        supporting_document_recall=_optional_number(
            record.get("supporting_document_recall"),
            "metrics.supporting_document_recall",
        ),
        complete_support_set=_optional_boolean(
            record.get("complete_support_set"), "metrics.complete_support_set"
        ),
    )


def evaluation_item_from_dict(value: object) -> EvaluationItem:
    """Validate and reconstruct one decoded evaluation record."""

    record = _mapping(value, "evaluation record")
    if _integer(record.get("schema_version"), "schema_version") != EVALUATION_RECORD_SCHEMA_VERSION:
        raise EvaluationFormatError("unsupported evaluation schema_version")
    if _string(record.get("protocol_version"), "protocol_version") != BASELINE_EVALUATION_PROTOCOL:
        raise EvaluationFormatError("unsupported evaluation protocol_version")
    record_type = _string(record.get("record_type"), "record_type")
    run_id = _string(record.get("run_id"), "run_id")
    example_id = _string(record.get("example_id"), "example_id")
    condition = _condition(record.get("policy_condition"))
    scope = _scope(record.get("result_scope"))
    try:
        if record_type == "evaluation_infrastructure_failure":
            return EvaluationFailure(
                run_id=run_id,
                policy_condition=condition,
                result_scope=scope,
                example_id=example_id,
                error_type=_string(record.get("error_type"), "error_type"),
                message=_string(record.get("message"), "message"),
            )
        if record_type != "evaluation_example":
            raise EvaluationFormatError(f"unsupported record_type: {record_type}")
        rollout = agent_rollout_from_dict(record.get("trajectory"))
        if rollout.example.example_id != example_id:
            raise EvaluationFormatError("example_id does not match embedded trajectory")
        retrieved_ids = tuple(
            _string(item, "retrieved_document_ids[]")
            for item in _list(record.get("retrieved_document_ids"), "retrieved_document_ids")
        )
        return EvaluationRecord(
            run_id=run_id,
            policy_condition=condition,
            result_scope=scope,
            rollout=rollout,
            metrics=_metrics_from_record(record.get("metrics")),
            retrieved_document_ids=retrieved_ids,
        )
    except ValueError as error:
        if isinstance(error, EvaluationFormatError):
            raise
        raise EvaluationFormatError(str(error)) from error


def evaluation_item_from_json(line: str) -> EvaluationItem:
    """Decode one evaluation JSON line."""

    try:
        value: object = json.loads(line)
    except json.JSONDecodeError as error:
        raise EvaluationFormatError(f"invalid evaluation JSON: {error}") from error
    return evaluation_item_from_dict(value)


def read_evaluation_jsonl(path: str | Path) -> tuple[EvaluationItem, ...]:
    """Read every non-empty record and identify the exact failing line."""

    input_path = Path(path)
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise EvaluationFormatError(
            f"could not read evaluation records {input_path}: {error}"
        ) from error
    items: list[EvaluationItem] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            items.append(evaluation_item_from_json(line))
        except EvaluationFormatError as error:
            raise EvaluationFormatError(f"{input_path}:{line_number}: {error}") from error
    return tuple(items)


def write_json_artifact(path: str | Path, value: object) -> Path:
    """Write one stable JSON artifact atomically."""

    output_path = Path(path)
    write_bytes_atomic(output_path, stable_json_bytes(value))
    return output_path
