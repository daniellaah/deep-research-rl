"""Resolved configuration and self-checking run-manifest artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from deep_research_rl.evaluation.contracts import EvaluationFailure, EvaluationItem
from deep_research_rl.evaluation.metrics import (
    EVALUATION_PROTOCOL_VERSION,
    EvaluationIntegrityError,
    aggregate_evaluation,
)
from deep_research_rl.evaluation.reporting import write_aggregate_csv
from deep_research_rl.evaluation.serialization import write_evaluation_jsonl, write_json_artifact
from deep_research_rl.retrieval.index import sha256_file

PER_EXAMPLE_FILENAME = "per-example.jsonl"
AGGREGATE_JSON_FILENAME = "aggregate.json"
AGGREGATE_CSV_FILENAME = "aggregate.csv"
RESOLVED_CONFIG_FILENAME = "resolved-config.json"
RUN_MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class EvaluationArtifacts:
    """Paths and status emitted for one evaluation attempt."""

    output_dir: Path
    manifest_path: Path
    status: str
    aggregate: dict[str, object] | None


def capture_code_state(repository_root: str | Path) -> dict[str, object]:
    """Capture the exact Git revision and whether tracked/untracked content is present."""

    root = Path(repository_root)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise EvaluationIntegrityError(f"could not capture code revision: {error}") from error
    return {"dirty": bool(status.strip()), "revision": revision}


def capture_hardware() -> dict[str, object]:
    """Capture dependency-light host/runtime identity for a run."""

    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
    }


def _ids_sha256(example_ids: Sequence[str]) -> str:
    payload = json.dumps(list(example_ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_resolved_config(
    *,
    run_id: str,
    policy_condition: str,
    result_scope: str,
    examples_path: Path,
    examples_sha256: str,
    dataset_source: str,
    source_records: int,
    requested_example_ids: Sequence[str],
    model_name: str,
    model_revision: str,
    prompt_format: str,
    device: str,
    dtype: str,
    retrieval: dict[str, object],
    code_state: dict[str, object],
    command: Sequence[str],
) -> dict[str, object]:
    """Materialize every frozen inference/evaluation control and input revision."""

    no_search = policy_condition == "no_search"
    return {
        "code": code_state,
        "command": list(command),
        "config_kind": "resolved",
        "dataset": {
            "examples_path": str(examples_path.resolve()),
            "examples_sha256": examples_sha256,
            "name": "hotpot_qa",
            "requested_example_ids_sha256": _ids_sha256(requested_example_ids),
            "requested_examples": len(requested_example_ids),
            "selection": "all" if result_scope == "baseline_validation" else "ordered_prefix",
            "source": dataset_source,
            "source_records": source_records,
            "variant": "distractor",
        },
        "environment": {
            "actions": ["SEARCH", "ANSWER"],
            "context_policy": "append_only",
            "max_executed_searches": 0 if no_search else 5,
            "max_policy_steps": 1 if no_search else 8,
        },
        "evaluation": {
            "answer_alias_reduction": "maximum",
            "answer_normalization": "hotpotqa_lower_punctuation_articles_whitespace",
            "coverage": "exactly_once_in_requested_order",
            "evidence_documents": "union_across_executed_searches",
            "infrastructure_failure": "invalidate_run",
            "per_example_source_of_truth": True,
        },
        "hardware": capture_hardware(),
        "inference": {
            "device": device,
            "do_sample": False,
            "dtype": dtype,
            "max_new_tokens_per_step": 96,
            "max_prompt_tokens": 8192,
            "seed": 0,
        },
        "model": {
            "checkpoint": model_name,
            "prompt_format": prompt_format,
            "revision": model_revision,
        },
        "policy_condition": policy_condition,
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "result_scope": result_scope,
        "retrieval": retrieval,
        "run_id": run_id,
        "schema_version": 1,
    }


def _artifact_metadata(
    path: Path,
    output_dir: Path,
    *,
    records: int | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "bytes": path.stat().st_size,
        "path": path.relative_to(output_dir).as_posix(),
        "sha256": sha256_file(path),
    }
    if records is not None:
        metadata["records"] = records
    return metadata


def write_evaluation_artifacts(
    output_dir: str | Path,
    *,
    items: Sequence[EvaluationItem],
    expected_example_ids: Sequence[str],
    resolved_config: dict[str, object],
) -> EvaluationArtifacts:
    """Write source records, recomputed aggregates, resolved config, and final manifest."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    per_example_path = write_evaluation_jsonl(directory / PER_EXAMPLE_FILENAME, items)
    resolved_path = write_json_artifact(directory / RESOLVED_CONFIG_FILENAME, resolved_config)
    failures = [item for item in items if isinstance(item, EvaluationFailure)]
    artifacts: dict[str, object] = {
        "per_example": _artifact_metadata(per_example_path, directory, records=len(items)),
        "resolved_config": _artifact_metadata(resolved_path, directory),
    }
    aggregate: dict[str, object] | None = None
    status = "invalid" if failures else "completed"
    integrity_error: str | None = None
    if not failures:
        try:
            aggregate = aggregate_evaluation(items, expected_example_ids=expected_example_ids)
        except EvaluationIntegrityError as error:
            status = "invalid"
            integrity_error = str(error)
        else:
            aggregate_json = write_json_artifact(directory / AGGREGATE_JSON_FILENAME, aggregate)
            aggregate_csv = write_aggregate_csv(directory / AGGREGATE_CSV_FILENAME, aggregate)
            artifacts["aggregate_json"] = _artifact_metadata(aggregate_json, directory)
            artifacts["aggregate_csv"] = _artifact_metadata(aggregate_csv, directory, records=1)
    if status == "invalid":
        (directory / AGGREGATE_JSON_FILENAME).unlink(missing_ok=True)
        (directory / AGGREGATE_CSV_FILENAME).unlink(missing_ok=True)

    manifest = {
        "artifacts": artifacts,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "failure_count": len(failures),
        "integrity_error": integrity_error,
        "policy_condition": resolved_config.get("policy_condition"),
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "record_type": "evaluation_run_manifest",
        "requested_examples": len(expected_example_ids),
        "result_scope": resolved_config.get("result_scope"),
        "run_id": resolved_config.get("run_id"),
        "schema_version": 1,
        "status": status,
        "written_records": len(items),
    }
    manifest_path = write_json_artifact(directory / RUN_MANIFEST_FILENAME, manifest)
    return EvaluationArtifacts(
        output_dir=directory,
        manifest_path=manifest_path,
        status=status,
        aggregate=aggregate,
    )
