"""Materialize verified canonical HotpotQA rows as Agent-R1 Parquet inputs."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deep_research_rl.data.hotpotqa import verify_hotpotqa_build
from deep_research_rl.retrieval.index import sha256_file, stable_json_bytes, write_bytes_atomic
from deep_research_rl.training.episode import AGENT_FLOW_NAME

TRAINING_DATA_SCHEMA_VERSION = 1
TRAINING_DATA_MANIFEST = "manifest.json"


class TrainingDataError(ValueError):
    """Raised when training data cannot satisfy the Agent-R1 input contract."""


@dataclass(frozen=True, slots=True)
class TrainingDataBuild:
    """Verified paths and counts for one Parquet materialization."""

    output_dir: Path
    train_path: Path
    validation_path: Path
    manifest_path: Path
    train_rows: int
    validation_rows: int


def _pyarrow_modules() -> tuple[Any, Any]:
    try:
        pyarrow = importlib.import_module("pyarrow")
        parquet = importlib.import_module("pyarrow.parquet")
    except (ImportError, ModuleNotFoundError) as error:
        raise TrainingDataError(
            "Agent-R1 Parquet export requires pyarrow from the pinned training container"
        ) from error
    return pyarrow, parquet


def _training_rows(path: Path, *, limit: int | None) -> Iterator[dict[str, object]]:
    if limit is not None and limit < 1:
        raise TrainingDataError("row limits must be positive")
    emitted = 0
    try:
        input_file = path.open(encoding="utf-8")
    except OSError as error:
        raise TrainingDataError(f"could not read Agent-R1 JSONL {path}: {error}") from error
    with input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                value: object = json.loads(line)
            except json.JSONDecodeError as error:
                raise TrainingDataError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
                raise TrainingDataError(f"training row at {path}:{line_number} must be an object")
            row = {str(key): item for key, item in value.items()}
            if set(row) != {"data_source", "extra_info", "prompt", "reward_model"}:
                raise TrainingDataError(
                    f"training row at {path}:{line_number} has unexpected logical columns"
                )
            row["agent_name"] = AGENT_FLOW_NAME
            yield row
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def _write_parquet(source: Path, destination: Path, *, limit: int | None) -> int:
    pyarrow, parquet = _pyarrow_modules()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    writer = None
    records = 0
    batch: list[dict[str, object]] = []
    try:
        for row in _training_rows(source, limit=limit):
            batch.append(row)
            if len(batch) < 1024:
                continue
            table = pyarrow.Table.from_pylist(batch)
            if writer is None:
                writer = parquet.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            records += len(batch)
            batch.clear()
        if batch:
            table = pyarrow.Table.from_pylist(batch)
            if writer is None:
                writer = parquet.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            records += len(batch)
        if writer is None:
            raise TrainingDataError(f"Agent-R1 JSONL has no rows: {source}")
        writer.close()
        writer = None
        os.replace(temporary, destination)
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return records


def _file_metadata(path: Path, *, output_dir: Path, records: int) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(output_dir).as_posix(),
        "records": records,
        "sha256": sha256_file(path),
    }


def prepare_training_data(
    processed_dir: str | Path,
    output_dir: str | Path,
    *,
    max_train: int | None = None,
    max_validation: int | None = None,
) -> TrainingDataBuild:
    """Verify canonical inputs, add the flow selector, and write Parquet files."""

    source_dir = Path(processed_dir)
    source_manifest = verify_hotpotqa_build(source_dir)
    destination = Path(output_dir)
    train_path = destination / "train.parquet"
    validation_path = destination / "validation.parquet"
    train_source = source_dir / "agent_r1" / "train.jsonl"
    validation_source = source_dir / "agent_r1" / "validation.jsonl"
    train_rows = _write_parquet(train_source, train_path, limit=max_train)
    validation_rows = _write_parquet(
        validation_source,
        validation_path,
        limit=max_validation,
    )
    source_manifest_path = source_dir / "manifest.json"
    manifest = {
        "agent_flow": AGENT_FLOW_NAME,
        "build_scope": (
            "complete_source_build"
            if max_train is None and max_validation is None
            else "ordered_prefix_engineering_only"
        ),
        "outputs": {
            "train": _file_metadata(train_path, output_dir=destination, records=train_rows),
            "validation": _file_metadata(
                validation_path,
                output_dir=destination,
                records=validation_rows,
            ),
        },
        "record_type": "agent_r1_training_data_manifest",
        "schema_version": TRAINING_DATA_SCHEMA_VERSION,
        "source": {
            "build_mode": source_manifest.get("build_mode"),
            "manifest_path": str(source_manifest_path.resolve()),
            "manifest_sha256": sha256_file(source_manifest_path),
            "train_jsonl_sha256": sha256_file(train_source),
            "validation_jsonl_sha256": sha256_file(validation_source),
        },
    }
    manifest_path = destination / TRAINING_DATA_MANIFEST
    write_bytes_atomic(manifest_path, stable_json_bytes(manifest))
    result = verify_training_data(destination)
    if result.train_rows != train_rows or result.validation_rows != validation_rows:
        raise TrainingDataError("training-data verification changed output row counts")
    return result


def _manifest_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TrainingDataError(f"{field} must be an object")
    return {str(key): item for key, item in value.items()}


def _manifest_integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TrainingDataError(f"{field} must be an integer")
    return value


def _verify_parquet(
    output_dir: Path,
    metadata_value: object,
    *,
    split: str,
) -> tuple[Path, int]:
    _, parquet = _pyarrow_modules()
    metadata = _manifest_mapping(metadata_value, f"outputs.{split}")
    relative_value = metadata.get("path")
    if not isinstance(relative_value, str):
        raise TrainingDataError(f"outputs.{split}.path must be a string")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise TrainingDataError(f"outputs.{split}.path must stay inside the output directory")
    path = output_dir / relative
    try:
        size = path.stat().st_size
    except OSError as error:
        raise TrainingDataError(f"could not inspect training data {path}: {error}") from error
    if size != _manifest_integer(metadata.get("bytes"), f"outputs.{split}.bytes"):
        raise TrainingDataError(f"training data size mismatch: {path}")
    expected_hash = metadata.get("sha256")
    if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
        raise TrainingDataError(f"training data hash mismatch: {path}")
    expected_rows = _manifest_integer(metadata.get("records"), f"outputs.{split}.records")
    parquet_file = parquet.ParquetFile(path)
    if parquet_file.metadata.num_rows != expected_rows:
        raise TrainingDataError(f"training data row count mismatch: {path}")
    required_columns = {"agent_name", "data_source", "extra_info", "prompt", "reward_model"}
    if set(parquet_file.schema_arrow.names) != required_columns:
        raise TrainingDataError(f"training data columns differ from Agent-R1 contract: {path}")
    agent_names = parquet.read_table(path, columns=["agent_name"])["agent_name"].to_pylist()
    if any(name != AGENT_FLOW_NAME for name in agent_names):
        raise TrainingDataError(f"training data contains an unexpected agent_name: {path}")
    return path, expected_rows


def verify_training_data(output_dir: str | Path) -> TrainingDataBuild:
    """Verify hashes, row counts, columns, and flow routing for saved Parquet data."""

    destination = Path(output_dir)
    manifest_path = destination / TRAINING_DATA_MANIFEST
    try:
        value: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise TrainingDataError(
            f"could not read training manifest {manifest_path}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise TrainingDataError(f"invalid training manifest {manifest_path}: {error}") from error
    manifest = _manifest_mapping(value, "training manifest")
    if manifest.get("schema_version") != TRAINING_DATA_SCHEMA_VERSION:
        raise TrainingDataError("unsupported training-data manifest schema_version")
    if manifest.get("record_type") != "agent_r1_training_data_manifest":
        raise TrainingDataError("unexpected training-data manifest record_type")
    if manifest.get("agent_flow") != AGENT_FLOW_NAME:
        raise TrainingDataError("training-data manifest agent_flow differs from baseline")
    outputs = _manifest_mapping(manifest.get("outputs"), "outputs")
    if set(outputs) != {"train", "validation"}:
        raise TrainingDataError("training-data manifest must contain train and validation")
    train_path, train_rows = _verify_parquet(destination, outputs["train"], split="train")
    validation_path, validation_rows = _verify_parquet(
        destination,
        outputs["validation"],
        split="validation",
    )
    return TrainingDataBuild(
        output_dir=destination,
        train_path=train_path,
        validation_path=validation_path,
        manifest_path=manifest_path,
        train_rows=train_rows,
        validation_rows=validation_rows,
    )
