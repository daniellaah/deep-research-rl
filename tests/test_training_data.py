import json
from pathlib import Path
from typing import Any

import pytest

from deep_research_rl.cli import main
from deep_research_rl.data.hotpotqa import build_hotpotqa
from deep_research_rl.data.source import load_source_config
from deep_research_rl.training.data import (
    TrainingDataError,
    prepare_training_data,
    verify_training_data,
)
from deep_research_rl.training.episode import AGENT_FLOW_NAME

parquet: Any = pytest.importorskip("pyarrow.parquet")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "tests" / "fixtures" / "hotpotqa"
FIXTURE_CONFIG = FIXTURE_ROOT / "source.json"
FIXTURE_RAW = FIXTURE_ROOT / "raw"


def _canonical_build(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    build_hotpotqa(load_source_config(FIXTURE_CONFIG), FIXTURE_RAW, processed)
    return processed


def test_training_export_adds_agent_flow_without_changing_canonical_jsonl(
    tmp_path: Path,
) -> None:
    processed = _canonical_build(tmp_path)
    canonical_path = processed / "agent_r1" / "train.jsonl"
    before = canonical_path.read_bytes()

    result = prepare_training_data(
        processed,
        tmp_path / "training",
        max_train=2,
        max_validation=1,
    )

    assert canonical_path.read_bytes() == before
    assert (result.train_rows, result.validation_rows) == (2, 1)
    assert verify_training_data(result.output_dir) == result
    table = parquet.read_table(result.train_path)
    assert set(table.column_names) == {
        "agent_name",
        "data_source",
        "extra_info",
        "prompt",
        "reward_model",
    }
    assert table["agent_name"].to_pylist() == [AGENT_FLOW_NAME, AGENT_FLOW_NAME]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["build_scope"] == "ordered_prefix_engineering_only"


def test_training_data_verification_detects_artifact_tampering(tmp_path: Path) -> None:
    result = prepare_training_data(
        _canonical_build(tmp_path),
        tmp_path / "training",
        max_train=1,
        max_validation=1,
    )
    result.train_path.write_bytes(result.train_path.read_bytes() + b"tampered")

    with pytest.raises(TrainingDataError, match="size mismatch"):
        verify_training_data(result.output_dir)


def test_training_data_cli_prepares_and_verifies(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    processed = _canonical_build(tmp_path)
    output = tmp_path / "training"

    assert (
        main(
            [
                "training",
                "prepare-data",
                "--processed-dir",
                str(processed),
                "--output-dir",
                str(output),
                "--max-train",
                "2",
                "--max-validation",
                "1",
            ]
        )
        == 0
    )
    assert "train=2, validation=1" in capsys.readouterr().out
    assert main(["training", "verify-data", "--training-data-dir", str(output)]) == 0
    assert "training data verified" in capsys.readouterr().out
