import json
from pathlib import Path

import pytest

from deep_research_rl.core.serialization import (
    TrajectoryFormatError,
    read_trajectory_jsonl,
    trajectory_as_json,
    trajectory_from_json,
)
from deep_research_rl.core.smoke import run_synthetic_smoke


def test_trajectory_round_trips_through_json_and_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "trajectory.jsonl"
    trajectory = run_synthetic_smoke(output_path)

    serialized = trajectory_as_json(trajectory)
    record: object = json.loads(serialized)
    assert isinstance(record, dict)
    assert record["synthetic"] is True
    assert record["benchmark_eligible"] is False
    assert trajectory_from_json(serialized) == trajectory
    assert read_trajectory_jsonl(output_path) == (trajectory,)


def test_rejects_inconsistent_top_level_provenance_marker(tmp_path: Path) -> None:
    trajectory = run_synthetic_smoke(tmp_path / "trajectory.jsonl")
    record: object = json.loads(trajectory_as_json(trajectory))
    assert isinstance(record, dict)
    record["synthetic"] = False

    with pytest.raises(TrajectoryFormatError, match="synthetic marker"):
        trajectory_from_json(json.dumps(record))
